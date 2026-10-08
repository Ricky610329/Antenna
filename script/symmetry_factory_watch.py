"""Run the authorized private symmetry controller every 30 minutes.

The foreground process is intended to be launched once with a hidden window.
It never starts HFSS. A local STOP file ends monitoring between cycles; errors
are recorded and stop the controller without changing worker claims/results.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from script import profiled_batch as pb
from script import symmetry_factory_cycle as cycle


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _campaign_stop(settings: dict) -> str | None:
    scope = settings.get('_bound_scope')
    if not isinstance(scope, str) or not scope:
        raise ValueError('watch campaign stop requires a bound scope')
    state = Path(settings['dataset_root']) / 'jobs_state'
    for name in ('STOP', f'STOP_{scope}'):
        if (state / name).exists():
            return name
    return None


def _read_settings_once(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    value = json.loads(raw.decode('utf-8'))
    if not isinstance(value, dict):
        raise ValueError('watch settings must be a JSON object')
    return value, hashlib.sha256(raw).hexdigest()


def _profile_binding(path: Path) -> tuple[object, dict[str, str]]:
    before = pb.file_sha256(path)
    profile = pb.load_profile_config(path)
    if profile is None:
        raise ValueError('watch requires a named measurement profile')
    if pb.file_sha256(path) != before:
        raise ValueError('watch profile changed while binding the campaign')
    return profile, {'path': str(path.resolve()), 'sha256': before, 'scope': profile.scope}


def _require_campaign_binding(settings_file: Path, settings_sha256: str,
                              profile_binding: dict[str, str],
                              pilot_binding: dict | None = None) -> None:
    if pb.file_sha256(settings_file) != settings_sha256:
        raise ValueError('watch settings changed; restart explicitly with the new settings')
    profile = Path(profile_binding['path'])
    if pb.file_sha256(profile) != profile_binding['sha256']:
        raise ValueError('watch profile changed; restart explicitly with the new profile')
    if pilot_binding is not None:
        cycle._require_pilot_request_binding(
            pilot_binding['request_path'], profile, pilot_binding)


def _validate_cycle_receipt(receipt: dict, retry_workers: tuple[str, ...],
                            pilot_binding: dict | None = None) -> None:
    if receipt.get('expected_retry_workers') != list(retry_workers):
        raise ValueError('controller receipt retry worker roster differs from watch binding')
    valid = receipt.get('valid_unique')
    target = receipt.get('target_valid_unique')
    if (isinstance(valid, bool) or not isinstance(valid, int) or
            isinstance(target, bool) or not isinstance(target, int)):
        raise ValueError('controller receipt lacks integer valid-unique accounting')
    if valid > target:
        raise ValueError('controller receipt exceeds exact valid-unique target')
    if valid == target and receipt.get('pending_unique') != 0:
        raise ValueError('controller receipt reaches target with residual pending unique')
    actual_pilot = receipt.get('pilot_request')
    if actual_pilot != pilot_binding or (pilot_binding is None and 'pilot_request' in receipt):
        raise ValueError('controller receipt pilot request differs from watch binding')


def _write_state(state_file: Path, attempt_file: Path, attempt: dict, state: dict) -> None:
    pb.atomic_json(state_file, state)
    record = dict(attempt)
    record['current_watch_status'] = state
    pb.atomic_json(attempt_file, record)


def _next_scheduled_due(started: float, now: float, interval: int) -> float:
    due = started + interval
    while due <= now:
        due += interval
    return due


@contextlib.contextmanager
def _exclusive_watch(path: Path):
    # Kernel locks disappear when the process exits, including a hard crash.
    # The harmless lock file itself is retained for subsequent launches.
    with path.open('a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'0'); stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError('a watcher already owns this work directory') from exc
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError('a watcher already owns this work directory') from exc
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def run(settings_path: str | Path, **kwargs) -> dict:
    settings_file = Path(settings_path).resolve()
    settings, settings_sha256 = _read_settings_once(settings_file)
    if 'training_workdir' not in settings:
        raise ValueError('watch settings must name training_workdir')
    training = Path(settings['training_workdir']).resolve()
    if not training.is_dir():
        raise FileNotFoundError(f'training work directory does not exist: {training}')
    with _exclusive_watch(training / 'factory-watch.lock'):
        return _run(settings_file, settings_snapshot=settings,
                    settings_sha256=settings_sha256, **kwargs)


def _run(settings_path: str | Path, *, prepare: Callable = cycle.run_once,
        dispatch: Callable = cycle.commit_dispatch, sleep: Callable = time.sleep,
        monotonic: Callable = time.monotonic, max_cycles: int | None = None,
        campaign_stop: Callable = _campaign_stop,
        settings_snapshot: dict | None = None,
        settings_sha256: str | None = None) -> dict:
    settings_file = Path(settings_path).resolve()
    if settings_snapshot is None or settings_sha256 is None:
        settings, bound_settings_sha256 = _read_settings_once(settings_file)
    else:
        settings = dict(settings_snapshot)
        bound_settings_sha256 = str(settings_sha256)
    required = {'local_workdir', 'dataset_root', 'profile_config', 'training_workdir',
                'seed_inputs', 'interval_seconds', 'expected_retry_workers'}
    if not required <= settings.keys() or set(settings) - required - {
            'prepared_blind_pool', 'pilot_request'}:
        raise ValueError(
            'watch settings must name explicit controller inputs, interval, and '
            'expected_retry_workers')
    interval = int(settings.pop('interval_seconds'))
    if interval < 1800:
        raise ValueError('routine controller interval must be at least 1800 seconds')
    if max_cycles is not None and max_cycles < 1:
        raise ValueError('max_cycles must be positive')
    retry_workers = cycle._normalize_retry_workers(settings['expected_retry_workers'])
    if not retry_workers:
        raise ValueError('watch requires a nonempty expected_retry_workers roster')
    settings['expected_retry_workers'] = list(retry_workers)
    work = Path(settings['local_workdir']).resolve()
    work.mkdir(parents=True, exist_ok=True)
    state_file = work / 'watch_status.json'
    stop_file = work / 'STOP'
    profile_path = Path(settings['profile_config']).resolve()
    _profile, profile_binding = _profile_binding(profile_path)
    if 'pilot_request' in settings and settings['pilot_request'] is None:
        raise ValueError('watch pilot_request must be an explicit request path when present')
    _pilot, pilot_binding = cycle._load_pilot_request(settings.get('pilot_request'), profile_path)
    if pilot_binding is not None:
        settings['pilot_request'] = pilot_binding['request_path']
    stop_settings = dict(settings)
    stop_settings['_bound_scope'] = profile_binding['scope']
    launch_id = uuid.uuid4().hex
    attempts = work / 'watch_attempts'
    attempts.mkdir(parents=True, exist_ok=True)
    attempt_file = attempts / f'{launch_id}.json'
    prior_state = pb.read_json(state_file) if state_file.is_file() else None
    prior_sha256 = pb.file_sha256(state_file) if state_file.is_file() else None
    attempt = {'schema_version': 1, 'launch_id': launch_id, 'started_utc': _utc(),
               'settings_path': str(settings_file), 'settings_sha256': bound_settings_sha256,
               'profile_binding': profile_binding,
               'prior_watch_status': prior_state,
               'prior_watch_status_sha256': prior_sha256}
    state = {'schema_version': 1, 'status': 'starting', 'started_utc': _utc(),
             'launch_id': launch_id, 'attempt_record': str(attempt_file),
              'settings_path': str(settings_file), 'settings_sha256': bound_settings_sha256,
              'profile_binding': profile_binding,
              'expected_retry_workers': list(retry_workers),
              'interval_seconds': interval, 'completed_cycles': 0, 'hfss_started_here': False}
    if pilot_binding is not None:
        attempt['pilot_request'] = pilot_binding
        state['pilot_request'] = pilot_binding
    _write_state(state_file, attempt_file, attempt, state)
    due = monotonic()
    try:
        while max_cycles is None or state['completed_cycles'] < max_cycles:
            if stop_file.exists():
                state.update(status='stopped', stop_reason='local_STOP', finished_utc=_utc())
                break
            remaining = due - monotonic()
            if remaining > 0:
                sleep(min(30.0, remaining))
                continue
            _require_campaign_binding(settings_file, bound_settings_sha256, profile_binding,
                                      pilot_binding)
            stopped = campaign_stop(stop_settings)
            if stopped:
                state.update(status='stopped', stop_reason=stopped, finished_utc=_utc())
                break
            started = monotonic()
            state.update(status='running_cycle', cycle_started_utc=_utc())
            _write_state(state_file, attempt_file, attempt, state)
            receipt_path = None
            try:
                receipt_path = Path(prepare(**settings))
                receipt = pb.read_json(receipt_path)
                _require_campaign_binding(settings_file, bound_settings_sha256, profile_binding,
                                          pilot_binding)
                stopped = 'local_STOP' if stop_file.exists() else campaign_stop(stop_settings)
                if stopped:
                    state.update(status='stopped', stop_reason=stopped, finished_utc=_utc(),
                                 prepared_receipt=str(receipt_path.resolve()))
                    break
                _validate_cycle_receipt(receipt, retry_workers, pilot_binding)
                if receipt['status'] in ('prepared', 'dispatching'):
                    # Preparation produces the concrete immutable proposal. The user
                    # already authorized continuing this bounded private campaign.
                    dispatch(receipt_path)
                    receipt = pb.read_json(receipt_path)
                    _validate_cycle_receipt(receipt, retry_workers, pilot_binding)
                if receipt['status'] not in ('idle', 'dispatched'):
                    raise ValueError(f"controller did not complete: {receipt['status']}")
            except cycle.LiveSnapshotChanged as exc:
                number = int(state.get('deferred_retry_count', 0)) + 1
                retry_dir = work / 'deferred_retries'
                retry_dir.mkdir(parents=True, exist_ok=True)
                retry_path = retry_dir / f'{launch_id}-{number:04d}.json'
                event = {'schema_version': 1, 'event': 'live_snapshot_changed',
                         'deferred_utc': _utc(), 'error': str(exc),
                         'exception_type': 'LiveSnapshotChanged',
                         'cycle_started_utc': state['cycle_started_utc'],
                         'controller_receipt': (str(receipt_path.resolve())
                                                if receipt_path is not None else None)}
                pb.atomic_json(retry_path, event)
                event_binding = {'path': str(retry_path.resolve()),
                                 'sha256': pb.file_sha256(retry_path), **event}
                state.update(status='waiting', deferred_retry_count=number,
                             last_deferred_retry=event_binding)
                due = _next_scheduled_due(started, monotonic(), interval)
                state['next_cycle_in_seconds'] = max(0.0, due - monotonic())
                _write_state(state_file, attempt_file, attempt, state)
                continue
            state['completed_cycles'] += 1
            state.update(last_receipt=str(receipt_path.resolve()),
                         last_receipt_sha256=pb.file_sha256(receipt_path),
                         last_cycle_seconds=monotonic() - started,
                         valid_unique=receipt['valid_unique'],
                         latest_data_version=receipt['latest_data_version'],
                          status='waiting', last_completed_utc=_utc())
            if receipt['valid_unique'] == receipt['target_valid_unique']:
                state.update(status='symmetry_collection_complete', finished_utc=_utc(),
                             next_phase='R81 engineering checks remain; no filter jobs launched here')
                break
            now = monotonic()
            due = _next_scheduled_due(started, now, interval)
            state['next_cycle_in_seconds'] = max(0.0, due - monotonic())
            _write_state(state_file, attempt_file, attempt, state)
        else:
            state.update(status='bounded_run_complete', finished_utc=_utc())
    except BaseException as exc:
        state.update(status='failed', finished_utc=_utc(), error=repr(exc),
                     traceback=traceback.format_exc())
        _write_state(state_file, attempt_file, attempt, state)
        raise
    _write_state(state_file, attempt_file, attempt, state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', required=True, type=Path)
    parser.add_argument('--max-cycles', type=int)
    args = parser.parse_args()
    print(json.dumps(run(args.settings, max_cycles=args.max_cycles), ensure_ascii=False))


if __name__ == '__main__':
    main()
