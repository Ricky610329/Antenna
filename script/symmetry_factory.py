"""Scoped R80 factory policy and read-only, physically audited progress snapshots.

The rolling factory is separate from exploration's original three-batch pilot.
Workers continue to consume the same named measurement profile.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import statistics

from antenna.measurement import measurement_id, score_spec_id, validate_measurement, validate_score_spec
from script import profiled_batch as pb
from script.batch_scope import validate_scope


def validate_factory_config(config):
    exp = dict(config.get('exploration', {}))
    expected = {'mode', 'target_valid_unique', 'seed', 'wave_size', 'shard_size',
                'update_every_valid_unique', 'max_guided_outstanding',
                'prepared_blind_pool_size', 'guided_priority', 'blind_priority',
                'selection_fractions', 'historical_data_role',
                'historical_radiation_role', 'performance_gate'}
    if set(exp) != expected:
        raise ValueError('factory policy fields are missing or unknown')
    if exp['mode'] != 'symmetry_factory_v1':
        raise ValueError('wrong factory mode')
    for key in ('target_valid_unique', 'wave_size', 'shard_size', 'update_every_valid_unique',
                'max_guided_outstanding', 'prepared_blind_pool_size',
                'guided_priority', 'blind_priority'):
        if type(exp[key]) is not int or exp[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if type(exp['seed']) is not int or exp['seed'] < 0:
        raise ValueError('seed must be a nonnegative integer')
    if exp['wave_size'] % exp['shard_size'] or exp['max_guided_outstanding'] < exp['wave_size']:
        raise ValueError('wave/shard/outstanding budgets are inconsistent')
    if not exp['guided_priority'] < exp['blind_priority'] < 8:
        raise ValueError('priorities must avoid the legacy tier2 yield-before-save path')
    if exp['selection_fractions'] != {'performance_lcb': .4, 'disagreement': .3, 'blind': .3}:
        raise ValueError('factory selection fractions must be 40/30/30')
    if (exp['performance_gate'] is not False or
            exp['historical_data_role'] != 'pretraining_prior' or
            exp['historical_radiation_role'] != 'soft_signal_pending_prospective_validation'):
        raise ValueError('invalid factory data/prediction roles')
    measurement = validate_measurement(config['measurement'])
    score = validate_score_spec(config['score_spec'], measurement=measurement)
    if (config['port'] != 'single' or measurement['port'] != 'single' or
            measurement['geometry']['diag_bridge_w'] != .1 or score['bands']):
        raise ValueError('factory requires observation-only single-port 0.1 mm profile')
    validate_scope(config['scope'])
    return exp


def _child(root, name):
    if not isinstance(name, str) or not name or Path(name).name != name or any(c in name for c in '/\\:'):
        raise ValueError('unsafe queue path')
    path = root / name
    path.resolve().relative_to(root.resolve())
    return path


def snapshot(dataset_root, scope):
    """Audit saved successes, retain partial/error counts, deduplicate physical patterns."""
    root = Path(dataset_root).resolve()
    scope = validate_scope(scope)
    if not scope:
        raise ValueError('explicit scope required')
    queue_hash = pb.file_sha256(root / 'jobs.json')
    jobs = [j for j in pb.read_json(root / 'jobs.json') if j.get('scope') == scope]
    items, patterns, times, completed = [], set(), [], []
    measurement_ids = set()
    for job in jobs:
        inp, store = _child(root, job['input']), _child(root, job['store'])
        input_hashes = {name: pb.file_sha256(inp / name) for name in
                        ('config.yaml', 'measurement.json', 'score_spec.json', 'manifest.json')}
        cfg = pb.load_profile_config(inp / 'config.yaml')
        if cfg is None or cfg.scope != scope:
            raise ValueError('job configuration scope mismatch')
        expected = pb.read_json(inp / 'manifest.json')
        mid = measurement_id(cfg.measurement)
        if (measurement_id(pb.read_json(inp / 'measurement.json')) != mid or
                score_spec_id(pb.read_json(inp / 'score_spec.json')) != score_spec_id(cfg.score_spec)):
            raise ValueError('input measurement/score differs from configuration')
        measurement_ids.add(mid)
        markers = {}
        for suffix in ('claim', 'done', 'fail'):
            path = root / 'jobs_state' / (job['store'] + '.' + suffix)
            if path.exists():
                markers[suffix] = {'mtime_utc': datetime.fromtimestamp(
                    path.stat().st_mtime, timezone.utc).isoformat(), 'payload': pb.read_json(path)}
        results = {}
        result_path = store / 'results.json'
        source_hashes = None
        if result_path.exists():
            source_hashes = {name: pb.file_sha256(store / name) for name in
                             ('measurement.json', 'score_spec.json', 'manifest.json', 'results.json')}
            if measurement_id(pb.read_json(store / 'measurement.json')) != mid:
                raise ValueError('store measurement differs from queued input')
            if score_spec_id(pb.read_json(store / 'score_spec.json')) != score_spec_id(cfg.score_spec):
                raise ValueError('store score definition differs from queued input')
            if pb.read_json(store / 'manifest.json') != expected:
                raise ValueError('store manifest differs from queued input')
            results = pb.validate_store(store, require_complete=False)
            if any(pb.file_sha256(store / name) != digest for name, digest in source_hashes.items()):
                raise ValueError('live store changed during snapshot; retry with a fresh snapshot')
        ok = {key: value for key, value in results.items()
              if value.get('status') == 'ok' and 'error' not in value}
        patterns.update((mid, value['pattern_sha256']) for value in ok.values())
        times.extend(value['time_s'] for value in ok.values())
        if len(ok) == len(expected):
            completed.append(str(store))
        if any(pb.file_sha256(inp / name) != digest for name, digest in input_hashes.items()):
            raise ValueError('input changed during snapshot; retry with a fresh snapshot')
        items.append({'store': job['store'], 'input': job['input'], 'prio': job['prio'],
                      'expected': len(expected), 'audited_ok': len(ok),
                      'errors': {key: value for key, value in results.items() if 'error' in value},
                      'markers': markers,
                      'input_metadata_sha256': input_hashes,
                      'source_metadata_sha256': source_hashes})
    if len(measurement_ids) > 1:
        raise ValueError('one factory snapshot cannot mix different measurements')
    if pb.file_sha256(root / 'jobs.json') != queue_hash:
        raise ValueError('queue changed during snapshot; retry with a fresh snapshot')
    return {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'scope': scope,
            'dataset_root': str(root), 'jobs_sha256': queue_hash,
            'measurement_ids': sorted(measurement_ids),
            'valid_unique_patterns': len(patterns), 'audited_ok_including_repeats': len(times),
            'solver_time_median_s': statistics.median(times) if times else None,
            'completed_stores': completed, 'jobs': items,
            'progress_limit': 'Snapshot of saved artifacts; claims do not prove a live worker heartbeat.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--scope', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('snapshot output must be fresh')
    result = snapshot(args.dataset_root, args.scope)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pb.atomic_json(args.output, result)
    print(f"audited unique={result['valid_unique_patterns']}; "
          f"completed stores={len(result['completed_stores'])}; output={args.output}")


if __name__ == '__main__':
    main()
