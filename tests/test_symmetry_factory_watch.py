import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from script import profiled_batch as pb
from script import symmetry_factory_cycle as cycle
from script.symmetry_factory_watch import run, _exclusive_watch


REPO = Path(__file__).resolve().parents[1]
PROFILE = REPO / 'configs' / 'single_r80_symmetry_factory.yaml'
RETRY_WORKERS = ['140.123.106.216', '140.123.106.218', '140.123.106.37']


def settings(tmp_path, interval=1800, *, name='settings.json', local='work', profile=PROFILE):
    (tmp_path / 'training').mkdir(exist_ok=True)
    (tmp_path / 'dataset' / 'jobs_state').mkdir(parents=True, exist_ok=True)
    path = tmp_path / name
    path.write_text(json.dumps(dict(local_workdir=str(tmp_path / local),
                                    dataset_root=str(tmp_path / 'dataset'),
                                    profile_config=str(profile),
                                    training_workdir=str(tmp_path / 'training'),
                                    seed_inputs=['seed'], interval_seconds=interval,
                                    expected_retry_workers=RETRY_WORKERS)))
    return path


def test_waits_30_minutes_and_dispatches_each_prepared_cycle(tmp_path):
    path = settings(tmp_path)
    clock = [0.0]
    starts, dispatched, sleeps = [], [], []

    def prepare(**kw):
        starts.append(clock[0])
        p = Path(kw['local_workdir']) / f'receipt{len(starts)}.json'
        p.write_text(json.dumps(dict(status='prepared', valid_unique=48 * len(starts),
                                    target_valid_unique=10240, latest_data_version=len(starts),
                                    expected_retry_workers=RETRY_WORKERS)))
        clock[0] += 120
        return p

    def dispatch(p):
        value = json.loads(p.read_text())
        value['status'] = 'dispatched'
        p.write_text(json.dumps(value))
        dispatched.append(p)

    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    result = run(path, prepare=prepare, dispatch=dispatch, sleep=sleep,
                 monotonic=lambda: clock[0], max_cycles=2, campaign_stop=lambda _: None)
    assert starts == [0, 1800]
    assert len(dispatched) == 2 and max(sleeps) <= 30
    assert result['status'] == 'bounded_run_complete'


@pytest.mark.parametrize('value', [None, [], ['worker', 'worker']])
def test_watch_requires_explicit_nonempty_unique_retry_worker_roster(tmp_path, value):
    path = settings(tmp_path)
    payload = json.loads(path.read_text())
    if value is None:
        payload.pop('expected_retry_workers')
    else:
        payload['expected_retry_workers'] = value
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match='expected_retry_workers'):
        run(path, prepare=lambda **_: pytest.fail('invalid roster must not prepare'))


def test_watch_rejects_present_null_pilot_request(tmp_path):
    path = settings(tmp_path)
    payload = json.loads(path.read_text())
    payload['pilot_request'] = None
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='explicit request path'):
        run(path, prepare=lambda **_: pytest.fail('null request must not prepare'))


def test_watch_binds_pilot_request_bytes_and_rejects_in_place_change_before_dispatch(
        tmp_path, monkeypatch):
    path = settings(tmp_path)
    request_path = tmp_path / 'pilot-request.json'
    request_path.write_text('{"version": 1}', encoding='utf-8')
    payload = json.loads(path.read_text())
    payload['pilot_request'] = str(request_path)
    path.write_text(json.dumps(payload))
    pilot_id = 'a' * 64

    def load(value, _profile):
        if value is None:
            return None, None
        supplied = Path(value).resolve()
        binding = {'request_path': str(supplied),
                   'request_sha256': hashlib.sha256(supplied.read_bytes()).hexdigest(),
                   'pilot_id': pilot_id}
        return SimpleNamespace(binding=binding, pilot_id=pilot_id), binding

    monkeypatch.setattr(cycle, '_load_pilot_request', load)
    dispatched = []

    def prepare(**kw):
        assert kw['pilot_request'] == str(request_path.resolve())
        _, binding = load(request_path, PROFILE)
        receipt = Path(kw['local_workdir']) / 'receipt.json'
        receipt.write_text(json.dumps({
            'status': 'prepared', 'valid_unique': 100, 'target_valid_unique': 5000,
            'latest_data_version': 1, 'expected_retry_workers': RETRY_WORKERS,
            'pilot_request': binding,
        }))
        request_path.write_text('{"version": 2}', encoding='utf-8')
        return receipt

    with pytest.raises(ValueError, match='pilot request presence or exact binding changed'):
        run(path, prepare=prepare, dispatch=lambda receipt: dispatched.append(receipt),
            campaign_stop=lambda _: None)
    assert not dispatched
    state = json.loads((tmp_path / 'work/watch_status.json').read_text())
    assert state['status'] == 'failed' and state['pilot_request']['pilot_id'] == pilot_id


def test_watch_rejects_receipt_that_drops_bound_pilot_before_dispatch(tmp_path, monkeypatch):
    path = settings(tmp_path)
    request_path = tmp_path / 'pilot-request.json'
    request_path.write_text('{}', encoding='utf-8')
    payload = json.loads(path.read_text())
    payload['pilot_request'] = str(request_path)
    path.write_text(json.dumps(payload))
    binding = {'request_path': str(request_path.resolve()),
               'request_sha256': hashlib.sha256(request_path.read_bytes()).hexdigest(),
               'pilot_id': 'b' * 64}
    monkeypatch.setattr(cycle, '_load_pilot_request', lambda value, profile: (
        SimpleNamespace(binding=binding, pilot_id=binding['pilot_id']), dict(binding)))
    dispatched = []

    def prepare(**kw):
        receipt = Path(kw['local_workdir']) / 'receipt.json'
        receipt.write_text(json.dumps({
            'status': 'prepared', 'valid_unique': 100, 'target_valid_unique': 5000,
            'latest_data_version': 1, 'expected_retry_workers': RETRY_WORKERS,
        }))
        return receipt

    with pytest.raises(ValueError, match='pilot request differs'):
        run(path, prepare=prepare, dispatch=lambda receipt: dispatched.append(receipt),
            campaign_stop=lambda _: None)
    assert not dispatched


def test_stop_and_failure_do_not_dispatch(tmp_path):
    path = settings(tmp_path)
    work = tmp_path / 'work'
    work.mkdir()
    (work / 'STOP').touch()
    calls = []
    assert run(path, prepare=lambda **kw: calls.append(kw))['status'] == 'stopped'
    assert not calls
    (work / 'STOP').unlink()

    def fail(**kw):
        raise ValueError('corrupt immutable input')

    with pytest.raises(ValueError, match='corrupt'):
        run(path, prepare=fail, dispatch=lambda p: calls.append(p), campaign_stop=lambda _: None)
    assert json.loads((work / 'watch_status.json').read_text())['status'] == 'failed'
    assert not calls


def test_stops_after_symmetry_collection_without_filter_dispatch(tmp_path):
    path = settings(tmp_path)

    def prepare(**kw):
        p = Path(kw['local_workdir']) / 'receipt.json'
        p.write_text(json.dumps(dict(status='idle', valid_unique=10240,
                                    target_valid_unique=10240, latest_data_version=210,
                                    pending_unique=0,
                                    expected_retry_workers=RETRY_WORKERS)))
        return p

    result = run(path, prepare=prepare, dispatch=lambda p: pytest.fail('unexpected dispatch'),
                 campaign_stop=lambda _: None)
    assert result['status'] == 'symmetry_collection_complete'
    assert result['completed_cycles'] == 1 and not result['hfss_started_here']


def test_rejects_receipt_above_exact_target(tmp_path):
    path = settings(tmp_path)
    dispatched = []

    def prepare(**kw):
        receipt = Path(kw['local_workdir']) / 'receipt.json'
        receipt.write_text(json.dumps({
            'status': 'prepared', 'valid_unique': 5001, 'target_valid_unique': 5000,
            'latest_data_version': 100, 'expected_retry_workers': RETRY_WORKERS,
        }))
        return receipt

    with pytest.raises(ValueError, match='exceeds exact'):
        run(path, prepare=prepare, dispatch=lambda receipt: dispatched.append(receipt),
            campaign_stop=lambda _: None)
    state = json.loads((tmp_path / 'work' / 'watch_status.json').read_text())
    assert state['status'] == 'failed' and not dispatched


def test_does_not_complete_exact_target_with_residual_pending(tmp_path):
    path = settings(tmp_path)

    def prepare(**kw):
        receipt = Path(kw['local_workdir']) / 'receipt.json'
        receipt.write_text(json.dumps({
            'status': 'idle', 'valid_unique': 5000, 'pending_unique': 1,
            'target_valid_unique': 5000, 'latest_data_version': 100,
            'expected_retry_workers': RETRY_WORKERS,
        }))
        return receipt

    with pytest.raises(ValueError, match='residual pending'):
        run(path, prepare=prepare, campaign_stop=lambda _: None)


def test_rejects_wrong_receipt_roster_before_dispatch(tmp_path):
    path = settings(tmp_path)
    dispatched = []

    def prepare(**kw):
        receipt = Path(kw['local_workdir']) / 'receipt.json'
        receipt.write_text(json.dumps({
            'status': 'prepared', 'valid_unique': 4999, 'target_valid_unique': 5000,
            'latest_data_version': 100, 'expected_retry_workers': RETRY_WORKERS[:-1],
        }))
        return receipt

    with pytest.raises(ValueError, match='roster differs'):
        run(path, prepare=prepare, dispatch=lambda receipt: dispatched.append(receipt),
            campaign_stop=lambda _: None)
    assert not dispatched


def test_rejects_frequent_polling(tmp_path):
    with pytest.raises(ValueError, match='1800'):
        run(settings(tmp_path, interval=60))


def test_kernel_lock_prevents_duplicate_watcher_and_releases(tmp_path):
    lock = tmp_path / 'watch.lock'
    with _exclusive_watch(lock):
        with pytest.raises(RuntimeError, match='already owns'):
            with _exclusive_watch(lock):
                pytest.fail('duplicate watcher acquired lock')
    with _exclusive_watch(lock):
        pass


def test_stop_arriving_during_prepare_prevents_dispatch(tmp_path):
    path = settings(tmp_path)

    def prepare(**kw):
        work = Path(kw['local_workdir'])
        (work / 'STOP').touch()
        receipt = work / 'receipt.json'
        receipt.write_text(json.dumps({'status': 'prepared'}))
        return receipt

    result = run(path, prepare=prepare, campaign_stop=lambda _: None,
                 dispatch=lambda p: pytest.fail('STOP must block dispatch'))
    assert result['status'] == 'stopped' and result['completed_cycles'] == 0


@pytest.mark.parametrize('global_stop', [True, False])
def test_campaign_global_and_scope_stop_prevent_prepare(tmp_path, global_stop):
    path = settings(tmp_path)
    scope = pb.load_profile_config(PROFILE).scope
    name = 'STOP' if global_stop else f'STOP_{scope}'
    (tmp_path / 'dataset' / 'jobs_state' / name).touch()
    calls = []

    result = run(path, prepare=lambda **kw: calls.append(kw))

    assert result['status'] == 'stopped' and result['stop_reason'] == name
    assert not calls


def test_settings_change_during_prepare_fails_before_dispatch(tmp_path):
    path = settings(tmp_path)
    dispatched = []

    def prepare(**kw):
        value = json.loads(path.read_text())
        value['interval_seconds'] = 3600
        path.write_text(json.dumps(value))
        receipt = Path(kw['local_workdir']) / 'receipt.json'
        receipt.write_text(json.dumps({'status': 'prepared'}))
        return receipt

    with pytest.raises(ValueError, match='settings changed'):
        run(path, prepare=prepare, dispatch=lambda p: dispatched.append(p),
            campaign_stop=lambda _: None)
    assert not dispatched
    assert json.loads((tmp_path / 'work' / 'watch_status.json').read_text())['status'] == 'failed'


def test_profile_change_during_prepare_fails_before_dispatch(tmp_path):
    profile = tmp_path / 'profile.yaml'
    profile.write_bytes(PROFILE.read_bytes())
    path = settings(tmp_path, profile=profile)
    dispatched = []

    def prepare(**kw):
        profile.write_bytes(profile.read_bytes() + b'\n# changed during prepare\n')
        receipt = Path(kw['local_workdir']) / 'receipt.json'
        receipt.write_text(json.dumps({'status': 'prepared'}))
        return receipt

    with pytest.raises(ValueError, match='profile changed'):
        run(path, prepare=prepare, dispatch=lambda p: dispatched.append(p),
            campaign_stop=lambda _: None)
    assert not dispatched


def test_long_cycle_skips_missed_tick_without_immediate_catchup(tmp_path):
    path = settings(tmp_path)
    clock = [0.0]
    starts = []

    def prepare(**kw):
        starts.append(clock[0])
        receipt = Path(kw['local_workdir']) / f'receipt-{len(starts)}.json'
        receipt.write_text(json.dumps({'status': 'idle', 'valid_unique': len(starts),
                                       'target_valid_unique': 10240,
                                       'latest_data_version': 0,
                                       'expected_retry_workers': RETRY_WORKERS}))
        clock[0] += 1900
        return receipt

    def sleep(seconds):
        clock[0] += seconds

    run(path, prepare=prepare, sleep=sleep, monotonic=lambda: clock[0], max_cycles=2,
        campaign_stop=lambda _: None)
    assert starts == [0.0, 3600.0]


def test_restart_attempt_preserves_prior_status_and_uses_new_launch_id(tmp_path):
    path = settings(tmp_path)

    def prepare(**kw):
        receipt = Path(kw['local_workdir']) / 'receipt.json'
        receipt.write_text(json.dumps({'status': 'idle', 'valid_unique': 1,
                                       'target_valid_unique': 10240,
                                       'latest_data_version': 0,
                                       'expected_retry_workers': RETRY_WORKERS}))
        return receipt

    first = run(path, prepare=prepare, max_cycles=1, campaign_stop=lambda _: None)
    second = run(path, prepare=prepare, max_cycles=1, campaign_stop=lambda _: None)
    attempt = json.loads(Path(second['attempt_record']).read_text())

    assert first['launch_id'] != second['launch_id']
    assert attempt['prior_watch_status']['launch_id'] == first['launch_id']
    assert attempt['prior_watch_status_sha256']


def test_same_training_workdir_lock_blocks_second_local_watcher(tmp_path):
    first_path = settings(tmp_path, name='first.json', local='work-a')
    second_path = settings(tmp_path, name='second.json', local='work-b')

    def prepare(**kw):
        with pytest.raises(RuntimeError, match='already owns'):
            run(second_path, prepare=lambda **_: pytest.fail('second watcher prepared'),
                max_cycles=1, campaign_stop=lambda _: None)
        receipt = Path(kw['local_workdir']) / 'receipt.json'
        receipt.write_text(json.dumps({'status': 'idle', 'valid_unique': 1,
                                       'target_valid_unique': 10240,
                                       'latest_data_version': 0,
                                       'expected_retry_workers': RETRY_WORKERS}))
        return receipt

    run(first_path, prepare=prepare, max_cycles=1, campaign_stop=lambda _: None)


def test_live_snapshot_race_defers_exactly_one_tick_then_recovers(tmp_path):
    path = settings(tmp_path)
    clock = [0.0]
    starts, dispatched = [], []

    def prepare(**kw):
        starts.append(clock[0])
        if len(starts) == 1:
            raise cycle.LiveSnapshotChanged('results changed during bounded scan')
        receipt = Path(kw['local_workdir']) / 'receipt.json'
        receipt.write_text(json.dumps({'status': 'prepared', 'valid_unique': 48,
                                       'target_valid_unique': 10240,
                                       'latest_data_version': 1,
                                       'expected_retry_workers': RETRY_WORKERS}))
        return receipt

    def dispatch(receipt):
        value = json.loads(receipt.read_text())
        value['status'] = 'dispatched'
        receipt.write_text(json.dumps(value))
        dispatched.append(receipt)

    def sleep(seconds):
        clock[0] += seconds

    result = run(path, prepare=prepare, dispatch=dispatch, sleep=sleep,
                 monotonic=lambda: clock[0], max_cycles=1,
                 campaign_stop=lambda _: None)
    event = result['last_deferred_retry']

    assert starts == [0.0, 1800.0]
    assert len(dispatched) == 1 and result['completed_cycles'] == 1
    assert result['deferred_retry_count'] == 1
    assert event['exception_type'] == 'LiveSnapshotChanged'
    assert 'results changed' in event['error']
    assert pb.file_sha256(Path(event['path'])) == event['sha256']


def test_non_live_snapshot_corruption_remains_fatal(tmp_path):
    path = settings(tmp_path)
    dispatched = []

    with pytest.raises(ValueError, match='immutable store corruption'):
        run(path, prepare=lambda **_: (_ for _ in ()).throw(
            ValueError('immutable store corruption')),
            dispatch=lambda receipt: dispatched.append(receipt),
            campaign_stop=lambda _: None)

    status = json.loads((tmp_path / 'work' / 'watch_status.json').read_text())
    assert status['status'] == 'failed'
    assert 'deferred_retry_count' not in status
    assert not dispatched
