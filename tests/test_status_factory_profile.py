import json
import os

import pytest

from script.status import _factory_scan


def test_factory_counts_observations_without_wm_and_preserves_legacy(tmp_path):
    (tmp_path / 'jobs_state').mkdir()
    (tmp_path / 'jobs.json').write_text(json.dumps([
        {'input': 'input', 'store': 'result', 'scope': 'ours'},
        {'input': 'other_input', 'store': 'other', 'scope': 'other'},
    ]))
    (tmp_path / 'input').mkdir()
    (tmp_path / 'input/manifest.json').write_text(json.dumps([{}, {}, {}, {}]))
    (tmp_path / 'result').mkdir()
    (tmp_path / 'result/results.json').write_text(json.dumps({
        'new': {'status': 'ok', 'measurement_id': 'a'},
        'legacy': {'wm': -2.0},
        'bad': {'status': 'ok', 'wm': 0, 'error': 'RPC'},
        'pending': {},
    }))
    (tmp_path / 'jobs_state/other.fail').write_text('{}')
    lines, alarms = _factory_scan(dataset_root=tmp_path, scope='ours')
    assert len(lines) == 1 and '2/4' in lines[0]
    assert alarms == []


def test_scoped_factory_keeps_own_failure_and_ignores_unrelated_done(tmp_path):
    (tmp_path / 'jobs_state').mkdir()
    (tmp_path / 'jobs.json').write_text(json.dumps([
        {'input': 'input', 'store': 'result', 'scope': 'ours'}]))
    (tmp_path / 'jobs_state/result.fail').write_text('{}')
    (tmp_path / 'jobs_state/other.done').write_text('{"errors":5}')
    lines, alarms = _factory_scan(dataset_root=tmp_path, scope='ours')
    assert len(lines) == 1 and 'result' in lines[0]
    assert len(alarms) == 1 and 'result.fail' in alarms[0]


@pytest.mark.parametrize('worker_continues', [True, False, None])
def test_partial_scoped_failure_reports_decision_without_claiming_worker_alive(tmp_path, worker_continues):
    state = tmp_path / 'jobs_state'
    state.mkdir()
    (tmp_path / 'jobs.json').write_text(json.dumps([
        {'input': 'input', 'store': 'partial', 'scope': 'ours'}]))
    failure = {'failure_kind': 'incomplete_profile_hfss'}
    if worker_continues is not None:
        failure['worker_continues'] = worker_continues
    (state / 'partial.fail').write_text(json.dumps(failure))
    _, alarms = _factory_scan(dataset_root=tmp_path, scope='ours')
    assert len(alarms) == 1 and '批次部分失敗' in alarms[0]
    assert '待跨機補測' in alarms[0] and '需另查近期進展' in alarms[0]
    assert ('原worker決定續跑' in alarms[0]) == (worker_continues is True)
    _, legacy_alarms = _factory_scan(dataset_root=tmp_path)
    assert len(legacy_alarms) == 1 and '工廠停機' in legacy_alarms[0]


@pytest.mark.parametrize('contents', ['not json', '[]', '{}'])
def test_unknown_or_corrupt_failure_retains_stop_alarm(tmp_path, contents):
    state = tmp_path / 'jobs_state'
    state.mkdir()
    (tmp_path / 'jobs.json').write_text(json.dumps([
        {'input': 'input', 'store': 'failed', 'scope': 'ours'}]))
    (state / 'failed.fail').write_text(contents)
    _, alarms = _factory_scan(dataset_root=tmp_path, scope='ours')
    assert len(alarms) == 1 and '工廠停機' in alarms[0]


def test_preserved_partial_claim_is_failed_recovery_not_a_stalled_active_solve(tmp_path):
    state = tmp_path / 'jobs_state'
    state.mkdir()
    (tmp_path / 'jobs.json').write_text(json.dumps([
        {'input': 'input', 'store': 'partial', 'scope': 'ours'}]))
    (state / 'partial.fail').write_text(json.dumps({
        'failure_kind': 'incomplete_profile_hfss', 'worker_continues': True}))
    claim = state / 'partial.claim'
    claim.write_text('{}')
    os.utime(claim, (1, 1))
    _, alarms = _factory_scan(dataset_root=tmp_path, scope='ours')
    assert len(alarms) == 1 and '批次部分失敗' in alarms[0]
    result = tmp_path / 'partial/results.json'
    result.parent.mkdir()
    result.write_text(json.dumps({'ok': {'status': 'ok'}, 'bad': {'error': 'COM'}}))
    os.utime(result, (1, 1))
    lines, alarms = _factory_scan(dataset_root=tmp_path, scope='ours')
    assert len(alarms) == 1 and '批次部分失敗' in alarms[0]
    assert '1/?' in lines[0]
