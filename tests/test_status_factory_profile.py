import json

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
