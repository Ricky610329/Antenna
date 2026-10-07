from copy import deepcopy

import pytest

from script.profiled_batch import load_profile_config
from script.symmetry_factory import validate_factory_config, _child


def test_snapshot_counts_repeated_geometry_once_and_calls_physical_audit(tmp_path, monkeypatch):
    from script import profiled_batch as pb
    from script.symmetry_factory import snapshot
    cfg = load_profile_config('configs/single_r80_symmetry_factory.yaml')
    rows = [{'id': 'a'}, {'id': 'repeat'}]
    result = {
        'a': {'status': 'ok', 'pattern_sha256': 'same-pattern', 'time_s': 120},
        'repeat': {'status': 'ok', 'pattern_sha256': 'same-pattern', 'time_s': 180},
    }
    for sub in ('input', 'result', 'jobs_state'):
        (tmp_path / sub).mkdir()
    pb.atomic_json(tmp_path / 'jobs.json', [{'input': 'input', 'store': 'result',
                                           'prio': 1, 'scope': cfg.scope}])
    for sub in ('input', 'result'):
        for name, data in [('measurement', cfg.measurement), ('score_spec', cfg.score_spec),
                           ('manifest', rows)]:
            pb.atomic_json(tmp_path / sub / (name + '.json'), data)
    (tmp_path / 'input/config.yaml').write_text('fixture configuration loader is injected')
    pb.atomic_json(tmp_path / 'result/results.json', result)
    monkeypatch.setattr(pb, 'load_profile_config', lambda path: cfg)
    calls = []
    def audit(path, *, require_complete):
        calls.append((path, require_complete))
        return result
    monkeypatch.setattr(pb, 'validate_store', audit)
    observed = snapshot(tmp_path, cfg.scope)
    assert observed['valid_unique_patterns'] == 1
    assert observed['audited_ok_including_repeats'] == 2
    assert observed['solver_time_median_s'] == 150
    assert calls == [(tmp_path / 'result', False)]
    assert observed['jobs'][0]['source_metadata_sha256']['results.json'] == pb.file_sha256(
        tmp_path / 'result/results.json')


def test_rolling_policy_keeps_worker_measurement_and_unique_budget():
    cfg = vars(load_profile_config('configs/single_r80_symmetry_factory.yaml'))
    exp = validate_factory_config(cfg)
    assert exp['target_valid_unique'] == 10240
    assert exp['update_every_valid_unique'] == 48
    assert exp['guided_priority'] < exp['blind_priority'] < 8


@pytest.mark.parametrize('key,value', [('blind_priority', 8), ('target_valid_unique', 9999),
                                     ('performance_gate', True), ('wave_size', 47)])
def test_rolling_policy_rejects_waste_or_silent_budget_change(key, value):
    cfg = deepcopy(vars(load_profile_config('configs/single_r80_symmetry_factory.yaml')))
    cfg['exploration'][key] = value
    with pytest.raises(ValueError):
        validate_factory_config(cfg)


def test_factory_queue_path_cannot_escape_private_root(tmp_path):
    with pytest.raises(ValueError):
        _child(tmp_path, '../public')
