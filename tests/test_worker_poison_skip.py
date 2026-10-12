"""Execution skips must free workers without promoting incomplete truth."""
import pytest

from antenna.utils.utils import Path
from script import dedust, profiled_batch as pb
from tests import test_scoped_worker_recovery as recovery

WORKER_IP = "140.123.106.216"


def test_scoped_worker_continues_four_poison_jobs_then_good_without_simulator(tmp_path, monkeypatch):
    dataset = tmp_path / 'dataset'
    dataset.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dedust, 'DATASET_PATH', Path(str(dataset)))
    monkeypatch.setattr('antenna.utils.web.get_local_ip', lambda: WORKER_IP)
    monkeypatch.setattr('antenna.patch.SinglePortRadSimulator',
                        lambda **_kw: pytest.fail('settled results must not open HFSS'))
    jobs = []
    for index in range(5):
        name = f'job{index}'
        ids = [f'ok{index}', f'bad{index}'] if index < 4 else ['next']
        input_root, rows, cfg = recovery._write_input(dataset, name + '_input', ids)
        errors = {f'bad{index}': {'error': 'DISP_E_EXCEPTION old COM', 'attempts': 3}} if index < 4 else {}
        recovery._write_store(dataset, input_root, name, rows, cfg, errors=errors)
        jobs.append({'input': name + '_input', 'store': name, 'prio': index,
                     'scope': recovery.SCOPE, 'config': name + '_input/config.yaml'})
    pb.atomic_json(dataset / 'jobs.json', jobs)
    calls = []
    actual_run = dedust.run

    def observe_run(args):
        calls.append(args.store)
        value = actual_run(args)
        if args.store == 'job4':
            (dataset / 'jobs_state' / f'STOP_{recovery.SCOPE}').write_text('stop', encoding='utf-8')
        return value

    monkeypatch.setattr(dedust, 'run', observe_run)
    dedust.worker(recovery._worker_args())
    assert calls == [f'job{index}' for index in range(5)]
    for index in range(5):
        state = dataset / 'jobs_state'
        done = pb.read_json(state / f'job{index}.done')
        assert done['errors'] == (1 if index < 4 else 0)
        assert done['error_ids'] == ([f'bad{index}'] if index < 4 else [])
        assert not (state / f'job{index}.fail').exists()
        if index < 4:
            assert pb.read_json(dataset / f'job{index}/results.json')[f'bad{index}']['status'] == 'skipped'


@pytest.mark.parametrize('error', [
    {'error': 'DISP_E_EXCEPTION COM', 'attempts': True},
    {'error': 'profile_observation_invalid: missing rad', 'attempts': 3,
     'error_kind': 'profile_observation_invalid'},
])
def test_corrupt_or_non_hfss_error_stops_before_simulator(tmp_path, monkeypatch, error):
    dataset = tmp_path / 'dataset'
    dataset.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dedust, 'DATASET_PATH', Path(str(dataset)))
    monkeypatch.setattr('antenna.utils.web.get_local_ip', lambda: WORKER_IP)
    monkeypatch.setattr('antenna.patch.SinglePortRadSimulator',
                        lambda **_kw: pytest.fail('invalid result must stop before HFSS'))
    input_root, rows, cfg = recovery._write_input(dataset, 'invalid_input', ['bad'])
    store = recovery._write_store(dataset, input_root, 'invalid', rows, cfg, errors={'bad': error})
    before = (store / 'results.json').read_bytes()
    pb.atomic_json(dataset / 'jobs.json', [{'input': 'invalid_input', 'store': 'invalid',
        'prio': 1, 'scope': recovery.SCOPE, 'config': 'invalid_input/config.yaml'}])
    with pytest.raises(SystemExit, match='完整性驗證失敗'):
        dedust.worker(recovery._worker_args(once=True))
    assert (store / 'results.json').read_bytes() == before
    assert not (dataset / 'jobs_state/invalid.done').exists()
    assert (dataset / 'jobs_state/invalid.fail').exists()
