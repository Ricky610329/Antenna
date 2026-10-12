"""Execution skips must free workers without promoting incomplete truth."""
import sys

import pytest

from antenna.utils.utils import Path
from script import dedust, profiled_batch as pb
from script import r81_worker_entry as base
from script import r81_formal_sharded_entry as formal
from tests import test_scoped_worker_recovery as recovery
from tests import test_r81_worker_entry as entry_fixture
from tests import test_r81_formal_sharded_entry as shard_fixture


def _skip_one(dataset, entry):
    store = dataset / entry['job']['store']
    results = pb.read_json(store / 'results.json')
    row_id = next(iter(results))
    results[row_id] = {'error': 'COM fixture failure', 'error_kind': 'hfss_simulation',
                       'attempts': 3, 'status': 'skipped',
                       'skip_reason': 'hfss_attempts_exhausted'}
    pb.atomic_json(store / 'results.json', results)
    shard_fixture._mark_done(dataset, entry)
    done_path = dataset / 'jobs_state' / f"{entry['job']['store']}.done"
    done = pb.read_json(done_path)
    done.update(errors=1, error_ids=[row_id])
    pb.atomic_json(done_path, done)
    return done_path, row_id


def test_scoped_worker_continues_four_poison_jobs_then_good_without_simulator(tmp_path, monkeypatch):
    dataset = tmp_path / 'dataset'
    dataset.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dedust, 'DATASET_PATH', Path(str(dataset)))
    monkeypatch.setattr('antenna.utils.web.get_local_ip', lambda: base.EXPECTED_WORKERS[0])
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
    monkeypatch.setattr('antenna.utils.web.get_local_ip', lambda: base.EXPECTED_WORKERS[0])
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


@pytest.fixture
def runtime_binding(monkeypatch):
    monkeypatch.setattr(base, '_git_is_ancestor', lambda _commit: True)
    monkeypatch.setattr(base, '_git_path_clean', lambda _commit, _relative: True)
    monkeypatch.setattr(base, '_git_blob_sha', lambda _commit, relative: pb.file_sha256(base.REPO / relative))
    entry_fixture._CENSUS_FIXTURES.clear()

    def census(binding, _commit):
        path = entry_fixture._CENSUS_FIXTURES[binding['repo_path']]
        assert pb.file_sha256(path) == binding['sha256']
        return path, pb.read_json(path)

    monkeypatch.setattr(base, '_bound_census_status', census)


def test_engineering_queue_runs_siblings_then_reports_skips_without_promoting_truth(tmp_path, runtime_binding):
    path, release = entry_fixture._release(tmp_path, 'EngineeringOnly')
    dataset = base.Path(release['dataset_root'])
    entries = release['queue']['jobs']
    done_path, _ = _skip_one(dataset, entries[0])
    digest = pb.file_sha256(path)
    assert base.queue_status(path, digest, 'EngineeringOnly') == 'active'
    for entry in entries[1:]:
        shard_fixture._mark_done(dataset, entry)
    assert base.queue_status(path, digest, 'EngineeringOnly') == 'terminal_skips'
    with pytest.raises(ValueError):
        pb.validate_store(dataset / entries[0]['job']['store'])
    done = pb.read_json(done_path)
    done.update(errors=0, error_ids=[])
    pb.atomic_json(done_path, done)
    with pytest.raises(ValueError, match='execution terminal'):
        base.queue_status(path, digest, 'EngineeringOnly')


def test_formal_skip_keeps_other_shards_running_and_never_merges(tmp_path, runtime_binding):
    path, release, shards = shard_fixture._release(tmp_path)
    dataset = base.Path(release['dataset_root'])
    shard_fixture._complete_shard(dataset, shards[0])
    _skip_one(dataset, shards[0])
    digest = pb.file_sha256(path)
    assert formal.queue_status(path, digest) == 'active'
    for shard in shards[1:]:
        shard_fixture._complete_shard(dataset, shard)
    assert formal.queue_status(path, digest) == 'terminal_skips'
    with pytest.raises(ValueError, match='all three native shards'):
        formal.aggregate_batch(path, digest, 1)
    assert not (dataset / 'dedust_r81b1').exists()


@pytest.mark.parametrize('entry', [base, formal])
def test_skipped_queue_cli_exits_cleanly(entry, monkeypatch):
    monkeypatch.setattr(entry, 'queue_status', lambda *_args: 'terminal_skips')
    argv = ['entry', '--release', 'unused.json', '--release-sha256', 'a' * 64, '--queue-status']
    if entry is base:
        argv.extend(['--stage', 'EngineeringOnly'])
    monkeypatch.setattr(sys, 'argv', argv)
    if entry is base:
        with pytest.raises(SystemExit) as ended:
            entry.main()
        assert ended.value.code == 0
    else:
        assert entry.main() == 0
