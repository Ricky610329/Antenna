"""Generated CPU fixtures exercise the complete opt-in dual feedback loop."""
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from script import exploration as ex
from script import filter_training as ft
from script import profiled_batch as pb


def _prepare(tmp_path, *, shared_canonical=False):
    cfg = yaml.safe_load((Path(__file__).parents[1] / 'configs/dual_r81_wide_filter.yaml').read_text(encoding='utf-8'))
    prior_root = tmp_path / 'prior'
    seeds = tmp_path / 'seeds'
    prior_root.mkdir(); seeds.mkdir()
    old_rows, seed_rows = [], []
    for i in range(16):
        pattern = torch.tensor(ex.enforce_dual_geometry(np.random.default_rng(i + 77).random((25, 25))), dtype=torch.float32)
        response = torch.stack((torch.full((17,), -12.0 - i / 10),
                                torch.full((17,), -2.0), torch.full((17,), -11.0)))
        source = prior_root / f'old{i}.pt'
        torch.save((pattern, response), source)
        old_rows.append({'id': f'old{i}', 'sample_file': source.name,
                         'sample_sha256': pb.file_sha256(source),
                         'pattern_sha256': pb.pattern_sha256(pattern),
                         'lineage_id': f'seed-family-{i}', 'canonical_group_id': f'seed-family-{i}',
                         'labels': ['S11', 'S21', 'S22'], 'freqs_ghz': np.arange(24, 32.1, .5).tolist(),
                         'geometry': cfg['measurement']['geometry'], 'provenance': {'kind': 'generated_fixture'}})
        if i < 12:
            torch.save(pattern, seeds / f'seed{i}.pt')
            seed_rows.append({'id': f'seed{i}', 'pattern_file': f'seed{i}.pt',
                              'pattern_sha256': pb.pattern_sha256(pattern),
                              'lineage_id': f'seed-family-{i}',
                              'source_group': ('history', 'specialist', 'random')[i // 4]})
            if shared_canonical and i < 2:
                seed_rows[-1]['canonical_group_id'] = 'shared-root'
    pb.atomic_json(prior_root / 'manifest.json', old_rows)
    pb.atomic_json(seeds / 'manifest.json', seed_rows)
    cfg['exploration'].update(training_protocol=ft.PROTOCOL, seed=11, batch_size=12,
                              candidate_pool_size=48, hidden_dims=[8], epochs=3, pretrain_epochs=2,
                              batch_train_size=4, holdout_fraction=.34,
                              arms={'history': 4, 'specialist': 4, 'random': 4},
                              later_arms={'best_min_margin': 4, 'uncertainty': 4, 'random': 4},
                              filter_prior={'manifest': str(prior_root / 'manifest.json'),
                                            'sample_root': str(prior_root),
                                            'expected_manifest_sha256': pb.file_sha256(prior_root / 'manifest.json')})
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    dataset = tmp_path / 'dataset'; dataset.mkdir()
    work = tmp_path / 'work'
    ex.prepare_run(config, dataset, work, [seeds])
    return work, dataset, prior_root


def _feedback(work, dataset, batch, *, edit_result=None):
    folder = work / 'batches' / f'batch-{batch:03d}_input'
    cfg = pb.load_profile_config(folder / 'config.yaml')
    store = dataset / f'batch-{batch:03d}'
    rows = pb.prepare_store(folder, store, cfg)
    result = {}
    for i, row in enumerate(rows):
        pattern = torch.load(folder / row['pattern_file'], weights_only=True)
        # Not HFSS: varied deterministic finite curves, all 49 frequencies retained.
        response = torch.stack((torch.linspace(-17, -6, 49) + i / 10,
                                torch.linspace(-27, -1, 49) + float(pattern.mean()),
                                torch.linspace(-13, -8, 49) - i / 10))
        entry = pb.observation(response, None, row, pattern, cfg, elapsed=.1)
        sample = store / entry['sample_file']
        torch.save((pattern, response), sample)
        entry['sample_sha256'] = pb.file_sha256(sample)
        result[row['id']] = entry
    if edit_result is not None:
        edit_result(result)
    pb.atomic_json(store / 'results.json', result)
    audit = ex.import_feedback(work, batch, store / 'results.json')
    return pb.read_json(audit)


def _equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _equal(x, y)
    else:
        assert a == b


def test_masked_mse_has_true_covered_denominator_and_zero_stopband_gradient():
    pred = torch.tensor([[1., 2., 3., 100., -100.]], requires_grad=True)
    loss = ft.masked_mse(pred, torch.zeros_like(pred), torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool))
    assert loss.item() == pytest.approx(14 / 3)
    loss.backward()
    torch.testing.assert_close(pred.grad, torch.tensor([[2/3, 4/3, 2., 0., 0.]]))


def test_split_unifies_cross_field_families_and_rejects_transitive_conflict():
    rows = [{'pattern_sha256': 'a', 'lineage_id': 'shared'},
            {'pattern_sha256': 'b', 'canonical_group_id': 'shared'}]
    train, hold, ledger = ft.plan_split(rows, .34, 11)
    assert train == [0, 1] or hold == [0, 1]
    _, _, expanded = ft.plan_split(rows + [{'pattern_sha256': 'c', 'lineage_id': 'new'}], .34, 11, ledger)
    assert all(expanded[k] == v for k, v in ledger.items())
    bridge = [{'pattern_sha256': 'a', 'lineage_id': 'middle'},
              {'pattern_sha256': 'b', 'canonical_group_id': 'middle'}]
    with pytest.raises(ValueError, match='retroactive'):
        ft.plan_split(bridge, .34, 11, {'pattern:a': 'train', 'pattern:b': 'holdout'})


def test_two_cumulative_dual_batches_keep_holdout_norm_and_prior_mask_evidence(tmp_path):
    work, dataset, prior_root = _prepare(tmp_path)
    _feedback(work, dataset, 1)
    ex.train_models(work, 1)
    summary1 = pb.read_json(work / 'models/batch-001/training_summary.json')
    ledger1 = pb.read_json(work / ft.SPLIT_FILE)['assignments']
    assert summary1['prior']['exclusions']
    validation = summary1['validation']
    absolute = np.abs(np.array(validation['prediction']) - np.array(validation['target']))
    np.testing.assert_allclose(list(validation['mae_by_band'][name] for name in validation['band_names']),
                               absolute.mean(axis=0), rtol=1e-6)
    assert validation['used_for_model_selection'] is False
    assert all(r['margin_mask'] == [True, True, True, False, False]
               for r in summary1['prior']['records_kept'])
    rows = pb.read_json(work / 'dataset_manifest.json')
    xs = [torch.load(work / row['scoped_sample_file'], weights_only=True)[0].reshape(-1)
          for row in rows if row['id'] in summary1['train_ids']]
    mean, std = ex._fit_norm(torch.stack(xs))
    for seed in (0, 1):
        saved = torch.load(work / f'models/batch-001/member-{seed}.pt', weights_only=False)
        assert len(saved['norms']) == 1
        torch.testing.assert_close(saved['norms'][0][0], mean, rtol=0, atol=0)
        torch.testing.assert_close(saved['norms'][0][1], std, rtol=0, atol=0)
    next_input = ex.select_batch(work, 2)
    next_rows = pb.read_json(next_input / 'manifest.json')
    assert len(next_rows) == 12
    assert all(r['prediction_valid'] and len(r['prediction']) == 5 and r['model_hash'] for r in next_rows)
    audit = _feedback(work, dataset, 2)
    assert audit['n_with_predictions'] == 12
    ex.train_models(work, 2)
    summary2 = pb.read_json(work / 'models/batch-002/training_summary.json')
    assert len(summary2['current_samples']) == 24
    ledger2 = pb.read_json(work / ft.SPLIT_FILE)['assignments']
    assert all(ledger2[k] == v for k, v in ledger1.items())
    assert set(summary1['holdout_ids']) <= set(summary2['holdout_ids'])
    assert set(summary1['train_ids']) <= set(summary2['train_ids'])
    third = ex.select_batch(work, 3)
    assert len(pb.read_json(third / 'manifest.json')) == 12
    # Both raw formats survive; placeholders have never become new-band truth.
    assert torch.load(prior_root / 'old0.pt', weights_only=True)[1].shape == (3, 17)
    assert torch.load(work / rows[0]['scoped_sample_file'], weights_only=True)[1].shape == (3, 49)


@pytest.mark.parametrize('boundary', [1, 2, 3])
def test_epoch_recovery_matches_uninterrupted_across_prior_to_current(tmp_path, boundary):
    work, dataset, _ = _prepare(tmp_path)
    _feedback(work, dataset, 1)
    uninterrupted = tmp_path / 'uninterrupted'
    resumed = tmp_path / 'resumed'
    shutil.copytree(work, uninterrupted); shutil.copytree(work, resumed)
    ex.train_models(uninterrupted, 1)
    with pytest.raises(ex.TrainingInterrupted):
        ex.train_models(resumed, 1, interrupt_after_epochs=boundary)
    ex.train_models(resumed, 1)
    for seed in (0, 1):
        a = torch.load(uninterrupted / f'models/batch-001/member-{seed}.pt', weights_only=False)
        b = torch.load(resumed / f'models/batch-001/member-{seed}.pt', weights_only=False)
        for key in ('model_state', 'optimizer_state', 'torch_rng_state', 'numpy_rng_state', 'norms', 'complete'):
            _equal(a[key], b[key])


@pytest.mark.parametrize('kind', ['prior_sample', 'prior_manifest', 'current_sample', 'config'])
def test_training_rejects_changed_bound_inputs_without_replacing_checkpoint(tmp_path, kind):
    work, dataset, prior_root = _prepare(tmp_path)
    _feedback(work, dataset, 1)
    ex.train_models(work, 1)
    checkpoint = work / 'models/batch-001/member-0.pt'
    digest = pb.file_sha256(checkpoint)
    ledger = (work / ft.SPLIT_FILE).read_bytes()
    if kind == 'prior_sample':
        (prior_root / 'old0.pt').write_bytes(b'tampered')
    elif kind == 'prior_manifest':
        with (prior_root / 'manifest.json').open('a') as stream:
            stream.write(' ')
    elif kind == 'current_sample':
        row = pb.read_json(work / 'dataset_manifest.json')[0]
        (work / row['scoped_sample_file']).write_bytes(b'tampered')
    else:
        cfg = yaml.safe_load((work / 'config.yaml').read_text(encoding='utf-8'))
        cfg['exploration']['epochs'] += 1
        (work / 'config.yaml').write_text(yaml.safe_dump(cfg), encoding='utf-8')
    with pytest.raises(ValueError, match='hash|SHA-256|configuration'):
        ex.train_models(work, 1)
    assert pb.file_sha256(checkpoint) == digest
    assert (work / ft.SPLIT_FILE).read_bytes() == ledger


def test_prior_preflight_rejects_manifest_drift_before_creating_workdir(tmp_path):
    work, dataset, prior_root = _prepare(tmp_path)
    fresh = tmp_path / 'must-not-exist'
    with (prior_root / 'manifest.json').open('a') as stream:
        stream.write(' ')
    with pytest.raises(ValueError, match='SHA-256'):
        ex.prepare_run(tmp_path / 'config.yaml', dataset, fresh, [])
    assert not fresh.exists()


def test_selection_rejects_changed_completed_model(tmp_path):
    work, dataset, _ = _prepare(tmp_path)
    _feedback(work, dataset, 1)
    ex.train_models(work, 1)
    path = work / 'models/batch-001/member-0.pt'
    saved = torch.load(path, weights_only=False)
    next(iter(saved['model_state'].values())).add_(1.0)
    torch.save(saved, path)
    with pytest.raises(ValueError, match='model hash/signature'):
        ex.select_batch(work, 2)
    assert not (work / 'batches/batch-002_input').exists()


def test_new_protocol_is_rejected_for_single_port_before_training(tmp_path):
    cfg = yaml.safe_load((Path(__file__).parents[1] / 'configs/single_r80_symmetry_explore.yaml').read_text(encoding='utf-8'))
    cfg['exploration']['training_protocol'] = ft.PROTOCOL
    with pytest.raises(ValueError, match='dual-only'):
        ex.validate_exploration_config(cfg)


@pytest.mark.parametrize('kind', ['delete', 'batch', 'lineage_id', 'canonical_group_id', 'pattern_sha256', 'append_unaudited'])
def test_imported_dataset_cannot_be_relabelled_or_shrunk_before_first_train(tmp_path, kind):
    work, dataset, _ = _prepare(tmp_path)
    _feedback(work, dataset, 1)
    path = work / 'dataset_manifest.json'
    rows = pb.read_json(path)
    if kind == 'delete':
        rows.pop()
    elif kind == 'batch':
        rows[0]['batch'] = 2
    elif kind == 'append_unaudited':
        rows.append({**rows[0], 'id': 'not-selected', 'batch': 0})
    else:
        rows[0][kind] = 'changed'
    pb.atomic_json(path, rows)
    with pytest.raises(ValueError, match='dataset binding'):
        ex.train_models(work, 1)
    with pytest.raises(ValueError, match='dataset binding'):
        ex.import_feedback(work, 1, dataset / 'batch-001/results.json')
    assert not (work / 'models/batch-001').exists()


def test_feedback_rejects_family_different_from_frozen_selected_input(tmp_path):
    work, dataset, _ = _prepare(tmp_path)
    def relabel(result):
        next(iter(result.values()))['lineage_id'] = 'different-family'
    with pytest.raises(ValueError):
        _feedback(work, dataset, 1, edit_result=relabel)
    assert pb.read_json(work / 'dataset_manifest.json') == []


def test_selected_input_hash_is_bound_at_prepare(tmp_path):
    work, dataset, _ = _prepare(tmp_path)
    path = work / 'batches/batch-001_input/manifest.json'
    rows = pb.read_json(path)
    rows[0]['lineage_id'] = 'changed-before-observation'
    pb.atomic_json(path, rows)
    with pytest.raises(ValueError, match='input manifest hash'):
        _feedback(work, dataset, 1)
    assert pb.read_json(work / 'dataset_manifest.json') == []


def test_next_training_rejects_prior_authenticated_split_ledger_drift(tmp_path):
    work, dataset, _ = _prepare(tmp_path)
    _feedback(work, dataset, 1)
    ex.train_models(work, 1)
    ex.select_batch(work, 2)
    _feedback(work, dataset, 2)
    path = work / ft.SPLIT_FILE
    ledger = pb.read_json(path)
    key = next(iter(ledger['assignments']))
    ledger['assignments'][key] = 'holdout' if ledger['assignments'][key] == 'train' else 'train'
    pb.atomic_json(path, ledger)
    with pytest.raises(ValueError, match='previously authenticated'):
        ex.train_models(work, 2)
    assert not (work / 'models/batch-002').exists()


@pytest.mark.parametrize('kind', ['older_batch', 'wrong_member'])
def test_selection_rejects_valid_checkpoint_from_wrong_batch_or_member(tmp_path, kind):
    work, dataset, _ = _prepare(tmp_path)
    _feedback(work, dataset, 1)
    ex.train_models(work, 1)
    if kind == 'wrong_member':
        shutil.copyfile(work / 'models/batch-001/member-0.pt', work / 'models/batch-001/member-1.pt')
        next_batch = 2
    else:
        ex.select_batch(work, 2)
        _feedback(work, dataset, 2)
        ex.train_models(work, 2)
        shutil.copytree(work / 'models/batch-001', work / 'models/batch-002', dirs_exist_ok=True)
        next_batch = 3
    with pytest.raises(ValueError, match='signature|binding'):
        ex.select_batch(work, next_batch)
    assert not (work / f'batches/batch-{next_batch:03d}_input').exists()


def test_seed_and_current_parent_variants_keep_canonical_family(tmp_path):
    work, dataset, _ = _prepare(tmp_path, shared_canonical=True)
    seeds = ex.load_seed_rows([work / 'seed_snapshot'])
    related = [row for row in seeds if row.get('canonical_group_id') == 'shared-root']
    assert len(related) == 2 and len({row['lineage_id'] for row in related}) == 2
    variants = [row for row in ex.generate_candidates(related, 12, seed=19, single=False)
                if row['parent_id'] is not None]
    assert len(variants) == 6 and all(row['canonical_group_id'] == 'shared-root' for row in variants)
    _, holdout, _ = ft.plan_split(variants, .34, 11, {'group:shared-root': 'holdout'})
    assert len(holdout) == len(variants)
    _feedback(work, dataset, 1)
    current = ex.load_scoped_parent_rows(work)
    assert sum(row.get('canonical_group_id') == 'shared-root' for row in current) == 2


def test_nonfinite_holdout_predictions_cannot_produce_completed_evidence(tmp_path, monkeypatch):
    work, dataset, _ = _prepare(tmp_path)
    _feedback(work, dataset, 1)
    forward = ex.CurveMLP.forward
    def nonfinite_eval(model, x):
        value = forward(model, x)
        return value if model.training else value * float('nan')
    monkeypatch.setattr(ex.CurveMLP, 'forward', nonfinite_eval)
    with pytest.raises(ValueError, match='non-finite filter holdout'):
        ex.train_models(work, 1)
    assert not (work / 'models/batch-001/training_summary.json').exists()
    assert pb.read_json(work / 'state.json')['batches']['1']['trained_model_dir'] is None


def test_relative_prior_paths_are_rejected_before_output(tmp_path):
    work, dataset, _ = _prepare(tmp_path)
    cfg = yaml.safe_load((work / 'config.yaml').read_text(encoding='utf-8'))
    cfg['exploration']['filter_prior']['manifest'] = 'prior/manifest.json'
    config = tmp_path / 'relative.yaml'
    config.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    fresh = tmp_path / 'must-not-exist'
    with pytest.raises(ValueError, match='absolute paths'):
        ex.prepare_run(config, dataset, fresh, [])
    assert not fresh.exists()
