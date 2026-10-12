"""Focused generated-fixture tests for the additive S21-only trainer."""
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from antenna.measurement import measurement_id, score_spec_id
from script import filter_s21_training as s21
from script import profiled_batch as pb


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "dual_r81_s21_discrete.yaml"


def _pattern(number: int, *, offset: int = 0) -> torch.Tensor:
    pattern = np.zeros((25, 25), dtype=np.float32)
    pattern[:5, 10:15] = 1
    pattern[20:, 10:15] = 1
    value = number + offset + 1
    for bit in range(12):
        if value & (1 << bit):
            row, column = divmod(bit * 17 + offset * 3, 25)
            pattern[row % 20, column] = 1
    return torch.from_numpy(pattern)


def _wide_response(number: int = 0) -> torch.Tensor:
    freqs = np.arange(16, 40.1, .5)
    s21_values = np.full(49, -10.0, dtype=np.float32)
    s21_values[(freqs >= 16) & (freqs <= 20)] = -21.0 - number / 100
    s21_values[(freqs >= 26) & (freqs <= 30)] = -2.5 + number / 100
    s21_values[(freqs >= 36) & (freqs <= 40)] = -22.0 - number / 100
    return torch.stack((torch.full((49,), -12.0), torch.from_numpy(s21_values),
                        torch.full((49,), -13.0)))


def _input_and_store(tmp_path: Path, count: int = 20):
    input_dir = tmp_path / "canonical"
    input_dir.mkdir()
    shutil.copy2(CONFIG, input_dir / "config.yaml")
    cfg = pb.load_profile_config(input_dir / "config.yaml")
    pb.atomic_json(input_dir / "measurement.json", cfg.measurement)
    pb.atomic_json(input_dir / "score_spec.json", cfg.score_spec)
    mid, sid = measurement_id(cfg.measurement), score_spec_id(cfg.score_spec)
    rows = []
    for index in range(count):
        pattern = _pattern(index)
        name = f"s21_{index:03d}"
        torch.save(pattern, input_dir / f"{name}.pt")
        rows.append({"id": name, "pattern_file": f"{name}.pt",
                     "pattern_sha256": pb.pattern_sha256(pattern), "port": "dual",
                     "measurement_id": mid, "score_spec_id": sid,
                     "geometry_profile": "dual25_p01_db075",
                     "lineage_id": f"line-{index}", "canonical_group_id": f"family-{index}"})
    pb.atomic_json(input_dir / "manifest.json", rows)
    pb.validate_input(input_dir, cfg)

    store = tmp_path / "store"
    pb.prepare_store(input_dir, store, cfg)
    results = {}
    for index, row in enumerate(rows):
        pattern = torch.load(input_dir / row["pattern_file"], weights_only=True)
        response = _wide_response(index)
        entry = pb.observation(response, None, row, pattern, cfg, elapsed=.1)
        sample = store / entry["sample_file"]
        torch.save((pattern, response), sample)
        entry["sample_sha256"] = pb.file_sha256(sample)
        results[row["id"]] = entry
    pb.atomic_json(store / "results.json", results)
    pb.validate_store(store, require_complete=True)
    return input_dir, store


def _prior(tmp_path: Path, count: int = 12):
    root = tmp_path / "prior"
    root.mkdir()
    rows = []
    freqs = np.arange(24, 32.1, .5).tolist()
    for index in range(count):
        pattern = _pattern(index, offset=100)
        response = torch.stack((torch.full((17,), -11.0),
                                torch.linspace(-3.0, -1.0, 17) + index / 100,
                                torch.full((17,), -12.0)))
        sample = root / f"prior-{index}.pt"
        torch.save((pattern, response), sample)
        rows.append({"id": f"prior-{index}", "sample_file": sample.name,
                     "sample_sha256": pb.file_sha256(sample),
                     "pattern_sha256": pb.pattern_sha256(pattern),
                     "lineage_id": f"prior-line-{index}",
                     "canonical_group_id": f"prior-family-{index}",
                     "labels": ["S11", "S21", "S22"], "freqs_ghz": freqs,
                     "geometry": {"geom": "p01", "pixel_count": 25,
                                  "diag_bridge_w": .075},
                     "provenance": {"kind": "generated_fixture"}})
    manifest = root / "manifest.json"
    pb.atomic_json(manifest, rows)
    return manifest, root, pb.file_sha256(manifest)


def test_authoritative_three_margins_and_reference_only_channels():
    protocol = s21.load_protocol()
    result = s21.score_s21(_wide_response(), protocol,
                           labels=["S11", "S21", "S22"],
                           freqs_ghz=np.arange(16, 40.1, .5))
    np.testing.assert_allclose(result, [.5, 1.0, 2.0], rtol=0, atol=1e-6)
    reference = s21._reference_diagnostics(_wide_response().numpy(), protocol)
    assert set(reference) == {"s11_match_margin_db", "s22_match_margin_db",
                              "low_buffer_max_s21_db", "high_buffer_max_s21_db"}
    assert tuple(protocol["band_names"]) == ("s21_pass", "s21_low_stop", "s21_high_stop")


def test_historical_prior_masks_both_unobserved_stop_outputs(tmp_path):
    protocol = s21.load_protocol()
    manifest, root, digest = _prior(tmp_path)
    x, y, mask, receipt = s21.load_historical_prior(
        protocol, manifest, root, digest, holdout_rows=[])
    assert x.shape == (12, 625) and y.shape == mask.shape == (12, 3)
    assert torch.equal(mask, torch.tensor([[True, False, False]]).expand_as(mask))
    assert receipt["target_mask"] == [True, False, False]
    assert receipt["historical_full_three_band_worst_margin_available"] is False
    # Mask placeholders are never accepted as a historical min-of-three target.
    assert not torch.equal(y.min(dim=1).values, y[:, 0])


def test_partial_snapshot_training_is_bound_and_rank_requires_wide_model(tmp_path):
    input_dir, store = _input_and_store(tmp_path)
    manifest, prior_root, prior_sha = _prior(tmp_path)
    output = tmp_path / "artifacts"
    paths, summary = s21.train(
        input_store_pairs=[(input_dir, store)], split_input=input_dir,
        output_dir=output, prior_manifest=manifest, prior_root=prior_root,
        expected_prior_sha256=prior_sha, min_successes=20,
        pretrain_epochs=1, epochs=1, batch_size=8, hidden_dims=[8],
        holdout_fraction=.3, seed=17)
    assert len(paths) == 2 and summary["rank_ready"] is True
    assert summary["trained_output_mask"] == [True, True, True]
    assert summary["validation"]["used_for_model_selection"] is False
    prior_summary = next((output / "prior").glob("*/training_summary.json"))
    assert pb.read_json(prior_summary)["rank_ready"] is False

    candidates = []
    for index in range(3):
        pattern = _pattern(index, offset=200).numpy()
        candidates.append({"id": f"candidate-{index}", "pattern": pattern,
                           "pattern_sha256": pb.pattern_sha256(pattern)})
    model_dir = paths[0].parent
    completion_sha = pb.file_sha256(model_dir / "completion_receipt.json")
    prediction, order, receipt = s21.predict_and_rank(
        model_dir, candidates, expected_completion_sha256=completion_sha)
    np.testing.assert_array_equal(order, np.argsort(-prediction.min(axis=1), kind="stable"))
    assert receipt["ordered_ids"] == [candidates[index]["id"] for index in order]
    assert receipt["model_signature"] == summary["signature"]

    binding_path = next((output / "snapshots").glob("*/source_bindings.json"))
    binding = pb.read_json(binding_path)
    binding["items"][next(iter(binding["items"]))]["source_sample_sha256"] = "0" * 64
    pb.atomic_json(binding_path, binding)
    with pytest.raises(ValueError, match="provenance"):
        s21.freeze_successes([(input_dir, store)], output)


def test_rejects_under_twenty_and_tampered_plan(tmp_path):
    input_dir, store = _input_and_store(tmp_path)
    manifest, prior_root, prior_sha = _prior(tmp_path)
    with pytest.raises(ValueError, match="min_successes >= 20"):
        s21.train(input_store_pairs=[(input_dir, store)], split_input=input_dir,
                  output_dir=tmp_path / "bad", prior_manifest=manifest,
                  prior_root=prior_root, expected_prior_sha256=prior_sha,
                  min_successes=19, pretrain_epochs=1, epochs=1,
                  batch_size=8, hidden_dims=[8], holdout_fraction=.3, seed=17)

    paths, _ = s21.train(input_store_pairs=[(input_dir, store)], split_input=input_dir,
                         output_dir=tmp_path / "good", prior_manifest=manifest,
                         prior_root=prior_root, expected_prior_sha256=prior_sha,
                         min_successes=20, pretrain_epochs=1, epochs=1,
                         batch_size=8, hidden_dims=[8], holdout_fraction=.3, seed=17)
    plan_path = paths[0].parent / "training_plan.json"
    plan = pb.read_json(plan_path); plan["recipe"]["epochs"] = 2
    pb.atomic_json(plan_path, plan)
    pattern = _pattern(0, offset=300).numpy()
    with pytest.raises(ValueError, match="plan/summary/signature"):
        s21.predict_and_rank(paths[0].parent, [{"id": "candidate", "pattern": pattern,
                                               "pattern_sha256": pb.pattern_sha256(pattern)}],
                             expected_completion_sha256=pb.file_sha256(
                                 paths[0].parent / "completion_receipt.json"))
