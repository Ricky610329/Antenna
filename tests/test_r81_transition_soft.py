"""Focused generated checks for measured-only R81 transition preparation."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from script import exploration as ex
from script import filter_training as ft
from script import profiled_batch as pb
from script import r81_transition_soft as soft


REPO = Path(__file__).parents[1]


def _response(*, ripple=False) -> torch.Tensor:
    freqs = soft.EXPECTED_FREQS
    s21 = np.full(49, -25.0, dtype=np.float32)
    s21[(freqs >= 26.0) & (freqs <= 30.0)] = -2.0
    left = (freqs >= 20.0) & (freqs <= 26.0)
    right = (freqs >= 30.0) & (freqs <= 36.0)
    s21[left] = np.linspace(-25.0, -2.0, int(left.sum()), dtype=np.float32)
    s21[right] = np.linspace(-2.0, -25.0, int(right.sum()), dtype=np.float32)
    if ripple:
        interior = np.flatnonzero((freqs > 20.0) & (freqs < 26.0))
        s21[interior] += np.where(np.arange(len(interior)) % 2, 3.0, -3.0).astype(np.float32)
        interior = np.flatnonzero((freqs > 30.0) & (freqs < 36.0))
        s21[interior] += np.where(np.arange(len(interior)) % 2, -3.0, 3.0).astype(np.float32)
    return torch.stack((torch.full((49,), -12.0), torch.from_numpy(s21),
                        torch.full((49,), -12.0)))


def _prepare_feedback(tmp_path: Path) -> Path:
    cfg = yaml.safe_load((REPO / "configs/dual_r81_wide_filter.yaml").read_text(encoding="utf-8"))
    prior = tmp_path / "prior"; prior.mkdir()
    seeds = tmp_path / "seeds"; seeds.mkdir()
    prior_rows, seed_rows = [], []
    for index in range(6):
        pattern = torch.tensor(ex.enforce_dual_geometry(
            np.random.default_rng(index + 901).random((25, 25)) < 0.5), dtype=torch.float32)
        old_response = torch.stack((torch.full((17,), -12.0), torch.full((17,), -2.0),
                                    torch.full((17,), -12.0)))
        sample = prior / f"old-{index}.pt"
        torch.save((pattern, old_response), sample)
        prior_rows.append({
            "id": f"old-{index}", "sample_file": sample.name,
            "sample_sha256": pb.file_sha256(sample),
            "pattern_sha256": pb.pattern_sha256(pattern),
            "lineage_id": f"family-{index}", "canonical_group_id": f"family-{index}",
            "labels": list(soft.EXPECTED_LABELS),
            "freqs_ghz": np.arange(24.0, 32.1, 0.5).tolist(),
            "geometry": cfg["measurement"]["geometry"],
            "provenance": {"kind": "generated_old_profile_fixture"},
        })
        torch.save(pattern, seeds / f"seed-{index}.pt")
        seed_rows.append({
            "id": f"seed-{index}", "pattern_file": f"seed-{index}.pt",
            "pattern_sha256": pb.pattern_sha256(pattern), "lineage_id": f"family-{index}",
            "source_group": ("history", "specialist", "random")[index // 2],
        })
    pb.atomic_json(prior / "manifest.json", prior_rows)
    pb.atomic_json(seeds / "manifest.json", seed_rows)
    cfg["exploration"].update(
        training_protocol=ft.PROTOCOL, seed=81, batch_size=6, candidate_pool_size=12,
        hidden_dims=[8], epochs=1, pretrain_epochs=1, batch_train_size=2,
        holdout_fraction=0.34,
        arms={"history": 2, "specialist": 2, "random": 2},
        later_arms={"best_min_margin": 2, "uncertainty": 2, "random": 2},
        filter_prior={"manifest": str(prior / "manifest.json"), "sample_root": str(prior),
                      "expected_manifest_sha256": pb.file_sha256(prior / "manifest.json")},
    )
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    dataset = tmp_path / "dataset"; dataset.mkdir()
    work = tmp_path / "work"
    ex.prepare_run(config, dataset, work, [seeds])

    input_dir = work / "batches/batch-001_input"
    profile = pb.load_profile_config(input_dir / "config.yaml")
    store = dataset / "batch-001"
    rows = pb.prepare_store(input_dir, store, profile)
    results = {}
    for row in rows:
        pattern = torch.load(input_dir / row["pattern_file"], weights_only=True)
        response = _response()
        entry = pb.observation(response, None, row, pattern, profile, elapsed=0.01)
        sample = store / entry["sample_file"]
        torch.save((pattern, response), sample)
        entry["sample_sha256"] = pb.file_sha256(sample)
        results[row["id"]] = entry
    pb.atomic_json(store / "results.json", results)
    ex.import_feedback(work, 1, store / "results.json")
    return work


def _rank_record(name: str, pattern: str, wm: float, *, good: bool) -> dict:
    cfg = ex._load_yaml(soft.CANONICAL_CONFIG)
    side = {
        "mean_relative_attenuation_db": 10.0 if good else 0.0,
        "outward_rise_rms_db": 0.0 if good else 10.0,
        "curvature_rms_db": 0.0 if good else 10.0,
    }
    return {
        "truth_kind": soft.TRUTH_KIND, "id": name, "pattern_sha256": pattern,
        "scoped_sample_sha256": "c" * 64, "raw_response_f32le_sha256": "d" * 64,
        "measurement_id": soft.measurement_id(cfg["measurement"]),
        "score_spec_id": soft.score_spec_id(cfg["score_spec"]),
        "labels": list(soft.EXPECTED_LABELS), "frequencies_ghz": soft.EXPECTED_FREQS.tolist(),
        "hard_band_truth": {"worst_margin_db": wm},
        "transition": {"left": dict(side), "right": dict(side)},
    }


def test_smooth_and_ripple_are_softly_distinguished_at_identical_wm():
    cfg = ex._load_yaml(soft.CANONICAL_CONFIG)
    smooth = soft.diagnose_response(_response(), soft.EXPECTED_LABELS,
                                    soft.EXPECTED_FREQS, cfg["score_spec"])
    ripple = soft.diagnose_response(_response(ripple=True), soft.EXPECTED_LABELS,
                                    soft.EXPECTED_FREQS, cfg["score_spec"])
    assert smooth["hard_band_truth"] == ripple["hard_band_truth"]
    assert smooth["hard_band_truth"]["worst_margin_db"] == pytest.approx(1.0)
    for side in ("left", "right"):
        assert smooth["transition"][side]["curvature_rms_db"] < 1e-5
        assert ripple["transition"][side]["curvature_rms_db"] > 1.0
        assert ripple["transition"][side]["outward_rise_rms_db"] > 0.0
        assert len(smooth["transition"][side]["s21_db_pass_to_stop"]) == 13
        assert len(smooth["transition"][side]["interior_relative_attenuation_db"]) == 11


def test_diagnostics_reject_old_axis_nonfinite_and_changed_score_spec():
    cfg = ex._load_yaml(soft.CANONICAL_CONFIG)
    with pytest.raises(ValueError, match="16-40"):
        soft.diagnose_response(torch.zeros((3, 17)), soft.EXPECTED_LABELS,
                               np.arange(24.0, 32.1, 0.5), cfg["score_spec"])
    bad = _response(); bad[1, 10] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        soft.diagnose_response(bad, soft.EXPECTED_LABELS, soft.EXPECTED_FREQS,
                               cfg["score_spec"])
    changed = copy.deepcopy(cfg["score_spec"]); changed["bands"][0]["threshold"] = -9
    with pytest.raises(ValueError, match="exact current R81"):
        soft.diagnose_response(_response(), soft.EXPECTED_LABELS,
                               soft.EXPECTED_FREQS, changed)


def test_midrank_all_ties_and_input_permutation_are_exact():
    records = [_rank_record("c", "3" * 64, 0.5, good=True),
               _rank_record("a", "1" * 64, 0.5, good=True),
               _rank_record("b", "2" * 64, 0.5, good=True)]
    forward = soft.rank_measured_records(records)
    reverse = soft.rank_measured_records(list(reversed(records)))
    assert forward == reverse
    assert [row["id"] for row in forward] == ["a", "b", "c"]
    assert all(row["transition_quality_0_to_1"] == pytest.approx(0.5) for row in forward)
    assert all(set(row["transition_component_midranks"].values()) == {0.5} for row in forward)


def test_bounded_quality_cannot_overturn_more_than_point_one_db_wm_deficit():
    high_wm_bad_transition = _rank_record("high", "a" * 64, 1.0, good=False)
    lower_wm_good_transition = _rank_record("low", "b" * 64, 0.899, good=True)
    ranked = soft.rank_measured_records([lower_wm_good_transition, high_wm_bad_transition])
    assert [row["id"] for row in ranked] == ["high", "low"]
    by_id = {row["id"]: row for row in ranked}
    assert by_id["high"]["transition_quality_0_to_1"] == 0.0
    assert by_id["low"]["transition_quality_0_to_1"] == 1.0
    assert by_id["high"]["bounded_parent_score_db"] > by_id["low"]["bounded_parent_score_db"]
    assert all(row["cap_is_noise_estimate"] is False and row["transition_enters_wm"] is False
               for row in ranked)
    with pytest.raises(ValueError, match="fixes.*0.1"):
        soft.rank_measured_records([high_wm_bad_transition], max_wm_tradeoff_db=0.2)


def test_authenticated_feedback_sidecar_binds_lifecycle_and_is_deterministic(tmp_path):
    work = _prepare_feedback(tmp_path)
    first = soft.build_sidecar(work, 1, tmp_path / "first.json")
    second = soft.build_sidecar(work, 1, tmp_path / "second.json")
    assert pb.file_sha256(first) == pb.file_sha256(second)
    value = pb.read_json(first)
    assert value["kind"] == soft.SIDECAR_KIND
    assert value["immutable_input_sha256"]["before"] == value["immutable_input_sha256"]["after"]
    assert value["sample_sha256"]["before"] == value["sample_sha256"]["after"]
    assert len(value["records"]) == 6
    assert all(row["truth_kind"] == soft.TRUTH_KIND and row["hard_band_truth"]["transition_enters_wm"] is False
               for row in value["records"])
    ranking = soft.rank_sidecar(first, tmp_path / "ranking.json")
    ranked = pb.read_json(ranking)
    assert len(ranked["ranking"]) == 6
    assert ranked["select_batch_or_candidate_ranking_integrated"] is False
    assert ranked["source_feedback_revalidated"] is True
    assert set(ranked["source_sidecar_sha256"].values()) == {pb.file_sha256(first)}
    assert all(row["input_authentication_performed"] is False for row in ranked["ranking"])


def test_tampered_scoped_sample_is_rejected_without_destination(tmp_path):
    work = _prepare_feedback(tmp_path)
    row = pb.read_json(work / "dataset_manifest.json")[0]
    sample = work / row["scoped_sample_file"]
    pattern, response = torch.load(sample, weights_only=True)
    torch.save((pattern, response + 0.25), sample)
    output = tmp_path / "must-not-exist.json"
    with pytest.raises(ValueError, match="hash changed"):
        soft.build_sidecar(work, 1, output)
    assert not output.exists()


def test_stale_feedback_binding_and_duplicate_measured_patterns_are_rejected(tmp_path):
    work = _prepare_feedback(tmp_path)
    audit = work / pb.read_json(work / "state.json")["batches"]["1"]["feedback_audit"]
    value = pb.read_json(audit); value["n_imported"] += 1
    pb.atomic_json(audit, value)
    output = tmp_path / "stale-must-not-exist.json"
    with pytest.raises(ValueError, match="hash binding"):
        soft.build_sidecar(work, 1, output)
    assert not output.exists()
    duplicate = [_rank_record("a", "f" * 64, 1.0, good=True),
                 _rank_record("b", "f" * 64, 1.0, good=False)]
    with pytest.raises(ValueError, match="unique"):
        soft.rank_measured_records(duplicate)
    pretending = _rank_record("fresh", "e" * 64, 1.0, good=True)
    pretending.pop("scoped_sample_sha256")
    with pytest.raises(ValueError, match="structurally marked measured"):
        soft.rank_measured_records([pretending])


def test_create_only_and_sidecar_content_binding_reject_substitution(tmp_path):
    work = _prepare_feedback(tmp_path)
    sidecar = soft.build_sidecar(work, 1, tmp_path / "sidecar.json")
    original = sidecar.read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        soft.build_sidecar(work, 1, sidecar)
    assert sidecar.read_bytes() == original
    value = json.loads(original)
    value["records"][0]["hard_band_truth"]["worst_margin_db"] += 1.0
    value["records"][0]["transition"]["left"]["curvature_rms_db"] += 2.0
    value.pop("content_id")
    value["content_id"] = ex.content_id(value)
    substituted = tmp_path / "substituted.json"
    substituted.write_text(json.dumps(value), encoding="utf-8")
    output = tmp_path / "ranking-must-not-exist.json"
    with pytest.raises(ValueError, match="feedback replay"):
        soft.rank_sidecar(substituted, output)
    assert not output.exists()


def test_rank_rejects_stale_measured_source_after_sidecar_creation(tmp_path):
    work = _prepare_feedback(tmp_path)
    sidecar = soft.build_sidecar(work, 1, tmp_path / "sidecar.json")
    row = pb.read_json(work / "dataset_manifest.json")[0]
    sample = work / row["scoped_sample_file"]
    pattern, response = torch.load(sample, weights_only=True)
    torch.save((pattern, response - 0.5), sample)
    output = tmp_path / "stale-ranking-must-not-exist.json"
    with pytest.raises(ValueError, match="sample hash changed"):
        soft.rank_sidecar(sidecar, output)
    assert not output.exists()
