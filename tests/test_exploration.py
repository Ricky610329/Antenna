# -*- coding: utf-8 -*-
"""CPU-only tests for the bounded exploration state machine."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from script.exploration import (TrainingInterrupted, enforce_dual_geometry,
                                enforce_single_geometry,
                                generate_candidates,
                                file_sha256, grouped_holdout_indices, hamming, import_feedback,
                                load_scoped_parent_rows,
                                maxmin_hamming_indices, radiation_vector,
                                pattern_sha256, prepare_run, select_batch,
                                select_candidate_arms, train_models)


def test_single_geometry_is_exact_half12_mirror_and_feed():
    rng = np.random.default_rng(9)
    p = enforce_single_geometry(rng.random((25, 25)))
    assert np.array_equal(p[:, :12], p[:, 13:][:, ::-1])
    assert p[24, 12]


def test_dual_geometry_always_preserves_both_five_by_five_port_pads():
    p = enforce_dual_geometry(np.zeros((25, 25), dtype=bool))
    assert p[:5, 10:15].all()
    assert p[20:25, 10:15].all()


def test_grouped_holdout_keeps_lineage_profile_together():
    rows = [
        {"lineage_id": "a", "geometry_profile": "p"},
        {"lineage_id": "a", "geometry_profile": "p"},
        {"lineage_id": "b", "geometry_profile": "p"},
        {"lineage_id": "a", "geometry_profile": "q"},
    ]
    tr, hold = grouped_holdout_indices(rows, 0.34, 7)
    assert set(tr).isdisjoint(hold)
    assert ({0, 1} <= set(tr)) or ({0, 1} <= set(hold))
    assert set(tr) | set(hold) == set(range(4))


def test_maxmin_hamming_prefers_geometry_distance():
    p0 = np.zeros((2, 4), bool)
    p1 = p0.copy(); p1[0, 0] = 1
    p2 = np.ones((2, 4), bool)
    got = maxmin_hamming_indices([p1, p2], 1, references=[p0])
    assert got == [1]
    assert hamming(p0, p2) == 8


def test_invalid_predictions_fall_back_to_geometry_and_are_recorded():
    candidates = [{"pattern": np.unpackbits(np.array([i], np.uint8)).reshape(2, 4),
                   "candidate_index": i} for i in range(8)]
    pred = np.zeros((8, 3), np.float32); pred[2, 0] = np.nan
    rows = select_candidate_arms(candidates, {"uncertainty": 2, "predcurve_diversity": 2},
                                 seed=2, mode="single_fullcurve", predictions=pred,
                                 uncertainty=np.full(8, np.nan))
    assert len(rows) == 4
    assert all(r["fallback"] == "invalid_or_missing_prediction_to_geometry" for r in rows)
    assert len({r["candidate_index"] for r in rows}) == 4


def test_candidate_pool_is_half_fresh_and_random_arm_uses_only_fresh():
    parent = enforce_single_geometry(np.random.default_rng(1).random((25, 25)))
    candidates = generate_candidates(
        [{"id": "parent", "pattern": parent, "lineage_id": "family-a",
          "source": "current_scoped_observation"}], 20, seed=12, single=True)
    groups = [row["candidate_group"] for row in candidates]
    assert groups.count("fresh_random") == 10
    assert groups.count("parent_variant") == 10
    assert all(np.array_equal(row["pattern"][:, :12], row["pattern"][:, 13:][:, ::-1])
               for row in candidates)
    assert {row["lineage_id"] for row in candidates if row["candidate_group"] == "parent_variant"} == {
        "family-a"}
    selected = select_candidate_arms(candidates, {"geometry": 4, "random": 4}, seed=9,
                                     mode="single_fullcurve", predictions=None, uncertainty=None)
    assert all(row["candidate_group"] == "fresh_random"
               for row in selected if row["selection_arm"] == "random")


def test_single_selection_does_not_use_predicted_goodness():
    rng = np.random.default_rng(3)
    candidates = [{"pattern": rng.random((5, 5)) > .5, "candidate_index": i}
                  for i in range(12)]
    # Changing absolute curve levels must not affect random/Hamming/curve-diversity
    # selections.  Curve diversity sees only pairwise differences.
    pred = rng.normal(size=(12, 4)).astype(np.float32)
    unc = np.linspace(0, 1, 12)
    arms = {"random": 2, "geometry": 2, "predcurve_diversity": 2}
    a = select_candidate_arms(candidates, arms, seed=5, mode="single_fullcurve",
                              predictions=pred, uncertainty=unc)
    b = select_candidate_arms(candidates, arms, seed=5, mode="single_fullcurve",
                              predictions=pred + 1000.0, uncertainty=unc)
    assert [r["candidate_index"] for r in a] == [r["candidate_index"] for r in b]


def test_radiation_vector_is_two_91_point_cuts_and_rejects_nan():
    theta = torch.linspace(-90, 90, 181)
    rad = {"theta": theta, "phi0": theta / 90, "phi90": -theta / 90}
    out = radiation_vector(rad)
    assert out.shape == (182,)
    assert out[0] == pytest.approx(-1) and out[90] == pytest.approx(1)
    rad["phi0"][4] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        radiation_vector(rad)


def test_prepare_uses_measurement_api_and_writes_self_contained_bundle(tmp_path):
    measurement = {"name": "single17", "port": "single", "labels": ["S11", "Gain"],
                   "geometry": {"geom": "single_v1", "pixel_count": 25, "diag_bridge_w": .1},
                   "sweep": {"start_ghz": 24, "stop_ghz": 32, "step_ghz": .5,
                             "type": "Interpolating"},
                   "solver": {"setup_freq_ghz": 28, "open_region_freq_ghz": 28,
                              "max_delta_s": .02, "max_passes": 10, "min_passes": 2,
                              "min_converged": 2}}
    cfg = {"measurement": measurement, "score_spec": {"name": "record_only", "bands": []},
           "exploration": {"mode": "single_fullcurve", "batch_size": 4, "max_batches": 3,
                           "seed": 2, "candidate_pool_size": 12, "ensemble_seeds": [0, 1],
                           "hidden_dims": [4], "epochs": 1, "pretrain_epochs": 0,
                           "batch_train_size": 2, "geometry_profile": "single_v1",
                           "arms": {"random": 1, "geometry": 1, "uncertainty": 1,
                                    "predcurve_diversity": 1},
                           "later_arms": {"random": 1, "geometry": 1, "uncertainty": 1,
                                          "predcurve_diversity": 1}}}
    config = tmp_path / "config.yaml"; config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    dataset = tmp_path / "dataset"; dataset.mkdir()
    out = prepare_run(config, dataset, tmp_path / "work", [])
    assert (out / "measurement.json").is_file()
    assert (out / "score_spec.json").is_file()
    assert (out / "config.yaml").is_file()
    assert not (out / "hfss_setup.json").exists()
    rows = json.loads((out / "manifest.json").read_text())
    assert {r["selection_arm"] for r in rows} == {
        "random", "geometry", "uncertainty", "predcurve_diversity"}
    assert all(r["measurement_id"] and r["score_spec_id"] for r in rows)
    assert all(r["port"] == "single" for r in rows)
    assert all(torch.load(out / r["pattern_file"], weights_only=True)[24, 12] for r in rows)


def _fake_work(tmp_path: Path, n: int = 6) -> Path:
    work = tmp_path / "work"; work.mkdir()
    dataset = tmp_path / "dataset"; dataset.mkdir()
    measurement = {"name": "fake", "port": "single", "labels": ["S11", "Gain"],
                   "geometry": {"geom": "single_v1", "pixel_count": 25, "diag_bridge_w": .1},
                   "sweep": {"start_ghz": 24, "stop_ghz": 32, "step_ghz": .5,
                             "type": "Interpolating"},
                   "solver": {"setup_freq_ghz": 28, "open_region_freq_ghz": 28,
                              "max_delta_s": .02, "max_passes": 10, "min_passes": 2,
                              "min_converged": 2}}
    cfg = {"measurement": measurement, "score_spec": {"name": "unused", "bands": []},
           "exploration": {"mode": "single_fullcurve", "batch_size": n,
                           "max_batches": 3, "seed": 4, "candidate_pool_size": 24,
                           "ensemble_seeds": [0, 1], "hidden_dims": [8], "epochs": 2,
                           "pretrain_epochs": 0, "batch_train_size": 3,
                           "learning_rate": .001, "holdout_fraction": .34,
                           "geometry_profile": "single_v1",
                           "arms": {"random": 2, "geometry": 2, "uncertainty": 1,
                                    "predcurve_diversity": 1},
                           "later_arms": {"random": 2, "geometry": 2, "uncertainty": 1,
                                          "predcurve_diversity": 1}}}
    (work / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    (work / "measurement.json").write_text(json.dumps(measurement), encoding="utf-8")
    (work / "score_spec.json").write_text(json.dumps(cfg["score_spec"]), encoding="utf-8")
    batch_dir = work / "batches" / "batch-001_input"; batch_dir.mkdir(parents=True)
    from antenna.measurement import measurement_id, score_spec_id
    mid, sid = measurement_id(measurement), score_spec_id(cfg["score_spec"])
    (dataset / "measurement.json").write_text(json.dumps(measurement), encoding="utf-8")
    (dataset / "score_spec.json").write_text(json.dumps(cfg["score_spec"]), encoding="utf-8")
    batch_rows, result_rows = [], {}
    (dataset / "rad").mkdir()
    for i in range(n):
        p = torch.as_tensor(enforce_single_geometry(np.random.default_rng(i).random((25, 25))),
                            dtype=torch.float32)
        response = torch.stack((torch.linspace(-15, -5, 17) + i,
                                torch.linspace(0, 5, 17) - i))
        pid = f"p{i}"
        torch.save(p, batch_dir / f"{pid}.pt")
        sha = pattern_sha256(p)
        batch_rows.append({"id": pid, "pattern_file": f"{pid}.pt", "pattern_sha256": sha,
                           "lineage_id": f"line-{i // 2}", "prediction": None})
        sample = dataset / f"sample-{i}.pt"; torch.save((p, response), sample)
        rad = {"theta": torch.linspace(-90, 90, 181),
               "phi0": torch.linspace(-2, 2, 181) + i,
               "phi90": torch.linspace(2, -2, 181) - i}
        torch.save(rad, dataset / "rad" / f"{pid}.pt")
        rad_path = dataset / "rad" / f"{pid}.pt"
        result_rows[pid] = {"status": "ok", "sample_file": sample.name,
                            "sample_sha256": file_sha256(sample), "rad_file": f"rad/{pid}.pt",
                            "rad_sha256": file_sha256(rad_path), "lineage_id": f"line-{i // 2}",
                            "pattern_sha256": sha, "measurement_id": mid, "score_spec_id": sid,
                            "labels": ["S11", "Gain"], "freqs": np.linspace(24, 32, 17).tolist(),
                            "geom": "single_v1", "dbw": .1}
    (batch_dir / "manifest.json").write_text(json.dumps(batch_rows), encoding="utf-8")
    (dataset / "results.json").write_text(json.dumps(result_rows), encoding="utf-8")
    state = {"schema_version": 1, "config_hash": "test", "dataset_root": str(dataset),
             "measurement_id": mid, "score_spec_id": sid, "id_prefix": "fake_loop",
             "frequencies_ghz": np.linspace(24, 32, 17).tolist(), "seed_inputs": [],
             "max_batches": 3, "measured_budget": 0, "next_batch": 2,
             "batches": {"1": {"input_dir": "batches/batch-001_input", "selected": n,
                                  "model_hash": None, "feedback_audit": None,
                                  "trained_model_dir": None}}}
    (work / "state.json").write_text(json.dumps(state), encoding="utf-8")
    (work / "dataset_manifest.json").write_text("[]", encoding="utf-8")
    return work


def test_feedback_then_training_interruption_resumes_bitwise(tmp_path):
    work = _fake_work(tmp_path)
    audit = import_feedback(work, 1, tmp_path / "dataset" / "results.json")
    assert json.loads(audit.read_text())["order"] == "saved_predictions_before_training_new_batch"
    assert len(json.loads((work / "dataset_manifest.json").read_text())) == 6
    with pytest.raises(TrainingInterrupted):
        train_models(work, 1, interrupt_after_epochs=1)
    resumed = train_models(work, 1)

    # A fresh run from the same imported scoped data must end byte-equivalent at
    # the tensor/state level (zip serialization bytes themselves need not match).
    fresh = tmp_path / "fresh"
    import shutil
    shutil.copytree(work, fresh, ignore=shutil.ignore_patterns("models"))
    st = json.loads((fresh / "state.json").read_text())
    st["batches"]["1"]["trained_model_dir"] = None
    (fresh / "state.json").write_text(json.dumps(st), encoding="utf-8")
    clean = train_models(fresh, 1)
    for a, b in zip(resumed, clean):
        sa = torch.load(a, weights_only=False)["model_state"]
        sb = torch.load(b, weights_only=False)["model_state"]
        assert sa.keys() == sb.keys()
        assert all(torch.equal(sa[k], sb[k]) for k in sa)


def test_feedback_recovers_after_dataset_write_before_state_write(tmp_path, monkeypatch):
    import script.exploration as exploration

    work = _fake_work(tmp_path)
    results = tmp_path / "dataset" / "results.json"
    real_atomic_json = exploration._atomic_json
    inject = {"once": True}

    def fail_once_before_state(path, value):
        if Path(path).name == "state.json" and inject["once"]:
            inject["once"] = False
            raise RuntimeError("injected crash before feedback state commit")
        return real_atomic_json(path, value)

    monkeypatch.setattr(exploration, "_atomic_json", fail_once_before_state)
    with pytest.raises(RuntimeError, match="injected crash"):
        exploration.import_feedback(work, 1, results)
    assert len(json.loads((work / "dataset_manifest.json").read_text())) == 6
    stale = json.loads((work / "state.json").read_text())
    assert stale["measured_budget"] == 0
    assert stale["batches"]["1"]["feedback_audit"] is None

    audit_path = exploration.import_feedback(work, 1, results)
    entries = json.loads((work / "dataset_manifest.json").read_text())
    state = json.loads((work / "state.json").read_text())
    audit = json.loads(audit_path.read_text())
    assert len(entries) == 6
    assert len({(row["batch"], row["id"]) for row in entries}) == 6
    assert state["measured_budget"] == 6
    assert audit["n_idempotent_replayed"] == 6


def test_two_batch_fake_loop_preserves_pretrain_prediction_audit(tmp_path):
    work = _fake_work(tmp_path)
    dataset = tmp_path / "dataset"
    import_feedback(work, 1, dataset / "results.json")
    train_models(work, 1)
    batch2 = select_batch(work, 2)
    current_parents = load_scoped_parent_rows(work)
    assert len(current_parents) == 6
    assert {row["source"] for row in current_parents} == {"current_scoped_observation"}
    selected = json.loads((batch2 / "manifest.json").read_text())
    assert len(selected) == 6
    assert all(row["model_hash"] for row in selected)
    assert all(row["prediction"] is not None for row in selected)
    assert all(row["candidate_group"] == "fresh_random"
               for row in selected if row["selection_arm"] == "random")

    result_rows = json.loads((dataset / "results.json").read_text())
    for i, row in enumerate(selected):
        p = torch.load(batch2 / row["pattern_file"], weights_only=True)
        response = torch.stack((torch.linspace(-25, 3, 17) + i,
                                torch.linspace(-4, 9, 17) - i))
        sample = dataset / f"batch2-{i}.pt"; torch.save((p, response), sample)
        torch.save({"theta": torch.linspace(-90, 90, 181),
                    "phi0": torch.linspace(-10, 10, 181),
                    "phi90": torch.linspace(10, -10, 181)},
                   dataset / "rad" / f"{row['id']}.pt")
        rad_path = dataset / "rad" / f"{row['id']}.pt"
        result_rows[row["id"]] = {"status": "ok", "sample_file": sample.name,
                                  "sample_sha256": file_sha256(sample),
                                  "rad_file": f"rad/{row['id']}.pt",
                                  "rad_sha256": file_sha256(rad_path),
                                  "lineage_id": row["lineage_id"],
                                  "pattern_sha256": row["pattern_sha256"],
                                  "measurement_id": row["measurement_id"],
                                  "score_spec_id": row["score_spec_id"],
                                  "labels": ["S11", "Gain"],
                                  "freqs": np.linspace(24, 32, 17).tolist(),
                                  "geom": "single_v1", "dbw": .1}
    second_manifest = dataset / "results-after-batch2.json"
    second_manifest.write_text(json.dumps(result_rows), encoding="utf-8")
    audit_path = import_feedback(work, 2, second_manifest)
    audit = json.loads(audit_path.read_text())
    assert audit["n_with_predictions"] == 6
    assert audit["order"] == "saved_predictions_before_training_new_batch"
    train_models(work, 2)
    state = json.loads((work / "state.json").read_text())
    assert state["measured_budget"] == 12
    assert state["batches"]["2"]["trained_model_dir"] == "models/batch-002"


def test_dual_checkpoint_shape_guard_rejects_old_17_point_model():
    from script.exploration import _validate_checkpoint
    state = {"measurement_id": "m49", "score_spec_id": "s5"}
    old = {"schema_version": 1, "mode": "dual_margin5", "output_dim": 17,
           "measurement_id": "m17", "score_spec_id": "s5"}
    with pytest.raises(ValueError, match="output_dim|measurement_id"):
        _validate_checkpoint(old, state, "dual_margin5", 5)


def test_dual_49_point_response_becomes_five_margins_in_declared_order():
    from script.exploration import _score_dual
    freqs = np.linspace(16, 40, 49)
    response = torch.stack((torch.linspace(-20, -8, 49),
                            torch.linspace(-30, -10, 49),
                            torch.linspace(-18, -6, 49)))
    bands = [
        {"name": "s11_lo", "label": "S11", "start_ghz": 16.0, "stop_ghz": 20.0,
         "relation": "le", "threshold": -10.0},
        {"name": "s11_hi", "label": "S11", "start_ghz": 20.5, "stop_ghz": 24.0,
         "relation": "le", "threshold": -9.0},
        {"name": "iso", "label": "S21", "start_ghz": 24.5, "stop_ghz": 30.0,
         "relation": "le", "threshold": -12.0},
        {"name": "s22_lo", "label": "S22", "start_ghz": 30.5, "stop_ghz": 35.0,
         "relation": "le", "threshold": -8.0},
        {"name": "s22_hi", "label": "S22", "start_ghz": 35.5, "stop_ghz": 40.0,
         "relation": "le", "threshold": -5.0},
    ]
    got = _score_dual(response, {}, {"name": "five", "bands": bands},
                      ["S11", "S21", "S22"], freqs)
    assert got.shape == (5,)
    assert np.all(np.isfinite(got))


def test_formal_single_and_dual_configs_prepare_profiled_worker_inputs(tmp_path):
    import shutil
    from script.profiled_batch import load_profile_config, validate_input

    repo = Path(__file__).resolve().parents[1]
    dataset_root = tmp_path / "dataset"; dataset_root.mkdir()

    single_config = repo / "configs" / "single_r80_symmetry_explore.yaml"
    single_work = tmp_path / "single-work"
    single_input = prepare_run(single_config, dataset_root, single_work, [])
    single_cfg = load_profile_config(single_config)
    single_rows = validate_input(single_input, single_cfg)
    assert len(single_rows) == 48
    moved_single = tmp_path / "nas-copy" / "single" / single_input.name
    shutil.copytree(single_input, moved_single)
    assert len(validate_input(moved_single, load_profile_config(moved_single / "config.yaml"))) == 48

    seed_dir = tmp_path / "dual-seeds"; seed_dir.mkdir()
    rng = np.random.default_rng(811007)
    seed_rows = []
    roles = ["history"] * 20 + ["specialist"] * 20 + ["random"] * 20
    for i, role in enumerate(roles):
        p = enforce_dual_geometry(rng.random((25, 25)) < 0.5)
        sid = f"seed_{i:03d}"
        torch.save(torch.as_tensor(p, dtype=torch.float32), seed_dir / f"{sid}.pt")
        seed_rows.append({"id": sid, "pattern_file": f"{sid}.pt", "source_group": role,
                          "lineage_id": sid, "source": "generated_contract_fixture"})
    (seed_dir / "manifest.json").write_text(json.dumps(seed_rows), encoding="utf-8")
    dual_config = repo / "configs" / "dual_r81_wide_filter.yaml"
    dual_work = tmp_path / "dual-work"
    dual_input = prepare_run(dual_config, dataset_root, dual_work, [seed_dir])
    dual_cfg = load_profile_config(dual_config)
    dual_rows = validate_input(dual_input, dual_cfg)
    assert len(dual_rows) == 60
    moved_dual = tmp_path / "nas-copy" / "dual" / dual_input.name
    shutil.copytree(dual_input, moved_dual)
    assert len(validate_input(moved_dual, load_profile_config(moved_dual / "config.yaml"))) == 60
    assert not ({row["id"] for row in single_rows} & {row["id"] for row in dual_rows})
