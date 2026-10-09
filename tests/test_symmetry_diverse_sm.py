import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from antenna.measurement import measurement_id, score_spec_id
from script import exploration as ex
from script import profiled_batch as pb
from script import symmetry_diverse_sm as diverse
from script import symmetry_sm_pool as sm


REPO = Path(__file__).resolve().parents[1]
POLICY_PATH = REPO / "configs" / "r80_diverse_sm_search_v1.json"
PROFILE = REPO / "configs" / "single_r80_symmetry_factory.yaml"
PROFILE_CFG = pb.load_profile_config(PROFILE)


def _policy():
    return diverse.load_policy(POLICY_PATH)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _fixture_protocol():
    value = {
        "measurement_id": measurement_id(PROFILE_CFG.measurement),
        "score_spec_id": score_spec_id(PROFILE_CFG.score_spec),
        "geometry_profile": "single_v1",
        "frequencies_ghz": list(np.arange(24.0, 32.01, 0.5)),
    }
    value["protocol_id"] = ex.content_id(value)
    return value


def _training_fixture(root: Path, count: int = 128):
    root.mkdir()
    (root / "profile/samples").mkdir(parents=True)
    (root / "profile/rad").mkdir(parents=True)
    mid = measurement_id(PROFILE_CFG.measurement)
    sid = score_spec_id(PROFILE_CFG.score_spec)
    protocol = _fixture_protocol()
    rng = np.random.default_rng(404)
    rows = []
    theta = torch.arange(-180, 181, 2, dtype=torch.float32)
    for index in range(count):
        pattern = ex.random_single_pattern(rng)
        response = torch.stack((torch.full((17,), -30.0),
                                torch.full((17,), 20.0 - index * 0.05)))
        rad = {"theta": theta, "phi0": torch.zeros_like(theta),
               "phi90": torch.ones_like(theta)}
        sample = root / f"profile/samples/sample-{index:04d}.pt"
        rad_path = root / f"profile/rad/rad-{index:04d}.pt"
        torch.save((torch.as_tensor(pattern, dtype=torch.float32), response), sample)
        torch.save(rad, rad_path)
        rows.append({
            "id": f"measured-{index:04d}", "source_observation_id": f"obs-{index:04d}",
            "source_store": "fixture", "source_sample_sha256": pb.file_sha256(sample),
            "source_rad_sha256": pb.file_sha256(rad_path),
            "sample_file": str(sample.relative_to(root)).replace("\\", "/"),
            "rad_file": str(rad_path.relative_to(root)).replace("\\", "/"),
            "pattern_sha256": ex.pattern_sha256(pattern),
            "lineage_id": f"line-{index % 16:02d}",
            "canonical_group_id": f"group-{index % 16:02d}",
            "measurement_id": mid, "score_spec_id": sid,
            "geometry_profile": "single_v1", "split": "profile_train",
        })
    manifest_id = ex.content_id(rows)
    _write_json(root / "protocol.json", protocol)
    _write_json(root / "state.json", {"latest_data_version": 1})
    _write_json(root / "data/data-v001/manifest.json", rows)
    _write_json(root / "data/data-v001/receipt.json", {
        "data_version": 1, "manifest_id": manifest_id,
        "cumulative_observations": count, "cumulative_valid_unique": count,
    })
    return rows, manifest_id


class FakePredictor:
    model_role = "current_profile"
    model_ids = [{"file": "fixture.pt", "sha256": "1" * 64,
                  "protocol_id": "pending", "manifest_id": "pending",
                  "data_version": "1"}]

    def __init__(self, manifest_id, count, protocol_id=None):
        protocol_id = protocol_id or _fixture_protocol()["protocol_id"]
        self.binding = {
            "measurement_id": measurement_id(PROFILE_CFG.measurement),
            "score_spec_id": score_spec_id(PROFILE_CFG.score_spec),
            "protocol_id": protocol_id, "data_version": 1,
            "manifest_id": manifest_id, "cumulative_valid_unique": count,
        }
        self.model_ids = [{**self.model_ids[0], "manifest_id": manifest_id,
                           "protocol_id": protocol_id}]


def _candidate_rows(count=150, *, aliases=False):
    rng = np.random.default_rng(808)
    rows = []
    for index in range(count):
        pattern = ex.random_single_pattern(rng)
        group = "one-group" if aliases else f"candidate-{index}"
        rows.append({
            "pattern": pattern, "pattern_sha256": ex.pattern_sha256(pattern),
            "candidate_group": "fresh_random" if index % 2 == 0 else "parent_variant",
            "canonical_group_id": group, "lcb_score": float(count - index),
            "navigation_mean_score": float(index),
            "response_disagreement": float((index * 17) % count),
        })
    return rows


def test_load_policy_accepts_only_the_fixed_complete_contract():
    policy = _policy()
    assert policy["candidate_pool_size"] == 20_000
    assert policy["source_pool_fractions"] == {"fresh_random": 0.5,
                                                "parent_variant": 0.5}


@pytest.mark.parametrize("change,match", [
    ({"surprise": 1}, "keys differ"),
    ({"candidate_pool_size": 19999}, "fixed v1 contract"),
    ({"arm_weights": [1, float("nan"), 1]}, "finite positive"),
    ({"measurement_profile": "configs/other.yaml"}, "measurement_profile"),
])
def test_policy_rejects_unknown_tampered_or_nonfinite_values(tmp_path, change, match):
    raw = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    raw.update(change)
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        diverse.load_policy(path)


def test_equal_largest_remainder_is_deterministic_for_short_tail():
    assert diverse._largest_remainder(8, _policy()["arm_order"], [1, 1, 1]) == {
        "global_lcb": 3, "parent_lcb": 3, "high_disagreement": 2}
    assert diverse._largest_remainder(1, _policy()["arm_order"], [1, 1, 1]) == {
        "global_lcb": 1, "parent_lcb": 0, "high_disagreement": 0}


def test_actual_wm_uses_only_exact_265_to_295_s11_gain_band():
    frequencies = np.arange(24.0, 32.01, 0.5)
    response = np.stack((np.full(17, -12.0), np.full(17, 6.0)))
    response[:, :5] = np.asarray([99.0, -99.0])[:, None]
    response[:, 12:] = np.asarray([99.0, -99.0])[:, None]
    assert diverse._wm(response, frequencies) == pytest.approx(2.0)
    with pytest.raises(ValueError, match="exact 26.5"):
        diverse._wm(np.delete(response, 5, axis=1), np.delete(frequencies, 5))


def test_measured_parent_choice_starts_at_best_and_enforces_cap_distance_groups(tmp_path):
    _training_fixture(tmp_path / "train")
    rows, _, _ = diverse._training_snapshot(tmp_path / "train", 1, PROFILE_CFG)
    parents = diverse._select_parents(rows, _policy())
    assert len(parents) == 32 and parents[0]["id"] == "measured-0000"
    assert max(Counter(row["canonical_group_id"] for row in parents).values()) <= 2
    assert len({row["canonical_group_id"] for row in parents}) >= 8
    assert min(ex.hamming(a["pattern"], b["pattern"])
               for i, a in enumerate(parents) for b in parents[i + 1:]) >= 64


def test_parent_choice_has_no_distance_or_group_fallback():
    pattern = ex.random_single_pattern(np.random.default_rng(4))
    rows = [{"id": str(i), "pattern": pattern.copy(), "pattern_sha256": f"{i:064x}",
             "canonical_group_id": "same", "actual_wm": float(100 - i)} for i in range(128)]
    with pytest.raises(RuntimeError, match="too few canonical groups"):
        diverse._select_parents(rows, _policy())


def test_parent_count_is_a_maximum_when_strict_geometry_leaves_8_to_31():
    rng = np.random.default_rng(87)
    rows = []
    for index in range(64):
        pattern = ex.random_single_pattern(rng)
        rows.append({"id": str(index), "pattern": pattern,
                     "pattern_sha256": ex.pattern_sha256(pattern),
                     "canonical_group_id": f"group-{index % 8}",
                     "actual_wm": float(100 - index)})
    parents = diverse._select_parents(rows, _policy())
    assert len(parents) == 16
    assert len({row["canonical_group_id"] for row in parents}) == 8


def test_whole_cohort_selection_obeys_origin_quota_distance_and_alias_cap():
    selected, audit = diverse._select(_candidate_rows(), 12, _policy())
    assert audit["arm_quotas"] == {"global_lcb": 4, "parent_lcb": 4,
                                    "high_disagreement": 4}
    assert all(row["candidate_group"] == "fresh_random" for row in selected
               if row["selection_arm"] == "global_lcb")
    assert all(row["candidate_group"] == "parent_variant" for row in selected
               if row["selection_arm"] == "parent_lcb")
    assert min(ex.hamming(a["pattern"], b["pattern"])
               for i, a in enumerate(selected) for b in selected[i + 1:]) >= 64
    assert max(Counter(row["canonical_group_id"] for row in selected).values()) <= 2


def test_whole_cohort_selection_strictly_fails_when_alias_cap_cannot_fill():
    with pytest.raises(RuntimeError, match="no-relaxation"):
        diverse._select(_candidate_rows(80, aliases=True), 6, _policy())


def test_actual_tensor_snapshot_binds_raw_hashes_and_measured_wm(tmp_path):
    rows, manifest_id = _training_fixture(tmp_path / "train")
    loaded, binding, tracked = diverse._training_snapshot(tmp_path / "train", 1, PROFILE_CFG)
    assert binding["manifest_id"] == manifest_id
    assert loaded[0]["actual_wm"] == pytest.approx(16.0)
    assert loaded[31]["actual_wm"] == pytest.approx(14.45)
    assert tracked[f"sample:{rows[0]['id']}"] == rows[0]["source_sample_sha256"]
    assert tracked[f"rad:{rows[0]['id']}"] == rows[0]["source_rad_sha256"]


def test_changed_sample_bytes_are_rejected_before_parent_use(tmp_path):
    rows, _ = _training_fixture(tmp_path / "train")
    sample = tmp_path / "train" / rows[0]["sample_file"]
    pattern, response = torch.load(sample, weights_only=True)
    torch.save((pattern, response + 1), sample)
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        diverse._training_snapshot(tmp_path / "train", 1, PROFILE_CFG)


def test_sample_change_between_load_and_final_rehash_is_rejected(tmp_path, monkeypatch):
    rows, _ = _training_fixture(tmp_path / "train")
    sample = tmp_path / "train" / rows[0]["sample_file"]
    original = ex.radiation_vector
    changed = False

    def mutate_after_sample_load(value):
        nonlocal changed
        if not changed:
            pattern, response = torch.load(sample, weights_only=True)
            torch.save((pattern, response + 0.25), sample)
            changed = True
        return original(value)

    monkeypatch.setattr(ex, "radiation_vector", mutate_after_sample_load)
    with pytest.raises(ValueError, match="cache changed"):
        diverse._training_snapshot(tmp_path / "train", 1, PROFILE_CFG)


def test_predictor_manifest_tamper_is_rejected_before_generation(tmp_path, monkeypatch):
    _, manifest_id = _training_fixture(tmp_path / "train")
    predictor = FakePredictor("f" * 64, 128)
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("generation must not run")

    monkeypatch.setattr(diverse, "_generate_pool", forbidden)
    cfg = sm.PoolConfig(profile_config=PROFILE, candidate_pool_size=20_000,
                        selected_count=48, seed=7)
    with pytest.raises(ValueError, match="predictor binding differs"):
        diverse.build_diverse_pool(pool_cfg=cfg, policy=_policy(),
                                   training_workdir=tmp_path / "train", version=1,
                                   exclusions=[], predictor=predictor)
    assert manifest_id != predictor.binding["manifest_id"] and not called


def test_predictor_protocol_tamper_is_rejected_before_generation(tmp_path, monkeypatch):
    _, manifest_id = _training_fixture(tmp_path / "train")
    predictor = FakePredictor(manifest_id, 128, protocol_id="e" * 64)
    monkeypatch.setattr(diverse, "_generate_pool",
                        lambda *args, **kwargs: pytest.fail("generation must not run"))
    cfg = sm.PoolConfig(profile_config=PROFILE, candidate_pool_size=20_000,
                        selected_count=48, seed=7)
    with pytest.raises(ValueError, match="predictor binding differs"):
        diverse.build_diverse_pool(pool_cfg=cfg, policy=_policy(),
                                   training_workdir=tmp_path / "train", version=1,
                                   exclusions=[], predictor=predictor)


def test_generated_candidates_exclude_all_measured_and_queued_and_use_all_parents(tmp_path):
    rows, _ = _training_fixture(tmp_path / "train")
    measured, _, _ = diverse._training_snapshot(tmp_path / "train", 1, PROFILE_CFG)
    parents = diverse._select_parents(measured, _policy())
    queued = ex.pattern_sha256(ex.random_single_pattern(np.random.default_rng(99)))
    pool = diverse._generate_pool(parents, _policy(), PROFILE_CFG,
                                  {row["pattern_sha256"] for row in rows} | {queued})
    assert len(pool) == 20_000
    assert Counter(row["candidate_group"] for row in pool) == {
        "fresh_random": 10_000, "parent_variant": 10_000}
    assert not ({row["pattern_sha256"] for row in pool} &
                ({row["pattern_sha256"] for row in rows} | {queued}))
    assert {row["parent_id"] for row in pool if row["candidate_group"] == "parent_variant"} == {
        row["id"] for row in parents}


def test_successful_build_is_deterministic_binds_model_and_emits_json_safe_audit(
        tmp_path, monkeypatch):
    rows, manifest_id = _training_fixture(tmp_path / "train")
    measured, _, _ = diverse._training_snapshot(tmp_path / "train", 1, PROFILE_CFG)
    parents = diverse._select_parents(measured, _policy())
    blocked = {row["pattern_sha256"] for row in rows}
    frozen_pool = diverse._generate_pool(parents, _policy(), PROFILE_CFG, blocked)

    def fixed_pool(*args, **kwargs):
        return [{**row, "pattern": row["pattern"].copy()} for row in frozen_pool]

    def fixed_scores(candidate_rows, predictor, cfg):
        for index, row in enumerate(candidate_rows):
            density = float(np.mean(row["pattern"]))
            row.update({
                "s11_margin": density, "gain_margin": density,
                "factory_score": density, "radiation_margin": 0.0,
                "response_disagreement": float((index * 31) % 997) / 997.0,
                "navigation_mean_score": density, "lcb_score": density - 0.1,
                "navigation_score": density - 0.01,
                "predicted_s11": [-12.0] * 17, "predicted_gain": [6.0] * 17,
                "predicted_radiation_phi0": [0.0] * 91,
                "predicted_radiation_phi90": [0.0] * 91,
                "predicted_radiation_theta": np.linspace(-90, 90, 91).tolist(),
            })

    monkeypatch.setattr(diverse, "_generate_pool", fixed_pool)
    monkeypatch.setattr(sm, "_score_rows", fixed_scores)
    predictor = FakePredictor(manifest_id, 128)
    cfg = sm.PoolConfig(profile_config=PROFILE, candidate_pool_size=20_000,
                        selected_count=12, seed=7, surrogate_generation="data-v001",
                        valid_observations_at_fit=128)
    first = diverse.build_diverse_pool(
        pool_cfg=cfg, policy=_policy(), training_workdir=tmp_path / "train",
        version=1, exclusions=[], predictor=predictor)
    second = diverse.build_diverse_pool(
        pool_cfg=cfg, policy=_policy(), training_workdir=tmp_path / "train",
        version=1, exclusions=[], predictor=predictor)
    assert first["predictor_binding"] == predictor.binding
    assert first["diversity_audit"]["candidate_pool_sha256"] == \
        second["diversity_audit"]["candidate_pool_sha256"]
    assert first["diversity_audit"]["selected_cohort_sha256"] == \
        second["diversity_audit"]["selected_cohort_sha256"]
    assert first["diversity_audit"]["candidate_origin_counts"] == {
        "fresh_random": 10_000, "parent_variant": 10_000}
    json.dumps(first["diversity_audit"], allow_nan=False)

