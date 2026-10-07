from pathlib import Path

import numpy as np
import pytest

from antenna.measurement import measurement_id, score_spec_id
from script import exploration as ex
from script import profiled_batch as pb
from script.symmetry_sm_pool import (
    LegacySurrogatePredictor,
    PoolConfig,
    PredictionBatch,
    build_pool,
    score_predictions,
    write_bundle,
)


REPO = Path(__file__).resolve().parents[1]
PROFILE = REPO / "configs" / "single_r80_symmetry_explore.yaml"
LOCAL_MODELS = Path(r"C:\Users\ricky\antenna_nas_backup\dataset")
MODEL_NAMES = ("sm_reanchor108.pth", "sm_ens108_1.pth", "sm_ens108_2.pth")
RAD_NAME = "rad_head108.pth"
PROFILE_CFG = pb.load_profile_config(PROFILE)


class FakePredictor:
    model_role = "current_profile"
    binding = {"measurement_id": measurement_id(PROFILE_CFG.measurement),
               "score_spec_id": score_spec_id(PROFILE_CFG.score_spec),
               "protocol_id": "fixture-protocol", "data_version": 1,
               "manifest_id": "fixture-manifest", "cumulative_valid_unique": 48}
    model_ids = [{"file": "fixture", "sha256": "0" * 64,
                  "architecture": "deterministic_fixture", "source": __file__,
                  "protocol_id": "fixture-protocol", "data_version": "1",
                  "manifest_id": "fixture-manifest"}]

    def predict(self, patterns, *, batch_size):
        x = np.stack([ex.enforce_single_geometry(p) for p in patterns]).astype(np.float32)
        n = len(x)
        density = x.mean((1, 2))
        asym_feature = x[:, :12, :6].mean((1, 2)) - x[:, 12:, :6].mean((1, 2))
        response = np.empty((n, 2, 17), dtype=np.float32)
        response[:, 0] = (-8.0 - 6.0 * density)[:, None]
        response[:, 1] = (2.0 + 7.0 * density + asym_feature)[:, None]
        theta = np.linspace(-90.0, 90.0, 91, dtype=np.float32)
        radiation = np.empty((n, 2, 91), dtype=np.float32)
        shape = -2.0 * np.abs(theta) / 90.0
        radiation[:, 0] = density[:, None] + shape
        radiation[:, 1] = density[:, None] + 0.8 * shape
        disagreement = np.abs(density - 0.5).astype(np.float32)
        return PredictionBatch(response, disagreement, radiation, theta)


class DifferentFakePredictor(FakePredictor):
    def predict(self, patterns, *, batch_size):
        pred = super().predict(patterns, batch_size=batch_size)
        response = pred.response_mean.copy()
        response[:, 0] = -3.0 * response[:, 0]
        response[:, 1] = -2.0 * response[:, 1]
        disagreement = 5.0 - pred.response_disagreement
        return PredictionBatch(response, disagreement, pred.radiation, pred.radiation_theta)


def _pool_config(**updates):
    values = dict(profile_config=PROFILE, candidate_pool_size=96, selected_count=24,
                  initial_pool_size=32, generations=2, parent_count=12,
                  max_per_ancestry=6, min_mutation_flips=2, max_mutation_flips=12,
                  seed=77, prediction_batch_size=32)
    values.update(updates)
    return PoolConfig(**values)


def _seed_rows(n=5):
    rng = np.random.default_rng(123)
    return [{"id": f"history-{i}", "pattern": ex.random_single_pattern(rng)} for i in range(n)]


def test_factory_score_uses_265_to_295_and_rad_is_navigation_only():
    response = np.empty((2, 2, 17), dtype=np.float32)
    response[:, 0] = -12.0
    response[:, 1] = 6.0
    # Very bad values outside the exact seven-point factory band must not matter.
    response[:, 0, :5] = 99.0
    response[:, 0, 12:] = 99.0
    response[:, 1, :5] = -99.0
    response[:, 1, 12:] = -99.0
    theta = np.asarray([-90.0, -45.0, 0.0, 45.0, 90.0], dtype=np.float32)
    radiation = np.asarray([
        [[0.0, 7.0, 10.0, 7.0, 0.0], [0.0, 8.0, 10.0, 8.0, 0.0]],
        [[0.0, 5.0, 10.0, 5.0, 0.0], [0.0, 6.0, 10.0, 6.0, 0.0]],
    ], dtype=np.float32)
    pred = PredictionBatch(response, np.asarray([0.0, 1.0]), radiation, theta)
    scores = score_predictions(pred, radiation_weight=0.2, uncertainty_weight=0.1)
    assert scores["s11_margin"].tolist() == pytest.approx([2.0, 2.0])
    assert scores["gain_margin"].tolist() == pytest.approx([2.0, 2.0])
    assert scores["factory_score"].tolist() == pytest.approx([2.0, 2.0])
    assert scores["radiation_margin"].tolist() == pytest.approx([0.0, -2.0])
    assert scores["navigation_score"].tolist() == pytest.approx([2.0, 1.5])
    assert scores["lcb_score"].tolist() == pytest.approx([2.0, 0.6])


def test_build_pool_is_finite_exact_symmetric_deduplicated_and_excluded():
    seeds = _seed_rows()
    excluded = ex.pattern_sha256(seeds[0]["pattern"])
    result = build_pool(_pool_config(), seeds, [excluded], predictor=FakePredictor())
    rows, selected = result["rows"], result["selected_rows"]
    assert len(rows) == 96
    assert len(selected) == 24
    hashes = [row["pattern_sha256"] for row in rows]
    assert len(hashes) == len(set(hashes)) and excluded not in hashes
    for row in rows:
        p = row["pattern"]
        assert p.shape == (25, 25) and p.dtype == np.bool_
        assert np.array_equal(p, p[:, ::-1])
        assert p[24, 12]
        assert np.isfinite([row["s11_margin"], row["gain_margin"],
                            row["factory_score"], row["radiation_margin"],
                            row["response_disagreement"], row["navigation_score"]]).all()
        assert len(row["predicted_s11"]) == len(row["predicted_gain"]) == 17
        assert len(row["predicted_radiation_theta"]) == 91
    assert {row["selection_arm"] for row in selected} == {
        "lcb_performance", "high_disagreement", "blind_random"
    }
    assert len({row["ancestry_id"] for row in selected}) >= 2


def test_distinct_seed_bits_keep_shared_explicit_lineage():
    seeds = _seed_rows(2)
    seeds[0]["lineage_id"] = seeds[1]["lineage_id"] = "t07_top"
    result = build_pool(_pool_config(candidate_pool_size=16, initial_pool_size=8,
                                     selected_count=4, generations=1),
                        seeds, [], predictor=FakePredictor())
    explicit = [row for row in result["rows"] if row["proposal"] == "explicit_seed"]
    assert len(explicit) == 2
    assert {row["source_id"] for row in explicit} == {"history-0", "history-1"}
    assert {row["ancestry_id"] for row in explicit} == {"t07_top"}
    descendants = [row for row in result["rows"] if row["source_id"] in {"history-0", "history-1"}]
    assert descendants and {row["ancestry_id"] for row in descendants} == {"t07_top"}


def test_three_arm_counts_are_13_10_9_and_disagreement_obeys_p60_cut():
    result = build_pool(_pool_config(candidate_pool_size=128, initial_pool_size=40,
                                     selected_count=32),
                        _seed_rows(), [], predictor=FakePredictor())
    selected = result["selected_rows"]
    assert {arm: sum(row["selection_arm"] == arm for row in selected) for arm in (
        "lcb_performance", "high_disagreement", "blind_random"
    )} == {"lcb_performance": 13, "high_disagreement": 10, "blind_random": 9}
    disagreement = [row for row in selected if row["selection_arm"] == "high_disagreement"]
    assert all(row["navigation_mean_score"] >= row["disagreement_eligibility_mean_p60"]
               for row in disagreement)


def test_blind_arm_is_independent_of_sm_predictions():
    cfg = _pool_config(candidate_pool_size=128, initial_pool_size=40, selected_count=32,
                       id_prefix="r80b2")
    seeds = _seed_rows()
    first = build_pool(cfg, seeds, [], predictor=FakePredictor())
    second = build_pool(cfg, seeds, [], predictor=DifferentFakePredictor())
    blind = lambda result: [row["pattern_sha256"] for row in result["selected_rows"]
                            if row["selection_arm"] == "blind_random"]
    assert blind(first) == blind(second)
    assert all(not row["blind_origin_fallback"] for row in first["selected_rows"]
               if row["selection_arm"] == "blind_random")


def test_bundle_is_accepted_by_profiled_batch_bridge_gate(tmp_path):
    result = build_pool(_pool_config(candidate_pool_size=48, initial_pool_size=16,
                                     selected_count=12), _seed_rows(), [],
                        predictor=FakePredictor())
    output = write_bundle(result, tmp_path / "blind_input")
    cfg = pb.load_profile_config(output / "config.yaml")
    rows = pb.validate_input(output, cfg)
    assert len(rows) == 12
    assert cfg.measurement["geometry"]["diag_bridge_w"] == pytest.approx(0.1)
    assert all(row["prediction_is_navigation_only"] for row in rows)
    assert all(row["predicted_before_hfss"] for row in rows)
    assert all(row["surrogate_generation"] == "data-v001" for row in rows)
    assert all(row["valid_observations_at_fit"] == 48 for row in rows)
    assert all(row["update_every_valid_points"] == 48 for row in rows)
    assert all("legacy_geometry_limitation" not in row for row in rows)
    assert all(row["id"].startswith("r80b2_") for row in rows)


def test_selected_one_has_only_lcb_and_no_empty_arm_warning():
    result = build_pool(_pool_config(candidate_pool_size=8, initial_pool_size=4,
                                     selected_count=1, generations=1),
                        _seed_rows(2), [], predictor=FakePredictor())
    assert len(result["selected_rows"]) == 1
    assert result["selected_rows"][0]["selection_arm"] == "lcb_performance"


def test_wrong_bridge_profile_is_rejected(tmp_path):
    raw = (PROFILE.read_text(encoding="utf-8").replace("diag_bridge_w: 0.1",
                                                        "diag_bridge_w: 0.075"))
    wrong = tmp_path / "wrong.yaml"
    wrong.write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError, match="diag_bridge_w=0.1"):
        build_pool(_pool_config(profile_config=wrong), _seed_rows(), [],
                   predictor=FakePredictor())


def test_malicious_caller_cannot_override_predictor_fit_count():
    cfg = _pool_config(candidate_pool_size=8, initial_pool_size=4, selected_count=1,
                       generations=1, valid_observations_at_fit=53)
    with pytest.raises(ValueError, match="valid_observations_at_fit disagrees"):
        build_pool(cfg, _seed_rows(2), [], predictor=FakePredictor())


def test_current_predictor_measurement_mismatch_is_rejected():
    class WrongProfilePredictor(FakePredictor):
        binding = {**FakePredictor.binding, "measurement_id": "wrong-measurement"}

    with pytest.raises(ValueError, match="measurement_id does not match"):
        build_pool(_pool_config(candidate_pool_size=8, initial_pool_size=4,
                                selected_count=1, generations=1),
                   _seed_rows(2), [], predictor=WrongProfilePredictor())


@pytest.mark.skipif(not all((LOCAL_MODELS / name).is_file()
                            for name in (*MODEL_NAMES, RAD_NAME)),
                    reason="local historical v108 checkpoints are unavailable")
def test_actual_local_v108_models_predict_finite_symmetric_smoke():
    predictor = LegacySurrogatePredictor(LOCAL_MODELS, MODEL_NAMES, RAD_NAME)
    rng = np.random.default_rng(9)
    patterns = [ex.random_single_pattern(rng), ex.random_single_pattern(rng)]
    pred = predictor.predict(patterns, batch_size=2)
    scores = score_predictions(pred)
    assert pred.response_mean.shape == (2, 2, 17)
    assert pred.response_disagreement.shape == (2,)
    assert pred.radiation.shape == (2, 2, 91)
    assert pred.radiation_theta.shape == (91,)
    assert all(np.isfinite(value).all() for value in scores.values())
    assert [item["architecture"] for item in predictor.model_ids] == [
        "HFSSNet", "ResCNNNet", "ResCNNNet", "legacy_rad_mlp_cosine_head"
    ]
    cold = build_pool(_pool_config(candidate_pool_size=8, initial_pool_size=4,
                                   selected_count=1, generations=1),
                      _seed_rows(2), [], predictor=predictor)
    assert cold["surrogate_generation"] == "historical-v108-cold-start"
    assert cold["valid_observations_at_fit"] == 0
    assert cold["predictor_binding"] is None
