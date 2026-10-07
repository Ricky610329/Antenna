"""量測身份、wide 頻率網格與規格評分的純 CPU 契約。"""
from copy import deepcopy

import numpy as np
import pytest

from antenna.measurement import (
    frequency_grid,
    measurement_id,
    score_response,
    score_spec_id,
    validate_measurement,
    validate_score_spec,
)


def _measurement():
    return {
        "name": "dual-wide-v1",
        "port": "dual",
        "labels": ["S11", "S21", "S22"],
        "geometry": {"geom": "p01", "pixel_count": 25, "diag_bridge_w": None},
        "sweep": {"start_ghz": 16, "stop_ghz": 40, "step_ghz": 0.5, "type": "Fast"},
        "solver": {
            "setup_freq_ghz": 28,
            "open_region_freq_ghz": 16,
            "max_delta_s": 0.02,
            "max_passes": 6,
            "min_passes": 5,
            "min_converged": 5,
        },
    }


def _score_spec():
    return {
        "name": "dual-wide-score-v1",
        "bands": [
            {"name": "pass_s21", "label": "S21", "start_ghz": 26, "stop_ghz": 30,
             "relation": "ge", "threshold": -3},
            {"name": "stop_low_s21", "label": "S21", "start_ghz": 16, "stop_ghz": 20,
             "relation": "le", "threshold": -20},
            {"name": "stop_high_s21", "label": "S21", "start_ghz": 36, "stop_ghz": 40,
             "relation": "le", "threshold": -20},
            {"name": "match_s11", "label": "S11", "start_ghz": 26.5, "stop_ghz": 29.5,
             "relation": "le", "threshold": -10},
            {"name": "match_s22", "label": "S22", "start_ghz": 26.5, "stop_ghz": 29.5,
             "relation": "le", "threshold": -10},
        ],
    }


def test_wide_frequency_grid_has_49_exact_points():
    grid = frequency_grid(_measurement())
    assert grid.shape == (49,)
    assert grid[0] == 16 and grid[-1] == 40
    np.testing.assert_array_equal(grid, np.linspace(16, 40, 49))


def test_frequency_grid_accepts_sweep_and_rejects_nonintegral_range():
    assert frequency_grid(_measurement()["sweep"]).tolist() == np.linspace(16, 40, 49).tolist()
    bad = deepcopy(_measurement()["sweep"])
    bad["step_ghz"] = 0.7
    with pytest.raises(ValueError, match="整除"):
        frequency_grid(bad)


def test_validation_normalizes_without_mutating_input():
    raw = _measurement()
    got = validate_measurement(raw)
    assert got is not raw and got["geometry"] is not raw["geometry"]
    assert isinstance(got["sweep"]["start_ghz"], float)
    got["labels"].append("extra")
    assert raw["labels"] == ["S11", "S21", "S22"]
    assert validate_score_spec(_score_spec(), measurement=raw)["bands"][0]["threshold"] == -3.0


@pytest.mark.parametrize("edit, message", [
    (lambda m: m["labels"].append("S21"), "不可重複"),
    (lambda m: m.update(port="triple"), "port 只支援"),
    (lambda m: m["sweep"].update(type="Linear"), "type 只支援"),
    (lambda m: m["sweep"].update(start_ghz=0), "start_ghz 必須大於 0"),
    (lambda m: m["geometry"].update(diag_bridge_w=-0.05), "大於等於 0"),
    (lambda m: m["solver"].update(setup_freq_ghz=41), "sweep 範圍"),
    (lambda m: m["solver"].update(open_region_freq_ghz=float("nan")), "有限數值"),
    (lambda m: m["solver"].update(min_passes=7), "max_passes"),
])
def test_measurement_validation_is_strict(edit, message):
    measurement = _measurement()
    edit(measurement)
    with pytest.raises(ValueError, match=message):
        validate_measurement(measurement)


def test_canonical_ids_ignore_dict_order_and_numeric_spelling():
    first = _measurement()
    second = {key: deepcopy(first[key]) for key in reversed(first)}
    second["sweep"]["start_ghz"] = 16.0
    assert measurement_id(first) == measurement_id(second)
    assert len(measurement_id(first)) == 64

    score = _score_spec()
    reordered = {"bands": deepcopy(score["bands"]), "name": score["name"]}
    assert score_spec_id(score) == score_spec_id(reordered)
    changed = deepcopy(score)
    changed["bands"][0]["threshold"] = -2.5
    assert score_spec_id(score) != score_spec_id(changed)
    assert measurement_id(first) == measurement_id(_measurement())


def test_score_spec_checks_measurement_labels_and_grid():
    bad_label = _score_spec()
    bad_label["bands"][0]["label"] = "S12"
    with pytest.raises(ValueError, match="沒有的 label"):
        validate_score_spec(bad_label, measurement=_measurement())
    off_grid = _score_spec()
    off_grid["bands"][0]["start_ghz"] = 26.1
    with pytest.raises(ValueError, match="不在 measurement 網格"):
        validate_score_spec(off_grid, measurement=_measurement())


def _passing_response():
    freqs = frequency_grid(_measurement())
    response = np.full((3, len(freqs)), -12.0)
    s21 = response[1]
    s21[(freqs >= 26) & (freqs <= 30)] = -2.0
    s21[freqs == 28] = -2.8
    s21[(freqs >= 16) & (freqs <= 20)] = -21.0
    s21[freqs == 18] = -20.5
    s21[(freqs >= 36) & (freqs <= 40)] = -22.0
    response[0, freqs == 27.5] = -10.2
    response[2, freqs == 29] = -10.5
    # transition 刻意極差；未列入 bands，不能影響 margin。
    s21[(freqs > 20) & (freqs < 26)] = 100.0
    s21[(freqs > 30) & (freqs < 36)] = 100.0
    return response, freqs


def test_score_response_reports_each_worst_margin_and_frequency():
    response, freqs = _passing_response()
    result = score_response(response, ["S11", "S21", "S22"], freqs, _score_spec())
    assert result["all_passed"] is True
    assert result["margins"] == pytest.approx({
        "pass_s21": 0.2,
        "stop_low_s21": 0.5,
        "stop_high_s21": 2.0,
        "match_s11": 0.2,
        "match_s22": 0.5,
    })
    assert result["worst_margin"] == pytest.approx(0.2)
    assert result["worst_band"] == "match_s11"
    assert result["worst_frequencies"]["pass_s21"] == 28.0
    assert result["worst_frequencies"]["stop_low_s21"] == 18.0
    assert result["worst_frequencies"]["match_s11"] == 27.5


def test_score_response_requires_finite_values_and_band_endpoints():
    response, freqs = _passing_response()
    missing_start = freqs[1:]
    with pytest.raises(ValueError, match="兩端點"):
        score_response(response[:, 1:], ["S11", "S21", "S22"], missing_start, _score_spec())
    response[1, 10] = np.nan  # transition 也必須有限，不能把壞 CSV 藏在不計分區。
    with pytest.raises(ValueError, match="全為有限值"):
        score_response(response, ["S11", "S21", "S22"], freqs, _score_spec())

    response, freqs = _passing_response()
    keep = np.arange(len(freqs)) != 5
    with pytest.raises(ValueError, match="缺少中間頻點"):
        score_response(response[:, keep], ["S11", "S21", "S22"], freqs[keep], _score_spec())


def test_empty_bands_are_valid_for_exploration():
    response, freqs = _passing_response()
    empty = {"name": "explore-no-score", "bands": []}
    assert validate_score_spec(empty) == empty
    result = score_response(response, ["S11", "S21", "S22"], freqs, empty)
    assert result == {
        "margins": {}, "worst_margin": None, "worst_band": None,
        "worst_frequencies": {}, "passed": {}, "all_passed": True,
    }
