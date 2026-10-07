"""量測與評分規格的驗證、識別及純函式評分工具。

量測規格描述「量了什麼」，評分規格描述「怎樣才算通過」。兩者分開雜湊，
可避免只修改門檻時誤把既有量測資料認成另一種儀器輸出。
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy

import numpy as np


_MEASUREMENT_KEYS = {"name", "port", "labels", "geometry", "sweep", "solver"}
_GEOMETRY_KEYS = {"geom", "pixel_count", "diag_bridge_w"}
_SWEEP_KEYS = {"start_ghz", "stop_ghz", "step_ghz", "type"}
_SOLVER_KEYS = {
    "setup_freq_ghz",
    "open_region_freq_ghz",
    "max_delta_s",
    "max_passes",
    "min_passes",
    "min_converged",
}
_SCORE_KEYS = {"name", "bands"}
_BAND_KEYS = {"name", "label", "start_ghz", "stop_ghz", "relation", "threshold"}
_PORTS = {"single", "dual"}
_SWEEP_TYPES = {"Fast", "Interpolating", "Discrete"}


def _mapping(value, name: str) -> dict:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} 必須是 dict")
    return deepcopy(dict(value))


def _exact_keys(value: Mapping, expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{name} 欄位不符：缺少 {missing}，多出 {extra}")


def _text(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必須是非空字串")
    return value.strip()


def _finite(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} 必須是有限數值")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必須是有限數值") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} 必須是有限數值")
    return out


def _positive_int(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} 必須是正整數")
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必須是正整數") from exc
    if out <= 0 or out != value:
        raise ValueError(f"{name} 必須是正整數")
    return out


def _validate_sweep(sweep) -> dict:
    out = _mapping(sweep, "sweep")
    _exact_keys(out, _SWEEP_KEYS, "sweep")
    start = _finite(out["start_ghz"], "sweep.start_ghz")
    stop = _finite(out["stop_ghz"], "sweep.stop_ghz")
    step = _finite(out["step_ghz"], "sweep.step_ghz")
    if stop <= start:
        raise ValueError("sweep.stop_ghz 必須大於 start_ghz")
    if start <= 0:
        raise ValueError("sweep.start_ghz 必須大於 0")
    if step <= 0:
        raise ValueError("sweep.step_ghz 必須大於 0")
    intervals = (stop - start) / step
    rounded = round(intervals)
    if not math.isclose(intervals, rounded, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("sweep 範圍必須能被 step_ghz 整除，才能包含兩端點")
    sweep_type = _text(out["type"], "sweep.type")
    if sweep_type not in _SWEEP_TYPES:
        raise ValueError(f"sweep.type 只支援 {sorted(_SWEEP_TYPES)}")
    out.update(start_ghz=start, stop_ghz=stop, step_ghz=step, type=sweep_type)
    return out


def validate_measurement(measurement) -> dict:
    """驗證並正規化 measurement，回傳不與輸入共享容器的 dict。"""
    out = _mapping(measurement, "measurement")
    _exact_keys(out, _MEASUREMENT_KEYS, "measurement")
    out["name"] = _text(out["name"], "measurement.name")
    out["port"] = _text(out["port"], "measurement.port")
    if out["port"] not in _PORTS:
        raise ValueError(f"measurement.port 只支援 {sorted(_PORTS)}")

    labels = out["labels"]
    if isinstance(labels, (str, bytes)) or not isinstance(labels, Sequence) or not labels:
        raise ValueError("measurement.labels 必須是非空字串序列")
    labels = [_text(label, "measurement.labels[]") for label in labels]
    if len(set(labels)) != len(labels):
        raise ValueError("measurement.labels 不可重複")
    out["labels"] = labels

    geometry = _mapping(out["geometry"], "measurement.geometry")
    _exact_keys(geometry, _GEOMETRY_KEYS, "measurement.geometry")
    geometry["geom"] = _text(geometry["geom"], "measurement.geometry.geom")
    geometry["pixel_count"] = _positive_int(geometry["pixel_count"], "measurement.geometry.pixel_count")
    bridge = geometry["diag_bridge_w"]
    geometry["diag_bridge_w"] = None if bridge is None else _finite(bridge, "measurement.geometry.diag_bridge_w")
    if geometry["diag_bridge_w"] is not None and geometry["diag_bridge_w"] < 0:
        raise ValueError("measurement.geometry.diag_bridge_w 必須大於等於 0 或為 None")
    out["geometry"] = geometry
    out["sweep"] = _validate_sweep(out["sweep"])

    solver = _mapping(out["solver"], "measurement.solver")
    _exact_keys(solver, _SOLVER_KEYS, "measurement.solver")
    for key in ("setup_freq_ghz", "open_region_freq_ghz", "max_delta_s"):
        solver[key] = _finite(solver[key], f"measurement.solver.{key}")
    if solver["max_delta_s"] <= 0:
        raise ValueError("measurement.solver.max_delta_s 必須大於 0")
    for key in ("max_passes", "min_passes", "min_converged"):
        solver[key] = _positive_int(solver[key], f"measurement.solver.{key}")
    if solver["min_passes"] > solver["max_passes"]:
        raise ValueError("measurement.solver.min_passes 不可大於 max_passes")
    if solver["min_converged"] > solver["min_passes"]:
        raise ValueError("measurement.solver.min_converged 不可大於 min_passes")
    start, stop = out["sweep"]["start_ghz"], out["sweep"]["stop_ghz"]
    if not start <= solver["setup_freq_ghz"] <= stop:
        raise ValueError("measurement.solver.setup_freq_ghz 必須落在 sweep 範圍內")
    if solver["open_region_freq_ghz"] <= 0:
        raise ValueError("measurement.solver.open_region_freq_ghz 必須大於 0")
    out["solver"] = solver
    return out


def validate_score_spec(score_spec, *, measurement=None) -> dict:
    """驗證評分 bands；可選 measurement 會再檢查 label 與頻率端點。"""
    out = _mapping(score_spec, "score_spec")
    _exact_keys(out, _SCORE_KEYS, "score_spec")
    out["name"] = _text(out["name"], "score_spec.name")
    bands = out["bands"]
    if isinstance(bands, (str, bytes)) or not isinstance(bands, Sequence):
        raise ValueError("score_spec.bands 必須是序列")

    checked = []
    names = set()
    for index, raw_band in enumerate(bands):
        band = _mapping(raw_band, f"score_spec.bands[{index}]")
        _exact_keys(band, _BAND_KEYS, f"score_spec.bands[{index}]")
        band["name"] = _text(band["name"], f"score_spec.bands[{index}].name")
        if band["name"] in names:
            raise ValueError(f"score_spec band name 重複：{band['name']}")
        names.add(band["name"])
        band["label"] = _text(band["label"], f"score_spec.bands[{index}].label")
        band["start_ghz"] = _finite(band["start_ghz"], f"score_spec.bands[{index}].start_ghz")
        band["stop_ghz"] = _finite(band["stop_ghz"], f"score_spec.bands[{index}].stop_ghz")
        if band["stop_ghz"] < band["start_ghz"]:
            raise ValueError(f"score_spec band {band['name']} 的 stop_ghz 小於 start_ghz")
        relation = _text(band["relation"], f"score_spec.bands[{index}].relation").lower()
        if relation not in {"ge", "le"}:
            raise ValueError(f"score_spec band {band['name']} relation 只能是 ge 或 le")
        band["relation"] = relation
        band["threshold"] = _finite(band["threshold"], f"score_spec.bands[{index}].threshold")
        checked.append(band)
    out["bands"] = checked

    if measurement is not None:
        measured = validate_measurement(measurement)
        grid = frequency_grid(measured)
        label_set = set(measured["labels"])
        tolerance = max(1e-9, measured["sweep"]["step_ghz"] * 1e-9)
        for band in checked:
            if band["label"] not in label_set:
                raise ValueError(f"score_spec band {band['name']} 使用 measurement 沒有的 label：{band['label']}")
            for key in ("start_ghz", "stop_ghz"):
                if not np.any(np.isclose(grid, band[key], rtol=0.0, atol=tolerance)):
                    raise ValueError(f"score_spec band {band['name']} 的 {key} 不在 measurement 網格上")
    return out


def _canonical_id(value: dict) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def measurement_id(measurement) -> str:
    """回傳只由量測規格決定的 canonical SHA-256。"""
    return _canonical_id(validate_measurement(measurement))


def score_spec_id(score_spec) -> str:
    """回傳含 threshold 的獨立 canonical SHA-256。"""
    return _canonical_id(validate_score_spec(score_spec))


def frequency_grid(spec) -> np.ndarray:
    """由 measurement 或 sweep dict 建立包含兩端點的 GHz 網格。"""
    if not isinstance(spec, Mapping):
        raise TypeError("spec 必須是 measurement 或 sweep dict")
    if set(spec) == _SWEEP_KEYS:
        sweep = _validate_sweep(spec)
    else:
        sweep = validate_measurement(spec)["sweep"]
    count = int(round((sweep["stop_ghz"] - sweep["start_ghz"]) / sweep["step_ghz"])) + 1
    return np.linspace(sweep["start_ghz"], sweep["stop_ghz"], count, dtype=float)


def _numpy_response(response) -> np.ndarray:
    if hasattr(response, "detach"):
        response = response.detach()
    if hasattr(response, "cpu"):
        response = response.cpu()
    if hasattr(response, "numpy"):
        response = response.numpy()
    try:
        return np.asarray(response, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("response 必須能轉成有限數值陣列") from exc


def score_response(response, labels, freqs, score_spec) -> dict:
    """依每個 band 的最差點計算 margin；transition 頻帶因未列入 bands 而不評分。"""
    spec = validate_score_spec(score_spec)
    if isinstance(labels, (str, bytes)) or not isinstance(labels, Sequence) or not labels:
        raise ValueError("labels 必須是非空字串序列")
    labels = [_text(label, "labels[]") for label in labels]
    if len(set(labels)) != len(labels):
        raise ValueError("labels 不可重複")

    values = _numpy_response(response)
    frequencies = np.asarray(freqs, dtype=float)
    if values.ndim != 2 or values.shape[0] != len(labels):
        raise ValueError(f"response 形狀必須是 ({len(labels)}, F)")
    if frequencies.ndim != 1 or values.shape[1] != frequencies.size:
        raise ValueError("freqs 必須是一維，且長度等於 response 的頻率軸")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(frequencies)):
        raise ValueError("response 與 freqs 必須全為有限值")
    if frequencies.size == 0 or (frequencies.size > 1 and not np.all(np.diff(frequencies) > 0)):
        raise ValueError("freqs 必須嚴格遞增且非空")
    if frequencies.size > 2:
        steps = np.diff(frequencies)
        if not np.allclose(steps, steps[0], rtol=0.0, atol=max(1e-12, abs(steps[0]) * 1e-9)):
            raise ValueError("freqs 必須是完整的等步距網格，不可缺少中間頻點")

    label_index = {label: index for index, label in enumerate(labels)}
    margins, worst_frequencies, passed = {}, {}, {}
    for band in spec["bands"]:
        if band["label"] not in label_index:
            raise ValueError(f"score_spec band {band['name']} 使用 response 沒有的 label：{band['label']}")
        tolerance = max(1e-9, abs(band["stop_ghz"] - band["start_ghz"]) * 1e-12)
        start_hits = np.flatnonzero(np.isclose(frequencies, band["start_ghz"], rtol=0.0, atol=tolerance))
        stop_hits = np.flatnonzero(np.isclose(frequencies, band["stop_ghz"], rtol=0.0, atol=tolerance))
        if start_hits.size != 1 or stop_hits.size != 1:
            raise ValueError(f"score_spec band {band['name']} 的兩端點未被 freqs 完整覆蓋")
        lo, hi = int(start_hits[0]), int(stop_hits[0])
        if lo > hi:
            raise ValueError(f"score_spec band {band['name']} 的頻率索引順序錯誤")
        band_values = values[label_index[band["label"]], lo:hi + 1]
        point_margins = (band_values - band["threshold"] if band["relation"] == "ge"
                         else band["threshold"] - band_values)
        worst_index = int(np.argmin(point_margins))
        margin = float(point_margins[worst_index])
        margins[band["name"]] = margin
        worst_frequencies[band["name"]] = float(frequencies[lo + worst_index])
        passed[band["name"]] = bool(margin >= 0.0)

    if margins:
        worst_band = min(margins, key=margins.get)
        worst_margin = margins[worst_band]
    else:
        worst_band = None
        worst_margin = None
    return {
        "margins": margins,
        "worst_margin": worst_margin,
        "worst_band": worst_band,
        "worst_frequencies": worst_frequencies,
        "passed": passed,
        "all_passed": all(passed.values()),
    }
