"""Measured-only R81 transition diagnostics and bounded parent ranking.

This opt-in preparation helper does not train a model, predict an unmeasured
candidate, select a batch, or dispatch work.  It authenticates already imported
R81 feedback, preserves the existing five-band worst margin, and records a
separate soft description of the two transition regions.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

from antenna.measurement import measurement_id, score_response, score_spec_id
from script import exploration as ex
from script import filter_training as ft
from script import profiled_batch as pb


REPO = Path(__file__).resolve().parents[1]
CANONICAL_CONFIG = REPO / "configs" / "dual_r81_wide_filter.yaml"
POLICY_ID = "r81_measured_transition_soft_v1"
TRUTH_KIND = "authenticated_current_r81_feedback"
SIDECAR_KIND = "r81_measured_transition_sidecar_v1"
RANKING_KIND = "r81_measured_transition_parent_ranking_v1"
DEFAULT_MAX_WM_TRADEOFF_DB = 0.1
EXPECTED_LABELS = ("S11", "S21", "S22")
EXPECTED_FREQS = np.linspace(16.0, 40.0, 49, dtype=np.float64)
IMPLEMENTATION_PATHS = {
    "transition_soft": Path(__file__).resolve(),
    "exploration": REPO / "script" / "exploration.py",
    "filter_training": REPO / "script" / "filter_training.py",
    "profiled_batch": REPO / "script" / "profiled_batch.py",
    "measurement": REPO / "antenna" / "measurement.py",
    "canonical_config": CANONICAL_CONFIG,
}


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    return pb.file_sha256(path)


def _path_label(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return str(path.resolve())


def _capture_files(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    captured: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        resolved = Path(path).resolve()
        try:
            raw = resolved.read_bytes()
        except (FileNotFoundError, IsADirectoryError) as exc:
            raise ValueError(f"required transition binding is missing: {name}") from exc
        captured[name] = {"path": resolved, "bytes": raw, "sha256": _sha_bytes(raw)}
    return captured


def _capture_hashes(captured: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    return {name: str(item["sha256"]) for name, item in captured.items()}


def _assert_unchanged(captured: Mapping[str, Mapping[str, Any]], message: str) -> None:
    for name, item in captured.items():
        if _sha_file(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"{message}: {name}")


def _json_bytes(item: Mapping[str, Any], name: str) -> Any:
    try:
        return json.loads(bytes(item["bytes"]).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid bound JSON: {name}") from exc


def _yaml_bytes(item: Mapping[str, Any], name: str) -> Any:
    try:
        return yaml.safe_load(bytes(item["bytes"]).decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid bound YAML: {name}") from exc


def _implementation_bindings(captured: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        name: {"path": _path_label(Path(item["path"])), "sha256": item["sha256"]}
        for name, item in captured.items()
        if name in IMPLEMENTATION_PATHS
    }


def _exact_axis(labels: Sequence[str], freqs: Sequence[float]) -> tuple[list[str], np.ndarray]:
    label_list = list(labels)
    frequency = np.asarray(freqs, dtype=np.float64)
    if label_list != list(EXPECTED_LABELS):
        raise ValueError("transition truth requires exact R81 labels [S11, S21, S22]")
    if frequency.shape != (49,) or not np.array_equal(frequency, EXPECTED_FREQS):
        raise ValueError("transition truth requires exact 16-40 GHz / 0.5 GHz 49-point axis")
    return label_list, frequency


def _side(values: np.ndarray, freqs: np.ndarray, ordered_freqs: np.ndarray) -> dict[str, Any]:
    indices = []
    for value in ordered_freqs:
        hits = np.flatnonzero(np.isclose(freqs, value, rtol=0.0, atol=1e-12))
        if hits.size != 1:
            raise ValueError(f"transition frequency {value:g} GHz is missing or duplicated")
        indices.append(int(hits[0]))
    oriented = values[indices].astype(np.float64, copy=False)
    interior = oriented[1:-1]
    relative = oriented[0] - interior
    outward_steps = np.diff(oriented)
    upward = np.maximum(outward_steps, 0.0)
    curvature = np.diff(oriented, n=2)
    return {
        "frequencies_ghz_pass_to_stop": ordered_freqs.astype(float).tolist(),
        "s21_db_pass_to_stop": oriented.astype(float).tolist(),
        "interior_frequencies_ghz": ordered_freqs[1:-1].astype(float).tolist(),
        "interior_relative_attenuation_db": relative.astype(float).tolist(),
        "mean_relative_attenuation_db": float(np.mean(relative)),
        "outward_rise_rms_db": float(np.sqrt(np.mean(upward ** 2))),
        "curvature_rms_db": float(np.sqrt(np.mean(curvature ** 2))),
    }


def diagnose_response(response: Any, labels: Sequence[str], freqs: Sequence[float],
                      score_spec: Mapping[str, Any]) -> dict[str, Any]:
    """Return five-band truth plus separate measured transition diagnostics."""
    label_list, frequency = _exact_axis(labels, freqs)
    values = np.asarray(torch.as_tensor(response, dtype=torch.float32).cpu(), dtype=np.float32)
    if values.shape != (3, 49) or not np.all(np.isfinite(values)):
        raise ValueError("transition truth requires a finite dual 3x49 response")
    canonical = ex._load_yaml(CANONICAL_CONFIG)
    if score_spec_id(score_spec) != score_spec_id(canonical["score_spec"]):
        raise ValueError("transition truth requires the exact current R81 five-band score spec")
    scored = score_response(values, label_list, frequency, score_spec)
    if (list(scored["margins"]) != [band["name"] for band in score_spec["bands"]]
            or len(scored["margins"]) != 5):
        raise ValueError("transition truth must preserve exactly five ordered hard-band margins")
    s21 = values[label_list.index("S21")].astype(np.float64)
    left = _side(s21, frequency, np.arange(26.0, 19.9, -0.5, dtype=np.float64))
    right = _side(s21, frequency, np.arange(30.0, 36.1, 0.5, dtype=np.float64))
    return {
        "hard_band_truth": {
            "margins_db": {key: float(value) for key, value in scored["margins"].items()},
            "worst_margin_db": float(scored["worst_margin"]),
            "worst_band": scored["worst_band"],
            "worst_frequencies_ghz": {
                key: float(value) for key, value in scored["worst_frequencies"].items()
            },
            "passed": dict(scored["passed"]),
            "all_passed": bool(scored["all_passed"]),
            "transition_enters_wm": False,
        },
        "transition": {"left": left, "right": right},
    }


def _canonical_profile(work_dir: Path, captured: Mapping[str, Mapping[str, Any]]) \
        -> tuple[dict[str, Any], dict[str, Any], Any]:
    cfg = ex.validate_exploration_config(_yaml_bytes(captured["config"], "config"))
    canonical = ex.validate_exploration_config(
        _yaml_bytes(captured["canonical_config"], "canonical_config"))
    state = _json_bytes(captured["state"], "state")
    measurement_marker = _json_bytes(captured["measurement_marker"], "measurement_marker")
    score_marker = _json_bytes(captured["score_spec_marker"], "score_spec_marker")
    profile = pb.load_profile_config(work_dir / "config.yaml")
    if (measurement_id(cfg["measurement"]) != measurement_id(canonical["measurement"])
            or score_spec_id(cfg["score_spec"]) != score_spec_id(canonical["score_spec"])
            or cfg.get("runtime") != canonical.get("runtime")
            or cfg["exploration"].get("mode") != "dual_margin5"
            or cfg["exploration"].get("training_protocol") != ft.PROTOCOL):
        raise ValueError("sidecar requires the exact current R81 masked-prior profile")
    measurement = cfg["measurement"]
    geometry = measurement.get("geometry", {})
    if (measurement.get("port") != "dual" or measurement.get("labels") != list(EXPECTED_LABELS)
            or geometry != {"geom": "p01", "pixel_count": 25, "diag_bridge_w": 0.075}):
        raise ValueError("sidecar requires the exact R81 dual geometry identity")
    _exact_axis(measurement["labels"], state.get("frequencies_ghz", ()))
    if (state.get("config_hash") != ex.content_id(cfg)
            or state.get("measurement_id") != measurement_id(cfg["measurement"])
            or state.get("score_spec_id") != score_spec_id(cfg["score_spec"])
            or measurement_id(measurement_marker) != state["measurement_id"]
            or score_spec_id(score_marker) != state["score_spec_id"]):
        raise ValueError("prepared R81 config/state/profile markers differ")
    return cfg, state, profile


def _binding_paths(work_dir: Path, batch: int, state: Mapping[str, Any]) -> dict[str, Path]:
    info = state.get("batches", {}).get(str(batch))
    if not isinstance(info, Mapping) or not info.get("feedback_audit"):
        raise ValueError("batch feedback is not completed and audited")
    return {
        "config": work_dir / "config.yaml",
        "state": work_dir / "state.json",
        "measurement": work_dir / "measurement.json",
        "score_spec": work_dir / "score_spec.json",
        "dataset_manifest": work_dir / "dataset_manifest.json",
        "input_manifest": ex._safe_input_path(work_dir, info["input_dir"]) / "manifest.json",
        "feedback_audit": ex._safe_input_path(work_dir, info["feedback_audit"]),
    }


def _validate_unique_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    ids = [row.get("id") for row in rows]
    patterns = [row.get("pattern_sha256") for row in rows]
    if (any(not isinstance(value, str) or not value for value in ids + patterns)
            or len(set(ids)) != len(ids) or len(set(patterns)) != len(patterns)):
        raise ValueError("measured transition rows require unique nonempty ids and physical patterns")


def _validate_rank_truth(row: Mapping[str, Any]) -> None:
    canonical = ex._load_yaml(CANONICAL_CONFIG)
    hashes = (row.get("pattern_sha256"), row.get("scoped_sample_sha256"),
              row.get("raw_response_f32le_sha256"))
    if (row.get("truth_kind") != TRUTH_KIND
            or any(not isinstance(value, str) or len(value) != 64
                   or any(char not in "0123456789abcdef" for char in value)
                   for value in hashes)
            or row.get("measurement_id") != measurement_id(canonical["measurement"])
            or row.get("score_spec_id") != score_spec_id(canonical["score_spec"])):
        raise ValueError("parent arithmetic accepts structurally marked measured R81 truth only")
    _exact_axis(row.get("labels", ()), row.get("frequencies_ghz", ()))


def _build_authenticated_payload(work_dir: str | Path, batch: int) -> dict[str, Any]:
    """Rebuild a sidecar payload from current bound feedback and sample bytes."""
    work = Path(work_dir).resolve()
    if isinstance(batch, bool) or int(batch) != batch or int(batch) < 1:
        raise ValueError("batch must be a positive integer")
    batch = int(batch)

    fixed_paths = {
        **IMPLEMENTATION_PATHS,
        "config": work / "config.yaml",
        "state": work / "state.json",
        "measurement_marker": work / "measurement.json",
        "score_spec_marker": work / "score_spec.json",
        "dataset_manifest": work / "dataset_manifest.json",
    }
    fixed = _capture_files(fixed_paths)
    # This immediate filesystem snapshot makes the captured byte set explicit
    # before any profile, state, measurement, or score parsing occurs.
    _assert_unchanged(fixed, "R81 initial binding changed after capture")
    cfg, state, profile = _canonical_profile(work, fixed)
    paths = _binding_paths(work, batch, state)
    dynamic = _capture_files({
        "input_manifest": paths["input_manifest"],
        "feedback_audit": paths["feedback_audit"],
    })
    _assert_unchanged(dynamic, "R81 feedback binding changed after capture")
    ft.validate_feedback_binding(work, batch, state)
    dataset_value = _json_bytes(fixed["dataset_manifest"], "dataset_manifest")
    if isinstance(dataset_value, dict):
        dataset_value = dataset_value.get("samples", dataset_value.get("entries"))
    if not isinstance(dataset_value, list):
        raise ValueError("bound dataset manifest must be a list or contain samples")
    all_rows = [dict(row) for row in dataset_value]
    _validate_unique_rows(all_rows)
    rows = sorted((row for row in all_rows if row.get("batch") == batch), key=lambda row: row["id"])
    info = state["batches"][str(batch)]
    if len(rows) != info.get("selected") or not rows:
        raise ValueError("authenticated feedback batch row count differs")
    _validate_unique_rows(rows)

    records = []
    sample_captures: dict[str, dict[str, Any]] = {}
    for row in rows:
        if (row.get("measurement_id") != state["measurement_id"]
                or row.get("score_spec_id") != state["score_spec_id"]
                or row.get("geometry_profile") != cfg["exploration"]["geometry_profile"]
                or row.get("response_labels") != list(EXPECTED_LABELS)):
            raise ValueError(f"measured row metadata differs from R81 profile: {row.get('id')}")
        _exact_axis(row["response_labels"], row.get("response_freqs_ghz", ()))
        sample = ex._safe_input_path(work, row["scoped_sample_file"])
        capture = _capture_files({row["id"]: sample})[row["id"]]
        sample_captures[row["id"]] = capture
        raw = bytes(capture["bytes"])
        current_hash = capture["sha256"]
        if current_hash != row.get("scoped_sample_sha256"):
            raise ValueError(f"scoped measured sample hash changed: {row['id']}")
        pattern, response = torch.load(io.BytesIO(raw), weights_only=True, map_location="cpu")
        pattern = torch.as_tensor(pattern, dtype=torch.float32).reshape(25, 25)
        response = torch.as_tensor(response, dtype=torch.float32)
        if (pb.validate_pattern(pattern, profile) != row.get("pattern_sha256")
                or ex.fingerprint(pattern, response) != row.get("sample_fingerprint")):
            raise ValueError(f"scoped measured sample identity changed: {row['id']}")
        diagnostic = diagnose_response(response, row["response_labels"],
                                       row["response_freqs_ghz"], cfg["score_spec"])
        response_bytes = np.asarray(response, dtype="<f4").tobytes(order="C")
        records.append({
            "truth_kind": TRUTH_KIND,
            "id": row["id"],
            "batch": batch,
            "pattern_sha256": row["pattern_sha256"],
            "lineage_id": row.get("lineage_id"),
            "canonical_group_id": row.get("canonical_group_id"),
            "sample_fingerprint": row["sample_fingerprint"],
            "scoped_sample_file": row["scoped_sample_file"],
            "scoped_sample_sha256": current_hash,
            "source_sample_sha256": row.get("source_sample_sha256"),
            "raw_response_f32le_sha256": _sha_bytes(response_bytes),
            "measurement_id": row["measurement_id"],
            "score_spec_id": row["score_spec_id"],
            "labels": list(row["response_labels"]),
            "frequencies_ghz": list(row["response_freqs_ghz"]),
            **diagnostic,
        })

    _assert_unchanged(fixed, "R81 fixed binding changed during sidecar validation")
    _assert_unchanged(dynamic, "R81 feedback binding changed during sidecar validation")
    _assert_unchanged(sample_captures, "R81 measured sample changed during sidecar validation")
    fixed_hashes = _capture_hashes(fixed)
    dynamic_hashes = _capture_hashes(dynamic)
    sample_hashes = _capture_hashes(sample_captures)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": SIDECAR_KIND,
        "status": "authenticated_measured_transition_preparation_only",
        "policy": {
            "id": POLICY_ID,
            "measured_parent_only": True,
            "candidate_sm_transition_output": False,
            "transition_enters_wm": False,
            "transition_hard_gate": False,
            "strict_monotonic_requirement": False,
            "default_max_wm_tradeoff_db": DEFAULT_MAX_WM_TRADEOFF_DB,
            "cap_is_noise_estimate": False,
        },
        "implementation_bindings": _implementation_bindings(fixed),
        "work_dir": str(work),
        "batch": batch,
        "measurement_id": state["measurement_id"],
        "score_spec_id": state["score_spec_id"],
        "records": records,
        "immutable_input_sha256": {"before": fixed_hashes, "after": fixed_hashes},
        "feedback_input_sha256": {"before": dynamic_hashes, "after": dynamic_hashes},
        "sample_sha256": {"before": sample_hashes, "after": sample_hashes},
        "integration": "sidecar_and_ranking_core_only_no_select_batch_consumer",
    }
    payload["content_id"] = ex.content_id(payload)
    return payload


def build_sidecar(work_dir: str | Path, batch: int, output: str | Path) -> Path:
    """Revalidate one imported feedback batch and create its transition sidecar."""
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite transition sidecar: {destination}")
    payload = _build_authenticated_payload(work_dir, batch)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
    return destination


def _midrank(values: Sequence[float], *, higher_is_better: bool) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("midrank inputs must be a nonempty finite vector")
    key = array if higher_is_better else -array
    order = np.argsort(key, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and key[order[stop]] == key[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    if len(array) == 1:
        ranks[:] = 0.5
    else:
        ranks /= len(array) - 1
    return ranks.astype(float).tolist()


def rank_measured_records(records: Sequence[Mapping[str, Any]], *,
                          max_wm_tradeoff_db: float = DEFAULT_MAX_WM_TRADEOFF_DB) -> list[dict[str, Any]]:
    """Pure arithmetic helper; it does not authenticate arbitrary record mappings."""
    cap = float(max_wm_tradeoff_db)
    if not np.isfinite(cap) or cap != DEFAULT_MAX_WM_TRADEOFF_DB:
        raise ValueError("r81_measured_transition_soft_v1 fixes max_wm_tradeoff_db at 0.1")
    rows = [dict(row) for row in records]
    if not rows:
        raise ValueError("measured parent ranking requires at least one record")
    _validate_unique_rows(rows)
    for row in rows:
        _validate_rank_truth(row)
    component_specs = (
        ("left_mean_relative_attenuation", "left", "mean_relative_attenuation_db", True),
        ("left_outward_rise_rms", "left", "outward_rise_rms_db", False),
        ("left_curvature_rms", "left", "curvature_rms_db", False),
        ("right_mean_relative_attenuation", "right", "mean_relative_attenuation_db", True),
        ("right_outward_rise_rms", "right", "outward_rise_rms_db", False),
        ("right_curvature_rms", "right", "curvature_rms_db", False),
    )
    values: dict[str, list[float]] = {name: [] for name, _side_name, _metric, _good in component_specs}
    wms = []
    for row in rows:
        try:
            wm = float(row["hard_band_truth"]["worst_margin_db"])
            transition = row["transition"]
            for name, side_name, metric, _good in component_specs:
                values[name].append(float(transition[side_name][metric]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("measured parent record lacks transition or WM diagnostics") from exc
        if not np.isfinite(wm):
            raise ValueError("measured parent WM must be finite")
        wms.append(wm)
    component_ranks = {
        name: _midrank(values[name], higher_is_better=good)
        for name, _side_name, _metric, good in component_specs
    }
    ranked = []
    for index, row in enumerate(rows):
        ranks = {name: component_ranks[name][index] for name in component_ranks}
        quality = float(np.mean(list(ranks.values())))
        ranked.append({
            "id": row["id"],
            "pattern_sha256": row["pattern_sha256"],
            "truth_kind": TRUTH_KIND,
            "measured_wm_db": wms[index],
            "transition_component_midranks": ranks,
            "transition_quality_0_to_1": quality,
            "max_wm_tradeoff_db": cap,
            "bounded_parent_score_db": float(wms[index] + cap * quality),
            "cap_is_noise_estimate": False,
            "transition_enters_wm": False,
            "input_authentication_performed": False,
        })
    return sorted(ranked, key=lambda row: (-row["bounded_parent_score_db"],
                                           -row["measured_wm_db"],
                                           -row["transition_quality_0_to_1"],
                                           row["pattern_sha256"], row["id"]))


def rank_sidecar(sidecar: str | Path, output: str | Path, *,
                 max_wm_tradeoff_db: float = DEFAULT_MAX_WM_TRADEOFF_DB) -> Path:
    source = Path(sidecar).resolve()
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite transition ranking: {destination}")
    try:
        raw = source.read_bytes()
    except (FileNotFoundError, IsADirectoryError) as exc:
        raise ValueError("measured transition sidecar is missing") from exc
    source_sha_before = _sha_bytes(raw)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid measured transition sidecar JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("measured transition sidecar must be an object")
    if value.get("kind") != SIDECAR_KIND:
        raise ValueError("not an R81 measured transition sidecar")
    claimed = value.get("content_id")
    unsigned = dict(value); unsigned.pop("content_id", None)
    if claimed != ex.content_id(unsigned):
        raise ValueError("transition sidecar content binding changed")
    expected = _build_authenticated_payload(value.get("work_dir"), value.get("batch"))
    if value != expected:
        raise ValueError("transition sidecar differs from current authenticated feedback replay")
    source_sha_after = _sha_file(source)
    if source_sha_before != source_sha_after:
        raise ValueError("transition sidecar changed during ranking validation")
    ranking = rank_measured_records(expected["records"],
                                    max_wm_tradeoff_db=max_wm_tradeoff_db)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": RANKING_KIND,
        "status": "measured_parent_ranking_core_preparation_only",
        "policy_id": POLICY_ID,
        "source_sidecar": str(source),
        "source_sidecar_sha256": {"before": source_sha_before, "after": source_sha_after},
        "source_sidecar_content_id": claimed,
        "source_feedback_revalidated": True,
        "max_wm_tradeoff_db": float(max_wm_tradeoff_db),
        "cap_is_noise_estimate": False,
        "hard_gate": False,
        "wm_modified": False,
        "select_batch_or_candidate_ranking_integrated": False,
        "ranking": ranking,
        "implementation_bindings": expected["implementation_bindings"],
    }
    payload["content_id"] = ex.content_id(payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
    return destination


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sidecar = sub.add_parser("sidecar", help="authenticate imported feedback and write diagnostics")
    sidecar.add_argument("--work-dir", required=True, type=Path)
    sidecar.add_argument("--batch", required=True, type=int)
    sidecar.add_argument("--output", required=True, type=Path)
    ranking = sub.add_parser("rank", help="write a bounded measured-parent ranking receipt")
    ranking.add_argument("--sidecar", required=True, type=Path)
    ranking.add_argument("--output", required=True, type=Path)
    ranking.add_argument("--max-wm-tradeoff-db", type=float,
                         default=DEFAULT_MAX_WM_TRADEOFF_DB)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    if args.command == "sidecar":
        path = build_sidecar(args.work_dir, args.batch, args.output)
    else:
        path = rank_sidecar(args.sidecar, args.output,
                            max_wm_tradeoff_db=args.max_wm_tradeoff_db)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
