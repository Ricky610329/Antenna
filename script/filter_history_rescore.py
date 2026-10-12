# -*- coding: utf-8 -*-
"""Re-score an explicit historical filter snapshot under the wide S21 rules.

This is an offline census utility.  It reads one immutable manifest plus its
raw ``(pattern, response)`` samples, validates the declared hashes, and keeps
coverage separate from performance.  In particular, a 24--32 GHz response can
pass the 26--30 GHz transmission band but can never qualify for the current
wide standard because neither stop band was observed.

The primary verdict is S21-only:

* 26--30 GHz: S21 >= -3 dB
* 16--20 GHz: S21 <= -20 dB
* 36--40 GHz: S21 <= -20 dB

The legacy S11/S22 match check over 26.5--29.5 GHz (<= -10 dB) is reported as
an independent reference and never changes the S21 verdict.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from antenna.measurement import (frequency_grid, measurement_id, score_response,
                                 score_spec_id, validate_measurement,
                                 validate_score_spec)
from script.exploration import pattern_sha256


LABELS = ("S11", "S21", "S22")
OBSERVED_SCOPE_GHZ = np.arange(16.0, 40.0 + 0.25, 0.5, dtype=np.float64)
S21_BANDS = (
    {"name": "s21_pass", "label": "S21", "start_ghz": 26.0,
     "stop_ghz": 30.0, "relation": "ge", "threshold": -3.0},
    {"name": "s21_low_stop", "label": "S21", "start_ghz": 16.0,
     "stop_ghz": 20.0, "relation": "le", "threshold": -20.0},
    {"name": "s21_high_stop", "label": "S21", "start_ghz": 36.0,
     "stop_ghz": 40.0, "relation": "le", "threshold": -20.0},
)
MATCH_BANDS = (
    {"name": "s11_match", "label": "S11", "start_ghz": 26.5,
     "stop_ghz": 29.5, "relation": "le", "threshold": -10.0},
    {"name": "s22_match", "label": "S22", "start_ghz": 26.5,
     "stop_ghz": 29.5, "relation": "le", "threshold": -10.0},
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64 and
            all(char in "0123456789abcdefABCDEF" for char in value))


def _load_manifest(path: Path) -> tuple[list[dict[str, Any]], str]:
    snapshot = path.read_bytes()
    try:
        value = json.loads(snapshot.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("historical manifest must be valid UTF-8 JSON") from exc
    if not isinstance(value, list) or not value:
        raise ValueError("historical manifest must be a non-empty list")
    if any(not isinstance(row, Mapping) for row in value):
        raise ValueError("every historical manifest row must be a mapping")
    rows = [dict(row) for row in value]
    ids = [row.get("id") for row in rows]
    if any(not isinstance(row_id, str) or not row_id.strip() for row_id in ids):
        raise ValueError("every historical row needs a non-empty id")
    if len(ids) != len(set(ids)):
        raise ValueError("historical manifest ids must be unique")
    return rows, _sha256(snapshot)


def _safe_sample(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("sample_file must be a non-empty relative path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"sample_file escapes sample root: {relative}") from exc
    if not path.is_file():
        raise ValueError(f"historical sample is missing: {relative}")
    return path


def _grid(start: float, stop: float) -> np.ndarray:
    return np.arange(start, stop + 0.25, 0.5, dtype=np.float64)


def _validate_geometry(value: Any, row_id: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{row_id}: geometry must be an explicit mapping")
    geometry = dict(value)
    try:
        bridge = float(geometry.get("diag_bridge_w"))
    except (TypeError, ValueError):
        bridge = float("nan")
    if (geometry.get("geom") != "p01" or geometry.get("pixel_count") != 25 or
            not np.isclose(bridge, 0.075, rtol=0.0, atol=1e-12)):
        raise ValueError(f"{row_id}: historical geometry is not p01/25/0.075 mm")
    return {"geom": "p01", "pixel_count": 25, "diag_bridge_w": 0.075}


def _covers(freqs: np.ndarray, start: float, stop: float) -> bool:
    """Require every point of the current 0.5 GHz band grid."""
    expected = _grid(start, stop)
    return all(np.count_nonzero(np.isclose(freqs, point, rtol=0.0, atol=1e-9)) == 1
               for point in expected)


def _score_bands(response: np.ndarray, freqs: np.ndarray,
                 bands: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], bool]:
    details: dict[str, Any] = {}
    all_covered = True
    for raw_band in bands:
        band = dict(raw_band)
        covered = _covers(freqs, band["start_ghz"], band["stop_ghz"])
        all_covered = all_covered and covered
        entry: dict[str, Any] = {
            "covered": covered,
            "start_ghz": band["start_ghz"],
            "stop_ghz": band["stop_ghz"],
            "relation": band["relation"],
            "threshold_db": band["threshold"],
        }
        if covered:
            in_band = ((freqs >= band["start_ghz"] - 1e-9) &
                       (freqs <= band["stop_ghz"] + 1e-9))
            scored = score_response(
                response[:, in_band], list(LABELS), freqs[in_band],
                {"name": "historical-wide-rescore", "bands": [band]},
            )
            name = band["name"]
            entry.update({
                "margin_db": float(scored["margins"][name]),
                "worst_frequency_ghz": float(scored["worst_frequencies"][name]),
                "passed": bool(scored["passed"][name]),
            })
        else:
            entry.update({"margin_db": None, "worst_frequency_ghz": None,
                          "passed": None})
        details[band["name"]] = entry
    return details, all_covered


def _validate_sample(row: Mapping[str, Any], root: Path) -> tuple[np.ndarray, np.ndarray, str, str, dict[str, Any]]:
    row_id = row["id"]
    if row.get("labels") != list(LABELS):
        raise ValueError(f"{row_id}: labels must be exactly [S11, S21, S22]")
    geometry = _validate_geometry(row.get("geometry"), row_id)
    if not isinstance(row.get("provenance"), Mapping) or not row["provenance"]:
        raise ValueError(f"{row_id}: provenance must be an explicit non-empty mapping")
    freqs = np.asarray(row.get("freqs_ghz"), dtype=np.float64)
    if (freqs.ndim != 1 or freqs.size == 0 or not np.isfinite(freqs).all() or
            (freqs.size > 1 and not np.all(np.diff(freqs) > 0))):
        raise ValueError(f"{row_id}: frequencies must be finite, one-dimensional, and increasing")
    if not _is_sha256(row.get("sample_sha256")):
        raise ValueError(f"{row_id}: sample_sha256 must be an explicit SHA-256")
    sample_path = _safe_sample(root, row.get("sample_file"))
    snapshot = sample_path.read_bytes()
    sample_hash = _sha256(snapshot)
    if sample_hash != row["sample_sha256"].lower():
        raise ValueError(f"{row_id}: historical sample hash mismatch")
    sample = torch.load(io.BytesIO(snapshot), weights_only=True, map_location="cpu")
    if not isinstance(sample, (tuple, list)) or len(sample) != 2:
        raise ValueError(f"{row_id}: sample must contain (pattern, response)")
    pattern = torch.as_tensor(sample[0]).detach().cpu().numpy()
    if (pattern.shape != (25, 25) or not np.isfinite(pattern).all() or
            not np.isin(pattern, [0, 1]).all()):
        raise ValueError(f"{row_id}: pattern must be finite binary 25x25")
    computed_pattern_hash = pattern_sha256(pattern.astype(np.uint8))
    if (not _is_sha256(row.get("pattern_sha256")) or
            computed_pattern_hash != row["pattern_sha256"].lower()):
        raise ValueError(f"{row_id}: pattern_sha256 mismatch")
    response = torch.as_tensor(sample[1]).detach().cpu().numpy()
    if response.shape != (len(LABELS), freqs.size) or not np.isfinite(response).all():
        raise ValueError(
            f"{row_id}: response must be finite with shape ({len(LABELS)}, {freqs.size})")
    response = response.astype(np.float32, copy=False)
    response_hash = _sha256(np.asarray(response, dtype="<f4").tobytes(order="C"))
    return freqs, response, sample_hash, response_hash, geometry


def rescore_history(manifest: str | Path, sample_root: str | Path) -> dict[str, Any]:
    """Validate and re-score all rows, retaining missing coverage as a failure."""
    manifest_path = Path(manifest).resolve()
    root = Path(sample_root).resolve()
    if not manifest_path.is_file() or not root.is_dir():
        raise ValueError("historical manifest and sample root must already exist")
    rows, manifest_hash = _load_manifest(manifest_path)

    records: list[dict[str, Any]] = []
    axes: Counter[tuple[int, float, float, str]] = Counter()
    stores: Counter[str] = Counter()
    total_sample_bytes = 0
    for row in rows:
        freqs, response, sample_hash, response_hash, geometry = _validate_sample(row, root)
        sample_path = _safe_sample(root, row["sample_file"])
        total_sample_bytes += sample_path.stat().st_size
        axis_hash = _sha256(np.asarray(freqs, dtype="<f8").tobytes(order="C"))
        axes[(int(freqs.size), float(freqs[0]), float(freqs[-1]), axis_hash)] += 1
        provenance = row.get("provenance")
        if isinstance(provenance, Mapping):
            store = provenance.get("source_store", provenance.get("store", "<unspecified>"))
        else:
            store = "<unspecified>"
        stores[str(store)] += 1

        s21, s21_bands_covered = _score_bands(response, freqs, S21_BANDS)
        match, match_covered = _score_bands(response, freqs, MATCH_BANDS)
        observed_scope_complete = _covers(freqs, 16.0, 40.0)
        # Only the three hard bands determine WM availability.  Extra/denser
        # observations are valid, and missing buffer points are not a fourth
        # hard constraint.
        s21_qualification_covered = bool(s21_bands_covered)
        s21_worst_margin = (min(item["margin_db"] for item in s21.values())
                            if s21_qualification_covered else None)
        s21_threshold_met = bool(
            s21_qualification_covered and
            all(item["passed"] is True for item in s21.values())
        )
        s21_positive = bool(s21_worst_margin is not None and s21_worst_margin > 0.0)
        passband_only_passed = s21["s21_pass"]["passed"] is True
        match_qualified = bool(
            match_covered and all(item["passed"] is True for item in match.values())
        )
        missing = [name for name, item in s21.items() if not item["covered"]]
        if not observed_scope_complete and not s21_bands_covered:
            missing.append("observed_scope_16_40")
        records.append({
            "id": row["id"],
            "sample_file": row["sample_file"],
            "sample_sha256": sample_hash,
            "pattern_sha256": row["pattern_sha256"].lower(),
            "raw_response_sha256": response_hash,
            "geometry": geometry,
            "lineage_id": row.get("lineage_id"),
            "canonical_group_id": row.get("canonical_group_id"),
            "provenance": dict(provenance) if isinstance(provenance, Mapping) else None,
            "frequency_axis": {
                "count": int(freqs.size), "start_ghz": float(freqs[0]),
                "stop_ghz": float(freqs[-1]), "sha256_float64_le": axis_hash,
                "observed_scope_16_40_complete": bool(observed_scope_complete),
            },
            "s21": {
                "qualification_covered": s21_qualification_covered,
                "qualified": s21_threshold_met,
                "threshold_met": s21_threshold_met,
                "worst_margin_db": s21_worst_margin,
                "positive_margin_observed": s21_positive,
                "independent_repeat_confirmed": False,
                "desired_positive_and_repeat_confirmed": False,
                "missing_coverage": missing,
                "passband_only_passed": passband_only_passed,
                "passband_margin_db": s21["s21_pass"]["margin_db"],
                "bands": s21,
            },
            "retained_match_reference": {
                "covered": bool(match_covered),
                "qualified": match_qualified,
                "bands": match,
                "affects_s21_qualification": False,
            },
        })

    # Keep every physical measurement instance.  Duplicate pattern metadata is
    # descriptive only: no row is removed and no duplicate is silently treated
    # as an independent confirmation run.
    pattern_counts = Counter(record["pattern_sha256"] for record in records)
    pattern_seen: Counter[str] = Counter()
    for record in records:
        digest = record["pattern_sha256"]
        pattern_seen[digest] += 1
        record["physical_pattern_instances"] = {
            "multiplicity": pattern_counts[digest],
            "instance_index": pattern_seen[digest],
            "deduplicated": False,
            "counts_as_independent_repeat": False,
        }

    def count(predicate) -> int:
        return sum(1 for record in records if predicate(record))

    summary = {
        "rows_validated": len(records),
        "full_observed_scope_16_40": count(
            lambda r: r["frequency_axis"]["observed_scope_16_40_complete"]),
        "s21_qualification_covered": count(lambda r: r["s21"]["qualification_covered"]),
        "s21_qualified": count(lambda r: r["s21"]["qualified"]),
        "s21_positive_margin_observed": count(
            lambda r: r["s21"]["positive_margin_observed"]),
        "s21_positive_and_repeat_confirmed": count(
            lambda r: r["s21"]["desired_positive_and_repeat_confirmed"]),
        "s21_passband_only_passed": count(lambda r: r["s21"]["passband_only_passed"]),
        "s21_low_stop_covered": count(lambda r: r["s21"]["bands"]["s21_low_stop"]["covered"]),
        "s21_high_stop_covered": count(lambda r: r["s21"]["bands"]["s21_high_stop"]["covered"]),
        "retained_match_covered": count(lambda r: r["retained_match_reference"]["covered"]),
        "retained_match_qualified": count(lambda r: r["retained_match_reference"]["qualified"]),
        "s21_and_retained_match_qualified": count(
            lambda r: r["s21"]["qualified"] and r["retained_match_reference"]["qualified"]),
        "total_sample_bytes": total_sample_bytes,
        "unique_physical_patterns": len(pattern_counts),
        "duplicate_measurement_instances": len(records) - len(pattern_counts),
    }
    axis_inventory = [
        {"count": count_value, "points": key[0], "start_ghz": key[1],
         "stop_ghz": key[2], "sha256_float64_le": key[3]}
        for key, count_value in sorted(axes.items(), key=lambda item: (-item[1], item[0]))
    ]
    source_store_inventory = [
        {"source_store": name, "count": number}
        for name, number in sorted(stores.items(), key=lambda item: (-item[1], item[0]))
    ]
    pending_stop = sorted((
        {
            "id": record["id"],
            "pattern_sha256": record["pattern_sha256"],
            "sample_file": record["sample_file"],
            "sample_sha256": record["sample_sha256"],
            "lineage_id": record["lineage_id"],
            "canonical_group_id": record["canonical_group_id"],
            "source_store": ((record["provenance"] or {}).get(
                "source_store", (record["provenance"] or {}).get("store"))),
            "passband_margin_db": record["s21"]["passband_margin_db"],
            "missing_coverage": record["s21"]["missing_coverage"],
        }
        for record in records
        if (record["s21"]["passband_margin_db"] is not None and
            record["s21"]["passband_margin_db"] > 0.0 and
            (not record["s21"]["bands"]["s21_low_stop"]["covered"] or
             not record["s21"]["bands"]["s21_high_stop"]["covered"]))
    ), key=lambda row: (-row["passband_margin_db"], row["id"]))
    summary["positive_passband_pending_stop_measurement"] = len(pending_stop)
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "role": "offline_historical_filter_rescore_not_current_truth",
        "primary_standard": {
            "basis": "S21_only",
            "observed_scope_ghz": [16.0, 40.0],
            "observed_step_ghz": 0.5,
            "bands": [dict(band) for band in S21_BANDS],
            "missing_stop_coverage_can_qualify": False,
            "threshold_met_rule": "worst_margin_db >= 0",
            "desired_result_rule": "worst_margin_db > 0 plus an independently identified repeat",
            "repeat_confirmation_inferred_from_duplicate_patterns": False,
        },
        "optional_reference": {
            "basis": "retained_S11_S22_match",
            "bands": [dict(band) for band in MATCH_BANDS],
            "affects_primary_standard": False,
        },
        "input_bindings": {
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_hash,
            "sample_root": str(root),
        },
        "summary": summary,
        "frequency_axis_inventory": axis_inventory,
        "source_store_inventory": source_store_inventory,
        "qualified_s21_ids": [r["id"] for r in records if r["s21"]["qualified"]],
        "positive_s21_ids": [r["id"] for r in records
                             if r["s21"]["positive_margin_observed"]],
        "positive_passband_pending_stop_measurement": pending_stop,
        "records": records,
    }


def rescore_completed_wide_store(store_dir: str | Path, *, source_method: str) -> dict[str, Any]:
    """Validate and assess one completed 16--40 GHz dual engineering store."""
    from script import profiled_batch as pb

    root = Path(store_dir).resolve()
    results = pb.validate_store(root, require_complete=True)
    measurement_snapshot = (root / "measurement.json").read_bytes()
    score_snapshot = (root / "score_spec.json").read_bytes()
    manifest_snapshot = (root / "manifest.json").read_bytes()
    measurement = validate_measurement(json.loads(measurement_snapshot.decode("utf-8")))
    score = validate_score_spec(json.loads(score_snapshot.decode("utf-8")), measurement=measurement)
    rows = json.loads(manifest_snapshot.decode("utf-8"))
    if (measurement["port"] != "dual" or measurement["labels"] != list(LABELS) or
            not np.array_equal(frequency_grid(measurement), OBSERVED_SCOPE_GHZ)):
        raise ValueError("wide completed store must be dual [S11,S21,S22] at 16..40/0.5 GHz")
    if not isinstance(rows, list) or {row["id"] for row in rows} != set(results):
        raise ValueError("wide completed store manifest/results identity mismatch")

    observations = []
    for row in rows:
        entry = results[row["id"]]
        path = _safe_sample(root, entry.get("sample_file"))
        snapshot = path.read_bytes()
        if _sha256(snapshot) != entry.get("sample_sha256"):
            raise ValueError(f"{row['id']}: completed wide sample hash mismatch")
        sample = torch.load(io.BytesIO(snapshot), weights_only=True, map_location="cpu")
        if not isinstance(sample, (tuple, list)) or len(sample) != 2:
            raise ValueError(f"{row['id']}: completed wide sample must contain (pattern, response)")
        pattern = torch.as_tensor(sample[0]).detach().cpu().numpy()
        response = torch.as_tensor(sample[1]).detach().cpu().numpy()
        if (pattern.shape != (25, 25) or response.shape != (3, 49) or
                not np.isfinite(pattern).all() or not np.isfinite(response).all()):
            raise ValueError(f"{row['id']}: completed wide sample has invalid shape or values")
        pattern_hash = pattern_sha256(pattern)
        if pattern_hash != row.get("pattern_sha256") or pattern_hash != entry.get("pattern_sha256"):
            raise ValueError(f"{row['id']}: completed wide pattern hash mismatch")
        freqs = np.asarray(entry.get("freqs"), dtype=np.float64)
        if not np.array_equal(freqs, OBSERVED_SCOPE_GHZ) or entry.get("labels") != list(LABELS):
            raise ValueError(f"{row['id']}: completed wide response identity mismatch")
        s21, covered = _score_bands(response.astype(np.float32, copy=False), freqs, S21_BANDS)
        match, match_covered = _score_bands(
            response.astype(np.float32, copy=False), freqs, MATCH_BANDS)
        worst_margin = min(item["margin_db"] for item in s21.values()) if covered else None
        threshold_met = bool(covered and worst_margin is not None and worst_margin >= 0.0)
        observations.append({
            "id": row["id"], "source_method": source_method,
            "pattern_sha256": pattern_hash, "sample_file": entry["sample_file"],
            "sample_sha256": entry["sample_sha256"],
            "raw_response_sha256": _sha256(
                np.asarray(response, dtype="<f4").tobytes(order="C")),
            "measurement_id": measurement_id(measurement),
            "score_spec_id": score_spec_id(score),
            "s21": {"qualification_covered": bool(covered),
                    "threshold_met": threshold_met, "qualified": threshold_met,
                    "worst_margin_db": worst_margin,
                    "positive_margin_observed": bool(worst_margin is not None and worst_margin > 0.0),
                    "independent_repeat_confirmed": False,
                    "desired_positive_and_repeat_confirmed": False,
                    "bands": s21},
            "retained_match_reference": {
                "covered": bool(match_covered),
                "qualified": bool(match_covered and all(v["passed"] is True
                                                         for v in match.values())),
                "bands": match, "affects_s21_qualification": False,
            },
        })
    return {
        "source_method": source_method,
        "store": str(root),
        "bindings": {
            "measurement_sha256": _sha256(measurement_snapshot),
            "score_spec_sha256": _sha256(score_snapshot),
            "manifest_sha256": _sha256(manifest_snapshot),
            "measurement_id": measurement_id(measurement),
            "score_spec_id": score_spec_id(score),
        },
        "observation_count": len(observations),
        "s21_threshold_met": sum(1 for row in observations if row["s21"]["threshold_met"]),
        "s21_positive_margin_observed": sum(
            1 for row in observations if row["s21"]["positive_margin_observed"]),
        "observations": observations,
    }


def attach_wide_assessment(report: dict[str, Any], stores: Sequence[tuple[str, str | Path]]) -> None:
    """Attach current wide observations without mixing them into historical counts."""
    assessed = [rescore_completed_wide_store(path, source_method=method)
                for method, path in stores]
    observations = [row for store in assessed for row in store["observations"]]
    multiplicity = Counter(row["pattern_sha256"] for row in observations)
    for row in observations:
        row["cross_store_pattern_multiplicity"] = multiplicity[row["pattern_sha256"]]
        row["counts_as_independent_repeat"] = False
    report["current_wide_engineering_assessment"] = {
        "role": "current_wide_engineering_observations_separate_from_historical_prior",
        "stores": assessed,
        "summary": {
            "observations": len(observations),
            "unique_physical_patterns": len(multiplicity),
            "s21_threshold_met": sum(1 for row in observations if row["s21"]["threshold_met"]),
            "s21_positive_margin_observed": sum(
                1 for row in observations if row["s21"]["positive_margin_observed"]),
        },
        "engineering_consistency_gate_assessed": False,
        "instrument_or_machine_confound_resolved": False,
        "mesh_observations_included": False,
        "full_engineering_release_allowed_claim": False,
    }


def wide_stores_from_worker_descriptor(path: str | Path) -> tuple[list[tuple[str, Path]], dict[str, Any]]:
    """Resolve the pinned private release as UTF-8 data, never as shell code."""
    descriptor_path = Path(path).resolve()
    descriptor_snapshot = descriptor_path.read_bytes()
    descriptor = json.loads(descriptor_snapshot.decode("utf-8"))
    if not isinstance(descriptor, Mapping) or not _is_sha256(descriptor.get("release_sha256")):
        raise ValueError("worker descriptor must pin release and release_sha256")
    release_path = Path(descriptor.get("release", ""))
    release_snapshot = release_path.read_bytes()
    if _sha256(release_snapshot) != descriptor["release_sha256"].lower():
        raise ValueError("worker descriptor release hash mismatch")
    release = json.loads(release_snapshot.decode("utf-8"))
    if release.get("stage") != "EngineeringOnly" or not release.get("dataset_root"):
        raise ValueError("worker descriptor must reference an EngineeringOnly release")
    dataset = Path(release["dataset_root"])
    stores = [("Fast", dataset / "dedust_r81smoke"),
              ("Discrete", dataset / "dedust_r81discrete")]
    binding = {
        "descriptor": str(descriptor_path),
        "descriptor_sha256": _sha256(descriptor_snapshot),
        "release": str(release_path),
        "release_sha256": _sha256(release_snapshot),
        "release_id": release.get("release_id"),
        "stage": release.get("stage"),
    }
    return stores, binding


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def _atomic_pending_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    columns = ("id", "pattern_sha256", "sample_file", "sample_sha256", "lineage_id",
               "canonical_group_id", "source_store", "passband_margin_db", "missing_coverage")
    lines = ["\t".join(columns)]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            if isinstance(value, list):
                value = ",".join(str(item) for item in value)
            text = "" if value is None else str(value)
            values.append(text.replace("\t", " ").replace("\r", " ").replace("\n", " "))
        lines.append("\t".join(values))
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--sample-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pending-tsv", type=Path)
    parser.add_argument("--pending-json", type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument(
        "--wide-store", action="append", default=[], metavar="METHOD=PATH",
        help="optional completed 16--40 GHz store; repeat for Fast/Discrete",
    )
    parser.add_argument(
        "--worker-descriptor", type=Path,
        help="optional UTF-8 descriptor resolving the pinned Fast6/Discrete2 release",
    )
    args = parser.parse_args(argv)
    report = rescore_history(args.manifest, args.sample_root)
    wide_stores = []
    for item in args.wide_store:
        if "=" not in item or not item.split("=", 1)[0]:
            parser.error("--wide-store must be METHOD=PATH")
        wide_stores.append(tuple(item.split("=", 1)))
    if args.worker_descriptor is not None:
        if wide_stores:
            parser.error("use either --worker-descriptor or --wide-store, not both")
        wide_stores, descriptor_binding = wide_stores_from_worker_descriptor(
            args.worker_descriptor)
    if wide_stores:
        attach_wide_assessment(report, wide_stores)
        if args.worker_descriptor is not None:
            report["current_wide_engineering_assessment"]["worker_descriptor_binding"] = (
                descriptor_binding)
    _atomic_json(args.output.resolve(), report)
    if args.pending_json is not None:
        _atomic_json(args.pending_json.resolve(), {
            "schema_version": 1,
            "role": "positive_26_30_passband_pending_both_stop_measurements",
            "manifest_sha256": report["input_bindings"]["manifest_sha256"],
            "count": len(report["positive_passband_pending_stop_measurement"]),
            "records": report["positive_passband_pending_stop_measurement"],
        })
    if args.pending_tsv is not None:
        _atomic_pending_tsv(
            args.pending_tsv.resolve(), report["positive_passband_pending_stop_measurement"])
    if args.summary_output is not None:
        summary_report = {
            key: report[key] for key in (
                "schema_version", "created_utc", "role", "primary_standard",
                "optional_reference", "input_bindings", "summary",
                "frequency_axis_inventory", "source_store_inventory")
        }
        if "current_wide_engineering_assessment" in report:
            summary_report["current_wide_engineering_assessment"] = report[
                "current_wide_engineering_assessment"]
        _atomic_json(args.summary_output.resolve(), summary_report)
    output_hash = _sha256(args.output.resolve().read_bytes())
    print(json.dumps({"output": str(args.output.resolve()), "output_sha256": output_hash,
                      "summary": report["summary"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
