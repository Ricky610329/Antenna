# -*- coding: utf-8 -*-
"""Prepare an audited, masked historical prior for the R81 filter SM.

The caller supplies one explicit manifest and its sample root.  This module does
not discover datasets, train a model, or run HFSS.  Historical 24--32 GHz
responses may supervise only the three R81 margin bands that they fully cover;
the two new stop-band targets remain masked and are never interpreted as a
historical worst margin.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from antenna.measurement import (frequency_grid, measurement_id, score_response,
                                 score_spec_id, validate_measurement,
                                 validate_score_spec)
from script.exploration import pattern_sha256


HISTORICAL_LABELS = ("S11", "S21", "S22")
HISTORICAL_FREQUENCIES_GHZ = np.arange(24.0, 32.0 + 0.25, 0.5, dtype=np.float64)
HISTORICAL_GEOMETRY = {"geom": "p01", "pixel_count": 25, "diag_bridge_w": 0.075}
R81_FREQUENCIES_GHZ = np.arange(16.0, 40.0 + 0.25, 0.5, dtype=np.float64)
R81_BAND_NAMES = (
    "s11_match", "s22_match", "s21_pass", "s21_low_stop", "s21_high_stop",
)
COVERED_BAND_NAMES = frozenset({"s11_match", "s22_match", "s21_pass"})
MASKED_BAND_NAMES = frozenset({"s21_low_stop", "s21_high_stop"})


@dataclass(frozen=True)
class FilterPrior:
    """Audited arrays and provenance for later masked-loss integration."""

    patterns: np.ndarray
    raw_responses: np.ndarray
    margins: np.ndarray
    margin_masks: np.ndarray
    band_names: tuple[str, ...]
    records: tuple[dict[str, Any], ...]
    exclusions: tuple[dict[str, Any], ...]
    audit: dict[str, Any]


def _read_manifest(path: Path) -> tuple[list[dict[str, Any]], str]:
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
    if any(not isinstance(value, str) or not value.strip() for value in ids):
        raise ValueError("every historical row needs a non-empty id")
    if len(ids) != len(set(ids)):
        raise ValueError("historical manifest ids must be unique")
    return rows, hashlib.sha256(snapshot).hexdigest()


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


def _is_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64 and
            all(char in "0123456789abcdefABCDEF" for char in value))


def _validate_current_profile(measurement: Mapping[str, Any],
                              score_spec: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    current = validate_measurement(measurement)
    score = validate_score_spec(score_spec, measurement=current)
    if current["port"] != "dual" or current["labels"] != list(HISTORICAL_LABELS):
        raise ValueError("R81 current profile must use dual labels [S11, S21, S22]")
    if current["geometry"] != HISTORICAL_GEOMETRY:
        raise ValueError("R81 current geometry must be exactly p01/25/0.075 mm")
    if not np.array_equal(frequency_grid(current), R81_FREQUENCIES_GHZ):
        raise ValueError("R81 current sweep must be exactly 16..40 GHz in 0.5 GHz steps")
    names = tuple(band["name"] for band in score["bands"])
    if len(names) != 5 or set(names) != set(R81_BAND_NAMES):
        raise ValueError(f"R81 score must contain exactly {list(R81_BAND_NAMES)}")

    historical_lo, historical_hi = HISTORICAL_FREQUENCIES_GHZ[[0, -1]]
    covered, masked = set(), set()
    for band in score["bands"]:
        lo, hi = band["start_ghz"], band["stop_ghz"]
        intersects = hi >= historical_lo and lo <= historical_hi
        fully_covered = lo >= historical_lo and hi <= historical_hi
        if intersects and not fully_covered:
            raise ValueError(f"score band {band['name']} only partially overlaps historical sweep")
        (covered if fully_covered else masked).add(band["name"])
    if covered != COVERED_BAND_NAMES or masked != MASKED_BAND_NAMES:
        raise ValueError("historical prior must cover exactly the three in-band margins")
    return current, score


def _validate_geometry(value: Any, row_id: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(HISTORICAL_GEOMETRY):
        raise ValueError(f"{row_id}: historical geometry must explicitly name geom/pixel_count/diag_bridge_w")
    geometry = dict(value)
    bridge = geometry.get("diag_bridge_w")
    try:
        valid_bridge = (not isinstance(bridge, bool) and
                        np.isclose(float(bridge), 0.075, rtol=0.0, atol=1e-12))
    except (TypeError, ValueError):
        valid_bridge = False
    if (geometry.get("geom") != "p01" or geometry.get("pixel_count") != 25 or
            not valid_bridge):
        raise ValueError(f"{row_id}: historical geometry is not p01/25/0.075 mm")
    return copy.deepcopy(HISTORICAL_GEOMETRY)


def _validate_pattern(value: Any, row_id: str) -> tuple[np.ndarray, str]:
    pattern = torch.as_tensor(value).detach().cpu().numpy()
    if (pattern.shape != (25, 25) or not np.isfinite(pattern).all() or
            not np.isin(pattern, [0, 1]).all()):
        raise ValueError(f"{row_id}: pattern must be finite binary 25x25")
    binary = pattern.astype(np.uint8)
    if not binary[:5, 10:15].all() or not binary[20:25, 10:15].all():
        raise ValueError(f"{row_id}: p01 pattern lacks a required 5x5 feed pad")
    return binary, pattern_sha256(binary)


def _holdout_sets(rows: Sequence[Mapping[str, Any]]) -> tuple[set[str], set[str]]:
    patterns, groups = set(), set()
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ValueError(f"current holdout row {index} must be a mapping")
        row = dict(raw)
        declared = row.get("pattern_sha256")
        if declared is not None and not _is_sha256(declared):
            raise ValueError(f"current holdout {index} has invalid pattern_sha256")
        if "pattern" in row:
            _pattern, computed = _validate_pattern(row["pattern"], f"current holdout {index}")
            if declared is not None and declared.lower() != computed:
                raise ValueError(f"current holdout {index} pattern hash mismatch")
            declared = computed
        if declared is not None:
            patterns.add(declared.lower())
        if row.get("lineage_id") is not None:
            groups.add(str(row["lineage_id"]))
        if row.get("canonical_group_id") is not None:
            groups.add(str(row["canonical_group_id"]))
        if declared is None and row.get("lineage_id") is None and row.get("canonical_group_id") is None:
            raise ValueError(f"current holdout {index} has no pattern or group identity")
    return patterns, groups


def _response_sha256(response: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(response, dtype="<f4").tobytes(order="C")).hexdigest()


def _masked_margins(response: np.ndarray, score: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    values, masks = [], []
    for band in score["bands"]:
        if band["name"] in COVERED_BAND_NAMES:
            result = score_response(response, list(HISTORICAL_LABELS),
                                    HISTORICAL_FREQUENCIES_GHZ,
                                    {"name": f"historical-{band['name']}", "bands": [band]})
            values.append(float(result["margins"][band["name"]]))
            masks.append(True)
        else:
            values.append(0.0)
            masks.append(False)
    return np.asarray(values, dtype=np.float32), np.asarray(masks, dtype=bool)


def prepare_filter_prior(
    measurement: Mapping[str, Any],
    score_spec: Mapping[str, Any],
    historical_manifest: str | Path,
    sample_root: str | Path,
    *,
    current_holdout: Sequence[Mapping[str, Any]] = (),
) -> FilterPrior:
    """Validate an explicit historical manifest and return masked training arrays.

    ``margins`` placeholders are zero wherever ``margin_masks`` is false.  A
    consumer must apply the mask; no full historical worst margin is produced.
    """
    current, score = _validate_current_profile(measurement, score_spec)
    manifest_path = Path(historical_manifest).resolve()
    root = Path(sample_root).resolve()
    if not manifest_path.is_file() or not root.is_dir():
        raise ValueError("historical manifest and sample root must already exist")
    manifest_rows, manifest_sha = _read_manifest(manifest_path)
    hold_patterns, hold_groups = _holdout_sets(current_holdout)

    validated = []
    for row in manifest_rows:
        row_id = row["id"]
        if row.get("labels") != list(HISTORICAL_LABELS):
            raise ValueError(f"{row_id}: historical labels must be exactly [S11, S21, S22]")
        freqs = np.asarray(row.get("freqs_ghz"), dtype=np.float64)
        if freqs.shape != (17,) or not np.array_equal(freqs, HISTORICAL_FREQUENCIES_GHZ):
            raise ValueError(f"{row_id}: historical frequencies must be exactly 24..32 GHz / 0.5")
        geometry = _validate_geometry(row.get("geometry"), row_id)
        if not isinstance(row.get("lineage_id"), str) or not row["lineage_id"]:
            raise ValueError(f"{row_id}: lineage_id must be explicit")
        if not isinstance(row.get("provenance"), Mapping) or not row["provenance"]:
            raise ValueError(f"{row_id}: provenance must be an explicit non-empty mapping")
        if not _is_sha256(row.get("sample_sha256")):
            raise ValueError(f"{row_id}: sample_sha256 must be explicit SHA-256")
        sample_path = _safe_sample(root, row.get("sample_file"))
        sample_snapshot = sample_path.read_bytes()
        actual_sample_sha = hashlib.sha256(sample_snapshot).hexdigest()
        if actual_sample_sha != row["sample_sha256"].lower():
            raise ValueError(f"{row_id}: historical sample hash mismatch")
        sample = torch.load(io.BytesIO(sample_snapshot), weights_only=True, map_location="cpu")
        if not isinstance(sample, (tuple, list)) or len(sample) != 2:
            raise ValueError(f"{row_id}: sample must contain (pattern, response)")
        pattern, pattern_hash = _validate_pattern(sample[0], row_id)
        if not _is_sha256(row.get("pattern_sha256")) or row["pattern_sha256"].lower() != pattern_hash:
            raise ValueError(f"{row_id}: pattern_sha256 mismatch")
        response = torch.as_tensor(sample[1]).detach().cpu().numpy()
        if response.shape != (3, 17) or not np.isfinite(response).all():
            raise ValueError(f"{row_id}: response must be finite 3x17")
        response = response.astype(np.float32, copy=False)
        margins, masks = _masked_margins(response, score)
        validated.append((row, geometry, pattern, pattern_hash, response,
                          actual_sample_sha, margins, masks))

    # A group-level holdout exclusion also forbids every physical duplicate of
    # that row, even if another manifest entry gives the same bits a new alias.
    group_forbidden_patterns = {
        item[3] for item in validated
        if (item[0]["lineage_id"] in hold_groups or
            (item[0].get("canonical_group_id") is not None and
             str(item[0]["canonical_group_id"]) in hold_groups))
    }
    # Identity aliases can span several patterns and family names. Exclude the
    # complete connected component, not only the first physical duplicate.
    pattern_groups, group_patterns = defaultdict(set), defaultdict(set)
    for row, _geometry, _pattern, digest, *_rest in validated:
        groups = {row["lineage_id"]}
        if row.get("canonical_group_id") is not None:
            groups.add(str(row["canonical_group_id"]))
        pattern_groups[digest].update(groups)
        for group in groups:
            group_patterns[group].add(digest)
    blocked_patterns, blocked_groups = set(hold_patterns), set(hold_groups)
    pending = deque([("pattern", p) for p in blocked_patterns] +
                    [("group", g) for g in blocked_groups])
    while pending:
        kind, identity = pending.popleft()
        neighbors = pattern_groups[identity] if kind == "pattern" else group_patterns[identity]
        blocked = blocked_groups if kind == "pattern" else blocked_patterns
        next_kind = "group" if kind == "pattern" else "pattern"
        for neighbor in neighbors - blocked:
            blocked.add(neighbor)
            pending.append((next_kind, neighbor))
    kept, exclusions, seen_patterns = [], [], set()
    for row, geometry, pattern, pattern_hash, response, sample_sha, margins, masks in validated:
        reasons = []
        if pattern_hash in hold_patterns:
            reasons.append("current_holdout_pattern")
        if row["lineage_id"] in hold_groups:
            reasons.append("current_holdout_lineage")
        canonical = row.get("canonical_group_id")
        if canonical is not None and str(canonical) in hold_groups:
            reasons.append("current_holdout_canonical_group")
        if pattern_hash in group_forbidden_patterns and not reasons:
            reasons.append("current_holdout_physical_alias")
        if pattern_hash in blocked_patterns and not reasons:
            reasons.append("current_holdout_alias_component")
        if pattern_hash in seen_patterns:
            reasons.append("duplicate_physical_pattern")
        if reasons:
            exclusions.append({"id": row["id"], "pattern_sha256": pattern_hash,
                               "reasons": reasons, "sample_sha256": sample_sha})
            continue
        seen_patterns.add(pattern_hash)
        record = {
            "id": row["id"], "lineage_id": row["lineage_id"],
            "canonical_group_id": row.get("canonical_group_id"),
            "pattern_sha256": pattern_hash, "source_sample_sha256": sample_sha,
            "raw_response_sha256": _response_sha256(response),
            "sample_file": row["sample_file"], "labels": list(HISTORICAL_LABELS),
            "freqs_ghz": HISTORICAL_FREQUENCIES_GHZ.tolist(), "geometry": geometry,
            "provenance": copy.deepcopy(dict(row["provenance"])),
            "role": "historical_prior_not_current_truth",
            "full_current_worst_margin_available": False,
            "margin_names": [band["name"] for band in score["bands"]],
            "margin_mask": masks.tolist(),
        }
        kept.append((record, pattern, response, margins, masks))

    if not kept:
        raise ValueError("historical prior is empty after holdout exclusion and deduplication")
    band_names = tuple(band["name"] for band in score["bands"])
    audit = {
        "schema_version": 1,
        "role": "historical_prior_not_current_truth",
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "current_measurement_id": measurement_id(current),
        "current_score_spec_id": score_spec_id(score),
        "historical_rows_validated": len(validated),
        "records_kept": len(kept),
        "records_excluded": len(exclusions),
        "band_names": list(band_names),
        "covered_bands": [name for name in band_names if name in COVERED_BAND_NAMES],
        "masked_bands": [name for name in band_names if name in MASKED_BAND_NAMES],
        "full_current_worst_margin_available": False,
    }
    return FilterPrior(
        patterns=np.stack([item[1] for item in kept]).astype(np.float32),
        raw_responses=np.stack([item[2] for item in kept]).astype(np.float32),
        margins=np.stack([item[3] for item in kept]).astype(np.float32),
        margin_masks=np.stack([item[4] for item in kept]).astype(bool),
        band_names=band_names,
        records=tuple(item[0] for item in kept),
        exclusions=tuple(exclusions),
        audit=audit,
    )
