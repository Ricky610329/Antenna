# -*- coding: utf-8 -*-
"""Historical geometry/field-symmetry census from the pattern-browser snapshot.

This script deliberately treats the browser curves as an id-aligned cache.  The
incremental producer can carry an old curve forward by bare id, so ``meta.store``
is retained as the metric store candidate rather than asserted as curve origin.
Only small JSON metadata files are read from the optional backup root.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO / "application" / "pattern_browser" / "data"
DEFAULT_BACKUP = Path(r"C:\Users\ricky\antenna_nas_backup\dataset")
DEFAULT_OUT = REPO / "docs" / "log" / "assets" / "analysis-18"
FREQ_GHZ = np.linspace(24.0, 32.0, 17)
SOLVER_KEYS = ("max_delta_s", "max_passes", "min_passes", "min_converged", "timeout")


def _finite_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def pattern_geometry(pattern):
    """Return left/right mirror mismatch and simple geometry descriptors.

    Mirror A is the mean XOR over the 25x12 unique left/right pixel pairs; the
    centre column is not compared with itself.
    """
    p = np.asarray(pattern, dtype=bool).reshape(25, 25)
    mismatch = float(np.mean(np.logical_xor(p[:, :12], p[:, :12:-1])))
    metal = float(np.mean(p))
    upper = float(np.mean(p[:12]))
    lower = float(np.mean(p[13:]))
    return {
        "geometry_mismatch_fraction": mismatch,
        "symmetry_class": ("exact" if mismatch == 0.0 else
                           "near" if mismatch <= 0.1 else "asymmetric"),
        "metal_fraction": metal,
        "upper_metal_fraction": upper,
        "lower_metal_fraction": lower,
        "upper_lower_metal_imbalance": upper - lower,
    }


def _paired_values(theta, values, limit):
    theta = np.asarray(theta, dtype=float).reshape(-1)
    values = np.asarray(values, dtype=float).reshape(-1)
    if theta.size != values.size:
        raise ValueError("theta and curve length differ")
    lookup = {round(float(t), 8): i for i, t in enumerate(theta) if np.isfinite(t)}
    left, right = [], []
    for i, t in enumerate(theta):
        if not np.isfinite(t) or t <= 0 or t > limit:
            continue
        j = lookup.get(round(float(-t), 8))
        if j is not None and np.isfinite(values[i]) and np.isfinite(values[j]):
            left.append(values[j])
            right.append(values[i])
    return np.asarray(left), np.asarray(right)


def _db_to_power(db):
    db = np.asarray(db, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        power = np.power(10.0, db / 10.0)
    power[~np.isfinite(power)] = np.nan
    return power


def radial_profile(theta, curve_db):
    """Compute mirror residuals, front-hemisphere power centroid and F/B balance.

    Mirror power residual is ``sum |P(+t)-P(-t)| / sum(P(+t)+P(-t))``.
    The centroid uses the finite peak-minus-3 dB main-lobe samples within the
    front hemisphere (±90 degrees), avoiding a single-bin argmax conclusion.
    """
    theta = np.asarray(theta, dtype=float).reshape(-1)
    curve = np.asarray(curve_db, dtype=float).reshape(-1)
    if theta.size != curve.size:
        raise ValueError("theta and curve length differ")
    out = {}
    for limit in (45, 90):
        neg, pos = _paired_values(theta, curve, limit)
        if neg.size:
            pn, pp = _db_to_power(neg), _db_to_power(pos)
            valid = np.isfinite(pn) & np.isfinite(pp)
            denom = float(np.sum(pn[valid] + pp[valid])) if np.any(valid) else 0.0
            out[f"mirror_power_{limit}"] = (float(np.sum(np.abs(pp[valid] - pn[valid])) / denom)
                                              if denom > 0 else math.nan)
            out[f"mirror_db_mae_{limit}"] = (float(np.mean(np.abs(pos - neg)))
                                               if neg.size else math.nan)
            out[f"mirror_pairs_{limit}"] = int(neg.size)
        else:
            out[f"mirror_power_{limit}"] = math.nan
            out[f"mirror_db_mae_{limit}"] = math.nan
            out[f"mirror_pairs_{limit}"] = 0

    power = _db_to_power(curve)
    valid = np.isfinite(theta) & np.isfinite(power)
    front = valid & (np.abs(theta) <= 90.0)
    back = valid & (np.abs(theta) > 90.0)
    peak = float(np.max(curve[front])) if np.any(front) else math.nan
    main = front & (curve >= peak - 3.0) if np.isfinite(peak) else np.zeros(theta.shape, bool)
    sm = float(np.sum(power[main])) if np.any(main) else 0.0
    if sm > 0:
        out["power_centroid_deg"] = float(np.sum(theta[main] * power[main]) / sm)
    else:
        out["power_centroid_deg"] = math.nan
    # Compare hemisphere mean powers, so a flat curve is exactly balanced even
    # when an endpoint is duplicated in the -180..180 sampling convention.
    mf = float(np.mean(power[front])) if np.any(front) else math.nan
    mb = float(np.mean(power[back])) if np.any(back) else math.nan
    denom = mf + mb
    out["front_back_imbalance"] = ((mf - mb) / denom
                                    if np.isfinite(denom) and denom > 0 else math.nan)
    out["front_back_db"] = (10.0 * math.log10(mf / mb)
                            if np.isfinite(mf) and np.isfinite(mb) and mf > 0 and mb > 0
                            else math.nan)
    finite_curve = np.where(np.isfinite(curve), curve, -np.inf)
    out["argmax_deg_diagnostic"] = (float(theta[int(np.argmax(finite_curve))])
                                     if np.any(np.isfinite(curve)) else math.nan)
    return out


def radial_profiles_batch(theta, curves_db):
    """Vectorized counterpart of :func:`radial_profile` for the full cache."""
    theta = np.asarray(theta, dtype=np.float32).reshape(-1)
    curves = np.asarray(curves_db, dtype=np.float32)
    if curves.ndim != 2 or curves.shape[1] != theta.size:
        raise ValueError("curves must have shape [N, len(theta)]")
    out = {}
    lookup = {round(float(t), 8): i for i, t in enumerate(theta) if np.isfinite(t)}
    for limit in (45, 90):
        pos_idx, neg_idx = [], []
        for i, t in enumerate(theta):
            if np.isfinite(t) and 0 < t <= limit:
                j = lookup.get(round(float(-t), 8))
                if j is not None:
                    pos_idx.append(i)
                    neg_idx.append(j)
        pos, neg = curves[:, pos_idx], curves[:, neg_idx]
        valid = np.isfinite(pos) & np.isfinite(neg)
        with np.errstate(over="ignore", invalid="ignore"):
            pp, pn = np.power(10.0, pos / 10.0), np.power(10.0, neg / 10.0)
        valid &= np.isfinite(pp) & np.isfinite(pn)
        pp = np.where(valid, pp, 0.0)
        pn = np.where(valid, pn, 0.0)
        denom = np.sum(pp + pn, axis=1, dtype=np.float64)
        numer = np.sum(np.abs(pp - pn), axis=1, dtype=np.float64)
        out[f"mirror_power_{limit}"] = np.divide(
            numer, denom, out=np.full(curves.shape[0], np.nan), where=denom > 0)
        db_numer = np.sum(np.where(valid, np.abs(pos - neg), 0.0), axis=1, dtype=np.float64)
        counts = np.sum(valid, axis=1)
        out[f"mirror_db_mae_{limit}"] = np.divide(
            db_numer, counts, out=np.full(curves.shape[0], np.nan), where=counts > 0)
        out[f"mirror_pairs_{limit}"] = counts

    with np.errstate(over="ignore", invalid="ignore"):
        power = np.power(10.0, curves / 10.0)
    power[~np.isfinite(power)] = np.nan
    front = np.abs(theta) <= 90.0
    back = np.abs(theta) > 90.0
    fp, bp = power[:, front], power[:, back]
    sf = np.nansum(fp, axis=1, dtype=np.float64)
    front_curves = curves[:, front]
    safe_front = np.where(np.isfinite(front_curves), front_curves, -np.inf)
    peak = np.max(safe_front, axis=1)
    main = np.isfinite(front_curves) & (front_curves >= peak[:, None] - 3.0)
    main_power = np.where(main, fp, 0.0)
    sm = np.sum(main_power, axis=1, dtype=np.float64)
    out["power_centroid_deg"] = np.divide(
        np.sum(main_power * theta[front], axis=1, dtype=np.float64), sm,
        out=np.full(curves.shape[0], np.nan), where=sm > 0)
    nf, nb = np.sum(np.isfinite(fp), axis=1), np.sum(np.isfinite(bp), axis=1)
    mf = np.divide(sf, nf, out=np.full(curves.shape[0], np.nan), where=nf > 0)
    mb = np.divide(np.nansum(bp, axis=1, dtype=np.float64), nb,
                   out=np.full(curves.shape[0], np.nan), where=nb > 0)
    denom = mf + mb
    out["front_back_imbalance"] = np.divide(
        mf - mb, denom, out=np.full(curves.shape[0], np.nan),
        where=np.isfinite(denom) & (denom > 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        out["front_back_db"] = 10.0 * np.log10(mf / mb)
    finite = np.isfinite(curves)
    arg = np.argmax(np.where(finite, curves, -np.inf), axis=1)
    out["argmax_deg_diagnostic"] = theta[arg].astype(float)
    out["argmax_deg_diagnostic"][~np.any(finite, axis=1)] = np.nan
    return out


def classify_measurement_profile(setup, manifest_row):
    """Classify the metric-store candidate without claiming curve provenance."""
    setup = setup if isinstance(setup, dict) else {}
    manifest_row = manifest_row if isinstance(manifest_row, dict) else {}
    kind = manifest_row.get("kind")
    bridge = setup.get("diag_bridge_w", manifest_row.get("diag_bridge_w"))
    pixels = setup.get("pixel_count", manifest_row.get("pixel_count"))
    slot = setup.get("slot_spec", manifest_row.get("slot_spec"))
    solver = {k: setup[k] for k in SOLVER_KEYS if k in setup}
    parts = []
    if kind in {"meshconv", "diagbridge", "slotw"}:
        parts.append(str(kind))
    if bridge is not None:
        parts.append(f"bridge_w={bridge}")
    if pixels is not None:
        parts.append(f"pixels={pixels}")
    if slot is not None:
        parts.append("slot_spec")
    if solver:
        digest = hashlib.sha256(json.dumps(solver, sort_keys=True).encode()).hexdigest()[:8]
        parts.append(f"solver={digest}")
    if not parts:
        parts.append("standard_or_legacy_unspecified")
    profile = {
        "kind": kind,
        "diag_bridge_w": bridge,
        "pixel_count": pixels,
        "slot_spec": slot,
        "solver": solver,
        "port": manifest_row.get("port"),
    }
    canonical = json.dumps(profile, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "+".join(parts), hashlib.sha256(canonical.encode("utf-8")).hexdigest(), profile


def _read_json(path, audit):
    try:
        raw = path.read_bytes()
        audit["metadata_bytes_read"] += len(raw)
        return json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def load_candidate_profiles(meta, backup_root):
    """Read only setup/manifest JSON for stores already named by the cache."""
    root = Path(backup_root)
    wanted = defaultdict(set)
    for row in meta:
        if row.get("store"):
            wanted[str(row["store"])].add(str(row["id"]))
    audit = {"stores_requested": len(wanted), "metadata_files_read": 0,
             "metadata_bytes_read": 0, "stores_with_any_metadata": 0}
    profiles = {}
    for store, ids in sorted(wanted.items()):
        input_dir = root / f"{store}_input"
        store_dir = root / store
        setup = None
        setup_sources = []
        for path in (input_dir / "hfss_setup.json", store_dir / "hfss_setup.json"):
            if path.is_file():
                value = _read_json(path, audit)
                audit["metadata_files_read"] += 1
                if isinstance(value, dict):
                    setup_sources.append(str(path.relative_to(root)))
                    if setup is None:
                        setup = value
        manifest = None
        manifest_path = input_dir / "manifest.json"
        if manifest_path.is_file():
            manifest = _read_json(manifest_path, audit)
            audit["metadata_files_read"] += 1
        rows = {}
        if isinstance(manifest, list):
            for entry in manifest:
                if isinstance(entry, dict) and str(entry.get("id")) in ids:
                    rows.setdefault(str(entry["id"]), entry)
        if setup is not None or manifest is not None:
            audit["stores_with_any_metadata"] += 1
        for pid in ids:
            mrow = rows.get(pid)
            stratum, digest, profile = classify_measurement_profile(setup, mrow)
            status = ("resolved_candidate" if setup is not None or mrow is not None
                      else "legacy_unresolved")
            profiles[(store, pid)] = {
                "measurement_stratum": stratum,
                "measurement_profile_sha256": digest,
                "measurement_profile_status": status,
                "measurement_profile": profile,
                "measurement_profile_sources": setup_sources +
                    ([str(manifest_path.relative_to(root))] if mrow is not None else []),
            }
    return profiles, audit


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _cache_provenance(data_dir, paths, curve_state):
    files = {}
    for path in paths:
        st = path.stat()
        files[path.name] = {
            "bytes": st.st_size,
            "mtime_local": datetime.fromtimestamp(st.st_mtime).astimezone().isoformat(),
            "sha256": _sha256_file(path),
        }
    vals = [float(v) for v in curve_state.values() if isinstance(v, (int, float))]
    return {
        "data_dir": str(Path(data_dir).resolve()),
        "files": files,
        "curve_state_store_count": len(curve_state),
        "curve_state_snapshot_range_local": {
            "min": datetime.fromtimestamp(min(vals)).astimezone().isoformat() if vals else None,
            "max": datetime.fromtimestamp(max(vals)).astimezone().isoformat() if vals else None,
        },
        "curve_store_provenance": "unresolved_incremental_bare_id_carry_forward",
        "interpretation": "2026-08-10 browser snapshot; not asserted complete or current history",
    }


def _response_metrics(resp):
    resp = np.asarray(resp, dtype=float).reshape(2, 17)
    s11, gain = resp
    i28 = int(np.argmin(np.abs(FREQ_GHZ - 28.0)))
    band = (FREQ_GHZ >= 26.5) & (FREQ_GHZ <= 29.5)
    def safe(fun, x):
        x = x[np.isfinite(x)]
        return float(fun(x)) if x.size else math.nan
    return {
        "s11_db_28": _finite_float(s11[i28]),
        "gain_db_28": _finite_float(gain[i28]),
        "s11_db_band_mean": safe(np.mean, s11[band]),
        "s11_db_band_max": safe(np.max, s11[band]),
        "gain_db_band_mean": safe(np.mean, gain[band]),
        "gain_db_band_min": safe(np.min, gain[band]),
    }


def _json_number(value):
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def _json_ready(value):
    """Recursively replace non-finite numpy/Python scalars for strict JSON."""
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return _json_number(value)


PROFILE_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
PROFILE_SCALAR_KEYS = (
    "geometry_mismatch_fraction", "metal_fraction", "upper_metal_fraction",
    "lower_metal_fraction", "upper_lower_metal_imbalance", "s11_db_28",
    "gain_db_28", "s11_db_band_mean", "s11_db_band_max",
    "gain_db_band_mean", "gain_db_band_min", "s11_band_margin_db",
    "gain_band_margin_db",
)
RADIAL_SCALAR_KEYS = (
    "mirror_power_45", "mirror_power_90", "mirror_db_mae_45",
    "mirror_db_mae_90", "mirror_pairs_45", "mirror_pairs_90",
    "power_centroid_deg", "front_back_imbalance", "front_back_db",
    "argmax_deg_diagnostic",
)


def _profile_response_metrics(response, freqs):
    """Descriptive 28 GHz and 26.5--29.5 GHz values for a named profile."""
    response = np.asarray(response, dtype=float)
    freqs = np.asarray(freqs, dtype=float).reshape(-1)
    if response.shape != (2, freqs.size):
        raise ValueError("named-profile response must have shape [2, len(freqs)]")
    band = (freqs >= 26.5) & (freqs <= 29.5)
    if not np.any(band) or not np.any(np.isclose(freqs, 28.0)):
        raise ValueError("named-profile response lacks the 26.5--29.5 GHz band or 28 GHz")
    i28 = int(np.flatnonzero(np.isclose(freqs, 28.0))[0])
    s11, gain = response

    def safe(fun, values):
        values = values[np.isfinite(values)]
        return float(fun(values)) if values.size else math.nan

    s11_max = safe(np.max, s11[band])
    gain_min = safe(np.min, gain[band])
    return {
        "s11_db_28": _finite_float(s11[i28]),
        "gain_db_28": _finite_float(gain[i28]),
        "s11_db_band_mean": safe(np.mean, s11[band]),
        "s11_db_band_max": s11_max,
        "gain_db_band_mean": safe(np.mean, gain[band]),
        "gain_db_band_min": gain_min,
        # These fixed thresholds are descriptive coordinates used by the
        # factory figures.  No row is accepted or rejected using them.
        "s11_band_margin_db": -10.0 - s11_max,
        "gain_band_margin_db": gain_min - 4.0,
    }


def _quantile_summary(rows, keys):
    summary = {}
    for key in keys:
        values = np.asarray([row.get(key, math.nan) for row in rows], dtype=float)
        values = values[np.isfinite(values)]
        quantiles = np.quantile(values, PROFILE_QUANTILES) if values.size else []
        summary[key] = {
            "finite_count": int(values.size),
            **{f"q{int(q * 100):02d}": float(value)
               for q, value in zip(PROFILE_QUANTILES, quantiles)},
        }
    return summary


def _profile_source_provenance(root, entry_hashes):
    metadata = {}
    for name in ("measurement.json", "score_spec.json", "manifest.json",
                 "results.json", "source_bindings.json", "config.yaml"):
        path = root / name
        if path.is_file():
            metadata[name] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
    canonical = json.dumps(entry_hashes, sort_keys=True, separators=(",", ":"))
    return {
        "path": str(root),
        "metadata_files": metadata,
        "validated_artifact_count": len(entry_hashes),
        "validated_artifact_receipt_sha256": hashlib.sha256(
            canonical.encode("utf-8")).hexdigest(),
    }


def _write_profile_outputs(summary, arrays, out_json, out_npz):
    out_json, out_npz = Path(out_json), Path(out_npz)
    if out_json.resolve() == out_npz.resolve():
        raise ValueError("profile JSON and NPZ outputs must be different files")
    if out_json.exists() or out_npz.exists():
        raise ValueError("profile analysis outputs must be fresh; refusing overwrite")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    json_temp = npz_temp = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="wb", dir=out_npz.parent, prefix=out_npz.name + ".",
                suffix=".tmp", delete=False) as stream:
            npz_temp = Path(stream.name)
            np.savez_compressed(stream, **arrays)
        summary["artifacts"] = {"npz": {
            "path": str(out_npz.resolve()),
            "sha256": _sha256_file(npz_temp),
            "keys": sorted(arrays),
        }}
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n", dir=out_json.parent,
                prefix=out_json.name + ".", suffix=".tmp", delete=False) as stream:
            json_temp = Path(stream.name)
            json.dump(_json_ready(summary), stream, ensure_ascii=False, indent=2,
                      allow_nan=False)
            stream.write("\n")
        os.replace(npz_temp, out_npz)
        npz_temp = None
        os.replace(json_temp, out_json)
        json_temp = None
    finally:
        for path in (json_temp, npz_temp):
            if path is not None:
                path.unlink(missing_ok=True)


def analyze_profile_stores(stores, out_json, out_npz):
    """Validate and describe explicit complete named-profile frozen stores.

    Every input is replayed with :func:`script.profiled_batch.validate_store`,
    which verifies the measurement snapshot, physical bridge geometry, sample
    and radiation hashes.  Rows are deduplicated only by
    ``(measurement_id, pattern_sha256)``; non-repeat observations win before
    stable caller/store order.  The function never discovers stores, runs
    HFSS, filters on performance, or trains a model.
    """
    from antenna.measurement import measurement_id, score_spec_id
    from script import profiled_batch as pb
    import torch

    roots = [Path(value).resolve() for value in stores]
    if not roots:
        raise ValueError("at least one explicit frozen store is required")
    candidates = []
    provenance = []
    common_measurement = common_score = None
    common_mid = common_sid = None
    for store_index, root in enumerate(roots):
        results = pb.validate_store(root, require_complete=True)
        measurement = pb.read_json(root / "measurement.json")
        score = pb.read_json(root / "score_spec.json")
        mid, sid = measurement_id(measurement), score_spec_id(score)
        if measurement.get("port") != "single" or measurement.get("labels") != ["S11", "Gain"]:
            raise ValueError("profile symmetry analysis requires a single-port S11/Gain measurement")
        if common_mid is None:
            common_measurement, common_score = measurement, score
            common_mid, common_sid = mid, sid
        elif (mid, sid) != (common_mid, common_sid):
            raise ValueError("profile stores do not share measurement_id and score_spec_id")
        manifest = pb.read_json(root / "manifest.json")
        bindings_path = root / "source_bindings.json"
        if not bindings_path.is_file():
            raise ValueError(f"profile input is not a snapshot_successes frozen store: {root}")
        bindings_doc = pb.read_json(bindings_path)
        if not isinstance(bindings_doc, dict):
            raise ValueError(f"invalid snapshot_successes source bindings: {root}")
        bindings = bindings_doc.get("items", {})
        manifest_ids = {row["id"] for row in manifest}
        if (bindings_doc.get("schema_version") != 1
                or bindings_doc.get("partial_snapshot") is not True
                or bindings_doc.get("measurement_id") != mid
                or bindings_doc.get("score_spec_id") != sid
                or set(bindings) != manifest_ids):
            raise ValueError(f"invalid snapshot_successes source bindings: {root}")
        entry_hashes = []
        for row_index, manifest_row in enumerate(manifest):
            name = manifest_row["id"]
            entry = results[name]
            binding = bindings[name]
            if (not isinstance(binding, dict)
                    or binding.get("source_sample_sha256") != entry["sample_sha256"]
                    or binding.get("source_rad_sha256") != entry["rad_sha256"]):
                raise ValueError(f"{name}: frozen source binding differs from validated artifacts")
            repeat = manifest_row.get("kind") in ("repeat", "notarize")
            entry_hashes.append({
                "id": name,
                "sample_file": entry["sample_file"],
                "sample_sha256": entry["sample_sha256"],
                "rad_file": entry["rad_file"],
                "rad_sha256": entry["rad_sha256"],
            })
            candidates.append({
                "store_index": store_index,
                "row_index": row_index,
                "store": root,
                "manifest_row": manifest_row,
                "raw_result": entry,
                "source_binding": binding,
                "repeat": repeat,
                "key": (mid, entry["pattern_sha256"]),
            })
        source = _profile_source_provenance(root, entry_hashes)
        source["manifest_rows"] = len(manifest)
        source["measurement_id"] = mid
        source["score_spec_id"] = sid
        provenance.append(source)

    winners = {}
    for candidate in candidates:
        priority = (candidate["repeat"], candidate["store_index"], candidate["row_index"])
        previous = winners.get(candidate["key"])
        if previous is None or priority < previous[0]:
            winners[candidate["key"]] = (priority, candidate)
    selected = sorted((item[1] for item in winners.values()),
                      key=lambda item: (item["store_index"], item["row_index"]))
    selected_identity = {(item["store_index"], item["row_index"]) for item in selected}
    duplicate_rows = []
    for candidate in candidates:
        identity = (candidate["store_index"], candidate["row_index"])
        if identity not in selected_identity:
            winner = winners[candidate["key"]][1]
            duplicate_rows.append({
                "id": candidate["manifest_row"]["id"],
                "source_store_index": candidate["store_index"],
                "pattern_sha256": candidate["key"][1],
                "kind": candidate["manifest_row"].get("kind"),
                "preferred_id": winner["manifest_row"]["id"],
                "preferred_source_store_index": winner["store_index"],
                "raw_manifest_row": candidate["manifest_row"],
                "raw_result": candidate["raw_result"],
            })

    rows, patterns, responses, radiation = [], [], [], []
    common_freqs = common_theta = None
    for array_index, candidate in enumerate(selected):
        root = candidate["store"]
        entry = candidate["raw_result"]
        manifest_row = candidate["manifest_row"]
        pattern, response = torch.load(
            root / entry["sample_file"], weights_only=True, map_location="cpu")
        rad = torch.load(root / entry["rad_file"], weights_only=True, map_location="cpu")
        pattern = np.asarray(pattern).reshape(25, 25).astype(bool)
        response = np.asarray(response)
        freqs = np.asarray(entry["freqs"], dtype=float)
        theta = np.asarray(rad["theta"], dtype=float)
        cuts = np.stack((np.asarray(rad["phi0"]), np.asarray(rad["phi90"])))
        if response.shape != (2, 17) or freqs.shape != (17,):
            raise ValueError(f"{manifest_row['id']}: expected complete 2x17 S11/Gain response")
        if theta.shape != (181,) or cuts.shape != (2, 181):
            raise ValueError(f"{manifest_row['id']}: expected complete phi0/phi90 181-angle radiation")
        if common_freqs is None:
            common_freqs, common_theta = freqs, theta
        elif not np.array_equal(freqs, common_freqs) or not np.array_equal(theta, common_theta):
            raise ValueError("profile stores contain inconsistent frequency or radiation grids")
        geom = pattern_geometry(pattern)
        metrics = {**geom, **_profile_response_metrics(response, freqs)}
        for cut_name, curve in zip(("phi0", "phi90"), cuts):
            metrics.update({f"{cut_name}_{key}": value
                            for key, value in radial_profile(theta, curve).items()})
        for key in ("mirror_power_45", "mirror_power_90", "mirror_db_mae_45",
                    "mirror_db_mae_90", "front_back_imbalance", "front_back_db"):
            values = np.asarray([metrics[f"phi0_{key}"], metrics[f"phi90_{key}"]])
            metrics[f"field_{key}"] = (float(np.mean(values[np.isfinite(values)]))
                                         if np.any(np.isfinite(values)) else math.nan)
        centroids = np.abs([metrics["phi0_power_centroid_deg"],
                            metrics["phi90_power_centroid_deg"]])
        metrics["field_abs_power_centroid_deg"] = float(np.mean(centroids))
        arm = manifest_row.get("selection_arm", manifest_row.get("arm", "unspecified"))
        row = {
            "array_index": array_index,
            "id": manifest_row["id"],
            "source_store_index": candidate["store_index"],
            "source_manifest_index": candidate["row_index"],
            "measurement_id": common_mid,
            "score_spec_id": common_sid,
            "pattern_sha256": entry["pattern_sha256"],
            "lineage_id": manifest_row.get("lineage_id", entry.get("lineage_id")),
            "kind": manifest_row.get("kind"),
            "selection_arm": str(arm),
            "repeat_or_notarize": candidate["repeat"],
            "raw_manifest_row": manifest_row,
            "raw_result": entry,
            "source_binding": candidate["source_binding"],
            **metrics,
        }
        rows.append(row)
        patterns.append(pattern)
        responses.append(response)
        radiation.append(cuts)

    response_array = np.stack(responses)
    radiation_array = np.stack(radiation)
    scalar_keys = list(PROFILE_SCALAR_KEYS)
    scalar_keys += [f"{cut}_{key}" for cut in ("phi0", "phi90")
                    for key in RADIAL_SCALAR_KEYS]
    scalar_keys += ["field_mirror_power_45", "field_mirror_power_90",
                    "field_mirror_db_mae_45", "field_mirror_db_mae_90",
                    "field_front_back_imbalance", "field_front_back_db",
                    "field_abs_power_centroid_deg"]
    arrays = {
        "ids": np.asarray([row["id"] for row in rows], dtype=np.str_),
        "source_store_index": np.asarray([row["source_store_index"] for row in rows], dtype=np.int32),
        "pattern_sha256": np.asarray([row["pattern_sha256"] for row in rows], dtype=np.str_),
        "lineage_id": np.asarray([row["lineage_id"] or "" for row in rows], dtype=np.str_),
        "selection_arm": np.asarray([row["selection_arm"] for row in rows], dtype=np.str_),
        "patterns": np.stack(patterns),
        "response_freqs_ghz": common_freqs,
        "responses": response_array,
        "s11_db": response_array[:, 0],
        "gain_db": response_array[:, 1],
        "radiation_theta_deg": common_theta,
        "radiation": radiation_array,
        "phi0_db": radiation_array[:, 0],
        "phi90_db": radiation_array[:, 1],
        **{key: np.asarray([row[key] for row in rows]) for key in scalar_keys},
    }
    arm_counts = Counter(row["selection_arm"] for row in rows)
    repeat_candidates = sum(item["repeat"] for item in candidates)
    repeat_selected = sum(item["repeat"] for item in selected)
    summary = {
        "schema_version": 1,
        "analysis_kind": "named_profile_frozen_store_symmetry",
        "created_local": datetime.now().astimezone().isoformat(),
        "measurement_id": common_mid,
        "score_spec_id": common_sid,
        "measurement": common_measurement,
        "score_spec": common_score,
        "scope": {
            "descriptive_only": True,
            "causal_claim": False,
            "performance_filter_applied": False,
            "hfss_or_training_run": False,
            "input_discovery": False,
            "dedup_key": ["measurement_id", "pattern_sha256"],
            "duplicate_preference": "nonrepeat_then_explicit_store_and_manifest_order",
        },
        "counts": {
            "source_stores": len(roots),
            "validated_complete_source_rows": len(candidates),
            "valid_response_rows": len(candidates),
            "valid_radiation_rows": len(candidates),
            "unique_selected_rows": len(rows),
            "duplicate_rows_removed": len(duplicate_rows),
            "repeat_or_notarize_candidates": repeat_candidates,
            "repeat_or_notarize_selected": repeat_selected,
            "symmetry_class": dict(Counter(row["symmetry_class"] for row in rows)),
        },
        "arms": {"selected_counts": dict(sorted(arm_counts.items()))},
        "quantiles": _quantile_summary(rows, scalar_keys),
        "definitions": {
            "geometry_mismatch_fraction": "mean XOR over 25x12 left/right pixel pairs",
            "upper_lower_metal_imbalance": "upper metal fraction minus lower metal fraction; centre row omitted",
            "field_mirror_power": "sum(abs(P(+theta)-P(-theta)))/sum(P(+theta)+P(-theta)) in linear power",
            "band": "inclusive 26.5--29.5 GHz",
            "s11_band_margin_db": "-10 dB minus the maximum S11 in band; descriptive only",
            "gain_band_margin_db": "minimum Gain in band minus 4 dB; descriptive only",
            "physical_validation": "profiled_batch.validate_store replay including actual bridge symmetry and artifact hashes",
        },
        "sources": provenance,
        "duplicate_rows": duplicate_rows,
        "rows": rows,
        "curve_storage": "aligned full curves are in the bound NPZ; array_index links each JSON row",
        "limitations": [
            "This artifact is a descriptive census of explicitly supplied complete frozen stores.",
            "Selection arms are provenance labels and are not interpreted as randomized treatments.",
            "No performance threshold was used to include, exclude, or deduplicate a row.",
            "Duplicate observations are retained in duplicate_rows as audit records but omitted from aligned arrays.",
        ],
    }
    _write_profile_outputs(summary, arrays, out_json, out_npz)
    return summary


def _spot_check(rows, patterns, resp, phi0, phi90, theta, backup_root, limit,
                max_store_files=64):
    """Deterministic, bounded checks against raw files in candidate metric stores."""
    if limit <= 0:
        return []
    try:
        import torch
    except ImportError:
        return [{"status": "torch_unavailable"}]
    picked, seen = [], set()
    for i, row in enumerate(rows):
        key = row["measurement_stratum"]
        if (key not in seen and row["metric_store"] and row["has_resp"] and row["has_rad"]):
            picked.append((i, row))
            seen.add(key)
        if len(picked) >= limit:
            break
    root = Path(backup_root)
    checks = []
    for i, row in picked:
        store_dir = root / row["metric_store"]
        target = np.packbits(patterns[i].reshape(-1)).tobytes()
        raw_resp = None
        files_seen = 0
        truncated = False
        if store_dir.is_dir():
            for path in sorted(store_dir.glob("*.pt")):
                if files_seen >= max_store_files:
                    truncated = True
                    break
                files_seen += 1
                try:
                    pat, rr = torch.load(path, weights_only=True, map_location="cpu")
                    packed = np.packbits((np.asarray(pat).reshape(25, 25) > 0.5)
                                         .astype(np.uint8).reshape(-1)).tobytes()
                    if packed == target:
                        raw_resp = np.asarray(rr, dtype=float).reshape(2, 17)
                        break
                except Exception:
                    continue
        raw_rad = None
        rpath = store_dir / "rad" / f"{row['id']}.pt"
        if rpath.is_file():
            try:
                value = torch.load(rpath, weights_only=True, map_location="cpu")
                raw_rad = (np.asarray(value["theta"], dtype=float),
                           np.asarray(value["phi0"], dtype=float),
                           np.asarray(value["phi90"], dtype=float))
            except Exception:
                pass
        resp_err = (float(np.nanmax(np.abs(raw_resp - resp[i])))
                    if raw_resp is not None else math.nan)
        rad_err = (float(np.nanmax(np.abs(np.r_[raw_rad[1] - phi0[i],
                                                  raw_rad[2] - phi90[i]])))
                   if raw_rad is not None and np.allclose(raw_rad[0], theta, atol=0.01)
                   else math.nan)
        checks.append({
            "id": row["id"], "metric_store": row["metric_store"],
            "measurement_stratum": row["measurement_stratum"],
            "store_pt_files_read_until_match": files_seen,
            "store_pt_scan_limit": max_store_files,
            "store_pt_scan_truncated": truncated,
            "response_match_max_abs": _json_number(resp_err),
            "rad_match_max_abs": _json_number(rad_err),
            "response_status": ("match" if np.isfinite(resp_err) and resp_err <= 0.02
                                else "mismatch" if np.isfinite(resp_err) else "missing"),
            "rad_status": ("match" if np.isfinite(rad_err) and rad_err <= 0.02
                           else "mismatch" if np.isfinite(rad_err) else "missing"),
        })
    return checks


def run(data_dir=DEFAULT_DATA, backup_root=DEFAULT_BACKUP, out_dir=DEFAULT_OUT,
        raw_checks=4):
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ppath, mpath = data_dir / "patterns.npz", data_dir / "meta.json"
    rpath, gpath = data_dir / "resp.npz", data_dir / "rad.npz"
    cpath, vpath = data_dir / "curve_state.json", data_dir / "variant_resp.json"
    pz, rz, gz = np.load(ppath), np.load(rpath), np.load(gpath)
    meta = json.loads(mpath.read_text(encoding="utf-8"))
    curve_state = json.loads(cpath.read_text(encoding="utf-8"))
    # np.load(.npz) decompresses an array on every ``z[key]`` access.  Materialize
    # each member once before the row loop; repeated indexing into the NpzFile
    # would otherwise expand the full archive tens of thousands of times.
    pattern_ids = np.asarray(pz["ids"])
    packed = np.asarray(pz["packed"])
    response_ids = np.asarray(rz["ids"])
    responses = np.asarray(rz["resp"])
    has_resp = np.asarray(rz["has_resp"], dtype=bool)
    radiation_ids = np.asarray(gz["ids"])
    theta = np.asarray(gz["theta"], dtype=float)
    phi0 = np.asarray(gz["phi0"])
    phi90 = np.asarray(gz["phi90"])
    has_rad = np.asarray(gz["has_rad"], dtype=bool)
    ids = [str(x) for x in pattern_ids]
    if len(ids) != len(set(ids)):
        raise ValueError("patterns ids are not unique")
    if ids != [str(x.get("id")) for x in meta]:
        raise ValueError("meta ids are not aligned with patterns")
    if not np.array_equal(pattern_ids, response_ids) or not np.array_equal(pattern_ids, radiation_ids):
        raise ValueError("curve ids are not aligned with patterns")
    patterns = np.unpackbits(packed, axis=1)[:, :625].reshape(-1, 25, 25).astype(bool)
    profiles, profile_audit = load_candidate_profiles(meta, backup_root)
    rows = []
    rad_curves = {"phi0": phi0, "phi90": phi90}
    rad_batch = {cut: radial_profiles_batch(theta, rad_curves[cut])
                 for cut in ("phi0", "phi90")}
    for i, (pid, m) in enumerate(zip(ids, meta)):
        geom = pattern_geometry(patterns[i])
        store = m.get("store")
        profile = profiles.get((str(store), pid), {}) if store else {}
        row = {
            "id": pid,
            "pattern_sha256": hashlib.sha256(packed[i].tobytes()).hexdigest(),
            "pattern_source_folder": m.get("folder"),
            "metric_store": store,
            "curve_provenance": "cache_id_aligned_store_unresolved",
            "measurement_profile_status": profile.get("measurement_profile_status", "unmeasured_or_unindexed"),
            "measurement_stratum": profile.get("measurement_stratum", "unmeasured_or_unindexed"),
            "measurement_profile_sha256": profile.get("measurement_profile_sha256"),
            "measurement_profile_sources": "|".join(profile.get("measurement_profile_sources", [])),
            **geom,
            "n8": m.get("n8"), "total_metal_pixels": m.get("total"),
            "has_resp": bool(has_resp[i]), "has_rad": bool(has_rad[i]),
            "missing_metric_store": store is None,
            "missing_response": not bool(has_resp[i]),
            "missing_radiation": not bool(has_rad[i]),
            "missing_curve_count": int(not bool(has_resp[i])) + int(not bool(has_rad[i])),
            "wm_cache": _finite_float(m.get("wm")), "rad_margin_cache": _finite_float(m.get("rad")),
            "lo_cache": _finite_float(m.get("lo")),
        }
        if row["has_resp"]:
            row.update(_response_metrics(responses[i]))
        else:
            row.update({k: math.nan for k in ("s11_db_28", "gain_db_28", "s11_db_band_mean",
                                               "s11_db_band_max", "gain_db_band_mean", "gain_db_band_min")})
        for cut in ("phi0", "phi90"):
            row.update({f"{cut}_{k}": values[i] for k, values in rad_batch[cut].items()})
        for name in ("mirror_power_45", "mirror_power_90", "mirror_db_mae_45",
                     "mirror_db_mae_90", "front_back_imbalance", "front_back_db"):
            vals = np.array([row[f"phi0_{name}"], row[f"phi90_{name}"]], dtype=float)
            row[f"field_{name}"] = float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else math.nan
        cents = np.abs([row["phi0_power_centroid_deg"], row["phi90_power_centroid_deg"]])
        row["field_abs_power_centroid_deg"] = (float(np.nanmean(cents))
                                                  if np.any(np.isfinite(cents)) else math.nan)
        rows.append(row)

    spot_checks = _spot_check(rows, patterns, responses, phi0, phi90,
                              theta, backup_root, raw_checks)
    csv_path = out_dir / "symmetry_census.csv.gz"
    fieldnames = list(rows[0]) if rows else []
    with open(csv_path, "wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=6, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)

    classes = Counter(r["symmetry_class"] for r in rows)
    strata = Counter(r["measurement_stratum"] for r in rows)
    missing = {
        "metric_store": sum(r["metric_store"] is None for r in rows),
        "response": sum(not r["has_resp"] for r in rows),
        "radiation": sum(not r["has_rad"] for r in rows),
        "both_response_and_radiation": sum(not r["has_resp"] and not r["has_rad"] for r in rows),
        "candidate_measurement_profile_unresolved": sum(
            r["measurement_profile_status"] != "resolved_candidate" for r in rows),
    }
    valid_rad = [r for r in rows if np.isfinite(r["field_mirror_power_90"])]
    exact_rad = [r for r in valid_rad if r["symmetry_class"] == "exact"]
    asym_rad = [r for r in valid_rad if r["symmetry_class"] == "asymmetric"]
    plane_contrasts = {
        "exact_geometry_largest_phi0_phi90_contrast": [
            {"id": r["id"], "phi0_mirror_power_90": r["phi0_mirror_power_90"],
             "phi90_mirror_power_90": r["phi90_mirror_power_90"],
             "metric_store": r["metric_store"]}
            for r in sorted(exact_rad,
                            key=lambda x: abs(x["phi0_mirror_power_90"] - x["phi90_mirror_power_90"]),
                            reverse=True)[:5]],
        "asymmetric_geometry_low_phi90_mismatch": [
            {"id": r["id"], "geometry_mismatch_fraction": r["geometry_mismatch_fraction"],
             "phi0_mirror_power_90": r["phi0_mirror_power_90"],
             "phi90_mirror_power_90": r["phi90_mirror_power_90"],
             "metric_store": r["metric_store"]}
            for r in sorted(asym_rad, key=lambda x: x["phi90_mirror_power_90"])[:5]],
    }
    field_distributions = {}
    for cls in ("exact", "near", "asymmetric"):
        group = [r for r in valid_rad if r["symmetry_class"] == cls]
        field_distributions[cls] = {"n": len(group)}
        for cut in ("phi0", "phi90"):
            for window in (45, 90):
                key = f"{cut}_mirror_power_{window}"
                vals = np.asarray([r[key] for r in group], dtype=float)
                vals = vals[np.isfinite(vals)]
                field_distributions[cls][key] = {
                    "median": float(np.median(vals)) if vals.size else None,
                    "p90": float(np.quantile(vals, 0.9)) if vals.size else None,
                }
    summary = {
        "schema_version": 1,
        "created_local": datetime.now().astimezone().isoformat(),
        "scope": "cache-level historical census; no new HFSS values",
        "definitions": {
            "geometry_mirror_A": "mean XOR over 25x12 left/right pixel pairs",
            "symmetry_classes": {"exact": "A=0", "near": "0<A<=0.1", "asymmetric": "A>0.1"},
            "field_mirror_power": "sum(abs(P(+theta)-P(-theta)))/sum(P(+theta)+P(-theta))",
            "power_centroid": "linear-power centroid of peak-minus-3 dB samples within -90..90 degrees",
            "front_back_imbalance": "(mean front linear power - mean back)/(sum of hemisphere means)",
        },
        "counts": {"rows": len(rows), "symmetry_class": dict(classes),
                   "valid_radiation_metrics": len(valid_rad), "exact_with_radiation": len(exact_rad)},
        "missing_counts": missing,
        "field_mirror_power_distribution_by_geometry_class": field_distributions,
        "measurement_strata_candidate_metric_store": dict(strata),
        "profile_metadata_audit": profile_audit,
        "raw_spot_checks": spot_checks,
        "plane_contrast_representatives": plane_contrasts,
        "cache_provenance": _cache_provenance(data_dir,
            [ppath, mpath, rpath, gpath, cpath, vpath], curve_state),
        "limitations": [
            "The cache producer can carry curves forward by bare id; curve-to-store provenance is unresolved.",
            "Measurement strata describe the current meta.store candidate only and are not used for causal contrasts.",
            "The browser snapshot is dated 2026-08-10 and is not asserted to contain later history.",
            "No raw .pt spot check was included in the final low-load artifact.",
            "Gain, S11 and radiation values are existing cached HFSS outputs; no new simulation was run.",
        ],
    }
    json_path = out_dir / "symmetry_census_summary.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_number) + "\n",
                         encoding="utf-8")
    print(f"rows={len(rows)} classes={dict(classes)}")
    print(f"missing={missing}")
    print(f"csv={csv_path}")
    print(f"json={json_path}")
    return summary


def _profile_main(argv):
    parser = argparse.ArgumentParser(
        description="具名 profile frozen store 對稱／響應盤點（零 HFSS、零訓練）")
    parser.add_argument("--store", action="append", required=True, type=Path,
                        help="snapshot_successes 產生的完整 frozen store；可重複指定")
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-npz", required=True, type=Path)
    args = parser.parse_args(argv)
    summary = analyze_profile_stores(args.store, args.out_json, args.out_npz)
    print(f"validated={summary['counts']['validated_complete_source_rows']} "
          f"unique={summary['counts']['unique_selected_rows']} "
          f"duplicates={summary['counts']['duplicate_rows_removed']}")
    print(f"json={args.out_json}")
    print(f"npz={args.out_npz}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "profile":
        _profile_main(sys.argv[2:])
        return
    parser = argparse.ArgumentParser(description="歷史 pattern/場型對稱盤點（零 HFSS）")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--backup-root", type=Path, default=DEFAULT_BACKUP)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--raw-checks", type=int, default=4,
                        help="候選 meta.store 的小量 raw 核對列數（0=不讀 torch）")
    args = parser.parse_args()
    run(args.data_dir, args.backup_root, args.out_dir, args.raw_checks)


if __name__ == "__main__":
    main()
