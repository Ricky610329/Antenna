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
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


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


def main():
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
