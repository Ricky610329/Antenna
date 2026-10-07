# -*- coding: utf-8 -*-
"""Bounded surrogate-model exploration with explicit, scoped inputs.

This module prepares immutable input bundles for the existing simulation worker.
It never runs HFSS, discovers historical stores, or writes outside ``repo/tmp``
when used through the CLI.  All observations enter through an explicit dataset
manifest and are copied into the run's scoped :class:`SampleStore`.

The supported loops are:

``single_fullcurve``
    A 25x25, exactly left/right symmetric single-port pattern.  The model predicts
    the complete two 17-point response curves plus two 91-point radiation cuts.
    Selection is deliberately performance-neutral: random, max-min Hamming,
    ensemble uncertainty, and predicted-curve diversity.

``dual_margin5``
    A dual-port loop whose current-profile 49-point response is converted to five
    configured margins by :mod:`antenna.measurement`.  Batch one is supplied by
    explicit history/specialist seeds; later batches use predicted minimum margin,
    uncertainty, and random arms.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch import nn

from antenna.utils.store import SampleStore, fingerprint


REPO = Path(__file__).resolve().parents[1]
TMP_ROOT = (REPO / "tmp").resolve()
SCHEMA_VERSION = 1
FEED = (24, 12)
RAD_THETA = np.linspace(-90.0, 90.0, 91, dtype=np.float32)
SINGLE_ARMS = ("random", "geometry", "uncertainty", "predcurve_diversity")
DUAL_FIRST_ARMS = ("history", "specialist", "random")
DUAL_LATER_ARMS = ("best_min_margin", "uncertainty", "random")


class TrainingInterrupted(RuntimeError):
    """Test hook used to verify exact epoch-boundary recovery."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def content_id(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def pattern_sha256(pattern: np.ndarray | torch.Tensor) -> str:
    arr = np.asarray(pattern)
    if arr.shape != (25, 25) or not np.isfinite(arr).all() or not np.isin(arr, [0, 1]).all():
        raise ValueError("pattern must be a finite binary 25x25 array")
    # Kept byte-for-byte identical to script.profiled_batch.pattern_sha256.
    return hashlib.sha256(np.packbits(arr.astype(bool)).tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True),
                   encoding="utf-8")
    os.replace(tmp, path)


def _atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _require_cli_output(path: Path) -> None:
    if not _inside(path, TMP_ROOT):
        raise ValueError(f"CLI output must be below {TMP_ROOT}, got {path.resolve()}")


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("configuration must be a YAML mapping")
    return value


def _measurement_api(measurement: Mapping[str, Any], score_spec: Mapping[str, Any]):
    # Lazy import lets the exploration unit tests run while the measurement module
    # is developed independently, but real preparation always exercises its API.
    from antenna.measurement import (frequency_grid, measurement_id, score_spec_id,
                                     validate_measurement, validate_score_spec)

    checked_m = validate_measurement(copy.deepcopy(dict(measurement)))
    if not isinstance(checked_m, Mapping):
        checked_m = dict(measurement)
    checked_s = validate_score_spec(copy.deepcopy(dict(score_spec)), measurement=checked_m)
    if not isinstance(checked_s, Mapping):
        checked_s = dict(score_spec)
    freqs = np.asarray(frequency_grid(checked_m), dtype=np.float32)
    return (dict(checked_m), dict(checked_s), str(measurement_id(checked_m)),
            str(score_spec_id(checked_s)), freqs)


def validate_exploration_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(cfg.get("measurement"), Mapping):
        raise ValueError("configuration needs a measurement mapping")
    score = cfg.get("score_spec", cfg.get("score"))
    if not isinstance(score, Mapping):
        raise ValueError("configuration needs a score_spec mapping")
    exp = copy.deepcopy(dict(cfg.get("exploration", {})))
    mode = exp.get("mode")
    if mode not in ("single_fullcurve", "dual_margin5"):
        raise ValueError("exploration.mode must be single_fullcurve or dual_margin5")
    defaults = {
        "batch_size": 48 if mode == "single_fullcurve" else 60,
        "max_batches": 3,
        "seed": 0,
        "candidate_pool_size": 1024,
        "ensemble_seeds": [0, 1],
        "hidden_dims": [128, 128],
        "epochs": 20,
        "pretrain_epochs": 10,
        "batch_train_size": 32,
        "learning_rate": 1e-3,
        "holdout_fraction": 0.2,
        "geometry_profile": "unspecified",
    }
    for key, val in defaults.items():
        exp.setdefault(key, val)
    expected = SINGLE_ARMS if mode == "single_fullcurve" else DUAL_FIRST_ARMS
    later = SINGLE_ARMS if mode == "single_fullcurve" else DUAL_LATER_ARMS
    exp.setdefault("arms", {k: exp["batch_size"] // len(expected) for k in expected})
    exp.setdefault("later_arms", {k: exp["batch_size"] // len(later) for k in later})
    for name in ("batch_size", "max_batches", "candidate_pool_size", "epochs",
                 "pretrain_epochs", "batch_train_size"):
        exp[name] = int(exp[name])
        if exp[name] < (0 if name == "pretrain_epochs" else 1):
            raise ValueError(f"exploration.{name} must be positive")
    exp["seed"] = int(exp["seed"])
    exp["ensemble_seeds"] = [int(s) for s in exp["ensemble_seeds"]]
    if exp["ensemble_seeds"] != [0, 1]:
        raise ValueError("exploration.ensemble_seeds must be [0, 1]")
    exp["hidden_dims"] = [int(v) for v in exp["hidden_dims"]]
    if any(v < 1 for v in exp["hidden_dims"]):
        raise ValueError("hidden_dims must be positive")
    exp["learning_rate"] = float(exp["learning_rate"])
    exp["holdout_fraction"] = float(exp["holdout_fraction"])
    if exp["max_batches"] != 3:
        raise ValueError("bounded exploration requires exactly three batches")
    if exp["candidate_pool_size"] < exp["batch_size"]:
        raise ValueError("candidate_pool_size must be at least batch_size")
    if exp["learning_rate"] <= 0:
        raise ValueError("learning_rate must be positive")
    if not 0 <= exp["holdout_fraction"] < 1:
        raise ValueError("holdout_fraction must be in [0, 1)")
    if not str(exp["geometry_profile"]).strip():
        raise ValueError("geometry_profile must be non-empty")
    for key in ("arms", "later_arms"):
        exp[key] = {str(k): int(v) for k, v in dict(exp[key]).items()}
        wanted = set(expected if key == "arms" else later)
        if set(exp[key]) != wanted or any(v <= 0 for v in exp[key].values()):
            raise ValueError(f"exploration.{key} must contain positive counts for {sorted(wanted)}")
        if sum(exp[key].values()) != exp["batch_size"]:
            raise ValueError(f"exploration.{key} must sum to batch_size")
    if mode == "single_fullcurve" and exp["batch_size"] == 48:
        if any(v != 12 for v in exp["arms"].values()) or any(v != 12 for v in exp["later_arms"].values()):
            raise ValueError("single batch 48 requires four arms of 12")
    if mode == "dual_margin5" and exp["batch_size"] == 60:
        if any(v != 20 for v in exp["arms"].values()) or any(v != 20 for v in exp["later_arms"].values()):
            raise ValueError("dual batch 60 requires three arms of 20")
    cfg2 = copy.deepcopy(dict(cfg))
    cfg2["exploration"] = exp
    cfg2["score_spec"] = copy.deepcopy(dict(score))
    cfg2.pop("score", None)
    return cfg2


def enforce_single_geometry(pattern: np.ndarray | torch.Tensor) -> np.ndarray:
    """Return an exact 25x25 half-12 mirror and force the feed pixel on."""
    p = np.asarray(pattern, dtype=np.float32)
    if p.size != 625:
        raise ValueError(f"single pattern must contain 625 pixels, got {p.size}")
    p = (p.reshape(25, 25) > 0.5)
    p[:, 13:] = p[:, :12][:, ::-1]
    p[FEED] = True
    if not np.array_equal(p[:, :12], p[:, 13:][:, ::-1]):
        raise AssertionError("internal symmetry failure")
    return p


def enforce_dual_geometry(pattern: np.ndarray | torch.Tensor) -> np.ndarray:
    """Force the two 5x5 port pads required by the p01 dual simulator."""
    p = np.asarray(pattern, dtype=np.float32)
    if p.size != 625:
        raise ValueError(f"dual pattern must contain 625 pixels, got {p.size}")
    p = p.reshape(25, 25) > 0.5
    p[:5, 10:15] = True
    p[20:25, 10:15] = True
    return p


def random_single_pattern(rng: np.random.Generator) -> np.ndarray:
    p = np.zeros((25, 25), dtype=bool)
    p[:, :13] = rng.random((25, 13)) < rng.uniform(0.25, 0.75)
    return enforce_single_geometry(p)


def mutate_pattern(parent: np.ndarray, rng: np.random.Generator, *, single: bool,
                   min_flips: int = 4, max_flips: int = 64) -> np.ndarray:
    p = (np.asarray(parent).reshape(25, 25) > 0.5).copy()
    n = int(rng.integers(min_flips, max_flips + 1))
    if single:
        # Flip independent variables only; mirroring is applied afterwards.
        rr = rng.integers(0, 25, size=n)
        cc = rng.integers(0, 13, size=n)
        p[rr, cc] = ~p[rr, cc]
        return enforce_single_geometry(p)
    idx = rng.choice(625, size=min(n, 625), replace=False)
    p.reshape(-1)[idx] = ~p.reshape(-1)[idx]
    return enforce_dual_geometry(p)


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero((np.asarray(a).reshape(-1) > 0.5) !=
                                (np.asarray(b).reshape(-1) > 0.5)))


def _pattern_order_key(pattern: np.ndarray) -> str:
    bits = (np.asarray(pattern).reshape(-1) > 0.5).astype(np.uint8)
    return hashlib.sha256(np.packbits(bits).tobytes()).hexdigest()


def maxmin_hamming_indices(patterns: Sequence[np.ndarray], n: int, *,
                           references: Sequence[np.ndarray] = (),
                           eligible: Sequence[int] | None = None) -> list[int]:
    pool = list(range(len(patterns))) if eligible is None else list(eligible)
    refs = list(references)
    picked: list[int] = []
    while pool and len(picked) < n:
        if not refs and not picked:
            choice = min(pool, key=lambda i: _pattern_order_key(patterns[i]))
        else:
            prior = refs + [patterns[i] for i in picked]
            choice = min(pool, key=lambda i: (-min(hamming(patterns[i], q) for q in prior),
                                               _pattern_order_key(patterns[i])))
        picked.append(choice)
        pool.remove(choice)
    return picked


def maxmin_vector_indices(vectors: np.ndarray, n: int, *, eligible: Sequence[int],
                          reference_indices: Sequence[int] = ()) -> list[int]:
    pool = list(eligible)
    picked: list[int] = []
    if len(pool) == 0:
        return picked
    scale = np.nanstd(vectors[pool], axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    z = vectors / scale
    while pool and len(picked) < n:
        refs = list(reference_indices) + picked
        if not refs:
            choice = min(pool)
        else:
            choice = min(pool, key=lambda i: (-min(float(np.linalg.norm(z[i] - z[j]))
                                                   for j in refs), i))
        picked.append(choice)
        pool.remove(choice)
    return picked


def grouped_holdout_indices(rows: Sequence[Mapping[str, Any]], fraction: float,
                            seed: int) -> tuple[list[int], list[int]]:
    """Split whole (lineage, geometry-profile) groups to prevent variant leakage."""
    groups: dict[tuple[str, str], list[int]] = {}
    for i, row in enumerate(rows):
        lineage = str(row.get("lineage_id") or row.get("pattern_sha256") or i)
        profile = str(row.get("geometry_profile") or "unspecified")
        groups.setdefault((lineage, profile), []).append(i)
    keys = sorted(groups)
    if len(keys) <= 1 or fraction <= 0:
        return list(range(len(rows))), []
    rng = np.random.default_rng(int(seed))
    perm = [keys[i] for i in rng.permutation(len(keys))]
    n_hold = min(len(keys) - 1, max(1, int(round(len(keys) * fraction))))
    hold_keys = set(perm[:n_hold])
    train, hold = [], []
    for key in keys:
        (hold if key in hold_keys else train).extend(groups[key])
    return sorted(train), sorted(hold)


def radiation_vector(rad: Mapping[str, Any]) -> np.ndarray:
    theta = np.asarray(rad["theta"], dtype=np.float64).reshape(-1)
    if theta.size < 2 or not np.all(np.isfinite(theta)) or np.any(np.diff(theta) <= 0):
        raise ValueError("radiation theta must be finite and strictly increasing")
    cuts = []
    for key in ("phi0", "phi90"):
        val = np.asarray(rad[key], dtype=np.float64).reshape(-1)
        if val.shape != theta.shape or not np.all(np.isfinite(val)):
            raise ValueError(f"radiation {key} must be finite and match theta")
        if theta[0] > -90.0 or theta[-1] < 90.0:
            raise ValueError("radiation must cover -90 through 90 degrees")
        cuts.append(np.interp(RAD_THETA, theta, val).astype(np.float32))
    return np.concatenate(cuts)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_rows(path: Path) -> list[dict[str, Any]]:
    value = _read_json(path)
    if isinstance(value, dict):
        value = value.get("samples", value.get("entries"))
    if not isinstance(value, list):
        raise ValueError(f"manifest must be a list or contain samples: {path}")
    return [dict(row) for row in value]


def _result_rows(path: Path) -> list[dict[str, Any]]:
    """Read profiled_batch ``results.json`` directly (id -> observation)."""
    value = _read_json(path)
    if isinstance(value, list):
        return [dict(row) for row in value]
    if not isinstance(value, dict):
        raise ValueError(f"results must be an id mapping: {path}")
    rows = []
    for sample_id, raw in value.items():
        if not isinstance(raw, Mapping):
            raise ValueError(f"result {sample_id} must be a mapping")
        row = dict(raw)
        if "id" in row and row["id"] != sample_id:
            raise ValueError(f"result id mismatch for {sample_id}")
        row["id"] = sample_id
        rows.append(row)
    return rows


def _safe_input_path(root: Path, value: str | Path) -> Path:
    p = Path(value)
    p = p if p.is_absolute() else root / p
    if not _inside(p, root):
        raise ValueError(f"input escapes dataset root: {p}")
    return p


def _safe_id_prefix(value: str) -> str:
    prefix = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_-")[:24]
    if not prefix:
        raise ValueError("exploration.id_prefix must contain a safe filename character")
    return prefix


def load_seed_rows(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for supplied in paths:
        supplied = supplied.resolve()
        if supplied.is_dir():
            manifest_path, base = supplied / "manifest.json", supplied
        else:
            manifest_path, base = supplied, supplied.parent
        for row in _manifest_rows(manifest_path):
            rel = row.get("pattern_file", f"{row['id']}.pt")
            path = _safe_input_path(base, rel)
            value = torch.load(path, weights_only=True)
            pattern = value[0] if isinstance(value, (tuple, list)) else value
            arr = (np.asarray(pattern, dtype=np.float32).reshape(25, 25) > 0.5)
            rows.append({**row, "pattern": arr, "seed_input": str(manifest_path)})
    return rows


def snapshot_seed_inputs(work_dir: Path, paths: Sequence[Path]) -> list[Path]:
    """Copy explicit seeds into the run so later batches survive path/NAS moves."""
    if not paths:
        return []
    rows = load_seed_rows(paths)
    out = work_dir / "seed_snapshot"
    out.mkdir(parents=True, exist_ok=False)
    manifest = []
    seen = set()
    for i, row in enumerate(rows):
        pattern = np.asarray(row["pattern"], dtype=bool).reshape(25, 25)
        sha = pattern_sha256(pattern)
        if sha in seen:
            raise ValueError("seed inputs contain duplicate patterns")
        seen.add(sha)
        sid = f"seed_{i:04d}_{sha[:8]}"
        torch.save(torch.as_tensor(pattern, dtype=torch.float32), out / f"{sid}.pt")
        clean = {k: v for k, v in row.items()
                 if k not in ("pattern", "seed_input", "pattern_file", "id")}
        clean.update({"id": sid,
                      "original_id": row.get("original_id", row.get("id")),
                      "snapshot_source_id": row.get("id"),
                      "pattern_file": f"{sid}.pt", "pattern_sha256": sha})
        manifest.append(clean)
    _atomic_json(out / "manifest.json", manifest)
    return [Path("seed_snapshot")]


def diverse_parents(seed_rows: Sequence[Mapping[str, Any]], limit: int = 32) -> list[dict[str, Any]]:
    if not seed_rows:
        return []
    pats = [np.asarray(r["pattern"]) for r in seed_rows]
    idx = maxmin_hamming_indices(pats, min(limit, len(pats)))
    return [dict(seed_rows[i]) for i in idx]


def generate_candidates(seed_rows: Sequence[Mapping[str, Any]], count: int, *, seed: int,
                        single: bool) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    parents = diverse_parents(seed_rows)
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    # With usable parents, reserve exactly half the bounded pool for globally
    # fresh legal geometries.  The other half explores local variants around a
    # diverse mix of explicit seeds and observations from this scoped run.
    fresh_target = count if not parents else (count + 1) // 2
    variant_target = count - fresh_target

    def append_candidate(pattern, *, group, source, parent=None):
        key = pattern_sha256(pattern)
        if key in seen:
            return False
        seen.add(key)
        if parent is None:
            parent_id, lineage, parent_source = None, key, None
        else:
            parent_id = str(parent.get("id", "seed"))
            lineage = str(parent.get("lineage_id") or parent.get("pattern_sha256") or
                          pattern_sha256(parent["pattern"]))
            parent_source = parent.get("source")
        out.append({"pattern": pattern, "seed": seed, "candidate_index": len(out),
                    "candidate_group": group, "parent_id": parent_id,
                    "parent_source": parent_source, "lineage_id": lineage,
                    "source": source, "pattern_sha256": key})
        return True

    attempts = 0
    fresh_count = 0
    while fresh_count < fresh_target and attempts < max(100, count * 100):
        attempts += 1
        p = (random_single_pattern(rng) if single
             else enforce_dual_geometry(rng.random((25, 25)) < 0.5))
        if append_candidate(p, group="fresh_random", source="fresh_legal_random"):
            fresh_count += 1

    attempts = 0
    variant_count = 0
    while variant_count < variant_target and attempts < max(100, count * 100):
        parent = parents[attempts % len(parents)]
        attempts += 1
        p = mutate_pattern(np.asarray(parent["pattern"]), rng, single=single)
        if append_candidate(p, group="parent_variant", source="diverse_parent_mutation",
                            parent=parent):
            variant_count += 1
    if len(out) != count:
        raise RuntimeError(f"could generate only {len(out)}/{count} unique candidates")
    return out


def load_scoped_parent_rows(work_dir: Path) -> list[dict[str, Any]]:
    """Load only observations imported into this run; never discover history."""
    manifest_path = work_dir / "dataset_manifest.json"
    if not manifest_path.is_file():
        return []
    rows = []
    for row in _manifest_rows(manifest_path):
        sample = _safe_input_path(work_dir, row["scoped_sample_file"])
        pattern, _response = torch.load(sample, weights_only=True, map_location="cpu")
        rows.append({"id": row["id"], "pattern": np.asarray(pattern).reshape(25, 25),
                     "pattern_sha256": row["pattern_sha256"],
                     "lineage_id": row["lineage_id"],
                     "source": "current_scoped_observation"})
    return rows


def mixed_diverse_parents(seed_rows: Sequence[Mapping[str, Any]],
                          current_rows: Sequence[Mapping[str, Any]],
                          limit: int = 32) -> list[dict[str, Any]]:
    """Guarantee current observations and explicit seeds both contribute."""
    def unique(rows, excluded=()):
        seen = set(excluded)
        out = []
        for row in rows:
            key = pattern_sha256(row["pattern"])
            if key not in seen:
                seen.add(key); out.append(row)
        return out, seen

    unique_current, current_hashes = unique(current_rows)
    unique_seeds, _ = unique(seed_rows, current_hashes)
    current_limit = min(len(unique_current), limit // 2 if unique_seeds else limit)
    current = diverse_parents(unique_current, current_limit)
    remaining = limit - len(current)
    seeds = diverse_parents(unique_seeds, min(len(unique_seeds), remaining))
    return current + seeds


class CurveMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], output_dim: int):
        super().__init__()
        dims = [input_dim, *hidden_dims, output_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i + 1 < len(dims) - 1:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


@dataclass
class TrainingSet:
    x: torch.Tensor
    y: torch.Tensor
    rows: list[dict[str, Any]]


def _score_dual(response: torch.Tensor, measurement: Mapping[str, Any],
                score_spec: Mapping[str, Any], labels: Sequence[str],
                freqs: Sequence[float]) -> np.ndarray:
    from antenna.measurement import score_response

    scored = score_response(np.asarray(response, dtype=np.float32), labels=list(labels),
                            freqs=np.asarray(freqs, dtype=np.float32), score_spec=score_spec)
    raw_margins = scored["margins"]
    margins = np.asarray(list(raw_margins.values()) if isinstance(raw_margins, Mapping)
                         else raw_margins, dtype=np.float32).reshape(-1)
    if margins.size != 5 or not np.all(np.isfinite(margins)):
        raise ValueError("dual score_response must yield five finite margins")
    return margins


def load_training_set(work_dir: Path) -> TrainingSet:
    state = _read_json(work_dir / "state.json")
    cfg = _load_yaml(work_dir / "config.yaml")
    measurement = _read_json(work_dir / "measurement.json")
    score = _read_json(work_dir / "score_spec.json")
    mode = cfg["exploration"]["mode"]
    freqs = np.asarray(state["frequencies_ghz"], dtype=np.float32)
    entries = _manifest_rows(work_dir / "dataset_manifest.json")
    xs, ys, good_rows = [], [], []
    for row in entries:
        sample_path = _safe_input_path(work_dir, row["scoped_sample_file"])
        x, response = torch.load(sample_path, weights_only=True)
        x = torch.as_tensor(x, dtype=torch.float32).reshape(25, 25)
        response = torch.as_tensor(response, dtype=torch.float32)
        if mode == "single_fullcurve":
            if tuple(response.shape) != (2, len(freqs)):
                raise ValueError(f"single response shape mismatch: {tuple(response.shape)}")
            rad = torch.load(_safe_input_path(work_dir, row["scoped_rad_file"]), weights_only=True)
            target = np.concatenate([response.numpy().reshape(-1), radiation_vector(rad)])
        else:
            labels = row.get("response_labels", measurement["labels"])
            row_freqs = row.get("response_freqs_ghz", freqs)
            if tuple(response.shape) != (len(labels), len(freqs)):
                raise ValueError(f"dual response shape mismatch: {tuple(response.shape)}")
            target = _score_dual(response, measurement, score, labels, row_freqs)
        if not np.all(np.isfinite(target)):
            raise ValueError(f"non-finite target for {row['id']}")
        xs.append(x.reshape(-1))
        ys.append(torch.as_tensor(target, dtype=torch.float32))
        good_rows.append(row)
    if not xs:
        return TrainingSet(torch.empty((0, 625)), torch.empty((0, 0)), [])
    return TrainingSet(torch.stack(xs), torch.stack(ys), good_rows)


def _model_output_dim(mode: str, n_freq: int) -> int:
    return 2 * n_freq + 2 * len(RAD_THETA) if mode == "single_fullcurve" else 5


def _training_signature(rows: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any],
                        through_batch: int, member_seed: int) -> str:
    return content_id({"samples": [r["sample_fingerprint"] for r in rows],
                       "config": cfg["exploration"], "through_batch": through_batch,
                       "member_seed": member_seed})


def _epoch_order(n: int, seed: int, phase: int, epoch: int) -> np.ndarray:
    return np.random.default_rng(seed * 1000003 + phase * 10007 + epoch).permutation(n)


def _fit_norm(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False)
    std[std < 1e-6] = 1.0
    return mean, std


def _load_old_pretrain(cfg: Mapping[str, Any], mode: str, measurement_id_value: str,
                       score_spec_id_value: str, freqs: np.ndarray) -> TrainingSet | None:
    old = cfg["exploration"].get("old_pretrain")
    if not old:
        return None
    root = Path(old["dataset_root"]).resolve()
    manifest = _safe_input_path(root, old.get("manifest", "manifest.json"))
    rows = _manifest_rows(manifest)
    xs, ys, kept = [], [], []
    measurement = cfg["measurement"]
    score = cfg["score_spec"]
    for row in rows:
        if row.get("status") != "ok":
            continue
        sample_path = _safe_input_path(root, row["sample_file"])
        x, response = torch.load(sample_path, weights_only=True)
        x = torch.as_tensor(x, dtype=torch.float32).reshape(25, 25)
        response = torch.as_tensor(response, dtype=torch.float32)
        if mode == "single_fullcurve":
            if response.shape != (2, len(freqs)):
                raise ValueError("old single profile has incompatible sweep")
            rad_path = _safe_input_path(root, row.get("rad_file", f"rad/{row['id']}.pt"))
            target = np.concatenate([response.numpy().reshape(-1),
                                     radiation_vector(torch.load(rad_path, weights_only=True))])
        else:
            # An old 17-point dual profile is rejected instead of silently feeding
            # it into the current 49-point model.
            if response.shape != (len(measurement["labels"]), len(freqs)):
                raise ValueError("old dual profile sweep is incompatible with current measurement")
            target = _score_dual(response, measurement, score,
                                 row.get("response_labels", measurement["labels"]),
                                 row.get("response_freqs_ghz", freqs))
        xs.append(x.reshape(-1)); ys.append(torch.as_tensor(target)); kept.append(row)
    if not xs:
        return None
    return TrainingSet(torch.stack(xs), torch.stack(ys).float(), kept)


def _save_member_checkpoint(path: Path, *, model: nn.Module, optimizer: torch.optim.Optimizer,
                            signature: str, mode: str, output_dim: int, measurement_id_value: str,
                            score_spec_id_value: str, geometry_profile: str, member_seed: int,
                            phase: int, epoch: int, norms: Sequence[tuple[torch.Tensor, torch.Tensor]],
                            complete: bool, through_batch: int) -> None:
    _atomic_torch(path, {
        "schema_version": SCHEMA_VERSION, "signature": signature, "mode": mode,
        "output_dim": output_dim, "measurement_id": measurement_id_value,
        "score_spec_id": score_spec_id_value, "geometry_profile": geometry_profile,
        "member_seed": member_seed, "phase": phase, "epoch": epoch,
        "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
        "norms": [(m.cpu(), s.cpu()) for m, s in norms], "complete": complete,
        "through_batch": through_batch, "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
    })


def train_models(work_dir: Path, through_batch: int, *,
                 interrupt_after_epochs: int | None = None) -> list[Path]:
    cfg = validate_exploration_config(_load_yaml(work_dir / "config.yaml"))
    exp = cfg["exploration"]
    state = _read_json(work_dir / "state.json")
    for batch in range(1, through_batch + 1):
        info = state["batches"].get(str(batch), {})
        if not info.get("feedback_audit"):
            raise ValueError(f"batch {batch} feedback must be audited before training")
    current = load_training_set(work_dir)
    use = [i for i, r in enumerate(current.rows) if int(r["batch"]) <= through_batch]
    current = TrainingSet(current.x[use], current.y[use], [current.rows[i] for i in use])
    train_idx, hold_idx = grouped_holdout_indices(current.rows, exp["holdout_fraction"], exp["seed"])
    if not train_idx:
        raise ValueError("no current-profile training samples")
    mode = exp["mode"]
    freqs = np.asarray(state["frequencies_ghz"], dtype=np.float32)
    output_dim = _model_output_dim(mode, len(freqs))
    if current.y.shape[1] != output_dim:
        raise ValueError(f"model target is {current.y.shape[1]}, expected {output_dim}")
    old = _load_old_pretrain(cfg, mode, state["measurement_id"], state["score_spec_id"], freqs)
    phases: list[tuple[str, TrainingSet, int]] = []
    if old is not None and exp["pretrain_epochs"]:
        phases.append(("old_pretrain", old, exp["pretrain_epochs"]))
    phases.append(("current_finetune", TrainingSet(current.x[train_idx], current.y[train_idx],
                                                    [current.rows[i] for i in train_idx]),
                   exp["epochs"]))
    model_dir = work_dir / "models" / f"batch-{through_batch:03d}"
    paths: list[Path] = []
    completed_epochs_this_call = 0
    for member_seed in exp["ensemble_seeds"]:
        torch.manual_seed(member_seed)
        np.random.seed(member_seed)
        model = CurveMLP(625, exp["hidden_dims"], output_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=exp["learning_rate"])
        path = model_dir / f"member-{member_seed}.pt"
        signature = _training_signature(current.rows, cfg, through_batch, member_seed)
        phase_start, epoch_start, norms = 0, 0, []
        if path.exists():
            saved = torch.load(path, weights_only=False, map_location="cpu")
            _validate_checkpoint(saved, state, mode, output_dim, signature=signature)
            if saved.get("complete"):
                paths.append(path)
                continue
            model.load_state_dict(saved["model_state"])
            optimizer.load_state_dict(saved["optimizer_state"])
            phase_start, epoch_start = int(saved["phase"]), int(saved["epoch"])
            norms = list(saved["norms"])
        for phase_i, (_phase_name, dataset, epochs) in enumerate(phases):
            if phase_i < phase_start:
                continue
            x, y = dataset.x.float(), dataset.y.float()
            if phase_i >= len(norms):
                norms.append(_fit_norm(x))
            mean, std = norms[phase_i]
            first_epoch = epoch_start if phase_i == phase_start else 0
            for epoch in range(first_epoch, epochs):
                model.train()
                order = _epoch_order(len(x), member_seed, phase_i, epoch)
                bs = exp["batch_train_size"]
                for start in range(0, len(order), bs):
                    idx = torch.as_tensor(order[start:start + bs], dtype=torch.long)
                    pred = model((x[idx] - mean) / std)
                    loss = torch.mean((pred - y[idx]) ** 2)
                    if not torch.isfinite(loss):
                        raise ValueError("non-finite training loss")
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                next_phase, next_epoch = phase_i, epoch + 1
                if next_epoch >= epochs:
                    next_phase, next_epoch = phase_i + 1, 0
                complete = next_phase >= len(phases)
                _save_member_checkpoint(path, model=model, optimizer=optimizer,
                                        signature=signature, mode=mode, output_dim=output_dim,
                                        measurement_id_value=state["measurement_id"],
                                        score_spec_id_value=state["score_spec_id"],
                                        geometry_profile=exp["geometry_profile"],
                                        member_seed=member_seed, phase=next_phase,
                                        epoch=next_epoch, norms=norms, complete=complete,
                                        through_batch=through_batch)
                completed_epochs_this_call += 1
                if (interrupt_after_epochs is not None and
                        completed_epochs_this_call >= interrupt_after_epochs):
                    raise TrainingInterrupted("intentional epoch-boundary interruption")
        paths.append(path)
    metrics = {"through_batch": through_batch, "n_total": len(current.rows),
               "n_train": len(train_idx), "n_holdout": len(hold_idx),
               "train_ids": [current.rows[i]["id"] for i in train_idx],
               "holdout_ids": [current.rows[i]["id"] for i in hold_idx],
               "model_files": [p.relative_to(work_dir).as_posix() for p in paths]}
    _atomic_json(model_dir / "training_summary.json", metrics)
    state = _read_json(work_dir / "state.json")
    state["batches"][str(through_batch)]["trained_model_dir"] = model_dir.relative_to(work_dir).as_posix()
    _atomic_json(work_dir / "state.json", state)
    return paths


def _validate_checkpoint(saved: Mapping[str, Any], state: Mapping[str, Any], mode: str,
                         output_dim: int, signature: str | None = None) -> None:
    expected = {"schema_version": SCHEMA_VERSION, "mode": mode, "output_dim": output_dim,
                "measurement_id": state["measurement_id"],
                "score_spec_id": state["score_spec_id"]}
    for key, val in expected.items():
        if saved.get(key) != val:
            raise ValueError(f"checkpoint {key} mismatch: {saved.get(key)!r} != {val!r}")
    if signature is not None and saved.get("signature") != signature:
        raise ValueError("checkpoint training signature mismatch")


def _ensemble_hash(paths: Sequence[Path]) -> str:
    h = hashlib.sha256()
    for path in paths:
        h.update(path.read_bytes())
    return h.hexdigest()


def predict_ensemble(work_dir: Path, batch: int, patterns: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray, str]:
    cfg = validate_exploration_config(_load_yaml(work_dir / "config.yaml"))
    exp = cfg["exploration"]
    state = _read_json(work_dir / "state.json")
    model_dir = work_dir / "models" / f"batch-{batch:03d}"
    paths = [model_dir / f"member-{seed}.pt" for seed in exp["ensemble_seeds"]]
    x = torch.as_tensor(np.asarray(patterns), dtype=torch.float32).reshape(len(patterns), -1)
    preds = []
    output_dim = _model_output_dim(exp["mode"], len(state["frequencies_ghz"]))
    for path in paths:
        saved = torch.load(path, weights_only=False, map_location="cpu")
        _validate_checkpoint(saved, state, exp["mode"], output_dim)
        if not saved.get("complete"):
            raise ValueError(f"incomplete model checkpoint: {path}")
        model = CurveMLP(625, exp["hidden_dims"], output_dim)
        model.load_state_dict(saved["model_state"]); model.eval()
        mean, std = saved["norms"][-1]
        with torch.no_grad():
            preds.append(model((x - mean) / std).numpy())
    stacked = np.stack(preds)
    mean = stacked.mean(axis=0)
    uncertainty = stacked.std(axis=0).mean(axis=1)
    return mean, uncertainty, _ensemble_hash(paths)


def _fallback_geometry(candidates: Sequence[Mapping[str, Any]], n: int, used: set[int]) -> list[int]:
    pats = [np.asarray(c["pattern"]) for c in candidates]
    refs = [pats[i] for i in sorted(used)]
    eligible = [i for i in range(len(candidates)) if i not in used]
    return maxmin_hamming_indices(pats, n, references=refs, eligible=eligible)


def select_candidate_arms(candidates: Sequence[Mapping[str, Any]], arms: Mapping[str, int], *,
                          seed: int, mode: str, predictions: np.ndarray | None,
                          uncertainty: np.ndarray | None) -> list[dict[str, Any]]:
    """Select disjoint arms. Single mode never ranks by predicted performance."""
    rng = np.random.default_rng(seed)
    used: set[int] = set()
    selected: list[dict[str, Any]] = []
    valid_pred = (np.all(np.isfinite(predictions), axis=1)
                  if predictions is not None and predictions.shape[0] == len(candidates)
                  else np.zeros(len(candidates), dtype=bool))
    for arm, count in arms.items():
        available = [i for i in range(len(candidates)) if i not in used]
        fallback = None
        if arm == "random":
            has_strata = any("candidate_group" in row for row in candidates)
            fresh = ([i for i in available if candidates[i].get("candidate_group") == "fresh_random"]
                     if has_strata else available)
            if len(fresh) < count:
                raise ValueError(f"random arm needs {count} unused fresh candidates, got {len(fresh)}")
            chosen = list(rng.permutation(fresh)[:count])
        elif arm == "geometry":
            chosen = _fallback_geometry(candidates, count, used)
        elif arm == "uncertainty":
            if (uncertainty is None or len(uncertainty) != len(candidates) or
                    not np.all(np.isfinite(uncertainty[available]))):
                chosen = _fallback_geometry(candidates, count, used)
                fallback = "invalid_or_missing_prediction_to_geometry"
            else:
                chosen = sorted(available, key=lambda i: (-float(uncertainty[i]), i))[:count]
        elif arm == "predcurve_diversity":
            eligible = [i for i in available if bool(valid_pred[i])]
            if predictions is None or len(eligible) != len(available):
                chosen = _fallback_geometry(candidates, count, used)
                fallback = "invalid_or_missing_prediction_to_geometry"
            else:
                chosen = maxmin_vector_indices(predictions, count, eligible=eligible,
                                               reference_indices=sorted(used))
        elif arm == "best_min_margin":
            eligible = [i for i in available if bool(valid_pred[i])]
            if predictions is None or len(eligible) != len(available):
                chosen = _fallback_geometry(candidates, count, used)
                fallback = "invalid_or_missing_prediction_to_geometry"
            else:
                chosen = sorted(eligible, key=lambda i: (-float(np.min(predictions[i])), i))[:count]
        else:
            raise ValueError(f"unknown selection arm {arm}")
        if len(chosen) != count:
            raise ValueError(f"arm {arm} selected {len(chosen)}/{count}")
        for i in chosen:
            used.add(i)
            row = dict(candidates[i])
            row["selection_arm"] = arm
            row["fallback"] = fallback
            row["prediction_valid"] = bool(valid_pred[i]) if predictions is not None else False
            row["prediction"] = (predictions[i].astype(float).tolist()
                                 if predictions is not None and bool(valid_pred[i]) else None)
            row["predicted_uncertainty"] = (float(uncertainty[i]) if uncertainty is not None and
                                             np.isfinite(uncertainty[i]) else None)
            selected.append(row)
    return selected


def _dual_first(seed_rows: Sequence[Mapping[str, Any]], arms: Mapping[str, int],
                seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    selected, used_hashes = [], set()
    for arm, count in arms.items():
        if arm == "random":
            pool = [r for r in seed_rows if str(r.get("source_group", r.get("role", ""))) == arm]
            if len(pool) < count:
                # Random means random among explicitly supplied eligible seeds; it
                # does not authorize history discovery.
                pool = [r for r in seed_rows if pattern_sha256(r["pattern"]) not in used_hashes]
            order = list(rng.permutation(len(pool)))[:count]
            chosen = [pool[i] for i in order]
        else:
            pool = [r for r in seed_rows if str(r.get("source_group", r.get("role", ""))) == arm]
            if len(pool) < count:
                raise ValueError(f"dual first batch needs {count} explicit {arm} seeds")
            idx = maxmin_hamming_indices([r["pattern"] for r in pool], count)
            chosen = [pool[i] for i in idx]
        if len(chosen) < count:
            raise ValueError(f"dual first arm {arm} has only {len(chosen)}/{count}")
        for src in chosen:
            pattern = np.asarray(src["pattern"], dtype=bool).reshape(25, 25)
            if not (pattern[:5, 10:15].all() and pattern[20:25, 10:15].all()):
                raise ValueError(f"dual explicit seed {src.get('id')} lacks required port pads")
            key = pattern_sha256(pattern)
            if key in used_hashes:
                raise ValueError("dual first-batch seed appears in multiple arms")
            used_hashes.add(key)
            selected.append({**dict(src), "pattern": pattern,
                             "selection_arm": arm, "fallback": None,
                             "prediction_valid": False, "prediction": None,
                             "predicted_uncertainty": None, "pattern_sha256": key,
                             "lineage_id": str(src.get("lineage_id") or key),
                             "source": str(src.get("source", "explicit_seed")), "seed": seed})
    return selected


def _batch_dir(work_dir: Path, batch: int) -> Path:
    return work_dir / "batches" / f"batch-{batch:03d}_input"


def write_batch_bundle(work_dir: Path, batch: int, rows: Sequence[Mapping[str, Any]], *,
                       model_hash: str | None) -> Path:
    state = _read_json(work_dir / "state.json")
    cfg = _load_yaml(work_dir / "config.yaml")
    out = _batch_dir(work_dir, batch)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing batch bundle {out}")
    out.mkdir(parents=True, exist_ok=False)
    manifest = []
    for ordinal, source in enumerate(rows):
        pattern = np.asarray(source["pattern"], dtype=bool).reshape(25, 25)
        if cfg["exploration"]["mode"] == "single_fullcurve":
            pattern = enforce_single_geometry(pattern)
        else:
            pattern = enforce_dual_geometry(pattern)
        sha = pattern_sha256(pattern)
        pid = f"{state['id_prefix']}_b{batch:02d}_{ordinal:03d}_{sha[:8]}"
        torch.save(torch.as_tensor(pattern, dtype=torch.float32), out / f"{pid}.pt")
        clean = {k: v for k, v in dict(source).items() if k != "pattern" and k != "seed_input"}
        clean.update({"id": pid, "batch": batch, "pattern_file": f"{pid}.pt",
                      "pattern_sha256": sha, "model_hash": model_hash,
                      "port": cfg["measurement"]["port"],
                      "measurement_id": state["measurement_id"],
                      "score_spec_id": state["score_spec_id"],
                      "geometry_profile": cfg["exploration"]["geometry_profile"]})
        manifest.append(clean)
    _atomic_json(out / "manifest.json", manifest)
    _atomic_json(out / "measurement.json", _read_json(work_dir / "measurement.json"))
    _atomic_json(out / "score_spec.json", _read_json(work_dir / "score_spec.json"))
    shutil.copy2(work_dir / "config.yaml", out / "config.yaml")
    state["batches"][str(batch)] = {"input_dir": out.relative_to(work_dir).as_posix(),
                                     "selected": len(manifest), "model_hash": model_hash,
                                     "feedback_audit": None, "trained_model_dir": None}
    state["next_batch"] = batch + 1
    _atomic_json(work_dir / "state.json", state)
    return out


def select_batch(work_dir: Path, batch: int, seed_inputs: Sequence[Path] = ()) -> Path:
    cfg = validate_exploration_config(_load_yaml(work_dir / "config.yaml"))
    exp = cfg["exploration"]
    state = _read_json(work_dir / "state.json")
    if batch < 1 or batch > exp["max_batches"]:
        raise ValueError("batch is outside configured budget")
    if str(batch) in state["batches"]:
        return work_dir / state["batches"][str(batch)]["input_dir"]
    if batch > 1 and not state["batches"].get(str(batch - 1), {}).get("trained_model_dir"):
        raise ValueError("previous batch must be fed back and trained first")
    saved_seed_paths = [(Path(p) if Path(p).is_absolute() else work_dir / p)
                        for p in state.get("seed_inputs", [])]
    seeds = load_seed_rows(list(seed_inputs) or saved_seed_paths)
    mode = exp["mode"]
    if mode == "dual_margin5" and batch == 1:
        rows = _dual_first(seeds, exp["arms"], exp["seed"])
        return write_batch_bundle(work_dir, batch, rows, model_hash=None)
    candidate_seed = exp["seed"] + batch * 1009
    current = load_scoped_parent_rows(work_dir) if batch > 1 else []
    parents = mixed_diverse_parents(seeds, current)
    candidates = generate_candidates(parents, exp["candidate_pool_size"], seed=candidate_seed,
                                     single=(mode == "single_fullcurve"))
    predictions = uncertainty = None
    model_hash = None
    if batch > 1:
        predictions, uncertainty, model_hash = predict_ensemble(work_dir, batch - 1,
                                                                [c["pattern"] for c in candidates])
    arms = exp["arms"] if batch == 1 else exp["later_arms"]
    rows = select_candidate_arms(candidates, arms, seed=candidate_seed + 1, mode=mode,
                                 predictions=predictions, uncertainty=uncertainty)
    return write_batch_bundle(work_dir, batch, rows, model_hash=model_hash)


def prepare_run(config_path: Path, dataset_root: Path, work_dir: Path,
                seed_inputs: Sequence[Path]) -> Path:
    cfg = validate_exploration_config(_load_yaml(config_path))
    measurement, score, mid, sid, freqs = _measurement_api(cfg["measurement"], cfg["score_spec"])
    mode = cfg["exploration"]["mode"]
    if mode == "single_fullcurve":
        if (measurement["port"] != "single" or len(measurement["labels"]) != 2 or
                len(freqs) != 17):
            raise ValueError("single_fullcurve requires single port, two labels, and 17 sweep points")
        geometry = measurement["geometry"]
        if geometry["pixel_count"] != 25 or float(geometry.get("diag_bridge_w") or -1) != 0.1:
            raise ValueError("single_fullcurve requires 25 pixels and fixed diag_bridge_w=0.1 mm")
    else:
        if measurement["port"] != "dual" or len(freqs) != 49 or len(score["bands"]) != 5:
            raise ValueError("dual_margin5 requires dual port, 49 sweep points, and five score bands")
    cfg["measurement"], cfg["score_spec"] = measurement, score
    cfg_hash = content_id(cfg)
    dataset_root = dataset_root.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"explicit dataset root does not exist: {dataset_root}")
    if work_dir.exists() and any(work_dir.iterdir()):
        state_path = work_dir / "state.json"
        if state_path.exists() and _read_json(state_path).get("config_hash") == cfg_hash:
            return select_batch(work_dir, 1)
        raise FileExistsError(f"refusing to overwrite non-empty work dir {work_dir}")
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    _atomic_json(work_dir / "measurement.json", measurement)
    _atomic_json(work_dir / "score_spec.json", score)
    _atomic_json(work_dir / "dataset_manifest.json", [])
    snapped_seeds = snapshot_seed_inputs(work_dir, seed_inputs)
    id_prefix = _safe_id_prefix(cfg["exploration"].get(
        "id_prefix", f"{cfg.get('name', mode)}_{cfg_hash[:8]}"))
    state = {"schema_version": SCHEMA_VERSION, "config_hash": cfg_hash,
             "dataset_root": str(dataset_root), "measurement_id": mid, "score_spec_id": sid,
             "id_prefix": id_prefix,
             "frequencies_ghz": freqs.astype(float).tolist(),
             "seed_inputs": [p.as_posix() for p in snapped_seeds],
             "max_batches": cfg["exploration"]["max_batches"], "measured_budget": 0,
             "next_batch": 1, "batches": {}}
    _atomic_json(work_dir / "state.json", state)
    return select_batch(work_dir, 1)


def _prediction_audit(batch_rows: Sequence[Mapping[str, Any]], actual: Mapping[str, np.ndarray],
                      target_groups: Mapping[str, tuple[int, int]] | None = None) -> dict[str, Any]:
    errors = []
    group_errors = {name: [] for name in (target_groups or {})}
    per_id = []
    for row in batch_rows:
        pred = row.get("prediction")
        if pred is None or row["id"] not in actual:
            per_id.append({"id": row["id"], "prediction_available": False})
            continue
        p = np.asarray(pred, dtype=np.float32).reshape(-1)
        y = np.asarray(actual[row["id"]], dtype=np.float32).reshape(-1)
        if p.shape != y.shape or not np.all(np.isfinite(p)) or not np.all(np.isfinite(y)):
            per_id.append({"id": row["id"], "prediction_available": True,
                           "valid_for_error": False})
            continue
        mae = float(np.mean(np.abs(p - y)))
        errors.append(mae)
        for name, (start, stop) in (target_groups or {}).items():
            group_errors[name].append(float(np.mean(np.abs(p[start:stop] - y[start:stop]))))
        per_id.append({"id": row["id"], "prediction_available": True,
                       "valid_for_error": True, "mae": mae})
    return {"order": "saved_predictions_before_training_new_batch",
            "n_with_predictions": len(errors),
            "mean_absolute_error": float(np.mean(errors)) if errors else None,
            "group_mean_absolute_error": {
                name: (float(np.mean(values)) if values else None)
                for name, values in group_errors.items()},
            "items": per_id}


def import_feedback(work_dir: Path, batch: int, result_manifest: Path,
                    dataset_root_override: Path | None = None) -> Path:
    cfg = validate_exploration_config(_load_yaml(work_dir / "config.yaml"))
    state = _read_json(work_dir / "state.json")
    info = state["batches"].get(str(batch))
    if not info:
        raise ValueError(f"batch {batch} was not prepared")
    if info.get("feedback_audit"):
        return work_dir / info["feedback_audit"]
    dataset_root = (dataset_root_override or Path(state["dataset_root"])).resolve()
    result_manifest = result_manifest.resolve()
    if not _inside(result_manifest, dataset_root):
        raise ValueError("result manifest must be under the explicit dataset root")
    result_root = result_manifest.parent
    from antenna.measurement import measurement_id, score_spec_id
    measurement_marker = result_root / "measurement.json"
    score_marker = result_root / "score_spec.json"
    if (not measurement_marker.is_file() or
            measurement_id(_read_json(measurement_marker)) != state["measurement_id"]):
        raise ValueError("result store measurement.json is missing or mismatched")
    if (not score_marker.is_file() or
            score_spec_id(_read_json(score_marker)) != state["score_spec_id"]):
        raise ValueError("result store score_spec.json is missing or mismatched")
    batch_rows = _manifest_rows(work_dir / info["input_dir"] / "manifest.json")
    result_rows = _result_rows(result_manifest)
    if len({row["id"] for row in result_rows}) != len(result_rows):
        raise ValueError("result manifest contains duplicate ids")
    by_id = {row["id"]: row for row in result_rows}
    expected = {row["id"] for row in batch_rows}
    missing = sorted(expected - set(by_id))
    if missing:
        raise ValueError(f"result manifest misses {len(missing)} batch ids: {missing[:3]}")
    scoped = SampleStore(work_dir / "scoped_store", verbose=False)
    scoped_rad = work_dir / "scoped_rad"; scoped_rad.mkdir(parents=True, exist_ok=True)
    dataset_entries = _manifest_rows(work_dir / "dataset_manifest.json")
    prior_entries = [entry for entry in dataset_entries if int(entry["batch"]) != batch]
    old_batch_entries = [entry for entry in dataset_entries if int(entry["batch"]) == batch]
    if len({entry["id"] for entry in old_batch_entries}) != len(old_batch_entries):
        raise ValueError(f"dataset manifest has duplicate ids in batch {batch}")
    old_by_id = {entry["id"]: entry for entry in old_batch_entries}
    unexpected_old = sorted(set(old_by_id) - expected)
    if unexpected_old:
        raise ValueError(f"dataset manifest has unexpected batch {batch} ids: {unexpected_old[:3]}")
    actual_targets: dict[str, np.ndarray] = {}
    pending: list[dict[str, Any]] = []
    replayed = 0
    input_dir = work_dir / info["input_dir"]
    mode = cfg["exploration"]["mode"]
    n_freq = len(state["frequencies_ghz"])
    for selected in batch_rows:
        row = by_id[selected["id"]]
        if row.get("status") != "ok":
            raise ValueError(f"feedback {row.get('id')} is not status=ok")
        for key in ("measurement_id", "score_spec_id"):
            if row.get(key) != state[key]:
                raise ValueError(f"feedback {row['id']} {key} mismatch")
        if (row.get("geometry_profile") is not None and
                row["geometry_profile"] != cfg["exploration"]["geometry_profile"]):
            raise ValueError(f"feedback {row['id']} geometry_profile mismatch")
        if (row.get("geom") != cfg["measurement"]["geometry"]["geom"] or
                not np.isclose(float(row.get("dbw")),
                               float(cfg["measurement"]["geometry"]["diag_bridge_w"]),
                               rtol=0.0, atol=1e-12)):
            raise ValueError(f"feedback {row['id']} geometry identity mismatch")
        source_path = _safe_input_path(result_root, row["sample_file"])
        if row.get("sample_sha256") != file_sha256(source_path):
            raise ValueError(f"feedback sample hash mismatch for {row['id']}")
        source_sample_sha = row["sample_sha256"]
        x, response = torch.load(source_path, weights_only=True)
        x = torch.as_tensor(x, dtype=torch.float32).reshape(25, 25)
        response = torch.as_tensor(response, dtype=torch.float32)
        expected_pattern = torch.load(input_dir / selected["pattern_file"], weights_only=True)
        if (pattern_sha256(x) != selected["pattern_sha256"] or
                not np.array_equal(np.asarray(x), np.asarray(expected_pattern))):
            raise ValueError(f"feedback pattern mismatch for {row['id']}")
        labels = row.get("labels", row.get("response_labels", cfg["measurement"]["labels"]))
        row_freqs = row.get("freqs", row.get("response_freqs_ghz", state["frequencies_ghz"]))
        if list(labels) != list(cfg["measurement"]["labels"]):
            raise ValueError(f"feedback labels mismatch for {row['id']}")
        if not np.allclose(row_freqs, state["frequencies_ghz"], rtol=0.0, atol=1e-9):
            raise ValueError(f"feedback frequencies mismatch for {row['id']}")
        if response.shape != (len(labels), n_freq):
            raise ValueError(f"feedback response shape mismatch for {row['id']}")
        if not torch.isfinite(response).all():
            raise ValueError(f"feedback response is non-finite for {row['id']}")
        rad_rel = None
        source_rad_sha = None
        if mode == "single_fullcurve":
            rad_src = _safe_input_path(result_root, row.get("rad_file", f"rad/{row['id']}.pt"))
            if row.get("rad_sha256") != file_sha256(rad_src):
                raise ValueError(f"feedback radiation hash mismatch for {row['id']}")
            source_rad_sha = row["rad_sha256"]
            rad = torch.load(rad_src, weights_only=True)
            rv = radiation_vector(rad)
            rad_name = f"{row['id']}_{hashlib.sha256(rv.tobytes()).hexdigest()[:16]}.pt"
            scoped_rad_path = scoped_rad / rad_name
            if scoped_rad_path.exists():
                existing_rad = torch.load(scoped_rad_path, weights_only=True)
                if not np.array_equal(radiation_vector(existing_rad), rv):
                    raise ValueError(f"scoped radiation content mismatch for {row['id']}")
            else:
                _atomic_torch(scoped_rad_path, rad)
            rad_rel = scoped_rad_path.relative_to(work_dir).as_posix()
            target = np.concatenate([response.numpy().reshape(-1), rv])
        else:
            target = _score_dual(response, cfg["measurement"], cfg["score_spec"],
                                 labels, row_freqs)
        actual_targets[row["id"]] = target
        added = scoped.add(x, response)
        sample_fp = fingerprint(x, response)
        scoped_file = work_dir / "scoped_store" / f"{sample_fp}.pt"
        if not scoped_file.is_file():
            raise ValueError(f"scoped sample is missing after import: {row['id']}")
        scoped_x, scoped_y = torch.load(scoped_file, weights_only=True)
        if fingerprint(scoped_x, scoped_y) != sample_fp:
            raise ValueError(f"scoped sample content mismatch for {row['id']}")
        if any(entry["id"] == row["id"] for entry in prior_entries):
            raise ValueError(f"feedback id already belongs to another batch: {row['id']}")
        if any(entry["sample_fingerprint"] == sample_fp for entry in prior_entries):
            raise ValueError(f"feedback sample duplicates another batch: {row['id']}")
        if any(entry["id"] == row["id"] for entry in pending):
            raise ValueError(f"feedback id is duplicated within batch: {row['id']}")
        if any(entry["sample_fingerprint"] == sample_fp for entry in pending):
            raise ValueError(f"feedback sample is duplicated within batch: {row['id']}")
        new_entry = {"id": row["id"], "batch": batch,
                     "sample_fingerprint": sample_fp,
                     "scoped_sample_file": scoped_file.relative_to(work_dir).as_posix(),
                     "scoped_sample_sha256": file_sha256(scoped_file),
                     "source_sample_sha256": source_sample_sha,
                     "scoped_rad_file": rad_rel,
                     "scoped_rad_sha256": (file_sha256(work_dir / rad_rel) if rad_rel else None),
                     "source_rad_sha256": source_rad_sha,
                     "lineage_id": str(row.get("lineage_id") or selected.get("lineage_id") or
                                       row["pattern_sha256"]),
                     "pattern_sha256": row["pattern_sha256"],
                     "geometry_profile": cfg["exploration"]["geometry_profile"],
                     "measurement_id": state["measurement_id"],
                     "score_spec_id": state["score_spec_id"],
                     "response_labels": labels,
                     "response_freqs_ghz": row_freqs,
                     "source_manifest": str(result_manifest.resolve())}
        old_entry = old_by_id.get(row["id"])
        if old_entry is not None:
            stable_keys = ("id", "batch", "sample_fingerprint", "scoped_sample_file",
                           "scoped_sample_sha256", "source_sample_sha256", "scoped_rad_file",
                           "scoped_rad_sha256", "source_rad_sha256", "lineage_id",
                           "pattern_sha256", "geometry_profile", "measurement_id",
                           "score_spec_id", "response_labels", "response_freqs_ghz")
            if any(old_entry.get(key) != new_entry.get(key) for key in stable_keys):
                raise ValueError(f"partial feedback replay changed content for {row['id']}")
            replayed += 1
        elif not added:
            # A crash may have persisted the scoped file before dataset_manifest.
            # Its content was verified above, so it is safe to finish this batch.
            replayed += 1
        pending.append(new_entry)
    if mode == "single_fullcurve":
        target_groups = {"S11": (0, n_freq), "Gain": (n_freq, 2 * n_freq),
                         "phi0": (2 * n_freq, 2 * n_freq + 91),
                         "phi90": (2 * n_freq + 91, 2 * n_freq + 182)}
    else:
        target_groups = {band["name"]: (i, i + 1)
                         for i, band in enumerate(cfg["score_spec"]["bands"])}
    audit = _prediction_audit(batch_rows, actual_targets, target_groups)
    audit.update({"batch": batch, "n_imported": len(pending),
                  "n_idempotent_replayed": replayed,
                  "truth_used_for_selection": False,
                  "single_performance_filter_or_optimization": False if mode == "single_fullcurve" else None})
    audit_path = work_dir / "feedback" / f"batch-{batch:03d}-pretrain-audit.json"
    _atomic_json(audit_path, audit)  # Deliberately written before exposing new data to training.
    final_entries = prior_entries + pending
    _atomic_json(work_dir / "dataset_manifest.json", final_entries)
    state = _read_json(work_dir / "state.json")
    state["batches"][str(batch)]["feedback_audit"] = audit_path.relative_to(work_dir).as_posix()
    state["measured_budget"] = len(final_entries)
    _atomic_json(work_dir / "state.json", state)
    return audit_path


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--seed-input", type=Path, action="append", default=[])
    p.add_argument("--work-dir", type=Path, required=True)
    p = sub.add_parser("feedback")
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--dataset-root", type=Path, required=True,
                   help="explicit root containing the completed profiled store")
    p.add_argument("--result-manifest", type=Path, required=True)
    p = sub.add_parser("train")
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--through-batch", type=int, required=True)
    p = sub.add_parser("select-batch")
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--seed-input", type=Path, action="append", default=[])
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _require_cli_output(args.work_dir)
    if args.command == "prepare":
        out = prepare_run(args.config, args.dataset_root, args.work_dir, args.seed_input)
    elif args.command == "feedback":
        out = import_feedback(args.work_dir, args.batch, args.result_manifest, args.dataset_root)
    elif args.command == "train":
        out = train_models(args.work_dir, args.through_batch)
    else:
        out = select_batch(args.work_dir, args.batch, args.seed_input)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
