# -*- coding: utf-8 -*-
"""Versioned, local-only training for the single-port symmetry campaign.

The module deliberately has no dataset discovery and no HFSS entry point.  A
caller supplies completed, hash-audited profile stores.  Each accepted chunk
creates an immutable cumulative data manifest, and each model version is
freshly initialized.  Historical data may be registered as an explicitly
provenanced pretraining prior; it never enters current-profile validation.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from antenna.measurement import (
    frequency_grid,
    measurement_id,
    score_spec_id,
    validate_measurement,
    validate_score_spec,
)
from script.exploration import (
    CurveMLP,
    TMP_ROOT,
    TrainingInterrupted,
    _atomic_json,
    _atomic_torch,
    _epoch_order,
    _fit_norm,
    _inside,
    content_id,
    file_sha256,
    pattern_sha256,
    radiation_vector,
)
from script.profiled_batch import verify_completed


SCHEMA_VERSION = 2
OUTPUT_DIM = 216


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _mapping(value: Mapping[str, Any] | str | Path, name: str) -> dict[str, Any]:
    if isinstance(value, (str, Path)):
        value = _read_json(Path(value))
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping or JSON path")
    return dict(value)


def _safe_file(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not _inside(path, root) or not path.is_file():
        raise ValueError(f"unsafe or missing file below {root}: {relative}")
    return path


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if file_sha256(destination) != file_sha256(source):
            raise ValueError(f"immutable cache collision: {destination}")
        return
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, tmp)
    os.replace(tmp, destination)


def fixed_group_split(lineage_id: str, fraction: float, seed: int) -> str:
    """Assign a lineage permanently, independent of arrival order/data size."""
    if not 0.0 < float(fraction) < 1.0:
        raise ValueError("holdout_fraction must be between zero and one")
    digest = content_id({"split_seed": int(seed), "lineage_id": str(lineage_id)})
    value = int(digest, 16) / float(1 << 256)
    return "holdout" if value < float(fraction) else "train"


def _protocol(work_dir: Path) -> dict[str, Any]:
    value = _read_json(work_dir / "protocol.json")
    if content_id({k: v for k, v in value.items() if k != "protocol_id"}) != value["protocol_id"]:
        raise ValueError("protocol hash mismatch")
    return value


def _hex64(value: Any, name: str) -> str:
    if (not isinstance(value, str) or len(value) != 64 or
            any(char not in "0123456789abcdefABCDEF" for char in value)):
        raise ValueError(f"{name} must be 64 hexadecimal characters")
    return value.lower()


def _canonical_group(group_id: str, aliases: Mapping[str, str]) -> str:
    current = str(group_id)
    seen: set[str] = set()
    while current in aliases and str(aliases[current]) != current:
        if current in seen:
            raise ValueError("lineage alias cycle")
        seen.add(current)
        current = str(aliases[current])
    return current


def _profile_role(canonical_group_id: str, protocol: Mapping[str, Any],
                  aliases: Mapping[str, str]) -> dict[str, str]:
    """Return the immutable effective/reference role for one canonical group."""
    canonical = _canonical_group(canonical_group_id, aliases)
    reference = "profile_" + fixed_group_split(
        canonical, protocol["split"]["fraction"], protocol["split"]["seed"])
    declared = protocol.get("development_train_groups")
    if declared is None:
        return {"split": reference}
    development = {_canonical_group(str(group), aliases) for group in declared}
    effective = "profile_train" if canonical in development else reference
    return {"split": effective, "reference_split": reference}


def _assign_profile_role(row: dict[str, Any], protocol: Mapping[str, Any],
                         aliases: Mapping[str, str]) -> None:
    row.pop("reference_split", None)
    row.update(_profile_role(str(row["canonical_group_id"]), protocol, aliases))


def _validate_development_aliases(protocol: Mapping[str, Any],
                                  aliases: Mapping[str, str]) -> None:
    if "development_train_groups" not in protocol:
        return
    initial = dict(protocol["lineage_aliases"]["mapping"])
    frozen_roots = set(initial.values()) | set(protocol["development_train_groups"])
    for root in frozen_roots:
        if _canonical_group(str(root), aliases) != root:
            raise ValueError("protocol canonical alias root was retargeted in state")
    for lineage, root in initial.items():
        if _canonical_group(str(lineage), aliases) != root:
            raise ValueError("protocol lineage alias was retargeted in state")


def _validate_profile_roles(rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any],
                            aliases: Mapping[str, str]) -> None:
    effective_by_group: dict[str, str] = {}
    opt_in = "development_train_groups" in protocol
    for row in rows:
        recorded_canonical = row.get("canonical_group_id")
        if not isinstance(recorded_canonical, str) or not recorded_canonical:
            raise ValueError("profile row lacks a canonical group")
        canonical = _canonical_group(recorded_canonical, aliases)
        if opt_in:
            lineage = row.get("lineage_id")
            if not isinstance(lineage, str) or not lineage:
                raise ValueError("development profile row lacks a lineage identifier")
            if recorded_canonical != canonical:
                raise ValueError("profile canonical group is not a state alias root")
            if _canonical_group(lineage, aliases) != canonical:
                raise ValueError("profile lineage and canonical group closure mismatch")
        expected = _profile_role(canonical, protocol, aliases)
        actual = {"split": row.get("split")}
        if opt_in:
            actual["reference_split"] = row.get("reference_split")
        elif "reference_split" in row:
            raise ValueError("legacy profile row unexpectedly contains a reference split")
        if actual != expected:
            raise ValueError(f"profile role proof mismatch for {row.get('id', '<unknown>')}")
        prior = effective_by_group.setdefault(canonical, str(row["split"]))
        if prior != row["split"]:
            raise ValueError("one canonical group spans effective train and holdout roles")


def _development_metadata(rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any],
                          aliases: Mapping[str, str]) -> dict[str, Any] | None:
    if "development_train_groups" not in protocol:
        return None
    development = {_canonical_group(str(group), aliases)
                   for group in protocol["development_train_groups"]}
    represented = {_canonical_group(str(row["canonical_group_id"]), aliases) for row in rows}
    missing = sorted(development - represented)
    if missing:
        raise ValueError(f"declared development groups are absent from current manifest: {missing}")
    development_rows = [row for row in rows
                        if _canonical_group(str(row["canonical_group_id"]), aliases) in development]
    reference_counts = Counter(str(row["reference_split"]) for row in rows)
    effective_counts = Counter(str(row["split"]) for row in rows)
    return {
        "role_policy": dict(protocol["development_role_policy"]),
        "reference_protocol_id": protocol["reference_protocol_id"],
        "development_train_groups": list(protocol["development_train_groups"]),
        "counts": {
            "development_rows": len(development_rows),
            "development_reference_holdout_rows": sum(
                row["reference_split"] == "profile_holdout" for row in development_rows),
            "effective_profile_train": effective_counts["profile_train"],
            "effective_profile_holdout": effective_counts["profile_holdout"],
            "reference_profile_train": reference_counts["profile_train"],
            "reference_profile_holdout": reference_counts["profile_holdout"],
        },
    }


def initialize_workdir(
    work_dir: str | Path,
    measurement: Mapping[str, Any] | str | Path,
    score_spec: Mapping[str, Any] | str | Path,
    *,
    geometry_profile: str,
    holdout_seed: int = 20261007,
    holdout_fraction: float = 0.2,
    ensemble_seeds: Sequence[int] = (0, 1),
    hidden_dims: Sequence[int] = (128, 128),
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    legacy_epochs: int = 10,
    current_epochs: int = 100,
    update_min: int = 48,
    update_max: int = 96,
    initial_lineage_aliases: Mapping[str, str] | None = None,
    alias_source_sha256: str | None = None,
    development_train_groups: Sequence[str] = (),
    reference_protocol_id: str | None = None,
) -> Path:
    """Create or verify a scoped training work directory."""
    root = Path(work_dir).resolve()
    m = validate_measurement(_mapping(measurement, "measurement"))
    s = validate_score_spec(_mapping(score_spec, "score_spec"), measurement=m)
    freqs = frequency_grid(m)
    if m["port"] != "single" or len(m["labels"]) != 2 or len(freqs) != 17:
        raise ValueError("symmetry training requires a single-port 2x17 response profile")
    if m["geometry"]["geom"] != geometry_profile:
        raise ValueError("geometry_profile must equal measurement.geometry.geom")
    seeds = [int(x) for x in ensemble_seeds]
    hidden = [int(x) for x in hidden_dims]
    if not seeds or len(set(seeds)) != len(seeds) or not hidden or min(hidden) <= 0:
        raise ValueError("ensemble seeds must be unique and hidden dimensions positive")
    if learning_rate <= 0 or batch_size <= 0 or legacy_epochs < 0 or current_epochs <= 0:
        raise ValueError("invalid training settings")
    if update_min <= 0 or update_max < update_min:
        raise ValueError("invalid update range")
    aliases = {str(key): str(value) for key, value in (initial_lineage_aliases or {}).items()}
    if any(not key or not value for key, value in aliases.items()):
        raise ValueError("initial lineage aliases must contain non-empty strings")
    if any(value in aliases and value != key for key, value in aliases.items()):
        raise ValueError("initial lineage aliases must be one-level mappings to canonical roots")
    if aliases:
        if (not isinstance(alias_source_sha256, str) or len(alias_source_sha256) != 64 or
                any(char not in "0123456789abcdefABCDEF" for char in alias_source_sha256)):
            raise ValueError("a 64-hex alias_source_sha256 is required with initial aliases")
    elif alias_source_sha256 is not None:
        raise ValueError("alias_source_sha256 requires initial_lineage_aliases")
    if isinstance(development_train_groups, (str, bytes)):
        raise TypeError("development_train_groups must be a sequence of group identifiers")
    if any(not isinstance(group, str) or not group for group in development_train_groups):
        raise ValueError("development_train_groups must contain non-empty strings")
    development = sorted({_canonical_group(group, aliases)
                          for group in development_train_groups})
    if development:
        reference_protocol_id = _hex64(reference_protocol_id, "reference_protocol_id")
    elif reference_protocol_id is not None:
        raise ValueError("reference_protocol_id requires development_train_groups")
    protocol: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "measurement_id": measurement_id(m),
        "score_spec_id": score_spec_id(s),
        "geometry_profile": geometry_profile,
        "frequencies_ghz": freqs.tolist(),
        "output_dim": OUTPUT_DIM,
        "split": {"method": "sha256_lineage_threshold_v1", "seed": int(holdout_seed),
                  "fraction": float(holdout_fraction)},
        "model": {"ensemble_seeds": seeds, "hidden_dims": hidden,
                  "learning_rate": float(learning_rate), "batch_size": int(batch_size),
                  "legacy_epochs": int(legacy_epochs), "current_epochs": int(current_epochs)},
        "updates": {"min_new_valid": int(update_min), "max_new_valid": int(update_max)},
        "lineage_aliases": {"mapping": dict(sorted(aliases.items())),
                            "source_sha256": alias_source_sha256},
        "policy": {"fresh_initialization_each_version": True,
                   "legacy_usage": "pretraining_prior_only",
                   "validation": "current_profile_holdout_only",
                   "early_stopping": False},
    }
    if development:
        protocol.update({
            "development_train_groups": development,
            "reference_protocol_id": reference_protocol_id,
            "development_role_policy": {
                "name": "whole_canonical_group_train_override_v1",
                "reference_split_method": "sha256_lineage_threshold_v1",
                "future_same_group_labels": "profile_train",
                "validation_scope": "remaining_reference_group_holdouts_descriptive_only",
            },
        })
    protocol["protocol_id"] = content_id(protocol)
    if (root / "protocol.json").exists():
        if _read_json(root / "protocol.json") != protocol:
            raise ValueError("work directory already has a different immutable protocol")
        return root
    if root.exists() and any(root.iterdir()):
        raise ValueError("new work directory must be empty")
    root.mkdir(parents=True, exist_ok=True)
    _atomic_json(root / "measurement.json", m)
    _atomic_json(root / "score_spec.json", s)
    _atomic_json(root / "protocol.json", protocol)
    _atomic_json(root / "state.json", {"schema_version": SCHEMA_VERSION,
                 "protocol_id": protocol["protocol_id"], "latest_data_version": 0,
                 "chunks": [], "lineage_aliases": dict(sorted(aliases.items()))})
    return root


def _target(sample_path: Path, rad_path: Path, frequencies: Sequence[float]) -> tuple[torch.Tensor, torch.Tensor]:
    pattern, response = torch.load(sample_path, weights_only=True, map_location="cpu")
    x = torch.as_tensor(pattern, dtype=torch.float32).reshape(25, 25)
    pattern_sha256(x)
    y = torch.as_tensor(response, dtype=torch.float32)
    if tuple(y.shape) != (2, len(frequencies)) or not torch.isfinite(y).all():
        raise ValueError(f"incompatible response in {sample_path}")
    rad = torch.load(rad_path, weights_only=True, map_location="cpu")
    target = torch.cat((y.reshape(-1), torch.as_tensor(radiation_vector(rad))))
    if target.numel() != OUTPUT_DIM or not torch.isfinite(target).all():
        raise ValueError(f"invalid target in {sample_path}")
    return x, target.float()


def _latest_rows(root: Path, version: int) -> list[dict[str, Any]]:
    if version == 0:
        return []
    value = _read_json(root / "data" / f"data-v{version:03d}" / "manifest.json")
    if not isinstance(value, list):
        raise ValueError("data manifest must be a list")
    return value


def ingest_profile_stores(
    work_dir: str | Path,
    stores: Sequence[str | Path],
    *,
    min_new: int | None = None,
    max_new: int | None = None,
) -> Path:
    """Validate explicit completed stores and create one cumulative data version."""
    root = Path(work_dir).resolve()
    protocol = _protocol(root)
    state = _read_json(root / "state.json")
    low = protocol["updates"]["min_new_valid"] if min_new is None else int(min_new)
    high = protocol["updates"]["max_new_valid"] if max_new is None else int(max_new)
    if low <= 0 or high < low or not stores:
        raise ValueError("invalid update bounds or empty store list")
    prior = _latest_rows(root, int(state["latest_data_version"]))
    by_id = {r["id"]: r for r in prior}
    aliases = dict(state.get("lineage_aliases", {}))
    _validate_development_aliases(protocol, aliases)
    _validate_profile_roles(prior, protocol, aliases)
    pattern_groups = {r["pattern_sha256"]: r["canonical_group_id"] for r in prior}
    frozen_groups = set(aliases.values()) | set(pattern_groups.values())
    known_unique = {r["pattern_sha256"] for r in prior}
    new_unique: set[str] = set()
    pending: list[tuple[dict[str, Any], Path, Path]] = []
    chunk_fingerprints: list[str] = []
    measurement = _read_json(root / "measurement.json")
    score = _read_json(root / "score_spec.json")
    cfg = SimpleNamespace(port="single", measurement=measurement, score_spec=score)
    for supplied in stores:
        store = Path(supplied).resolve()
        if measurement_id(_read_json(store / "measurement.json")) != protocol["measurement_id"]:
            raise ValueError(f"measurement profile mismatch: {store}")
        if score_spec_id(_read_json(store / "score_spec.json")) != protocol["score_spec_id"]:
            raise ValueError(f"score profile mismatch: {store}")
        results = verify_completed(store, cfg, require_complete=True)
        rows = {r["id"]: r for r in _read_json(store / "manifest.json")}
        store_id = content_id({"manifest": file_sha256(store / "manifest.json"),
                               "results": file_sha256(store / "results.json")})
        chunk_fingerprints.append(store_id)
        for name, entry in sorted(results.items()):
            source_sample = _safe_file(store, entry["sample_file"])
            source_rad = _safe_file(store, entry["rad_file"])
            x, _ = _target(source_sample, source_rad, protocol["frequencies_ghz"])
            if pattern_sha256(x) != entry["pattern_sha256"]:
                raise ValueError(f"pattern mismatch for {name}")
            uid = content_id({"measurement_id": protocol["measurement_id"],
                              "completed_store_id": store_id, "observation_id": name})
            source_row = rows[name]
            raw_lineage = str(source_row.get("lineage_id") or source_row.get("ancestry_id") or
                              source_row.get("root_parent_id") or source_row.get("source_id") or
                              entry.get("lineage_id") or uid)
            existing_group = pattern_groups.get(entry["pattern_sha256"])
            aliased_group = aliases.get(raw_lineage)
            if existing_group is not None and aliased_group not in (None, existing_group):
                if existing_group in frozen_groups and aliased_group in frozen_groups:
                    raise ValueError("duplicate pattern would merge two already-fixed lineage groups")
                if existing_group in frozen_groups:
                    canonical, merged = existing_group, aliased_group
                elif aliased_group in frozen_groups:
                    canonical, merged = aliased_group, existing_group
                else:
                    canonical, merged = min(existing_group, aliased_group), max(existing_group, aliased_group)
                aliases = {key: (canonical if value == merged else value)
                           for key, value in aliases.items()}
                if "development_train_groups" in protocol:
                    aliases[merged] = canonical
                pattern_groups = {key: (canonical if value == merged else value)
                                  for key, value in pattern_groups.items()}
                for prior_pending, _sample, _rad in pending:
                    if prior_pending["canonical_group_id"] == merged:
                        prior_pending["canonical_group_id"] = canonical
                        _assign_profile_role(prior_pending, protocol, aliases)
                existing_group = canonical
                aliased_group = canonical
            canonical_group = existing_group or aliased_group or raw_lineage
            aliases[raw_lineage] = canonical_group
            pattern_groups[entry["pattern_sha256"]] = canonical_group
            sample_hash, rad_hash = file_sha256(source_sample), file_sha256(source_rad)
            record = {
                "id": uid, "source_observation_id": name,
                "source_store": by_id.get(uid, {}).get("source_store", str(store)),
                "source_sample_sha256": sample_hash, "source_rad_sha256": rad_hash,
                "sample_file": f"profile/samples/{sample_hash}.pt",
                "rad_file": f"profile/rad/{rad_hash}.pt",
                "pattern_sha256": entry["pattern_sha256"], "lineage_id": raw_lineage,
                "canonical_group_id": canonical_group,
                "measurement_id": protocol["measurement_id"],
                "score_spec_id": protocol["score_spec_id"],
                "geometry_profile": protocol["geometry_profile"],
            }
            _assign_profile_role(record, protocol, aliases)
            if uid in by_id:
                if by_id[uid] != record:
                    raise ValueError(f"observation identity collision: {name}")
                continue
            pending.append((record, source_sample, source_rad))
            if entry["pattern_sha256"] not in known_unique:
                new_unique.add(entry["pattern_sha256"])
    chunk_id = content_id(sorted(chunk_fingerprints))
    prior_chunk = next((c for c in state["chunks"] if c["chunk_id"] == chunk_id), None)
    if prior_chunk is not None:
        return root / "data" / f"data-v{int(prior_chunk['data_version']):03d}"
    if not low <= len(new_unique) <= high:
        raise ValueError(f"chunk has {len(new_unique)} new valid unique patterns "
                         f"({len(pending)} observations); required {low}..{high}")
    for record, source_sample, source_rad in pending:
        _atomic_copy(source_sample, root / record["sample_file"])
        _atomic_copy(source_rad, root / record["rad_file"])
        by_id[record["id"]] = record
    version = int(state["latest_data_version"]) + 1
    version_dir = root / "data" / f"data-v{version:03d}"
    rows_out = sorted(by_id.values(), key=lambda x: x["id"])
    _validate_profile_roles(rows_out, protocol, aliases)
    manifest_hash = content_id(rows_out)
    _atomic_json(version_dir / "manifest.json", rows_out)
    _atomic_json(version_dir / "receipt.json", {
        "schema_version": SCHEMA_VERSION, "data_version": version, "chunk_id": chunk_id,
        "new_valid_unique": len(new_unique), "new_observations": len(pending),
        "cumulative_observations": len(rows_out),
        "cumulative_valid_unique": len({row["pattern_sha256"] for row in rows_out}),
        "manifest_id": manifest_hash, "source_store_count": len(stores),
    })
    state["latest_data_version"] = version
    state["lineage_aliases"] = aliases
    state["chunks"].append({"chunk_id": chunk_id, "data_version": version,
                            "new_valid_unique": len(new_unique),
                            "new_observations": len(pending), "manifest_id": manifest_hash})
    _atomic_json(root / "state.json", state)
    return version_dir


def register_legacy_prior(
    work_dir: str | Path,
    manifest: str | Path,
    dataset_root: str | Path,
    provenance: Mapping[str, Any] | str | Path,
    *,
    max_samples: int | None = None,
) -> Path:
    """Register compatible legacy curves solely as an explicit pretraining prior."""
    root = Path(work_dir).resolve()
    protocol = _protocol(root)
    source_root = Path(dataset_root).resolve()
    manifest_path = Path(manifest).resolve()
    raw = _read_json(manifest_path)
    if not isinstance(raw, list):
        raise ValueError("legacy manifest must be a list")
    prov = _mapping(provenance, "provenance")
    required = {"source_name", "source_geometry_profile", "compatibility_note"}
    if any(not str(prov.get(k, "")).strip() for k in required):
        raise ValueError(f"legacy provenance requires {sorted(required)}")
    source_manifest_sha256 = file_sha256(manifest_path)
    entries: list[tuple[dict[str, Any], Path, Path]] = []
    for index, row in enumerate(raw):
        if row.get("status", "ok") != "ok":
            continue
        sample = _safe_file(source_root, row["sample_file"])
        rad = _safe_file(source_root, row["rad_file"])
        if row.get("sample_sha256") not in (None, file_sha256(sample)):
            raise ValueError(f"legacy sample hash mismatch: {row.get('id', index)}")
        if row.get("rad_sha256") not in (None, file_sha256(rad)):
            raise ValueError(f"legacy radiation hash mismatch: {row.get('id', index)}")
        x, _ = _target(sample, rad, protocol["frequencies_ghz"])
        ph = pattern_sha256(x)  # Deliberately no symmetry constraint for historical prior.
        lineage = str(row.get("lineage_id") or row.get("id") or ph)
        sh, rh = file_sha256(sample), file_sha256(rad)
        entry = {
            "id": content_id({"legacy_manifest": source_manifest_sha256, "row": index,
                              "sample": sh, "rad": rh}),
            "source_id": str(row.get("id", index)), "source_sample_sha256": sh,
            "source_rad_sha256": rh, "sample_file": f"legacy/samples/{sh}.pt",
            "rad_file": f"legacy/rad/{rh}.pt", "pattern_sha256": ph,
            "lineage_id": lineage,
            "source_geometry_profile": str(row.get("geometry_profile") or
                                             prov["source_geometry_profile"]),
            "split": "legacy_" + fixed_group_split(lineage, protocol["split"]["fraction"],
                                                     protocol["split"]["seed"]),
        }
        entries.append((entry, sample, rad))
    entries.sort(key=lambda item: content_id({"seed": protocol["split"]["seed"],
                                               "id": item[0]["id"]}))
    if max_samples is not None:
        if int(max_samples) <= 0:
            raise ValueError("max_samples must be positive")
        entries = entries[:int(max_samples)]
    if not entries:
        raise ValueError("legacy prior is empty")
    legacy_manifest = [x[0] for x in entries]
    provenance_out = dict(prov)
    provenance_out.update({
        "schema_version": SCHEMA_VERSION, "usage": "pretraining_prior_only",
        "current_measurement_id": protocol["measurement_id"],
        "current_geometry_profile": protocol["geometry_profile"],
        "legacy_manifest_sha256": source_manifest_sha256,
        "selected_samples": len(legacy_manifest),
        "nonmirror_patterns_allowed": True,
        "excluded_from_current_validation": True,
    })
    target_manifest = root / "legacy" / "manifest.json"
    if target_manifest.exists():
        if (_read_json(target_manifest) != legacy_manifest or
                _read_json(root / "legacy" / "provenance.json") != provenance_out):
            raise ValueError("a different immutable legacy prior is already registered")
        return target_manifest
    for entry, sample, rad in entries:
        _atomic_copy(sample, root / entry["sample_file"])
        _atomic_copy(rad, root / entry["rad_file"])
    _atomic_json(target_manifest, legacy_manifest)
    _atomic_json(root / "legacy" / "provenance.json", provenance_out)
    return target_manifest


def prepare_legacy_manifest(dataset_root: str | Path, output: str | Path, *,
                            max_samples: int | None = None,
                            max_stores: int | None = None) -> Path:
    """Build a strict read-only inventory from legacy ``store/rad`` pairs.

    A row is accepted only when the named input pattern maps to exactly one
    finite 2x17 response in the result store and its radiation sidecar passes
    the full-curve interpolation checks.  Ambiguous repeats are excluded rather
    than pairing a response and radiation result from different HFSS runs.
    """
    source_root = Path(dataset_root).resolve()
    target = Path(output).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    if max_samples is not None and int(max_samples) <= 0:
        raise ValueError("max_samples must be positive")
    if max_stores is not None and int(max_stores) <= 0:
        raise ValueError("max_stores must be positive")
    accepted: list[dict[str, Any]] = []
    counts = Counter()
    stores = [p for p in sorted(source_root.iterdir())
              if p.is_dir() and not p.name.endswith("_input") and (p / "rad").is_dir()]
    if max_stores is not None:
        stores = stores[:int(max_stores)]
    for store in stores:
        input_dir = source_root / f"{store.name}_input"
        if not input_dir.is_dir():
            counts["missing_input_store"] += 1
            continue
        if (store / "measurement.json").is_file() or (input_dir / "measurement.json").is_file():
            counts["named_profile_store_excluded"] += 1
            continue
        setup_path = (store / "hfss_setup.json" if (store / "hfss_setup.json").is_file()
                      else input_dir / "hfss_setup.json")
        setup = _read_json(setup_path) if setup_path.is_file() else None
        if setup is not None and not isinstance(setup, Mapping):
            counts["invalid_setup_metadata"] += 1
            continue
        if isinstance(setup, Mapping) and setup.get("diag_bridge_w") is not None:
            source_geometry_profile = f"legacy_diag_bridge_w_{float(setup['diag_bridge_w']):g}"
        else:
            source_geometry_profile = "unprofiled_legacy_unknown"
        metadata = {}
        input_manifest = input_dir / "manifest.json"
        if input_manifest.is_file():
            value = _read_json(input_manifest)
            if isinstance(value, list):
                metadata = {str(row.get("id")): row for row in value if isinstance(row, Mapping)}
        response_by_pattern: dict[str, list[tuple[Path, str]]] = {}
        for sample in sorted(store.glob("*.pt")):
            try:
                pattern, response = torch.load(sample, weights_only=True, map_location="cpu")
                ph = pattern_sha256(torch.as_tensor(pattern).reshape(25, 25))
                y = torch.as_tensor(response)
                if tuple(y.shape) != (2, 17) or not torch.isfinite(y).all():
                    raise ValueError("incompatible response")
            except (OSError, RuntimeError, TypeError, ValueError):
                counts["invalid_response_file"] += 1
                continue
            response_by_pattern.setdefault(ph, []).append((sample, file_sha256(sample)))
        for rad in sorted((store / "rad").glob("*.pt")):
            name = rad.stem
            pattern_path = input_dir / f"{name}.pt"
            if not pattern_path.is_file():
                counts["missing_named_pattern"] += 1
                continue
            try:
                pattern = torch.load(pattern_path, weights_only=True, map_location="cpu")
                ph = pattern_sha256(torch.as_tensor(pattern).reshape(25, 25))
                radiation_vector(torch.load(rad, weights_only=True, map_location="cpu"))
            except (OSError, RuntimeError, TypeError, ValueError):
                counts["invalid_pattern_or_radiation"] += 1
                continue
            matches = response_by_pattern.get(ph, [])
            if len(matches) != 1:
                counts["ambiguous_or_missing_response"] += 1
                continue
            sample, sample_hash = matches[0]
            row = metadata.get(name, {})
            lineage = str(row.get("lineage_id") or row.get("source_id") or ph)
            accepted.append({
                "id": f"{store.name}:{name}", "status": "ok", "lineage_id": lineage,
                "sample_file": sample.relative_to(source_root).as_posix(),
                "rad_file": rad.relative_to(source_root).as_posix(),
                "sample_sha256": sample_hash, "rad_sha256": file_sha256(rad),
                "pattern_sha256": ph, "source_store": store.name,
                "geometry_profile": source_geometry_profile,
                "source_hfss_setup": dict(setup) if isinstance(setup, Mapping) else None,
                "source_hfss_setup_sha256": file_sha256(setup_path) if setup_path.is_file() else None,
                "source_metadata": {str(k): v for k, v in row.items()
                                    if k in {"kind", "family", "source_id", "lineage_id",
                                             "geom", "dbw", "diag_bridge_w"}},
            })
            counts["accepted"] += 1
    accepted.sort(key=lambda row: content_id({"legacy_inventory_v1": row["id"],
                                               "sample": row["sample_sha256"],
                                               "rad": row["rad_sha256"]}))
    total_eligible = len(accepted)
    if max_samples is not None:
        accepted = accepted[:int(max_samples)]
    if not accepted:
        raise ValueError("no unambiguous compatible legacy response/radiation pairs")
    if target.exists() and _read_json(target) != accepted:
        raise ValueError("legacy inventory output already exists with different content")
    _atomic_json(target, accepted)
    _atomic_json(target.with_suffix(target.suffix + ".inventory.json"), {
        "schema_version": SCHEMA_VERSION, "dataset_root": str(source_root),
        "stores_scanned": len(stores), "eligible_before_cap": total_eligible,
        "selected": len(accepted), "counts": dict(sorted(counts.items())),
        "selection": "sha256_stable_order_v1",
        "safety": "unique-pattern response match; finite 2x17; valid full radiation cuts",
    })
    return target


def _load_dataset(root: Path, rows: Sequence[Mapping[str, Any]], split: str,
                  frequencies: Sequence[float]) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    xs, ys, kept = [], [], []
    for row in rows:
        if row["split"] != split:
            continue
        sample = _safe_file(root, row["sample_file"])
        rad = _safe_file(root, row["rad_file"])
        if file_sha256(sample) != row["source_sample_sha256"] or file_sha256(rad) != row["source_rad_sha256"]:
            raise ValueError(f"cached input hash mismatch: {row['id']}")
        x, y = _target(sample, rad, frequencies)
        xs.append(x.reshape(-1)); ys.append(y); kept.append(dict(row))
    if not xs:
        return torch.empty((0, 625)), torch.empty((0, OUTPUT_DIM)), []
    return torch.stack(xs), torch.stack(ys), kept


def _weights(rows: Sequence[Mapping[str, Any]]) -> torch.Tensor:
    counts = Counter(str(row.get("canonical_group_id", row["lineage_id"])) for row in rows)
    values = torch.tensor([1.0 / counts[str(row.get("canonical_group_id", row["lineage_id"]))]
                           for row in rows])
    return values * (len(values) / values.sum())


def _metrics(model: CurveMLP, data: tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]],
             input_norm: tuple[torch.Tensor, torch.Tensor] | None,
             target_norm: tuple[torch.Tensor, torch.Tensor] | None) -> dict[str, Any] | None:
    x, y, rows = data
    if not rows or input_norm is None or target_norm is None:
        return None
    model.eval()
    with torch.no_grad():
        pred_norm = model((x - input_norm[0]) / input_norm[1])
        pred = pred_norm * target_norm[1] + target_norm[0]
    return _error_metrics(pred, y, len(rows))


def _error_metrics(pred: torch.Tensor, y: torch.Tensor, n: int) -> dict[str, Any]:
    error = (pred - y).abs()
    return {"n": int(n), "mae": float(error.mean()),
            "s11_mae": float(error[:, :17].mean()), "gain_mae": float(error[:, 17:34].mean()),
            "phi0_mae": float(error[:, 34:125].mean()), "phi90_mae": float(error[:, 125:].mean())}


def train_version(work_dir: str | Path, version: int | None = None, *,
                  interrupt_after_epochs: int | None = None) -> list[Path]:
    """Train/resume one data version on CPU; every new version starts fresh."""
    root = Path(work_dir).resolve()
    protocol = _protocol(root)
    state = _read_json(root / "state.json")
    data_version = int(state["latest_data_version"] if version is None else version)
    if data_version <= 0 or data_version > int(state["latest_data_version"]):
        raise ValueError("unknown data version")
    version_dir = root / "data" / f"data-v{data_version:03d}"
    current_rows = _read_json(version_dir / "manifest.json")
    receipt = _read_json(version_dir / "receipt.json")
    if content_id(current_rows) != receipt["manifest_id"]:
        raise ValueError("data manifest hash mismatch")
    aliases = dict(state.get("lineage_aliases", {}))
    _validate_development_aliases(protocol, aliases)
    _validate_profile_roles(current_rows, protocol, aliases)
    development_metadata = _development_metadata(current_rows, protocol, aliases)
    current_train = _load_dataset(root, current_rows, "profile_train", protocol["frequencies_ghz"])
    current_hold = _load_dataset(root, current_rows, "profile_holdout", protocol["frequencies_ghz"])
    if not current_train[2] or not current_hold[2]:
        raise ValueError("fixed split must contain current-profile train and holdout lineages")
    legacy_rows: list[dict[str, Any]] = []
    if (root / "legacy" / "manifest.json").exists():
        legacy_rows = _read_json(root / "legacy" / "manifest.json")
    hold_patterns = {row["pattern_sha256"] for row in current_hold[2]}
    hold_groups = ({str(row.get("canonical_group_id", row["lineage_id"]))
                    for row in current_hold[2]} |
                   {str(row["lineage_id"]) for row in current_hold[2]})
    legacy_excluded = [row for row in legacy_rows
                       if row["split"] == "legacy_train" and
                       (row["pattern_sha256"] in hold_patterns or
                        str(row["lineage_id"]) in hold_groups)]
    excluded_ids = {row["id"] for row in legacy_excluded}
    usable_legacy_rows = [row for row in legacy_rows if row["id"] not in excluded_ids]
    legacy_train = _load_dataset(root, usable_legacy_rows, "legacy_train", protocol["frequencies_ghz"])
    legacy_hold = _load_dataset(root, legacy_rows, "legacy_holdout", protocol["frequencies_ghz"])
    cfg = protocol["model"]
    phases = []
    if legacy_train[2] and cfg["legacy_epochs"]:
        phases.append(("legacy_pretrain", legacy_train, int(cfg["legacy_epochs"])))
    if cfg["current_epochs"]:
        phases.append(("current_profile", current_train, int(cfg["current_epochs"])))
    if not phases:
        raise ValueError("training has zero epochs")
    input_norm = _fit_norm(current_train[0])
    target_norm = _fit_norm(current_train[1])
    legacy_id = content_id(legacy_rows) if legacy_rows else None
    binding = {
        "measurement_id": protocol["measurement_id"],
        "score_spec_id": protocol["score_spec_id"],
        "protocol_id": protocol["protocol_id"],
        "data_version": data_version,
        "manifest_id": receipt["manifest_id"],
        "cumulative_valid_unique": int(receipt["cumulative_valid_unique"]),
    }
    model_dir = root / "models" / f"data-v{data_version:03d}"
    paths, epochs_this_call = [], 0
    summaries = []
    for seed in cfg["ensemble_seeds"]:
        torch.manual_seed(int(seed)); np.random.seed(int(seed))
        model = CurveMLP(625, cfg["hidden_dims"], OUTPUT_DIM)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
        path = model_dir / f"member-{int(seed)}.pt"
        signature = content_id({"protocol_id": protocol["protocol_id"],
                                "manifest_id": receipt["manifest_id"],
                                "legacy_manifest_id": legacy_id, "member_seed": int(seed)})
        phase_start, epoch_start = 0, 0
        if path.exists():
            saved = torch.load(path, weights_only=False, map_location="cpu")
            if saved.get("signature") != signature or saved.get("data_version") != data_version:
                raise ValueError("checkpoint signature mismatch")
            if saved.get("binding") != binding or int(saved.get("member_seed", -1)) != int(seed):
                raise ValueError("checkpoint factory binding/member mismatch")
            if saved.get("legacy_manifest_id") != legacy_id:
                raise ValueError("checkpoint legacy manifest mismatch")
            if development_metadata is not None:
                if saved.get("development_training") != development_metadata:
                    raise ValueError("checkpoint development role binding mismatch")
            elif "development_training" in saved:
                raise ValueError("default checkpoint unexpectedly has development metadata")
            if saved.get("complete"):
                paths.append(path)
                summaries.append(saved.get("metrics"))
                continue
            model.load_state_dict(saved["model_state"])
            optimizer.load_state_dict(saved["optimizer_state"])
            phase_start, epoch_start = int(saved["phase"]), int(saved["epoch"])
            saved_input_norm = tuple(saved["input_norm"])
            saved_target_norm = tuple(saved["target_norm"])
            if any(not torch.equal(a, b) for a, b in zip(saved_input_norm, input_norm)):
                raise ValueError("checkpoint input normalization mismatch")
            if any(not torch.equal(a, b) for a, b in zip(saved_target_norm, target_norm)):
                raise ValueError("checkpoint target normalization mismatch")
            torch.set_rng_state(saved["torch_rng_state"])
            np.random.set_state(saved["numpy_rng_state"])
        for phase_i, (_name, dataset, epochs) in enumerate(phases):
            if phase_i < phase_start:
                continue
            x, y, rows = dataset
            weights = _weights(rows)
            first = epoch_start if phase_i == phase_start else 0
            for epoch in range(first, epochs):
                order = _epoch_order(len(x), int(seed), phase_i, epoch)
                model.train()
                for start in range(0, len(order), int(cfg["batch_size"])):
                    index = torch.as_tensor(order[start:start + int(cfg["batch_size"])], dtype=torch.long)
                    pred = model((x[index] - input_norm[0]) / input_norm[1])
                    normalized_y = (y[index] - target_norm[0]) / target_norm[1]
                    per_row = (pred - normalized_y).square().mean(dim=1)
                    loss = (per_row * weights[index]).mean()
                    if not torch.isfinite(loss):
                        raise ValueError("non-finite training loss")
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                next_phase, next_epoch = phase_i, epoch + 1
                if next_epoch == epochs:
                    next_phase, next_epoch = phase_i + 1, 0
                complete = next_phase == len(phases)
                phase_names = [p[0] for p in phases]
                metrics = None
                if complete:
                    metrics = {"profile_train": _metrics(model, current_train, input_norm, target_norm),
                               "profile_holdout": _metrics(model, current_hold, input_norm, target_norm),
                               "legacy_holdout": _metrics(model, legacy_hold, input_norm, target_norm)}
                checkpoint = {
                    "schema_version": SCHEMA_VERSION, "signature": signature,
                    "binding": binding,
                    "protocol_id": protocol["protocol_id"], "data_version": data_version,
                    "manifest_id": receipt["manifest_id"], "legacy_manifest_id": legacy_id,
                    "member_seed": int(seed), "fresh_initialization": True,
                    "phase": next_phase, "epoch": next_epoch, "phase_names": phase_names,
                    "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                    "input_norm": tuple(t.cpu() for t in input_norm),
                    "target_norm": tuple(t.cpu() for t in target_norm), "complete": complete,
                    "metrics": metrics, "torch_rng_state": torch.get_rng_state(),
                    "numpy_rng_state": np.random.get_state(),
                }
                if development_metadata is not None:
                    checkpoint["development_training"] = development_metadata
                _atomic_torch(path, checkpoint)
                epochs_this_call += 1
                if interrupt_after_epochs is not None and epochs_this_call >= interrupt_after_epochs:
                    raise TrainingInterrupted("intentional epoch-boundary interruption")
        paths.append(path)
        summaries.append(torch.load(path, weights_only=False, map_location="cpu")["metrics"])
    ensemble_predictions = []
    with torch.inference_mode():
        for path in paths:
            saved = torch.load(path, weights_only=False, map_location="cpu")
            member = CurveMLP(625, cfg["hidden_dims"], OUTPUT_DIM)
            member.load_state_dict(saved["model_state"], strict=True); member.eval()
            xnorm, ynorm = saved["input_norm"], saved["target_norm"]
            normalized = member((current_hold[0] - xnorm[0]) / xnorm[1])
            ensemble_predictions.append(normalized * ynorm[1] + ynorm[0])
    ensemble_holdout = _error_metrics(torch.stack(ensemble_predictions).mean(dim=0),
                                      current_hold[1], len(current_hold[2]))
    member_models = [{"member_seed": int(torch.load(path, weights_only=False,
                                                      map_location="cpu")["member_seed"]),
                      "file": path.name, "sha256": file_sha256(path)} for path in paths]
    summary = {
        "schema_version": SCHEMA_VERSION, "data_version": data_version,
        "protocol_id": protocol["protocol_id"], "manifest_id": receipt["manifest_id"],
        "legacy_manifest_id": legacy_id, "member_seeds": cfg["ensemble_seeds"],
        "binding": binding, "member_models": member_models,
        "counts": {"profile_train": len(current_train[2]), "profile_holdout": len(current_hold[2]),
                   "legacy_train": len(legacy_train[2]), "legacy_holdout": len(legacy_hold[2])},
        "profile_train_ids": [r["id"] for r in current_train[2]],
        "profile_holdout_ids": [r["id"] for r in current_hold[2]],
        "legacy_train_exclusions_for_current_holdout": [r["id"] for r in legacy_excluded],
        "metrics": summaries,
        "ensemble_profile_holdout": ensemble_holdout,
        "validation_policy": "current_profile_holdout_only_no_early_stopping",
    }
    if development_metadata is not None:
        summary["development_training"] = development_metadata
        summary["validation_policy"] = (
            "remaining_reference_group_holdouts_descriptive_only_no_early_stopping")
    _atomic_json(model_dir / "summary.json", summary)
    return paths


class CurrentProfilePredictor:
    """Adapter from completed versioned checkpoints to the factory Predictor API."""

    model_role = "current_profile"

    def __init__(self, checkpoints: Sequence[str | Path], *,
                 expected_seeds: Sequence[int] | None = None,
                 expected_binding: Mapping[str, Any] | None = None,
                 expected_legacy_manifest_id: str | None = None):
        if not checkpoints:
            raise ValueError("at least one current-profile checkpoint is required")
        self.models: list[CurveMLP] = []
        self.input_norms: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.target_norms: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.model_ids: list[dict[str, str]] = []
        common: tuple[str, int, str, str | None] | None = None
        common_binding: dict[str, Any] | None = None
        seen_seeds: set[int] = set()
        for supplied in checkpoints:
            path = Path(supplied).resolve()
            saved = torch.load(path, weights_only=False, map_location="cpu")
            if not saved.get("complete") or not saved.get("fresh_initialization"):
                raise ValueError(f"checkpoint is incomplete or not versioned training: {path}")
            phase_names = list(saved.get("phase_names", ()))
            if "current_profile" not in phase_names:
                raise ValueError(f"checkpoint has no current-profile phase: {path}")
            state = saved["model_state"]
            linear_weights = [v for k, v in state.items() if k.endswith(".weight")]
            if len(linear_weights) < 2 or linear_weights[0].shape[1] != 625 or linear_weights[-1].shape[0] != OUTPUT_DIM:
                raise ValueError(f"unsupported CurveMLP checkpoint: {path}")
            hidden = [int(weight.shape[0]) for weight in linear_weights[:-1]]
            model = CurveMLP(625, hidden, OUTPUT_DIM)
            model.load_state_dict(state, strict=True); model.eval(); model.requires_grad_(False)
            input_norm = saved.get("input_norm")
            target_norm = saved.get("target_norm")
            if input_norm is None or target_norm is None:
                raise ValueError(f"checkpoint lacks shared train-only normalization: {path}")
            member_seed = int(saved["member_seed"])
            if member_seed in seen_seeds:
                raise ValueError(f"duplicate ensemble member seed: {member_seed}")
            seen_seeds.add(member_seed)
            signature = (str(saved["protocol_id"]), int(saved["data_version"]),
                         str(saved["manifest_id"]), saved.get("legacy_manifest_id"))
            if common is not None and signature != common:
                raise ValueError("current-profile ensemble checkpoints are from different data versions")
            common = signature
            recomputed = content_id({"protocol_id": signature[0], "manifest_id": signature[2],
                                     "legacy_manifest_id": signature[3],
                                     "member_seed": member_seed})
            if saved.get("signature") != recomputed:
                raise ValueError(f"checkpoint training signature mismatch: {path}")
            checkpoint_binding = saved.get("binding")
            if not isinstance(checkpoint_binding, Mapping):
                raise ValueError(f"checkpoint lacks immutable factory binding: {path}")
            checkpoint_binding = dict(checkpoint_binding)
            required_binding = {"measurement_id", "score_spec_id", "protocol_id", "data_version",
                                "manifest_id", "cumulative_valid_unique"}
            if set(checkpoint_binding) != required_binding:
                raise ValueError(f"checkpoint factory binding keys mismatch: {path}")
            if (not isinstance(checkpoint_binding["data_version"], int) or
                    checkpoint_binding["data_version"] <= 0 or
                    not isinstance(checkpoint_binding["cumulative_valid_unique"], int) or
                    checkpoint_binding["cumulative_valid_unique"] < 0):
                raise ValueError(f"checkpoint factory binding counts are invalid: {path}")
            if (checkpoint_binding["protocol_id"] != signature[0] or
                    checkpoint_binding["data_version"] != signature[1] or
                    checkpoint_binding["manifest_id"] != signature[2]):
                raise ValueError(f"checkpoint factory binding contradicts checkpoint: {path}")
            if common_binding is not None and checkpoint_binding != common_binding:
                raise ValueError("current-profile ensemble checkpoint bindings differ")
            common_binding = checkpoint_binding
            if expected_binding is not None and checkpoint_binding != dict(expected_binding):
                raise ValueError(f"checkpoint factory binding mismatch: {path}")
            if (expected_legacy_manifest_id is not None and
                    signature[3] != expected_legacy_manifest_id):
                raise ValueError(f"checkpoint legacy manifest mismatch: {path}")
            self.models.append(model)
            self.input_norms.append((torch.as_tensor(input_norm[0]), torch.as_tensor(input_norm[1])))
            self.target_norms.append((torch.as_tensor(target_norm[0]), torch.as_tensor(target_norm[1])))
            self.model_ids.append({"file": str(path), "sha256": file_sha256(path),
                                   "architecture": "CurveMLP-216",
                                   "source": "script.symmetry_training",
                                   "protocol_id": signature[0], "data_version": str(signature[1]),
                                   "manifest_id": signature[2]})
        self.data_version = common[1] if common else -1
        if expected_seeds is not None and seen_seeds != {int(seed) for seed in expected_seeds}:
            raise ValueError("checkpoint member seeds do not match the configured ensemble")
        self.binding = dict(expected_binding or common_binding or {})

    def predict(self, patterns: Sequence[np.ndarray], *, batch_size: int = 256):
        from script.symmetry_sm_pool import PredictionBatch

        if batch_size <= 0 or not patterns:
            raise ValueError("patterns must be non-empty and batch_size positive")
        x = torch.stack([torch.as_tensor(pattern, dtype=torch.float32).reshape(625)
                         for pattern in patterns])
        for pattern in x:
            pattern_sha256(pattern.reshape(25, 25))
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(x), batch_size):
                xb = x[start:start + batch_size]
                chunks.append(torch.stack([
                    model((xb - xnorm[0]) / xnorm[1]) * ynorm[1] + ynorm[0]
                    for model, xnorm, ynorm in zip(self.models, self.input_norms,
                                                   self.target_norms)
                ], dim=0))
        members = torch.cat(chunks, dim=1)
        mean = members.mean(dim=0)
        response_members = members[:, :, :34].reshape(len(self.models), len(x), 2, 17)
        response_mean = mean[:, :34].reshape(len(x), 2, 17)
        radiation = mean[:, 34:].reshape(len(x), 2, 91)
        disagreement = (response_members.std(dim=0, unbiased=False).mean(dim=(1, 2))
                        if len(self.models) > 1 else torch.zeros(len(x)))
        return PredictionBatch(
            response_mean=response_mean.numpy(),
            response_disagreement=disagreement.numpy(),
            radiation=radiation.numpy(),
            radiation_theta=np.linspace(-90.0, 90.0, 91, dtype=np.float32),
            response_members=response_members.numpy(),
        )


def load_current_predictor(work_dir: str | Path, version: int | None = None) -> CurrentProfilePredictor:
    """Load a completed local model version for direct factory injection."""
    root = Path(work_dir).resolve()
    state = _read_json(root / "state.json")
    selected = int(state["latest_data_version"] if version is None else version)
    protocol = _protocol(root)
    model_dir = root / "models" / f"data-v{selected:03d}"
    receipt = _read_json(root / "data" / f"data-v{selected:03d}" / "receipt.json")
    expected_binding = {"measurement_id": protocol["measurement_id"],
                        "score_spec_id": protocol["score_spec_id"],
                        "protocol_id": protocol["protocol_id"], "data_version": selected,
                        "manifest_id": receipt["manifest_id"],
                        "cumulative_valid_unique": int(receipt["cumulative_valid_unique"])}
    paths = [model_dir / f"member-{int(seed)}.pt" for seed in protocol["model"]["ensemble_seeds"]]
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError(f"model version is incomplete: {model_dir}")
    summary = _read_json(model_dir / "summary.json")
    if summary.get("binding") != expected_binding:
        raise ValueError("model summary binding mismatch")
    expected_models = [{"member_seed": int(seed), "file": path.name, "sha256": file_sha256(path)}
                       for seed, path in zip(protocol["model"]["ensemble_seeds"], paths)]
    if summary.get("member_models") != expected_models:
        raise ValueError("model member hash receipt mismatch")
    return CurrentProfilePredictor(
        paths, expected_seeds=protocol["model"]["ensemble_seeds"],
        expected_binding=expected_binding,
        expected_legacy_manifest_id=summary.get("legacy_manifest_id"),
    )


def _cli_work(path: str) -> Path:
    root = Path(path).resolve()
    if not _inside(root, TMP_ROOT):
        raise ValueError(f"CLI work directory must be below {TMP_ROOT}")
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description="Local versioned symmetry-SM training")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--work-dir", required=True); init.add_argument("--measurement", required=True)
    init.add_argument("--score-spec", required=True); init.add_argument("--geometry-profile", required=True)
    ingest = sub.add_parser("ingest")
    ingest.add_argument("--work-dir", required=True); ingest.add_argument("--store", action="append", required=True)
    ingest.add_argument("--min-new", type=int); ingest.add_argument("--max-new", type=int)
    legacy = sub.add_parser("register-legacy")
    legacy.add_argument("--work-dir", required=True); legacy.add_argument("--manifest", required=True)
    legacy.add_argument("--dataset-root", required=True); legacy.add_argument("--provenance", required=True)
    legacy.add_argument("--max-samples", type=int)
    prepare = sub.add_parser("prepare-legacy")
    prepare.add_argument("--work-dir", required=True,
                         help="used only to enforce that --output is in this local work tree")
    prepare.add_argument("--dataset-root", required=True); prepare.add_argument("--output", required=True)
    prepare.add_argument("--max-samples", type=int); prepare.add_argument("--max-stores", type=int)
    train = sub.add_parser("train")
    train.add_argument("--work-dir", required=True); train.add_argument("--version", type=int)
    args = parser.parse_args()
    work = _cli_work(args.work_dir)
    if args.command == "init":
        result = initialize_workdir(work, args.measurement, args.score_spec,
                                    geometry_profile=args.geometry_profile)
    elif args.command == "ingest":
        result = ingest_profile_stores(work, args.store, min_new=args.min_new, max_new=args.max_new)
    elif args.command == "register-legacy":
        result = register_legacy_prior(work, args.manifest, args.dataset_root, args.provenance,
                                       max_samples=args.max_samples)
    elif args.command == "prepare-legacy":
        output = Path(args.output).resolve()
        if not _inside(output, work):
            raise ValueError("prepared legacy manifest must be below --work-dir")
        result = prepare_legacy_manifest(args.dataset_root, output,
                                         max_samples=args.max_samples,
                                         max_stores=args.max_stores)
    else:
        result = train_version(work, args.version)
    print(result)


if __name__ == "__main__":
    main()
