# -*- coding: utf-8 -*-
"""Transactional bootstrap of a family-development symmetry training workdir.

The migration consumes one explicitly named, frozen parent data version.  It
does not follow ``state.json``, copy checkpoints, train a model, or touch the
factory queue.  All source payloads are local and hash checked before and
after the copy.  A child becomes visible at its destination only after the
complete derived manifest, unchanged legacy prior, receipts, and state have
been validated in a private sibling staging directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from antenna.measurement import measurement_id, score_spec_id
from script.exploration import TMP_ROOT, _inside, content_id, file_sha256, pattern_sha256
from script import symmetry_training as training


MIGRATION_SCHEMA_VERSION = 1
POLICY_METHOD = "explicit_whole_canonical_group_development_override_v1"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_local_child(path: Path, allowed_root: Path, label: str) -> Path:
    resolved = path.resolve()
    root = allowed_root.resolve()
    if not _inside(resolved, root):
        raise ValueError(f"{label} must resolve below the explicit local allowed root")
    return resolved


def _require_safe_file(root: Path, relative: str, label: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"{label} has an unsafe relative path")
    path = (root / relative).resolve()
    if not _inside(path, root) or not path.is_file():
        raise ValueError(f"{label} is missing or escapes its workdir: {relative}")
    return path


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    array = tensor.numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _metadata_binding(parent: Path, version: int) -> dict[str, Any]:
    names = {
        "protocol": parent / "protocol.json",
        "measurement": parent / "measurement.json",
        "score_spec": parent / "score_spec.json",
        "profile_manifest": parent / "data" / f"data-v{version:03d}" / "manifest.json",
        "profile_receipt": parent / "data" / f"data-v{version:03d}" / "receipt.json",
        "legacy_manifest": parent / "legacy" / "manifest.json",
        "legacy_provenance": parent / "legacy" / "provenance.json",
    }
    missing = [name for name, path in names.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"frozen parent inputs are incomplete: {missing}")
    hashes = {name: file_sha256(path) for name, path in names.items()}
    return {"files": hashes, "binding_id": content_id(hashes)}


def _validate_hex(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64 or
            any(char not in "0123456789abcdef" for char in value)):
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


def _identity(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a nonempty normalized string")
    return value


def _resolve_alias(value: str, aliases: Mapping[str, str]) -> str:
    current = _identity(value, "lineage identity")
    seen: set[str] = set()
    while current in aliases and aliases[current] != current:
        if current in seen:
            raise ValueError("lineage alias cycle")
        seen.add(current)
        current = _identity(aliases[current], "lineage alias target")
    return current


def _derived_aliases(protocol: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    raw_aliases = protocol.get("lineage_aliases", {}).get("mapping", {})
    if not isinstance(raw_aliases, Mapping):
        raise ValueError("parent protocol lineage aliases must be a mapping")
    aliases: dict[str, str] = {}
    for raw_key, raw_value in raw_aliases.items():
        key = _identity(raw_key, "parent protocol alias key")
        value = _identity(raw_value, "parent protocol alias target")
        aliases[key] = value
    for key, value in aliases.items():
        if _resolve_alias(value, aliases) != value or _resolve_alias(key, aliases) != value:
            raise ValueError("parent protocol aliases do not preserve frozen canonical roots")
    for row in rows:
        lineage = _identity(row.get("lineage_id"), "profile lineage_id")
        canonical = _identity(row.get("canonical_group_id"), "profile canonical_group_id")
        if _resolve_alias(canonical, aliases) != canonical:
            raise ValueError("frozen manifest canonical group is not an alias root")
        if lineage == canonical:
            continue
        prior = aliases.get(lineage)
        if prior is not None and _resolve_alias(lineage, aliases) != canonical:
            raise ValueError(f"lineage maps to conflicting canonical groups: {lineage}")
        if prior is None:
            aliases[lineage] = canonical
    for row in rows:
        lineage = _identity(row.get("lineage_id"), "profile lineage_id")
        canonical = _identity(row.get("canonical_group_id"), "profile canonical_group_id")
        if (_resolve_alias(canonical, aliases) != canonical or
                _resolve_alias(lineage, aliases) != canonical):
            raise ValueError("frozen manifest lineage/canonical closure was retargeted")
    return dict(sorted(aliases.items()))


def _profile_sources(parent: Path, protocol: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected_keys = {
        "measurement_id": protocol["measurement_id"],
        "score_spec_id": protocol["score_spec_id"],
        "geometry_profile": protocol["geometry_profile"],
    }
    ids: set[str] = set()
    patterns: set[str] = set()
    lineage_groups: dict[str, str] = {}
    payloads: dict[str, str] = {}
    truths: list[dict[str, str]] = []
    for raw in rows:
        row = dict(raw)
        uid = _identity(row.get("id"), "profile observation id")
        if uid in ids:
            raise ValueError("profile observation IDs must be nonempty and unique")
        ids.add(uid)
        for key, value in expected_keys.items():
            if row.get(key) != value:
                raise ValueError(f"profile row {uid} has a different {key}")
        lineage = _identity(row.get("lineage_id"), f"profile row {uid} lineage_id")
        canonical = _identity(row.get("canonical_group_id"),
                              f"profile row {uid} canonical_group_id")
        prior_group = lineage_groups.setdefault(lineage, canonical)
        if prior_group != canonical:
            raise ValueError(f"lineage appears in multiple canonical groups: {lineage}")
        reference_split = "profile_" + training.fixed_group_split(
            canonical, protocol["split"]["fraction"], protocol["split"]["seed"])
        if row.get("split") != reference_split or "reference_split" in row:
            raise ValueError(f"parent profile row {uid} is not an unmodified reference split")
        pattern_hash = _validate_hex(row.get("pattern_sha256"), "pattern_sha256")
        if pattern_hash in patterns:
            raise ValueError(
                "frozen parent profile manifest contains repeated/nonunique patterns; "
                "bootstrap requires one label per valid unique pattern")
        patterns.add(pattern_hash)
        sample = _require_safe_file(parent, row.get("sample_file"), f"profile sample {uid}")
        rad = _require_safe_file(parent, row.get("rad_file"), f"profile radiation {uid}")
        sample_hash = file_sha256(sample)
        rad_hash = file_sha256(rad)
        if sample_hash != row.get("source_sample_sha256") or rad_hash != row.get("source_rad_sha256"):
            raise ValueError(f"profile cache hash mismatch: {uid}")
        pattern, target = training._target(sample, rad, protocol["frequencies_ghz"])
        if pattern_sha256(pattern) != pattern_hash:
            raise ValueError(f"profile cached pattern identity mismatch: {uid}")
        payloads[row["sample_file"]] = sample_hash
        payloads[row["rad_file"]] = rad_hash
        truths.append({"id": uid, "pattern_sha256": pattern_hash,
                       "target_sha256": _tensor_sha256(target)})
    if not rows:
        raise ValueError("frozen parent profile manifest is empty")
    return {
        "observation_count": len(rows),
        "valid_unique_count": len(patterns),
        "payload_count": len(payloads),
        "payloads": dict(sorted(payloads.items())),
        "payload_tree_id": content_id(dict(sorted(payloads.items()))),
        "truth_id": content_id(sorted(truths, key=lambda item: item["id"])),
        "repeat_nonunique_proof": {
            "status": "pattern_unique_proved_from_frozen_training_manifest",
            "manifest_rows": len(rows),
            "unique_pattern_sha256": len(patterns),
            "limitation": "training cache rows do not retain the upstream source kind; "
                          "the migration proves one cached observation per unique pattern, "
                          "not the original queue kind label",
        },
    }


def _legacy_sources(parent: Path, protocol: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
                    provenance: Mapping[str, Any]) -> dict[str, Any]:
    if not rows:
        raise ValueError("frozen parent legacy manifest is empty")
    if (provenance.get("usage") != "pretraining_prior_only" or
            provenance.get("excluded_from_current_validation") is not True or
            provenance.get("current_measurement_id") != protocol["measurement_id"] or
            provenance.get("current_geometry_profile") != protocol["geometry_profile"]):
        raise ValueError("legacy provenance does not describe the immutable pretraining prior")
    ids: set[str] = set()
    payloads: dict[str, str] = {}
    truths: list[dict[str, str]] = []
    for raw in rows:
        row = dict(raw)
        uid = _identity(row.get("id"), "legacy observation id")
        if uid in ids:
            raise ValueError("legacy IDs must be nonempty and unique")
        ids.add(uid)
        lineage = _identity(row.get("lineage_id"), f"legacy row {uid} lineage_id")
        expected_split = "legacy_" + training.fixed_group_split(
            lineage, protocol["split"]["fraction"], protocol["split"]["seed"])
        if row.get("split") != expected_split:
            raise ValueError(f"legacy role changed for {uid}")
        sample = _require_safe_file(parent, row.get("sample_file"), f"legacy sample {uid}")
        rad = _require_safe_file(parent, row.get("rad_file"), f"legacy radiation {uid}")
        sample_hash = file_sha256(sample)
        rad_hash = file_sha256(rad)
        if sample_hash != row.get("source_sample_sha256") or rad_hash != row.get("source_rad_sha256"):
            raise ValueError(f"legacy cache hash mismatch: {uid}")
        pattern, target = training._target(sample, rad, protocol["frequencies_ghz"])
        if pattern_sha256(pattern) != row.get("pattern_sha256"):
            raise ValueError(f"legacy cached pattern identity mismatch: {uid}")
        payloads[row["sample_file"]] = sample_hash
        payloads[row["rad_file"]] = rad_hash
        truths.append({"id": uid, "pattern_sha256": row["pattern_sha256"],
                       "target_sha256": _tensor_sha256(target)})
    return {
        "observation_count": len(rows),
        "payload_count": len(payloads),
        "payloads": dict(sorted(payloads.items())),
        "payload_tree_id": content_id(dict(sorted(payloads.items()))),
        "truth_id": content_id(sorted(truths, key=lambda item: item["id"])),
    }


def _source_binding(parent: Path, version: int) -> dict[str, Any]:
    protocol = training._protocol(parent)
    if "development_train_groups" in protocol or "reference_protocol_id" in protocol:
        raise ValueError("parent must be an original reference protocol, not a development child")
    measurement = _read_json(parent / "measurement.json")
    score = _read_json(parent / "score_spec.json")
    if (measurement_id(measurement) != protocol["measurement_id"] or
            score_spec_id(score) != protocol["score_spec_id"]):
        raise ValueError("parent measurement/score identity differs from protocol")
    version_dir = parent / "data" / f"data-v{version:03d}"
    rows = _read_json(version_dir / "manifest.json")
    receipt = _read_json(version_dir / "receipt.json")
    if not isinstance(rows, list) or receipt.get("data_version") != version:
        raise ValueError("explicit frozen parent version is invalid")
    manifest_id = content_id(rows)
    if receipt.get("manifest_id") != manifest_id:
        raise ValueError("parent profile manifest hash mismatch")
    legacy_rows = _read_json(parent / "legacy" / "manifest.json")
    provenance = _read_json(parent / "legacy" / "provenance.json")
    if not isinstance(legacy_rows, list) or not isinstance(provenance, Mapping):
        raise ValueError("parent legacy evidence is invalid")
    profile = _profile_sources(parent, protocol, rows)
    if (receipt.get("cumulative_observations") != profile["observation_count"] or
            receipt.get("cumulative_valid_unique") != profile["valid_unique_count"]):
        raise ValueError("parent receipt counts differ from physical profile cache")
    legacy = _legacy_sources(parent, protocol, legacy_rows, provenance)
    aliases = _derived_aliases(protocol, rows)
    metadata = _metadata_binding(parent, version)
    return {
        "protocol": protocol,
        "measurement": measurement,
        "score_spec": score,
        "profile_rows": rows,
        "profile_receipt": receipt,
        "legacy_rows": legacy_rows,
        "legacy_provenance": dict(provenance),
        "aliases": aliases,
        "metadata": metadata,
        "profile": profile,
        "legacy": legacy,
        "source_binding_id": content_id({
            "metadata": metadata, "profile": {k: v for k, v in profile.items() if k != "payloads"},
            "legacy": {k: v for k, v in legacy.items() if k != "payloads"},
        }),
    }


def _copy_payloads(source_root: Path, destination_root: Path,
                   payloads: Mapping[str, str]) -> None:
    for relative, expected_hash in sorted(payloads.items()):
        source = _require_safe_file(source_root, relative, "bound payload")
        if file_sha256(source) != expected_hash:
            raise ValueError(f"bound source payload changed before copy: {relative}")
        destination = (destination_root / relative).resolve()
        if not _inside(destination, destination_root):
            raise ValueError(f"unsafe child payload path: {relative}")
        training._atomic_copy(source, destination)
        if file_sha256(destination) != expected_hash:
            raise ValueError(f"child payload differs after copy: {relative}")


def _role_fields(canonical: str, protocol: Mapping[str, Any],
                 aliases: Mapping[str, str]) -> dict[str, str]:
    helper = getattr(training, "_profile_role", None)
    if helper is None:
        raise RuntimeError("symmetry_training lacks the development-role helper")
    value = helper(canonical, protocol, aliases)
    if not isinstance(value, Mapping) or set(value) != {"split", "reference_split"}:
        raise ValueError("development-role helper returned an unsupported schema")
    split = _identity(value["split"], "development effective split")
    reference = _identity(value["reference_split"], "development reference split")
    return {"split": split, "reference_split": reference}


def _derive_child_rows(parent_rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any],
                       aliases: Mapping[str, str]) -> list[dict[str, Any]]:
    out = []
    for raw in parent_rows:
        row = dict(raw)
        reference_split = row["split"]
        roles = _role_fields(
            _identity(row.get("canonical_group_id"), "child canonical_group_id"),
            protocol, aliases)
        if roles["reference_split"] != reference_split:
            raise ValueError(f"child reference role differs from parent: {row['id']}")
        row.update(roles)
        out.append(row)
    return sorted(out, key=lambda row: row["id"])


def _child_payload_binding(root: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = {}
    for row in rows:
        for key in ("sample_file", "rad_file"):
            path = _require_safe_file(root, row[key], "child payload")
            values[row[key]] = file_sha256(path)
    return {"payload_count": len(values), "payload_tree_id": content_id(dict(sorted(values.items())))}


def _validate_payload_inventory(root: Path, profile_rows: Sequence[Mapping[str, Any]],
                                legacy_rows: Sequence[Mapping[str, Any]]) -> None:
    expected = {_identity(row.get(key), f"child {key}")
                for row in [*profile_rows, *legacy_rows]
                for key in ("sample_file", "rad_file")}
    actual = {path.relative_to(root).as_posix()
              for directory in (root / "profile", root / "legacy") if directory.is_dir()
              for path in directory.rglob("*.pt") if path.is_file()}
    if actual != expected:
        raise ValueError("child payload inventory contains missing or unreferenced tensor files")


def _fresh_state(protocol_id: str, aliases: Mapping[str, str]) -> dict[str, Any]:
    return {
        "schema_version": training.SCHEMA_VERSION,
        "protocol_id": protocol_id,
        "latest_data_version": 0,
        "chunks": [],
        "lineage_aliases": dict(sorted(aliases.items())),
    }


def _validate_child(stage: Path, source: Mapping[str, Any], child_rows: Sequence[Mapping[str, Any]],
                    groups: Sequence[str], policy_id: str) -> dict[str, Any]:
    protocol = training._protocol(stage)
    state = _read_json(stage / "state.json")
    if state != _fresh_state(protocol["protocol_id"], source["aliases"]):
        raise ValueError("fresh child state protocol or alias closure differs from frozen sources")
    if protocol.get("reference_protocol_id") != source["protocol"]["protocol_id"]:
        raise ValueError("child reference protocol binding mismatch")
    if protocol.get("development_train_groups") != list(groups):
        raise ValueError("child development group binding mismatch")
    if not isinstance(protocol.get("development_role_policy"), Mapping):
        raise ValueError("child protocol lacks explicit development role policy")
    if content_id(protocol["development_role_policy"]) != policy_id:
        raise ValueError("child development role policy identity mismatch")
    for key in ("measurement_id", "score_spec_id", "geometry_profile", "frequencies_ghz",
                "output_dim", "split", "model", "updates", "policy"):
        if protocol.get(key) != source["protocol"].get(key):
            raise ValueError(f"child changed the single-factor parent setting: {key}")
    manifest_path = stage / "data" / "data-v001" / "manifest.json"
    receipt_path = stage / "data" / "data-v001" / "receipt.json"
    rows = _read_json(manifest_path)
    receipt = _read_json(receipt_path)
    if rows != list(child_rows) or receipt.get("manifest_id") != content_id(rows):
        raise ValueError("child bootstrap manifest/receipt mismatch")
    if (receipt.get("new_valid_unique") != 0 or receipt.get("new_observations") != 0 or
            receipt.get("collection_delta_valid_unique") != 0 or
            receipt.get("collection_delta_observations") != 0 or
            receipt.get("bootstrap_valid_unique") != source["profile"]["valid_unique_count"] or
            receipt.get("cumulative_valid_unique") != source["profile"]["valid_unique_count"]):
        raise ValueError("bootstrap counts are not separated from new collection counts")
    groups_set = set(groups)
    for row in rows:
        canonical = row["canonical_group_id"]
        expected = _role_fields(canonical, protocol, source["aliases"])
        if {key: row.get(key) for key in expected} != expected:
            raise ValueError(f"child row role mismatch: {row['id']}")
        if canonical in groups_set and row["split"] != "profile_train":
            raise ValueError("development canonical group is not wholly training")
    for group in groups_set:
        members = [row for row in rows if row["canonical_group_id"] == group]
        if not members or {row["split"] for row in members} != {"profile_train"}:
            raise ValueError(f"development group closure is incomplete: {group}")
    reference_rows = []
    for row in rows:
        reference_row = {key: value for key, value in row.items() if key != "reference_split"}
        reference_row["split"] = row["reference_split"]
        reference_rows.append(reference_row)
    profile_validation = _profile_sources(stage, protocol, reference_rows)
    if (profile_validation["truth_id"] != source["profile"]["truth_id"] or
            profile_validation["payload_tree_id"] != source["profile"]["payload_tree_id"]):
        raise ValueError("child profile truth/cache differs from parent")
    legacy_manifest = stage / "legacy" / "manifest.json"
    legacy_provenance = stage / "legacy" / "provenance.json"
    if (legacy_manifest.read_bytes() != (Path(source["parent"]) / "legacy" / "manifest.json").read_bytes() or
            legacy_provenance.read_bytes() !=
            (Path(source["parent"]) / "legacy" / "provenance.json").read_bytes()):
        raise ValueError("child legacy manifest/provenance bytes changed")
    legacy_rows = _read_json(legacy_manifest)
    legacy_validation = _legacy_sources(stage, protocol, legacy_rows, _read_json(legacy_provenance))
    if (legacy_validation["truth_id"] != source["legacy"]["truth_id"] or
            legacy_validation["payload_tree_id"] != source["legacy"]["payload_tree_id"]):
        raise ValueError("child legacy truth/cache differs from parent")
    _validate_payload_inventory(stage, rows, legacy_rows)
    if (stage / "models").exists() or any(stage.rglob("*.pt.tmp")):
        raise ValueError("migration copied a model/checkpoint or left a partial tensor")
    return {
        "protocol_sha256": file_sha256(stage / "protocol.json"),
        "measurement_sha256": file_sha256(stage / "measurement.json"),
        "score_spec_sha256": file_sha256(stage / "score_spec.json"),
        "profile_manifest_sha256": file_sha256(manifest_path),
        "profile_receipt_sha256": file_sha256(receipt_path),
        "legacy_manifest_sha256": file_sha256(legacy_manifest),
        "legacy_provenance_sha256": file_sha256(legacy_provenance),
        "profile_payload_tree_id": _child_payload_binding(stage, rows)["payload_tree_id"],
        "legacy_payload_tree_id": _child_payload_binding(stage, legacy_rows)["payload_tree_id"],
    }


def migrate_development_workdir(
    parent_workdir: str | Path,
    parent_data_version: int,
    destination: str | Path,
    development_train_groups: Sequence[str],
    *,
    allowed_root: str | Path = TMP_ROOT,
) -> Path:
    """Create one complete child from an explicit frozen local parent version."""

    if (isinstance(parent_data_version, bool) or
            not isinstance(parent_data_version, int) or parent_data_version <= 0):
        raise ValueError("parent_data_version must be a positive explicit integer")
    version = parent_data_version
    root = Path(allowed_root).resolve()
    parent = _require_local_child(Path(parent_workdir), root, "parent_workdir")
    child = _require_local_child(Path(destination), root, "destination")
    if not parent.is_dir():
        raise FileNotFoundError(parent)
    if parent == child or _inside(child, parent) or _inside(parent, child):
        raise ValueError("parent and destination workdirs must not overlap")
    if child.exists() and (not child.is_dir() or any(child.iterdir())):
        raise ValueError("destination must be absent or an empty fresh directory")
    if isinstance(development_train_groups, (str, bytes)):
        raise TypeError("development_train_groups must be a sequence, not a string")
    supplied_groups = [
        _identity(value, "development_train_groups element")
        for value in development_train_groups
    ]
    groups = sorted(set(supplied_groups))
    if (not groups or len(groups) != len(supplied_groups) or
            any(not value or value != value.strip() for value in groups)):
        raise ValueError("development_train_groups must be nonempty, normalized, and unique")

    before = _source_binding(parent, version)
    before["parent"] = str(parent)
    canonical_groups = {row["canonical_group_id"] for row in before["profile_rows"]}
    missing_groups = sorted(set(groups) - canonical_groups)
    if missing_groups:
        raise ValueError(f"development groups are absent from frozen parent: {missing_groups}")
    alias_source_id = content_id({
        "method": "frozen_parent_manifest_alias_closure_v1",
        "parent_protocol_id": before["protocol"]["protocol_id"],
        "parent_data_version": version,
        "parent_manifest_id": before["profile_receipt"]["manifest_id"],
        "aliases": before["aliases"],
    })
    migration_id = content_id({
        "schema_version": MIGRATION_SCHEMA_VERSION,
        "method": POLICY_METHOD,
        "source_binding_id": before["source_binding_id"],
        "development_train_groups": groups,
        "alias_source_id": alias_source_id,
    })
    pending = child.parent / f".{child.name}.migration-pending-{migration_id[:16]}"
    if pending.exists():
        raise FileExistsError(
            f"incomplete or conflicting migration staging exists and will not be overwritten: {pending}")

    protocol = before["protocol"]
    model = protocol["model"]
    training.initialize_workdir(
        pending,
        before["measurement"],
        before["score_spec"],
        geometry_profile=protocol["geometry_profile"],
        holdout_seed=int(protocol["split"]["seed"]),
        holdout_fraction=float(protocol["split"]["fraction"]),
        ensemble_seeds=protocol["model"]["ensemble_seeds"],
        hidden_dims=model["hidden_dims"],
        learning_rate=float(model["learning_rate"]),
        batch_size=int(model["batch_size"]),
        legacy_epochs=int(model["legacy_epochs"]),
        current_epochs=int(model["current_epochs"]),
        update_min=int(protocol["updates"]["min_new_valid"]),
        update_max=int(protocol["updates"]["max_new_valid"]),
        initial_lineage_aliases=before["aliases"],
        alias_source_sha256=alias_source_id,
        development_train_groups=groups,
        reference_protocol_id=protocol["protocol_id"],
    )
    child_protocol = training._protocol(pending)
    policy = child_protocol.get("development_role_policy")
    if not isinstance(policy, Mapping):
        raise ValueError("initialized child protocol lacks development_role_policy")
    policy_id = content_id(policy)
    child_rows = _derive_child_rows(before["profile_rows"], child_protocol, before["aliases"])

    _copy_payloads(parent, pending, before["profile"]["payloads"])
    _copy_payloads(parent, pending, before["legacy"]["payloads"])
    training._atomic_copy(parent / "legacy" / "manifest.json", pending / "legacy" / "manifest.json")
    training._atomic_copy(parent / "legacy" / "provenance.json", pending / "legacy" / "provenance.json")

    manifest_id = content_id(child_rows)
    chunk_id = content_id({"migration_id": migration_id, "manifest_id": manifest_id})
    version_dir = pending / "data" / "data-v001"
    training._atomic_json(version_dir / "manifest.json", child_rows)
    data_receipt = {
        "schema_version": training.SCHEMA_VERSION,
        "data_version": 1,
        "chunk_id": chunk_id,
        "manifest_id": manifest_id,
        "new_valid_unique": 0,
        "new_observations": 0,
        "collection_delta_valid_unique": 0,
        "collection_delta_observations": 0,
        "bootstrap_valid_unique": before["profile"]["valid_unique_count"],
        "bootstrap_observations": before["profile"]["observation_count"],
        "cumulative_valid_unique": before["profile"]["valid_unique_count"],
        "cumulative_observations": before["profile"]["observation_count"],
        "source_store_count": 0,
        "bootstrap_kind": "frozen_parent_migration_not_new_hfss_collection",
        "migration_id": migration_id,
        "parent_protocol_id": protocol["protocol_id"],
        "parent_data_version": version,
        "parent_manifest_id": before["profile_receipt"]["manifest_id"],
        "development_policy_id": policy_id,
    }
    training._atomic_json(version_dir / "receipt.json", data_receipt)

    # This validation intentionally runs while latest_data_version is still zero.
    source_for_validation = dict(before)
    source_for_validation["parent"] = str(parent)
    child_binding = _validate_child(
        pending, source_for_validation, child_rows, groups, policy_id)

    after = _source_binding(parent, version)
    if after["source_binding_id"] != before["source_binding_id"]:
        raise ValueError("parent source changed during migration")
    if after["metadata"] != before["metadata"] or after["profile"] != before["profile"] or after["legacy"] != before["legacy"]:
        raise ValueError("parent metadata, payload, or truth binding changed during migration")

    state = _read_json(pending / "state.json")
    if state != _fresh_state(child_protocol["protocol_id"], before["aliases"]):
        raise ValueError("fresh child state protocol or alias closure changed before publication")
    state["latest_data_version"] = 1
    state["chunks"] = [{
        "chunk_id": chunk_id,
        "data_version": 1,
        "manifest_id": manifest_id,
        "new_valid_unique": 0,
        "new_observations": 0,
        "bootstrap_valid_unique": before["profile"]["valid_unique_count"],
        "bootstrap_observations": before["profile"]["observation_count"],
        "migration_id": migration_id,
    }]
    training._atomic_json(pending / "state.json", state)

    receipt = {
        "schema_version": MIGRATION_SCHEMA_VERSION,
        "status": "complete",
        "migration_id": migration_id,
        "method": POLICY_METHOD,
        "destination": str(child),
        "parent": {
            "workdir": str(parent),
            "protocol_id": protocol["protocol_id"],
            "data_version": version,
            "manifest_id": before["profile_receipt"]["manifest_id"],
            "before": {
                "source_binding_id": before["source_binding_id"],
                "metadata": before["metadata"],
                "profile_payload_tree_id": before["profile"]["payload_tree_id"],
                "profile_truth_id": before["profile"]["truth_id"],
                "legacy_payload_tree_id": before["legacy"]["payload_tree_id"],
                "legacy_truth_id": before["legacy"]["truth_id"],
            },
            "after": {
                "source_binding_id": after["source_binding_id"],
                "metadata": after["metadata"],
                "profile_payload_tree_id": after["profile"]["payload_tree_id"],
                "profile_truth_id": after["profile"]["truth_id"],
                "legacy_payload_tree_id": after["legacy"]["payload_tree_id"],
                "legacy_truth_id": after["legacy"]["truth_id"],
            },
        },
        "child": {
            "protocol_id": child_protocol["protocol_id"],
            "reference_protocol_id": protocol["protocol_id"],
            "development_train_groups": groups,
            "development_role_policy": dict(policy),
            "development_policy_id": policy_id,
            "alias_source_id": alias_source_id,
            "lineage_aliases": before["aliases"],
            "lineage_aliases_id": content_id(before["aliases"]),
            "data_version": 1,
            "manifest_id": manifest_id,
            "profile_observations": before["profile"]["observation_count"],
            "profile_valid_unique": before["profile"]["valid_unique_count"],
            "new_hfss_valid_unique": 0,
            "new_hfss_observations": 0,
            "legacy_observations": before["legacy"]["observation_count"],
            "binding": child_binding,
        },
        "repeat_nonunique_exclusion": before["profile"]["repeat_nonunique_proof"],
        "limits": [
            "The frozen training manifest proves one cached observation per pattern hash but does not retain upstream queue kind labels.",
            "This migration creates no model and performs no training, HFSS, queue, controller, or network action.",
            "Remaining reference holdout groups are descriptive; this receipt is not prospective validation evidence.",
        ],
    }
    training._atomic_json(pending / "migration_receipt.json", receipt)
    state_hash = file_sha256(pending / "state.json")
    receipt_hash = file_sha256(pending / "migration_receipt.json")
    training._atomic_json(pending / "migration_complete.json", {
        "schema_version": MIGRATION_SCHEMA_VERSION,
        "migration_id": migration_id,
        "state_sha256": state_hash,
        "migration_receipt_sha256": receipt_hash,
    })

    if child.exists():
        child.rmdir()  # It was proven empty above; never remove a nonempty destination.
    os.replace(pending, child)
    verify_migrated_workdir(child, allowed_root=root)
    return child


def verify_migrated_workdir(destination: str | Path, *,
                            allowed_root: str | Path = TMP_ROOT) -> dict[str, Any]:
    """Verify the terminal marker and immutable child metadata without training."""

    root = Path(allowed_root).resolve()
    child = _require_local_child(Path(destination), root, "destination")
    receipt_path = child / "migration_receipt.json"
    marker_path = child / "migration_complete.json"
    if not receipt_path.is_file() or not marker_path.is_file():
        raise ValueError("migration is not completely published")
    receipt = _read_json(receipt_path)
    marker = _read_json(marker_path)
    if (receipt.get("status") != "complete" or
            Path(str(receipt.get("destination", ""))).resolve() != child or
            marker.get("migration_id") != receipt.get("migration_id") or
            marker.get("migration_receipt_sha256") != file_sha256(receipt_path) or
            marker.get("state_sha256") != file_sha256(child / "state.json")):
        raise ValueError("migration terminal binding mismatch")
    state = _read_json(child / "state.json")
    data_receipt = _read_json(child / "data" / "data-v001" / "receipt.json")
    manifest = _read_json(child / "data" / "data-v001" / "manifest.json")
    protocol = training._protocol(child)
    child_receipt = receipt.get("child", {})
    protocol_aliases = protocol.get("lineage_aliases", {}).get("mapping")
    if not isinstance(protocol_aliases, Mapping):
        raise ValueError("published protocol lacks a lineage alias mapping")
    derived_aliases = _derived_aliases(protocol, manifest)
    if (state.get("protocol_id") != protocol.get("protocol_id") or
            state.get("lineage_aliases") != derived_aliases or
            dict(protocol_aliases) != derived_aliases or
            child_receipt.get("lineage_aliases") != derived_aliases or
            child_receipt.get("lineage_aliases_id") != content_id(derived_aliases)):
        raise ValueError("published migration state alias/protocol binding mismatch")
    training._validate_development_aliases(protocol, state["lineage_aliases"])
    training._validate_profile_roles(manifest, protocol, state["lineage_aliases"])
    if (state.get("latest_data_version") != 1 or len(state.get("chunks", [])) != 1 or
            data_receipt.get("manifest_id") != content_id(manifest) or
            state["chunks"][0].get("manifest_id") != data_receipt.get("manifest_id") or
            protocol.get("protocol_id") != child_receipt.get("protocol_id") or
            child_receipt.get("manifest_id") != data_receipt.get("manifest_id")):
        raise ValueError("published migration state/data/protocol binding mismatch")
    if (data_receipt.get("new_valid_unique") != 0 or data_receipt.get("new_observations") != 0):
        raise ValueError("migration bootstrap was counted as new collection")
    expected_binding = receipt.get("child", {}).get("binding", {})
    for name, relative in (
            ("protocol_sha256", "protocol.json"),
            ("measurement_sha256", "measurement.json"),
            ("score_spec_sha256", "score_spec.json"),
            ("profile_manifest_sha256", "data/data-v001/manifest.json"),
            ("profile_receipt_sha256", "data/data-v001/receipt.json"),
            ("legacy_manifest_sha256", "legacy/manifest.json"),
            ("legacy_provenance_sha256", "legacy/provenance.json")):
        if expected_binding.get(name) != file_sha256(child / relative):
            raise ValueError(f"published migration metadata changed: {relative}")
    profile_payload = _child_payload_binding(child, manifest)
    legacy_rows = _read_json(child / "legacy" / "manifest.json")
    legacy_payload = _child_payload_binding(child, legacy_rows)
    _validate_payload_inventory(child, manifest, legacy_rows)
    if (profile_payload["payload_tree_id"] != expected_binding.get("profile_payload_tree_id") or
            legacy_payload["payload_tree_id"] != expected_binding.get("legacy_payload_tree_id")):
        raise ValueError("published migration payload tree changed")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-workdir", required=True)
    parser.add_argument("--parent-data-version", required=True, type=int)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--development-train-group", required=True, action="append")
    parser.add_argument("--allowed-root", default=str(TMP_ROOT))
    args = parser.parse_args()
    result = migrate_development_workdir(
        args.parent_workdir,
        args.parent_data_version,
        args.destination,
        args.development_train_group,
        allowed_root=args.allowed_root,
    )
    print(result / "migration_receipt.json")


if __name__ == "__main__":
    main()
