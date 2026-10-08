"""Metadata-only LOW backlog maintenance for one symmetry factory watcher.

This module has no standalone controller entry point.  One
``symmetry_factory_watch`` process owns a maintainer thread and shares its
``DatasetWriteCoordinator`` with the main cycle.  Routine polling reads queue,
input-manifest, and worker-marker metadata only; it never reads HFSS results or
loads/trains a surrogate model.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from antenna.measurement import measurement_id, score_spec_id
from script import exploration as ex
from script import profiled_batch as pb
from script import symmetry_factory_cycle as cycle


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BacklogPolicy:
    poll_seconds: int = 90
    low_watermark: int = 48
    target_pending: int = 96
    max_refill: int = 96
    min_refill: int = 48
    shard_size: int = 16
    priority: int = 6
    owner: str = "r80-low-backlog-v1"
    reservoir_seed: int = 80100791
    reservoir_size: int = 2048

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "BacklogPolicy":
        if not isinstance(raw, Mapping):
            raise ValueError("low_backlog settings must be an object")
        allowed = set(cls.__dataclass_fields__) | {"enabled"}
        if set(raw) - allowed:
            raise ValueError(f"unknown low_backlog settings: {sorted(set(raw) - allowed)}")
        if raw.get("enabled", True) is not True:
            raise ValueError("low_backlog must be omitted or explicitly enabled")
        policy = cls(**{key: value for key, value in raw.items() if key != "enabled"})
        integers = (policy.poll_seconds, policy.low_watermark, policy.target_pending,
                    policy.max_refill, policy.min_refill, policy.shard_size,
                    policy.priority, policy.reservoir_seed, policy.reservoir_size)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integers):
            raise ValueError("low_backlog numeric settings must be integers")
        if policy.poll_seconds != 90:
            raise ValueError("low_backlog poll_seconds must be exactly 90")
        if (policy.low_watermark, policy.target_pending, policy.max_refill,
                policy.min_refill, policy.shard_size, policy.priority) != (48, 96, 96, 48, 16, 6):
            raise ValueError("LOW keeper contract is fixed to 48/96/96/48 rows, shard16, prio6")
        if not policy.owner or policy.owner != policy.owner.strip():
            raise ValueError("low_backlog owner must be a normalized nonempty string")
        if policy.reservoir_size < policy.max_refill:
            raise ValueError("low_backlog reservoir_size must cover one refill")
        return policy


def plan_refill(own_pending: int, capacity: int,
                policy: BacklogPolicy = BacklogPolicy()) -> int:
    """Return a hard-high-water refill count in complete shards."""

    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
           for value in (own_pending, capacity)):
        raise ValueError("pending and capacity must be non-negative integers")
    if own_pending >= policy.low_watermark or capacity < policy.min_refill:
        return 0
    room = policy.target_pending - own_pending
    requested = min(policy.max_refill, max(policy.min_refill, room), capacity)
    requested = (requested // policy.shard_size) * policy.shard_size
    while requested and own_pending + requested > policy.target_pending:
        requested -= policy.shard_size
    return requested


def _stable_json(path: Path) -> tuple[Any, str]:
    first = path.read_bytes()
    value = json.loads(first.decode("utf-8"))
    if path.read_bytes() != first:
        raise cycle.LiveSnapshotChanged(f"metadata changed during keeper scan: {path}")
    return value, hashlib.sha256(first).hexdigest()


def _input_record(input_dir: Path, cfg: Any, *, owner: str | None = None,
                  expected_profile_sha256: str | None = None) -> dict[str, Any]:
    config_sha = pb.file_sha256(input_dir / "config.yaml")
    score, score_sha = _stable_json(input_dir / "score_spec.json")
    if expected_profile_sha256 is not None and config_sha != expected_profile_sha256:
        raise ValueError(f"physical input config differs from current bound profile: {input_dir}")
    if score_spec_id(score) != score_spec_id(cfg.score_spec):
        raise ValueError(f"keeper index input has a different score specification: {input_dir}")
    measurement, measurement_sha = _stable_json(input_dir / "measurement.json")
    if measurement_id(measurement) != measurement_id(cfg.measurement):
        raise ValueError(f"keeper index input has a different measurement: {input_dir}")
    manifest, manifest_sha = _stable_json(input_dir / "manifest.json")
    if not isinstance(manifest, list) or not manifest:
        raise ValueError(f"keeper index input has an invalid manifest: {input_dir}")
    hashes = [str(row["pattern_sha256"]) for row in manifest]
    if len(hashes) != len(set(hashes)):
        raise ValueError(f"keeper index input contains duplicate patterns: {input_dir}")
    marker = input_dir / "backlog_job.json"
    marker_sha = pb.file_sha256(marker) if marker.is_file() else None
    metadata = pb.read_json(marker) if marker.is_file() else None
    if metadata is not None:
        required = {"schema_version", "owner", "tier", "priority", "scope",
                    "profile_sha256", "measurement_id", "score_spec_id",
                    "hashes", "cohort_id", "source_bindings",
                    "preparation_receipt_sha256"}
        identity = ({key: value for key, value in metadata.items() if key != "cohort_id"}
                    if isinstance(metadata, dict) else {})
        if (not isinstance(metadata, dict) or not required <= set(metadata) or
                metadata["schema_version"] != 1 or metadata["tier"] != "LOW" or
                int(metadata["priority"]) != 6 or metadata["scope"] != cfg.scope or
                metadata["profile_sha256"] != pb.file_sha256(input_dir / "config.yaml") or
                metadata["measurement_id"] != measurement_id(cfg.measurement) or
                metadata["score_spec_id"] != score_spec_id(cfg.score_spec) or
                list(metadata["hashes"]) != hashes or
                metadata["cohort_id"] != _digest(identity) or
                (input_dir.name.startswith("dedust_r80k") and
                 input_dir.name != f"dedust_r80k{metadata['cohort_id'][:16]}_input")):
            raise ValueError(f"keeper ownership metadata differs from its input: {input_dir}")
    actual_owner = metadata.get("owner") if isinstance(metadata, dict) else owner
    return {"input": input_dir.name, "count": len(hashes), "hashes": hashes,
            "metadata_sha256": {"config.yaml": config_sha,
                                "measurement.json": measurement_sha,
                                "score_spec.json": score_sha,
                                "manifest.json": manifest_sha,
                                "backlog_job.json": marker_sha},
            "owner": actual_owner}


def _recheck_index_bindings(dataset: Path, index: Mapping[str, Any]) -> None:
    """Rehash every indexed immutable metadata file; result payloads stay untouched."""

    for name, record in index.get("inputs", {}).items():
        root = dataset / str(name)
        expected = record.get("metadata_sha256")
        if not isinstance(expected, dict):
            raise ValueError(f"keeper index lacks immutable metadata bindings: {root}")
        actual = {}
        for filename in ("config.yaml", "measurement.json", "score_spec.json",
                         "manifest.json", "backlog_job.json"):
            path = root / filename
            actual[filename] = pb.file_sha256(path) if path.is_file() else None
        if actual != expected:
            raise ValueError(f"indexed physical input metadata changed: {root}")


def _jobs_snapshot(dataset: Path) -> tuple[list[dict[str, Any]], str]:
    value, digest = _stable_json(dataset / "jobs.json")
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError("jobs.json must contain a list of objects")
    return value, digest


def rebuild_index(dataset: Path, cfg: Any, index_path: Path,
                  policy: BacklogPolicy, *,
                  expected_profile_sha256: str | None = None) -> dict[str, Any]:
    """Bind every physical same-measurement input, including queue orphans."""

    jobs, jobs_sha = _jobs_snapshot(dataset)
    job_by_input = {str(job.get("input")): job for job in jobs}
    inputs: dict[str, dict[str, Any]] = {}
    before = sorted(path.name for path in dataset.glob("*_input") if path.is_dir())
    for name in before:
        root = dataset / name
        if not (root / "measurement.json").is_file() or not (root / "manifest.json").is_file():
            if name in job_by_input or name.startswith("dedust_r80k"):
                raise ValueError(f"queued/keeper input is partial: {root}")
            continue
        try:
            if measurement_id(pb.read_json(root / "measurement.json")) != measurement_id(
                    cfg.measurement):
                continue
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid physical measurement metadata: {root}") from exc
        inputs[name] = _input_record(
            root, cfg, expected_profile_sha256=expected_profile_sha256)
    after = sorted(path.name for path in dataset.glob("*_input") if path.is_dir())
    jobs_after, jobs_after_sha = _jobs_snapshot(dataset)
    if before != after or jobs != jobs_after or jobs_sha != jobs_after_sha:
        raise cycle.LiveSnapshotChanged("physical input or queue changed during index rebuild")
    missing = sorted(str(job.get("input")) for job in jobs if str(job.get("input")) not in inputs)
    if missing:
        raise ValueError(f"queue references missing or partial physical inputs: {missing[:10]}")
    value = {"schema_version": 1, "scope": cfg.scope,
             "measurement_id": measurement_id(cfg.measurement),
             "profile_sha256": expected_profile_sha256,
             "jobs": jobs, "jobs_sha256": jobs_sha, "inputs": inputs,
             "rebuilt_utc": _utc(), "owner": policy.owner}
    index_path.parent.mkdir(parents=True, exist_ok=True)
    pb.atomic_json(index_path, value)
    return value


def refresh_index(dataset: Path, cfg: Any, index_path: Path,
                  policy: BacklogPolicy, *, force_rebuild: bool = False,
                  expected_profile_sha256: str | None = None) -> dict[str, Any]:
    """Refresh the rebuildable local metadata cache without reading result stores."""

    if force_rebuild or not index_path.is_file():
        return rebuild_index(dataset, cfg, index_path, policy,
                             expected_profile_sha256=expected_profile_sha256)
    index = pb.read_json(index_path)
    if index.get("profile_sha256") != expected_profile_sha256:
        raise ValueError("keeper index belongs to a different bound profile")
    jobs, jobs_sha = _jobs_snapshot(dataset)
    prior = index.get("jobs")
    if not isinstance(prior, list) or jobs[:len(prior)] != prior:
        raise ValueError("jobs.json is not an append-only extension of the keeper index")
    inputs = dict(index.get("inputs", {}))
    for job in jobs[len(prior):]:
        name = str(job.get("input"))
        root = dataset / name
        if not root.is_dir():
            raise cycle.LiveSnapshotChanged(f"new queue input is not visible yet: {root}")
        inputs[name] = _input_record(
            root, cfg, expected_profile_sha256=expected_profile_sha256)
    index.update(jobs=jobs, jobs_sha256=jobs_sha, inputs=inputs,
                 refreshed_utc=_utc())
    pb.atomic_json(index_path, index)
    return index


def fast_snapshot(dataset: Path, cfg: Any, index_path: Path, policy: BacklogPolicy,
                  expected_retry_workers: Sequence[str], *,
                  protected_hashes: Sequence[str] = (),
                  force_rebuild: bool = False,
                  expected_profile_sha256: str | None = None) -> dict[str, Any]:
    """Read queue/input/marker metadata only and return conservative accounting."""

    index = refresh_index(
        dataset, cfg, index_path, policy, force_rebuild=force_rebuild,
        expected_profile_sha256=expected_profile_sha256)
    _recheck_index_bindings(dataset, index)
    jobs_before, jobs_sha = _jobs_snapshot(dataset)
    if jobs_before != index["jobs"] or jobs_sha != index["jobs_sha256"]:
        raise cycle.LiveSnapshotChanged("queue advanced after keeper index refresh")
    workers = cycle._normalize_retry_workers(expected_retry_workers)
    own_pending: set[str] = set()
    inputs = index["inputs"]
    for job in jobs_before:
        record = inputs[str(job["input"])]
        if record.get("owner") != policy.owner:
            continue
        state_root = dataset / "jobs_state"
        first, payloads = cycle._job_state_binding(state_root, str(job["store"]))
        second, _ = cycle._job_state_binding(state_root, str(job["store"]))
        if first != second:
            raise cycle.LiveSnapshotChanged(f"worker marker changed: {job['store']}")
        terminal = first["done"]["exists"] or (
            first["fail"]["exists"] and cycle._terminal_fail(payloads["fail"], workers))
        if not terminal:
            own_pending.update(record["hashes"])
    jobs_after, jobs_after_sha = _jobs_snapshot(dataset)
    if jobs_after != jobs_before or jobs_after_sha != jobs_sha:
        raise cycle.LiveSnapshotChanged("queue advanced during keeper marker scan")
    # Close the scan window for immutable metadata too.  This second binding
    # check makes a coordinated metadata edit during marker inspection fatal
    # instead of allowing it to survive until the following poll.
    _recheck_index_bindings(dataset, index)
    physical = {value for record in inputs.values() for value in record["hashes"]}
    protected = {str(value) for value in protected_hashes}
    reserved = physical | protected
    target = int(cfg.exploration["target_valid_unique"])
    return {"jobs_sha256": jobs_sha, "jobs": jobs_before,
            "own_pending_hashes": own_pending, "own_pending": len(own_pending),
            "physical_hashes": physical, "protected_hashes": protected,
            "reserved_hashes": reserved, "reserved_unique": len(reserved),
            "capacity": max(0, target - len(reserved)), "target": target,
            "index": index}


def generate_reservoir(profile: Path, output: Path, *, seed: int, count: int,
                       owner: str) -> Path:
    """Create one stable blind-only reservoir; no SM or HFSS dependency."""

    cfg = pb.load_profile_config(profile)
    if cfg is None or cfg.port != "single":
        raise ValueError("keeper reservoir requires a named single-port profile")
    if output.exists():
        pb.validate_input(output, cfg)
        return output
    pending = output.with_name(f".{output.name}.pending")
    pending.mkdir(parents=True, exist_ok=True)
    shutil.copy2(profile, pending / "config.yaml")
    pb.atomic_json(pending / "measurement.json", cfg.measurement)
    pb.atomic_json(pending / "score_spec.json", cfg.score_spec)
    rng = np.random.default_rng(seed)
    rows, seen = [], set()
    attempts = 0
    while len(rows) < count and attempts < count * 100:
        attempts += 1
        pattern = ex.random_single_pattern(rng)
        digest = pb.validate_pattern(pattern, cfg)
        if digest in seen:
            continue
        seen.add(digest)
        name = f"r80kr{seed:08x}_{len(rows):05d}_{digest[:8]}"
        torch.save(torch.as_tensor(pattern, dtype=torch.float32), pending / f"{name}.pt")
        rows.append({"id": name, "port": cfg.port,
                     "measurement_id": measurement_id(cfg.measurement),
                     "score_spec_id": score_spec_id(cfg.score_spec),
                     "pattern_sha256": digest, "kind": "symmetry_factory",
                     "selection_arm": "blind_buffer", "candidate_group": output.name,
                     "source": owner, "lineage_id": name,
                     "predicted_before_hfss": False})
    if len(rows) != count:
        raise RuntimeError(f"keeper reservoir produced only {len(rows)}/{count} unique rows")
    pb.atomic_json(pending / "manifest.json", rows)
    pb.validate_input(pending, cfg)
    os.replace(pending, output)
    return output


def prepare_low_shard(profile: Path, pools: Sequence[Path], output_root: Path,
                      excluded: set[str], policy: BacklogPolicy, *,
                      validated_rows: Mapping[Path, Sequence[Mapping[str, Any]]] | None = None
                      ) -> tuple[Path, dict[str, Any]]:
    cfg = pb.load_profile_config(profile)
    if cfg is None:
        raise ValueError("keeper profile is not a named measurement profile")
    selected: list[tuple[Path, dict[str, Any]]] = []
    seen = set(excluded)
    for pool in pools:
        rows = (pb.validate_input(pool, cfg) if validated_rows is None else
                validated_rows[pool.resolve()])
        for row in rows:
            digest = str(row["pattern_sha256"])
            if digest in seen:
                continue
            seen.add(digest)
            selected.append((pool, row))
            if len(selected) == policy.shard_size:
                break
        if len(selected) == policy.shard_size:
            break
    if len(selected) != policy.shard_size:
        raise LookupError("keeper pools do not contain one complete unreserved shard")
    source_bindings = []
    for pool, row in selected:
        pool = pool.resolve()
        source_bindings.append({
            "pool": str(pool), "source_id": row["id"],
            "pattern_sha256": row["pattern_sha256"],
            "source_file_sha256": pb.file_sha256(pool / f"{row['id']}.pt"),
            "pool_metadata_sha256": {
                name: pb.file_sha256(pool / name) for name in
                ("config.yaml", "measurement.json", "score_spec.json", "manifest.json")}})
    preparation_receipt_sha256 = _digest({
        "profile_sha256": pb.file_sha256(profile), "sources": source_bindings})
    identity = {"schema_version": 1, "owner": policy.owner,
                "tier": "LOW", "priority": policy.priority, "scope": cfg.scope,
                "profile_sha256": pb.file_sha256(profile),
                "measurement_id": measurement_id(cfg.measurement),
                "score_spec_id": score_spec_id(cfg.score_spec),
                "source_bindings": source_bindings,
                "preparation_receipt_sha256": preparation_receipt_sha256,
                "hashes": [row["pattern_sha256"] for _, row in selected]}
    cohort = _digest(identity)
    output = output_root / f"cohort-{cohort[:16]}_input"
    if not output.exists():
        pending = output.with_name(f".{output.name}.pending")
        pending.mkdir(parents=True, exist_ok=True)
        shutil.copy2(profile, pending / "config.yaml")
        pb.atomic_json(pending / "measurement.json", cfg.measurement)
        pb.atomic_json(pending / "score_spec.json", cfg.score_spec)
        manifest = []
        for ordinal, (pool, source) in enumerate(selected):
            digest = str(source["pattern_sha256"])
            name = f"r80kl{ordinal:02d}_{digest[:12]}"
            shutil.copy2(pool / f"{source['id']}.pt", pending / f"{name}.pt")
            row = dict(source)
            row.update(id=name, keeper_owner=policy.owner, keeper_tier="LOW",
                       keeper_priority=policy.priority, blind_source_id=source["id"])
            manifest.append(row)
        pb.atomic_json(pending / "manifest.json", manifest)
        pb.atomic_json(pending / "backlog_job.json", {**identity, "cohort_id": cohort})
        pb.validate_input(pending, cfg)
        os.replace(pending, output)
    pb.validate_input(output, cfg)
    if pb.read_json(output / "backlog_job.json") != {**identity, "cohort_id": cohort}:
        raise ValueError("existing keeper shard identity differs")
    return output, {**identity, "cohort_id": cohort}


def _recheck_source_bindings(metadata: Mapping[str, Any]) -> None:
    sources = metadata.get("source_bindings")
    if not isinstance(sources, list) or len(sources) != len(metadata.get("hashes", [])):
        raise ValueError("keeper cohort lacks exact selected-source bindings")
    for item in sources:
        pool = Path(item["pool"]).resolve()
        expected = item.get("pool_metadata_sha256")
        actual = {name: pb.file_sha256(pool / name) for name in
                  ("config.yaml", "measurement.json", "score_spec.json", "manifest.json")}
        if actual != expected:
            raise ValueError(f"keeper source pool metadata changed: {pool}")
        rows = pb.read_json(pool / "manifest.json")
        matches = [row for row in rows if row.get("id") == item.get("source_id")]
        if len(matches) != 1 or matches[0].get("pattern_sha256") != item.get(
                "pattern_sha256"):
            raise ValueError(f"keeper source row differs from its binding: {pool}")
        source_file = pool / f"{item.get('source_id')}.pt"
        if pb.file_sha256(source_file) != item.get("source_file_sha256"):
            raise ValueError(f"keeper source tensor differs from its binding: {source_file}")
    expected_receipt = _digest({"profile_sha256": metadata.get("profile_sha256"),
                                "sources": sources})
    if metadata.get("preparation_receipt_sha256") != expected_receipt:
        raise ValueError("keeper preparation receipt binding differs")


def _commit_receipt(index_path: Path, cohort: str, job: Mapping[str, Any],
                    metadata: Mapping[str, Any], input_record: Mapping[str, Any]) -> dict[str, Any]:
    """Create or verify one deterministic receipt from canonical queue/input authority."""

    payload = {"schema_version": 1, "status": "committed", "cohort_id": cohort,
               "job": dict(job), "hashes": list(metadata["hashes"]),
               "profile_sha256": metadata["profile_sha256"],
               "preparation_receipt_sha256": metadata["preparation_receipt_sha256"],
               "source_bindings_sha256": _digest(metadata["source_bindings"]),
               "input_metadata_sha256": dict(input_record["metadata_sha256"])}
    path = index_path.parent / "commit_receipts" / f"{cohort}.json"
    if path.is_file():
        if pb.read_json(path) != payload:
            raise ValueError(f"immutable keeper commit receipt differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        pb.atomic_json(path, payload)
    return {"path": str(path.resolve()), "sha256": pb.file_sha256(path)}


def _reconcile_queued_receipts(snapshot: Mapping[str, Any], *, dataset: Path,
                               index_path: Path, cfg: Any, policy: BacklogPolicy,
                               recheck_sources: bool) -> list[dict[str, Any]]:
    """Create/verify receipts for canonical queued keeper cohorts after restart."""

    reconciled = []
    inputs = snapshot["index"]["inputs"]
    for job in snapshot["jobs"]:
        name = str(job["input"])
        physical = inputs[name]
        if physical.get("owner") != policy.owner:
            continue
        root = dataset / name
        metadata = pb.read_json(root / "backlog_job.json")
        cohort = str(metadata.get("cohort_id", ""))
        store = f"dedust_r80k{cohort[:16]}"
        expected = {"input": f"{store}_input", "store": store,
                    "prio": policy.priority, "scope": cfg.scope,
                    "config": f"{store}_input/config.yaml"}
        if job != expected or name != expected["input"]:
            raise ValueError(f"queued keeper cohort differs from canonical job: {name}")
        if recheck_sources:
            _recheck_source_bindings(metadata)
        reconciled.append(_commit_receipt(
            index_path, cohort, expected, metadata, physical))
    return reconciled


def _complete_copy(staged: Path, pending: Path) -> None:
    pending.mkdir(parents=True, exist_ok=True)
    staged_files = {path.relative_to(staged): path for path in staged.rglob("*") if path.is_file()}
    marker = Path("backlog_job.json")
    if marker not in staged_files:
        raise ValueError(f"keeper staged shard lacks ownership metadata: {staged}")
    marker_destination = pending / marker
    if marker_destination.is_file():
        if pb.file_sha256(marker_destination) != pb.file_sha256(staged_files[marker]):
            raise ValueError(f"conflicting partial keeper ownership: {marker_destination}")
    else:
        shutil.copy2(staged_files[marker], marker_destination)
    for existing in [path for path in pending.rglob("*") if path.is_file()]:
        relative = existing.relative_to(pending)
        if relative not in staged_files or pb.file_sha256(existing) != pb.file_sha256(
                staged_files[relative]):
            raise ValueError(f"conflicting partial keeper copy: {existing}")
    for relative, source in staged_files.items():
        if relative == marker:
            continue
        destination = pending / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.is_file():
            shutil.copy2(source, destination)


def recover_pending_inputs(dataset: Path, staging_root: Path, cfg: Any) -> list[Path]:
    """Complete only exact keeper-owned pending trees, then atomically publish them."""

    recovered = []
    for pending in sorted(dataset.glob(".dedust_r80k*_input.pending-*")):
        if not pending.is_dir():
            raise ValueError(f"keeper pending path is not a directory: {pending}")
        marker = pending / "backlog_job.json"
        suffix = pending.name.rsplit(".pending-", 1)[-1]
        if marker.is_file():
            metadata = pb.read_json(marker)
            cohort = str(metadata.get("cohort_id", ""))
            candidates = [staging_root / f"cohort-{cohort[:16]}_input"]
        else:
            candidates = sorted(staging_root.glob(f"cohort-{suffix}*_input"))
            if len(candidates) != 1:
                raise ValueError(
                    f"partial keeper copy lacks one deterministic local source: {pending}")
            metadata = pb.read_json(candidates[0] / "backlog_job.json")
            cohort = str(metadata.get("cohort_id", ""))
        if len(cohort) != 64:
            raise ValueError(f"partial keeper copy has invalid cohort identity: {pending}")
        staged = candidates[0]
        if not staged.is_dir() or pb.read_json(staged / "backlog_job.json") != metadata:
            raise ValueError(f"partial keeper copy lacks its exact local source: {pending}")
        store = f"dedust_r80k{cohort[:16]}"
        expected_pending = f".{store}_input.pending-{cohort[:12]}"
        if pending.name != expected_pending:
            raise ValueError(f"partial keeper copy name differs from its cohort: {pending}")
        expected_name = pending.name[1:].split(".pending-", 1)[0]
        destination = dataset / expected_name
        if destination.exists():
            raise ValueError(f"keeper pending and final destination both exist: {pending}")
        _complete_copy(staged, pending)
        pb.validate_input(pending, cfg)
        if cycle._tree_hashes(pending) != cycle._tree_hashes(staged):
            raise ValueError(f"completed keeper pending tree differs from source: {pending}")
        os.replace(pending, destination)
        recovered.append(destination)
    return recovered


def commit_low_shard(staged: Path, metadata: Mapping[str, Any], *, dataset: Path,
                     profile: Path, index_path: Path, policy: BacklogPolicy,
                     expected_retry_workers: Sequence[str],
                     coordinator: cycle.DatasetWriteCoordinator,
                     cancel_event: threading.Event | None = None,
                     queue_add: Callable = cycle._dedust_add,
                     duplicate_check: Callable = pb.check_duplicates) -> dict[str, Any]:
    """Commit at most one LOW shard under one priority-6 writer turn."""

    cfg = pb.load_profile_config(profile)
    if cfg is None:
        raise ValueError("keeper profile is not a named measurement profile")
    rows = pb.validate_input(staged, cfg)
    profile_sha256 = pb.file_sha256(profile)
    _recheck_source_bindings(metadata)
    hashes = {str(row["pattern_sha256"]) for row in rows}
    if len(rows) != policy.shard_size or hashes != set(metadata["hashes"]):
        raise ValueError("keeper staged shard differs from its cohort identity")
    cohort = str(metadata["cohort_id"])
    store = f"dedust_r80k{cohort[:16]}"
    job = {"kind": "b", "staged_input": str(staged.resolve()),
           "input": store + "_input", "store": store, "prio": policy.priority,
           "count": len(rows), "tree_sha256": cycle._content_id(cycle._tree_hashes(staged))}
    with coordinator.turn(policy.priority, cancel_event=cancel_event):
        with cycle._dataset_exclusive_lock(dataset, cfg.scope):
            snapshot = fast_snapshot(
                dataset, cfg, index_path, policy, expected_retry_workers,
                protected_hashes=coordinator.protected_hashes(),
                expected_profile_sha256=profile_sha256)
            existing_job = next((item for item in snapshot["jobs"]
                                 if item.get("store") == store), None)
            expected = {"input": job["input"], "store": store, "prio": policy.priority,
                        "scope": cfg.scope, "config": f"{job['input']}/config.yaml"}
            if existing_job is not None:
                if existing_job != expected:
                    raise ValueError(f"keeper queue store collision: {store}")
                physical = _input_record(
                    dataset / job["input"], cfg,
                    expected_profile_sha256=profile_sha256)
                receipt = _commit_receipt(
                    index_path, cohort, expected, metadata, physical)
                return {"status": "already_committed", "job": expected,
                        "own_pending": snapshot["own_pending"],
                        "reserved_unique": snapshot["reserved_unique"],
                        "commit_receipt": receipt}
            # The poll trigger is the low watermark.  Once a refill wave is
            # authorized, each independently rechecked shard may fill through
            # the hard target (96), never stopping again at 48.
            if snapshot["own_pending"] >= policy.target_pending:
                return {"status": "not_needed", "own_pending": snapshot["own_pending"],
                        "reserved_unique": snapshot["reserved_unique"]}
            unreserved = hashes - snapshot["reserved_hashes"]
            destination = dataset / job["input"]
            orphan = destination.is_dir()
            if orphan:
                physical = _input_record(
                    destination, cfg, expected_profile_sha256=profile_sha256)
                if set(physical["hashes"]) != hashes:
                    raise ValueError(f"keeper orphan destination differs: {destination}")
                unreserved = hashes
            if unreserved != hashes:
                return {"status": "replan", "reason": "candidate_became_reserved",
                        "reserved_unique": snapshot["reserved_unique"]}
            if not orphan and snapshot["capacity"] < policy.shard_size:
                return {"status": "capacity_tail", "capacity": snapshot["capacity"],
                        "reserved_unique": snapshot["reserved_unique"]}
            state_root = dataset / "jobs_state"
            if any((state_root / f"{store}.{suffix}").exists()
                   for suffix in ("claim", "done", "fail")):
                raise ValueError(f"unqueued keeper store already has worker state: {store}")
            pending = dataset / f".{job['input']}.pending-{cohort[:12]}"
            if not destination.exists():
                _complete_copy(staged, pending)
                pb.validate_input(pending, cfg)
                os.replace(pending, destination)
            elif cycle._tree_hashes(destination) != cycle._tree_hashes(staged):
                raise ValueError(f"immutable keeper input collision: {destination}")
            duplicate_check(destination, dataset)
            before = pb.read_json(dataset / "jobs.json")
            queue_add(dataset, job, cfg.scope)
            after = pb.read_json(dataset / "jobs.json")
            if after != [*before, expected]:
                raise ValueError("keeper queue mutation was not one exact append")
            # jobs.json is now an exact append, so extend the local reservation
            # index with only this new input.  Startup/orphan recovery owns the
            # expensive all-physical metadata rebuild.
            refresh_index(dataset, cfg, index_path, policy,
                          expected_profile_sha256=profile_sha256)
            final = fast_snapshot(dataset, cfg, index_path, policy, expected_retry_workers,
                                  protected_hashes=coordinator.protected_hashes(),
                                  expected_profile_sha256=profile_sha256)
            physical = _input_record(
                destination, cfg, expected_profile_sha256=profile_sha256)
            receipt = _commit_receipt(
                index_path, cohort, expected, metadata, physical)
            return {"status": "committed", "job": expected,
                    "own_pending": final["own_pending"],
                    "reserved_unique": final["reserved_unique"],
                    "capacity": final["capacity"],
                    "commit_receipt": receipt,
                    "conservative_over_target": final["reserved_unique"] > int(
                        cfg.exploration["target_valid_unique"])}


class BacklogMaintainer:
    """Interruptible LOW keeper hosted by the sole watcher process."""

    def __init__(self, *, settings: Mapping[str, Any], policy: BacklogPolicy,
                 coordinator: cycle.DatasetWriteCoordinator,
                 event_wait: Callable[[float], bool] | None = None) -> None:
        self.settings = dict(settings)
        self.policy = policy
        self.coordinator = coordinator
        self._stop = threading.Event()
        self._wait = event_wait or self._stop.wait
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._force_rebuild = True
        self.work = Path(self.settings["local_workdir"]).resolve() / "low_backlog"
        self.index_path = self.work / "reservation_index.json"
        self.events = self.work / "events"
        self.pools = [Path(self.settings["prepared_blind_pool"]).resolve()]
        self.pools.extend(sorted((self.work / "reservoirs").glob("blind-reservoir-v*")))
        self._pool_cache: dict[Path, tuple[dict[str, str], list[dict[str, Any]]]] = {}
        self.profile_sha256 = pb.file_sha256(Path(self.settings["profile_config"]))

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("LOW maintainer already started")
        # A dataset copy/queue transaction must finish before the watcher can
        # exit.  Coordinator waits remain interruptible through ``self._stop``;
        # an already-active transaction is joined to completion.
        self._thread = threading.Thread(target=self._run, name="r80-low-backlog", daemon=False)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            if self._thread is threading.current_thread():
                raise RuntimeError("LOW maintainer cannot join itself")
            self._thread.join()

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("LOW backlog maintainer failed") from self._error

    def _event(self, event: str, **fields: Any) -> Path:
        self.events.mkdir(parents=True, exist_ok=True)
        prior = sorted(self.events.glob("*.json"))
        previous = None
        for ordinal, path in enumerate(prior, 1):
            payload = pb.read_json(path)
            expected_name = f"{ordinal:08d}-{_digest(payload)[:16]}.json"
            if path.name != expected_name or payload.get(
                    "previous_receipt_sha256") != previous:
                raise ValueError(f"keeper event chain differs: {path}")
            previous = pb.file_sha256(path)
        payload = {"schema_version": 1, "event": event, "timestamp_utc": _utc(),
                   "previous_receipt_sha256": previous, **fields}
        name = f"{len(prior) + 1:08d}-{_digest(payload)[:16]}.json"
        path = self.events / name
        pb.atomic_json(path, payload)
        return path

    def _pool_rows(self, pool: Path, cfg: Any) -> list[dict[str, Any]]:
        pool = pool.resolve()
        files = ("config.yaml", "measurement.json", "score_spec.json", "manifest.json")
        binding = {name: pb.file_sha256(pool / name) for name in files}
        cached = self._pool_cache.get(pool)
        if cached is not None:
            if cached[0] != binding:
                raise ValueError(f"keeper pool metadata changed after validation: {pool}")
            return cached[1]
        rows = pb.validate_input(pool, cfg)
        if {name: pb.file_sha256(pool / name) for name in files} != binding:
            raise cycle.LiveSnapshotChanged(f"keeper pool changed during validation: {pool}")
        self._pool_cache[pool] = (binding, rows)
        return rows

    def _ensure_pool(self, excluded: set[str]) -> None:
        cfg = pb.load_profile_config(Path(self.settings["profile_config"]))
        while True:
            available = sum(
                row["pattern_sha256"] not in excluded
                for pool in self.pools for row in self._pool_rows(pool, cfg))
            if available >= self.policy.max_refill:
                return
            ordinal = len(self.pools)
            output = self.work / "reservoirs" / f"blind-reservoir-v{ordinal:04d}"
            output.parent.mkdir(parents=True, exist_ok=True)
            generate_reservoir(Path(self.settings["profile_config"]), output,
                               seed=self.policy.reservoir_seed + ordinal,
                               count=self.policy.reservoir_size, owner=self.policy.owner)
            self.pools.append(output)

    def poll_once(self) -> dict[str, Any]:
        dataset = Path(self.settings["dataset_root"]).resolve()
        profile = Path(self.settings["profile_config"]).resolve()
        cfg = pb.load_profile_config(profile)
        if cfg is None:
            raise ValueError("keeper profile is not a named measurement profile")
        startup_rebuild = self._force_rebuild
        if startup_rebuild:
            with self.coordinator.turn(self.policy.priority, cancel_event=self._stop):
                with cycle._dataset_exclusive_lock(dataset, cfg.scope):
                    recover_pending_inputs(dataset, self.work / "staged", cfg)
                    snapshot = fast_snapshot(
                        dataset, cfg, self.index_path, self.policy,
                        self.settings["expected_retry_workers"],
                        protected_hashes=self.coordinator.protected_hashes(),
                        force_rebuild=True,
                        expected_profile_sha256=self.profile_sha256)
        else:
            snapshot = fast_snapshot(
                dataset, cfg, self.index_path, self.policy,
                self.settings["expected_retry_workers"],
                protected_hashes=self.coordinator.protected_hashes(),
                expected_profile_sha256=self.profile_sha256)
        _reconcile_queued_receipts(
            snapshot, dataset=dataset, index_path=self.index_path, cfg=cfg,
            policy=self.policy, recheck_sources=startup_rebuild)
        self._force_rebuild = False
        queued_inputs = {str(job["input"]) for job in snapshot["jobs"]}
        orphans = [record for name, record in snapshot["index"]["inputs"].items()
                   if record.get("owner") == self.policy.owner and name not in queued_inputs]
        if orphans:
            record = orphans[0]
            staged = dataset / str(record["input"])
            metadata = pb.read_json(staged / "backlog_job.json")
            recovered = commit_low_shard(
                staged, metadata, dataset=dataset, profile=profile,
                index_path=self.index_path, policy=self.policy,
                expected_retry_workers=self.settings["expected_retry_workers"],
                coordinator=self.coordinator, cancel_event=self._stop)
            return {"status": "orphan_recovery", "commit": recovered}
        count = plan_refill(snapshot["own_pending"], snapshot["capacity"], self.policy)
        if count == 0:
            status = ("capacity_tail_deferred" if snapshot["capacity"] <
                      self.policy.min_refill else "not_needed")
            return {"status": status, "own_pending": snapshot["own_pending"],
                    "capacity": snapshot["capacity"],
                    "reserved_unique": snapshot["reserved_unique"],
                    "completion_claim": False}
        committed = []
        for _ in range(count // self.policy.shard_size):
            current = fast_snapshot(
                dataset, cfg, self.index_path, self.policy,
                self.settings["expected_retry_workers"],
                protected_hashes=self.coordinator.protected_hashes(),
                expected_profile_sha256=self.profile_sha256)
            if current["own_pending"] >= self.policy.target_pending:
                break
            self._ensure_pool(current["reserved_hashes"])
            staged, metadata = prepare_low_shard(
                profile, self.pools, self.work / "staged", current["reserved_hashes"],
                self.policy, validated_rows={pool.resolve(): self._pool_rows(pool, cfg)
                                             for pool in self.pools})
            result = commit_low_shard(
                staged, metadata, dataset=dataset, profile=profile,
                index_path=self.index_path, policy=self.policy,
                expected_retry_workers=self.settings["expected_retry_workers"],
                coordinator=self.coordinator, cancel_event=self._stop)
            committed.append(result)
            if result["status"] != "committed":
                break
        return {"status": "refilled" if committed else "not_needed",
                "planned": count, "commits": committed}

    def _run(self) -> None:
        try:
            self._event("keeper_started", policy=self.policy.__dict__)
            while not self._stop.is_set():
                try:
                    result = self.poll_once()
                    if result.get("status") != "not_needed":
                        self._event("keeper_poll", result=result)
                except cycle.CoordinatorCancelled:
                    if self._stop.is_set():
                        break
                    raise
                except (cycle.DatasetLockBusy, cycle.LiveSnapshotChanged) as exc:
                    self._event("live_snapshot_deferred", exception_type=type(exc).__name__,
                                error=str(exc))
                    self._force_rebuild = isinstance(exc, cycle.DatasetLockBusy)
                if self._wait(self.policy.poll_seconds):
                    break
            self._event("keeper_stopped")
        except BaseException as exc:
            self._error = exc
            self._event("keeper_failed", exception_type=type(exc).__name__, error=str(exc),
                        traceback=traceback.format_exc())
