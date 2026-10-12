"""Additive, fail-closed R81 formal entry for three native 20-item workers.

The worker-facing paths in this module are read-only.  ``--aggregate-batch`` is
an explicit coordinator action which materializes one fresh logical 60-item
store only after all three native stores have completed and replayed cleanly.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from antenna.measurement import measurement_id, score_spec_id
from script import exploration as ex
from script import prepare_symmetry_filter as wide
from script import profiled_batch as pb
from script import profiled_shards
from script import r81_worker_entry as base


REPO = Path(__file__).resolve().parents[1]
STAGE = "FormalSharded"
SCOPE = base.SCOPE
FORMAL_ONLY_FILES = {
    "script/profiled_shards.py",
    "script/r81_formal_sharded_entry.py",
    "script/start_r81_formal_sharded_worker.ps1",
}
INPUT_METADATA = ("measurement.json", "score_spec.json", "config.yaml", "manifest.json")
STORE_METADATA = ("measurement.json", "score_spec.json", "manifest.json", "results.json")
COMPLETE_NAME = "formal_merge_complete.json"
ENGINEERING_REUSE_KIND = "engineering_original_source_reuse"


def _transition_used(binding: Any) -> bool:
    detector = getattr(base, "_exploration_transition_used", None)
    if callable(detector):
        return bool(detector(binding))
    if not isinstance(binding, Mapping):
        return False
    if "transition_selection_receipts" in binding:
        return True
    files = binding.get("files")
    if isinstance(files, Mapping) and any(
            Path(str(name)).parts[:1] == ("transition_selection",) for name in files):
        return True
    work_value = binding.get("work_dir")
    if not isinstance(work_value, str) or not Path(work_value).is_absolute():
        return False
    work = Path(work_value)
    try:
        state = ex._read_json(work / "state.json")
        for key, info in state.get("batches", {}).items():
            if int(key) <= 1 or not isinstance(info, dict):
                continue
            rows = ex._manifest_rows(
                ex._safe_input_path(work, info["input_dir"]) / "manifest.json")
            if any("parent_truth_provenance" in row for row in rows):
                return True
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        # The normal exploration validator reports malformed/missing inputs.
        return False
    return False


def _runtime_files(binding: Any) -> tuple[str, ...]:
    transition = _transition_used(binding)
    if transition and not hasattr(base, "TRANSITION_RUNTIME_FILES"):
        raise ValueError("transition evidence requires the integrated transition-aware entry")
    inherited = (base.TRANSITION_RUNTIME_FILES if transition
                 else base.RUNTIME_FILES)
    return tuple(sorted(set(inherited) | FORMAL_ONLY_FILES))


def _validate_runtime(runtime: Any, exploration: Any) -> None:
    value = base._exact_keys(runtime, {"git_commit", "files"}, "runtime")
    commit = value["git_commit"]
    if not isinstance(commit, str) or base.HEX40.fullmatch(commit) is None:
        raise ValueError("runtime.git_commit must be a full lowercase commit id")
    if not base._git_is_ancestor(commit):
        raise ValueError("runtime Git commit is not an ancestor of this checkout")
    required = _runtime_files(exploration)
    files = value["files"]
    if not isinstance(files, dict) or set(files) != set(required):
        raise ValueError("runtime.files must bind the exact formal-sharded producer set")
    for relative in required:
        expected = base._sha(files[relative], f"runtime.files[{relative}]")
        if (base._git_blob_sha(commit, relative) != expected or
                not base._git_path_clean(commit, relative)):
            raise ValueError(f"runtime source changed: {relative}")


def _safe_relative(value: Any, label: str, *, parts: int | None = None) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a safe relative path")
    path = Path(value)
    if (path.is_absolute() or not path.parts or ".." in path.parts or
            path.as_posix() != value or any(part in {"", "."} for part in path.parts)):
        raise ValueError(f"{label} must be a safe relative path")
    if parts is not None and len(path.parts) != parts:
        raise ValueError(f"{label} must contain exactly {parts} path components")
    for part in path.parts:
        base._name(part, f"{label} component")
    return value


def _metadata_hashes(root: Path, names: tuple[str, ...]) -> dict[str, str]:
    return {name: pb.file_sha256(root / name) for name in names}


def _source_file(dataset: Path, relative: str, label: str) -> Path:
    """Resolve one receipt member without permitting an escape from its dataset."""
    safe = _safe_relative(relative, label)
    try:
        root = dataset.resolve(strict=True)
        path = (dataset / Path(safe)).resolve(strict=True)
        path.relative_to(root)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise ValueError(f"{label} escapes or is missing from the source dataset") from exc
    if not path.is_file():
        raise ValueError(f"{label} is not a source file")
    return path


def _source_file_closure(dataset: Path, entries: list[dict[str, Any]]) -> dict[str, str]:
    """Hash exactly the original engineering files consumed by the formal gate."""
    relative_files = {"jobs.json"}
    for entry in entries:
        job = entry["job"]
        input_name, store = job["input"], job["store"]
        for name in entry["input_files"]:
            relative_files.add((Path(input_name) / name).as_posix())
        for name in STORE_METADATA:
            relative_files.add((Path(store) / name).as_posix())
        results = pb.read_json(dataset / store / "results.json")
        if not isinstance(results, dict):
            raise ValueError(f"engineering source results are malformed: {store}")
        for row_id, result in results.items():
            if not isinstance(result, dict):
                raise ValueError(f"engineering source result is malformed: {store}/{row_id}")
            for file_key, sha_key in (("sample_file", "sample_sha256"),
                                      ("rad_file", "rad_sha256")):
                name = result.get(file_key)
                if name is None:
                    continue
                relative = (Path(store) / _safe_relative(
                    name, f"engineering source {store}/{row_id}.{file_key}")).as_posix()
                path = _source_file(dataset, relative,
                                    f"engineering source {store}/{row_id}.{file_key}")
                if pb.file_sha256(path) != result.get(sha_key):
                    raise ValueError(f"engineering source result hash differs: {store}/{row_id}")
                relative_files.add(relative)
        relative_files.add((Path("jobs_state") / f"{store}.claim").as_posix())
        relative_files.add((Path("jobs_state") / f"{store}.done").as_posix())
    return {relative: pb.file_sha256(_source_file(
                dataset, relative, f"engineering source files[{relative}]"))
            for relative in sorted(relative_files)}


def _validate_shard_input(dataset: Path, entry: dict[str, Any], label: str):
    job = entry["job"]
    if (not isinstance(job, dict) or
            set(job) != {"input", "store", "prio", "config", "scope"}):
        raise ValueError(f"{label}.job fields differ")
    input_name = _safe_relative(job["input"], f"{label}.job.input", parts=2)
    base._name(job["store"], f"{label}.job.store")
    if (job["scope"] != SCOPE or
            job["config"] != f"{input_name}/config.yaml"):
        raise ValueError(f"{label}.job must use its scoped shard config snapshot")
    if isinstance(job["prio"], bool) or not isinstance(job["prio"], int):
        raise ValueError(f"{label}.job.prio must be an integer")
    folder = dataset / Path(input_name)
    base._validate_file_map(folder, entry["input_files"], f"{label}.input_files")
    cfg = pb.load_profile_config(folder / "config.yaml")
    if cfg is None or cfg.scope != SCOPE or cfg.port != "dual":
        raise ValueError(f"{label} does not use the R81 dual profile scope")
    rows = pb.validate_input(folder, cfg)
    profile = {"measurement_id": measurement_id(cfg.measurement),
               "score_spec_id": score_spec_id(cfg.score_spec),
               "runtime": cfg.runtime}
    return profile, rows, folder


def _read_marker(path: Path, allowed: set[str], required: set[str], label: str):
    return base._marker_json(path, allowed, required, label) if path.exists() else None


def _job_state(dataset: Path, entry: dict[str, Any], roster: tuple[str, ...]) -> str:
    job = entry["job"]
    store = job["store"]
    state_root = dataset / "jobs_state"
    claim = _read_marker(state_root / f"{store}.claim", {"machine", "at", "prior_fail"},
                         {"machine", "at"}, f"{store}.claim")
    done = _read_marker(state_root / f"{store}.done",
                        {"machine", "at", "errors", "error_ids"},
                        {"machine", "at", "errors", "error_ids"}, f"{store}.done")
    fail = _read_marker(state_root / f"{store}.fail",
                        {"machines", "last", "at", "failure_kind", "worker_continues"},
                        {"machines", "last", "at"}, f"{store}.fail")
    if claim is not None:
        if claim["machine"] not in roster:
            raise ValueError(f"{store}.claim names an unauthorized worker")
        if "prior_fail" in claim:
            prior = base._marker_machines(claim["prior_fail"], roster,
                                          f"{store}.claim.prior_fail")
            if claim["machine"] in prior:
                raise ValueError(f"{store}.claim repeats its owner in prior_fail")
    if done is not None:
        if (done["machine"] not in roster or isinstance(done["errors"], bool) or
                not isinstance(done["errors"], int) or done["errors"] < 0 or
                not isinstance(done["error_ids"], list) or
                any(not isinstance(item, str) for item in done["error_ids"]) or
                done["errors"] < len(done["error_ids"])):
            raise ValueError(f"{store}.done marker fields are invalid")
    if fail is not None:
        base._marker_machines(fail["machines"], roster, f"{store}.fail.machines")
        if not isinstance(fail["last"], str):
            raise ValueError(f"{store}.fail.last is invalid")
    if done is not None and fail is not None:
        raise ValueError(f"{store} has contradictory done and fail markers")
    if done is not None and (claim is None or done["machine"] != claim["machine"]):
        raise ValueError(f"{store}.done lacks its matching native claim")
    if fail is not None:
        expected = ([] if claim is None else claim.get("prior_fail", [])) + (
            [] if claim is None else [claim["machine"]])
        if claim is None or fail["machines"] != expected:
            raise ValueError(f"{store}.fail differs from its native claim retry chain")
    if done is not None:
        input_dir = dataset / Path(job["input"])
        store_dir = dataset / store
        for name in ("measurement.json", "score_spec.json", "manifest.json"):
            if pb.read_json(input_dir / name) != pb.read_json(store_dir / name):
                raise ValueError(f"{store} completed store differs from released {name}")
        finished = pb.validate_finished_store(store_dir)
        skipped = finished["skipped_ids"]
        if done["errors"] != len(skipped) or done["error_ids"] != skipped:
            raise ValueError(f"{store}.done does not describe the verified execution terminal")
        return "skipped" if skipped else "done"
    if fail is not None and set(fail["machines"]) == set(roster):
        return "failed"
    return "active"


def _engineering_terminal(dataset: Path, entries: list[dict[str, Any]],
                          roster: tuple[str, ...]) -> dict[str, Any]:
    seen: dict[str, dict[str, Any]] = {}
    profile = None
    for index, entry in enumerate(entries):
        if entry.get("role") not in base.ENGINEERING:
            continue
        label = f"queue.jobs[{index}]"
        base._exact_keys(entry, {"role", "job", "input_files"}, label)
        role = entry["role"]
        if role in seen:
            raise ValueError(f"duplicate engineering role: {role}")
        current, rows = base._validate_input(dataset, entry, label)
        expected_store, expected_count = base.ENGINEERING[role]
        if (entry["job"]["store"] != expected_store or
                entry["job"]["input"] != expected_store + "_input" or
                len(rows) != expected_count):
            raise ValueError(f"{label} differs from the exact engineering job")
        if _job_state(dataset, entry, roster) != "done":
            raise ValueError("formal sharded release requires engineering all-success terminal")
        seen[role] = entry
        if role == "engineering_smoke":
            profile = current
    if set(seen) != set(base.ENGINEERING) or profile is None:
        raise ValueError("queue must contain each engineering job exactly once")
    return profile


def _engineering_source_context(path: str | Path, required_sha256: str):
    """Authenticate an original EngineeringOnly release and its completed truth."""
    if not isinstance(path, (str, os.PathLike)):
        raise ValueError("engineering source release path must be a path")
    release_path = Path(path)
    if not release_path.is_absolute():
        raise ValueError("engineering source release path must be absolute")
    expected_sha = base._sha(required_sha256, "engineering source release sha256")
    if pb.file_sha256(release_path) != expected_sha:
        raise ValueError("engineering source release hash differs")
    base.validate_release(release_path, expected_sha, "EngineeringOnly")
    release = pb.read_json(release_path)
    canonical_release = release_path.resolve(strict=True)
    canonical_dataset = Path(release["dataset_root"]).resolve(strict=True)
    entries = release["queue"]["jobs"]
    roster = base._retry_workers(release["queue"]["expected_retry_workers"])
    if (len(entries) != len(base.ENGINEERING) or
            Counter(entry.get("role") for entry in entries) !=
            Counter(base.ENGINEERING.keys())):
        raise ValueError("engineering source release must contain only the exact three engineering jobs")
    known_stores = {entry["job"]["store"] for entry in entries}
    state_root = canonical_dataset / "jobs_state"
    if state_root.exists():
        for marker in state_root.iterdir():
            if (marker.is_file() and marker.suffix in {".claim", ".done", ".fail"} and
                    marker.stem not in known_stores):
                raise ValueError(
                    f"engineering source jobs_state contains an unknown marker: {marker.name}")
    source_files = _source_file_closure(canonical_dataset, entries)
    profile = _engineering_terminal(canonical_dataset, entries, roster)
    check = wide.measurement_check(canonical_dataset)
    if (check.get("engineering_check_passed") is not True or
            check.get("full_batch_release_allowed") is not True or
            len(check.get("comparisons", [])) != 4 or
            any(max(row.get("max_curve_difference_db", float("inf")),
                    row.get("max_margin_difference_db", float("inf"))) > .3
                for row in check["comparisons"])):
        raise ValueError("source R81 engineering measurement check did not pass all four comparisons")
    if (_source_file_closure(canonical_dataset, entries) != source_files or
            pb.file_sha256(canonical_release) != expected_sha):
        raise ValueError("engineering source changed while its truth was authenticated")
    receipt = {
        "schema_version": 1,
        "kind": ENGINEERING_REUSE_KIND,
        "source_release": {"path": str(canonical_release), "sha256": expected_sha},
        "dataset_root": str(canonical_dataset),
        "jobs_sha256": pb.file_sha256(canonical_dataset / "jobs.json"),
        "source_files": source_files,
    }
    return receipt, profile, check, canonical_dataset


def engineering_source_binding(path: str | Path, required_sha256: str) -> dict[str, Any]:
    """Return the canonical read-only receipt for reusable engineering evidence."""
    receipt, _profile, _check, _dataset = _engineering_source_context(
        path, required_sha256)
    return receipt


def _validate_engineering_reuse(binding: Any, exploration: Any, dataset: Path):
    value = base._exact_keys(binding, {"source_release", "receipt"},
                             "queue.engineering_reuse")
    source_release = base._exact_keys(
        value["source_release"], {"path", "sha256"},
        "queue.engineering_reuse.source_release")
    receipt_binding = base._exact_keys(
        value["receipt"], {"name", "sha256"},
        "queue.engineering_reuse.receipt")
    if receipt_binding["name"] != "engineering_source_reuse.json":
        raise ValueError("engineering reuse receipt must use its fixed name")
    receipt_sha = base._sha(receipt_binding["sha256"],
                            "queue.engineering_reuse.receipt.sha256")
    if not isinstance(exploration, dict):
        raise ValueError("engineering reuse requires an exploration controller")
    work_value, files = exploration.get("work_dir"), exploration.get("files")
    if not isinstance(work_value, str) or not Path(work_value).is_absolute():
        raise ValueError("engineering reuse work_dir must be absolute")
    if (not isinstance(files, dict) or
            files.get(receipt_binding["name"]) != receipt_sha):
        raise ValueError("engineering reuse receipt is not bound by exploration.files")
    receipt_path = _source_file(Path(work_value), receipt_binding["name"],
                                "engineering reuse receipt")
    if pb.file_sha256(receipt_path) != receipt_sha:
        raise ValueError("engineering reuse receipt hash differs")
    expected, profile, check, source_dataset = _engineering_source_context(
        source_release.get("path"), source_release.get("sha256"))
    if source_release != expected["source_release"]:
        raise ValueError("engineering reuse source release binding is not canonical")
    if base._nested(source_dataset, dataset.resolve(strict=True)):
        raise ValueError("engineering reuse source and formal datasets must not overlap")
    if pb.read_json(receipt_path) != expected:
        raise ValueError("engineering reuse receipt differs from authenticated source closure")
    return profile, check, expected


def _canonical_input(dataset: Path, binding: Any, label: str):
    value = base._exact_keys(binding, {"name", "files"}, label)
    name = base._name(value["name"], f"{label}.name")
    folder = dataset / name
    base._validate_file_map(folder, value["files"], f"{label}.files")
    cfg = pb.load_profile_config(folder / "config.yaml")
    if cfg is None or cfg.scope != SCOPE or cfg.port != "dual":
        raise ValueError(f"{label} is not a scoped dual profile")
    rows = pb.validate_input(folder, cfg)
    return name, folder, cfg, rows


def _validate_split(dataset: Path, batch: int, plan: dict[str, Any],
                    entries: list[dict[str, Any]], expected_profile: dict[str, Any]):
    label = f"queue.formal_shards[{batch}]"
    value = base._exact_keys(
        plan,
        {"canonical_input", "canonical_input_files", "split_root",
         "split_receipt_sha256", "logical_store"}, label)
    canonical_name, canonical, cfg, canonical_rows = _canonical_input(
        dataset, {"name": value["canonical_input"], "files": value["canonical_input_files"]},
        f"{label}.canonical")
    if len(canonical_rows) != 60 or any(row.get("batch") != batch for row in canonical_rows):
        raise ValueError(f"logical formal batch {batch} must contain exactly 60 matching rows")
    expected_arms = ({"history": 20, "specialist": 20, "random": 20} if batch == 1 else
                     {"best_min_margin": 20, "uncertainty": 20, "random": 20})
    if Counter(row.get("selection_arm") for row in canonical_rows) != expected_arms:
        raise ValueError(f"logical formal batch {batch} arm allocation differs")
    profile = {"measurement_id": measurement_id(cfg.measurement),
               "score_spec_id": score_spec_id(cfg.score_spec), "runtime": cfg.runtime}
    if profile != expected_profile:
        raise ValueError(f"logical formal batch {batch} profile differs from engineering")
    split_root_name = _safe_relative(value["split_root"], f"{label}.split_root", parts=1)
    split_root = dataset / split_root_name
    receipt_path = split_root / "split_receipt.json"
    if pb.file_sha256(receipt_path) != base._sha(
            value["split_receipt_sha256"], f"{label}.split_receipt_sha256"):
        raise ValueError(f"logical formal batch {batch} split receipt hash differs")
    receipt = pb.read_json(receipt_path)
    base._exact_keys(receipt, {"schema_version", "canonical_input",
                               "canonical_manifest_sha256", "canonical_metadata_sha256",
                               "config_sha256", "measurement_id", "score_spec_id",
                               "round_robin", "shard_size", "shards"},
                     f"{label}.split_receipt")
    if (receipt["schema_version"] != 1 or
            base._norm(receipt["canonical_input"]) != base._norm(canonical) or
            receipt["canonical_manifest_sha256"] != pb.file_sha256(canonical / "manifest.json") or
            receipt["canonical_metadata_sha256"] != _metadata_hashes(canonical, INPUT_METADATA) or
            receipt["config_sha256"] != pb.file_sha256(canonical / "config.yaml") or
            receipt["measurement_id"] != profile["measurement_id"] or
            receipt["score_spec_id"] != profile["score_spec_id"] or
            receipt["round_robin"] is not True or receipt["shard_size"] != 20):
        raise ValueError(f"logical formal batch {batch} split receipt differs from canonical input")
    shards = receipt["shards"]
    if not isinstance(shards, list) or len(shards) != 3:
        raise ValueError(f"logical formal batch {batch} needs exactly three split shards")
    by_number = {entry["shard"]: entry for entry in entries}
    if set(by_number) != {1, 2, 3} or len(entries) != 3:
        raise ValueError(f"logical formal batch {batch} needs physical shards 1, 2, 3")
    stores = set()
    for offset, shard_receipt in enumerate(shards):
        shard_no = offset + 1
        s_label = f"{label}.shards[{shard_no}]"
        base._exact_keys(shard_receipt, {"name", "ids", "items", "manifest_sha256"}, s_label)
        expected_name = f"shard-{shard_no:03d}_input"
        if shard_receipt["name"] != expected_name:
            raise ValueError(f"{s_label} name differs")
        entry = by_number[shard_no]
        expected_input = f"{split_root_name}/{expected_name}"
        if entry["job"]["input"] != expected_input:
            raise ValueError(f"{s_label} job input is not its native split shard")
        child_profile, child_rows, child = _validate_shard_input(dataset, entry, s_label)
        expected_rows = canonical_rows[offset::3]
        if child_profile != profile or child_rows != expected_rows or len(child_rows) != 20:
            raise ValueError(f"{s_label} is not the exact round-robin canonical partition")
        ids = [row["id"] for row in child_rows]
        expected_items = [{"id": row["id"], "pattern_file": row["pattern_file"],
                           "file_sha256": pb.file_sha256(child / row["pattern_file"]),
                           "pattern_sha256": row["pattern_sha256"]}
                          for row in child_rows]
        if (shard_receipt["ids"] != ids or shard_receipt["items"] != expected_items or
                shard_receipt["manifest_sha256"] != pb.file_sha256(child / "manifest.json")):
            raise ValueError(f"{s_label} receipt differs from physical input")
        store = entry["job"]["store"]
        if store in stores:
            raise ValueError(f"logical formal batch {batch} repeats a physical store")
        stores.add(store)
    logical_store = base._name(value["logical_store"], f"{label}.logical_store")
    if logical_store in stores:
        raise ValueError("logical store must differ from every physical store")
    return {"batch": batch, "canonical_name": canonical_name, "canonical": canonical,
            "canonical_rows": canonical_rows, "split_root": split_root,
            "split_receipt_sha256": value["split_receipt_sha256"],
            "entries": [by_number[i] for i in (1, 2, 3)],
            "logical_store": logical_store, "profile": profile}


def _validate_queue(queue: Any, dataset: Path) -> dict[str, Any]:
    reuse = isinstance(queue, dict) and "engineering_reuse" in queue
    expected_keys = {"jobs_sha256", "jobs", "formal_input", "formal_shards",
                     "exploration", "expected_retry_workers"}
    if reuse:
        expected_keys.add("engineering_reuse")
    value = base._exact_keys(
        queue, expected_keys, "queue")
    roster = base._retry_workers(value["expected_retry_workers"])
    jobs_path = dataset / "jobs.json"
    if pb.file_sha256(jobs_path) != base._sha(value["jobs_sha256"], "queue.jobs_sha256"):
        raise ValueError("jobs.json differs from the release")
    entries = value["jobs"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("queue.jobs must be a nonempty list")
    if pb.read_json(jobs_path) != [entry.get("job") for entry in entries if isinstance(entry, dict)]:
        raise ValueError("queue contains unknown, reordered, or changed jobs")
    if reuse:
        if any(isinstance(entry, dict) and entry.get("role") in base.ENGINEERING
               for entry in entries):
            raise ValueError("engineering reuse queue must not dispatch engineering jobs")
        expected_profile, check, reuse_receipt = _validate_engineering_reuse(
            value["engineering_reuse"], value["exploration"], dataset)
    else:
        expected_profile = _engineering_terminal(dataset, entries, roster)
        check = wide.measurement_check(dataset)
        reuse_receipt = None
    formal_map = value["formal_shards"]
    if not isinstance(formal_map, dict):
        raise ValueError("queue.formal_shards must be a batch mapping")
    try:
        batches = sorted(int(key) for key in formal_map)
    except (TypeError, ValueError) as exc:
        raise ValueError("queue.formal_shards keys must be integer strings") from exc
    if batches not in ([1], [1, 2], [1, 2, 3]) or set(formal_map) != {str(x) for x in batches}:
        raise ValueError("formal shards must be a nonempty contiguous prefix of batches 1..3")
    shard_entries: dict[int, list[dict[str, Any]]] = {batch: [] for batch in batches}
    repeat_entries = []
    for index, entry in enumerate(entries):
        label = f"queue.jobs[{index}]"
        if not isinstance(entry, dict) or not isinstance(entry.get("role"), str):
            raise ValueError(f"{label} is malformed")
        role = entry["role"]
        if role in base.ENGINEERING:
            if reuse:
                raise ValueError("engineering reuse queue must not dispatch engineering jobs")
            continue
        if role == "formal_shard":
            base._exact_keys(entry, {"role", "batch", "shard", "job", "input_files"}, label)
            batch, shard = entry["batch"], entry["shard"]
            if (isinstance(batch, bool) or not isinstance(batch, int) or batch not in batches or
                    isinstance(shard, bool) or not isinstance(shard, int) or shard not in {1, 2, 3}):
                raise ValueError(f"{label} batch/shard differs")
            shard_entries[batch].append(entry)
        elif role == "positive_repeat":
            base._exact_keys(entry, {"role", "job", "input_files", "candidate"}, label)
            profile, rows = base._validate_input(dataset, entry, label)
            base._validate_candidate(dataset, entry["candidate"], rows, profile, label)
            if profile != expected_profile:
                raise ValueError("positive repeat profile differs from engineering")
            repeat_entries.append(entry)
        else:
            raise ValueError(f"unknown R81 formal-sharded job role: {role}")
    plans = {batch: _validate_split(dataset, batch, formal_map[str(batch)],
                                    shard_entries[batch], expected_profile)
             for batch in batches}
    formal = base._exact_keys(value["formal_input"], {"name", "files"}, "queue.formal_input")
    if (formal["name"] != plans[1]["canonical_name"] or
            formal["files"] != formal_map["1"]["canonical_input_files"]):
        raise ValueError("formal_input must be the canonical batch-1 logical bundle")
    synthetic = {batch: {"job": {"input": plan["canonical_name"]}}
                 for batch, plan in plans.items()}
    if _transition_used(value["exploration"]) and not hasattr(base, "TRANSITION_RUNTIME_FILES"):
        raise ValueError("transition evidence cannot be checked by this base entry")
    base._validate_exploration(value["exploration"], dataset, len(batches),
                               synthetic, expected_profile)
    if (check.get("engineering_check_passed") is not True or
            check.get("full_batch_release_allowed") is not True or
            len(check.get("comparisons", [])) != 4 or
            any(max(row.get("max_curve_difference_db", float("inf")),
                    row.get("max_margin_difference_db", float("inf"))) > .3
                for row in check["comparisons"])):
        raise ValueError("fresh R81 engineering measurement check did not pass all four comparisons")
    queued_stores = [entry["job"]["store"] for entry in entries]
    if len(set(queued_stores)) != len(queued_stores):
        raise ValueError("queue repeats a physical store")
    logical_stores = [plan["logical_store"] for plan in plans.values()]
    if (len(set(logical_stores)) != len(logical_stores) or
            set(logical_stores).intersection(queued_stores)):
        raise ValueError("logical stores must be unique and distinct from all physical stores")
    state_root = dataset / "jobs_state"
    if state_root.exists():
        known = set(queued_stores)
        for marker in state_root.iterdir():
            if marker.is_file() and marker.suffix in {".claim", ".done", ".fail"}:
                if marker.stem not in known:
                    raise ValueError(f"jobs_state contains an unknown released store marker: {marker.name}")
    for batch in batches[:-1]:
        _validate_logical_store(dataset, plans[batch], roster, require_receipt=True)
    return {"roster": roster, "entries": entries, "plans": plans,
            "engineering_check": check, "engineering_reuse": reuse_receipt,
            "repeat_entries": repeat_entries}


def _context(path: str | Path, required_sha256: str) -> dict[str, Any]:
    release_path = Path(path)
    expected_sha = base._sha(required_sha256, "release sha256")
    if pb.file_sha256(release_path) != expected_sha:
        raise ValueError("release metadata hash differs")
    release = pb.read_json(release_path)
    value = base._exact_keys(
        release, {"schema_version", "release_id", "stage", "dataset_root",
                  "runtime", "r80_completion", "queue"}, "release")
    if value["schema_version"] != 1 or value["stage"] != STAGE:
        raise ValueError("unsupported formal-sharded release schema or stage")
    base._name(value["release_id"], "release.release_id")
    dataset = Path(value["dataset_root"])
    if not dataset.is_absolute() or "r81" not in str(dataset).lower():
        raise ValueError("dataset_root must be an explicit separate R81 path")
    _validate_runtime(value["runtime"], value["queue"].get("exploration"))
    r80 = base._validate_r80(value["r80_completion"], dataset,
                             value["runtime"]["git_commit"])
    queue = _validate_queue(value["queue"], dataset)
    context = {"release": value, "release_path": release_path.resolve(),
               "release_sha256": expected_sha, "dataset": dataset, "r80": r80,
               **queue}
    for batch in sorted(context["plans"])[:-1]:
        if not _feedback_imported(context, batch):
            raise ValueError(f"batch {batch} logical feedback must precede the next prefix")
    return context


def validate_release(path: str | Path, required_sha256: str) -> dict[str, Any]:
    context = _context(path, required_sha256)
    return {"schema_version": 1, "status": "r81_formal_sharded_entry_passed",
            "release_sha256": context["release_sha256"],
            "release_id": context["release"]["release_id"], "stage": STAGE,
            "dataset_root": str(context["dataset"]), "scope": SCOPE,
            "logical_batch_count": len(context["plans"]),
            "physical_formal_job_count": 3 * len(context["plans"]),
            "job_count": len(context["entries"]), "r80": context["r80"],
            "engineering_check": context["engineering_check"],
            "hfss_started": False, "queue_or_source_mutation": False,
            "aggregation_performed": False}


def _source_binding(dataset: Path, plan: dict[str, Any]) -> dict[str, Any]:
    logical = dataset / plan["logical_store"]
    results = pb.validate_store(logical, require_complete=True)
    rows = pb.read_json(logical / "manifest.json")
    if rows != plan["canonical_rows"] or list(results) != [row["id"] for row in rows]:
        raise ValueError("logical merged store is not the exact ordered canonical batch")
    bindings = pb.read_json(logical / "source_bindings.json")
    base._exact_keys(
        bindings,
        {"schema_version", "canonical_input", "canonical_manifest_sha256",
         "canonical_metadata_sha256", "measurement_id", "score_spec_id", "stores", "items"},
        "logical source bindings")
    stores = [dataset / entry["job"]["store"] for entry in plan["entries"]]
    if (bindings.get("schema_version") != 1 or
            base._norm(bindings.get("canonical_input", "")) != base._norm(plan["canonical"]) or
            bindings.get("canonical_manifest_sha256") !=
            pb.file_sha256(plan["canonical"] / "manifest.json") or
            bindings.get("canonical_metadata_sha256") !=
            _metadata_hashes(plan["canonical"], INPUT_METADATA) or
            bindings.get("measurement_id") != plan["profile"]["measurement_id"] or
            bindings.get("score_spec_id") != plan["profile"]["score_spec_id"] or
            [base._norm(item.get("store", "")) for item in bindings.get("stores", [])] !=
            [base._norm(store) for store in stores]):
        raise ValueError("logical source bindings differ from the released shard plan")
    by_id = {}
    physical = []
    for entry, store, store_binding in zip(plan["entries"], stores, bindings["stores"]):
        base._exact_keys(store_binding,
                         {"store", "ids", "manifest_sha256", "results_sha256",
                          "metadata_sha256"},
                         f"logical source store {entry['shard']}")
        input_dir = dataset / Path(entry["job"]["input"])
        input_rows = pb.read_json(input_dir / "manifest.json")
        store_rows = pb.read_json(store / "manifest.json")
        store_results = pb.read_json(store / "results.json")
        store_metadata = _metadata_hashes(store, STORE_METADATA)
        if (store_rows != input_rows or
                store_binding.get("ids") != [row["id"] for row in input_rows] or
                store_binding.get("manifest_sha256") != store_metadata["manifest.json"] or
                store_binding.get("results_sha256") != store_metadata["results.json"] or
                store_binding.get("metadata_sha256") != store_metadata):
            raise ValueError("logical source store binding differs from a physical shard")
        physical.append({"shard": entry["shard"], "input": entry["job"]["input"],
                         "store": entry["job"]["store"],
                         "input_manifest_sha256": pb.file_sha256(input_dir / "manifest.json"),
                         "store_manifest_sha256": store_metadata["manifest.json"],
                         "store_results_sha256": store_metadata["results.json"]})
        for row in input_rows:
            by_id[row["id"]] = {"store": store, "metadata": store_metadata,
                                "result": store_results[row["id"]]}
    if set(bindings.get("items", {})) != set(by_id):
        raise ValueError("logical item source bindings are not an exact canonical union")
    for row_id, origin in by_id.items():
        store = origin["store"]
        source_result = origin["result"]
        item = bindings["items"][row_id]
        result = results[row_id]
        item_keys = {"source_store", "source_manifest_sha256", "source_results_sha256",
                     "source_sample_file", "source_sample_sha256"}
        if source_result.get("rad_file"):
            item_keys |= {"source_rad_file", "source_rad_sha256"}
        base._exact_keys(item, item_keys, f"logical item source binding {row_id}")
        if (base._norm(item.get("source_store", "")) != base._norm(store) or
                item.get("source_manifest_sha256") != origin["metadata"]["manifest.json"] or
                item.get("source_results_sha256") != origin["metadata"]["results.json"] or
                item.get("source_sample_file") != source_result.get("sample_file") or
                item.get("source_sample_sha256") != source_result.get("sample_sha256") or
                result != source_result or
                pb.file_sha256(store / source_result["sample_file"]) !=
                source_result.get("sample_sha256") or
                pb.file_sha256(logical / result["sample_file"]) != result.get("sample_sha256")):
            raise ValueError(f"logical item source binding differs: {row_id}")
        if source_result.get("rad_file") and (
                item.get("source_rad_file") != source_result.get("rad_file") or
                item.get("source_rad_sha256") != source_result.get("rad_sha256") or
                pb.file_sha256(store / source_result["rad_file"]) !=
                source_result.get("rad_sha256") or
                pb.file_sha256(logical / result["rad_file"]) != result.get("rad_sha256")):
            raise ValueError(f"logical radiation source binding differs: {row_id}")
    return {"logical": logical, "bindings": bindings, "physical": physical,
            "results": results}


def _native_terminal_bindings(dataset: Path, entries: list[dict[str, Any]],
                              roster: tuple[str, ...]) -> list[dict[str, Any]]:
    if any(_job_state(dataset, entry, roster) != "done" for entry in entries):
        raise ValueError("a materialized logical store requires all three native shard terminals")
    state_root = dataset / "jobs_state"
    bindings = []
    for entry in entries:
        store = entry["job"]["store"]
        claim_path = state_root / f"{store}.claim"
        done_path = state_root / f"{store}.done"
        done = pb.read_json(done_path)
        bindings.append({
            "machine": done["machine"],
            "claim_file": claim_path.name, "claim_sha256": pb.file_sha256(claim_path),
            "done_file": done_path.name, "done_sha256": pb.file_sha256(done_path),
        })
    return bindings


def _completion_payload(dataset: Path, plan: dict[str, Any], source: dict[str, Any],
                        roster: tuple[str, ...]):
    logical = source["logical"]
    physical = []
    terminals = _native_terminal_bindings(dataset, plan["entries"], roster)
    for item, terminal in zip(source["physical"], terminals):
        bound = dict(item)
        bound["native_terminal"] = terminal
        physical.append(bound)
    return {"schema_version": 1, "status": "complete",
            "batch": plan["batch"], "logical_store": plan["logical_store"],
            "canonical_input": plan["canonical_name"],
            "canonical_manifest_sha256": pb.file_sha256(plan["canonical"] / "manifest.json"),
            "split_receipt_sha256": plan["split_receipt_sha256"],
            "logical_manifest_sha256": pb.file_sha256(logical / "manifest.json"),
            "logical_results_sha256": pb.file_sha256(logical / "results.json"),
            "source_bindings_sha256": pb.file_sha256(logical / "source_bindings.json"),
            "physical_sources": physical,
            "counts": {"logical": 60, "physical_shards": 3, "per_shard": 20},
            "authoritative_feedback_imported": False}


def _validate_logical_store(dataset: Path, plan: dict[str, Any], roster: tuple[str, ...], *,
                            require_receipt: bool):
    source = _source_binding(dataset, plan)
    receipt_path = source["logical"] / COMPLETE_NAME
    expected = _completion_payload(dataset, plan, source, roster)
    if require_receipt:
        if not receipt_path.is_file() or pb.read_json(receipt_path) != expected:
            raise ValueError("logical merge completion receipt is absent or differs")
    elif receipt_path.exists() and pb.read_json(receipt_path) != expected:
        raise ValueError("logical merge completion receipt differs")
    return source, expected


def _write_new_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def aggregate_batch(path: str | Path, required_sha256: str, batch: int, *,
                    recover_existing: bool = False) -> Path:
    context = _context(path, required_sha256)
    if batch not in context["plans"]:
        raise ValueError("aggregate batch is outside the released logical prefix")
    plan = context["plans"][batch]
    for entry in plan["entries"]:
        if _job_state(context["dataset"], entry, context["roster"]) != "done":
            raise ValueError("all three native shards must be done with zero errors before merge")
    logical = context["dataset"] / plan["logical_store"]
    receipt_path = logical / COMPLETE_NAME
    if logical.exists():
        if not recover_existing:
            raise FileExistsError("logical store already exists; use explicit --recover-existing")
        source, payload = _validate_logical_store(context["dataset"], plan, context["roster"],
                                                  require_receipt=False)
        if not receipt_path.exists():
            _write_new_json(receipt_path, payload)
    else:
        if recover_existing:
            raise FileNotFoundError("no existing logical store is available for recovery")
        stores = [context["dataset"] / entry["job"]["store"] for entry in plan["entries"]]
        profiled_shards.merge_stores(plan["canonical"], stores, logical)
        source, payload = _validate_logical_store(context["dataset"], plan, context["roster"],
                                                  require_receipt=False)
        _write_new_json(receipt_path, payload)
    _validate_logical_store(context["dataset"], plan, context["roster"],
                            require_receipt=True)
    return receipt_path


def _feedback_imported(context: dict[str, Any], batch: int) -> bool:
    work = Path(context["release"]["queue"]["exploration"]["work_dir"])
    state = ex._read_json(work / "state.json")
    info = state.get("batches", {}).get(str(batch), {})
    audit = info.get("feedback_audit")
    if not audit:
        return False
    audit_path = ex._safe_input_path(work, audit)
    if not audit_path.is_file():
        raise ValueError(f"batch {batch} feedback audit is missing")
    if "feedback_audit_sha256" in info and pb.file_sha256(audit_path) != info["feedback_audit_sha256"]:
        raise ValueError(f"batch {batch} feedback audit hash differs")
    cfg = ex.validate_exploration_config(ex._load_yaml(work / "config.yaml"))
    if cfg["exploration"].get("training_protocol") == "filter_masked_prior_v1":
        from script.filter_training import validate_feedback_binding
        validate_feedback_binding(work, batch, state)
    plan = context["plans"][batch]
    source, _payload = _validate_logical_store(context["dataset"], plan, context["roster"],
                                                require_receipt=True)
    logical = source["logical"]
    expected_manifest = logical / "results.json"
    expected_results_sha = pb.file_sha256(expected_manifest)
    expected_measurement_sha = pb.file_sha256(logical / "measurement.json")
    expected_score_sha = pb.file_sha256(logical / "score_spec.json")
    imported = [row for row in ex._manifest_rows(work / "dataset_manifest.json")
                if row.get("batch") == batch]
    if (len(imported) != 60 or {row["id"] for row in imported} !=
            set(source["results"])):
        raise ValueError(f"batch {batch} authoritative feedback is not the logical 60")
    for row in imported:
        result = source["results"][row["id"]]
        if (base._norm(row.get("source_manifest", "")) != base._norm(expected_manifest) or
                row.get("source_sample_sha256") != result.get("sample_sha256")):
            raise ValueError(f"batch {batch} feedback source differs from logical merge")
        if cfg["exploration"].get("training_protocol") == "filter_masked_prior_v1" and (
                row.get("source_manifest_sha256") != expected_results_sha or
                row.get("source_measurement_sha256") != expected_measurement_sha or
                row.get("source_score_spec_sha256") != expected_score_sha):
            raise ValueError(f"batch {batch} feedback metadata differs from logical merge")
    return True


def queue_status(path: str | Path, required_sha256: str) -> str:
    context = _context(path, required_sha256)
    for plan in context["plans"].values():
        if (context["dataset"] / plan["logical_store"]).exists():
            _validate_logical_store(context["dataset"], plan, context["roster"],
                                    require_receipt=True)
    states = [_job_state(context["dataset"], entry, context["roster"])
              for entry in context["entries"]]
    if "failed" in states:
        return "terminal_failure"
    if "active" in states:
        return "active"
    if "skipped" in states:
        return "terminal_skips"
    missing_logical = False
    missing_feedback = False
    for batch, plan in context["plans"].items():
        logical = context["dataset"] / plan["logical_store"]
        if not logical.exists():
            missing_logical = True
            continue
        _validate_logical_store(context["dataset"], plan, context["roster"],
                                require_receipt=True)
        if not _feedback_imported(context, batch):
            missing_feedback = True
    if missing_logical:
        return "physical_complete_feedback_pending"
    if missing_feedback:
        return "logical_ready_feedback_pending"
    return "all_success"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--queue-status", action="store_true")
    parser.add_argument("--aggregate-batch", type=int)
    parser.add_argument("--recover-existing", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.recover_existing and args.aggregate_batch is None:
        raise ValueError("--recover-existing requires --aggregate-batch")
    if args.queue_status and args.aggregate_batch is not None:
        raise ValueError("--queue-status and --aggregate-batch are mutually exclusive")
    if args.aggregate_batch is not None:
        receipt = aggregate_batch(args.release, args.release_sha256,
                                  args.aggregate_batch,
                                  recover_existing=args.recover_existing)
        print(receipt)
        return 0
    if args.queue_status:
        status = queue_status(args.release, args.release_sha256)
        print(status)
        return {"all_success": 0,
                "terminal_skips": 0,
                "physical_complete_feedback_pending": 0,
                "logical_ready_feedback_pending": 0,
                "terminal_failure": 4,
                "active": 3}[status]
    report = validate_release(args.release, args.release_sha256)
    if args.report:
        base._write_report(args.report, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
