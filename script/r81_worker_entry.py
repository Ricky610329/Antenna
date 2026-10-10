"""Fail-closed entry gate for a developer-prepared R81 worker release.

This module never prepares data, changes the queue, or starts HFSS.  A release
binds the exact Git/runtime bytes,
the separate R81 dataset, every queued job and input file, and the final R80
completion evidence.  There is no executable example release in the tree.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from antenna.measurement import measurement_id, score_spec_id
from script import prepare_symmetry_filter as wide
from script import profiled_batch as pb
from script import exploration as ex


REPO = Path(__file__).resolve().parents[1]
SCOPE = wide.SCOPE
STAGES = ("EngineeringOnly", "Formal")
RUNTIME_FILES = tuple(sorted(
    {path.relative_to(REPO).as_posix() for path in (REPO / "antenna").rglob("*.py")} |
    {"script/batch_scope.py", "script/dedust.py", "script/exploration.py",
     "script/filter_training.py", "script/kill.py", "script/prepare_symmetry_filter.py",
     "script/profiled_batch.py", "script/r81_worker_entry.py", "script/start_r81_worker.ps1",
     "configs/dual_r81_wide_filter.yaml",
     "configs/single_r80_symmetry_factory.yaml"}
))
ENGINEERING = {
    "engineering_smoke": ("dedust_r81smoke", 6),
    "engineering_discrete": ("dedust_r81discrete", 2),
    "engineering_mesh": ("dedust_r81mesh", 2),
}
HEX64 = re.compile(r"[0-9a-f]{64}")
HEX40 = re.compile(r"[0-9a-f]{40}")
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}")
EXPECTED_WORKERS = ("140.123.106.216", "140.123.106.218", "140.123.106.37")


def _exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(f"{label} keys differ: {actual}")
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _name(value: Any, label: str) -> str:
    if not isinstance(value, str) or SAFE_NAME.fullmatch(value) is None:
        raise ValueError(f"{label} must be one safe path component")
    return value


def _retry_workers(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or tuple(value) != EXPECTED_WORKERS:
        raise ValueError("queue.expected_retry_workers must be the fixed authorized worker roster")
    return EXPECTED_WORKERS


def _norm(path: str | Path) -> str:
    return os.path.normcase(os.fspath(Path(path).resolve(strict=False)))


def _nested(left: str | Path, right: str | Path) -> bool:
    """Return whether either normalized path contains the other."""
    a, b = _norm(left), _norm(right)
    try:
        return os.path.commonpath((a, b)) in {a, b}
    except ValueError:
        return False


def _bound_path(binding: Any, label: str, *, producer_path: bool = False) -> Path:
    keys = {"path", "sha256"} | ({"producer_path"} if producer_path else set())
    item = _exact_keys(binding, keys, label)
    path = Path(item["path"])
    if not path.is_absolute():
        raise ValueError(f"{label}.path must be absolute")
    if producer_path:
        original = item["producer_path"]
        if not isinstance(original, str) or not Path(original).is_absolute():
            raise ValueError(f"{label}.producer_path must preserve the absolute producer path")
    expected = _sha(item["sha256"], f"{label}.sha256")
    if pb.file_sha256(path) != expected:
        raise ValueError(f"{label} hash differs")
    return path


def _bound_json(binding: Any, label: str, *, producer_path: bool = False) -> tuple[Path, dict[str, Any]]:
    path = _bound_path(binding, label, producer_path=producer_path)
    value = pb.read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return path, value


def _bound_census_status(binding: Any, commit: str) -> tuple[Path, dict[str, Any]]:
    item = _exact_keys(binding, {"repo_path", "sha256"}, "r80_completion.census")
    relative = Path(item["repo_path"]).as_posix()
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts or candidate.as_posix() != relative:
        raise ValueError("R80 census repo_path must be a safe repository-relative path")
    if not relative.startswith("docs/log/assets/r80_") or not relative.endswith(".json"):
        raise ValueError("R80 census must be a tracked docs/log/assets status record")
    path = REPO / relative
    expected = _sha(item["sha256"], "r80_completion.census.sha256")
    if _git_blob_sha(commit, relative) != expected or not _git_path_clean(commit, relative):
        raise ValueError("R80 tracked census record differs from the release commit")
    value = pb.read_json(path)
    if not isinstance(value, dict):
        raise ValueError("r80_completion.census must contain a JSON object")
    return path, value


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    return result.stdout.strip()


def _git_blob_sha(commit: str, relative: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative}"], cwd=REPO, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    import hashlib
    return hashlib.sha256(result.stdout).hexdigest()


def _git_is_ancestor(commit: str) -> bool:
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=REPO,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0


def _git_path_clean(commit: str, relative: str) -> bool:
    return subprocess.run(
        ["git", "diff", "--quiet", commit, "--", relative], cwd=REPO,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0


def _validate_runtime(runtime: Any) -> None:
    value = _exact_keys(runtime, {"git_commit", "files"}, "runtime")
    commit = value["git_commit"]
    if not isinstance(commit, str) or HEX40.fullmatch(commit) is None:
        raise ValueError("runtime.git_commit must be a full lowercase commit id")
    if not _git_is_ancestor(commit):
        raise ValueError("runtime Git commit is not an ancestor of this checkout")
    files = value["files"]
    if not isinstance(files, dict) or set(files) != set(RUNTIME_FILES):
        raise ValueError("runtime.files must bind the exact required producer set")
    for relative in RUNTIME_FILES:
        expected = _sha(files[relative], f"runtime.files[{relative}]")
        if (_git_blob_sha(commit, relative) != expected or
                not _git_path_clean(commit, relative)):
            raise ValueError(f"runtime source changed: {relative}")


def _validate_r80(binding: Any, dataset: Path, commit: str) -> dict[str, Any]:
    value = _exact_keys(
        binding,
        {"watch_status", "action_receipt", "census", "profile", "census_raw_report",
         "census_rows", "census_pair_proofs", "census_query_producer",
         "census_validation_producer",
         "measurement_id", "score_spec_id"},
        "r80_completion",
    )
    _, watch = _bound_json(value["watch_status"], "r80_completion.watch_status")
    _action_path, action = _bound_json(value["action_receipt"], "r80_completion.action_receipt",
                                       producer_path=True)
    _, census = _bound_census_status(value["census"], commit)
    raw_path, raw = _bound_json(value["census_raw_report"], "r80_completion.census_raw_report")
    rows_path = _bound_path(value["census_rows"], "r80_completion.census_rows")
    proofs_path = _bound_path(value["census_pair_proofs"], "r80_completion.census_pair_proofs")
    query_path = _bound_path(value["census_query_producer"], "r80_completion.census_query_producer")
    validation_path = _bound_path(value["census_validation_producer"],
                                  "r80_completion.census_validation_producer")
    profile_path = _bound_path(value["profile"], "r80_completion.profile", producer_path=True)
    canonical_profile = REPO / "configs/single_r80_symmetry_factory.yaml"
    profile = pb.load_profile_config(profile_path)
    canonical = pb.load_profile_config(canonical_profile)
    if profile != canonical:
        raise ValueError("archived R80 profile differs from the canonical Git factory profile")
    if profile is None or profile.port != "single" or profile.scope != SCOPE:
        raise ValueError("R80 profile is not the authoritative single-port scope")
    mid = measurement_id(profile.measurement)
    sid = score_spec_id(profile.score_spec)
    if value["measurement_id"] != mid or value["score_spec_id"] != sid:
        raise ValueError("R80 release profile ids differ from the bound profile")
    action_sha = value["action_receipt"]["sha256"]
    if (watch.get("status") != "symmetry_collection_complete" or
            watch.get("valid_unique") != 5000 or
            watch.get("last_receipt_sha256") != action_sha or
            not isinstance(watch.get("last_receipt"), str) or
            _norm(watch["last_receipt"]) != _norm(value["action_receipt"]["producer_path"])):
        raise ValueError("R80 watcher has not recorded exact 5000 completion")
    watch_profile = watch.get("profile_binding")
    if (not isinstance(watch_profile, dict) or
            watch_profile.get("path") is None or
            _norm(watch_profile["path"]) != _norm(value["profile"]["producer_path"]) or
            watch_profile.get("sha256") != value["profile"]["sha256"] or
            watch_profile.get("scope") != SCOPE):
        raise ValueError("R80 watcher profile binding differs from the bound source profile")
    audit = action.get("audit")
    if (action.get("status") != "idle" or
            action.get("scope") != SCOPE or action.get("profile_sha256") != value["profile"]["sha256"] or
            not isinstance(action.get("profile_config"), str) or
            _norm(action["profile_config"]) != _norm(value["profile"]["producer_path"]) or
            action.get("valid_unique") != 5000 or action.get("target_valid_unique") != 5000 or
            action.get("pending_unique") != 0 or action.get("planned_jobs") != [] or
            not isinstance(audit, dict) or audit.get("valid_unique_patterns") != 5000):
        raise ValueError("R80 action receipt is not a terminal 5000/0 census")
    r80_root = action.get("dataset_root")
    if not isinstance(r80_root, str) or _nested(r80_root, dataset):
        raise ValueError("R81 must use a dataset root separate from R80")
    if (census.get("status") != "read_only_query_complete" or
            census.get("validation_status") != "passed_saved_raw_summary_validation" or
            census.get("target_unique") != 5000 or
            census.get("queue_or_source_mutation") is not False or
            census.get("controller_locks_acquired") is not False or
            census.get("unique_successful_patterns") != 5000 or
            census.get("successful_including_repeats", 0) < 5000 or
            census.get("raw_replayed_this_query") != census.get("successful_including_repeats") or
            census.get("exact_prior_metrics_reused") != 0 or
            census.get("reported_best_independently_raw_replayed") is not True or
            census.get("measurement_id") != mid or census.get("score_spec_id") != sid):
        raise ValueError("R80 validated final census is not the required immutable 5000 receipt")
    if (census.get("rows_sha256") != pb.file_sha256(rows_path) or
            census.get("pair_proofs_sha256") != pb.file_sha256(proofs_path) or
            census.get("query_script_sha256") != pb.file_sha256(query_path) or
            census.get("query_report_sha256") != pb.file_sha256(raw_path) or
            census.get("validation_script_sha256") != pb.file_sha256(validation_path) or
            not isinstance(census.get("jobs_sha256_at_scan_start"), str) or
            HEX64.fullmatch(census["jobs_sha256_at_scan_start"]) is None):
        raise ValueError("R80 census producer or saved row/proof evidence differs")
    copied = ("status", "measurement_id", "score_spec_id", "unique_successful_patterns",
              "successful_including_repeats", "raw_replayed_this_query",
              "exact_prior_metrics_reused", "reported_best_independently_raw_replayed",
              "jobs_sha256_at_scan_start", "rows_sha256", "pair_proofs_sha256",
              "query_script_sha256", "controller_locks_acquired", "queue_or_source_mutation")
    if any(census.get(key) != raw.get(key) for key in copied):
        raise ValueError("R80 validated census differs from its bound raw report")
    return {"measurement_id": mid, "score_spec_id": sid}


def _validate_file_map(folder: Path, value: Any, label: str) -> None:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{label} must bind every direct input file")
    actual = {p.name: pb.file_sha256(p) for p in folder.iterdir() if p.is_file()}
    if set(actual) != set(value):
        raise ValueError(f"{label} file inventory differs")
    for name, digest in value.items():
        _name(name, f"{label} filename")
        if actual[name] != _sha(digest, f"{label}[{name}]"):
            raise ValueError(f"{label} file hash differs: {name}")


def _validate_tree(folder: Path, value: Any, label: str) -> None:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{label} must bind the complete controller tree")
    actual = {path.relative_to(folder).as_posix(): pb.file_sha256(path)
              for path in folder.rglob("*") if path.is_file()}
    if set(actual) != set(value):
        raise ValueError(f"{label} file inventory differs")
    for relative, digest in value.items():
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts or candidate.as_posix() != relative:
            raise ValueError(f"{label} contains an unsafe relative path")
        if actual[relative] != _sha(digest, f"{label}[{relative}]"):
            raise ValueError(f"{label} file hash differs: {relative}")


def _validate_exploration(binding: Any, dataset: Path, prefix: int,
                          formal_entries: dict[int, dict[str, Any]],
                          expected_profile: dict[str, Any]) -> None:
    value = _exact_keys(binding, {"work_dir", "files"}, "queue.exploration")
    work = Path(value["work_dir"])
    if not work.is_absolute():
        raise ValueError("queue.exploration.work_dir must be absolute")
    _validate_tree(work, value["files"], "queue.exploration.files")
    cfg = ex.validate_exploration_config(ex._load_yaml(work / "config.yaml"))
    state = ex._read_json(work / "state.json")
    exp = cfg["exploration"]
    profile = {"measurement_id": measurement_id(cfg["measurement"]),
               "score_spec_id": score_spec_id(cfg["score_spec"]),
               "runtime": cfg.get("runtime", {})}
    if (profile != expected_profile or exp.get("mode") != "dual_margin5" or
            exp.get("max_batches") != 3 or exp.get("batch_size") != 60 or
            ex.content_id(cfg) != state.get("config_hash") or
            _norm(state.get("dataset_root", "")) != _norm(dataset)):
        raise ValueError("exploration controller differs from the R81 release profile")
    expected_batches = {str(batch) for batch in range(1, prefix + 1)}
    if set(state.get("batches", {})) != expected_batches or state.get("next_batch") != prefix + 1:
        raise ValueError("exploration controller batch prefix is not exact and contiguous")

    ensemble_hashes: dict[int, str] = {}
    masked = exp.get("training_protocol") == "filter_masked_prior_v1"
    for previous in range(1, prefix):
        info = state["batches"][str(previous)]
        if not info.get("feedback_audit") or not info.get("trained_model_dir"):
            raise ValueError(f"batch {previous} needs authoritative feedback and trained model before the next release")
    training = ex.load_training_set(work) if prefix > 1 else None
    for previous in range(1, prefix):
        info = state["batches"][str(previous)]
        if masked:
            from script import filter_training
            filter_training.validate_feedback_binding(work, previous, state)
            _plan, summary = filter_training.validate_model_evidence(work, previous, cfg, state)
            paths = [work / relative for relative in summary["model_files"]]
        else:
            audit = ex._read_json(work / info["feedback_audit"])
            imported = [row for row in training.rows if int(row["batch"]) == previous]
            if (audit.get("batch") != previous or audit.get("n_imported") != info.get("selected") or
                    audit.get("truth_used_for_selection") is not False or
                    len(imported) != info.get("selected")):
                raise ValueError(f"batch {previous} feedback audit/training set binding differs")
            model_dir = work / info["trained_model_dir"]
            summary = ex._read_json(model_dir / "training_summary.json")
            expected_paths = [model_dir / f"member-{seed}.pt" for seed in exp["ensemble_seeds"]]
            relative_paths = [path.relative_to(work).as_posix() for path in expected_paths]
            if summary.get("through_batch") != previous or summary.get("model_files") != relative_paths:
                raise ValueError(f"batch {previous} training summary differs")
            through = [row for row in training.rows if int(row["batch"]) <= previous]
            for seed, path in zip(exp["ensemble_seeds"], expected_paths):
                saved = torch.load(path, weights_only=False, map_location="cpu")
                signature = ex._training_signature(through, cfg, previous, seed)
                ex._validate_checkpoint(saved, state, "dual_margin5", 5, signature=signature)
                if (saved.get("complete") is not True or saved.get("through_batch") != previous or
                        saved.get("member_seed") != seed):
                    raise ValueError(f"batch {previous} model checkpoint is incomplete or mismatched")
            paths = expected_paths
        ensemble_hashes[previous] = ex._ensemble_hash(paths)

    for batch in range(1, prefix + 1):
        info = state["batches"][str(batch)]
        controller_input = ex._safe_input_path(work, info["input_dir"])
        release_input = dataset / formal_entries[batch]["job"]["input"]
        if (_files_for_compare(controller_input) != _files_for_compare(release_input) or
                info.get("selected") != 60):
            raise ValueError(f"batch {batch} release input differs from controller selection")
        rows = ex._manifest_rows(controller_input / "manifest.json")
        arms = exp["arms"] if batch == 1 else exp["later_arms"]
        if (Counter(row.get("selection_arm") for row in rows) != Counter(arms) or
                any(row.get("batch") != batch for row in rows)):
            raise ValueError(f"batch {batch} does not use the authoritative selection arms")
        model_hash = None if batch == 1 else ensemble_hashes[batch - 1]
        if any(row.get("model_hash") != model_hash for row in rows):
            raise ValueError(f"batch {batch} model lineage differs from the trained predecessor")


def _files_for_compare(folder: Path) -> dict[str, str]:
    return {path.name: pb.file_sha256(path) for path in folder.iterdir() if path.is_file()}


def _validate_input(dataset: Path, entry: dict[str, Any], label: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    job = entry["job"]
    if not isinstance(job, dict) or set(job) - {"input", "store", "prio", "config", "scope"}:
        raise ValueError(f"{label}.job has unsupported worker fields")
    if not {"input", "store", "prio", "config", "scope"}.issubset(job):
        raise ValueError(f"{label}.job lacks required worker fields")
    input_name = _name(job["input"], f"{label}.job.input")
    _name(job["store"], f"{label}.job.store")
    if job["scope"] != SCOPE or job["config"] != f"{input_name}/config.yaml":
        raise ValueError(f"{label}.job must use the scoped input config snapshot")
    if isinstance(job["prio"], bool) or not isinstance(job["prio"], int):
        raise ValueError(f"{label}.job.prio must be an integer")
    folder = dataset / input_name
    _validate_file_map(folder, entry["input_files"], f"{label}.input_files")
    cfg = pb.load_profile_config(folder / "config.yaml")
    if cfg is None or cfg.scope != SCOPE or cfg.port != "dual":
        raise ValueError(f"{label} does not use the R81 dual profile scope")
    rows = pb.validate_input(folder, cfg)
    return {"measurement_id": measurement_id(cfg.measurement),
            "score_spec_id": score_spec_id(cfg.score_spec), "runtime": cfg.runtime}, rows


def _validate_candidate(dataset: Path, candidate: Any, repeat_rows: list[dict[str, Any]],
                        expected_profile: dict[str, Any], label: str) -> None:
    value = _exact_keys(candidate, {"store", "id", "files"}, f"{label}.candidate")
    store_name = _name(value["store"], f"{label}.candidate.store")
    candidate_id = _name(value["id"], f"{label}.candidate.id")
    store = dataset / store_name
    _validate_file_map(store, value["files"], f"{label}.candidate.files")
    results = pb.validate_store(store, require_complete=True)
    candidate_profile = {
        "measurement_id": measurement_id(pb.read_json(store / "measurement.json")),
        "score_spec_id": score_spec_id(pb.read_json(store / "score_spec.json")),
        "runtime": expected_profile["runtime"],
    }
    if candidate_profile != expected_profile:
        raise ValueError(f"{label} candidate profile differs from the R81 wide profile")
    result = results.get(candidate_id)
    manifest = {row["id"]: row for row in pb.read_json(store / "manifest.json")}
    if (not isinstance(result, dict) or result.get("all_passed") is not True or
            not isinstance(result.get("worst_margin"), (int, float)) or
            result["worst_margin"] <= 0 or candidate_id not in manifest):
        raise ValueError(f"{label} candidate is not a completed positive-margin original")
    if manifest[candidate_id].get("repeat") is True:
        raise ValueError(f"{label} candidate must be an original, not another repeat")
    if len(repeat_rows) != 1:
        raise ValueError(f"{label} must contain exactly one repeat")
    row = repeat_rows[0]
    if (row.get("kind") != "repeat" or row.get("repeat") is not True or
            row.get("repeat_reason") != "repeatability" or
            row.get("parent_batch_id") != candidate_id or
            row.get("pattern_sha256") != manifest[candidate_id].get("pattern_sha256")):
        raise ValueError(f"{label} does not independently repeat the bound positive candidate")


def _validate_queue(queue: Any, dataset: Path, stage: str) -> dict[str, Any]:
    value = _exact_keys(queue, {"jobs_sha256", "jobs", "formal_input", "exploration",
                                "expected_retry_workers"}, "queue")
    roster = _retry_workers(value["expected_retry_workers"])
    jobs_path = dataset / "jobs.json"
    if pb.file_sha256(jobs_path) != _sha(value["jobs_sha256"], "queue.jobs_sha256"):
        raise ValueError("jobs.json differs from the release")
    entries = value["jobs"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("queue.jobs must be a nonempty list")
    actual_jobs = pb.read_json(jobs_path)
    if actual_jobs != [entry.get("job") for entry in entries if isinstance(entry, dict)]:
        raise ValueError("queue contains unknown, reordered, or changed jobs")
    formal = _exact_keys(value["formal_input"], {"name", "files"}, "queue.formal_input")
    if formal["name"] != "dedust_r81b1_input":
        raise ValueError("formal input must be the declared R81 b1 bundle")
    _validate_file_map(dataset / formal["name"], formal["files"], "queue.formal_input.files")
    wide.validate_wide_inputs(dataset)
    roles: list[str] = []
    profiles: dict[str, dict[str, Any]] = {}
    formal_batches: dict[int, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
    formal_entries: dict[int, dict[str, Any]] = {}
    repeat_profiles: list[dict[str, Any]] = []
    repeats = 0
    for index, entry in enumerate(entries):
        label = f"queue.jobs[{index}]"
        if not isinstance(entry, dict) or not isinstance(entry.get("role"), str):
            raise ValueError(f"{label} is malformed")
        role = entry["role"]
        roles.append(role)
        common = {"role", "job", "input_files"}
        extra = {"batch"} if role == "formal_batch" else ({"candidate"} if role == "positive_repeat" else set())
        _exact_keys(entry, common | extra, label)
        profile, rows = _validate_input(dataset, entry, label)
        profiles[role] = profile
        if role in ENGINEERING:
            expected_store, count = ENGINEERING[role]
            if entry["job"]["store"] != expected_store or entry["job"]["input"] != expected_store + "_input" or len(rows) != count:
                raise ValueError(f"{label} differs from the exact engineering job")
        elif role == "formal_batch":
            batch = entry["batch"]
            if isinstance(batch, bool) or not isinstance(batch, int) or batch not in {1, 2, 3} or batch in formal_batches:
                raise ValueError("formal batches must be unique integers 1, 2, 3")
            formal_batches[batch] = (rows, profile)
            formal_entries[batch] = entry
        elif role == "positive_repeat":
            repeats += 1
            repeat_profiles.append(profile)
            _validate_candidate(dataset, entry["candidate"], rows, profile, label)
        else:
            raise ValueError(f"unknown R81 job role: {role}")

    if Counter(roles).most_common() and any(roles.count(role) != 1 for role in ENGINEERING):
        raise ValueError("queue must contain each engineering job exactly once")
    queued_inputs = {entry["job"]["input"] for entry in entries}
    if stage == "EngineeringOnly":
        if (Counter(roles) != Counter(ENGINEERING.keys()) or formal["name"] in queued_inputs or
                value["exploration"] is not None):
            raise ValueError("EngineeringOnly permits only 6 Fast + 2 Discrete + 2 mesh; formal stays unqueued")
        return {"job_count": 3, "engineering_check": None, "retry_workers": roster}

    check = wide.measurement_check(dataset)
    if (check.get("engineering_check_passed") is not True or
            check.get("full_batch_release_allowed") is not True or
            len(check.get("comparisons", [])) != 4 or
            any(max(row.get("max_curve_difference_db", float("inf")),
                    row.get("max_margin_difference_db", float("inf"))) > .3
                for row in check["comparisons"])):
        raise ValueError("fresh R81 engineering measurement check did not pass all four comparisons")
    prefix = len(formal_batches)
    if prefix not in {1, 2, 3} or set(formal_batches) != set(range(1, prefix + 1)):
        raise ValueError("Formal batches must be a nonempty contiguous prepared prefix from batch 1")
    if next(entry for entry in entries if entry.get("role") == "formal_batch" and entry.get("batch") == 1)["job"]["input"] != formal["name"]:
        raise ValueError("formal batch 1 must be the release-bound b1 input")
    expected = profiles["engineering_smoke"]
    for batch, (rows, profile) in formal_batches.items():
        expected_arms = ({"history": 20, "specialist": 20, "random": 20} if batch == 1 else
                         {"best_min_margin": 20, "uncertainty": 20, "random": 20})
        if len(rows) != 60 or Counter(row.get("selection_arm") for row in rows) != expected_arms:
            raise ValueError(f"formal batch {batch} is not the declared size-60 arm allocation")
        if profile != expected:
            raise ValueError(f"formal batch {batch} profile differs from the engineering profile")
    if any(profile != expected for profile in repeat_profiles):
        raise ValueError("positive-candidate repeats must use the same R81 wide profile")
    _validate_exploration(value["exploration"], dataset, prefix, formal_entries, expected)
    return {"job_count": len(entries), "engineering_check": check, "retry_workers": roster}


def validate_release(path: str | Path, required_sha256: str, stage: str) -> dict[str, Any]:
    release_path = Path(path)
    expected_sha = _sha(required_sha256, "release sha256")
    if pb.file_sha256(release_path) != expected_sha:
        raise ValueError("release metadata hash differs")
    release = pb.read_json(release_path)
    value = _exact_keys(
        release,
        {"schema_version", "release_id", "stage", "dataset_root", "runtime", "r80_completion", "queue"},
        "release",
    )
    if value["schema_version"] != 1:
        raise ValueError("unsupported release schema")
    _name(value["release_id"], "release.release_id")
    if stage not in STAGES or value["stage"] != stage:
        raise ValueError("requested stage differs from the immutable release stage")
    dataset = Path(value["dataset_root"])
    if not dataset.is_absolute() or "r81" not in str(dataset).lower():
        raise ValueError("dataset_root must be an explicit separate R81 path")
    _validate_runtime(value["runtime"])
    r80 = _validate_r80(value["r80_completion"], dataset, value["runtime"]["git_commit"])
    queue = _validate_queue(value["queue"], dataset, stage)
    return {"schema_version": 1, "status": "r81_worker_entry_passed", "release_sha256": expected_sha,
            "release_id": value["release_id"], "stage": stage, "dataset_root": str(dataset),
            "scope": SCOPE, "job_count": queue["job_count"], "r80": r80,
            "engineering_check": queue.get("engineering_check"), "hfss_started": False,
            "queue_or_source_mutation": False}


def _marker_json(path: Path, allowed: set[str], required: set[str], label: str) -> dict[str, Any]:
    try:
        value = pb.read_json(path)
    except Exception as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict) or not required <= set(value) or set(value) - allowed:
        raise ValueError(f"{label} marker schema differs")
    if not isinstance(value.get("at"), str) or not value["at"]:
        raise ValueError(f"{label}.at is invalid")
    return value


def _marker_machines(values: Any, roster: tuple[str, ...], label: str) -> list[str]:
    if (not isinstance(values, list) or not values or
            any(not isinstance(item, str) or not item for item in values) or
            len(set(values)) != len(values) or not set(values) <= set(roster)):
        raise ValueError(f"{label} does not contain a unique authorized worker subset")
    return values


def queue_status(path: str | Path, required_sha256: str, stage: str) -> str:
    """Return a read-only semantic status for the exact released queue."""
    validate_release(path, required_sha256, stage)
    release = pb.read_json(path)
    dataset = Path(release["dataset_root"])
    entries = release["queue"]["jobs"]
    roster = _retry_workers(release["queue"]["expected_retry_workers"])
    state_root = dataset / "jobs_state"
    stores = {entry["job"]["store"] for entry in entries}
    if state_root.exists():
        for marker in state_root.iterdir():
            if marker.is_file() and marker.suffix in {".claim", ".done", ".fail"}:
                if marker.stem not in stores:
                    raise ValueError(f"jobs_state contains an unknown released store marker: {marker.name}")

    states: list[str] = []
    by_store = {entry["job"]["store"]: entry for entry in entries}
    for store, entry in by_store.items():
        claim_path = state_root / f"{store}.claim"
        done_path = state_root / f"{store}.done"
        fail_path = state_root / f"{store}.fail"
        claim = done = fail = None
        if claim_path.exists():
            claim = _marker_json(claim_path, {"machine", "at", "prior_fail"},
                                 {"machine", "at"}, f"{store}.claim")
            if claim["machine"] not in roster:
                raise ValueError(f"{store}.claim names an unauthorized worker")
            if "prior_fail" in claim:
                prior = _marker_machines(claim["prior_fail"], roster, f"{store}.claim.prior_fail")
                if claim["machine"] in prior:
                    raise ValueError(f"{store}.claim repeats its owner in prior_fail")
        if done_path.exists():
            done = _marker_json(done_path, {"machine", "at", "errors", "error_ids"},
                                {"machine", "at", "errors", "error_ids"}, f"{store}.done")
            if (done["machine"] not in roster or isinstance(done["errors"], bool) or
                    not isinstance(done["errors"], int) or done["errors"] < 0 or
                    not isinstance(done["error_ids"], list) or
                    any(not isinstance(item, str) for item in done["error_ids"]) or
                    done["errors"] < len(done["error_ids"])):
                raise ValueError(f"{store}.done marker fields are invalid")
        if fail_path.exists():
            fail = _marker_json(
                fail_path, {"machines", "last", "at", "failure_kind", "worker_continues"},
                {"machines", "last", "at"}, f"{store}.fail")
            _marker_machines(fail["machines"], roster, f"{store}.fail.machines")
            if not isinstance(fail["last"], str):
                raise ValueError(f"{store}.fail.last is invalid")
        if done is not None and fail is not None:
            raise ValueError(f"{store} has contradictory done and fail markers")
        if done is not None and (claim is None or done["machine"] != claim["machine"]):
            raise ValueError(f"{store}.done lacks its matching native claim")
        if fail is not None:
            expected_chain = ([] if claim is None else claim.get("prior_fail", [])) + (
                [] if claim is None else [claim["machine"]])
            if claim is None or fail["machines"] != expected_chain:
                raise ValueError(f"{store}.fail differs from its native claim retry chain")
        if done is not None:
            input_dir = dataset / entry["job"]["input"]
            store_dir = dataset / store
            for name in ("measurement.json", "score_spec.json", "manifest.json"):
                if pb.read_json(input_dir / name) != pb.read_json(store_dir / name):
                    raise ValueError(f"{store} completed store differs from released {name}")
            results = pb.validate_store(store_dir, require_complete=True)
            if done["errors"] != 0 or done["error_ids"] or len(results) != len(pb.read_json(input_dir / "manifest.json")):
                raise ValueError(f"{store}.done does not describe the complete validated store")
            states.append("done")
        elif fail is not None and set(fail["machines"]) == set(roster):
            states.append("failed")
        else:
            states.append("active")
    if states and all(value == "done" for value in states):
        return "all_success"
    if states and all(value in {"done", "failed"} for value in states):
        return "terminal_failure"
    return "active"


def _write_report(path: Path, report: dict[str, Any]) -> None:
    target = path.resolve()
    target.relative_to((REPO / "tmp").resolve())
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--queue-status", action="store_true")
    args = parser.parse_args()
    if args.queue_status:
        status = queue_status(args.release, args.release_sha256, args.stage)
        raise SystemExit({"all_success": 0, "active": 3, "terminal_failure": 4}[status])
    report = validate_release(args.release, args.release_sha256, args.stage)
    if args.report:
        _write_report(args.report, report)


if __name__ == "__main__":
    main()
