"""Authenticated worker entry for the bounded live R81 S21 survey loop.

The initial batch remains the immutable ``ProvisionalS21Discrete`` release.
Successor batches are additive immutable releases advertised through a
create-only SHA-256 registry.  This module only authenticates the handoff and
chooses the next shard for one pinned host; it never writes the registry,
launches HFSS, trains a model, or changes an existing marker.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import stat
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from script import filter_s21_training as s21
from script import profiled_batch as pb
from script import r81_formal_sharded_entry as formal
from script import r81_provisional_filter_dispatch as dispatch
from script import r81_s21_survey_cycle as cycle
from script import r81_worker_entry as base


SCHEMA_VERSION = 1
STAGE = "S21SurveyLoop"
SUCCESSOR_STAGE = "S21SurveySuccessor"
SUCCESSOR_ROLE = "successor_s21_discrete"
REGISTRY_NAME = re.compile(r"batch-([0-9]{3})[.]json")
MAX_BATCHES = 3
BATCH_SIZE = 60
ROWS_PER_SHARD = 20
RUNTIME_FILES = tuple(sorted(set(dispatch.RUNTIME_FILES) | {
    "script/filter_s21_training.py",
    "script/r81_s21_survey_cycle.py",
    "script/r81_s21_live_controller.py",
    "script/r81_s21_live_entry.py",
    "script/start_r81_s21_live_worker.ps1",
}))
BUDGET = {
    "batch_size": BATCH_SIZE,
    "max_batches": MAX_BATCHES,
    "maximum_rows": BATCH_SIZE * MAX_BATCHES,
    "original_batch": 1,
    "success_trigger": cycle.TRIGGER_NEW_SUCCESSES,
    "arms": copy.deepcopy(cycle.ARM_COUNTS),
}
POLICY = {
    "provisional_exploration_only": True,
    "formal_release_allowed": False,
    "engineering_gate_satisfied": False,
    "engineering_gate_bypassed": False,
    "symmetry_required": False,
    "sweep_type": "Discrete",
    "score_labels": ["S21"],
    "rows_per_batch": BATCH_SIZE,
    "rows_per_shard": ROWS_PER_SHARD,
    "machine_pinned": True,
    "survey_complete_is_spec_pass": False,
    "performance_target_assessed": False,
}


def _sha(path: Path) -> str:
    return pb.file_sha256(Path(path))


def _absolute(path: Any, label: str, *, exists: bool = True) -> Path:
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    source = Path(path)
    value = source.resolve(strict=exists)
    if os.path.normcase(os.fspath(source)) != os.path.normcase(os.fspath(value)):
        raise ValueError(f"{label} must already be normalized")
    return value


def _inside(path: Path, root: Path, label: str) -> Path:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes the bootstrap shared root") from exc
    return path


def _plain_ancestors(path: Path, root: Path, label: str) -> None:
    """Reject redirected path components without walking mutable result trees."""
    _inside(path, root, label)
    current = path
    while True:
        info = current.lstat()
        if (stat.S_ISLNK(info.st_mode) or
                getattr(info, "st_file_attributes", 0) &
                getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)):
            raise ValueError(f"{label} contains a redirected path: {current}")
        if current == root:
            break
        current = current.parent


def _plain_external(path: Path, label: str) -> None:
    """Reject redirection in an external immutable batch-1 path."""
    current = path
    while True:
        info = current.lstat()
        if (stat.S_ISLNK(info.st_mode) or
                getattr(info, "st_file_attributes", 0) &
                getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)):
            raise ValueError(f"{label} contains a redirected path: {current}")
        parent = current.parent
        if parent == current:
            break
        current = parent


def _binding(value: Any, label: str, shared_root: Path) -> tuple[Path, str]:
    item = base._exact_keys(value, {"path", "sha256"}, label)
    path = _inside(_absolute(item["path"], f"{label}.path"), shared_root,
                   f"{label}.path")
    _plain_ancestors(path, shared_root, f"{label}.path")
    digest = base._sha(item["sha256"], f"{label}.sha256")
    if _sha(path) != digest:
        raise ValueError(f"{label} hash differs")
    return path, digest


def _directory_binding(value: Any, label: str, shared_root: Path) -> tuple[Path, dict[str, str]]:
    item = base._exact_keys(value, {"path", "files"}, label)
    root = _inside(_absolute(item["path"], f"{label}.path"), shared_root,
                   f"{label}.path")
    _plain_ancestors(root, shared_root, f"{label}.path")
    expected = item["files"]
    if not isinstance(expected, dict) or not expected:
        raise ValueError(f"{label}.files must be a nonempty recursive file map")
    actual: dict[str, str] = {}
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        info = path.lstat()
        if (stat.S_ISLNK(info.st_mode) or
                getattr(info, "st_file_attributes", 0) &
                getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)):
            raise ValueError(f"{label} contains a redirected member: {path}")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            actual[relative] = _sha(path)
    if actual != expected or any(not s21._is_sha(value) for value in expected.values()):
        raise ValueError(f"{label} recursive physical file map differs")
    return root, expected


def _runtime(value: Any) -> None:
    runtime = base._exact_keys(value, {"git_commit", "files"}, "runtime")
    commit = runtime["git_commit"]
    if not isinstance(commit, str) or base.HEX40.fullmatch(commit) is None:
        raise ValueError("runtime.git_commit must be a full lowercase commit id")
    if not base._git_is_ancestor(commit):
        raise ValueError("runtime commit is not an ancestor of this checkout")
    files = runtime["files"]
    if not isinstance(files, dict) or set(files) != set(RUNTIME_FILES):
        raise ValueError("runtime.files does not bind the exact live-loop closure")
    # The original 51 files are protected byte-for-byte by their own validator.
    for relative in RUNTIME_FILES:
        digest = base._sha(files[relative], f"runtime.files[{relative}]")
        if (base._git_blob_sha(commit, relative) != digest or
                not base._git_path_clean(commit, relative)):
            raise ValueError(f"live-loop runtime source changed: {relative}")


def _protocol(initial: Mapping[str, Any]) -> dict[str, Any]:
    canonical = Path(initial["dataset"]) / dispatch.CANONICAL_INPUT
    files = initial["release"]["queue"]["canonical_files"]
    return {
        "scope": dispatch.SCOPE,
        "workers": list(base.EXPECTED_WORKERS),
        "priority": dispatch.LOW_TIER_PRIORITY,
        "config_files": {name: files[name] for name in
                         ("config.yaml", "measurement.json", "score_spec.json")},
    }


def validate_bootstrap(path: str | Path, required_sha256: str) -> dict[str, Any]:
    """Authenticate the immutable loop bootstrap and unchanged external batch 1."""
    bootstrap_path = Path(path)
    digest = base._sha(required_sha256, "bootstrap sha256")
    if _sha(bootstrap_path) != digest:
        raise ValueError("live-loop bootstrap hash differs")
    value = base._exact_keys(
        pb.read_json(bootstrap_path),
        {"schema_version", "release_id", "stage", "actor_root", "shared_root",
         "runtime", "initial_survey", "registry", "budget", "protocol"},
        "bootstrap")
    if value["schema_version"] != SCHEMA_VERSION or value["stage"] != STAGE:
        raise ValueError("unsupported live-loop bootstrap schema or stage")
    base._name(value["release_id"], "bootstrap.release_id")
    actor = _absolute(value["actor_root"], "bootstrap.actor_root")
    shared = _absolute(value["shared_root"], "bootstrap.shared_root")
    _inside(shared, actor, "bootstrap.shared_root")
    _plain_ancestors(shared, actor, "bootstrap.shared_root")
    normalized_bootstrap = _inside(bootstrap_path.resolve(strict=True), shared,
                                   "bootstrap path")
    _plain_ancestors(normalized_bootstrap, shared, "bootstrap path")
    _runtime(value["runtime"])
    registry = base._exact_keys(value["registry"], {"directory"}, "registry")
    registry_dir = _inside(_absolute(registry["directory"], "registry.directory"),
                           actor, "registry.directory")
    _inside(registry_dir, shared, "registry.directory")
    _plain_ancestors(registry_dir, shared, "registry.directory")
    if value["budget"] != BUDGET:
        raise ValueError("bootstrap bounded 3x60 budget differs")
    initial_binding = base._exact_keys(
        value["initial_survey"], {"release", "dataset_root"}, "initial_survey")
    external_release = base._exact_keys(initial_binding["release"], {"path", "sha256"},
                                        "initial_survey.release")
    initial_release = _absolute(external_release["path"], "initial_survey.release.path")
    _plain_external(initial_release, "initial_survey.release.path")
    initial_sha = base._sha(external_release["sha256"], "initial_survey.release.sha256")
    if _sha(initial_release) != initial_sha:
        raise ValueError("initial_survey.release hash differs")
    initial_context = dispatch._context(initial_release, initial_sha)
    initial_context = dict(initial_context)
    initial_context["release_path"] = initial_release
    initial_dataset = _absolute(initial_binding["dataset_root"],
                                "initial_survey.dataset_root")
    _plain_external(initial_dataset, "initial_survey.dataset_root")
    if initial_dataset != Path(initial_context["dataset"]).resolve():
        raise ValueError("bootstrap initial dataset differs from the initial release")
    if value["protocol"] != _protocol(initial_context):
        raise ValueError("bootstrap live-loop protocol differs from batch 1")
    return {
        "bootstrap": value,
        "bootstrap_path": normalized_bootstrap,
        "bootstrap_sha256": digest,
        "actor_root": actor,
        "shared_root": shared,
        "registry_dir": registry_dir,
        "initial_context": initial_context,
    }


def _successor_entry(dataset: Path, entry: Any, batch: int, shard: int,
                     machine: str) -> dict[str, Any]:
    label = f"queue.jobs[{shard - 1}]"
    item = base._exact_keys(entry, {"role", "shard", "job", "input_files"}, label)
    job = base._exact_keys(item["job"],
                           {"input", "store", "prio", "config", "scope", "machine"},
                           f"{label}.job")
    relative = f"batch-{batch:03d}_split/shard-{shard:03d}_input"
    expected_store = f"dedust_r81s21live_b{batch:03d}_{shard:02d}"
    if (item["role"] != SUCCESSOR_ROLE or item["shard"] != shard or
            job != {"input": relative, "store": expected_store,
                    "prio": dispatch.LOW_TIER_PRIORITY,
                    "config": f"{relative}/config.yaml", "scope": dispatch.SCOPE,
                    "machine": machine}):
        raise ValueError(f"{label} identity, priority, store, or machine differs")
    native = {"role": SUCCESSOR_ROLE, "input_files": item["input_files"],
              "job": {key: job[key] for key in ("input", "store", "prio", "config", "scope")}}
    _profile, rows, folder = formal._validate_shard_input(dataset, native, label)
    config = pb.load_profile_config(folder / "config.yaml")
    if not dispatch._profile_ok(folder, config) or len(rows) != ROWS_PER_SHARD:
        raise ValueError(f"{label} is not an exact 20-row native S21 profile")
    allowed_arms = set(cycle.ARM_COUNTS)
    for row in rows:
        pattern = torch.load(folder / row["pattern_file"], weights_only=True,
                             map_location="cpu")
        if (row.get("batch") != batch or row.get("selection_arm") not in allowed_arms or
                not dispatch._has_dual_pads(pattern)):
            raise ValueError(f"{label} row batch, selection arm, or dual pads differ")
    return item


def _cycle_evidence(value: Any, *, batch: int, dataset: Path,
                    shared_root: Path) -> dict[str, Any]:
    evidence = base._exact_keys(
        value,
        {"batch", "selection_receipt", "candidate_pool_receipt", "rank_receipt",
         "model_completion_receipt", "frozen_snapshot", "canonical_manifest_sha256",
         "split_receipt_sha256"}, "cycle")
    if evidence["batch"] != batch:
        raise ValueError("cycle batch differs from successor release")
    selection_path, _ = _binding(evidence["selection_receipt"],
                                 "cycle.selection_receipt", shared_root)
    pool_path, _ = _binding(evidence["candidate_pool_receipt"],
                            "cycle.candidate_pool_receipt", shared_root)
    rank_path, _ = _binding(evidence["rank_receipt"], "cycle.rank_receipt", shared_root)
    completion_path, completion_sha = _binding(
        evidence["model_completion_receipt"], "cycle.model_completion_receipt", shared_root)
    snapshot_path, _snapshot_files = _directory_binding(
        evidence["frozen_snapshot"], "cycle.frozen_snapshot", shared_root)
    selection = pb.read_json(selection_path)
    pool = pb.read_json(pool_path)
    rank = pb.read_json(rank_path)
    completion = pb.read_json(completion_path)
    if not all(isinstance(item, dict) for item in (selection, pool, rank, completion)):
        raise ValueError("cycle evidence must contain JSON objects")
    model_dir = completion_path.parent
    summary = s21._validated_model_summary(
        model_dir, rank_ready=True, expected_completion_sha256=completion_sha)
    if (completion != s21._completion_receipt(model_dir, summary) or
            selection.get("model") != {
                "signature": summary["signature"],
                "completion_receipt_sha256": completion_sha,
                "model_sha256": summary["model_sha256"],
            }):
        raise ValueError("cycle model completion/summary/selection binding differs")
    for name, digest in summary["model_sha256"].items():
        if _sha(model_dir / name) != digest:
            raise ValueError("cycle physical ensemble model differs from completion receipt")
    current = s21.load_current_snapshot(snapshot_path, s21.load_protocol())
    trained_snapshot = selection.get("trained_snapshot")
    if (not isinstance(trained_snapshot, dict) or
            trained_snapshot.get("receipt") != current.receipt or
            summary.get("snapshot") != current.receipt):
        raise ValueError("cycle frozen snapshot differs from training plan/selection")
    snapshot_results = pb.read_json(snapshot_path / "results.json")
    expected_successes = sorted(
        ({"pattern_sha256": row["pattern_sha256"],
          "sample_sha256": snapshot_results[row["id"]]["sample_sha256"]}
         for row in current.rows), key=lambda item: item["pattern_sha256"])
    if trained_snapshot.get("successes") != expected_successes:
        raise ValueError("cycle frozen snapshot success facts differ")
    selected = selection.get("selected")
    hashes = ([row.get("pattern_sha256") for row in selected]
              if isinstance(selected, list) else [])
    arms = ([row.get("selection_arm") for row in selected]
            if isinstance(selected, list) else [])
    counts = {arm: arms.count(arm) for arm in cycle.ARM_COUNTS}
    frozen_successes = trained_snapshot.get("successes")
    trigger = selection.get("trigger_new_success_pattern_sha256")
    trigger_detail = selection.get("trigger")
    if not isinstance(trigger_detail, dict) or set(trigger_detail) != {
            "successes", "new_pattern_sha256", "required_batch",
            "required_batch_new_pattern_sha256", "source_inventory_id"}:
        raise ValueError("cycle trigger evidence fields differ")
    required_new = trigger_detail["required_batch_new_pattern_sha256"]
    if (trigger_detail["successes"] != cycle.TRIGGER_NEW_SUCCESSES or
            trigger_detail["new_pattern_sha256"] != trigger or
            not isinstance(required_new, list) or not set(required_new) <= set(trigger) or
            (batch == 2 and (trigger_detail["required_batch"] is not None or
                             required_new != trigger)) or
            (batch == 3 and (trigger_detail["required_batch"] != 2 or
                             len(required_new) < cycle.TRIGGER_NEW_SUCCESSES))):
        raise ValueError("cycle trigger or batch-3 new-batch-2 gate differs")
    if (selection.get("schema_version") != cycle.SCHEMA_VERSION or
            selection.get("batch") != batch or len(hashes) != BATCH_SIZE or
            len(set(hashes)) != BATCH_SIZE or any(not s21._is_sha(item) for item in hashes) or
            counts != cycle.ARM_COUNTS or
            selection.get("pool") != pool or selection.get("ranking") != rank or
            rank.get("candidate_set_id") != pool.get("candidate_set_id") or
            pool.get("rank_receipt_sha256") != _sha(rank_path) or
            not isinstance(trigger, list) or len(trigger) < cycle.TRIGGER_NEW_SUCCESSES or
            not isinstance(frozen_successes, list) or
            not set(trigger) <= {item.get("pattern_sha256") for item in frozen_successes
                                 if isinstance(item, dict)}):
        raise ValueError("cycle selection/model/pool/ranking/frozen-source evidence differs")
    pool_manifest_path = pool_path.parent / "manifest.json"
    if (_sha(pool_manifest_path) != pool.get("manifest_sha256") or
            not isinstance(pb.read_json(pool_manifest_path), list)):
        raise ValueError("cycle candidate-pool manifest differs from its receipt")
    pool_rows = pb.read_json(pool_manifest_path)
    if (pool.get("candidate_count") != len(pool_rows) or len(pool_rows) < BATCH_SIZE):
        raise ValueError("cycle candidate-pool count is invalid")
    pool_by_id: dict[str, dict[str, Any]] = {}
    file_bindings = []
    pool_root = pool_path.parent.resolve()
    for index, row in enumerate(pool_rows):
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or \
                row["id"] in pool_by_id:
            raise ValueError("cycle candidate-pool ids are malformed or repeated")
        relative = row.get("pattern_file")
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("cycle candidate-pool pattern path is unsafe")
        pattern_path = (pool_root / relative).resolve(strict=True)
        _inside(pattern_path, pool_root, "cycle candidate-pool pattern")
        file_sha = base._sha(row.get("pattern_file_sha256"),
                             "candidate_pool.pattern_file_sha256")
        if _sha(pattern_path) != file_sha or not s21._is_sha(row.get("pattern_sha256")):
            raise ValueError("cycle candidate-pool pattern bytes or identity differ")
        pool_by_id[row["id"]] = {**row, "pool_index": index}
        file_bindings.append({"id": row["id"], "file_sha256": file_sha})
    if cycle._json_id(file_bindings) != pool.get("pattern_files_id"):
        raise ValueError("cycle candidate-pool physical-file identity differs")
    claims = selection.get("claims")
    if (not isinstance(claims, dict) or claims.get("provisional_only") is not True or
            claims.get("formal_pass") is not False or
            claims.get("survey_complete_is_spec_pass") is not False):
        raise ValueError("cycle evidence makes an unsupported result claim")
    canonical = dataset / f"batch-{batch:03d}_input"
    split = dataset / f"batch-{batch:03d}_split" / "split_receipt.json"
    canonical_sha = base._sha(evidence["canonical_manifest_sha256"],
                              "cycle.canonical_manifest_sha256")
    split_sha = base._sha(evidence["split_receipt_sha256"],
                          "cycle.split_receipt_sha256")
    if (_sha(canonical / "manifest.json") != canonical_sha or _sha(split) != split_sha or
            selection.get("bundle", {}).get("split_receipt_sha256") != split_sha or
            selection.get("bundle", {}).get("canonical_files", {}).get("manifest.json") !=
            canonical_sha):
        raise ValueError("cycle canonical or split receipt differs from published data")
    canonical_rows = pb.read_json(canonical / "manifest.json")
    if len(canonical_rows) != len(selected):
        raise ValueError("cycle selection and canonical row counts differ")
    comparable = ("pattern_sha256", "prediction", "predicted_worst_margin_db",
                  "predicted_uncertainty")
    for ordinal, (picked, published) in enumerate(zip(selected, canonical_rows)):
        candidate = pool_by_id.get(picked.get("id"))
        if (candidate is None or picked.get("pool_index") != candidate["pool_index"] or
                published.get("candidate_id") != picked.get("id") or
                published.get("pool_index") != picked.get("pool_index") or
                published.get("selection_arm") != picked.get("selection_arm") or
                any(picked.get(key) != candidate.get(key) for key in comparable) or
                any(published.get(key) != picked.get(key) for key in comparable)):
            raise ValueError(f"cycle selected candidate differs from pool/canonical row {ordinal}")
    return {"selection": selection, "pool": pool, "rank": rank,
            "completion": completion}


def validate_successor_release(path: str | Path, required_sha256: str,
                               bootstrap_context: Mapping[str, Any]) -> dict[str, Any]:
    """Authenticate one immutable 60-row successor release and its cycle proof."""
    shared = Path(bootstrap_context["shared_root"])
    release_path = _inside(_absolute(os.fspath(path), "successor release"), shared,
                           "successor release")
    _plain_ancestors(release_path, shared, "successor release")
    digest = base._sha(required_sha256, "successor release sha256")
    if _sha(release_path) != digest:
        raise ValueError("successor release hash differs")
    value = base._exact_keys(
        pb.read_json(release_path),
        {"schema_version", "release_id", "stage", "sequence", "dataset_root",
         "bootstrap", "previous_registry", "cycle", "queue", "policy"}, "release")
    if (value["schema_version"] != SCHEMA_VERSION or value["stage"] != SUCCESSOR_STAGE or
            value["sequence"] not in (2, 3) or value["policy"] != POLICY):
        raise ValueError("unsupported successor release schema, sequence, stage, or policy")
    batch = value["sequence"]
    base._name(value["release_id"], "release.release_id")
    bootstrap_ref = base._exact_keys(value["bootstrap"], {"path", "sha256"},
                                     "release.bootstrap")
    if (bootstrap_ref["path"] != str(bootstrap_context["bootstrap_path"]) or
            bootstrap_ref["sha256"] != bootstrap_context["bootstrap_sha256"]):
        raise ValueError("successor release bootstrap binding differs")
    previous = base._exact_keys(value["previous_registry"],
                                {"sequence", "path", "sha256"},
                                "release.previous_registry")
    if previous["sequence"] != batch - 1:
        raise ValueError("successor release previous registry sequence differs")
    previous_path = _inside(_absolute(previous["path"], "release.previous_registry.path"),
                            shared, "release.previous_registry.path")
    _plain_ancestors(previous_path, shared, "release.previous_registry.path")
    if _sha(previous_path) != base._sha(previous["sha256"],
                                        "release.previous_registry.sha256"):
        raise ValueError("successor release previous registry hash differs")
    dataset = _inside(_absolute(value["dataset_root"], "release.dataset_root"), shared,
                      "release.dataset_root")
    _plain_ancestors(dataset, shared, "release.dataset_root")
    if dataset.name != "dataset":
        raise ValueError("successor dataset_root must be an explicit dataset directory")
    queue = base._exact_keys(value["queue"],
                             {"jobs_sha256", "jobs", "canonical_files",
                              "split_receipt_sha256"}, "queue")
    jobs_path = dataset / "jobs.json"
    if _sha(jobs_path) != base._sha(queue["jobs_sha256"], "queue.jobs_sha256"):
        raise ValueError("successor jobs.json differs")
    entries = queue["jobs"]
    if not isinstance(entries, list) or len(entries) != 3 or \
            pb.read_json(jobs_path) != [item.get("job") for item in entries]:
        raise ValueError("successor queue must contain exactly three ordered jobs")
    canonical = dataset / f"batch-{batch:03d}_input"
    base._validate_file_map(canonical, queue["canonical_files"], "queue.canonical_files")
    cfg = pb.load_profile_config(canonical / "config.yaml")
    rows = pb.validate_input(canonical, cfg)
    if not dispatch._profile_ok(canonical, cfg) or len(rows) != BATCH_SIZE:
        raise ValueError("successor canonical input is not exact 60-row native S21")
    validated = [_successor_entry(dataset, item, batch, shard, machine)
                 for shard, (item, machine) in enumerate(
                     zip(entries, base.EXPECTED_WORKERS), start=1)]
    split_path = dataset / f"batch-{batch:03d}_split" / "split_receipt.json"
    split_sha = base._sha(queue["split_receipt_sha256"], "queue.split_receipt_sha256")
    if _sha(split_path) != split_sha:
        raise ValueError("successor split receipt differs")
    split = pb.read_json(split_path)
    shards = split.get("shards") if isinstance(split, dict) else None
    if (split.get("canonical_manifest_sha256") != _sha(canonical / "manifest.json") or
            split.get("shard_size") != ROWS_PER_SHARD or split.get("round_robin") is not True or
            not isinstance(shards, list) or len(shards) != 3):
        raise ValueError("successor split is not the exact 3x20 round-robin partition")
    for offset, (entry, shard_receipt) in enumerate(zip(validated, shards)):
        folder = dataset / entry["job"]["input"]
        child_rows = pb.read_json(folder / "manifest.json")
        expected = rows[offset::3]
        if (child_rows != expected or shard_receipt.get("ids") != [row["id"] for row in expected] or
                shard_receipt.get("manifest_sha256") != _sha(folder / "manifest.json")):
            raise ValueError("successor shard differs from its exact canonical partition")
        for marker in ("config.yaml", "measurement.json", "score_spec.json"):
            if _sha(folder / marker) != _sha(canonical / marker):
                raise ValueError(f"successor shard changed canonical {marker}")
    evidence = _cycle_evidence(value["cycle"], batch=batch, dataset=dataset,
                               shared_root=shared)
    selection = evidence["selection"]
    if (selection.get("bundle", {}).get("canonical_files") != queue["canonical_files"] or
            selection.get("bundle", {}).get("split_receipt_sha256") != split_sha):
        raise ValueError("successor queue differs from authenticated cycle bundle")
    if batch == 3:
        previous_context = validate_registry_record(
            previous_path, previous["sha256"], bootstrap_context)
        if previous_context["record"]["sequence"] != 2:
            raise ValueError("batch 3 must follow the exact published batch 2")
        batch2_inventory = cycle._input_store_inventory({
            "dataset": previous_context["dataset"],
            "entries": previous_context["entries"],
        })
        snapshot_samples = {
            item["pattern_sha256"]: item["sample_sha256"]
            for item in selection["trained_snapshot"]["successes"]}
        required_new = selection["trigger"]["required_batch_new_pattern_sha256"]
        if any(digest not in batch2_inventory["successes"] or
               snapshot_samples.get(digest) !=
               batch2_inventory["successes"][digest]["sample_sha256"]
               for digest in required_new):
            raise ValueError("batch 3 trigger lacks twenty frozen successful batch-2 truths")
    allowed = {canonical.name, f"batch-{batch:03d}_split", "jobs.json", "jobs_state"} | {
        item["job"]["store"] for item in validated}
    unknown = {item.name for item in dataset.iterdir()} - allowed
    if unknown:
        raise ValueError(f"successor dataset contains unknown top-level entries: {sorted(unknown)}")
    state = dataset / "jobs_state"
    stores = {item["job"]["store"] for item in validated}
    if state.exists():
        for marker in state.iterdir():
            if marker.is_file() and marker.suffix in {".claim", ".done", ".fail"} and \
                    marker.stem not in stores:
                raise ValueError(f"successor jobs_state has unknown marker: {marker.name}")
    states = [formal._job_state(dataset, entry, (entry["job"]["machine"],))
              for entry in validated]
    return {"release": value, "release_path": release_path, "release_sha256": digest,
            "dataset": dataset, "entries": validated, "states": states,
            "cycle_evidence": evidence}


def validate_registry_record(path: str | Path, required_sha256: str,
                             bootstrap_context: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one create-only record and its referenced release/data root."""
    registry_dir = Path(bootstrap_context["registry_dir"])
    record_path = _inside(_absolute(os.fspath(path), "registry record"), registry_dir,
                          "registry record")
    match = REGISTRY_NAME.fullmatch(record_path.name)
    if match is None:
        raise ValueError("registry filename must be batch-NNN.json")
    digest = base._sha(required_sha256, "registry record sha256")
    if _sha(record_path) != digest:
        raise ValueError("registry record hash differs")
    record = base._exact_keys(pb.read_json(record_path),
                              {"schema_version", "sequence", "previous", "bootstrap", "release"},
                              "registry record")
    sequence = int(match.group(1))
    if (record["schema_version"] != SCHEMA_VERSION or record["sequence"] != sequence or
            sequence < 1 or sequence > MAX_BATCHES):
        raise ValueError("registry record schema or bounded sequence differs")
    bootstrap = base._exact_keys(record["bootstrap"], {"path", "sha256"},
                                 "registry.bootstrap")
    if (bootstrap["path"] != str(bootstrap_context["bootstrap_path"]) or
            bootstrap["sha256"] != bootstrap_context["bootstrap_sha256"]):
        raise ValueError("registry bootstrap binding differs")
    release_ref = base._exact_keys(record["release"], {"path", "sha256", "dataset_root"},
                                   "registry.release")
    release_path = _absolute(release_ref["path"], "registry.release.path")
    dataset = _absolute(release_ref["dataset_root"], "registry.release.dataset_root")
    if sequence == 1:
        if record["previous"] is not None:
            raise ValueError("batch 1 registry record must not have a previous record")
        initial = bootstrap_context["initial_context"]
        if (release_path != Path(initial["release_path"]).resolve() or
                release_ref["sha256"] != initial["release_sha256"] or
                dataset != Path(initial["dataset"]).resolve()):
            raise ValueError("batch 1 registry record differs from immutable initial survey")
        context = initial
    else:
        release_path = _inside(release_path, Path(bootstrap_context["shared_root"]),
                               "registry.release.path")
        dataset = _inside(dataset, Path(bootstrap_context["shared_root"]),
                          "registry.release.dataset_root")
        previous = base._exact_keys(record["previous"], {"sequence", "path", "sha256"},
                                    "registry.previous")
        expected_previous = registry_dir / f"batch-{sequence - 1:03d}.json"
        if (previous["sequence"] != sequence - 1 or
                _absolute(previous["path"], "registry.previous.path") != expected_previous or
                _sha(expected_previous) != base._sha(previous["sha256"],
                                                     "registry.previous.sha256")):
            raise ValueError("registry previous-record chain differs")
        context = validate_successor_release(release_path, release_ref["sha256"],
                                             bootstrap_context)
        if context["release"]["sequence"] != sequence or context["dataset"] != dataset:
            raise ValueError("registry successor release/data binding differs")
        if context["release"]["previous_registry"] != previous:
            raise ValueError("successor release and registry previous bindings differ")
    states = []
    for entry, state in zip(context["entries"], context["states"]):
        # Every live job has one immutable machine owner.  The legacy batch-1
        # validator models a three-host retry chain, but such a retry is neither
        # launchable nor authorized here.  Its already-authenticated first-owner
        # .fail marker is therefore terminal and is never reset or reassigned.
        fail = Path(context["dataset"]) / "jobs_state" / f"{entry['job']['store']}.fail"
        states.append("failed" if fail.is_file() else state)
    return {"record": record, "record_path": record_path, "record_sha256": digest,
            "release": context["release"], "dataset": Path(context["dataset"]),
            "entries": context["entries"], "states": states}


def registry_snapshot(bootstrap_path: str | Path, bootstrap_sha256: str) -> dict[str, Any]:
    """Read one contiguous, authenticated registry snapshot."""
    bootstrap = validate_bootstrap(bootstrap_path, bootstrap_sha256)
    paths = sorted(item for item in bootstrap["registry_dir"].iterdir()
                   if item.is_file() and REGISTRY_NAME.fullmatch(item.name))
    unknown = [item.name for item in bootstrap["registry_dir"].iterdir()
               if item.is_file() and not REGISTRY_NAME.fullmatch(item.name)]
    if unknown:
        raise ValueError(f"registry contains unknown files: {sorted(unknown)}")
    if not paths or [item.name for item in paths] != [
            f"batch-{index:03d}.json" for index in range(1, len(paths) + 1)]:
        raise ValueError("registry must start at batch 001 and remain contiguous")
    if len(paths) > MAX_BATCHES:
        raise ValueError("registry exceeds the bounded three-batch budget")
    records = [validate_registry_record(item, _sha(item), bootstrap) for item in paths]
    for index in range(1, len(records)):
        previous = records[index]["record"]["previous"]
        expected = records[index - 1]
        if previous != {"sequence": index, "path": str(expected["record_path"]),
                        "sha256": expected["record_sha256"]}:
            raise ValueError("registry SHA chain is not exact")
    return {"bootstrap_context": bootstrap, "records": records}


def stop_requested(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Return a read-only shared stop decision across the actor and every batch."""
    context = snapshot["bootstrap_context"]
    roots = [("actor", Path(context["actor_root"]))]
    roots.extend((f"batch-{item['record']['sequence']:03d}", Path(item["dataset"]) / "jobs_state")
                 for item in snapshot["records"])
    names = ("STOP", f"STOP_{dispatch.SCOPE}")
    found = [str(root / name) for _label, root in roots for name in names
             if (root / name).is_file()]
    return {"stopped": bool(found), "markers": found}


def worker_status(bootstrap_path: str | Path, bootstrap_sha256: str,
                  machine: str) -> dict[str, Any]:
    """Return the next published shard for one host without changing any state."""
    if machine not in base.EXPECTED_WORKERS:
        raise ValueError("machine is not in the fixed R81 worker roster")
    snapshot = registry_snapshot(bootstrap_path, bootstrap_sha256)
    stop = stop_requested(snapshot)
    if stop["stopped"]:
        return {"schema_version": SCHEMA_VERSION, "status": "stopped",
                "machine": machine, "stop_markers": stop["markers"]}
    for record in snapshot["records"]:
        matches = [(entry, state) for entry, state in zip(record["entries"], record["states"])
                   if entry["job"]["machine"] == machine]
        if len(matches) != 1:
            raise ValueError("each registry batch must assign exactly one shard to this host")
        entry, state = matches[0]
        if state == "failed":
            return {"schema_version": SCHEMA_VERSION, "status": "terminal_failure",
                    "machine": machine, "sequence": record["record"]["sequence"],
                    "dataset_root": str(record["dataset"]), "job": entry["job"]}
        if state == "active":
            return {"schema_version": SCHEMA_VERSION, "status": "ready",
                    "machine": machine, "sequence": record["record"]["sequence"],
                    "dataset_root": str(record["dataset"]), "job": entry["job"],
                    "registry_record": str(record["record_path"]),
                    "registry_sha256": record["record_sha256"]}
        # done and toxic/skipped are terminal execution states.  A skipped row
        # is never truth, but it must not strand this host behind other shards.
    count = len(snapshot["records"])
    return {"schema_version": SCHEMA_VERSION,
            "status": "budget_complete" if count == MAX_BATCHES else "waiting_for_publication",
            "machine": machine, "published_batches": count,
            "maximum_batches": MAX_BATCHES}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=Path, required=True)
    parser.add_argument("--bootstrap-sha256", required=True)
    parser.add_argument("--machine", required=True, choices=base.EXPECTED_WORKERS)
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = worker_status(args.bootstrap, args.bootstrap_sha256, args.machine)
    if args.report is not None:
        pb.atomic_json(args.report, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
