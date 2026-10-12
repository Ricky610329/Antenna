"""Prepare and validate an isolated three-worker R81 S21 exploration.

This stage is deliberately provisional.  It remeasures the already staged 60
dual-port geometries with a Discrete sweep and scores only three S21 bands.  It
does not consume, satisfy, or bypass the R81 Engineering -> Formal gate, and it
must never be represented as a formal campaign result.

``--prepare-local`` writes review artifacts only below this checkout's ``tmp``
directory.  A later coordinator may copy the prepared dataset to a fresh NAS
root and construct an immutable release after the additive runtime is committed.
This module itself never starts a worker or HFSS.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import stat
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml

from antenna.measurement import measurement_id, score_spec_id
from script import profiled_batch as pb
from script import profiled_shards
from script import r81_formal_sharded_entry as formal
from script import r81_sweep_probe_entry as probe
from script import r81_worker_entry as base


REPO = Path(__file__).resolve().parents[1]
STAGE = "ProvisionalS21Discrete"
SCOPE = base.SCOPE
PROTECTED_RUNTIME_COMMIT = "88d4c372e4cbab2f07712afcb869c6881cd7aed6"
ADDITIVE_FILES = {
    "configs/dual_r81_s21_discrete.yaml",
    "script/r81_provisional_filter_dispatch.py",
    "script/start_r81_provisional_filter_worker.ps1",
}
RUNTIME_FILES = tuple(sorted(set(probe.RUNTIME_FILES) | ADDITIVE_FILES))
PROFILE_CONFIG = REPO / "configs/dual_r81_s21_discrete.yaml"
CANONICAL_INPUT = "dedust_r81s21provisional_input"
SPLIT_ROOT = "dedust_r81s21provisional_split"
STORE_PREFIX = "dedust_r81s21provisional"
ROLE = "provisional_s21_discrete"
LOW_TIER_PRIORITY = 6
_PROFILE_DOCUMENT = yaml.safe_load(PROFILE_CONFIG.read_text(encoding="utf-8"))
S21_SPEC = _PROFILE_DOCUMENT["score_spec"]
POLICY = {
    "provisional_exploration_only": True,
    "formal_release_allowed": False,
    "engineering_gate_satisfied": False,
    "engineering_gate_bypassed": False,
    "symmetry_required": False,
    "sweep_type": "Discrete",
    "score_labels": ["S21"],
    "row_count": 60,
    "shard_count": 3,
    "rows_per_shard": 20,
    "machine_pinned": True,
    "campaign_complete": False,
    "performance_target_assessed": False,
    "survey_complete_is_spec_pass": False,
    "success_definition": "three-band worst_margin_db > 0 plus an independent repeat",
}


def _files(folder: Path) -> dict[str, str]:
    return {path.name: pb.file_sha256(path)
            for path in sorted(folder.iterdir(), key=lambda item: item.name)
            if path.is_file()}


def _plain_tree(root: Path, label: str) -> None:
    canonical = root.resolve(strict=True)
    for path in (root, *root.rglob("*")):
        metadata = path.lstat()
        if (stat.S_ISLNK(metadata.st_mode) or
                getattr(metadata, "st_file_attributes", 0) &
                getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)):
            raise ValueError(f"{label} contains a redirected path: {path}")
        try:
            path.resolve(strict=True).relative_to(canonical)
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            raise ValueError(f"{label} member escapes its root: {path}") from exc


def _nonoverlap(left: Path, right: Path) -> None:
    a, b = left.resolve(strict=True), right.resolve(strict=True)
    for child, parent in ((a, b), (b, a)):
        try:
            child.relative_to(parent)
        except ValueError:
            pass
        else:
            raise ValueError("provisional dataset and source dataset must not overlap")


def _source_context(path: str | Path, required_sha256: str) -> dict[str, Any]:
    release_path = Path(path)
    if not release_path.is_absolute():
        raise ValueError("source release path must be absolute")
    digest = base._sha(required_sha256, "source release sha256")
    if pb.file_sha256(release_path) != digest:
        raise ValueError("source EngineeringOnly release hash differs")
    # Structural authentication is useful even though its unfinished physical
    # measurement gate is explicitly irrelevant to this provisional stage.
    base.validate_release(release_path, digest, "EngineeringOnly")
    release = pb.read_json(release_path)
    dataset = Path(release["dataset_root"]).resolve(strict=True)
    canonical_binding = base._exact_keys(
        release["queue"]["formal_input"], {"name", "files"},
        "source queue.formal_input")
    if canonical_binding["name"] != "dedust_r81b1_input":
        raise ValueError("source must bind the original staged R81 batch-1 input")
    canonical = dataset / canonical_binding["name"]
    base._validate_file_map(canonical, canonical_binding["files"],
                            "source queue.formal_input.files")
    cfg = pb.load_profile_config(canonical / "config.yaml")
    rows = pb.validate_input(canonical, cfg)
    if cfg.port != "dual" or cfg.scope != SCOPE or len(rows) != 60:
        raise ValueError("source canonical input must be the 60-row dual R81 profile")
    files = {(Path(canonical.name) / name).as_posix(): digest
             for name, digest in sorted(canonical_binding["files"].items())}
    binding = {
        "release": {"path": str(release_path.resolve(strict=True)), "sha256": digest},
        "dataset_root": str(dataset),
        "canonical_input": canonical.name,
        "canonical_files": files,
    }
    return {"binding": binding, "release": release, "dataset": dataset,
            "canonical": canonical, "config": cfg, "rows": rows}


def _provisional_config(source_cfg: Any) -> dict[str, Any]:
    config = copy.deepcopy(_PROFILE_DOCUMENT)
    expected_geometry = config["measurement"]["geometry"]
    if source_cfg.measurement.get("geometry") != expected_geometry:
        raise ValueError("source staged geometry differs from the authoritative S21 profile")
    return config


def _has_dual_pads(pattern: torch.Tensor) -> bool:
    value = pattern.reshape(25, 25).bool()
    return bool(value[:5, 10:15].all() and value[20:, 10:15].all())


def _transformed_rows(source: Mapping[str, Any], cfg: Any) -> list[dict[str, Any]]:
    """Pure, ordered transformation from the bound source canonical manifest."""
    mid, sid = measurement_id(cfg.measurement), score_spec_id(cfg.score_spec)
    rows = []
    for index, old in enumerate(source["rows"]):
        row = copy.deepcopy(old)
        row["id"] = f"r81s21p_{index:03d}_{old['pattern_sha256'][:8]}"
        row["pattern_file"] = row["id"] + ".pt"
        row["parent_batch_id"] = old["id"]
        row["source_selection_arm"] = old.get("selection_arm")
        row["selection_arm"] = "provisional_existing_pool"
        row["measurement_id"], row["score_spec_id"] = mid, sid
        rows.append(row)
    return rows


def _write_canonical(source: Mapping[str, Any], target: Path) -> None:
    target.mkdir()
    config = _provisional_config(source["config"])
    (target / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8", newline="\n")
    cfg = pb.load_profile_config(target / "config.yaml")
    rows = _transformed_rows(source, cfg)
    for old, row in zip(source["rows"], rows):
        pattern = torch.load(source["canonical"] / old["pattern_file"],
                             weights_only=True, map_location="cpu")
        if not _has_dual_pads(pattern):
            raise ValueError(f"source row lacks both dual-port pads: {old['id']}")
        shutil.copy2(source["canonical"] / old["pattern_file"],
                     target / row["pattern_file"])
    pb.atomic_json(target / "measurement.json", cfg.measurement)
    pb.atomic_json(target / "score_spec.json", cfg.score_spec)
    pb.atomic_json(target / "manifest.json", rows)
    pb.validate_input(target, cfg)


def _jobs(dataset: Path, priority: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if (isinstance(priority, bool) or not isinstance(priority, int) or
            priority < LOW_TIER_PRIORITY):
        raise ValueError(f"provisional priority must remain low tier (>= {LOW_TIER_PRIORITY})")
    entries, jobs = [], []
    for shard, machine in enumerate(base.EXPECTED_WORKERS, start=1):
        relative = f"{SPLIT_ROOT}/shard-{shard:03d}_input"
        job = {"input": relative, "store": f"{STORE_PREFIX}{shard:02d}",
               "prio": priority, "config": f"{relative}/config.yaml",
               "scope": SCOPE, "machine": machine}
        folder = dataset / Path(relative)
        rows = pb.validate_input(folder, pb.load_profile_config(folder / "config.yaml"))
        if len(rows) != 20:
            raise ValueError("each provisional native shard must contain exactly 20 rows")
        entries.append({"role": ROLE, "shard": shard, "job": job,
                        "input_files": _files(folder)})
        jobs.append(job)
    return entries, jobs


def prepare_local(source_release: Path, source_sha256: str, output_root: Path,
                  *, priority: int = LOW_TIER_PRIORITY) -> dict[str, Any]:
    """Create reviewable data and a release template without publishing or running."""
    source = _source_context(source_release, source_sha256)
    root = output_root.resolve(strict=False)
    try:
        root.relative_to((REPO / "tmp").resolve())
    except ValueError as exc:
        raise ValueError("--prepare-local output must stay below repository tmp") from exc
    if root.exists():
        raise FileExistsError("use a fresh local preparation root")
    dataset = root / "dataset"
    dataset.mkdir(parents=True)
    canonical = dataset / CANONICAL_INPUT
    _write_canonical(source, canonical)
    split = dataset / SPLIT_ROOT
    profiled_shards.split_bundle(canonical, split, shard_size=20)
    entries, jobs = _jobs(dataset, priority)
    pb.atomic_json(dataset / "jobs.json", jobs)
    source_binding = source["binding"]
    preparation = {
        "schema_version": 1,
        "status": "provisional_s21_discrete_local_prepared",
        "stage": STAGE,
        "dataset_root": str(dataset),
        "source": source_binding,
        "policy": copy.deepcopy(POLICY),
        "profile": {
            "measurement_id": measurement_id(pb.read_json(canonical / "measurement.json")),
            "score_spec_id": score_spec_id(pb.read_json(canonical / "score_spec.json")),
            "score_spec": copy.deepcopy(S21_SPEC),
        },
        "queue": {"jobs_sha256": pb.file_sha256(dataset / "jobs.json"),
                  "jobs": entries,
                  "canonical_files": _files(canonical),
                  "split_receipt_sha256": pb.file_sha256(split / "split_receipt.json"),
                  "workers": list(base.EXPECTED_WORKERS),
                  "priority": priority},
        "publication": {"release_created": False, "current_pointer_changed": False,
                        "nas_written": False, "worker_started": False,
                        "hfss_started": False},
        "next_step": "commit additive runtime, copy to a fresh NAS root, seal an immutable release",
    }
    pb.atomic_json(root / "preparation.json", preparation)
    _plain_tree(root, "local provisional preparation")
    return preparation


def _validate_runtime(runtime: Any) -> None:
    value = base._exact_keys(runtime, {"git_commit", "files"}, "runtime")
    commit = value["git_commit"]
    if not isinstance(commit, str) or base.HEX40.fullmatch(commit) is None:
        raise ValueError("runtime.git_commit must be a full lowercase commit id")
    if not base._git_is_ancestor(commit):
        raise ValueError("runtime commit is not an ancestor of this checkout")
    files = value["files"]
    if (len(probe.RUNTIME_FILES) != 48 or len(RUNTIME_FILES) != 51 or
            not isinstance(files, dict) or set(files) != set(RUNTIME_FILES)):
        raise ValueError("runtime must bind protected SweepProbe48 plus three additives")
    for name in probe.RUNTIME_FILES:
        if base._git_blob_sha(commit, name) != base._git_blob_sha(
                PROTECTED_RUNTIME_COMMIT, name):
            raise ValueError(f"provisional stage changed protected SweepProbe runtime: {name}")
    for name in RUNTIME_FILES:
        expected = base._sha(files[name], f"runtime.files[{name}]")
        if (base._git_blob_sha(commit, name) != expected or
                not base._git_path_clean(commit, name)):
            raise ValueError(f"runtime source changed: {name}")


def _profile_ok(folder: Path, cfg: Any) -> bool:
    raw = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    return (raw == _PROFILE_DOCUMENT and cfg.port == "dual" and cfg.scope == SCOPE and
            cfg.measurement == _PROFILE_DOCUMENT["measurement"] and
            cfg.score_spec == _PROFILE_DOCUMENT["score_spec"] and
            cfg.runtime == _PROFILE_DOCUMENT["runtime"] and cfg.exploration == {})


def _validate_entry(dataset: Path, entry: Any, shard: int, machine: str) -> dict[str, Any]:
    label = f"queue.jobs[{shard - 1}]"
    value = base._exact_keys(entry, {"role", "shard", "job", "input_files"}, label)
    job = base._exact_keys(value["job"],
                           {"input", "store", "prio", "config", "scope", "machine"},
                           f"{label}.job")
    relative = f"{SPLIT_ROOT}/shard-{shard:03d}_input"
    if (value["role"] != ROLE or value["shard"] != shard or
            job != {"input": relative, "store": f"{STORE_PREFIX}{shard:02d}",
                    "prio": job["prio"], "config": f"{relative}/config.yaml",
                    "scope": SCOPE, "machine": machine} or
            isinstance(job["prio"], bool) or not isinstance(job["prio"], int) or
            job["prio"] < LOW_TIER_PRIORITY):
        raise ValueError(f"{label} identity, low-tier priority, or machine pin differs")
    native = {"role": ROLE, "input_files": value["input_files"],
              "job": {key: job[key] for key in ("input", "store", "prio", "config", "scope")}}
    profile, rows, _folder = formal._validate_shard_input(dataset, native, label)
    cfg = pb.load_profile_config(dataset / Path(relative) / "config.yaml")
    if not _profile_ok(dataset / Path(relative), cfg) or len(rows) != 20 or profile != {
            "measurement_id": measurement_id(cfg.measurement),
            "score_spec_id": score_spec_id(cfg.score_spec), "runtime": cfg.runtime}:
        raise ValueError(f"{label} is not an exact 20-row S21-only Discrete profile")
    for row in rows:
        if (row.get("selection_arm") != "provisional_existing_pool" or
                not isinstance(row.get("parent_batch_id"), str) or
                not _has_dual_pads(torch.load(dataset / Path(relative) / row["pattern_file"],
                                               weights_only=True, map_location="cpu"))):
            raise ValueError(f"{label} row lineage or dual pads differ")
    return value


def _context(path: str | Path, required_sha256: str) -> dict[str, Any]:
    release_path = Path(path)
    digest = base._sha(required_sha256, "release sha256")
    if pb.file_sha256(release_path) != digest:
        raise ValueError("provisional release hash differs")
    value = base._exact_keys(
        pb.read_json(release_path),
        {"schema_version", "release_id", "stage", "dataset_root", "runtime",
         "source", "policy", "queue"}, "release")
    if value["schema_version"] != 1 or value["stage"] != STAGE or value["policy"] != POLICY:
        raise ValueError("unsupported provisional release schema, stage, or policy")
    base._name(value["release_id"], "release.release_id")
    _validate_runtime(value["runtime"])
    source_release = base._exact_keys(value["source"],
                                      {"release", "dataset_root", "canonical_input",
                                       "canonical_files"}, "source")
    expected_source = _source_context(source_release["release"]["path"],
                                      source_release["release"]["sha256"])
    if source_release != expected_source["binding"]:
        raise ValueError("provisional source binding differs")
    dataset = Path(value["dataset_root"])
    if not dataset.is_absolute() or dataset.name != "dataset":
        raise ValueError("provisional dataset_root must be an explicit dataset directory")
    _nonoverlap(dataset, expected_source["dataset"])
    _plain_tree(dataset, "provisional dataset")
    queue = base._exact_keys(value["queue"],
                             {"jobs_sha256", "jobs", "canonical_files",
                              "split_receipt_sha256"}, "queue")
    jobs_path = dataset / "jobs.json"
    if pb.file_sha256(jobs_path) != base._sha(queue["jobs_sha256"], "queue.jobs_sha256"):
        raise ValueError("provisional jobs.json differs")
    entries = queue["jobs"]
    if not isinstance(entries, list) or len(entries) != 3:
        raise ValueError("provisional queue must contain exactly three native jobs")
    if pb.read_json(jobs_path) != [entry.get("job") for entry in entries]:
        raise ValueError("provisional jobs.json ordering or fields differ")
    allowed = {CANONICAL_INPUT, SPLIT_ROOT, "jobs.json", "jobs_state"} | {
        f"{STORE_PREFIX}{number:02d}" for number in (1, 2, 3)}
    unknown = {path.name for path in dataset.iterdir()} - allowed
    if unknown:
        raise ValueError(f"provisional dataset contains unknown top-level entries: {sorted(unknown)}")
    canonical = dataset / CANONICAL_INPUT
    base._validate_file_map(canonical, queue["canonical_files"], "queue.canonical_files")
    canonical_cfg = pb.load_profile_config(canonical / "config.yaml")
    canonical_rows = pb.validate_input(canonical, canonical_cfg)
    if not _profile_ok(canonical, canonical_cfg) or len(canonical_rows) != 60:
        raise ValueError("provisional canonical input is not the exact 60-row profile")
    expected_rows = _transformed_rows(expected_source, canonical_cfg)
    if canonical_rows != expected_rows:
        raise ValueError("provisional canonical rows differ from the ordered source transformation")
    for old, row in zip(expected_source["rows"], canonical_rows):
        source_pattern = expected_source["canonical"] / old["pattern_file"]
        target_pattern = canonical / row["pattern_file"]
        if (pb.file_sha256(source_pattern) != pb.file_sha256(target_pattern) or
                row["pattern_sha256"] != old["pattern_sha256"]):
            raise ValueError("provisional canonical pattern bytes differ from their source")
    split_receipt = dataset / SPLIT_ROOT / "split_receipt.json"
    if pb.file_sha256(split_receipt) != base._sha(
            queue["split_receipt_sha256"], "queue.split_receipt_sha256"):
        raise ValueError("provisional split receipt differs")
    validated = [_validate_entry(dataset, entry, shard, machine)
                 for shard, (entry, machine) in enumerate(
                     zip(entries, base.EXPECTED_WORKERS), start=1)]
    split = pb.read_json(split_receipt)
    if (split.get("canonical_manifest_sha256") != pb.file_sha256(canonical / "manifest.json") or
            split.get("shard_size") != 20 or len(split.get("shards", [])) != 3 or
            sum(len(item.get("ids", [])) for item in split["shards"]) != 60):
        raise ValueError("provisional split is not the exact 3x20 canonical partition")
    for offset, (entry, receipt) in enumerate(zip(validated, split["shards"])):
        folder = dataset / Path(entry["job"]["input"])
        shard_rows = pb.read_json(folder / "manifest.json")
        expected_shard = canonical_rows[offset::3]
        if (shard_rows != expected_shard or
                receipt.get("ids") != [row["id"] for row in expected_shard] or
                receipt.get("manifest_sha256") != pb.file_sha256(folder / "manifest.json")):
            raise ValueError("provisional shard is not its exact ordered round-robin partition")
        for marker in ("config.yaml", "measurement.json", "score_spec.json"):
            if pb.file_sha256(folder / marker) != pb.file_sha256(canonical / marker):
                raise ValueError(f"provisional shard changed canonical {marker}")
        for row in expected_shard:
            if pb.file_sha256(folder / row["pattern_file"]) != \
                    pb.file_sha256(canonical / row["pattern_file"]):
                raise ValueError("provisional shard pattern differs from canonical bytes")
    stores = {entry["job"]["store"] for entry in validated}
    state = dataset / "jobs_state"
    if state.exists():
        for marker in state.iterdir():
            if marker.is_file() and marker.suffix in {".claim", ".done", ".fail"}:
                if marker.stem not in stores:
                    raise ValueError(f"provisional jobs_state has unknown marker: {marker.name}")
    states = [formal._job_state(dataset, entry, (entry["job"]["machine"],))
              for entry in validated]
    return {"release": value, "release_sha256": digest, "dataset": dataset,
            "entries": validated, "states": states}


def _status(states: Sequence[str]) -> str:
    if states and all(state == "done" for state in states):
        return "all_success"
    if states and all(state in {"done", "skipped", "failed"} for state in states):
        return "terminal_failure" if "failed" in states else "terminal_skips"
    return "active"


def validate_release(path: str | Path, required_sha256: str) -> dict[str, Any]:
    context = _context(path, required_sha256)
    return {"schema_version": 1, "status": "r81_provisional_s21_entry_passed",
            "release_id": context["release"]["release_id"], "stage": STAGE,
            "release_sha256": context["release_sha256"],
            "dataset_root": str(context["dataset"]), "scope": SCOPE,
            "job_count": 3, "row_count": 60,
            "workers": list(base.EXPECTED_WORKERS),
            "queue_status": _status(context["states"]),
            "policy": copy.deepcopy(POLICY), "hfss_started": False,
            "queue_or_source_mutation": False}


def queue_status(path: str | Path, required_sha256: str) -> str:
    return _status(_context(path, required_sha256)["states"])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--release-sha256")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--queue-status", action="store_true")
    parser.add_argument("--prepare-local", action="store_true")
    parser.add_argument("--source-release", type=Path)
    parser.add_argument("--source-release-sha256")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--priority", type=int, default=LOW_TIER_PRIORITY)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.prepare_local:
        if not args.source_release or not args.source_release_sha256 or not args.output_root:
            raise ValueError("local preparation requires source release, hash, and output root")
        result = prepare_local(args.source_release, args.source_release_sha256,
                               args.output_root, priority=args.priority)
    else:
        if not args.release or not args.release_sha256:
            raise ValueError("release validation requires release and release hash")
        if args.queue_status:
            status = queue_status(args.release, args.release_sha256)
            print(status)
            return {"all_success": 0, "terminal_skips": 0,
                    "active": 3, "terminal_failure": 4}[status]
        result = validate_release(args.release, args.release_sha256)
    if args.report:
        base._write_report(args.report, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
