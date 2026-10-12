"""Validate one bounded same-host R81 Discrete sweep diagnostic.

This additive entry authenticates the already completed Fast6 and Discrete2
evidence from the original EngineeringOnly release while allowing its mesh job
to remain pending.  It authorizes exactly one two-pattern Discrete repeat on
the original Fast owner's host.  It never promotes a formal release.
"""
from __future__ import annotations

import argparse
import copy
import json
import stat
from pathlib import Path
from typing import Any, Mapping, Sequence

from antenna.measurement import measurement_id, score_spec_id
from script import profiled_batch as pb
from script import r81_formal_sharded_entry as formal
from script import r81_worker_entry as base


STAGE = "SweepProbe"
SCOPE = base.SCOPE
SOURCE_RUNTIME_COMMIT = "c6e19c18ca4636e909253d1f2a7277089cdffdaa"
INPUT_NAME = "dedust_r81samehostdiscrete_input"
STORE_NAME = "dedust_r81samehostdiscrete"
ADDITIVE_FILES = {
    "script/profiled_shards.py",
    "script/r81_formal_sharded_entry.py",
    "script/r81_sweep_probe_entry.py",
    "script/start_r81_sweep_probe_worker.ps1",
}
RUNTIME_FILES = tuple(sorted(set(base.RUNTIME_FILES) | ADDITIVE_FILES))
SOURCE_ROLES = ("engineering_smoke", "engineering_discrete")


def _profile(cfg: Any) -> dict[str, Any]:
    return {"measurement_id": measurement_id(cfg.measurement),
            "score_spec_id": score_spec_id(cfg.score_spec),
            "runtime": cfg.runtime}


def _nonoverlap(left: Path, right: Path) -> None:
    left = left.resolve(strict=True)
    right = right.resolve(strict=True)
    for child, parent in ((left, right), (right, left)):
        try:
            child.relative_to(parent)
        except ValueError:
            pass
        else:
            raise ValueError("sweep probe and engineering source roots must not overlap")


def _source_entry(release: Mapping[str, Any], role: str) -> dict[str, Any]:
    matches = [entry for entry in release["queue"]["jobs"]
               if isinstance(entry, dict) and entry.get("role") == role]
    if len(matches) != 1:
        raise ValueError(f"engineering source must contain exactly one {role} job")
    return matches[0]


def _source_files(dataset: Path, entries: Sequence[dict[str, Any]],
                  canonical: Mapping[str, Any]) -> dict[str, str]:
    files = formal._source_file_closure(dataset, list(entries))
    name = base._name(canonical["name"], "source canonical input")
    for leaf in canonical["files"]:
        relative = (Path(name) / leaf).as_posix()
        files[relative] = pb.file_sha256(formal._source_file(
            dataset, relative, f"source canonical files[{relative}]"))
    return dict(sorted(files.items()))


def _known_source_markers(dataset: Path, release: Mapping[str, Any]) -> None:
    known = {entry["job"]["store"] for entry in release["queue"]["jobs"]}
    state = dataset / "jobs_state"
    if not state.exists():
        return
    for marker in state.iterdir():
        if (marker.is_file() and marker.suffix in {".claim", ".done", ".fail"}
                and marker.stem not in known):
            raise ValueError(f"engineering source has an unknown native marker: {marker.name}")


def _source_context(path: str | Path, required_sha256: str) -> dict[str, Any]:
    release_path = Path(path)
    if not release_path.is_absolute():
        raise ValueError("source_release.path must be absolute")
    expected_sha = base._sha(required_sha256, "source_release.sha256")
    if pb.file_sha256(release_path) != expected_sha:
        raise ValueError("engineering source release hash differs")
    base.validate_release(release_path, expected_sha, "EngineeringOnly")
    release = pb.read_json(release_path)
    dataset = Path(release["dataset_root"]).resolve(strict=True)
    canonical_release = release_path.resolve(strict=True)
    _known_source_markers(dataset, release)
    entries = [_source_entry(release, role) for role in SOURCE_ROLES]
    canonical = release["queue"]["formal_input"]
    before = _source_files(dataset, entries, canonical)
    roster = base._retry_workers(release["queue"]["expected_retry_workers"])

    profiles: dict[str, dict[str, Any]] = {}
    rows_by_role: dict[str, list[dict[str, Any]]] = {}
    for role, entry in zip(SOURCE_ROLES, entries):
        label = f"engineering source {role}"
        base._exact_keys(entry, {"role", "job", "input_files"}, label)
        current, rows = base._validate_input(dataset, entry, label)
        expected_store, expected_count = base.ENGINEERING[role]
        if (entry["job"]["store"] != expected_store or
                entry["job"]["input"] != expected_store + "_input" or
                len(rows) != expected_count):
            raise ValueError(f"{label} differs from the native engineering job")
        if formal._job_state(dataset, entry, roster) != "done":
            raise ValueError(f"{label} must be complete with zero errors")
        profiles[role], rows_by_role[role] = current, rows

    smoke_entry, discrete_entry = entries
    smoke_input = dataset / smoke_entry["job"]["input"]
    discrete_input = dataset / discrete_entry["job"]["input"]
    smoke_cfg = pb.load_profile_config(smoke_input / "config.yaml")
    discrete_cfg = pb.load_profile_config(discrete_input / "config.yaml")
    expected_measurement = copy.deepcopy(smoke_cfg.measurement)
    expected_measurement["name"] += "_discrete"
    expected_measurement["sweep"]["type"] = "Discrete"
    if (profiles["engineering_discrete"] != _profile(discrete_cfg) or
            discrete_cfg.scope != smoke_cfg.scope or discrete_cfg.port != smoke_cfg.port or
            discrete_cfg.runtime != smoke_cfg.runtime or
            discrete_cfg.score_spec != smoke_cfg.score_spec or
            discrete_cfg.measurement != expected_measurement):
        raise ValueError("source Discrete profile is not the exact Fast profile with Discrete sweep")

    smoke_rows = {}
    for row in rows_by_role["engineering_smoke"]:
        parent = row.get("parent_batch_id", row.get("parent"))
        if not isinstance(parent, str) or parent in smoke_rows:
            raise ValueError("source Fast rows need distinct parent pointers")
        smoke_rows[parent] = row
    parents, patterns = [], []
    for row in rows_by_role["engineering_discrete"]:
        parent = row.get("parent_batch_id", row.get("parent"))
        reference = smoke_rows.get(parent)
        if (not isinstance(parent, str) or reference is None or parent in parents or
                row.get("pattern_sha256") != reference.get("pattern_sha256") or
                pb.file_sha256(discrete_input / row["pattern_file"]) !=
                pb.file_sha256(smoke_input / reference["pattern_file"])):
            raise ValueError("source Discrete rows do not match two exact Fast parent patterns")
        parents.append(parent)
        patterns.append({"parent_id": parent,
                         "pattern_sha256": row["pattern_sha256"],
                         "source_file": row["pattern_file"],
                         "source_file_sha256": pb.file_sha256(
                             discrete_input / row["pattern_file"])})
    if len(parents) != 2:
        raise ValueError("source Discrete evidence must contain exactly two parents")
    canonical_dir = dataset / canonical["name"]
    canonical_rows = pb.read_json(canonical_dir / "manifest.json")
    history_pair = canonical_rows[:2]
    if (len(canonical_rows) != 60 or
            any(row.get("selection_arm") != "history" for row in history_pair) or
            parents != [row["id"] for row in history_pair]):
        raise ValueError("source Discrete parents must be the first two canonical history rows")
    for expected, row in zip(patterns, history_pair):
        if (expected["pattern_sha256"] != row["pattern_sha256"] or
                expected["source_file_sha256"] !=
                pb.file_sha256(canonical_dir / row["pattern_file"])):
            raise ValueError("source Discrete patterns differ from canonical history geometry")
    done = pb.read_json(dataset / "jobs_state" / f"{smoke_entry['job']['store']}.done")
    owner = done.get("machine")
    if not isinstance(owner, str) or owner not in roster:
        raise ValueError("source Fast done marker lacks its authorized owner")
    after = _source_files(dataset, entries, canonical)
    if (after != before or pb.file_sha256(canonical_release) != expected_sha):
        raise ValueError("engineering source changed during sweep-probe authentication")
    binding = {
        "source_release": {"path": str(canonical_release), "sha256": expected_sha},
        "dataset_root": str(dataset),
        "files": before,
        "owner": owner,
        "parent_ids": parents,
    }
    return {"binding": binding, "release": release, "dataset": dataset,
            "discrete_entry": discrete_entry, "discrete_profile": profiles["engineering_discrete"],
            "discrete_config": discrete_cfg, "patterns": patterns}


def collect_source_binding(path: str | Path, required_sha256: str) -> dict[str, Any]:
    """Return the stable Fast6+Discrete2 evidence closure and pinned owner."""
    return _source_context(path, required_sha256)["binding"]


def _validate_runtime(runtime: Any) -> None:
    value = base._exact_keys(runtime, {"git_commit", "files"}, "runtime")
    commit = value["git_commit"]
    if not isinstance(commit, str) or base.HEX40.fullmatch(commit) is None:
        raise ValueError("runtime.git_commit must be a full lowercase commit id")
    if not base._git_is_ancestor(commit):
        raise ValueError("runtime Git commit is not an ancestor of this checkout")
    files = value["files"]
    if (len(base.RUNTIME_FILES) != 44 or len(RUNTIME_FILES) != 48 or
            not isinstance(files, dict) or set(files) != set(RUNTIME_FILES)):
        raise ValueError("runtime.files must bind exact base44 plus four additive files")
    for name in base.RUNTIME_FILES:
        if base._git_blob_sha(commit, name) != base._git_blob_sha(SOURCE_RUNTIME_COMMIT, name):
            raise ValueError(f"SweepProbe changed a protected base44 blob: {name}")
    for name in RUNTIME_FILES:
        expected = base._sha(files[name], f"runtime.files[{name}]")
        if base._git_blob_sha(commit, name) != expected or not base._git_path_clean(commit, name):
            raise ValueError(f"runtime source changed: {name}")


def _probe_entry(dataset: Path, entry: Any, source: Mapping[str, Any]) -> dict[str, Any]:
    value = base._exact_keys(entry, {"role", "job", "input_files"}, "queue.jobs[0]")
    if value["role"] != "same_host_discrete":
        raise ValueError("SweepProbe queue role must be same_host_discrete")
    job = base._exact_keys(value["job"],
                           {"input", "store", "prio", "config", "scope", "machine"},
                           "queue.jobs[0].job")
    if (job["input"] != INPUT_NAME or job["store"] != STORE_NAME or
            job["config"] != f"{INPUT_NAME}/config.yaml" or job["scope"] != SCOPE or
            job["machine"] != source["binding"]["owner"] or
            isinstance(job["prio"], bool) or not isinstance(job["prio"], int)):
        raise ValueError("SweepProbe job identity, profile, or owner pin differs")
    native_entry = {"role": value["role"], "input_files": value["input_files"],
                    "job": {key: job[key]
                            for key in ("input", "store", "prio", "config", "scope")}}
    profile, rows = base._validate_input(dataset, native_entry, "queue.jobs[0]")
    if profile != source["discrete_profile"] or len(rows) != 2:
        raise ValueError("SweepProbe input must contain exactly two source-Discrete-profile rows")
    input_dir = dataset / INPUT_NAME
    source_input = source["dataset"] / source["discrete_entry"]["job"]["input"]
    for name in ("config.yaml", "measurement.json", "score_spec.json"):
        if pb.file_sha256(input_dir / name) != pb.file_sha256(source_input / name):
            raise ValueError(f"SweepProbe changed source Discrete {name}")
    if [row.get("parent_batch_id", row.get("parent")) for row in rows] != \
            source["binding"]["parent_ids"]:
        raise ValueError("SweepProbe parent order differs from source Discrete evidence")
    for row, expected in zip(rows, source["patterns"]):
        if (row.get("kind") != "repeat" or row.get("repeat") is not True or
                row.get("repeat_reason") != "repeatability" or
                row.get("selection_arm") != "repeat" or
                row.get("pattern_sha256") != expected["pattern_sha256"] or
                pb.file_sha256(input_dir / row["pattern_file"]) !=
                expected["source_file_sha256"]):
            raise ValueError("SweepProbe row is not an exact same-pattern repeat")
    return value


def _validate_dataset_layout(dataset: Path) -> None:
    root = dataset.resolve(strict=True)
    _plain_dataset_member(dataset, root)
    for path in dataset.rglob("*"):
        _plain_dataset_member(path, root)
    allowed = {INPUT_NAME, "jobs.json", "jobs_state", STORE_NAME}
    unknown = {path.name for path in dataset.iterdir()} - allowed
    if unknown:
        raise ValueError(f"SweepProbe dataset contains unknown entries: {sorted(unknown)}")


def _plain_dataset_member(path: Path, root: Path) -> None:
    metadata = path.lstat()
    if (stat.S_ISLNK(metadata.st_mode) or
            getattr(metadata, "st_file_attributes", 0) &
            getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)):
        raise ValueError(f"SweepProbe dataset contains a redirected path: {path.name}")
    try:
        path.resolve(strict=True).relative_to(root)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        raise ValueError(f"SweepProbe dataset member escapes its root: {path.name}") from exc


def _job_state(dataset: Path, entry: dict[str, Any], owner: str) -> str:
    state_root = dataset / "jobs_state"
    if state_root.exists():
        for marker in state_root.iterdir():
            if (marker.is_file() and marker.suffix in {".claim", ".done", ".fail"}
                    and marker.stem != STORE_NAME):
                raise ValueError(f"SweepProbe jobs_state contains unknown marker: {marker.name}")
    return formal._job_state(dataset, entry, (owner,))


def _context(path: str | Path, required_sha256: str) -> dict[str, Any]:
    release_path = Path(path)
    expected_sha = base._sha(required_sha256, "release sha256")
    if pb.file_sha256(release_path) != expected_sha:
        raise ValueError("SweepProbe release hash differs")
    release = pb.read_json(release_path)
    value = base._exact_keys(
        release, {"schema_version", "release_id", "stage", "dataset_root", "runtime",
                  "source_release", "source_files", "queue"}, "release")
    if value["schema_version"] != 1 or value["stage"] != STAGE:
        raise ValueError("unsupported SweepProbe release schema or stage")
    base._name(value["release_id"], "release.release_id")
    dataset = Path(value["dataset_root"])
    if not dataset.is_absolute() or dataset.name != "dataset" or dataset.parent.name != "sweep_probe_v001":
        raise ValueError("SweepProbe dataset must be the explicit sweep_probe_v001/dataset")
    _validate_runtime(value["runtime"])
    source_release = base._exact_keys(value["source_release"], {"path", "sha256"},
                                      "source_release")
    source = _source_context(source_release["path"], source_release["sha256"])
    if source_release != source["binding"]["source_release"] or \
            value["source_files"] != source["binding"]["files"]:
        raise ValueError("SweepProbe source binding differs from current completed evidence")
    _nonoverlap(dataset, source["dataset"])
    if dataset.resolve(strict=True).parent.parent != source["dataset"].parent.parent:
        raise ValueError("SweepProbe dataset must be a sibling of the engineering root")
    _validate_dataset_layout(dataset)
    queue = base._exact_keys(value["queue"], {"jobs_sha256", "jobs"}, "queue")
    jobs_path = dataset / "jobs.json"
    if pb.file_sha256(jobs_path) != base._sha(queue["jobs_sha256"], "queue.jobs_sha256"):
        raise ValueError("SweepProbe jobs.json differs from release")
    entries = queue["jobs"]
    if not isinstance(entries, list) or len(entries) != 1:
        raise ValueError("SweepProbe queue must contain exactly one two-row job")
    if pb.read_json(jobs_path) != [entries[0].get("job")]:
        raise ValueError("SweepProbe jobs.json contains changed or reordered jobs")
    entry = _probe_entry(dataset, entries[0], source)
    state = _job_state(dataset, entry, source["binding"]["owner"])
    return {"release": value, "release_path": release_path.resolve(),
            "release_sha256": expected_sha, "dataset": dataset, "source": source,
            "entry": entry, "job_state": state}


def validate_release(path: str | Path, required_sha256: str) -> dict[str, Any]:
    context = _context(path, required_sha256)
    binding = context["source"]["binding"]
    return {"schema_version": 1, "status": "r81_sweep_probe_entry_passed",
            "release_id": context["release"]["release_id"], "stage": STAGE,
            "release_sha256": context["release_sha256"],
            "dataset_root": str(context["dataset"]), "scope": SCOPE,
            "job_count": 1, "row_count": 2, "owner": binding["owner"],
            "parent_ids": binding["parent_ids"], "queue_status": _status(context["job_state"]),
            "formal_release_allowed": False, "engineering_gate_assessed": False,
            "hfss_started": False, "queue_or_source_mutation": False}


def _status(job_state: str) -> str:
    return {"active": "active", "done": "all_success", "skipped": "terminal_skips",
            "failed": "terminal_failure"}[job_state]


def queue_status(path: str | Path, required_sha256: str) -> str:
    return _status(_context(path, required_sha256)["job_state"])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--queue-status", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.queue_status:
        status = queue_status(args.release, args.release_sha256)
        print(status)
        return {"all_success": 0, "terminal_skips": 0,
                "terminal_failure": 4, "active": 3}[status]
    report = validate_release(args.release, args.release_sha256)
    if args.report:
        base._write_report(args.report, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
