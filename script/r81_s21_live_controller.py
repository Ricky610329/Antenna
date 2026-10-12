"""Automatic cumulative S21 survey controller and create-only publisher.

This additive controller turns the first immutable ``ProvisionalS21Discrete``
release into a bounded, three-batch live survey.  It does not run HFSS itself.
Workers discover releases through an authenticated create-only registry.

The controller takes a physical cutoff only when at least twenty previously
unconsumed successful wide-band observations are visible.  It then calls the
existing :mod:`script.filter_s21_training` implementation, ranks a fresh pool
with the existing survey-cycle helpers, writes 20 worst-margin, 20 uncertainty,
and 20 random rows, validates the complete successor release, and appends the
registry record as the final publication step.

All mutable work and all successor evidence live below the explicit shared
``actor_root``.  No repository-local path is published.  Batch 3 additionally
requires twenty new successful rows from batch 2, so it cannot be generated
solely from late observations of the original batch.
"""
from __future__ import annotations

import argparse
import copy
import contextlib
import json
import os
import shutil
import socket
import stat
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from script import exploration
from script import filter_s21_training as s21
from script import profiled_batch as pb
from script import profiled_shards
from script import r81_provisional_filter_dispatch as dispatch
from script import r81_s21_survey_cycle as cycle
from script import r81_worker_entry as base


SCHEMA_VERSION = 1
STAGE = "S21SurveyLoop"
SUCCESSOR_STAGE = "S21SurveySuccessor"
ACTOR_ROOT_NAME = "survey_loop_v003"
BOOTSTRAP_NAME = "release/r81_s21_live_loop_v003.json"
REGISTRY_DIR = "registry"
CONTROLLER_DIR = "controller"
BATCHES_DIR = "batches"
FAILURES_DIR = "failures"
LEASE_NAME = "creator.lock"
PENDING_NAME = "pending_publication.json"
POLL_SECONDS = 120

BATCH_SIZE = cycle.BATCH_SIZE
MAX_BATCHES = cycle.MAX_BATCHES
SUCCESS_TRIGGER = cycle.TRIGGER_NEW_SUCCESSES
ARM_COUNTS = copy.deepcopy(cycle.ARM_COUNTS)
CHECKPOINT_REPLACE_DELAYS = (.1, .2, .4, .8, 1.6)
CHECKPOINT_TRANSIENT_WINERRORS = frozenset({5, 32, 33})


class _ScopedProfiledTempfile:
    """Delegate tempfile except for safe Windows-UNC directories in actor_root."""

    def __init__(self, delegate: Any, actor_root: Path):
        self._delegate = delegate
        self._actor_root = Path(actor_root).resolve(strict=True)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def mkdtemp(self, suffix: str | None = None, prefix: str | None = None,
                dir: str | os.PathLike[str] | None = None) -> str:
        if not self._eligible(dir, prefix, suffix):
            return self._delegate.mkdtemp(suffix=suffix, prefix=prefix, dir=dir)
        parent = Path(dir).resolve(strict=True)
        stem, ending = "tmp" if prefix is None else prefix, "" if suffix is None else suffix
        for _attempt in range(100):
            candidate = parent / f"{stem}{uuid.uuid4().hex}{ending}"
            try:
                # Deliberately omit mode.  On the affected Windows UNC share,
                # mkdir(mode=0o700) is rejected while the default directory
                # creation correctly inherits the parent NAS ACL.
                candidate.mkdir()
            except FileExistsError:
                continue
            return os.fspath(candidate)
        raise FileExistsError("could not create an exclusive profiled temporary directory")

    def _eligible(self, directory: Any, prefix: Any, suffix: Any) -> bool:
        if os.name != "nt" or directory is None:
            return False
        if any(value is not None and not isinstance(value, str)
               for value in (prefix, suffix)):
            return False
        if any(separator and separator in value
               for value in (prefix, suffix) if value is not None
               for separator in (os.sep, os.altsep)):
            return False
        raw = Path(directory)
        if not raw.is_absolute() or not os.fspath(raw).startswith("\\\\"):
            return False
        try:
            resolved = raw.resolve(strict=True)
        except (FileNotFoundError, OSError, RuntimeError):
            return False
        if os.path.normcase(os.path.normpath(os.fspath(raw))) != \
                os.path.normcase(os.path.normpath(os.fspath(resolved))):
            return False
        try:
            resolved.relative_to(self._actor_root)
        except ValueError:
            return False
        return True


@contextlib.contextmanager
def _scoped_profiled_tempfile(actor_root: Path):
    """Temporarily replace only profiled_shards' tempfile module reference."""
    original = profiled_shards.tempfile
    if isinstance(original, _ScopedProfiledTempfile):
        raise RuntimeError("profiled tempfile proxy is already active")
    proxy = _ScopedProfiledTempfile(original, actor_root)
    profiled_shards.tempfile = proxy
    try:
        yield proxy
    finally:
        profiled_shards.tempfile = original


class _ScopedExplorationOS:
    """Delegate os except transient checkpoint replacement in actor_root."""

    def __init__(self, delegate: Any, actor_root: Path, *, sleep=time.sleep):
        self._delegate = delegate
        self._actor_root = Path(actor_root).resolve(strict=True)
        self._sleep = sleep

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def replace(self, source: str | os.PathLike[str],
                destination: str | os.PathLike[str], *,
                src_dir_fd: int | None = None,
                dst_dir_fd: int | None = None) -> None:
        arguments = {}
        if src_dir_fd is not None:
            arguments["src_dir_fd"] = src_dir_fd
        if dst_dir_fd is not None:
            arguments["dst_dir_fd"] = dst_dir_fd
        if not self._eligible(source, destination, src_dir_fd, dst_dir_fd):
            return self._delegate.replace(source, destination, **arguments)

        first_error = None
        for attempt in range(len(CHECKPOINT_REPLACE_DELAYS) + 1):
            try:
                return self._delegate.replace(source, destination)
            except PermissionError as error:
                if getattr(error, "winerror", None) not in CHECKPOINT_TRANSIENT_WINERRORS:
                    raise
                if first_error is None:
                    first_error = error
                if attempt == len(CHECKPOINT_REPLACE_DELAYS):
                    raise first_error
                self._sleep(CHECKPOINT_REPLACE_DELAYS[attempt])
        raise AssertionError("unreachable checkpoint replacement retry state")

    def _eligible(self, source: Any, destination: Any,
                  src_dir_fd: Any, dst_dir_fd: Any) -> bool:
        if os.name != "nt" or src_dir_fd is not None or dst_dir_fd is not None:
            return False
        try:
            source_path, destination_path = Path(source), Path(destination)
        except TypeError:
            return False
        if (not source_path.is_absolute() or not destination_path.is_absolute()
                or not os.fspath(source_path).startswith("\\\\")
                or not os.fspath(destination_path).startswith("\\\\")
                or source_path.parent != destination_path.parent
                or source_path.name != destination_path.name + ".tmp"
                or destination_path.suffix != ".pt"):
            return False
        try:
            resolved_parent = source_path.parent.resolve(strict=True)
        except (FileNotFoundError, OSError, RuntimeError):
            return False
        literal_parent = os.path.normcase(os.fspath(source_path.parent))
        raw_parent = os.path.normcase(os.path.normpath(os.fspath(source_path.parent)))
        normalized_parent = os.path.normcase(os.path.normpath(os.fspath(resolved_parent)))
        if literal_parent != raw_parent or raw_parent != normalized_parent:
            return False
        try:
            resolved_parent.relative_to(self._actor_root)
        except ValueError:
            return False
        if (not source_path.is_file() or source_path.is_symlink()
                or destination_path.is_symlink()
                or (destination_path.exists() and not destination_path.is_file())):
            return False
        return True


@contextlib.contextmanager
def _scoped_exploration_os(actor_root: Path, *, sleep=time.sleep):
    """Temporarily replace only exploration's os module reference."""
    original = exploration.os
    if isinstance(original, _ScopedExplorationOS):
        raise RuntimeError("exploration os proxy is already active")
    proxy = _ScopedExplorationOS(original, actor_root, sleep=sleep)
    exploration.os = proxy
    try:
        yield proxy
    finally:
        exploration.os = original

def _live():
    # Delayed import keeps this module importable while the additive worker
    # entry is being reviewed independently, and avoids a circular import.
    from script import r81_s21_live_entry as live
    return live


def _sha(path: Path) -> str:
    return pb.file_sha256(Path(path))


def _json_id(value: Any) -> str:
    import hashlib
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _is_redirected(path: Path) -> bool:
    metadata = path.lstat()
    return bool(stat.S_ISLNK(metadata.st_mode) or
                getattr(metadata, "st_file_attributes", 0) &
                getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _plain_root(path: Path, *, create: bool = False) -> Path:
    raw = Path(path)
    if not raw.is_absolute():
        raise ValueError("actor_root must be an explicit absolute path")
    if raw.name != ACTOR_ROOT_NAME:
        raise ValueError(f"actor_root must end in {ACTOR_ROOT_NAME}")
    if create:
        parent = raw.parent.resolve(strict=True)
        if _is_redirected(raw.parent):
            raise ValueError("actor_root parent is redirected")
        value = parent / raw.name
    else:
        value = raw.resolve(strict=True)
    if value.exists() and _is_redirected(value):
        raise ValueError("actor_root is redirected")
    return value


def _inside(root: Path, value: Path, label: str, *, must_exist: bool = True) -> Path:
    raw = Path(value)
    if not raw.is_absolute():
        raise ValueError(f"{label} must be absolute")
    resolved = raw.resolve(strict=must_exist)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise ValueError(f"{label} escapes actor_root") from exc
    probe = resolved if must_exist else resolved.parent
    while True:
        if probe.exists() and _is_redirected(probe):
            raise ValueError(f"{label} traverses a redirected path")
        if probe == root:
            break
        if root not in probe.parents:
            raise ValueError(f"{label} escapes actor_root")
        probe = probe.parent
    return resolved


def _write_create_only(path: Path, value: Mapping[str, Any]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return _sha(path)


def _publish_json_atomic_noreplace(path: Path, value: Mapping[str, Any],
                                   draft_dir: Path) -> str:
    """Atomically advertise JSON without exposing partial bytes or replacing a prefix.

    The draft lives outside ``registry`` so polling workers never reject or read
    it.  A same-volume hard link creates the final name atomically and fails if
    that name already exists on both NTFS and POSIX filesystems.
    """
    path, draft_dir = Path(path), Path(draft_dir)
    if path.exists():
        raise FileExistsError(f"create-only publication already exists: {path}")
    draft = draft_dir / f".registry-draft-{path.stem}-{uuid.uuid4().hex}.json"
    _write_create_only(draft, value)
    try:
        os.link(draft, path)
    except Exception:
        if path.exists() and pb.read_json(path) == value:
            # Another owner may have completed the exact pending transaction;
            # callers still validate its hash before accepting it.
            return _sha(path)
        raise
    finally:
        if draft.exists():
            draft.unlink()
    return _sha(path)


def _runtime_files() -> tuple[str, ...]:
    live = _live()
    expected = tuple(sorted(set(dispatch.RUNTIME_FILES) | {
        "script/filter_s21_training.py",
        "script/r81_s21_survey_cycle.py",
        "script/r81_s21_live_controller.py",
        "script/r81_s21_live_entry.py",
        "script/start_r81_s21_live_worker.ps1",
    }))
    advertised = getattr(live, "RUNTIME_FILES", expected)
    if tuple(advertised) != expected:
        raise ValueError("live worker runtime closure differs from controller closure")
    return expected


def _runtime_binding(commit: str) -> dict[str, Any]:
    if not isinstance(commit, str) or base.HEX40.fullmatch(commit) is None:
        raise ValueError("producer_commit must be a full lowercase commit id")
    if not base._git_is_ancestor(commit):
        raise ValueError("producer_commit is not available in this checkout")
    files = {}
    for relative in _runtime_files():
        path = base.REPO / relative
        if not path.is_file() or not base._git_path_clean(commit, relative):
            raise ValueError(f"runtime file is missing or differs from producer commit: {relative}")
        digest = base._git_blob_sha(commit, relative)
        files[relative] = digest
    return {"git_commit": commit, "files": files}


def _profile_binding(context: Mapping[str, Any]) -> dict[str, str]:
    return cycle._profile_binding(context)


def _bootstrap_document(actor_root: Path, initial_release: Path,
                        initial_sha256: str, producer_commit: str) -> dict[str, Any]:
    initial = dispatch._context(initial_release, initial_sha256)
    profile = _profile_binding(initial)
    return {
        "schema_version": SCHEMA_VERSION,
        "release_id": ACTOR_ROOT_NAME,
        "stage": STAGE,
        "actor_root": str(actor_root),
        "shared_root": str(actor_root),
        "runtime": _runtime_binding(producer_commit),
        "initial_survey": {
            "release": {"path": str(Path(initial_release).resolve(strict=True)),
                        "sha256": initial["release_sha256"]},
            "dataset_root": str(Path(initial["dataset"]).resolve(strict=True)),
        },
        "registry": {"directory": str(actor_root / REGISTRY_DIR)},
        "budget": copy.deepcopy(_live().BUDGET),
        "protocol": {"scope": dispatch.SCOPE,
                     "workers": list(base.EXPECTED_WORKERS),
                     "priority": dispatch.LOW_TIER_PRIORITY,
                     "config_files": profile},
    }


def prepare_local(*, actor_root: Path, initial_release: Path,
                  initial_release_sha256: str, producer_commit: str,
                  cache_source: Path | None = None) -> dict[str, Any]:
    """Create the fixed bootstrap and batch-001 registry record.

    Despite the historical command name, this may target the explicit shared
    personal root.  It is create-only and never launches a worker or HFSS.
    """
    root = _plain_root(actor_root, create=True)
    if root.exists():
        raise FileExistsError("live controller preparation requires a fresh actor_root")
    root.mkdir()
    for name in (REGISTRY_DIR, CONTROLLER_DIR, BATCHES_DIR, FAILURES_DIR, "release"):
        (root / name).mkdir()
    cache_receipt = _initialize_model_cache(root, cache_source)
    bootstrap = _bootstrap_document(root, initial_release, initial_release_sha256,
                                    producer_commit)
    bootstrap_path = root / BOOTSTRAP_NAME
    bootstrap_sha = _write_create_only(bootstrap_path, bootstrap)
    first = {
        "schema_version": SCHEMA_VERSION,
        "sequence": 1,
        "previous": None,
        "bootstrap": {"path": str(bootstrap_path), "sha256": bootstrap_sha},
        "release": copy.deepcopy(bootstrap["initial_survey"]["release"]) |
                   {"dataset_root": bootstrap["initial_survey"]["dataset_root"]},
    }
    first_path = root / REGISTRY_DIR / "batch-001.json"
    first_sha = _publish_json_atomic_noreplace(first_path, first, root / CONTROLLER_DIR)
    context = _live().validate_bootstrap(bootstrap_path, bootstrap_sha)
    _live().validate_registry_record(first_path, first_sha, context)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "live_controller_prepared",
        "actor_root": str(root),
        "bootstrap": {"path": str(bootstrap_path), "sha256": bootstrap_sha},
        "registry_head": {"sequence": 1, "path": str(first_path),
                          "sha256": first_sha},
        "model_cache": cache_receipt,
        "budget": copy.deepcopy(bootstrap["budget"]),
        "worker_started": False, "hfss_started": False,
    }


def _initialize_model_cache(root: Path, source: Path | None) -> dict[str, Any]:
    target = root / CONTROLLER_DIR / "models"
    target.mkdir()
    if source is None:
        value = {"schema_version": 1, "status": "empty", "files": {}}
    else:
        source_root = Path(source).resolve(strict=True)
        if not source_root.is_dir() or _is_redirected(source_root):
            raise ValueError("model cache source must be a plain existing directory")
        split = source_root / s21.SPLIT_FILE
        prior = source_root / "prior"
        if not split.is_file() or not prior.is_dir():
            raise ValueError("model cache source lacks the completed split/prior closure")
        for item in (source_root, prior, *prior.rglob("*")):
            if _is_redirected(item):
                raise ValueError("model cache source contains a redirected path")
        shutil.copy2(split, target / split.name)
        shutil.copytree(prior, target / "prior")
        files = {path.relative_to(target).as_posix(): _sha(path)
                 for path in sorted(target.rglob("*")) if path.is_file()}
        value = {"schema_version": 1, "status": "copied_completed_prior",
                 "source_leaf": source_root.name, "files": files,
                 "files_id": _json_id(files)}
    receipt = root / CONTROLLER_DIR / "model_cache_initialization.json"
    _write_create_only(receipt, value)
    return {"directory": str(target), "receipt": str(receipt),
            "receipt_sha256": _sha(receipt), "status": value["status"]}


def _bootstrap_context(path: Path, sha256: str) -> tuple[dict[str, Any], dict[str, Any], Path]:
    bootstrap_path = Path(path).resolve(strict=True)
    context = _live().validate_bootstrap(bootstrap_path, sha256)
    document = pb.read_json(bootstrap_path)
    root = _plain_root(Path(document["actor_root"]))
    if bootstrap_path != root / BOOTSTRAP_NAME:
        raise ValueError("bootstrap is not the fixed actor_root bootstrap")
    return context, document, root


def _registry_prefix(bootstrap_context: Mapping[str, Any], bootstrap: Mapping[str, Any],
                     root: Path) -> list[dict[str, Any]]:
    directory = _inside(root, Path(bootstrap["registry"]["directory"]),
                        "registry.directory")
    paths = sorted(directory.glob("batch-*.json"))
    unknown = [path.name for path in directory.iterdir()
               if path.is_file() and path not in paths]
    if unknown:
        raise ValueError(f"live registry contains unknown files: {sorted(unknown)}")
    if not paths:
        raise ValueError("live registry has no batch-001 record")
    records = []
    previous = None
    for sequence, path in enumerate(paths, start=1):
        if path.name != f"batch-{sequence:03d}.json":
            raise ValueError("live registry is not a contiguous create-only prefix")
        digest = _sha(path)
        validated = _live().validate_registry_record(path, digest, bootstrap_context)
        value = pb.read_json(path)
        expected_previous = None if previous is None else {
            "sequence": sequence - 1, "path": str(previous["path"]),
            "sha256": previous["sha256"]}
        if value.get("sequence") != sequence or value.get("previous") != expected_previous:
            raise ValueError("registry SHA chain or sequence changed")
        item = {"sequence": sequence, "path": path.resolve(), "sha256": digest,
                "record": value, "validated": validated}
        records.append(item); previous = item
    if len(records) > MAX_BATCHES:
        raise ValueError("live registry exceeds the fixed three-batch budget")
    return records


def _stop_status(bootstrap_context: Mapping[str, Any],
                 prefix: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return _live().stop_requested({
        "bootstrap_context": bootstrap_context,
        "records": [item["validated"] for item in prefix],
    })


def _release_context(item: Mapping[str, Any]) -> dict[str, Any]:
    ref = item["record"]["release"]
    release_path = Path(ref["path"])
    release = pb.read_json(release_path)
    if item["sequence"] == 1:
        return dispatch._context(release_path, ref["sha256"])
    dataset = Path(release["dataset_root"]).resolve(strict=True)
    entries = release["queue"]["jobs"]
    return {"release": release, "release_sha256": ref["sha256"],
            "dataset": dataset, "entries": entries, "states": []}


def _aggregate_inventory(contexts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pairs: list[tuple[Path, Path]] = []
    queued: dict[str, dict[str, Any]] = {}
    successes: dict[str, dict[str, Any]] = {}
    sources = []
    for batch, context in enumerate(contexts, start=1):
        current = cycle._input_store_inventory(context)
        pairs.extend(current["pairs"])
        for digest, value in current["queued"].items():
            if digest in queued:
                raise ValueError("released live batches contain a duplicate physical pattern")
            queued[digest] = value
            queued[digest]["released_batch"] = batch
        for digest, value in current["successes"].items():
            if digest in successes:
                raise ValueError("live stores contain a duplicate successful physical pattern")
            successes[digest] = value
            successes[digest]["released_batch"] = batch
        sources.extend({**source, "released_batch": batch}
                       for source in current["sources"])
    return {"pairs": pairs, "queued": queued, "successes": successes,
            "sources": sources, "source_inventory_id": cycle._json_id(sources)}


def _selection_receipt(item: Mapping[str, Any]) -> dict[str, Any] | None:
    if item["sequence"] == 1:
        return None
    release = pb.read_json(Path(item["record"]["release"]["path"]))
    ref = release["cycle"]["selection_receipt"]
    path = Path(ref["path"])
    if _sha(path) != ref["sha256"]:
        raise ValueError("published selection receipt changed")
    return pb.read_json(path)


def _consumed(prefix: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    receipt = _selection_receipt(prefix[-1])
    if receipt is None:
        return {}
    rows = receipt.get("trained_snapshot", {}).get("successes")
    if not isinstance(rows, list):
        raise ValueError("latest published selection lacks its frozen training cutoff")
    result = {}
    for row in rows:
        if (not isinstance(row, dict) or set(row) != {"pattern_sha256", "sample_sha256"} or
                not s21._is_sha(row["pattern_sha256"]) or
                not s21._is_sha(row["sample_sha256"]) or
                row["pattern_sha256"] in result):
            raise ValueError("published consumed-success bindings are malformed")
        result[row["pattern_sha256"]] = row["sample_sha256"]
    return result


def _cutoff(prefix: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    contexts = [_release_context(item) for item in prefix]
    inventory = _aggregate_inventory(contexts)
    consumed = _consumed(prefix)
    for digest, sample_sha in consumed.items():
        if (digest not in inventory["successes"] or
                inventory["successes"][digest]["sample_sha256"] != sample_sha):
            raise ValueError("a previously consumed physical success changed or disappeared")
    new = sorted(set(inventory["successes"]) - set(consumed))
    next_batch = len(prefix) + 1
    required_batch = None
    eligible = new
    if next_batch == 3:
        required_batch = 2
        eligible = [digest for digest in new
                    if inventory["successes"][digest]["released_batch"] == 2]
    status = "budget_exhausted" if next_batch > MAX_BATCHES else (
        "ready" if len(new) >= SUCCESS_TRIGGER and
        (required_batch is None or len(eligible) >= SUCCESS_TRIGGER) else "waiting")
    return {"status": status, "next_batch": None if next_batch > MAX_BATCHES else next_batch,
            "inventory": inventory, "contexts": contexts, "consumed": consumed,
            "new": new, "required_batch": required_batch,
            "required_batch_new": eligible}


def _family_preflight(cutoff: Mapping[str, Any], model_output_root: Path,
                      seed: int) -> dict[str, Any]:
    """Freeze/authenticate the cutoff and require usable family train/holdout rows."""
    inventory = cutoff["inventory"]
    snapshot = s21.freeze_successes(inventory["pairs"], model_output_root)
    protocol = s21.load_protocol()
    current = s21.load_current_snapshot(snapshot, protocol)
    anchor_dir = Path(cutoff["contexts"][0]["dataset"]) / dispatch.CANONICAL_INPUT
    _anchor_x, anchor_rows = s21.load_split_anchor(anchor_dir, protocol)
    train_rows, holdout_rows, _assignments = s21._split(
        anchor_rows, current.rows, model_output_root, .2, seed)
    return {"snapshot": str(snapshot), "sample_count": len(current.rows),
            "train_count": len(train_rows), "holdout_count": len(holdout_rows),
            "ready": bool(train_rows and holdout_rows)}


def _acquire_lease(root: Path) -> tuple[Any, str]:
    path = root / CONTROLLER_DIR / LEASE_NAME
    token = uuid.uuid4().hex
    stream = path.open("a+b")
    try:
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            if path.stat().st_size == 0:
                stream.write(b"0"); stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except Exception:
        stream.close()
        raise RuntimeError("another live-controller creator owns the shared lease")
    leases = root / CONTROLLER_DIR / "leases"
    leases.mkdir(exist_ok=True)
    _write_create_only(leases / f"{token}.json", {
        "schema_version": 1, "token": token, "host": socket.gethostname(),
        "created_unix": time.time(), "automatic_release_on_process_exit": True})
    return stream, token


def _release_lease(stream: Any, token: str) -> None:
    if stream.closed:
        return
    stream.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    stream.close()


def _jobs(dataset: Path, batch: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    live = _live()
    entries, jobs = [], []
    split_name = f"batch-{batch:03d}_split"
    for shard, machine in enumerate(base.EXPECTED_WORKERS, start=1):
        relative = f"{split_name}/shard-{shard:03d}_input"
        job = {"input": relative,
               "store": f"dedust_r81s21live_b{batch:03d}_{shard:02d}",
               "prio": dispatch.LOW_TIER_PRIORITY,
               "config": f"{relative}/config.yaml",
               "scope": dispatch.SCOPE, "machine": machine}
        folder = dataset / Path(relative)
        rows = pb.validate_input(folder, pb.load_profile_config(folder / "config.yaml"))
        if len(rows) != 20:
            raise ValueError("successor jobs must remain exact 20-row shards")
        entries.append({"role": live.SUCCESSOR_ROLE,
                        "shard": shard, "job": job,
                        "input_files": cycle._files(folder)})
        jobs.append(job)
    return entries, jobs


def _build_release(*, root: Path, bootstrap_path: Path, bootstrap_sha: str,
                   prefix: Sequence[Mapping[str, Any]], cutoff: Mapping[str, Any],
                   prior_manifest: Path, prior_root: Path, expected_prior_sha256: str,
                   seed_inputs: Sequence[Path], candidate_pool_size: int,
                   seed: int, pretrain_epochs: int, epochs: int,
                   batch_train_size: int, hidden_dims: Sequence[int],
                   model_output_root: Path) -> tuple[Path, str]:
    batch = int(cutoff["next_batch"])
    batch_root = root / BATCHES_DIR / f"batch-{batch:03d}"
    if batch_root.exists():
        failed = root / FAILURES_DIR / (
            f"batch-{batch:03d}-unpublished-{int(time.time())}-{uuid.uuid4().hex[:8]}")
        batch_root.rename(failed)
    batch_root.mkdir()
    try:
        inventory = cutoff["inventory"]
        training_root = _inside(root, model_output_root, "model_output_root")
        paths, summary = s21.train(
            input_store_pairs=inventory["pairs"],
            split_input=Path(cutoff["contexts"][0]["dataset"]) / dispatch.CANONICAL_INPUT,
            output_dir=training_root, prior_manifest=prior_manifest,
            prior_root=prior_root, expected_prior_sha256=expected_prior_sha256,
            min_successes=SUCCESS_TRIGGER, pretrain_epochs=pretrain_epochs,
            epochs=epochs, batch_size=batch_train_size,
            hidden_dims=hidden_dims, seed=seed)
        snapshot_successes, snapshot_receipt = cycle._trained_snapshot_successes(summary)
        if not set(inventory["successes"]) <= set(snapshot_successes):
            raise ValueError("cumulative snapshot omitted a success from the physical cutoff")
        model_dir = paths[0].parent
        completion_path = model_dir / "completion_receipt.json"
        completion_sha = _sha(completion_path)
        snapshot_path = _inside(root, Path(snapshot_receipt["snapshot"]),
                                "frozen_snapshot.path")
        snapshot_files = {path.relative_to(snapshot_path).as_posix(): _sha(path)
                          for path in sorted(snapshot_path.rglob("*")) if path.is_file()}
        excluded = set(inventory["queued"])
        optional, optional_bindings = cycle._optional_seed_rows(seed_inputs, excluded)
        seeds = [*cycle._seed_rows(inventory), *optional]
        pool = cycle._candidate_pool(seeds, excluded, candidate_pool_size,
                                     seed + batch * 1009,
                                     direct_candidates=optional)
        prediction, order, rank = s21.predict_and_rank(
            model_dir, pool, expected_completion_sha256=completion_sha)
        uncertainty = cycle._ensemble_uncertainty(
            model_dir, [row["pattern"] for row in pool], prediction, summary)
        selected = cycle._select(pool, prediction, order, uncertainty,
                                 seed=seed + batch * 2003)
        if set(row["pattern_sha256"] for row in selected) & excluded:
            raise ValueError("successor overlaps a measured, queued, or published pattern")

        pool_dir = batch_root / "candidate_pool"
        pool_receipt = cycle._write_pool(pool_dir, pool, prediction, uncertainty, rank)
        dataset = batch_root / "dataset"
        dataset.mkdir()
        profile = _profile_binding(cutoff["contexts"][0])
        bundle = cycle._write_next_bundle(
            dataset, selected, batch, summary["signature"], rank["candidate_set_id"],
            Path(cutoff["contexts"][0]["dataset"]) / dispatch.CANONICAL_INPUT,
            profile)
        (dataset / "jobs_template.json").unlink()
        entries, jobs = _jobs(dataset, batch)
        pb.atomic_json(dataset / "jobs.json", jobs)
        (dataset / "jobs_state").mkdir()

        # Re-read all released sources at the publication cutoff.  New results
        # may appear, but every sample used by the frozen model must remain.
        latest_prefix = _registry_prefix(
            _live().validate_bootstrap(bootstrap_path, bootstrap_sha),
            pb.read_json(bootstrap_path), root)
        if [(item["sha256"], item["sequence"]) for item in latest_prefix] != [
                (item["sha256"], item["sequence"]) for item in prefix]:
            raise ValueError("registry changed during successor construction")
        final_inventory = _aggregate_inventory([_release_context(item) for item in latest_prefix])
        for digest, sample_sha in snapshot_successes.items():
            if (digest not in final_inventory["successes"] or
                    final_inventory["successes"][digest]["sample_sha256"] != sample_sha):
                raise ValueError("a frozen cumulative sample changed before publication")

        selection_rows = [{key: row[key] for key in (
            "id", "pattern_sha256", "candidate_group", "selection_arm", "pool_index",
            "prediction", "predicted_worst_margin_db", "predicted_uncertainty")}
                          for row in selected]
        selection_path = batch_root / "selection_receipt.json"
        selection = {
            "schema_version": 1, "batch": batch,
            "trigger_new_success_pattern_sha256": cutoff["new"],
            "trigger": {"successes": SUCCESS_TRIGGER,
                        "new_pattern_sha256": cutoff["new"],
                        "required_batch": cutoff["required_batch"],
                        "required_batch_new_pattern_sha256": cutoff["required_batch_new"],
                        "source_inventory_id": inventory["source_inventory_id"]},
            "trigger_source_bindings": inventory["sources"],
            "trained_snapshot": {"receipt": snapshot_receipt,
                                 "successes": [{"pattern_sha256": digest,
                                                "sample_sha256": snapshot_successes[digest]}
                                               for digest in sorted(snapshot_successes)]},
            "post_source_inventory_id": final_inventory["source_inventory_id"],
            "post_source_bindings": final_inventory["sources"],
            "post_snapshot_new_success_pattern_sha256": sorted(
                set(final_inventory["successes"]) - set(snapshot_successes)),
            "optional_narrow_history_seeds": optional_bindings,
            "narrow_history_policy": "eligible geometry only; never treated as observed wide stop truth",
            "excluded_measured_queued_published_pattern_sha256": sorted(excluded),
            "model": {"signature": summary["signature"],
                      "completion_receipt_sha256": completion_sha,
                      "model_sha256": summary["model_sha256"]},
            "pool": pool_receipt, "ranking": rank,
            "uncertainty": "mean per-output population std across two bound ensemble members",
            "arms": copy.deepcopy(ARM_COUNTS), "selected": selection_rows,
            "selected_id": _json_id(selection_rows), "bundle": bundle,
            "claims": {"provisional_only": True, "formal_pass": False,
                       "survey_complete_is_spec_pass": False},
        }
        pb.atomic_json(selection_path, selection)

        previous = prefix[-1]
        canonical = dataset / f"batch-{batch:03d}_input"
        split_receipt = dataset / f"batch-{batch:03d}_split" / "split_receipt.json"
        release = {
            "schema_version": 1,
            "release_id": f"{ACTOR_ROOT_NAME}_batch_{batch:03d}",
            "stage": SUCCESSOR_STAGE,
            "sequence": batch,
            "dataset_root": str(dataset),
            "bootstrap": {"path": str(bootstrap_path), "sha256": bootstrap_sha},
            "previous_registry": {"sequence": previous["sequence"],
                                  "path": str(previous["path"]),
                                  "sha256": previous["sha256"]},
            "cycle": {
                "batch": batch,
                "selection_receipt": {"path": str(selection_path),
                                      "sha256": _sha(selection_path)},
                "candidate_pool_receipt": {"path": str(pool_dir / "pool_receipt.json"),
                                           "sha256": _sha(pool_dir / "pool_receipt.json")},
                "rank_receipt": {"path": str(pool_dir / "rank_receipt.json"),
                                 "sha256": _sha(pool_dir / "rank_receipt.json")},
                "model_completion_receipt": {"path": str(completion_path),
                                             "sha256": completion_sha},
                "frozen_snapshot": {"path": str(snapshot_path),
                                    "files": snapshot_files},
                "canonical_manifest_sha256": _sha(canonical / "manifest.json"),
                "split_receipt_sha256": _sha(split_receipt),
            },
            "queue": {"jobs_sha256": _sha(dataset / "jobs.json"),
                      "jobs": entries, "canonical_files": cycle._files(canonical),
                      "split_receipt_sha256": _sha(split_receipt)},
            "policy": copy.deepcopy(_live().POLICY),
        }
        release_path = batch_root / "release.json"
        release_sha = _write_create_only(release_path, release)
        return release_path, release_sha
    except Exception as exc:
        if batch_root.exists():
            failure_path = batch_root / "failure.json"
            if not failure_path.exists():
                _write_create_only(failure_path, {
                    "schema_version": 1, "error_type": type(exc).__name__,
                    "error": str(exc), "published": False,
                    "worker_started": False, "hfss_started": False})
            failed = root / FAILURES_DIR / (
                f"batch-{batch:03d}-failed-{int(time.time())}-{uuid.uuid4().hex[:8]}")
            batch_root.rename(failed)
        raise


def _record_for_release(*, sequence: int, bootstrap_path: Path, bootstrap_sha: str,
                        previous: Mapping[str, Any], release_path: Path,
                        release_sha: str) -> dict[str, Any]:
    release = pb.read_json(release_path)
    return {"schema_version": 1, "sequence": sequence,
            "previous": {"sequence": previous["sequence"],
                         "path": str(previous["path"]),
                         "sha256": previous["sha256"]},
            "bootstrap": {"path": str(bootstrap_path), "sha256": bootstrap_sha},
            "release": {"path": str(release_path), "sha256": release_sha,
                        "dataset_root": release["dataset_root"]}}


def _finish_pending(root: Path, bootstrap_context: Mapping[str, Any]) -> dict[str, Any] | None:
    pending_path = root / CONTROLLER_DIR / PENDING_NAME
    if not pending_path.is_file():
        return None
    pending = pb.read_json(pending_path)
    if pending.get("schema_version") != 1:
        raise ValueError("pending publication schema changed")
    record_path = _inside(root, Path(pending["registry_path"]),
                          "pending.registry_path", must_exist=False)
    release_path = _inside(root, Path(pending["release"]["path"]),
                           "pending.release.path")
    if _sha(release_path) != pending["release"]["sha256"]:
        raise ValueError("pending successor release changed")
    _live().validate_successor_release(release_path, pending["release"]["sha256"],
                                       bootstrap_context)
    record = pending["record"]
    if record_path.exists():
        if pb.read_json(record_path) != record:
            raise ValueError("existing registry record differs from pending publication")
        record_sha = _sha(record_path)
        _live().validate_registry_record(record_path, record_sha, bootstrap_context)
        snapshot = _live().registry_snapshot(
            bootstrap_context["bootstrap_path"], bootstrap_context["bootstrap_sha256"])
        stop = _live().stop_requested(snapshot)
        pending_path.unlink()
        return {"schema_version": 1,
                "status": ("stopped_after_publication" if stop["stopped"]
                           else "published_successor"),
                "batch": record["sequence"],
                "registry_record": {"path": str(record_path), "sha256": record_sha},
                "release": copy.deepcopy(record["release"]),
                "recovered": True, "stopped": stop["stopped"],
                "markers": stop["markers"], "registry_changed": True,
                "publication_budget_only": True,
                "workers_may_still_be_running": not stop["stopped"]}
    snapshot = _live().registry_snapshot(
        bootstrap_context["bootstrap_path"], bootstrap_context["bootstrap_sha256"])
    stop = _live().stop_requested(snapshot)
    if stop["stopped"]:
        return {"schema_version": 1, "status": "stopped_before_publication",
                "markers": stop["markers"], "prepared_release": pending["release"],
                "pending_publication": str(pending_path),
                "registry_changed": False, "worker_started": False,
                "hfss_started": False}
    else:
        _publish_json_atomic_noreplace(record_path, record, root / CONTROLLER_DIR)
    record_sha = _sha(record_path)
    _live().validate_registry_record(record_path, record_sha, bootstrap_context)
    pending_path.unlink()
    return {"schema_version": 1, "status": "published_successor",
            "batch": record["sequence"],
            "registry_record": {"path": str(record_path), "sha256": record_sha},
            "release": copy.deepcopy(record["release"]),
            "recovered": bool(pending.get("recovery_expected")),
            "publication_budget_only": True,
            "workers_may_still_be_running": True}


def once(*, bootstrap: Path, bootstrap_sha256: str,
         prior_manifest: Path, prior_root: Path, expected_prior_sha256: str,
         seed_inputs: Sequence[Path] = (), candidate_pool_size: int = cycle.DEFAULT_POOL_SIZE,
         seed: int = cycle.DEFAULT_SEED, pretrain_epochs: int = 30,
         epochs: int = 100, batch_train_size: int = 32,
         hidden_dims: Sequence[int] = (128, 128),
         model_output_root: Path | None = None) -> dict[str, Any]:
    """Inspect a physical cutoff once and publish at most one successor."""
    context, document, root = _bootstrap_context(bootstrap, bootstrap_sha256)
    lease_stream, token = _acquire_lease(root)
    with contextlib.ExitStack() as stack:
        stack.callback(_release_lease, lease_stream, token)
        stack.enter_context(_scoped_profiled_tempfile(root))
        stack.enter_context(_scoped_exploration_os(root))
        recovered = _finish_pending(root, context)
        if recovered is not None:
            return recovered
        prefix = _registry_prefix(context, document, root)
        stop = _stop_status(context, prefix)
        if stop["stopped"]:
            return {"schema_version": 1, "status": "stopped",
                    "markers": stop["markers"], "released_batches": len(prefix),
                    "registry_changed": False, "worker_started": False,
                    "hfss_started": False}
        cutoff = _cutoff(prefix)
        public = {"schema_version": 1, "status": cutoff["status"],
                  "released_batches": len(prefix), "next_batch": cutoff["next_batch"],
                  "unique_successes": len(cutoff["inventory"]["successes"]),
                  "consumed_successes": len(cutoff["consumed"]),
                  "new_unique_successes": len(cutoff["new"]),
                  "required_batch": cutoff["required_batch"],
                  "required_batch_new_successes": len(cutoff["required_batch_new"]),
                  "trigger": SUCCESS_TRIGGER}
        if cutoff["status"] != "ready":
            return public
        selected_model_root = (root / CONTROLLER_DIR / "models" if model_output_root is None
                               else _inside(root, Path(model_output_root),
                                            "model_output_root"))
        family = _family_preflight(cutoff, selected_model_root, seed)
        if not family["ready"]:
            return {**public, "status": "waiting_for_family_validation",
                    "source_inventory_id": cutoff["inventory"]["source_inventory_id"],
                    "family_validation": family}
        release_path, release_sha = _build_release(
            root=root, bootstrap_path=Path(bootstrap).resolve(strict=True),
            bootstrap_sha=bootstrap_sha256.lower(), prefix=prefix, cutoff=cutoff,
            prior_manifest=Path(prior_manifest).resolve(strict=True),
            prior_root=Path(prior_root).resolve(strict=True),
            expected_prior_sha256=expected_prior_sha256,
            seed_inputs=[Path(path).resolve(strict=True) for path in seed_inputs],
            candidate_pool_size=candidate_pool_size, seed=seed,
            pretrain_epochs=pretrain_epochs, epochs=epochs,
            batch_train_size=batch_train_size, hidden_dims=hidden_dims,
            model_output_root=selected_model_root)
        _live().validate_successor_release(release_path, release_sha, context)
        sequence = int(cutoff["next_batch"])
        registry_path = root / REGISTRY_DIR / f"batch-{sequence:03d}.json"
        record = _record_for_release(
            sequence=sequence, bootstrap_path=Path(bootstrap).resolve(strict=True),
            bootstrap_sha=bootstrap_sha256.lower(), previous=prefix[-1],
            release_path=release_path, release_sha=release_sha)
        pending = {"schema_version": 1, "recovery_expected": True,
                   "registry_path": str(registry_path),
                   "release": {"path": str(release_path), "sha256": release_sha},
                   "record": record}
        _publish_json_atomic_noreplace(root / CONTROLLER_DIR / PENDING_NAME, pending,
                                       root / CONTROLLER_DIR)
        result = _finish_pending(root, context)
        if result is None:
            raise RuntimeError("pending publication unexpectedly disappeared")
        result["recovered"] = False
        return result


def _refresh_watch_prefix(cache: dict[str, Any]) -> list[dict[str, Any]]:
    """Hash small immutable records and physically validate only new records."""
    if _sha(cache["bootstrap_path"]) != cache["bootstrap_sha256"]:
        raise ValueError("bootstrap changed while the controller was watching")
    directory = Path(cache["document"]["registry"]["directory"])
    paths = sorted(directory.glob("batch-*.json"))
    prefix = cache["prefix"]
    if len(paths) < len(prefix):
        raise ValueError("registry prefix shrank while the controller was watching")
    for item, path in zip(prefix, paths):
        if path.resolve() != item["path"] or _sha(path) != item["sha256"]:
            raise ValueError("an existing registry prefix record changed")
    for sequence, path in enumerate(paths[len(prefix):], start=len(prefix) + 1):
        if path.name != f"batch-{sequence:03d}.json":
            raise ValueError("new registry records are not contiguous")
        digest = _sha(path)
        validated = _live().validate_registry_record(
            path, digest, cache["bootstrap_context"])
        value = pb.read_json(path)
        previous = prefix[-1]
        if value.get("previous") != {"sequence": sequence - 1,
                                     "path": str(previous["path"]),
                                     "sha256": previous["sha256"]}:
            raise ValueError("new registry record does not extend the cached prefix")
        prefix.append({"sequence": sequence, "path": path.resolve(),
                       "sha256": digest, "record": value,
                       "validated": validated})
    return prefix


def _metadata_count(bootstrap: Path, bootstrap_sha256: str,
                    cache: dict[str, Any] | None = None) -> dict[str, Any]:
    """Cheap watch poll: count status-ok rows without a physical source gate."""
    if cache is None:
        context, document, root = _bootstrap_context(bootstrap, bootstrap_sha256)
        prefix = _registry_prefix(context, document, root)
    else:
        prefix = _refresh_watch_prefix(cache)
    stop = _stop_status(context if cache is None else cache["bootstrap_context"], prefix)
    if stop["stopped"]:
        return {"released_batches": len(prefix), "next_batch": len(prefix) + 1,
                "successes": 0, "new_successes": 0,
                "required_new_successes": 0, "success_set_id": None,
                "potentially_ready": False, "stopped": True,
                "stop_markers": stop["markers"]}
    consumed = _consumed(prefix)
    seen = set()
    successes: dict[str, int] = {}
    for item in prefix:
        release = pb.read_json(Path(item["record"]["release"]["path"]))
        for entry in release["queue"]["jobs"]:
            job = entry["job"]
            dataset = Path(release["dataset_root"])
            manifest_path = dataset / job["input"] / "manifest.json"
            store_results = dataset / job["store"] / "results.json"
            if not manifest_path.is_file() or not store_results.is_file():
                continue
            rows = {row["id"]: row for row in pb.read_json(manifest_path)}
            for row_id, result in pb.read_json(store_results).items():
                row = rows.get(row_id)
                if (row is None or not isinstance(result, dict) or
                        result.get("status") != "ok" or "error" in result):
                    continue
                digest = row["pattern_sha256"]
                if digest in seen:
                    raise ValueError("metadata poll found a duplicate released pattern")
                seen.add(digest); successes[digest] = item["sequence"]
    new = set(successes) - set(consumed)
    next_batch = len(prefix) + 1
    required = ({digest for digest in new if successes[digest] == 2}
                if next_batch == 3 else new)
    return {"released_batches": len(prefix), "next_batch": next_batch,
            "successes": len(successes), "new_successes": len(new),
            "required_new_successes": len(required),
            "success_set_id": _json_id(sorted(successes)),
            "potentially_ready": next_batch <= MAX_BATCHES and
            len(new) >= SUCCESS_TRIGGER and len(required) >= SUCCESS_TRIGGER}


def watch(*, poll_seconds: int = POLL_SECONDS, **kwargs: Any) -> dict[str, Any]:
    if int(poll_seconds) < POLL_SECONDS:
        raise ValueError(f"live controller poll interval must be at least {POLL_SECONDS} seconds")
    context, document, root = _bootstrap_context(
        kwargs["bootstrap"], kwargs["bootstrap_sha256"])
    cache = {"bootstrap_context": context, "document": document, "root": root,
             "bootstrap_path": Path(kwargs["bootstrap"]).resolve(strict=True),
             "bootstrap_sha256": kwargs["bootstrap_sha256"].lower(),
             "prefix": _registry_prefix(context, document, root)}
    last_id = None
    blocked_success_set_id = None
    while True:
        metadata = _metadata_count(kwargs["bootstrap"], kwargs["bootstrap_sha256"], cache)
        marker = _json_id(metadata)
        if marker != last_id:
            print(json.dumps({"schema_version": 1, "status": "watching", **metadata},
                             ensure_ascii=False), flush=True)
            last_id = marker
        if metadata.get("stopped"):
            return {"schema_version": 1, "status": "stopped",
                    **metadata, "registry_changed": False,
                    "worker_started": False, "hfss_started": False}
        if (metadata["potentially_ready"] and
                metadata["success_set_id"] != blocked_success_set_id):
            result = once(**kwargs)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if result["status"] == "published_successor":
                blocked_success_set_id = None
                continue
            if result["status"] == "waiting_for_family_validation":
                blocked_success_set_id = metadata["success_set_id"]
        if metadata["next_batch"] > MAX_BATCHES:
            return {"schema_version": 1, "status": "publication_budget_exhausted",
                    **metadata, "publication_budget_only": True,
                    "workers_may_still_be_running": True,
                    "final_worker_health_owned_by_root": True}
        time.sleep(int(poll_seconds))


def _add_execution_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bootstrap", type=Path, required=True)
    parser.add_argument("--bootstrap-sha256", required=True)
    parser.add_argument("--prior-manifest", type=Path, required=True)
    parser.add_argument("--prior-root", type=Path, required=True)
    parser.add_argument("--expected-prior-sha256", required=True)
    parser.add_argument("--seed-input", type=Path, action="append", default=[])
    parser.add_argument("--candidate-pool-size", type=int, default=cycle.DEFAULT_POOL_SIZE)
    parser.add_argument("--seed", type=int, default=cycle.DEFAULT_SEED)
    parser.add_argument("--pretrain-epochs", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-train-size", type=int, default=32)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--model-output-root", type=Path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare-local")
    prepare_parser.add_argument("--actor-root", type=Path, required=True)
    prepare_parser.add_argument("--initial-release", type=Path, required=True)
    prepare_parser.add_argument("--initial-release-sha256", required=True)
    prepare_parser.add_argument("--producer-commit", required=True)
    prepare_parser.add_argument("--cache-source", type=Path)
    for name in ("once", "watch"):
        item = sub.add_parser(name)
        _add_execution_arguments(item)
        if name == "watch":
            item.add_argument("--poll-seconds", type=int, default=POLL_SECONDS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare-local":
        result = prepare_local(actor_root=args.actor_root,
                               initial_release=args.initial_release,
                               initial_release_sha256=args.initial_release_sha256,
                               producer_commit=args.producer_commit,
                               cache_source=args.cache_source)
    else:
        values = vars(args).copy(); values.pop("command")
        result = watch(**values) if args.command == "watch" else once(**values)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
