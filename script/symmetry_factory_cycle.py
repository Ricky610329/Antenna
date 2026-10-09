"""One recoverable prepare/dispatch cycle for the R80 symmetry data factory.

Preparation writes staged artifacts below the explicit local work directory and
short-lived ownership metadata below the dataset/training roots.  Dispatch is a
separate call: it copies frozen 16-item shards, re-runs the duplicate gate, and
uses the existing queue helper.  Neither path runs HFSS or starts a process.
"""
from __future__ import annotations

import argparse
import contextlib
import heapq
import hashlib
import itertools
import json
import os
import socket
import shutil
import statistics
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from antenna.measurement import measurement_id, score_spec_id
from script import profiled_batch as pb
from script import symmetry_factory as factory
from script import symmetry_incumbent_shell_pilot as shell_pilot
from script import symmetry_sm_pool as sm
from script import symmetry_training as training
from script.profiled_shards import snapshot_successes, split_bundle


class LiveSnapshotChanged(RuntimeError):
    """A worker advanced queue/results during a physical controller snapshot."""


class DatasetLockBusy(LiveSnapshotChanged):
    """Another process briefly owns the scope dataset mutation boundary."""


class CoordinatorCancelled(RuntimeError):
    """A waiting LOW transaction was cancelled during watcher shutdown."""


class DatasetWriteCoordinator:
    """Serialize one watcher's dataset transactions with priority and reservations.

    The kernel lock still excludes other processes.  This coordinator prevents the
    main cycle and its LOW maintainer thread from racing that non-blocking lock and
    keeps prepared, not-yet-dispatched mainline hashes unavailable to LOW planning.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._sequence = itertools.count()
        self._waiting: list[tuple[int, int]] = []
        self._active: tuple[int, int] | None = None
        self._protected: dict[str, frozenset[str]] = {}

    def _acquire(self, priority: int,
                 cancel_event: threading.Event | None = None) -> tuple[int, int]:
        ticket = (int(priority), next(self._sequence))
        with self._condition:
            heapq.heappush(self._waiting, ticket)
            while self._active is not None or self._waiting[0] != ticket:
                if cancel_event is not None and cancel_event.is_set():
                    self._waiting.remove(ticket)
                    heapq.heapify(self._waiting)
                    self._condition.notify_all()
                    raise CoordinatorCancelled("dataset coordinator wait cancelled")
                self._condition.wait(0.25 if cancel_event is not None else None)
            if heapq.heappop(self._waiting) != ticket:
                raise RuntimeError("dataset coordinator queue corruption")
            self._active = ticket
        return ticket

    def _release(self, ticket: tuple[int, int]) -> None:
        with self._condition:
            if self._active != ticket:
                raise RuntimeError("dataset coordinator released by a non-owner")
            self._active = None
            self._condition.notify_all()

    @contextmanager
    def turn(self, priority: int, *, cancel_event: threading.Event | None = None):
        ticket = self._acquire(priority, cancel_event)
        try:
            yield
        finally:
            self._release(ticket)

    def planning_guard(self, priority: int = 1) -> "_PlanningGuard":
        return _PlanningGuard(self, priority)

    def protect(self, receipt: str | Path, hashes: Sequence[str]) -> None:
        key = str(Path(receipt).resolve())
        values = frozenset(str(value) for value in hashes)
        with self._condition:
            prior = self._protected.get(key)
            if prior is not None and prior != values:
                raise ValueError("prepared mainline reservation changed for one receipt")
            self._protected[key] = values

    def release_receipt(self, receipt: str | Path) -> None:
        with self._condition:
            self._protected.pop(str(Path(receipt).resolve()), None)

    def protected_hashes(self) -> set[str]:
        with self._condition:
            return set().union(*self._protected.values()) if self._protected else set()


class _PlanningGuard:
    """Priority turn that can yield only around the actual SM training call."""

    def __init__(self, coordinator: DatasetWriteCoordinator, priority: int) -> None:
        self._coordinator = coordinator
        self._priority = int(priority)
        self._ticket: tuple[int, int] | None = None

    def __enter__(self) -> "_PlanningGuard":
        if self._ticket is not None:
            raise RuntimeError("planning guard entered twice")
        self._ticket = self._coordinator._acquire(self._priority)
        return self

    def release_for_training(self) -> None:
        if self._ticket is None:
            raise RuntimeError("planning guard is not held")
        ticket, self._ticket = self._ticket, None
        self._coordinator._release(ticket)

    def reacquire_after_training(self) -> None:
        if self._ticket is not None:
            raise RuntimeError("planning guard is already held")
        self._ticket = self._coordinator._acquire(self._priority)

    def reserve(self, receipt: str | Path, hashes: Sequence[str]) -> None:
        if self._ticket is None:
            raise RuntimeError("planning guard must be held while reserving a receipt")
        self._coordinator.protect(receipt, hashes)

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._ticket is not None:
            ticket, self._ticket = self._ticket, None
            self._coordinator._release(ticket)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pb.atomic_json(path, value)


def _content_id(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _safe_child(root: Path, name: str) -> Path:
    if not isinstance(name, str) or Path(name).name != name or any(c in name for c in "/\\:"):
        raise ValueError(f"unsafe dataset child name: {name!r}")
    path = (root / name).resolve()
    path.relative_to(root.resolve())
    return path


@contextmanager
def _exclusive_lock(path: Path):
    """Use an auto-released kernel lock while retaining owner metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    token = json.dumps({"pid": os.getpid(), "host": socket.gethostname(),
                        "created_utc": datetime.now(timezone.utc).isoformat(),
                        "path": str(path.resolve())}, sort_keys=True)
    with path.open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0"); stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(
                    f"another symmetry factory controller holds {path}; the retained lock file "
                    "contains its host/PID/time metadata") from exc
            unlock = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError(
                    f"another symmetry factory controller holds {path}; the retained lock file "
                    "contains its host/PID/time metadata") from exc
            unlock = lambda: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        try:
            stream.seek(1); stream.truncate(); stream.write(token.encode("utf-8")); stream.flush()
            os.fsync(stream.fileno())
            yield
        finally:
            stream.seek(0)
            with contextlib.suppress(OSError):
                unlock()


def _dataset_lock(dataset_root: Path, scope: str) -> Path:
    suffix = hashlib.sha256(scope.encode("utf-8")).hexdigest()[:16]
    return dataset_root / "jobs_state" / f"factory-controller-{suffix}.lock"


@contextmanager
def _dataset_exclusive_lock(dataset_root: Path, scope: str):
    """Acquire the short dataset boundary, classifying contention as retryable."""

    path = _dataset_lock(dataset_root, scope)
    manager = _exclusive_lock(path)
    try:
        manager.__enter__()
    except RuntimeError as exc:
        if "another symmetry factory controller holds" not in str(exc):
            raise
        raise DatasetLockBusy(str(exc)) from exc
    try:
        yield
    finally:
        manager.__exit__(None, None, None)


def _train_version_with_guard(train_root: Path, version: int,
                              planning_guard: _PlanningGuard | None) -> None:
    """Let LOW refill only while the expensive SM training call is active."""

    if planning_guard is not None:
        planning_guard.release_for_training()
    try:
        training.train_version(train_root, version)
    finally:
        if planning_guard is not None:
            planning_guard.reacquire_after_training()


def _normalize_retry_workers(values: Sequence[str] | None) -> tuple[str, ...]:
    """Return a stable explicit roster; an empty roster never proves terminal failure."""

    if values is None:
        return ()
    if isinstance(values, (str, bytes)):
        raise ValueError("expected_retry_workers must be a sequence of worker identities")
    workers = []
    for value in values:
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError("expected_retry_workers must contain nonempty normalized strings")
        workers.append(value)
    if len(set(workers)) != len(workers):
        raise ValueError("expected_retry_workers must be unique")
    return tuple(sorted(workers))


def _marker_binding(path: Path) -> tuple[dict[str, Any], Any | None]:
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return {"exists": False}, None
    binding = {"exists": True, "sha256": hashlib.sha256(raw).hexdigest()}
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        binding["valid_json"] = False
        return binding, None
    binding["valid_json"] = True
    return binding, payload


def _job_state_binding(state_root: Path, store: str) -> tuple[dict[str, Any], dict[str, Any]]:
    bindings, payloads = {}, {}
    for suffix in ("claim", "done", "fail"):
        binding, payload = _marker_binding(state_root / f"{store}.{suffix}")
        bindings[suffix] = binding
        payloads[suffix] = payload
    return bindings, payloads


def _terminal_fail(payload: Any, expected_workers: tuple[str, ...]) -> bool:
    """Only an exact, fully exhausted roster releases failed residual capacity."""

    if not expected_workers or not isinstance(payload, dict):
        return False
    machines = payload.get("machines")
    if (not isinstance(machines, list) or any(not isinstance(item, str) or not item
                                               for item in machines) or
            len(set(machines)) != len(machines)):
        return False
    observed = set(machines)
    expected = set(expected_workers)
    if not observed <= expected:
        return False
    return observed == expected


def _require_state_bindings(state_root: Path, bindings: Sequence[Mapping[str, Any]]) -> None:
    for item in bindings:
        actual, _payloads = _job_state_binding(state_root, str(item["store"]))
        if actual != item["markers"]:
            raise LiveSnapshotChanged(
                f"worker state changed during controller scan: {item['store']}")


def _physical_snapshot(dataset: Path, scope: str) -> dict[str, Any]:
    try:
        return factory.snapshot(dataset, scope)
    except ValueError as exc:
        message = str(exc)
        if "live store changed during snapshot" in message:
            # The legacy snapshot error does not name the changed file.  A
            # fresh pass turns stable identity/manifest corruption into its
            # specific fatal validation error, while a still-moving results
            # file remains a retryable race.
            try:
                return factory.snapshot(dataset, scope)
            except ValueError as retry_exc:
                if "live store changed during snapshot" in str(retry_exc):
                    raise LiveSnapshotChanged(str(retry_exc)) from retry_exc
                raise
        if "queue changed during snapshot" in message:
            raise LiveSnapshotChanged(message) from exc
        raise


def _tree_hashes(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): pb.file_sha256(path)
            for path in sorted(root.rglob("*")) if path.is_file()}


def _path_binding(path: Path) -> dict[str, str]:
    path = path.resolve()
    if path.is_file():
        return {"path": str(path), "sha256": pb.file_sha256(path)}
    if path.is_dir():
        return {"path": str(path), "sha256": _content_id(_tree_hashes(path))}
    raise FileNotFoundError(path)


def _load_pilot_request(pilot_request: str | Path | None,
                        profile_config: str | Path) -> tuple[Any | None, dict[str, Any] | None]:
    """Load the optional one-shot request and return its exact serialized binding."""

    if pilot_request is None:
        return None, None
    request = shell_pilot.load_request(Path(pilot_request).resolve(),
                                       Path(profile_config).resolve())
    binding = dict(request.binding)
    required = {"request_path", "request_sha256", "pilot_id"}
    if not required <= binding.keys() or binding["pilot_id"] != request.pilot_id:
        raise ValueError("pilot request lacks its exact path/SHA/pilot identity binding")
    # Require a JSON-safe stable value before it enters cycle and receipt identities.
    json.loads(json.dumps(binding, sort_keys=True, ensure_ascii=False))
    return request, binding


def _require_pilot_request_binding(pilot_request: str | Path | None,
                                   profile_config: str | Path,
                                   expected: Mapping[str, Any] | None) -> Any | None:
    request, actual = _load_pilot_request(pilot_request, profile_config)
    if actual != expected:
        raise ValueError("pilot request presence or exact binding changed")
    return request


def _pilot_decision_path(local: Path, pilot_id: str) -> Path:
    if (not isinstance(pilot_id, str) or len(pilot_id) != 64 or
            any(character not in "0123456789abcdef" for character in pilot_id)):
        raise ValueError("pilot_id must be a lowercase SHA-256 identity")
    return local / "pilot_decisions" / f"{pilot_id}.json"


def _pilot_terminal_decision(local: Path, binding: Mapping[str, Any], *,
                             code: str, detail: str) -> dict[str, Any]:
    """Create or validate the immutable exact-request terminal-unavailable decision."""

    path = _pilot_decision_path(local, str(binding["pilot_id"]))
    expected_identity = {"schema_version": 1, "status": "terminal_unavailable",
                         "pilot_request": dict(binding), "pilot_id": binding["pilot_id"]}
    if path.is_file():
        return _validate_pilot_terminal_decision(_read(path), expected_identity)
    decision = {**expected_identity, "code": code, "detail": detail}
    decision["decision_id"] = _content_id(decision)
    _write(path, decision)
    if _read(path) != decision:
        raise ValueError("pilot terminal decision publication differs")
    return decision


def _validate_pilot_terminal_decision(decision: Mapping[str, Any],
                                      expected_identity: Mapping[str, Any]) -> dict[str, Any]:
    if any(decision.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("existing pilot terminal decision has a different identity")
    if decision.get("code") not in {
            "final_remaining_below_16", "empty_stratum", "no_eligible_blind"}:
        raise ValueError("existing pilot terminal decision has an invalid reason")
    payload = {key: value for key, value in decision.items() if key != "decision_id"}
    if decision.get("decision_id") != _content_id(payload):
        raise ValueError("existing pilot terminal decision content changed")
    return dict(decision)


def _existing_pilot_terminal_decision(local: Path,
                                      binding: Mapping[str, Any]) -> dict[str, Any] | None:
    path = _pilot_decision_path(local, str(binding["pilot_id"]))
    if not path.is_file():
        return None
    decision = _read(path)
    expected = {"schema_version": 1, "status": "terminal_unavailable",
                "pilot_request": dict(binding), "pilot_id": binding["pilot_id"]}
    return _validate_pilot_terminal_decision(decision, expected)


def _validate_bundle(path: Path, expected_cfg: Any) -> None:
    cfg = pb.load_profile_config(path / "config.yaml")
    if (cfg is None or measurement_id(cfg.measurement) != measurement_id(expected_cfg.measurement) or
            score_spec_id(cfg.score_spec) != score_spec_id(expected_cfg.score_spec) or
            cfg.scope != expected_cfg.scope):
        raise ValueError(f"staged bundle profile differs from factory profile: {path}")
    pb.validate_input(path, cfg)


def _publish_bundle(output: Path, expected_cfg: Any, producer: Callable[[Path], None],
                    completion_validator: Callable[[Path], None] | None = None) -> Path:
    """Recover a deterministic local pending directory, then rename atomically."""

    if output.exists():
        _validate_bundle(output, expected_cfg)
        if completion_validator is not None:
            completion_validator(output)
        return output
    pending = output.with_name(f".{output.name}.pending")
    if pending.exists():
        try:
            _validate_bundle(pending, expected_cfg)
            if completion_validator is not None:
                completion_validator(pending)
        except Exception:
            resolved, parent = pending.resolve(), output.resolve().parent
            if resolved.parent != parent or resolved.name != f".{output.name}.pending":
                raise RuntimeError(f"refusing unsafe pending cleanup: {resolved}")
            shutil.rmtree(resolved)
    if not pending.exists():
        producer(pending)
        _validate_bundle(pending, expected_cfg)
        if completion_validator is not None:
            completion_validator(pending)
    if output.exists():
        raise FileExistsError(f"bundle appeared during atomic publish: {output}")
    os.replace(pending, output)
    return output


def _guided_completion(path: Path, *, cycle_id: str,
                       predictor_model_ids: Sequence[Mapping[str, Any]]) -> tuple[int, set[str]]:
    """Validate a completed guided cohort without imposing a newly computed capacity."""

    marker = _read(path / "factory_bundle_complete.json")
    manifest = _read(path / "manifest.json")
    count = marker.get("selected_count")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("guided bundle completion has an invalid selected count")
    hashes = [str(row["pattern_sha256"]) for row in manifest]
    expected = {
        "schema_version": 1, "cycle_id": cycle_id, "selected_count": count,
        "predictor_model_ids": list(predictor_model_ids),
        "manifest_sha256": pb.file_sha256(path / "manifest.json"),
        "sm_pool_audit_sha256": pb.file_sha256(path / "sm_pool_audit.json"),
    }
    if marker != expected:
        raise ValueError("guided bundle completion binding differs from cycle")
    if len(manifest) != count or len(set(hashes)) != count:
        raise ValueError("guided bundle completion count differs from unique manifest rows")
    return count, set(hashes)


def _guided_replan_proof(path: Path, *, cycle_id: str,
                         predictor_model_ids: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    plan = _read(path / "factory_guided_plan.json")
    if set(plan) != {"schema_version", "plan_id", "identity"} or plan.get("schema_version") != 1:
        raise ValueError("guided replan proof has an invalid schema")
    identity = plan.get("identity")
    if not isinstance(identity, dict) or plan.get("plan_id") != _content_id(identity):
        raise ValueError("guided replan proof identity differs")
    name = path.name
    if name.startswith(".") and name.endswith(".pending"):
        name = name[1:-len(".pending")]
    if name != f"guided_bundle_replan_{plan['plan_id'][:16]}":
        raise ValueError("guided replan directory differs from plan identity")
    count, hashes = _guided_completion(
        path, cycle_id=cycle_id, predictor_model_ids=predictor_model_ids)
    if (identity.get("cycle_id") != cycle_id or
            identity.get("predictor_model_ids") != list(predictor_model_ids) or
            identity.get("selected_count") != count):
        raise ValueError("guided replan proof differs from cycle or predictor")
    return {"plan": plan, "count": count, "hashes": hashes}


def _load_profile(path: Path):
    cfg = pb.load_profile_config(path)
    if cfg is None:
        raise ValueError("profile_config must be a profiled YAML file")
    policy = factory.validate_factory_config(vars(cfg))
    return cfg, policy


def _trained(root: Path) -> tuple[int, set[str], dict[str, Any] | None]:
    state_path = root / "state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"training work directory is not initialized: {root}")
    state = _read(state_path)
    version = int(state["latest_data_version"])
    if version == 0:
        return 0, set(), None
    version_dir = root / "data" / f"data-v{version:03d}"
    rows, receipt = _read(version_dir / "manifest.json"), _read(version_dir / "receipt.json")
    if training.content_id(rows) != receipt["manifest_id"]:
        raise ValueError("latest training manifest binding is invalid")
    return version, {row["pattern_sha256"] for row in rows}, receipt


def _protocol_binding(root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    path = root / "protocol.json"
    before = pb.file_sha256(path)
    protocol = training._protocol(root)
    if pb.file_sha256(path) != before:
        raise ValueError("training protocol changed while binding the cycle")
    return protocol, {"protocol_id": protocol["protocol_id"], "sha256": before,
                      "path": str(path.resolve())}


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def _spearman(predicted: Sequence[float], measured: Sequence[float]) -> dict[str, Any]:
    x, y = np.asarray(predicted, dtype=np.float64), np.asarray(measured, dtype=np.float64)
    result: dict[str, Any] = {"n": int(len(x)), "rho": None}
    if len(x) < 3:
        result["undefined_reason"] = "fewer_than_3_pairs"
        return result
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        result["undefined_reason"] = "constant_prediction_or_measurement"
        return result
    rho = float(np.corrcoef(_rank(x), _rank(y))[0, 1])
    if not np.isfinite(rho):
        result["undefined_reason"] = "non_finite_rank_correlation"
        return result
    result["rho"] = rho
    return result


def _prediction_summary(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    components = {}
    for name in ("s11", "gain", "radiation_phi0", "radiation_phi90", "full_vector"):
        errors = [float(item["mae"][name]) for item in items if item["mae"].get(name) is not None]
        components[name] = {"n": len(errors), "prediction_missing": len(items) - len(errors),
                            "mean_absolute_error": (float(np.mean(errors)) if errors else None)}
    ranks = {}
    for name in ("factory_score", "navigation_score"):
        paired = [(item["predicted_scores"].get(name), item["measured_scores"][name])
                  for item in items if item["predicted_scores"].get(name) is not None]
        ranks[name] = _spearman([pair[0] for pair in paired], [pair[1] for pair in paired])
    complete = sum(item["prediction_complete"] for item in items)
    fields = ("predicted_s11", "predicted_gain", "predicted_radiation_phi0",
              "predicted_radiation_phi90", "predicted_radiation_theta",
              "factory_score", "navigation_score")
    return {"n": len(items), "prediction_complete": complete,
            "prediction_missing": len(items) - complete,
            "prediction_explicitly_invalid": sum(
                item["prediction_explicitly_invalid"] for item in items),
            "prediction_missing_by_field": {
                field: sum(field in item["prediction_missing_fields"] for item in items)
                for field in fields},
            "components": components, "spearman": ranks}


def audit_frozen_predictions(snapshot_store: str | Path, training_workdir: str | Path,
                             output_path: str | Path) -> Path:
    """Freeze saved prospective predictions versus measured curves before training."""

    store = Path(snapshot_store).resolve(); train_root = Path(training_workdir).resolve()
    output = Path(output_path).resolve()
    protocol, protocol_binding = _protocol_binding(train_root)
    source_hashes = {name: pb.file_sha256(store / name) for name in
                     ("measurement.json", "score_spec.json", "manifest.json", "results.json",
                      "source_bindings.json")}
    if (measurement_id(_read(store / "measurement.json")) != protocol["measurement_id"] or
            score_spec_id(_read(store / "score_spec.json")) != protocol["score_spec_id"]):
        raise ValueError("prediction-audit store differs from training protocol")
    results = pb.validate_store(store, require_complete=True)
    rows = _read(store / "manifest.json")
    items = []
    theta = np.linspace(-90.0, 90.0, 91, dtype=np.float32)
    for row in rows:
        entry = results[row["id"]]
        _pattern, target = training._target(store / entry["sample_file"],
                                            store / entry["rad_file"],
                                            protocol["frequencies_ghz"])
        measured_response = target[:34].numpy().reshape(2, 17)
        measured_rad = target[34:].numpy().reshape(2, 91)
        measured_batch = sm.PredictionBatch(
            measured_response[None], np.zeros(1, dtype=np.float32), measured_rad[None], theta)
        measured_scores = {key: float(value[0]) for key, value in
                           sm.score_predictions(measured_batch).items()}
        specifications = {
            "s11": ("predicted_s11", measured_response[0], 17),
            "gain": ("predicted_gain", measured_response[1], 17),
            "radiation_phi0": ("predicted_radiation_phi0", measured_rad[0], 91),
            "radiation_phi90": ("predicted_radiation_phi90", measured_rad[1], 91),
        }
        explicitly_invalid = row.get("prediction_valid") is False
        errors, vectors, missing, missing_fields = {}, [], [], []
        for name, (key, measured, size) in specifications.items():
            value = None if explicitly_invalid else row.get(key)
            if value is None:
                errors[name] = None; missing.append(name); missing_fields.append(key)
                continue
            predicted = np.asarray(value, dtype=np.float64)
            if predicted.shape != (size,) or not np.isfinite(predicted).all():
                raise ValueError(f"{row['id']} has invalid saved {key}")
            errors[name] = float(np.mean(np.abs(predicted - measured)))
            vectors.append((name, predicted, measured))
        theta_value = None if explicitly_invalid else row.get("predicted_radiation_theta")
        if theta_value is None:
            missing_fields.append("predicted_radiation_theta")
        else:
            predicted_theta = np.asarray(theta_value, dtype=np.float64)
            if predicted_theta.shape != (91,) or not np.allclose(predicted_theta, theta,
                                                                  rtol=0.0, atol=1e-9):
                raise ValueError(f"{row['id']} has invalid predicted_radiation_theta")
        complete = not missing and theta_value is not None
        if complete:
            predicted_full = np.concatenate([value[1] for value in vectors])
            measured_full = np.concatenate([value[2] for value in vectors])
            errors["full_vector"] = float(np.mean(np.abs(predicted_full - measured_full)))
        else:
            errors["full_vector"] = None
        predicted_scores = {}
        for name in ("factory_score", "navigation_score"):
            value = None if explicitly_invalid else row.get(name)
            if value is not None and (not isinstance(value, (int, float)) or
                                      not np.isfinite(float(value))):
                raise ValueError(f"{row['id']} has invalid saved {name}")
            predicted_scores[name] = (float(value) if value is not None else None)
            if value is None:
                missing_fields.append(name)
        items.append({"id": row["id"], "selection_arm": row.get("selection_arm", "unspecified"),
                      "prediction_explicitly_invalid": explicitly_invalid,
                      "prediction_complete": complete, "prediction_missing_components": missing,
                      "prediction_missing_fields": missing_fields,
                      "mae": errors, "predicted_scores": predicted_scores,
                      "measured_scores": measured_scores})
    arms = {arm: _prediction_summary([item for item in items if item["selection_arm"] == arm])
            for arm in sorted({item["selection_arm"] for item in items})}
    audit = {
        "schema_version": 1, "order": "saved_predictions_audited_before_ingest_and_training",
        "quality_gate": None, "all_valid_observations_retained_for_training": True,
        "source_store": str(store), "source_metadata_sha256": source_hashes,
        "protocol": protocol_binding, "overall": _prediction_summary(items),
        "selection_arms": arms, "items": items,
    }
    if any(pb.file_sha256(store / name) != digest for name, digest in source_hashes.items()):
        raise ValueError("frozen snapshot changed during prediction audit")
    if output.exists():
        if _read(output) != audit:
            raise ValueError("immutable pretrain prediction audit differs on recovery")
        return output
    _write(output, audit)
    return output


def _training_summary(root: Path, version: int) -> dict[str, Any] | None:
    if version <= 0:
        return None
    path = root / "models" / f"data-v{version:03d}" / "summary.json"
    if not path.is_file():
        return None
    summary = _read(path)
    return {"path": str(path.resolve()), "sha256": pb.file_sha256(path),
            "binding": summary.get("binding"), "counts": summary.get("counts"),
            "ensemble_profile_holdout": summary.get("ensemble_profile_holdout"),
            "validation_policy": summary.get("validation_policy"),
            "quality_threshold": None}


def _input_physical_proof(input_dir: Path, cfg: Any,
                          metadata: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Physically validate one input and retain a same-invocation immutable proof."""

    names = ("config.yaml", "measurement.json", "score_spec.json", "manifest.json")
    hashes = dict(metadata) if metadata is not None else {
        name: pb.file_sha256(input_dir / name) for name in names}
    manifest_rows = _read(input_dir / "manifest.json")
    pattern_files = {row["id"]: pb.file_sha256(input_dir / f"{row['id']}.pt")
                     for row in manifest_rows}
    rows = pb.validate_input(input_dir, cfg)
    if rows != manifest_rows:
        raise ValueError(f"input manifest changed during physical validation: {input_dir}")
    if any(pb.file_sha256(input_dir / name) != digest for name, digest in hashes.items()):
        raise ValueError(f"input changed during physical validation: {input_dir}")
    if any(pb.file_sha256(input_dir / f"{name}.pt") != digest
           for name, digest in pattern_files.items()):
        raise ValueError(f"input pattern changed during physical validation: {input_dir}")
    return {"input": str(input_dir), "rows": rows,
            "input_metadata_sha256": hashes,
            "input_pattern_file_sha256": pattern_files}


def _successful_entries(results: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {name: entry for name, entry in results.items()
            if isinstance(entry, dict) and entry.get("status") == "ok" and
            "error" not in entry}


def _safe_store_file(store: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"successful observation lacks {label}")
    path = (store / relative).resolve()
    try:
        path.relative_to(store.resolve())
    except ValueError as exc:
        raise ValueError(f"successful observation {label} escapes store") from exc
    if not path.is_file():
        raise ValueError(f"successful observation lacks {label}: {path}")
    return path


def _recheck_success_raw(proof: Mapping[str, Any]) -> None:
    """Rehash every counted cutoff success without deserializing/replaying it."""

    store = Path(proof["store"])
    for name, entry in _successful_entries(proof["results"]).items():
        sample = _safe_store_file(store, entry.get("sample_file"), f"{name} sample")
        if pb.file_sha256(sample) != entry.get("sample_sha256"):
            raise ValueError(f"frozen successful sample changed: {store}: {name}")
        rad_file = entry.get("rad_file")
        if rad_file is not None:
            radiation = _safe_store_file(store, rad_file, f"{name} radiation")
            if pb.file_sha256(radiation) != entry.get("rad_sha256"):
                raise ValueError(f"frozen successful radiation changed: {store}: {name}")


def _recheck_input_proof(proof: Mapping[str, Any]) -> None:
    root = Path(proof["input"])
    changed = [name for name, digest in proof["input_metadata_sha256"].items()
               if pb.file_sha256(root / name) != digest]
    if changed:
        raise ValueError(f"input changed after physical audit: {root}: {changed}")
    changed_patterns = [name for name, digest in proof["input_pattern_file_sha256"].items()
                        if pb.file_sha256(root / f"{name}.pt") != digest]
    if changed_patterns:
        raise ValueError(
            f"input pattern bytes changed after physical audit: {root}: {changed_patterns[:20]}")


def _recheck_pair_proof(proof: Mapping[str, Any],
                        append_cache: dict[str, Any] | None = None) -> set[str]:
    """Accept validated nonterminal progress while retaining the original cutoff."""

    store = Path(proof["store"])
    for name, digest in proof["store_metadata_sha256"].items():
        if name != "results.json" and pb.file_sha256(store / name) != digest:
            raise ValueError(f"store identity changed after physical audit: {store}: {name}")
    results_path = store / "results.json"
    try:
        before_bytes = results_path.read_bytes()
    except FileNotFoundError as exc:
        if results_path.is_file():
            raise LiveSnapshotChanged(f"results appeared while rechecking proof: {store}") from exc
        raise ValueError(f"physically audited results disappeared: {store}") from exc
    try:
        current = json.loads(before_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        if results_path.read_bytes() != before_bytes:
            raise LiveSnapshotChanged(f"results changed while rechecking proof: {store}") from exc
        raise ValueError(f"results.json is stably malformed after physical audit: {store}") from exc
    if not isinstance(current, dict):
        raise ValueError(f"results.json must contain an object: {store}")
    frozen = proof["results"]
    current_sha = hashlib.sha256(before_bytes).hexdigest()
    if current_sha != proof["store_metadata_sha256"]["results.json"] and proof["terminal"]:
        raise ValueError(f"terminal store results changed after physical audit: {store}")
    removed = sorted(set(frozen) - set(current))
    modified = sorted(name for name in frozen if current.get(name) != frozen[name])
    frozen_ok = set(_successful_entries(frozen))
    immutable = sorted((set(removed) | set(modified)) & frozen_ok)
    if immutable:
        raise ValueError(f"frozen successful result entry changed: {store}: {immutable[:20]}")
    known = {row["id"] for row in proof["rows"]}
    extras = sorted(set(current) - set(frozen))
    unknown = sorted(set(extras) - known)
    if unknown:
        raise ValueError(f"results append has unknown manifest ids: {store}: {unknown[:20]}")
    retryable_changes = sorted((set(removed) | set(modified)) - frozen_ok)
    if extras or retryable_changes:
        cached = None if append_cache is None else append_cache.get(str(store))
        if not isinstance(cached, dict) or cached.get("results_sha256") != current_sha:
            try:
                validated = pb.validate_store(store, require_complete=False)
            except ValueError as exc:
                if pb.file_sha256(results_path) != current_sha:
                    raise LiveSnapshotChanged(
                        f"results changed during appended-entry validation: {store}") from exc
                raise
            if pb.file_sha256(results_path) != current_sha:
                raise LiveSnapshotChanged(
                    f"results changed during appended-entry validation: {store}")
            if any(validated.get(name) != current[name] for name in current):
                raise LiveSnapshotChanged(f"validated results differ from append snapshot: {store}")
            cached = {"results_sha256": current_sha, "results": current}
            if append_cache is not None:
                append_cache[str(store)] = cached
        _recheck_success_raw({"store": str(store), "results": cached["results"]})
    elif current_sha != proof["store_metadata_sha256"]["results.json"]:
        raise LiveSnapshotChanged(f"results bytes changed after physical audit: {store}")
    if hashlib.sha256(results_path.read_bytes()).hexdigest() != current_sha:
        raise LiveSnapshotChanged(f"results changed after proof recheck: {store}")
    _recheck_success_raw(proof)
    rows = {row["id"]: row for row in proof["rows"]}
    late_names = set(extras) | (set(modified) - set(removed))
    return {rows[name]["pattern_sha256"] for name in late_names
            if isinstance(current[name], dict) and current[name].get("status") == "ok" and
            "error" not in current[name]}


def _recheck_frozen_markers(state_root: Path, bindings: Sequence[Mapping[str, Any]]) -> None:
    """Retain cutoff reservations while detecting unsafe or ambiguous marker changes."""

    for item in bindings:
        actual, _payloads = _job_state_binding(state_root, str(item["store"]))
        if actual["done"]["exists"] and actual["fail"]["exists"]:
            raise ValueError(f"job has both fail and done markers: {item['store']}")
        frozen = item["markers"]
        for suffix in ("claim", "done", "fail"):
            if frozen[suffix]["exists"] and actual[suffix] != frozen[suffix]:
                raise LiveSnapshotChanged(
                    f"worker state changed after physical proof: {item['store']}; retry cycle")
        if item["terminal"] and actual != frozen:
            raise LiveSnapshotChanged(
                f"terminal worker state changed after physical proof: {item['store']}; retry cycle")


def _recheck_queue_proof(dataset_root: Path, queue: Mapping[str, Any]) -> set[str]:
    """Recheck one ephemeral queue proof; appended successes remain reserved."""

    jobs_path = dataset_root / "jobs.json"
    if pb.file_sha256(jobs_path) != queue["jobs_sha256"]:
        raise LiveSnapshotChanged("queue changed after physical proof; retry")
    for proof in queue.get("input_proofs", []):
        _recheck_input_proof(proof)
    deferred = set()
    append_cache = (queue.setdefault("append_validation_cache", {})
                    if isinstance(queue, dict) else {})
    for proof in queue.get("pair_proofs", []):
        deferred.update(_recheck_pair_proof(proof, append_cache))
    for binding in queue.get("result_absence_bindings", []):
        if Path(binding["results_path"]).is_file() and binding["terminal"]:
            raise ValueError(
                f"terminal store results appeared after physical audit: {binding['store']}")
    for binding in queue.get("exclusion_bindings", []):
        root = Path(binding["input"])
        if (pb.file_sha256(root / "measurement.json") != binding["measurement_sha256"] or
                pb.file_sha256(root / "manifest.json") != binding["manifest_sha256"]):
            raise ValueError(f"reserved input changed after physical proof: {root}")
    _recheck_frozen_markers(dataset_root / "jobs_state", queue.get("state_bindings", []))
    return deferred


def _audit_from_queue(dataset_root: Path, cfg: Any, queue: Mapping[str, Any]) -> dict[str, Any]:
    proofs = queue.get("pair_proofs", [])
    times = [entry["time_s"] for proof in proofs
             for entry in _successful_entries(proof["results"]).values()]
    measurement_ids = sorted({measurement_id(cfg.measurement)} if queue["jobs"] else set())
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "scope": cfg.scope,
        "dataset_root": str(dataset_root), "jobs_sha256": queue["jobs_sha256"],
        "measurement_ids": measurement_ids,
        "valid_unique_patterns": len(queue["successful_hashes"]),
        "audited_ok_including_repeats": len(times),
        "solver_time_median_s": statistics.median(times) if times else None,
        "completed_stores": [proof["store"] for proof in proofs
                             if len(_successful_entries(proof["results"])) ==
                             len(proof["rows"])],
        "jobs": queue.get("audit_jobs", []),
        "progress_limit": "Snapshot of saved artifacts; claims do not prove a live worker heartbeat.",
    }


def _queue_view(dataset_root: Path, cfg: Any, policy: Mapping[str, Any],
                expected_retry_workers: Sequence[str] | None = None) -> dict[str, Any]:
    jobs_path = dataset_root / "jobs.json"
    state_root = dataset_root / "jobs_state"
    retry_workers = _normalize_retry_workers(expected_retry_workers)
    before = pb.file_sha256(jobs_path)
    jobs = _read(jobs_path)
    if not isinstance(jobs, list):
        raise ValueError("jobs.json must contain a list")
    wanted_mid = measurement_id(cfg.measurement)
    wanted_sid = score_spec_id(cfg.score_spec)
    relevant, pending_hashes, guided_pending = [], set(), set()
    successful_hashes: set[str] = set()
    pairs, pair_bindings, state_bindings = [], [], []
    input_proofs, pair_proofs, result_absence_bindings, audit_jobs = [], [], [], []
    retryable_failed_jobs, terminal_failed_jobs = [], []
    for job in jobs:
        if job.get("scope") != cfg.scope:
            continue
        input_dir = _safe_child(dataset_root, job["input"])
        if measurement_id(_read(input_dir / "measurement.json")) != wanted_mid:
            raise ValueError(f"same-scope job has wrong measurement profile: {input_dir}")
        input_hashes = {name: pb.file_sha256(input_dir / name) for name in
                        ("config.yaml", "measurement.json", "score_spec.json", "manifest.json")}
        input_cfg = pb.load_profile_config(input_dir / "config.yaml")
        if input_cfg is None:
            raise ValueError(f"queued input lacks a profiled config: {input_dir}")
        if (input_cfg.scope != cfg.scope or
                measurement_id(input_cfg.measurement) != wanted_mid or
                score_spec_id(input_cfg.score_spec) != wanted_sid):
            raise ValueError(f"same-scope job config differs from factory profile: {input_dir}")
        input_proof = _input_physical_proof(input_dir, input_cfg, input_hashes)
        rows = input_proof["rows"]
        input_proofs.append(input_proof)
        store = _safe_child(dataset_root, job["store"])
        marker_binding, marker_payloads = _job_state_binding(state_root, job["store"])
        fail = marker_binding["fail"]["exists"]
        done = marker_binding["done"]["exists"]
        if fail and done:
            raise ValueError(f"job has both fail and done markers: {job['store']}")
        terminal_fail = fail and _terminal_fail(marker_payloads["fail"], retry_workers)
        terminal = done or terminal_fail
        state_bindings.append({"store": job["store"], "markers": marker_binding,
                               "terminal": terminal})
        if fail:
            (terminal_failed_jobs if terminal_fail else retryable_failed_jobs).append(job["store"])
        if not terminal:
            hashes = {row["pattern_sha256"] for row in rows}
            pending_hashes.update(hashes)
            if int(job.get("prio", 9)) == int(policy["guided_priority"]):
                guided_pending.update(hashes)
        has_results = (store / "results.json").is_file()
        results: Mapping[str, Any] = {}
        store_hashes = None
        if has_results:
            store_hashes = {name: pb.file_sha256(store / name) for name in
                            ("measurement.json", "score_spec.json", "manifest.json", "results.json")}
            pairs.append((input_dir, store))
            results = pb.validate_store(store, require_complete=False)
            if (measurement_id(_read(store / "measurement.json")) !=
                    measurement_id(_read(input_dir / "measurement.json")) or
                    score_spec_id(_read(store / "score_spec.json")) !=
                    score_spec_id(_read(input_dir / "score_spec.json"))):
                raise ValueError(f"source store identity differs from input: {store}")
            if _read(store / "manifest.json") != rows:
                raise ValueError(f"source store manifest differs from input: {store}")
            changed = [name for name, digest in store_hashes.items()
                       if pb.file_sha256(store / name) != digest]
            if changed:
                error = f"source store changed during queue scan: {store}: {changed}"
                if set(changed) <= {"results.json"}:
                    raise LiveSnapshotChanged(error)
                raise ValueError(error)
            pair_bindings.append({"input": str(input_dir), "store": str(store),
                                  "input_metadata_sha256": input_hashes,
                                  "store_metadata_sha256": store_hashes})
            pair_proofs.append({
                "input": str(input_dir), "store": str(store), "rows": rows,
                "results": results, "terminal": terminal,
                "input_metadata_sha256": input_hashes,
                "input_pattern_file_sha256": input_proof["input_pattern_file_sha256"],
                "store_metadata_sha256": store_hashes,
            })
            successful_hashes.update(entry["pattern_sha256"] for entry in results.values()
                                     if isinstance(entry, dict) and entry.get("status") == "ok"
                                     and "error" not in entry)
        else:
            result_absence_bindings.append({
                "store": job["store"], "results_path": str(store / "results.json"),
                "terminal": terminal,
            })
        if any(pb.file_sha256(input_dir / name) != digest for name, digest in input_hashes.items()):
            raise ValueError(f"input changed during queue scan: {input_dir}")
        relevant.append({**job, "terminal": terminal})
        ok = _successful_entries(results)
        audit_markers = {}
        for suffix in ("claim", "done", "fail"):
            if not marker_binding[suffix]["exists"]:
                continue
            marker_path = state_root / f"{job['store']}.{suffix}"
            try:
                mtime = marker_path.stat().st_mtime
            except FileNotFoundError as exc:
                raise LiveSnapshotChanged(
                    f"worker state changed during controller scan: {job['store']}") from exc
            audit_markers[suffix] = {
                "mtime_utc": datetime.fromtimestamp(mtime, timezone.utc).isoformat(),
                "payload": marker_payloads[suffix],
            }
        audit_jobs.append({
            "store": job["store"], "input": job["input"], "prio": job["prio"],
            "expected": len(rows), "audited_ok": len(ok),
            "errors": {name: entry for name, entry in
                       results.items()
                       if isinstance(entry, dict) and "error" in entry},
            "markers": audit_markers,
            "input_metadata_sha256": input_hashes,
            "source_metadata_sha256": store_hashes,
        })

    # Every same-measurement input reserves its physical pattern forever,
    # including failed jobs; this is broader than the currently pending budget.
    exclusions = set()
    exclusion_bindings = []
    for input_dir in sorted(dataset_root.glob("*_input")):
        marker = input_dir / "measurement.json"
        manifest = input_dir / "manifest.json"
        if marker.is_file() and manifest.is_file() and measurement_id(_read(marker)) == wanted_mid:
            marker_hash, manifest_hash = pb.file_sha256(marker), pb.file_sha256(manifest)
            rows = _read(manifest)
            exclusions.update(row["pattern_sha256"] for row in rows)
            exclusion_bindings.append({"input": str(input_dir.resolve()),
                                       "measurement_sha256": marker_hash,
                                       "manifest_sha256": manifest_hash})
            if pb.file_sha256(marker) != marker_hash or pb.file_sha256(manifest) != manifest_hash:
                raise ValueError(f"reserved input changed during exclusion scan: {input_dir}")
    if pb.file_sha256(jobs_path) != before:
        raise LiveSnapshotChanged("queue changed during controller scan; retry")
    _require_state_bindings(state_root, state_bindings)
    return {"jobs_sha256": before, "all_jobs": jobs, "jobs": relevant, "pairs": pairs,
            "pair_bindings": pair_bindings,
            "pending_hashes": pending_hashes, "guided_pending_hashes": guided_pending,
            "successful_hashes": successful_hashes, "exclusion_hashes": exclusions,
            "expected_retry_workers": list(retry_workers),
            "state_bindings": state_bindings,
            "input_proofs": input_proofs, "pair_proofs": pair_proofs,
            "result_absence_bindings": result_absence_bindings,
            "exclusion_bindings": exclusion_bindings, "audit_jobs": audit_jobs,
            "retryable_failed_jobs": retryable_failed_jobs,
            "terminal_failed_jobs": terminal_failed_jobs}


def _pilot_call(operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        return operation(*args, **kwargs)
    except shell_pilot.PilotSnapshotChanged as exc:
        raise LiveSnapshotChanged(str(exc)) from exc


def _queued_pilot_evidence(dataset_root: Path, queue: Mapping[str, Any],
                           request: Any, scope: str) -> dict[str, Any] | None:
    """Return the sole complete queued cohort; reject partial/conflicting evidence."""

    matches = []
    for job in queue["jobs"]:
        input_dir = _safe_child(dataset_root, str(job["input"]))
        rows = _read(input_dir / "manifest.json")
        if not any(row.get("pilot_id") == request.pilot_id for row in rows):
            continue
        store = f"dedust_r80p{request.pilot_id[:8]}p01"
        expected_job = {"input": store + "_input", "store": store,
                        "prio": int(request.priority), "scope": scope,
                        "config": f"{store}_input/config.yaml"}
        if any(job.get(key) != value for key, value in expected_job.items()):
            raise ValueError("queued pilot evidence has a noncanonical queue record")
        try:
            proof = shell_pilot.validate_bundle(input_dir, request)
        except shell_pilot.PilotError as exc:
            raise ValueError(
                f"queued pilot evidence is partial or conflicting: {input_dir}: {exc}") from exc
        matches.append({"input": str(input_dir), "store": job["store"],
                        "job": dict(job), "proof": proof})
    if len(matches) > 1:
        raise ValueError("multiple queued cohorts claim the same one-shot pilot_id")
    return matches[0] if matches else None


def _recover_local_pilot_preparation(local: Path, request: Any) -> dict[str, Any] | None:
    """Recover the sole completed request-bound bundle/reference across cycle IDs."""

    cycles = local / "cycles"
    if not cycles.is_dir():
        return None
    candidates = []
    for cycle_dir in sorted(path for path in cycles.iterdir() if path.is_dir()):
        reference = cycle_dir / "pilot_reference.json"
        bundle = cycle_dir / "pilot_bundle"
        reference_payload = _read(reference) if reference.is_file() else None
        audit_path = bundle / "pilot_audit.json"
        audit_payload = _read(audit_path) if audit_path.is_file() else None
        reference_matches = (isinstance(reference_payload, dict) and
                             reference_payload.get("pilot_id") == request.pilot_id)
        audit_matches = (isinstance(audit_payload, dict) and
                         audit_payload.get("pilot_id") == request.pilot_id)
        if not reference_matches and not audit_matches:
            continue
        if not reference_matches or not bundle.is_dir():
            raise ValueError("local pilot preparation is partial or conflicting")
        if reference_payload.get("request_binding") != request.binding:
            raise ValueError("local pilot reference has a different request binding")
        proof = _pilot_call(shell_pilot.validate_bundle, bundle, request)
        _pilot_call(shell_pilot.recheck_reference, reference)
        candidates.append({"bundle": bundle.resolve(), "reference": reference.resolve(),
                           "proof": proof})
    if len(candidates) > 1:
        raise ValueError("multiple completed local preparations claim the same pilot_id")
    return candidates[0] if candidates else None


def _require_pair_bindings(bindings: Sequence[Mapping[str, Any]]) -> None:
    for binding in bindings:
        for key, label in (("input_metadata_sha256", "input"),
                           ("store_metadata_sha256", "store")):
            root = Path(binding[label])
            changed = [name for name, digest in binding[key].items()
                       if pb.file_sha256(root / name) != digest]
            if changed:
                error = f"{label} changed after physical audit; retry cycle: {root}"
                if label == "store" and set(changed) <= {"results.json"}:
                    raise LiveSnapshotChanged(error)
                raise ValueError(f"{error}: {changed}")


def _success_order(pairs: Sequence[tuple[Path, Path]], excluded: set[str]) -> list[str]:
    choices: dict[str, tuple[tuple[bool, int, int], str]] = {}
    for pair_index, (input_dir, store) in enumerate(pairs):
        rows = _read(input_dir / "manifest.json")
        results = pb.validate_store(store, require_complete=False)
        for row_index, row in enumerate(rows):
            entry = results.get(row["id"])
            digest = row["pattern_sha256"]
            if (digest in excluded or not isinstance(entry, dict) or
                    entry.get("status") != "ok" or "error" in entry):
                continue
            priority = (row.get("kind") in ("repeat", "notarize"), pair_index, row_index)
            if digest not in choices or priority < choices[digest][0]:
                choices[digest] = (priority, digest)
    return [item[1] for item in sorted(choices.values())]


def _proof_success_candidates(queue: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates = []
    for pair_index, proof in enumerate(queue.get("pair_proofs", [])):
        for row_index, row in enumerate(proof["rows"]):
            entry = proof["results"].get(row["id"])
            if (not isinstance(entry, dict) or entry.get("status") != "ok" or
                    "error" in entry):
                continue
            candidates.append({
                "pair_index": pair_index, "row_index": row_index,
                "input": Path(proof["input"]), "store": Path(proof["store"]),
                "row": row, "entry": entry, "proof": proof,
                "repeat": row.get("kind") in ("repeat", "notarize"),
                "pattern_sha256": row["pattern_sha256"],
            })
    return candidates


def _success_order_from_proofs(queue: Mapping[str, Any], excluded: set[str]) -> list[str]:
    choices: dict[str, tuple[tuple[bool, int, int], str]] = {}
    for item in _proof_success_candidates(queue):
        digest = item["pattern_sha256"]
        if digest in excluded:
            continue
        priority = (item["repeat"], item["pair_index"], item["row_index"])
        if digest not in choices or priority < choices[digest][0]:
            choices[digest] = (priority, digest)
    return [item[1] for item in sorted(choices.values())]


def _copy_proof_artifact(source: Path, destination: Path, digest: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if pb.file_sha256(source) != digest:
        raise ValueError(f"frozen successful artifact changed: {source}")
    shutil.copy2(source, destination)
    if pb.file_sha256(destination) != digest or pb.file_sha256(source) != digest:
        raise ValueError(f"frozen successful artifact changed during copy: {source}")


def _snapshot_successes_from_proofs(queue: Mapping[str, Any], output: Path,
                                    selected_hashes: Sequence[str], cfg: Any) -> Path:
    """Publish an exact cutoff snapshot without rediscovering live representatives."""

    selected = list(selected_hashes)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("proof snapshot requires nonempty unique pattern hashes")
    winners: dict[str, tuple[tuple[bool, int, int], dict[str, Any]]] = {}
    for item in _proof_success_candidates(queue):
        digest = item["pattern_sha256"]
        if digest not in set(selected):
            continue
        priority = (item["repeat"], item["pair_index"], item["row_index"])
        if digest not in winners or priority < winners[digest][0]:
            winners[digest] = (priority, item)
    if set(winners) != set(selected):
        raise ValueError("proof snapshot selection is absent from the physical cutoff")
    chosen = [winners[digest][1] for digest in selected]
    ids = [item["row"]["id"] for item in chosen]
    if len(set(ids)) != len(ids):
        raise ValueError("selected proof successes contain an id collision")
    rows = [item["row"] for item in chosen]
    results = {item["row"]["id"]: item["entry"] for item in chosen}
    sources = [{
        "input": proof["input"], "store": proof["store"],
        "input_metadata_sha256": proof["input_metadata_sha256"],
        "store_metadata_sha256": proof["store_metadata_sha256"],
        "partial_source_results_sha256": proof["store_metadata_sha256"]["results.json"],
        "cutoff_manifest_count": len(proof["rows"]),
        "cutoff_successful_count": len(_successful_entries(proof["results"])),
    } for proof in queue["pair_proofs"]]
    items = {}
    for item in chosen:
        row, entry, proof = item["row"], item["entry"], item["proof"]
        binding = {
            "source_input": proof["input"], "source_store": proof["store"],
            "source_manifest_sha256": proof["store_metadata_sha256"]["manifest.json"],
            "partial_source_results_sha256": proof["store_metadata_sha256"]["results.json"],
            "cutoff_result_entry": entry,
            "cutoff_result_entry_id": _content_id(entry),
            "source_sample_file": entry["sample_file"],
            "source_sample_sha256": entry["sample_sha256"],
        }
        if entry.get("rad_file") is not None:
            binding.update({"source_rad_file": entry["rad_file"],
                            "source_rad_sha256": entry["rad_sha256"]})
        items[row["id"]] = binding
    source_bindings = {
        "schema_version": 2, "partial_snapshot": True,
        "source_batches_may_be_incomplete": True,
        "cutoff_kind": "same_cycle_ephemeral_physical_proof",
        "cutoff_id": _content_id(queue["pair_proofs"]),
        "measurement_id": measurement_id(cfg.measurement),
        "score_spec_id": score_spec_id(cfg.score_spec),
        "selected_pattern_sha256": selected, "sources": sources,
        "items": {name: items[name] for name in ids},
    }

    def validate_existing(root: Path) -> None:
        pb.validate_store(root, require_complete=True)
        if (_read(root / "manifest.json") != rows or
                _read(root / "results.json") != results):
            raise ValueError("existing proof snapshot differs from exact cutoff selection")
        if _read(root / "source_bindings.json") != source_bindings:
            raise ValueError("existing proof snapshot has a different physical cutoff")

    output = output.resolve()
    if output.exists():
        validate_existing(output)
        return output
    pending = output.with_name(f".{output.name}.pending")
    if pending.exists():
        resolved = pending.resolve()
        if resolved.parent != output.parent or resolved.name != f".{output.name}.pending":
            raise RuntimeError(f"refusing unsafe pending cleanup: {resolved}")
        shutil.rmtree(resolved)
    pending.mkdir(parents=True)
    try:
        base = Path(queue["pair_proofs"][0]["input"])
        for name in ("measurement.json", "score_spec.json", "config.yaml"):
            _copy_proof_artifact(
                base / name, pending / name,
                queue["pair_proofs"][0]["input_metadata_sha256"][name])
        pb.atomic_json(pending / "manifest.json", rows)
        for item in chosen:
            row, entry, proof = item["row"], item["entry"], item["proof"]
            store = Path(proof["store"])
            sample = _safe_store_file(store, entry.get("sample_file"), f"{row['id']} sample")
            _copy_proof_artifact(sample, pending / entry["sample_file"], entry["sample_sha256"])
            if entry.get("rad_file") is not None:
                radiation = _safe_store_file(
                    store, entry["rad_file"], f"{row['id']} radiation")
                _copy_proof_artifact(
                    radiation, pending / entry["rad_file"], entry["rad_sha256"])
        pb.atomic_json(pending / "results.json", results)
        pb.atomic_json(pending / "source_bindings.json", source_bindings)
        pb.validate_store(pending, require_complete=True)
        if output.exists():
            raise FileExistsError(f"proof snapshot appeared during publication: {output}")
        os.replace(pending, output)
    except Exception:
        if pending.exists():
            shutil.rmtree(pending)
        raise
    validate_existing(output)
    return output


def _seed_rows(inputs: Sequence[str | Path]) -> list[dict[str, Any]]:
    rows = []
    for supplied in inputs:
        path = Path(supplied).resolve()
        if path.is_dir():
            for row in _read(path / "manifest.json"):
                value = dict(row)
                value["pattern"] = torch.load(path / f"{row['id']}.pt", weights_only=True,
                                               map_location="cpu")
                rows.append(value)
        else:
            rows.extend(sm.load_seed_rows(path))
    if not rows:
        raise ValueError("seed_inputs must provide at least one pattern")
    return rows


def _blind_subset(source: Path, output: Path, count: int, excluded: set[str],
                  expected_cfg: Any) -> Path:
    cfg = pb.load_profile_config(source / "config.yaml")
    if (cfg is None or measurement_id(cfg.measurement) != measurement_id(expected_cfg.measurement) or
            score_spec_id(cfg.score_spec) != score_spec_id(expected_cfg.score_spec) or
            cfg.scope != expected_cfg.scope):
        raise ValueError("prepared blind pool profile differs from factory profile")
    rows = pb.validate_input(source, cfg)
    selected = [row for row in rows if row["pattern_sha256"] not in excluded][:count]
    if len(selected) != count:
        raise ValueError(f"prepared blind pool has only {len(selected)}/{count} unreserved patterns")
    def produce(destination: Path) -> None:
        destination.mkdir(parents=True)
        for name in ("config.yaml", "measurement.json", "score_spec.json"):
            shutil.copy2(source / name, destination / name)
        for row in selected:
            shutil.copy2(source / f"{row['id']}.pt", destination / f"{row['id']}.pt")
        pb.atomic_json(destination / "manifest.json", selected)

    _publish_bundle(output, expected_cfg, produce)
    if [row["pattern_sha256"] for row in _read(output / "manifest.json")] != [
            row["pattern_sha256"] for row in selected]:
        raise ValueError("existing blind staging bundle differs")
    return output


def _planned(shards: Sequence[Path], kind: str, priority: int, cycle_id: str,
             start: int = 1) -> list[dict[str, Any]]:
    jobs = []
    for ordinal, shard in enumerate(shards, start=start):
        store = f"dedust_r80c{cycle_id[:8]}{kind}{ordinal:02d}"
        jobs.append({"kind": kind, "staged_input": str(shard.resolve()),
                     "input": store + "_input", "store": store, "prio": int(priority),
                     "count": len(_read(shard / "manifest.json")),
                     "tree_sha256": _content_id(_tree_hashes(shard))})
    return jobs


def planned_receipt_hashes(receipt_path: str | Path) -> set[str]:
    """Return exact staged hashes protected between main preparation and dispatch."""

    receipt = _read(Path(receipt_path).resolve())
    hashes: set[str] = set()
    for planned in receipt.get("planned_jobs", []):
        staged = Path(planned["staged_input"]).resolve()
        if _content_id(_tree_hashes(staged)) != planned.get("tree_sha256"):
            raise ValueError(f"staged shard changed while reserving it: {staged}")
        rows = _read(staged / "manifest.json")
        values = {str(row["pattern_sha256"]) for row in rows}
        if len(values) != len(rows) or hashes & values:
            raise ValueError("prepared receipt contains duplicate pattern reservations")
        hashes.update(values)
    return hashes


def _split_or_validate(bundle: Path, output: Path, shard_size: int, cfg: Any) -> list[Path]:
    if not output.exists():
        return split_bundle(bundle, output, shard_size)
    receipt = _read(output / "split_receipt.json")
    canonical_rows = pb.validate_input(bundle, cfg)
    shard_count = (len(canonical_rows) + shard_size - 1) // shard_size
    assignments = [canonical_rows[index::shard_count] for index in range(shard_count)]
    expected_shards = []
    shards = []
    for number, group in enumerate(assignments, start=1):
        name = f"shard-{number:03d}_input"
        shard = output / name
        _validate_bundle(shard, cfg)
        if _read(shard / "manifest.json") != group:
            raise ValueError(f"existing shard differs from canonical round-robin split: {shard}")
        items = []
        expected_files = {"config.yaml", "measurement.json", "score_spec.json", "manifest.json"}
        for metadata_name in ("config.yaml", "measurement.json", "score_spec.json"):
            if pb.file_sha256(shard / metadata_name) != pb.file_sha256(bundle / metadata_name):
                raise ValueError(
                    f"existing shard metadata differs from canonical bundle: {shard}")
        for row in group:
            pattern_file = row.get("pattern_file", f"{row['id']}.pt")
            expected_files.add(pattern_file)
            canonical_sha = pb.file_sha256(bundle / pattern_file)
            if pb.file_sha256(shard / pattern_file) != canonical_sha:
                raise ValueError(f"existing shard pattern differs from canonical bundle: {shard}")
            items.append({"id": row["id"], "pattern_file": pattern_file,
                          "file_sha256": canonical_sha,
                          "pattern_sha256": row["pattern_sha256"]})
        actual_entries = {path.name for path in shard.iterdir()}
        if actual_entries != expected_files or any(
                not path.is_file() or path.is_symlink() for path in shard.iterdir()):
            raise ValueError(f"existing shard has missing or extra files: {shard}")
        expected_shards.append({
            "name": name, "ids": [row["id"] for row in group], "items": items,
            "manifest_sha256": pb.file_sha256(shard / "manifest.json"),
        })
        shards.append(shard)
    expected_receipt = {
        "schema_version": 1,
        "canonical_input": str(bundle.resolve()),
        "canonical_manifest_sha256": pb.file_sha256(bundle / "manifest.json"),
        "canonical_metadata_sha256": {
            name: pb.file_sha256(bundle / name) for name in
            ("measurement.json", "score_spec.json", "config.yaml", "manifest.json")},
        "config_sha256": pb.file_sha256(bundle / "config.yaml"),
        "measurement_id": measurement_id(_read(bundle / "measurement.json")),
        "score_spec_id": score_spec_id(_read(bundle / "score_spec.json")),
        "round_robin": True, "shard_size": shard_size, "shards": expected_shards,
    }
    if receipt != expected_receipt:
        raise ValueError(f"existing shard split receipt differs from canonical replay: {output}")
    actual_entries = {path.name for path in output.iterdir()}
    if actual_entries != {"split_receipt.json", *(entry["name"] for entry in expected_shards)}:
        raise ValueError(f"existing shard split has missing or extra children: {output}")
    return shards


def _run_once_unlocked(*, local_workdir: str | Path, dataset_root: str | Path,
                       profile_config: str | Path, training_workdir: str | Path,
                       seed_inputs: Sequence[str | Path],
                       prepared_blind_pool: str | Path | None = None,
                       cold_start_predictor: Any | None = None,
                       expected_retry_workers: Sequence[str] | None = None,
                       pilot_request: str | Path | None = None,
                       planning_guard: _PlanningGuard | None = None) -> Path:
    """Prepare one recoverable cycle and return its reviewable action receipt."""

    local = Path(local_workdir).resolve(); dataset = Path(dataset_root).resolve()
    profile = Path(profile_config).resolve(); train_root = Path(training_workdir).resolve()
    retry_workers = _normalize_retry_workers(expected_retry_workers)
    pilot, pilot_binding = _load_pilot_request(pilot_request, profile)
    local.mkdir(parents=True, exist_ok=True)
    active = local / "active_cycle.json"
    recovered_training_receipt = None
    recovered_prediction_audit = None
    recovered_training_snapshot = None
    recovered_training_path = None
    recovered_cycle_id = None
    if active.is_file():
        active_payload = _read(active)
        receipt = Path(active_payload["receipt"]).resolve()
        receipt.relative_to(local)
        if receipt.is_file():
            prior = _read(receipt)
            if prior.get("status") in {"prepared", "dispatching", "training"}:
                expected = (str(dataset), str(profile), str(train_root), list(retry_workers))
                actual = (prior.get("dataset_root"), prior.get("profile_config"),
                          prior.get("training_workdir"), prior.get("expected_retry_workers"))
                if actual != expected:
                    raise ValueError("active cycle belongs to different explicit inputs")
                prior_pilot = prior.get("pilot_request")
                if prior_pilot != pilot_binding or (
                        pilot_binding is None and "pilot_request" in prior):
                    raise ValueError("active cycle belongs to different pilot request binding")
            if prior.get("status") == "training":
                recovered_training_receipt = str(receipt)
                recovered_prediction_audit = prior.get("pretrain_prediction_audit")
                recovered_training_snapshot = prior.get("training_snapshot")
                recovered_training_path = receipt
                recovered_cycle_id = prior.get("cycle_id")
            elif prior.get("status") in {"prepared", "dispatching"}:
                return receipt

    cfg, policy = _load_profile(profile)
    _protocol, protocol_binding = _protocol_binding(train_root)
    version, trained_hashes, trained_receipt = _trained(train_root)
    seed_bindings = [_path_binding(Path(value)) for value in seed_inputs]
    blind_binding = (_path_binding(Path(prepared_blind_pool))
                     if prepared_blind_pool is not None else None)
    cold_binding = list(getattr(cold_start_predictor, "model_ids", ()))
    profile_sha256 = pb.file_sha256(profile)
    with _dataset_exclusive_lock(dataset, cfg.scope):
        queue = _queue_view(dataset, cfg, policy, retry_workers)
        if "pair_proofs" in queue:
            audit = _audit_from_queue(dataset, cfg, queue)
            _recheck_queue_proof(dataset, queue)
        else:  # Compatibility for bounded callers that provide a synthetic queue view.
            audit = _physical_snapshot(dataset, cfg.scope)
            if audit["jobs_sha256"] != queue["jobs_sha256"]:
                raise LiveSnapshotChanged(
                    "queue changed between physical audit and controller scan; retry")
            _require_pair_bindings(queue.get("pair_bindings", []))
    initial_valid = len(queue["successful_hashes"])
    initial_pending = len(queue["pending_hashes"] - queue["successful_hashes"])
    if initial_valid > int(policy["target_valid_unique"]):
        raise ValueError("audited valid unique exceeds target")
    if initial_valid + initial_pending > int(policy["target_valid_unique"]):
        raise ValueError("audited valid plus pending unique exceeds target")
    source_binding = queue["pair_bindings"]
    training_source_binding = source_binding
    cycle_identity = {"jobs": audit["jobs_sha256"], "sources": source_binding,
                      "trained": trained_receipt, "profile": profile_sha256,
                      "training_protocol": protocol_binding,
                      "seeds": seed_bindings, "blind": blind_binding,
                      "cold_predictor": cold_binding,
                      "expected_retry_workers": retry_workers}
    if pilot_binding is not None:
        cycle_identity["pilot_request"] = pilot_binding
    cycle_id = _content_id(cycle_identity)
    if recovered_training_path is not None:
        if (not isinstance(recovered_cycle_id, str) or
                recovered_training_path.parent.parent != (local / "cycles").resolve()):
            raise ValueError("active training receipt has an invalid local cycle binding")
        cycle_id = recovered_cycle_id
        cycle = recovered_training_path.parent
        receipt_path = recovered_training_path
    else:
        cycle = local / "cycles" / cycle_id
        receipt_path = cycle / "action_receipt.json"
    cycle.mkdir(parents=True, exist_ok=True)
    _write(active, {"cycle_id": cycle_id, "receipt": str(receipt_path)})

    predictor = None
    trained_this_cycle = False
    prediction_audit_binding = recovered_prediction_audit
    training_snapshot_binding = recovered_training_snapshot
    if "pair_proofs" in queue:
        _recheck_queue_proof(dataset, queue)
        new_order = _success_order_from_proofs(queue, trained_hashes)
        _recheck_queue_proof(dataset, queue)
    else:
        _require_pair_bindings(queue.get("pair_bindings", []))
        new_order = _success_order(queue["pairs"], trained_hashes)
        _require_pair_bindings(queue.get("pair_bindings", []))
    recovering_training = recovered_training_path is not None
    if recovering_training or len(new_order) >= int(policy["update_every_valid_unique"]):
        if recovering_training:
            if not isinstance(recovered_training_snapshot, dict):
                raise ValueError("active training receipt lacks its immutable snapshot binding")
            frozen = Path(recovered_training_snapshot.get("path", "")).resolve()
            if frozen != (cycle / "training_snapshot").resolve() or not frozen.is_dir():
                raise ValueError("active training snapshot path differs from its cycle")
            frozen_rows = _read(frozen / "manifest.json")
            take = len(frozen_rows)
            if not int(policy["update_every_valid_unique"]) <= take <= 96:
                raise ValueError("active training snapshot has an invalid frozen count")
            actual_snapshot = {
                "path": str(frozen), "count": take,
                "manifest_sha256": pb.file_sha256(frozen / "manifest.json"),
                "results_sha256": pb.file_sha256(frozen / "results.json"),
                "source_bindings_sha256": pb.file_sha256(frozen / "source_bindings.json"),
            }
            if actual_snapshot != recovered_training_snapshot:
                raise ValueError("active training snapshot bytes differ from its receipt")
            pb.validate_store(frozen, require_complete=True)
        else:
            take = min(96, len(new_order))
            frozen = cycle / "training_snapshot"
        omit = set(new_order[take:])
        if not frozen.exists():
            if "pair_proofs" in queue:
                _recheck_queue_proof(dataset, queue)
                _snapshot_successes_from_proofs(queue, frozen, new_order[:take], cfg)
            else:
                try:
                    snapshot_successes(queue["pairs"], frozen,
                                       exclude_pattern_hashes=trained_hashes | omit)
                except ValueError as exc:
                    message = str(exc)
                    if ("source store " in message and
                            "metadata changed during operation: ['results.json']" in message):
                        raise LiveSnapshotChanged(message) from exc
                    raise
        if "pair_proofs" in queue:
            _recheck_queue_proof(dataset, queue)
        else:
            _require_pair_bindings(queue.get("pair_bindings", []))
        frozen_rows = _read(frozen / "manifest.json")
        frozen_hashes = [row["pattern_sha256"] for row in frozen_rows]
        if (not recovering_training and
                (len(frozen_hashes) != take or set(frozen_hashes) != set(new_order[:take]))):
            raise ValueError("frozen training snapshot differs from exact audited selection")
        frozen_binding = _read(frozen / "source_bindings.json")
        if not recovering_training:
            expected_results = {
                binding["store"]: binding["store_metadata_sha256"]["results.json"]
                for binding in queue["pair_bindings"]}
            actual_results = {item["store"]: item["partial_source_results_sha256"]
                              for item in frozen_binding["sources"]}
            if actual_results != expected_results:
                raise LiveSnapshotChanged(
                    "frozen training snapshot source hashes differ from physical audit")
        if (not recovering_training and "pair_proofs" in queue and
                frozen_binding.get("cutoff_id") != _content_id(queue["pair_proofs"])):
            raise ValueError("frozen training snapshot differs from physical cutoff proof")
        training_snapshot_binding = {
            "path": str(frozen), "count": len(frozen_rows),
            "manifest_sha256": pb.file_sha256(frozen / "manifest.json"),
            "results_sha256": pb.file_sha256(frozen / "results.json"),
            "source_bindings_sha256": pb.file_sha256(frozen / "source_bindings.json"),
        }
        prediction_audit_path = audit_frozen_predictions(
            frozen, train_root, cycle / "pretrain_prediction_audit.json")
        prediction_audit_binding = {"path": str(prediction_audit_path),
                                    "sha256": pb.file_sha256(prediction_audit_path)}
        if (recovering_training and
                prediction_audit_binding != recovered_prediction_audit):
            raise ValueError("active pretraining prediction audit differs from its receipt")
        training_receipt = {
            "schema_version": 1, "status": "training", "once": True,
            "cycle_id": cycle_id, "dataset_root": str(dataset), "scope": cfg.scope,
            "profile_config": str(profile), "training_workdir": str(train_root),
            "profile_sha256": profile_sha256,
            "expected_retry_workers": list(retry_workers),
            "training_protocol": protocol_binding,
            "audited_source_bindings": training_source_binding,
            "pretrain_prediction_audit": prediction_audit_binding,
            "training_snapshot": training_snapshot_binding,
            "quality_gate": None,
            "all_valid_observations_retained_for_training": True,
        }
        if pilot_binding is not None:
            training_receipt["pilot_request"] = pilot_binding
        if recovering_training:
            existing_training_receipt = _read(receipt_path)
            if (existing_training_receipt.get("profile_sha256") != profile_sha256 or
                    existing_training_receipt.get("training_protocol") != protocol_binding or
                    existing_training_receipt.get("training_snapshot") !=
                    training_snapshot_binding or
                    existing_training_receipt.get("pretrain_prediction_audit") !=
                    prediction_audit_binding):
                raise ValueError("active training receipt binding changed before recovery")
        else:
            _write(receipt_path, training_receipt)
        already_ingested = set(frozen_hashes) <= trained_hashes
        if recovering_training and already_ingested:
            if version <= 0:
                raise ValueError("active training snapshot is bound as ingested without a version")
        else:
            if recovering_training and set(frozen_hashes) & trained_hashes:
                raise ValueError("active training snapshot is only partially present in training state")
            version_dir = training.ingest_profile_stores(
                train_root, [frozen], min_new=int(policy["update_every_valid_unique"]), max_new=96)
            version = int(version_dir.name.removeprefix("data-v"))
        _train_version_with_guard(train_root, version, planning_guard)
        predictor = training.load_current_predictor(train_root, version)
        trained_this_cycle = True
    elif version > 0:
        try:
            predictor = training.load_current_predictor(train_root, version)
        except FileNotFoundError:
            _train_version_with_guard(train_root, version, planning_guard)
            predictor = training.load_current_predictor(train_root, version)
            trained_this_cycle = True
    elif predictor is None and cold_start_predictor is not None:
        if any(int(job.get("prio", 9)) == int(policy["guided_priority"])
               for job in queue["jobs"]):
            raise ValueError("historical cold start cannot be reused after guided jobs exist")
        predictor = cold_start_predictor

    # Actual training can outlive the cutoff, so it starts one fresh proof.  All
    # other work retains the original capacity/exclusion cutoff and only
    # rechecks its bytes; late completions cannot release reservations.
    if trained_this_cycle:
        with _dataset_exclusive_lock(dataset, cfg.scope):
            queue = _queue_view(dataset, cfg, policy, retry_workers)
            if "pair_proofs" in queue:
                audit = _audit_from_queue(dataset, cfg, queue)
                _recheck_queue_proof(dataset, queue)
            else:
                audit = _physical_snapshot(dataset, cfg.scope)
                if audit["jobs_sha256"] != queue["jobs_sha256"]:
                    raise LiveSnapshotChanged(
                        "queue changed between final physical audit and controller scan; retry")
                _require_pair_bindings(queue.get("pair_bindings", []))
        source_binding = queue["pair_bindings"]
    elif "pair_proofs" in queue:
        _recheck_queue_proof(dataset, queue)

    valid_unique = len(queue["successful_hashes"])
    if valid_unique > int(policy["target_valid_unique"]):
        raise ValueError("audited valid unique exceeds target")
    pending = queue["pending_hashes"] - queue["successful_hashes"]
    guided_pending = queue["guided_pending_hashes"] - queue["successful_hashes"]
    if valid_unique + len(pending) > int(policy["target_valid_unique"]):
        raise ValueError("audited valid plus pending unique exceeds target")
    remaining_budget = max(0, int(policy["target_valid_unique"]) - valid_unique - len(pending))
    planned_jobs: list[dict[str, Any]] = []
    planned_hashes: set[str] = set()
    pilot_receipt = None
    pilot_blocks_ordinary = False
    if pilot is not None:
        pilot = _require_pilot_request_binding(pilot_request, profile, pilot_binding)
        queued_pilot = _queued_pilot_evidence(dataset, queue, pilot, cfg.scope)
        terminal_decision = _existing_pilot_terminal_decision(local, pilot_binding)
        if queued_pilot is not None and terminal_decision is not None:
            raise ValueError("pilot cannot be both queued and terminal-unavailable")
        recovered_pilot = (None if queued_pilot is not None else
                           _recover_local_pilot_preparation(local, pilot))
        if recovered_pilot is not None and terminal_decision is not None:
            raise ValueError("pilot cannot be both locally prepared and terminal-unavailable")
        current_predictor = (predictor is not None and
                             getattr(predictor, "model_role", None) == "current_profile")
        guided_capacity = int(policy["max_guided_outstanding"]) - len(guided_pending)
        if queued_pilot is None and terminal_decision is None:
            if recovered_pilot is not None and remaining_budget < int(pilot.count):
                raise ValueError("completed local pilot preparation no longer fits exact target")
            if recovered_pilot is None and remaining_budget < int(pilot.count):
                terminal_decision = _pilot_terminal_decision(
                    local, pilot_binding, code="final_remaining_below_16",
                    detail=(f"exact target has only {remaining_budget}/{pilot.count} "
                            "unreserved slots; legacy final-tail planning resumes"))
            elif recovered_pilot is None and not current_predictor:
                # A historical cold-start model is never valid annotation provenance.
                # Keep the request pending while the exact ordinary planner proceeds.
                pilot_receipt = {"state": "waiting_current_profile_predictor",
                                 "request": pilot_binding}
            elif guided_capacity < int(pilot.count):
                pilot_blocks_ordinary = True
                pilot_receipt = {"state": "waiting_guided_capacity",
                                 "request": pilot_binding,
                                 "guided_capacity": guided_capacity,
                                 "required_capacity": int(pilot.count),
                                 "completed_preparation_recovered": recovered_pilot is not None}
            else:
                if recovered_pilot is not None:
                    bundle = recovered_pilot["bundle"]
                    reference = recovered_pilot["reference"]
                    proof = recovered_pilot["proof"]
                else:
                    try:
                        selection = _pilot_call(
                            shell_pilot.select_rows, pilot, set(queue["exclusion_hashes"]))
                    except shell_pilot.PilotUnavailable as exc:
                        terminal_decision = _pilot_terminal_decision(
                            local, pilot_binding, code=exc.code, detail=str(exc))
                        selection = None
                    if selection is not None:
                        if "pair_proofs" in queue:
                            _recheck_queue_proof(dataset, queue)
                            reference = _pilot_call(
                                shell_pilot.freeze_reference, queue["pairs"],
                                queue["pair_bindings"], pilot,
                                cycle / "pilot_reference.json",
                                validated_cutoff=queue["pair_proofs"])
                            _recheck_queue_proof(dataset, queue)
                        else:
                            reference = _pilot_call(
                                shell_pilot.freeze_reference, queue["pairs"],
                                queue["pair_bindings"], pilot,
                                cycle / "pilot_reference.json")
                        annotated = _pilot_call(
                            shell_pilot.annotate_selection, pilot, selection, predictor)
                        bundle = _pilot_call(
                            shell_pilot.write_pilot_bundle, pilot, annotated,
                            cycle / "pilot_bundle")
                        proof = _pilot_call(shell_pilot.validate_bundle, bundle, pilot)
                if terminal_decision is None:
                    hashes = tuple(proof["ordered_hashes"])
                    if (len(hashes) != int(pilot.count) or len(set(hashes)) != len(hashes) or
                            set(hashes) & set(queue["exclusion_hashes"])):
                        raise ValueError("prepared pilot cohort violates exact count or exclusions")
                    job = _planned([bundle], "p", int(pilot.priority), cycle_id)[0]
                    store = f"dedust_r80p{pilot.pilot_id[:8]}p01"
                    job.update({"input": store + "_input", "store": store,
                                "pilot_id": pilot.pilot_id,
                                "pilot_role": "incumbent_shell"})
                    planned_jobs.append(job)
                    planned_hashes.update(hashes)
                    pilot_blocks_ordinary = True
                    pilot_receipt = {
                        "state": "prepared", "request": pilot_binding,
                        "pilot_id": pilot.pilot_id,
                        "selection": {"ordered_ids": proof["ordered_ids"],
                                      "ordered_hashes": list(hashes),
                                      "role_counts": proof["role_counts"]},
                        "preparation_queue": {
                            "jobs_sha256": queue["jobs_sha256"],
                            "exclusion_count": len(queue["exclusion_hashes"]),
                            "exclusion_sha256": _content_id(
                                sorted(queue["exclusion_hashes"])),
                            "valid_unique": valid_unique,
                            "pending_unique": len(pending)},
                        "reference": _path_binding(reference),
                        "bundle": {"path": str(bundle.resolve()),
                                   "tree_sha256": _content_id(_tree_hashes(bundle)),
                                   "proof": proof},
                        "predictor_binding": dict(proof["predictor_binding"]),
                        "model_ids": list(proof["model_ids"]),
                        "surrogate_generation": proof["surrogate_generation"],
                        "valid_observations_at_fit": proof["valid_observations_at_fit"],
                        "completed_preparation_recovered": recovered_pilot is not None,
                        "one_shot": True, "replacement_after_freeze": False,
                    }

    guided_replan_receipt = None
    guided_replan_context = None
    guided_plan_identity = None
    if not pilot_blocks_ordinary and predictor is not None and remaining_budget > 0:
        shard_size = int(policy["shard_size"])
        capacity = min(int(policy["wave_size"]),
                       int(policy["max_guided_outstanding"]) - len(guided_pending),
                       remaining_budget)
        final_target_tail = remaining_budget < shard_size
        count = capacity if final_target_tail else (capacity // shard_size) * shard_size
        expected_models = list(getattr(predictor, "model_ids", ()))
        bundle = cycle / "guided_bundle"
        pending_bundle = bundle.with_name(f".{bundle.name}.pending")
        completed_candidate = None
        if bundle.exists():
            completed_candidate = bundle
        elif (recovering_training and
              (pending_bundle / "factory_bundle_complete.json").is_file()):
            completed_candidate = pending_bundle
        shards_root = cycle / "guided_shards"
        if completed_candidate is not None and recovering_training:
            _validate_bundle(completed_candidate, cfg)
            frozen_count, frozen_hashes = _guided_completion(
                completed_candidate, cycle_id=cycle_id, predictor_model_ids=expected_models)
            overlap = frozen_hashes & set(queue["exclusion_hashes"])
            if frozen_count <= capacity and not overlap:
                count = frozen_count
                final_target_tail = frozen_count < shard_size
            else:
                reason = {
                    "schema_version": 1,
                    "code": ("fresh_exclusion_overlap" if overlap else
                             "fresh_capacity_below_completed_count"),
                    "completed_bundle": str(completed_candidate.resolve()),
                    "completed_count": frozen_count,
                    "fresh_capacity": capacity,
                    "overlap_count": len(overlap),
                    "overlap_sha256": _content_id(sorted(overlap)),
                }
                canonical_binding = {
                    "path": str(completed_candidate.resolve()),
                    "tree_sha256": _content_id(_tree_hashes(completed_candidate)),
                    "completion_sha256": pb.file_sha256(
                        completed_candidate / "factory_bundle_complete.json"),
                }
                chosen = None
                versioned = sorted(
                    [path for path in cycle.glob("guided_bundle_replan_*") if path.is_dir()] +
                    [path for path in cycle.glob(".guided_bundle_replan_*.pending")
                     if (path / "factory_bundle_complete.json").is_file()])
                for candidate in versioned:
                    _validate_bundle(candidate, cfg)
                    proof = _guided_replan_proof(
                        candidate, cycle_id=cycle_id, predictor_model_ids=expected_models)
                    candidate_overlap = proof["hashes"] & set(queue["exclusion_hashes"])
                    if proof["count"] <= capacity and not candidate_overlap and chosen is None:
                        output_name = candidate.name
                        if output_name.startswith(".") and output_name.endswith(".pending"):
                            output_name = output_name[1:-len(".pending")]
                        chosen = {**proof, "source": candidate,
                                  "output": cycle / output_name}
                if chosen is not None:
                    count = chosen["count"]
                    final_target_tail = count < shard_size
                    bundle = chosen["output"]
                    plan_id = chosen["plan"]["plan_id"]
                    guided_plan_identity = chosen["plan"]["identity"]
                elif count > 0:
                    plan_identity = {
                        "schema_version": 1, "cycle_id": cycle_id,
                        "selected_count": count, "fresh_capacity": capacity,
                        "predictor_model_ids": expected_models,
                        "exclusion_sha256": _content_id(sorted(queue["exclusion_hashes"])),
                        "queue_jobs_sha256": queue["jobs_sha256"],
                        "queue_state_sha256": _content_id(queue.get("state_bindings", [])),
                        "canonical_bundle": canonical_binding, "reason": reason,
                    }
                    plan_id = _content_id(plan_identity)
                    guided_plan_identity = plan_identity
                    bundle = cycle / f"guided_bundle_replan_{plan_id[:16]}"
                else:
                    plan_id = None
                if plan_id is not None:
                    shards_root = cycle / f"guided_shards_replan_{plan_id[:16]}"
                guided_replan_context = {
                    "reason": reason, "canonical_bundle": canonical_binding,
                    "fresh_capacity": capacity,
                    "exclusion_count": len(queue["exclusion_hashes"]),
                    "exclusion_sha256": _content_id(sorted(queue["exclusion_hashes"])),
                    "queue_jobs_sha256": queue["jobs_sha256"],
                    "queue_state_sha256": _content_id(queue.get("state_bindings", [])),
                }
                if count <= 0:
                    decision = {
                        "schema_version": 1, **guided_replan_context,
                        "selected_plan_id": None, "selected_count": 0,
                        "selected_pattern_sha256": _content_id([]),
                        "fits_fresh_capacity": True, "disjoint_from_fresh_exclusions": True,
                    }
                    guided_replan_receipt = {
                        "schema_version": 1, "state": "no_plan",
                        "current_decision_id": _content_id(decision),
                        "current_decision": decision,
                        "selected_plan": None, "selected_bundle": None,
                        "selected_shards": None,
                    }
        if count > 0:
            pool_cfg = sm.PoolConfig(
                profile_config=profile, candidate_pool_size=max(10_000, count),
                selected_count=count, id_prefix=f"r80c{cycle_id[:8]}g",
                initial_pool_size=min(2_500, max(10_000, count)), generations=3,
                parent_count=256, seed=int(policy["seed"]) + version,
                update_every_valid_points=int(policy["update_every_valid_unique"]),
            )
            completion_path = "factory_bundle_complete.json"

            def validate_guided(destination: Path) -> None:
                completed_count, _hashes = _guided_completion(
                    destination, cycle_id=cycle_id, predictor_model_ids=expected_models)
                if completed_count != count:
                    raise ValueError("guided bundle completion binding differs from cycle")

            def produce_guided(destination: Path) -> None:
                result = sm.build_pool(pool_cfg, _seed_rows(seed_inputs),
                                       sorted(queue["exclusion_hashes"]), predictor=predictor)
                sm.write_bundle(result, destination)
                sm.write_audit(result, destination, 0.0)
                if guided_replan_context is not None:
                    if (guided_plan_identity is None or
                            _content_id(guided_plan_identity) != plan_id):
                        raise ValueError("guided replan identity changed before publication")
                    _write(destination / "factory_guided_plan.json", {
                        "schema_version": 1, "plan_id": plan_id,
                        "identity": guided_plan_identity,
                    })
                _write(destination / completion_path, {
                    "schema_version": 1, "cycle_id": cycle_id,
                    "selected_count": count, "predictor_model_ids": expected_models,
                    "manifest_sha256": pb.file_sha256(destination / "manifest.json"),
                    "sm_pool_audit_sha256": pb.file_sha256(
                        destination / "sm_pool_audit.json"),
                })

            _publish_bundle(bundle, cfg, produce_guided, validate_guided)
            if guided_replan_context is not None:
                proof = _guided_replan_proof(
                    bundle, cycle_id=cycle_id, predictor_model_ids=expected_models)
                if proof["plan"]["plan_id"] != plan_id:
                    raise ValueError("guided replan bundle differs from selected plan")
            if len(_read(bundle / "manifest.json")) != count:
                raise ValueError("guided bundle count differs from frozen plan")
            shards = _split_or_validate(bundle, shards_root, int(policy["shard_size"]), cfg)
            planned_jobs.extend(_planned(shards, "g", int(policy["guided_priority"]), cycle_id))
            if guided_replan_context is not None:
                pattern_sha256 = _content_id(sorted(proof["hashes"]))
                decision = {
                    "schema_version": 1, **guided_replan_context,
                    "selected_plan_id": proof["plan"]["plan_id"],
                    "selected_count": proof["count"],
                    "selected_pattern_sha256": pattern_sha256,
                    "fits_fresh_capacity": proof["count"] <= capacity,
                    "disjoint_from_fresh_exclusions": not bool(
                        proof["hashes"] & set(queue["exclusion_hashes"])),
                }
                guided_replan_receipt = {
                    "schema_version": 1, "state": "selected",
                    "current_decision_id": _content_id(decision),
                    "current_decision": decision,
                    "selected_plan": {
                        "plan_id": proof["plan"]["plan_id"],
                        "identity": proof["plan"]["identity"],
                        "proof_sha256": pb.file_sha256(bundle / "factory_guided_plan.json"),
                        "bundle": {
                            "path": str(bundle.resolve()),
                            "tree_sha256": _content_id(_tree_hashes(bundle)),
                            "completion_sha256": pb.file_sha256(
                                bundle / "factory_bundle_complete.json"),
                            "selected_count": proof["count"],
                            "pattern_sha256": pattern_sha256,
                        },
                    },
                    "selected_bundle": str(bundle.resolve()),
                    "selected_shards": str(shards_root.resolve()),
                }
            for job in planned_jobs:
                if job["kind"] == "g":
                    job["final_target_tail"] = final_target_tail
            planned_hashes.update(row["pattern_sha256"] for row in _read(bundle / "manifest.json"))

    pending_after_guided = len(pending | planned_hashes)
    remaining_budget = max(0, int(policy["target_valid_unique"]) - valid_unique - pending_after_guided)
    if (not pilot_blocks_ordinary and prepared_blind_pool is not None and
            pending_after_guided < 48 and remaining_budget >= 32):
        need = max(32, ((48 - pending_after_guided + 15) // 16) * 16)
        count = min(64, need, remaining_budget)
        count = count // int(policy["shard_size"]) * int(policy["shard_size"])
        blind = _blind_subset(Path(prepared_blind_pool).resolve(), cycle / "blind_bundle", count,
                              queue["exclusion_hashes"] | planned_hashes, cfg)
        shards_root = cycle / "blind_shards"
        shards = _split_or_validate(blind, shards_root, int(policy["shard_size"]), cfg)
        planned_jobs.extend(_planned(shards, "b", int(policy["blind_priority"]), cycle_id,
                                     start=1))

    if pb.file_sha256(profile) != profile_sha256:
        raise ValueError("profile config changed during cycle preparation")
    if _protocol_binding(train_root)[1] != protocol_binding:
        raise ValueError("training protocol changed during cycle preparation")
    if [_path_binding(Path(value)) for value in seed_inputs] != seed_bindings:
        raise ValueError("seed inputs changed during cycle preparation")
    if ((prepared_blind_pool is not None and
         _path_binding(Path(prepared_blind_pool)) != blind_binding)):
        raise ValueError("prepared blind pool changed during cycle preparation")
    if pilot_binding is not None:
        _require_pilot_request_binding(pilot_request, profile, pilot_binding)
    if "pair_proofs" in queue:
        _recheck_queue_proof(dataset, queue)
    receipt = {
        "schema_version": 1, "status": ("prepared" if planned_jobs else "idle"), "once": True,
        "cycle_id": cycle_id, "dataset_root": str(dataset), "scope": cfg.scope,
        "profile_config": str(profile), "training_workdir": str(train_root),
        "profile_sha256": profile_sha256,
        "expected_retry_workers": list(retry_workers),
        "training_protocol": protocol_binding,
        "audited_source_bindings": source_binding,
        "seed_inputs": seed_bindings, "prepared_blind_pool": blind_binding,
        "audit": audit, "queue_jobs_sha256": queue["jobs_sha256"],
        "latest_data_version": version, "trained_this_cycle": trained_this_cycle,
        "pretrain_prediction_audit": prediction_audit_binding,
        "training_snapshot": training_snapshot_binding,
        "recovered_training_receipt": recovered_training_receipt,
        "trained_model_summary": _training_summary(train_root, version),
        "model_quality_gate": None,
        "predictor_binding": (dict(getattr(predictor, "binding", {})) if predictor is not None
                              else None),
        "new_unique_available_before_training": len(new_order),
        "valid_unique": valid_unique, "pending_unique": len(pending),
        "guided_pending_unique": len(guided_pending), "target_valid_unique": policy["target_valid_unique"],
        "prepared_queue_state_bindings": queue.get("state_bindings", []),
        "retryable_failed_jobs": queue.get("retryable_failed_jobs", []),
        "terminal_failed_jobs": queue.get("terminal_failed_jobs", []),
        "planned_jobs": planned_jobs, "dispatch_gate": "not_run",
        "limits": {"max_guided_outstanding": policy["max_guided_outstanding"],
                   "valid_plus_pending_plus_planned": valid_unique + len(pending | planned_hashes) +
                   sum(job["count"] for job in planned_jobs if job["kind"] == "b")},
    }
    if pilot_binding is not None:
        receipt["pilot_request"] = pilot_binding
    if pilot_receipt is not None:
        receipt["pilot"] = pilot_receipt
    if guided_replan_receipt is not None:
        receipt["guided_replan"] = guided_replan_receipt
    _write(receipt_path, receipt)
    if not planned_jobs and active.is_file():
        active.unlink()
    return receipt_path


def run_once(*, local_workdir: str | Path, dataset_root: str | Path,
             profile_config: str | Path, training_workdir: str | Path,
             seed_inputs: Sequence[str | Path], prepared_blind_pool: str | Path | None = None,
             cold_start_predictor: Any | None = None,
             expected_retry_workers: Sequence[str] | None = None,
             pilot_request: str | Path | None = None,
             planning_guard: _PlanningGuard | None = None) -> Path:
    local = Path(local_workdir).resolve()
    train_root = Path(training_workdir).resolve()
    local.mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(local / "cycle-controller.lock"):
        with _exclusive_lock(train_root / "factory-cycle.lock"):
            return _run_once_unlocked(
                local_workdir=local, dataset_root=dataset_root, profile_config=profile_config,
                training_workdir=train_root, seed_inputs=seed_inputs,
                prepared_blind_pool=prepared_blind_pool,
                cold_start_predictor=cold_start_predictor,
                expected_retry_workers=expected_retry_workers,
                pilot_request=pilot_request, planning_guard=planning_guard)


single_cycle = run_once


def _dedust_add(dataset_root: Path, job: Mapping[str, Any], scope: str) -> None:
    from script import dedust
    previous = dedust.DATASET_PATH
    dedust.DATASET_PATH = dataset_root
    try:
        dedust.jobs_add(SimpleNamespace(input=job["input"], store=job["store"],
                                        prio=job["prio"], scope=scope, machine=None,
                                        config=f"{job['input']}/config.yaml"))
    finally:
        dedust.DATASET_PATH = previous


def _check_dispatch_budget(dataset: Path, cfg: Any, policy: Mapping[str, Any],
                           hashes: set[str], priority: int,
                           expected_retry_workers: Sequence[str] | None = None) -> None:
    queue = _queue_view(dataset, cfg, policy, expected_retry_workers)
    _check_dispatch_budget_from_queue(queue, policy, hashes, priority)


def _check_dispatch_budget_from_queue(queue: Mapping[str, Any], policy: Mapping[str, Any],
                                      hashes: set[str], priority: int) -> None:
    valid = queue["successful_hashes"]
    pending_hashes = queue["pending_hashes"] - valid
    guided = queue["guided_pending_hashes"] - valid
    if len(valid) + len(pending_hashes | hashes) > int(policy["target_valid_unique"]):
        raise ValueError("dispatch would exceed valid plus reserved target budget")
    if (priority == int(policy["guided_priority"]) and
            len(guided | hashes) > int(policy["max_guided_outstanding"])):
        raise ValueError("dispatch would exceed guided outstanding budget")


def _extend_queue_proof_after_add(dataset: Path, queue: dict[str, Any], cfg: Any,
                                  policy: Mapping[str, Any], job: Mapping[str, Any],
                                  input_proof: Mapping[str, Any]) -> None:
    """Extend a locked same-commit proof after one canonical queue append."""

    jobs_path = dataset / "jobs.json"
    raw = jobs_path.read_bytes()
    try:
        current_jobs = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        if jobs_path.read_bytes() != raw:
            raise LiveSnapshotChanged("queue changed after queue add; retry") from exc
        raise ValueError("jobs.json is stably malformed after queue add") from exc
    expected = {"input": job["input"], "store": job["store"], "prio": job["prio"],
                "scope": cfg.scope, "config": f"{job['input']}/config.yaml"}
    if current_jobs != [*queue["all_jobs"], expected]:
        raise ValueError(
            f"queue add was not one exact append-only transition: {job['store']}")
    jobs_sha = hashlib.sha256(raw).hexdigest()
    if pb.file_sha256(jobs_path) != jobs_sha:
        raise LiveSnapshotChanged("queue changed after exact queue append; retry")
    state_binding, _payloads = _job_state_binding(dataset / "jobs_state", job["store"])
    if state_binding["done"]["exists"] and state_binding["fail"]["exists"]:
        raise ValueError(f"job has both fail and done markers: {job['store']}")
    rows = input_proof["rows"]
    hashes = {row["pattern_sha256"] for row in rows}
    queue["all_jobs"] = current_jobs
    queue["jobs_sha256"] = jobs_sha
    queue["jobs"].append({**expected, "terminal": False})
    queue["input_proofs"].append(dict(input_proof))
    queue["state_bindings"].append({"store": job["store"], "markers": state_binding,
                                    "terminal": False})
    queue.setdefault("result_absence_bindings", []).append({
        "store": job["store"],
        "results_path": str(_safe_child(dataset, job["store"]) / "results.json"),
        "terminal": False,
    })
    queue["pending_hashes"].update(hashes)
    if int(job["prio"]) == int(policy["guided_priority"]):
        queue["guided_pending_hashes"].update(hashes)
    queue["exclusion_hashes"].update(hashes)
    root = Path(input_proof["input"])
    queue["exclusion_bindings"].append({
        "input": str(root),
        "measurement_sha256": input_proof["input_metadata_sha256"]["measurement.json"],
        "manifest_sha256": input_proof["input_metadata_sha256"]["manifest.json"],
    })
    queue["audit_jobs"].append({
        "store": job["store"], "input": job["input"], "prio": job["prio"],
        "expected": len(rows), "audited_ok": 0, "errors": {}, "markers": {},
        "input_metadata_sha256": input_proof["input_metadata_sha256"],
        "source_metadata_sha256": None,
    })


def _commit_dispatch_locked(path: Path, *,
                            queue_add: Callable[[Path, Mapping[str, Any], str], None],
                            duplicate_check: Callable[[Path, Path], int]) -> Path:
    path = path.resolve(); receipt = _read(path)
    if receipt.get("status") not in {"prepared", "dispatching", "dispatched"}:
        raise ValueError("receipt is not dispatchable")
    dataset = Path(receipt["dataset_root"]).resolve()
    jobs_path = dataset / "jobs.json"
    profile = Path(receipt["profile_config"]).resolve()
    if pb.file_sha256(profile) != receipt.get("profile_sha256"):
        raise ValueError("profile config differs from prepared receipt")
    cfg, policy = _load_profile(profile)
    if "expected_retry_workers" not in receipt:
        raise ValueError("dispatch receipt lacks expected retry worker roster")
    retry_workers = _normalize_retry_workers(receipt["expected_retry_workers"])
    if cfg.scope != receipt["scope"]:
        raise ValueError("receipt scope differs from bound profile")
    if _protocol_binding(Path(receipt["training_workdir"]).resolve())[1] != receipt.get(
            "training_protocol"):
        raise ValueError("training protocol differs from prepared receipt")
    pilot = None
    dispatch_queue = None
    pilot_binding = receipt.get("pilot_request")
    if pilot_binding is not None:
        if not isinstance(pilot_binding, dict) or "request_path" not in pilot_binding:
            raise ValueError("dispatch receipt has an invalid pilot request binding")
        pilot = _require_pilot_request_binding(
            pilot_binding["request_path"], profile, pilot_binding)
    elif "pilot_request" in receipt or "pilot" in receipt or any(
            item.get("kind") == "p" for item in receipt.get("planned_jobs", [])):
        raise ValueError("pilot dispatch data lacks an exact request binding")
    pilot_jobs = [item for item in receipt.get("planned_jobs", [])
                  if item.get("kind") == "p"]
    if pilot_jobs:
        if pilot is None:
            raise ValueError("pilot shard lacks its exact request binding")
        pilot_receipt = receipt.get("pilot")
        if not isinstance(pilot_receipt, dict) or pilot_receipt.get("state") != "prepared":
            raise ValueError("pilot dispatch receipt lacks prepared pilot evidence")
        if pilot_receipt.get("request") != pilot_binding:
            raise ValueError("prepared pilot evidence has a different request binding")
        if len(pilot_jobs) != 1 or any(item.get("kind") != "p"
                                      for item in receipt["planned_jobs"]):
            raise ValueError("pilot receipt must dispatch exactly one pilot shard")
        planned_pilot = pilot_jobs[0]
        bundle_binding = pilot_receipt.get("bundle")
        staged_pilot = Path(planned_pilot["staged_input"]).resolve()
        if (not isinstance(bundle_binding, dict) or
                Path(bundle_binding.get("path", "")).resolve() != staged_pilot or
                bundle_binding.get("tree_sha256") != _content_id(_tree_hashes(staged_pilot))):
            raise ValueError("pilot bundle binding differs from prepared shard")
        prepared_proof = _pilot_call(shell_pilot.validate_bundle, staged_pilot, pilot)
        if bundle_binding.get("proof") != prepared_proof:
            raise ValueError("pilot bundle proof differs from prepared receipt")
        proof_identity = {
            "predictor_binding": prepared_proof["predictor_binding"],
            "model_ids": prepared_proof["model_ids"],
            "surrogate_generation": prepared_proof["surrogate_generation"],
            "valid_observations_at_fit": prepared_proof["valid_observations_at_fit"],
        }
        if any(pilot_receipt.get(key) != value for key, value in proof_identity.items()):
            raise ValueError("pilot predictor identity differs from prepared proof")
        reference_binding = pilot_receipt.get("reference")
        if (not isinstance(reference_binding, dict) or
                _path_binding(Path(reference_binding.get("path", ""))) != reference_binding):
            raise ValueError("pilot frozen reference binding differs from prepared receipt")
        shell_pilot.recheck_reference(Path(reference_binding["path"]))
        dispatch_queue = _queue_view(dataset, cfg, policy, retry_workers)
        queued = _queued_pilot_evidence(dataset, dispatch_queue, pilot, cfg.scope)
        if queued is not None and (queued["job"].get("input") != pilot_jobs[0]["input"] or
                                   queued["job"].get("store") != pilot_jobs[0]["store"]):
            raise ValueError("one-shot pilot_id is already consumed by another queued cohort")
    elif isinstance(receipt.get("pilot"), dict) and receipt["pilot"].get("state") == "prepared":
        raise ValueError("prepared pilot evidence lacks its pilot shard")
    if receipt.get("status") == "dispatched":
        return path
    if dispatch_queue is None:
        dispatch_queue = _queue_view(dataset, cfg, policy, retry_workers)
    proof_dispatch = "pair_proofs" in dispatch_queue
    legacy_initial_queue = None if proof_dispatch else dispatch_queue
    if proof_dispatch:
        _recheck_queue_proof(dataset, dispatch_queue)
    for planned in receipt["planned_jobs"]:
        staged = Path(planned["staged_input"]).resolve()
        if _content_id(_tree_hashes(staged)) != planned["tree_sha256"]:
            raise ValueError(f"staged shard changed: {staged}")
        _validate_bundle(staged, cfg)
        if planned.get("kind") == "p":
            proof = shell_pilot.validate_bundle(staged, pilot)
            expected_pilot = receipt["pilot"]["selection"]
            actual_pilot = {"ordered_ids": proof["ordered_ids"],
                            "ordered_hashes": proof["ordered_hashes"],
                            "role_counts": proof["role_counts"]}
            if actual_pilot != expected_pilot or planned.get("pilot_id") != pilot.pilot_id:
                raise ValueError("staged pilot cohort differs from prepared receipt")
        current_jobs = _read(jobs_path)
        existing = next((job for job in current_jobs if job.get("store") == planned["store"]), None)
        expected = {"input": planned["input"], "store": planned["store"],
                    "prio": planned["prio"], "scope": receipt["scope"],
                    "config": f"{planned['input']}/config.yaml"}
        if existing is not None:
            if existing != expected:
                raise ValueError(f"queue store collision: {planned['store']}")
            if planned.get("kind") == "p":
                shell_pilot.validate_bundle(_safe_child(dataset, planned["input"]), pilot)
            continue
        rows = _read(staged / "manifest.json")
        hashes = {row["pattern_sha256"] for row in rows}
        if proof_dispatch:
            _recheck_queue_proof(dataset, dispatch_queue)
            _check_dispatch_budget_from_queue(
                dispatch_queue, policy, hashes, int(planned["prio"]))
        else:
            if legacy_initial_queue is not None:
                _check_dispatch_budget_from_queue(
                    legacy_initial_queue, policy, hashes, int(planned["prio"]))
                legacy_initial_queue = None
            else:
                _check_dispatch_budget(dataset, cfg, policy, hashes, int(planned["prio"]),
                                       retry_workers)
        if receipt["status"] == "prepared":
            receipt["status"] = "dispatching"
            _write(path, receipt)
        state_root = dataset / "jobs_state"
        if any((state_root / f"{planned['store']}.{suffix}").exists()
               for suffix in ("claim", "done", "fail")):
            raise ValueError(f"unqueued store already has worker state: {planned['store']}")
        if _safe_child(dataset, planned["store"]).exists():
            raise ValueError(f"unqueued store path already exists: {planned['store']}")
        destination = _safe_child(dataset, planned["input"])
        if destination.exists():
            if _tree_hashes(destination) != _tree_hashes(staged):
                raise ValueError(f"immutable input collision: {destination}")
        else:
            pending = dataset / f".{planned['input']}.pending-{receipt['cycle_id'][:12]}"
            if pending.exists() and _tree_hashes(pending) != _tree_hashes(staged):
                raise ValueError(f"pending input collision: {pending}")
            if not pending.exists():
                shutil.copytree(staged, pending)
            os.replace(pending, destination)
        if planned.get("kind") == "p":
            shell_pilot.validate_bundle(destination, pilot)
        current_jobs = _read(jobs_path)
        existing = next((job for job in current_jobs if job.get("store") == planned["store"]), None)
        if existing is not None:
            if existing != expected:
                raise ValueError(f"queue store collision after input copy: {planned['store']}")
            continue
        if proof_dispatch:
            _recheck_queue_proof(dataset, dispatch_queue)
            _check_dispatch_budget_from_queue(
                dispatch_queue, policy, hashes, int(planned["prio"]))
        else:
            _check_dispatch_budget(dataset, cfg, policy, hashes, int(planned["prio"]),
                                   retry_workers)
        duplicate_check(destination, dataset)
        input_proof = (_input_physical_proof(destination, cfg)
                       if proof_dispatch else None)
        queue_add(dataset, planned, receipt["scope"])
        if planned.get("kind") == "p":
            queued_jobs = _read(jobs_path)
            queued = next((job for job in queued_jobs
                           if job.get("store") == planned["store"]), None)
            if queued != expected:
                raise ValueError("pilot queue record differs immediately after queue add")
            shell_pilot.validate_bundle(destination, pilot)
        if proof_dispatch:
            _extend_queue_proof_after_add(
                dataset, dispatch_queue, cfg, policy, planned, input_proof)
            _recheck_queue_proof(dataset, dispatch_queue)
    receipt["status"] = "dispatched"; receipt["dispatch_gate"] = "passed_per_shard"
    _write(path, receipt)
    active = path.parents[2] / "active_cycle.json"
    if active.is_file() and _read(active).get("receipt") == str(path):
        active.unlink()
    return path


def commit_dispatch(receipt_path: str | Path, *,
                    queue_add: Callable[[Path, Mapping[str, Any], str], None] = _dedust_add,
                    duplicate_check: Callable[[Path, Path], int] = pb.check_duplicates,
                    coordinator: DatasetWriteCoordinator | None = None) -> Path:
    """Copy and enqueue a reviewed prepared receipt, idempotently per shard."""

    path = Path(receipt_path).resolve()
    receipt = _read(path)
    dataset = Path(receipt["dataset_root"]).resolve()
    scope = str(receipt["scope"])
    boundary = coordinator.turn(0) if coordinator is not None else contextlib.nullcontext()
    try:
        with boundary:
            with _dataset_exclusive_lock(dataset, scope):
                return _commit_dispatch_locked(path, queue_add=queue_add,
                                               duplicate_check=duplicate_check)
    finally:
        if coordinator is not None and path.is_file():
            status = _read(path).get("status")
            if status in {"dispatched", "idle"}:
                coordinator.release_receipt(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--local-workdir", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--profile-config", type=Path)
    parser.add_argument("--training-workdir", type=Path)
    parser.add_argument("--seed-input", type=Path, action="append")
    parser.add_argument("--prepared-blind-pool", type=Path)
    parser.add_argument("--pilot-request", type=Path)
    parser.add_argument("--expected-retry-worker", action="append", default=[])
    parser.add_argument("--commit-receipt", type=Path)
    args = parser.parse_args()
    if args.commit_receipt:
        result = commit_dispatch(args.commit_receipt)
    else:
        required = (args.local_workdir, args.dataset_root, args.profile_config,
                    args.training_workdir, args.seed_input)
        if not args.once or any(value is None for value in required):
            parser.error("prepare requires --once and all work/dataset/profile/training/seed inputs")
        result = run_once(local_workdir=args.local_workdir, dataset_root=args.dataset_root,
                          profile_config=args.profile_config,
                          training_workdir=args.training_workdir, seed_inputs=args.seed_input,
                          prepared_blind_pool=args.prepared_blind_pool,
                          expected_retry_workers=args.expected_retry_worker,
                          pilot_request=args.pilot_request)
    print(result)


if __name__ == "__main__":
    main()
