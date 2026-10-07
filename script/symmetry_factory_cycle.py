"""One recoverable prepare/dispatch cycle for the R80 symmetry data factory.

Preparation writes staged artifacts below the explicit local work directory and
short-lived ownership metadata below the dataset/training roots.  Dispatch is a
separate call: it copies frozen 16-item shards, re-runs the duplicate gate, and
uses the existing queue helper.  Neither path runs HFSS or starts a process.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import socket
import shutil
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
from script import symmetry_sm_pool as sm
from script import symmetry_training as training
from script.profiled_shards import snapshot_successes, split_bundle


class LiveSnapshotChanged(RuntimeError):
    """A worker advanced queue/results during a physical controller snapshot."""


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


def _queue_view(dataset_root: Path, cfg: Any, policy: Mapping[str, Any]) -> dict[str, Any]:
    jobs_path = dataset_root / "jobs.json"
    before = pb.file_sha256(jobs_path)
    jobs = _read(jobs_path)
    if not isinstance(jobs, list):
        raise ValueError("jobs.json must contain a list")
    wanted_mid = measurement_id(cfg.measurement)
    wanted_sid = score_spec_id(cfg.score_spec)
    relevant, pending_hashes, guided_pending = [], set(), set()
    successful_hashes: set[str] = set()
    pairs, pair_bindings = [], []
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
        rows = pb.validate_input(input_dir, input_cfg)
        store = _safe_child(dataset_root, job["store"])
        fail = (dataset_root / "jobs_state" / f"{job['store']}.fail").exists()
        done = (dataset_root / "jobs_state" / f"{job['store']}.done").exists()
        terminal = fail or done
        if not terminal:
            hashes = {row["pattern_sha256"] for row in rows}
            pending_hashes.update(hashes)
            if int(job.get("prio", 9)) == int(policy["guided_priority"]):
                guided_pending.update(hashes)
        if (store / "results.json").is_file():
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
            successful_hashes.update(entry["pattern_sha256"] for entry in results.values()
                                     if isinstance(entry, dict) and entry.get("status") == "ok"
                                     and "error" not in entry)
        if any(pb.file_sha256(input_dir / name) != digest for name, digest in input_hashes.items()):
            raise ValueError(f"input changed during queue scan: {input_dir}")
        relevant.append({**job, "terminal": terminal})

    # Every same-measurement input reserves its physical pattern forever,
    # including failed jobs; this is broader than the currently pending budget.
    exclusions = set()
    for input_dir in sorted(dataset_root.glob("*_input")):
        marker = input_dir / "measurement.json"
        manifest = input_dir / "manifest.json"
        if marker.is_file() and manifest.is_file() and measurement_id(_read(marker)) == wanted_mid:
            marker_hash, manifest_hash = pb.file_sha256(marker), pb.file_sha256(manifest)
            rows = _read(manifest)
            exclusions.update(row["pattern_sha256"] for row in rows)
            if pb.file_sha256(marker) != marker_hash or pb.file_sha256(manifest) != manifest_hash:
                raise ValueError(f"reserved input changed during exclusion scan: {input_dir}")
    if pb.file_sha256(jobs_path) != before:
        raise LiveSnapshotChanged("queue changed during controller scan; retry")
    return {"jobs_sha256": before, "jobs": relevant, "pairs": pairs,
            "pair_bindings": pair_bindings,
            "pending_hashes": pending_hashes, "guided_pending_hashes": guided_pending,
            "successful_hashes": successful_hashes, "exclusion_hashes": exclusions}


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


def _split_or_validate(bundle: Path, output: Path, shard_size: int, cfg: Any) -> list[Path]:
    if not output.exists():
        return split_bundle(bundle, output, shard_size)
    receipt = _read(output / "split_receipt.json")
    if (Path(receipt["canonical_input"]).resolve() != bundle.resolve() or
            receipt["canonical_manifest_sha256"] != pb.file_sha256(bundle / "manifest.json") or
            int(receipt["shard_size"]) != shard_size):
        raise ValueError(f"existing shard split differs from staged bundle: {output}")
    expected = [entry["name"] for entry in receipt["shards"]]
    actual = sorted(path.name for path in output.glob("*_input") if path.is_dir())
    if actual != sorted(expected):
        raise ValueError(f"existing shard split has missing or extra children: {output}")
    shards = [output / name for name in expected]
    for shard in shards:
        _validate_bundle(shard, cfg)
    if sum(len(_read(shard / "manifest.json")) for shard in shards) != len(
            _read(bundle / "manifest.json")):
        raise ValueError(f"existing shard split count differs from staged bundle: {output}")
    return shards


def _run_once_unlocked(*, local_workdir: str | Path, dataset_root: str | Path,
                       profile_config: str | Path, training_workdir: str | Path,
                       seed_inputs: Sequence[str | Path],
                       prepared_blind_pool: str | Path | None = None,
                       cold_start_predictor: Any | None = None) -> Path:
    """Prepare one recoverable cycle and return its reviewable action receipt."""

    local = Path(local_workdir).resolve(); dataset = Path(dataset_root).resolve()
    profile = Path(profile_config).resolve(); train_root = Path(training_workdir).resolve()
    local.mkdir(parents=True, exist_ok=True)
    active = local / "active_cycle.json"
    recovered_training_receipt = None
    recovered_prediction_audit = None
    recovered_training_snapshot = None
    if active.is_file():
        receipt = Path(_read(active)["receipt"]).resolve()
        receipt.relative_to(local)
        if receipt.is_file():
            prior = _read(receipt)
            if prior.get("status") in {"prepared", "dispatching", "training"}:
                expected = (str(dataset), str(profile), str(train_root))
                actual = (prior.get("dataset_root"), prior.get("profile_config"),
                          prior.get("training_workdir"))
                if actual != expected:
                    raise ValueError("active cycle belongs to different explicit inputs")
            if prior.get("status") == "training":
                recovered_training_receipt = str(receipt)
                recovered_prediction_audit = prior.get("pretrain_prediction_audit")
                recovered_training_snapshot = prior.get("training_snapshot")
            elif prior.get("status") in {"prepared", "dispatching"}:
                return receipt

    cfg, policy = _load_profile(profile)
    _protocol, protocol_binding = _protocol_binding(train_root)
    with _exclusive_lock(_dataset_lock(dataset, cfg.scope)):
        audit = _physical_snapshot(dataset, cfg.scope)
        queue = _queue_view(dataset, cfg, policy)
        if audit["jobs_sha256"] != queue["jobs_sha256"]:
            raise LiveSnapshotChanged(
                "queue changed between physical audit and controller scan; retry")
        _require_pair_bindings(queue["pair_bindings"])
    version, trained_hashes, trained_receipt = _trained(train_root)
    source_binding = queue["pair_bindings"]
    seed_bindings = [_path_binding(Path(value)) for value in seed_inputs]
    blind_binding = (_path_binding(Path(prepared_blind_pool))
                     if prepared_blind_pool is not None else None)
    cold_binding = list(getattr(cold_start_predictor, "model_ids", ()))
    profile_sha256 = pb.file_sha256(profile)
    cycle_id = _content_id({"jobs": audit["jobs_sha256"], "sources": source_binding,
                            "trained": trained_receipt, "profile": profile_sha256,
                            "training_protocol": protocol_binding,
                            "seeds": seed_bindings, "blind": blind_binding,
                            "cold_predictor": cold_binding})
    cycle = local / "cycles" / cycle_id
    receipt_path = cycle / "action_receipt.json"
    cycle.mkdir(parents=True, exist_ok=True)
    _write(active, {"cycle_id": cycle_id, "receipt": str(receipt_path)})

    predictor = None
    trained_this_cycle = False
    prediction_audit_binding = recovered_prediction_audit
    training_snapshot_binding = recovered_training_snapshot
    _require_pair_bindings(queue["pair_bindings"])
    new_order = _success_order(queue["pairs"], trained_hashes)
    _require_pair_bindings(queue["pair_bindings"])
    if len(new_order) >= int(policy["update_every_valid_unique"]):
        take = min(96, len(new_order))
        omit = set(new_order[take:])
        frozen = cycle / "training_snapshot"
        if not frozen.exists():
            try:
                snapshot_successes(queue["pairs"], frozen,
                                   exclude_pattern_hashes=trained_hashes | omit)
            except ValueError as exc:
                message = str(exc)
                if ("source store " in message and
                        "metadata changed during operation: ['results.json']" in message):
                    raise LiveSnapshotChanged(message) from exc
                raise
        _require_pair_bindings(queue["pair_bindings"])
        frozen_rows = _read(frozen / "manifest.json")
        frozen_hashes = [row["pattern_sha256"] for row in frozen_rows]
        if len(frozen_hashes) != take or set(frozen_hashes) != set(new_order[:take]):
            raise ValueError("frozen training snapshot differs from exact audited selection")
        frozen_binding = _read(frozen / "source_bindings.json")
        expected_results = {
            binding["store"]: binding["store_metadata_sha256"]["results.json"]
            for binding in queue["pair_bindings"]
        }
        actual_results = {item["store"]: item["partial_source_results_sha256"]
                          for item in frozen_binding["sources"]}
        if actual_results != expected_results:
            raise LiveSnapshotChanged(
                "frozen training snapshot source hashes differ from physical audit")
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
        _write(receipt_path, {
            "schema_version": 1, "status": "training", "once": True,
            "cycle_id": cycle_id, "dataset_root": str(dataset), "scope": cfg.scope,
            "profile_config": str(profile), "training_workdir": str(train_root),
            "profile_sha256": profile_sha256,
            "training_protocol": protocol_binding,
            "audited_source_bindings": source_binding,
            "pretrain_prediction_audit": prediction_audit_binding,
            "training_snapshot": training_snapshot_binding,
            "quality_gate": None,
            "all_valid_observations_retained_for_training": True,
        })
        version_dir = training.ingest_profile_stores(
            train_root, [frozen], min_new=int(policy["update_every_valid_unique"]), max_new=96)
        version = int(version_dir.name.removeprefix("data-v"))
        training.train_version(train_root, version)
        predictor = training.load_current_predictor(train_root, version)
        trained_this_cycle = True
    elif version > 0:
        try:
            predictor = training.load_current_predictor(train_root, version)
        except FileNotFoundError:
            training.train_version(train_root, version)
            predictor = training.load_current_predictor(train_root, version)
            trained_this_cycle = True
    elif predictor is None and cold_start_predictor is not None:
        if any(int(job.get("prio", 9)) == int(policy["guided_priority"])
               for job in queue["jobs"]):
            raise ValueError("historical cold start cannot be reused after guided jobs exist")
        predictor = cold_start_predictor

    # Re-audit after model work so proposal capacity and exclusions reflect the
    # current queue.  Training deliberately does not hold the dataset lock.
    with _exclusive_lock(_dataset_lock(dataset, cfg.scope)):
        audit = _physical_snapshot(dataset, cfg.scope)
        queue = _queue_view(dataset, cfg, policy)
        if audit["jobs_sha256"] != queue["jobs_sha256"]:
            raise LiveSnapshotChanged(
                "queue changed between final physical audit and controller scan; retry")
        _require_pair_bindings(queue["pair_bindings"])

    valid_unique = len(queue["successful_hashes"])
    pending = queue["pending_hashes"] - queue["successful_hashes"]
    guided_pending = queue["guided_pending_hashes"] - queue["successful_hashes"]
    remaining_budget = max(0, int(policy["target_valid_unique"]) - valid_unique - len(pending))
    planned_jobs: list[dict[str, Any]] = []
    planned_hashes: set[str] = set()
    if predictor is not None and remaining_budget > 0:
        shard_size = int(policy["shard_size"])
        capacity = min(int(policy["wave_size"]),
                       int(policy["max_guided_outstanding"]) - len(guided_pending),
                       remaining_budget)
        final_target_tail = remaining_budget < shard_size
        count = capacity if final_target_tail else (capacity // shard_size) * shard_size
        if count > 0:
            pool_cfg = sm.PoolConfig(
                profile_config=profile, candidate_pool_size=max(10_000, count),
                selected_count=count, id_prefix=f"r80c{cycle_id[:8]}g",
                initial_pool_size=min(2_500, max(10_000, count)), generations=3,
                parent_count=256, seed=int(policy["seed"]) + version,
                update_every_valid_points=int(policy["update_every_valid_unique"]),
            )
            bundle = cycle / "guided_bundle"
            expected_models = list(getattr(predictor, "model_ids", ()))
            completion_path = "factory_bundle_complete.json"

            def validate_guided(destination: Path) -> None:
                marker = _read(destination / completion_path)
                expected = {"schema_version": 1, "cycle_id": cycle_id,
                            "selected_count": count, "predictor_model_ids": expected_models,
                            "manifest_sha256": pb.file_sha256(destination / "manifest.json"),
                            "sm_pool_audit_sha256": pb.file_sha256(
                                destination / "sm_pool_audit.json")}
                if marker != expected:
                    raise ValueError("guided bundle completion binding differs from cycle")

            def produce_guided(destination: Path) -> None:
                result = sm.build_pool(pool_cfg, _seed_rows(seed_inputs),
                                       sorted(queue["exclusion_hashes"]), predictor=predictor)
                sm.write_bundle(result, destination)
                sm.write_audit(result, destination, 0.0)
                _write(destination / completion_path, {
                    "schema_version": 1, "cycle_id": cycle_id,
                    "selected_count": count, "predictor_model_ids": expected_models,
                    "manifest_sha256": pb.file_sha256(destination / "manifest.json"),
                    "sm_pool_audit_sha256": pb.file_sha256(
                        destination / "sm_pool_audit.json"),
                })

            _publish_bundle(bundle, cfg, produce_guided, validate_guided)
            if len(_read(bundle / "manifest.json")) != count:
                raise ValueError("guided bundle count differs from frozen plan")
            shards_root = cycle / "guided_shards"
            shards = _split_or_validate(bundle, shards_root, int(policy["shard_size"]), cfg)
            planned_jobs.extend(_planned(shards, "g", int(policy["guided_priority"]), cycle_id))
            for job in planned_jobs:
                if job["kind"] == "g":
                    job["final_target_tail"] = final_target_tail
            planned_hashes.update(row["pattern_sha256"] for row in _read(bundle / "manifest.json"))

    pending_after_guided = len(pending | planned_hashes)
    remaining_budget = max(0, int(policy["target_valid_unique"]) - valid_unique - pending_after_guided)
    if prepared_blind_pool is not None and pending_after_guided < 48 and remaining_budget >= 32:
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
    receipt = {
        "schema_version": 1, "status": ("prepared" if planned_jobs else "idle"), "once": True,
        "cycle_id": cycle_id, "dataset_root": str(dataset), "scope": cfg.scope,
        "profile_config": str(profile), "training_workdir": str(train_root),
        "profile_sha256": profile_sha256,
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
        "planned_jobs": planned_jobs, "dispatch_gate": "not_run",
        "limits": {"max_guided_outstanding": policy["max_guided_outstanding"],
                   "valid_plus_pending_plus_planned": valid_unique + len(pending | planned_hashes) +
                   sum(job["count"] for job in planned_jobs if job["kind"] == "b")},
    }
    _write(receipt_path, receipt)
    if not planned_jobs and active.is_file():
        active.unlink()
    return receipt_path


def run_once(*, local_workdir: str | Path, dataset_root: str | Path,
             profile_config: str | Path, training_workdir: str | Path,
             seed_inputs: Sequence[str | Path], prepared_blind_pool: str | Path | None = None,
             cold_start_predictor: Any | None = None) -> Path:
    local = Path(local_workdir).resolve()
    train_root = Path(training_workdir).resolve()
    local.mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(local / "cycle-controller.lock"):
        with _exclusive_lock(train_root / "factory-cycle.lock"):
            return _run_once_unlocked(
                local_workdir=local, dataset_root=dataset_root, profile_config=profile_config,
                training_workdir=train_root, seed_inputs=seed_inputs,
                prepared_blind_pool=prepared_blind_pool,
                cold_start_predictor=cold_start_predictor)


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
                           hashes: set[str], priority: int) -> None:
    queue = _queue_view(dataset, cfg, policy)
    valid = queue["successful_hashes"]
    pending_hashes = queue["pending_hashes"] - valid
    guided = queue["guided_pending_hashes"] - valid
    if len(valid) + len(pending_hashes | hashes) > int(policy["target_valid_unique"]):
        raise ValueError("dispatch would exceed valid plus reserved target budget")
    if (priority == int(policy["guided_priority"]) and
            len(guided | hashes) > int(policy["max_guided_outstanding"])):
        raise ValueError("dispatch would exceed guided outstanding budget")


def _commit_dispatch_locked(path: Path, *,
                            queue_add: Callable[[Path, Mapping[str, Any], str], None],
                            duplicate_check: Callable[[Path, Path], int]) -> Path:
    path = path.resolve(); receipt = _read(path)
    if receipt.get("status") == "dispatched":
        return path
    if receipt.get("status") not in {"prepared", "dispatching"}:
        raise ValueError("receipt is not dispatchable")
    dataset = Path(receipt["dataset_root"]).resolve()
    jobs_path = dataset / "jobs.json"
    profile = Path(receipt["profile_config"]).resolve()
    if pb.file_sha256(profile) != receipt.get("profile_sha256"):
        raise ValueError("profile config differs from prepared receipt")
    cfg, policy = _load_profile(profile)
    if cfg.scope != receipt["scope"]:
        raise ValueError("receipt scope differs from bound profile")
    if _protocol_binding(Path(receipt["training_workdir"]).resolve())[1] != receipt.get(
            "training_protocol"):
        raise ValueError("training protocol differs from prepared receipt")
    for planned in receipt["planned_jobs"]:
        staged = Path(planned["staged_input"]).resolve()
        if _content_id(_tree_hashes(staged)) != planned["tree_sha256"]:
            raise ValueError(f"staged shard changed: {staged}")
        _validate_bundle(staged, cfg)
        current_jobs = _read(jobs_path)
        existing = next((job for job in current_jobs if job.get("store") == planned["store"]), None)
        expected = {"input": planned["input"], "store": planned["store"],
                    "prio": planned["prio"], "scope": receipt["scope"],
                    "config": f"{planned['input']}/config.yaml"}
        if existing is not None:
            if existing != expected:
                raise ValueError(f"queue store collision: {planned['store']}")
            continue
        rows = _read(staged / "manifest.json")
        hashes = {row["pattern_sha256"] for row in rows}
        _check_dispatch_budget(dataset, cfg, policy, hashes, int(planned["prio"]))
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
        current_jobs = _read(jobs_path)
        existing = next((job for job in current_jobs if job.get("store") == planned["store"]), None)
        if existing is not None:
            if existing != expected:
                raise ValueError(f"queue store collision after input copy: {planned['store']}")
            continue
        _check_dispatch_budget(dataset, cfg, policy, hashes, int(planned["prio"]))
        duplicate_check(destination, dataset)
        queue_add(dataset, planned, receipt["scope"])
    receipt["status"] = "dispatched"; receipt["dispatch_gate"] = "passed_per_shard"
    _write(path, receipt)
    active = path.parents[2] / "active_cycle.json"
    if active.is_file() and _read(active).get("receipt") == str(path):
        active.unlink()
    return path


def commit_dispatch(receipt_path: str | Path, *,
                    queue_add: Callable[[Path, Mapping[str, Any], str], None] = _dedust_add,
                    duplicate_check: Callable[[Path, Path], int] = pb.check_duplicates) -> Path:
    """Copy and enqueue a reviewed prepared receipt, idempotently per shard."""

    path = Path(receipt_path).resolve()
    receipt = _read(path)
    dataset = Path(receipt["dataset_root"]).resolve()
    scope = str(receipt["scope"])
    with _exclusive_lock(_dataset_lock(dataset, scope)):
        return _commit_dispatch_locked(path, queue_add=queue_add,
                                       duplicate_check=duplicate_check)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--local-workdir", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--profile-config", type=Path)
    parser.add_argument("--training-workdir", type=Path)
    parser.add_argument("--seed-input", type=Path, action="append")
    parser.add_argument("--prepared-blind-pool", type=Path)
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
                          prepared_blind_pool=args.prepared_blind_pool)
    print(result)


if __name__ == "__main__":
    main()
