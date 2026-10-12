"""Bounded local feedback controller for the provisional R81 S21 survey.

The controller reads one immutable ``ProvisionalS21Discrete`` release and its
three partial stores, but writes only below this checkout's ``tmp`` directory.
It triggers after at least 20 previously unconsumed, unique successful patterns,
trains the additive S21 model on a frozen success snapshot, and prepares one
local 60-row successor bundle (20 predicted worst-margin, 20 ensemble-
uncertainty, 20 deterministic random).  It never publishes a release, mutates a
NAS queue, launches a worker, or starts HFSS.

The existing 51-file worker closure knows only the original first 60 rows.  A
separately versioned release/registry/worker entry is therefore required before
any successor bundle produced here can run.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from antenna.measurement import measurement_id, score_spec_id
from script import exploration as ex
from script import filter_s21_training as s21
from script import profiled_batch as pb
from script import profiled_shards
from script import r81_provisional_filter_dispatch as dispatch
from script import r81_worker_entry as base


REPO = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
TRIGGER_NEW_SUCCESSES = 20
BATCH_SIZE = 60
ARM_COUNTS = {"best_min_margin": 20, "uncertainty": 20, "random": 20}
MAX_BATCHES = 3
DEFAULT_POOL_SIZE = 1024
DEFAULT_SEED = 811007
LEDGER = "cycle_ledger.json"
PLAN = "controller_plan.json"
PENDING = "pending_cycle.json"


def _sha(path: Path) -> str:
    return pb.file_sha256(Path(path))


def _json_id(value: Any) -> str:
    import hashlib
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _local_path(path: Path, *, must_exist: bool = False) -> Path:
    root = (REPO / "tmp").resolve()
    value = Path(path).resolve(strict=must_exist)
    try:
        value.relative_to(root)
    except ValueError as exc:
        raise ValueError("survey-cycle work_dir must stay below repository tmp") from exc
    return value


def _files(folder: Path) -> dict[str, str]:
    return {path.name: _sha(path)
            for path in sorted(Path(folder).iterdir(), key=lambda item: item.name)
            if path.is_file()}


def _release_context(release: Path, release_sha256: str) -> dict[str, Any]:
    context = dispatch._context(release, release_sha256)
    if context["release"]["policy"] != dispatch.POLICY:
        raise ValueError("survey source is not the exact provisional S21 policy")
    if len(context["entries"]) != 3:
        raise ValueError("survey source must retain exactly three pinned jobs")
    return context


def _profile_binding(context: Mapping[str, Any]) -> dict[str, str]:
    """Return the three profile hashes authenticated by the release closure."""
    canonical = Path(context["dataset"]) / dispatch.CANONICAL_INPUT
    expected = context["release"]["queue"]["canonical_files"]
    binding = {}
    for marker in ("config.yaml", "measurement.json", "score_spec.json"):
        digest = expected.get(marker)
        if not s21._is_sha(digest) or _sha(canonical / marker) != digest:
            raise ValueError(f"authenticated canonical profile changed: {marker}")
        binding[marker] = digest
    return binding


def _input_store_inventory(context: Mapping[str, Any]) -> dict[str, Any]:
    dataset = Path(context["dataset"])
    queued: dict[str, dict[str, Any]] = {}
    successes: dict[str, dict[str, Any]] = {}
    pairs: list[tuple[Path, Path]] = []
    sources = []
    for entry in context["entries"]:
        job = entry["job"]
        input_dir = (dataset / job["input"]).resolve()
        store_dir = (dataset / job["store"]).resolve()
        cfg = pb.load_profile_config(input_dir / "config.yaml")
        rows = pb.validate_input(input_dir, cfg)
        for row in rows:
            digest = row["pattern_sha256"]
            if digest in queued:
                raise ValueError("released survey queues contain duplicate physical patterns")
            queued[digest] = {"row": dict(row), "input": input_dir}
        source = {
            "input": str(input_dir), "store": str(store_dir),
            "input_manifest_sha256": _sha(input_dir / "manifest.json"),
            "store_present": store_dir.is_dir(),
        }
        if not store_dir.is_dir():
            sources.append(source)
            continue
        results_path = store_dir / "results.json"
        results = pb.validate_store(store_dir, require_complete=False)
        if pb.read_json(store_dir / "manifest.json") != rows:
            raise ValueError("survey store manifest differs from its released input")
        source.update({
            "store_manifest_sha256": _sha(store_dir / "manifest.json"),
            "store_results_sha256": (_sha(results_path) if results_path.is_file() else None),
        })
        # filter_s21_training hashes results.json before taking its native
        # partial snapshot. A not-yet-started store is valid survey state but
        # cannot be passed as an input/store pair until that marker exists.
        if results_path.is_file():
            pairs.append((input_dir, store_dir))
        for row in rows:
            result = results.get(row["id"])
            if not isinstance(result, dict) or result.get("status") != "ok" or "error" in result:
                continue
            digest = row["pattern_sha256"]
            if digest in successes:
                raise ValueError("survey stores contain duplicate successful physical patterns")
            successes[digest] = {
                "id": row["id"], "pattern_sha256": digest,
                "input": str(input_dir), "store": str(store_dir),
                "sample_sha256": result["sample_sha256"],
            }
        sources.append(source)
    return {"pairs": pairs, "queued": queued, "successes": successes,
            "sources": sources,
            "source_inventory_id": _json_id(sources)}


def _initial_ledger(context: Mapping[str, Any]) -> dict[str, Any]:
    release = context["release"]
    return {
        "schema_version": SCHEMA_VERSION,
        "controller": "r81_s21_survey_cycle_v1",
        "source_release": {"path": None,
                           "release_id": release["release_id"],
                           "sha256": context["release_sha256"]},
        "budget": {"batch_size": BATCH_SIZE, "max_batches": MAX_BATCHES,
                   "maximum_rows": BATCH_SIZE * MAX_BATCHES,
                   "original_batch": 1, "success_trigger": TRIGGER_NEW_SUCCESSES,
                   "arms": copy.deepcopy(ARM_COUNTS)},
        "consumed_successes": [],
        "selected_pattern_sha256": [],
        "prepared_batches": [],
        "publication": {"nas_written": False, "release_created": False,
                        "queue_changed": False, "worker_started": False,
                        "hfss_started": False},
        "handoff": {
            "launchable": False,
            "reason": "the current 51-file closure binds only the original first-60 release",
            "required": "separately versioned live-loop release/registry/worker entry",
        },
    }


def prepare(release: Path, release_sha256: str, work_dir: Path) -> dict[str, Any]:
    """Create a local controller ledger after authenticating the source release."""
    context = _release_context(release, release_sha256)
    work = _local_path(work_dir)
    if work.exists():
        raise FileExistsError("survey-cycle preparation requires a fresh work_dir")
    work.mkdir(parents=True)
    ledger = _initial_ledger(context)
    ledger["source_release"]["path"] = str(Path(release).resolve())
    plan = {
        "schema_version": SCHEMA_VERSION,
        "status": "prepared_local_controller",
        "source_release": copy.deepcopy(ledger["source_release"]),
        "trigger": "at least 20 new unique verified successes since the last consumed snapshot",
        "model": {"protocol": s21.PROTOCOL, "ranking": "minimum of three S21 margins",
                  "prior_only_ranking_allowed": False},
        "candidate_pool": {"default_size": DEFAULT_POOL_SIZE, "maximum_size": s21.MAX_RANK_CANDIDATES,
                           "generator": "exploration.generate_candidates dual native"},
        "selection": copy.deepcopy(ARM_COUNTS),
        "budget": copy.deepcopy(ledger["budget"]),
        "publication": copy.deepcopy(ledger["publication"]),
        "handoff": copy.deepcopy(ledger["handoff"]),
    }
    pb.atomic_json(work / LEDGER, ledger)
    pb.atomic_json(work / PLAN, plan)
    return plan


def _load_ledger(work_dir: Path, context: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    work = _local_path(work_dir, must_exist=True)
    ledger = pb.read_json(work / LEDGER)
    expected = _initial_ledger(context)
    if (ledger.get("schema_version") != SCHEMA_VERSION or
            ledger.get("controller") != expected["controller"] or
            ledger.get("budget") != expected["budget"] or
            ledger.get("publication") != expected["publication"] or
            ledger.get("handoff") != expected["handoff"] or
            ledger.get("source_release", {}).get("release_id") !=
            context["release"]["release_id"] or
            ledger.get("source_release", {}).get("sha256") != context["release_sha256"]):
        raise ValueError("survey-cycle ledger recipe or release binding changed")
    if (not isinstance(ledger.get("consumed_successes"), list) or
            not isinstance(ledger.get("selected_pattern_sha256"), list) or
            not isinstance(ledger.get("prepared_batches"), list)):
        raise ValueError("survey-cycle ledger state is malformed")
    consumed_rows = ledger["consumed_successes"]
    consumed_hashes = [item.get("pattern_sha256") for item in consumed_rows
                       if isinstance(item, dict)]
    if (len(consumed_rows) != len(consumed_hashes) or
            len(consumed_hashes) != len(set(consumed_hashes)) or
            any(set(item) != {"pattern_sha256", "sample_sha256"} or
                not s21._is_sha(item["pattern_sha256"]) or
                not s21._is_sha(item["sample_sha256"]) for item in consumed_rows)):
        raise ValueError("survey-cycle consumed successes are not unique pattern/sample bindings")
    selected_values = ledger["selected_pattern_sha256"]
    if (len(selected_values) != len(set(selected_values)) or
            any(not s21._is_sha(value) for value in selected_values)):
        raise ValueError("survey-cycle selected patterns are not a unique SHA-256 set")
    if len(ledger["prepared_batches"]) > MAX_BATCHES - 1:
        raise ValueError("survey-cycle ledger exceeds its three-batch budget")
    if ([item.get("batch") for item in ledger["prepared_batches"]] !=
            list(range(2, 2 + len(ledger["prepared_batches"]))) or
            any(item.get("launchable") is not False for item in ledger["prepared_batches"])):
        raise ValueError("survey-cycle prepared-batch sequence or handoff role changed")
    prepared_selected: set[str] = set()
    cycles_root = (work / "cycles").resolve()
    for item in ledger["prepared_batches"]:
        if set(item) != {"batch", "rows", "directory", "selection_receipt_sha256",
                        "canonical_manifest_sha256", "split_receipt_sha256", "launchable"}:
            raise ValueError("survey-cycle prepared-batch receipt fields changed")
        batch = int(item["batch"])
        folder = Path(item["directory"]).resolve()
        try:
            folder.relative_to(cycles_root)
        except ValueError as exc:
            raise ValueError("prepared survey cycle escapes its work directory") from exc
        if folder != cycles_root / f"batch-{batch:03d}" or not folder.is_dir():
            raise ValueError("prepared survey cycle directory is missing or renamed")
        receipt_path = folder / "selection_receipt.json"
        canonical_manifest = folder / f"batch-{batch:03d}_input" / "manifest.json"
        split_receipt = folder / f"batch-{batch:03d}_split" / "split_receipt.json"
        if (_sha(receipt_path) != item["selection_receipt_sha256"] or
                _sha(canonical_manifest) != item["canonical_manifest_sha256"] or
                _sha(split_receipt) != item["split_receipt_sha256"]):
            raise ValueError("prepared survey cycle receipt or bundle changed")
        receipt = pb.read_json(receipt_path)
        rows = receipt.get("selected")
        hashes = ([row.get("pattern_sha256") for row in rows]
                  if isinstance(rows, list) else [])
        if (receipt.get("batch") != batch or item["rows"] != BATCH_SIZE or
                len(hashes) != BATCH_SIZE or len(set(hashes)) != BATCH_SIZE or
                any(not s21._is_sha(value) for value in hashes) or
                prepared_selected.intersection(hashes)):
            raise ValueError("prepared survey cycle selected rows changed")
        prepared_selected.update(hashes)
    if prepared_selected != set(selected_values):
        raise ValueError("survey-cycle selected hash ledger differs from prepared receipts")
    return work, ledger


def inspect(release: Path, release_sha256: str, work_dir: Path) -> dict[str, Any]:
    """Read one stable metadata snapshot and report whether a cycle may run."""
    context = _release_context(release, release_sha256)
    _work, ledger = _load_ledger(work_dir, context)
    inventory = _input_store_inventory(context)
    consumed_bindings = {item["pattern_sha256"]: item["sample_sha256"]
                         for item in ledger["consumed_successes"]}
    consumed = set(consumed_bindings)
    current = set(inventory["successes"])
    queued = set(inventory["queued"])
    selected = set(ledger["selected_pattern_sha256"])
    if (not consumed <= current or
            any(inventory["successes"][digest]["sample_sha256"] != sample_sha
                for digest, sample_sha in consumed_bindings.items()) or
            selected & queued):
        raise ValueError("survey-cycle ledger success/selection hashes contradict the source queue")
    new = sorted(current - consumed)
    next_batch = 2 + len(ledger["prepared_batches"])
    exhausted = next_batch > MAX_BATCHES
    handoff_required = bool(ledger["prepared_batches"]) and not exhausted
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "budget_exhausted" if exhausted else
                  ("handoff_required" if handoff_required else
                  ("ready" if len(new) >= TRIGGER_NEW_SUCCESSES else "waiting")),
        "source_release_sha256": context["release_sha256"],
        "next_batch": None if (exhausted or handoff_required) else next_batch,
        "unique_successes": len(current), "consumed_successes": len(consumed),
        "new_unique_successes": len(new), "trigger": TRIGGER_NEW_SUCCESSES,
        "new_success_pattern_sha256": new,
        "queued_unique_patterns": len(inventory["queued"]),
        "source_inventory_id": inventory["source_inventory_id"],
        "publication": copy.deepcopy(ledger["publication"]),
        "handoff": copy.deepcopy(ledger["handoff"]),
    }


def _recover_pending(work: Path, ledger: Mapping[str, Any],
                     context: Mapping[str, Any]) -> dict[str, Any]:
    """Finish an authenticated temp->final->ledger transaction after a crash."""
    path = work / PENDING
    if not path.is_file():
        return dict(ledger)
    pending = pb.read_json(path)
    if (pending.get("schema_version") != SCHEMA_VERSION or
            pending.get("source_release_sha256") != context["release_sha256"] or
            not isinstance(pending.get("ledger_after"), dict)):
        raise ValueError("pending survey transaction binding changed")
    after = pending["ledger_after"]
    current_id = _json_id(ledger)
    already_committed = current_id == _json_id(after)
    if not already_committed and pending.get("ledger_before_id") != current_id:
        raise ValueError("pending survey transaction ledger binding changed")
    temp = Path(pending.get("temp_dir", "")).resolve()
    final = Path(pending.get("cycle_dir", "")).resolve()
    cycles = (work / "cycles").resolve()
    try:
        temp.relative_to(cycles); final.relative_to(cycles)
    except ValueError as exc:
        raise ValueError("pending survey transaction escapes the work directory") from exc
    if final.exists() and temp.exists():
        raise ValueError("pending survey transaction has both temp and final directories")
    if not final.exists() and temp.exists():
        if _sha(temp / "selection_receipt.json") != pending.get("selection_receipt_sha256"):
            raise ValueError("pending temp selection receipt changed")
        temp.rename(final)
    elif not final.exists():
        # An in-process failure may have removed its temp before returning.
        path.unlink()
        return dict(ledger)
    if _sha(final / "selection_receipt.json") != pending.get("selection_receipt_sha256"):
        raise ValueError("pending final selection receipt changed")
    prepared = after.get("prepared_batches", [])
    if not prepared or prepared[-1].get("directory") != str(final):
        raise ValueError("pending ledger does not bind its final cycle directory")
    last = prepared[-1]
    batch = int(last["batch"])
    if (_sha(final / f"batch-{batch:03d}_input" / "manifest.json") !=
            last.get("canonical_manifest_sha256") or
            _sha(final / f"batch-{batch:03d}_split" / "split_receipt.json") !=
            last.get("split_receipt_sha256")):
        raise ValueError("pending successor bundle hashes changed")
    if not already_committed:
        pb.atomic_json(work / LEDGER, after)
    path.unlink()
    return after


def _seed_rows(inventory: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for digest, item in sorted(inventory["queued"].items()):
        row = item["row"]
        pattern = torch.load(item["input"] / row["pattern_file"],
                             weights_only=True, map_location="cpu")
        if pb.pattern_sha256(pattern) != digest:
            raise ValueError("queued source pattern changed while constructing candidates")
        rows.append({"id": row["id"], "pattern": np.asarray(pattern, dtype=bool),
                     "pattern_sha256": digest,
                     "lineage_id": row.get("lineage_id", digest),
                     "canonical_group_id": row.get("canonical_group_id"),
                     "source": "released_s21_queue"})
    return rows


def _optional_seed_rows(seed_inputs: Sequence[Path], excluded: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load explicit same-profile geometries that remain eligible for wide remeasurement."""
    rows: list[dict[str, Any]] = []
    bindings = []
    seen = set(excluded)
    for raw in seed_inputs:
        folder = Path(raw).resolve(strict=True)
        s21.load_protocol(folder / "config.yaml")
        cfg = pb.load_profile_config(folder / "config.yaml")
        manifest = pb.validate_input(folder, cfg)
        kept = 0
        for source in manifest:
            digest = source["pattern_sha256"]
            if digest in seen:
                continue
            pattern = torch.load(folder / source["pattern_file"],
                                 weights_only=True, map_location="cpu")
            if pb.pattern_sha256(pattern) != digest:
                raise ValueError("optional S21 seed pattern hash changed")
            seen.add(digest); kept += 1
            rows.append({
                "id": source["id"], "pattern": np.asarray(pattern, dtype=bool),
                "pattern_sha256": digest,
                "lineage_id": source.get("lineage_id", digest),
                "canonical_group_id": source.get("canonical_group_id"),
                "candidate_group": "pending_stop_remeasurement",
                "source": "narrow_history_geometry_eligible_for_new_wide_truth",
            })
        bindings.append({"path": str(folder),
                         "manifest_sha256": _sha(folder / "manifest.json"),
                         "files": _files(folder),
                         "manifest_count": len(manifest), "eligible_count": kept,
                         "historical_stop_truth_available": False})
    return rows, bindings


def _candidate_pool(seed_rows: Sequence[Mapping[str, Any]], excluded: set[str],
                    count: int, seed: int,
                    direct_candidates: Sequence[Mapping[str, Any]] = ()) -> list[dict[str, Any]]:
    if count < BATCH_SIZE or count > s21.MAX_RANK_CANDIDATES:
        raise ValueError(f"candidate_pool_size must be {BATCH_SIZE}..{s21.MAX_RANK_CANDIDATES}")
    pool: list[dict[str, Any]] = []
    seen = set(excluded)
    for candidate in direct_candidates:
        if len(pool) == count:
            break
        digest = candidate["pattern_sha256"]
        if digest in seen:
            continue
        pattern = ex.enforce_dual_geometry(candidate["pattern"])
        if pb.pattern_sha256(pattern) != digest:
            raise ValueError("direct candidate pattern hash changed")
        seen.add(digest)
        row = dict(candidate)
        row["id"] = f"r81s21_pool_{len(pool):04d}_{digest[:10]}"
        row["pattern"] = pattern
        pool.append(row)
    round_index = 0
    while len(pool) < count and round_index < 16:
        generated = ex.generate_candidates(seed_rows, count, seed=seed + round_index * 1009,
                                           single=False)
        for candidate in generated:
            digest = candidate["pattern_sha256"]
            if digest in seen:
                continue
            seen.add(digest)
            row = dict(candidate)
            row["id"] = f"r81s21_pool_{len(pool):04d}_{digest[:10]}"
            row["pattern"] = np.asarray(candidate["pattern"], dtype=bool)
            pool.append(row)
            if len(pool) == count:
                break
        round_index += 1
    if len(pool) != count:
        raise RuntimeError(f"could construct only {len(pool)}/{count} non-overlapping candidates")
    return pool


def _ensemble_uncertainty(model_dir: Path, patterns: Sequence[np.ndarray],
                          expected_mean: np.ndarray, summary: Mapping[str, Any]) -> np.ndarray:
    x = torch.as_tensor(np.asarray(patterns), dtype=torch.float32).reshape(len(patterns), -1)
    members = []
    for member_seed in s21.ENSEMBLE_SEEDS:
        saved = torch.load(model_dir / f"member-{member_seed}.pt",
                           weights_only=False, map_location="cpu")
        s21._checkpoint_ok(saved, summary["signature"], member_seed, rank_ready=True,
                           trained_mask=[True, True, True])
        model = ex.CurveMLP(625, summary["recipe"]["hidden_dims"], s21.OUTPUT_DIM)
        model.load_state_dict(saved["model_state"])
        model.eval()
        with torch.no_grad():
            members.append(model((x - saved["mean"]) / saved["std"]).numpy())
    stacked = np.stack(members)
    mean = stacked.mean(axis=0)
    if not np.allclose(mean, expected_mean, rtol=0, atol=1e-6):
        raise ValueError("uncertainty replay differs from the bound ranking predictions")
    uncertainty = stacked.std(axis=0).mean(axis=1)
    if uncertainty.shape != (len(patterns),) or not np.isfinite(uncertainty).all():
        raise ValueError("S21 ensemble uncertainty is non-finite")
    return uncertainty


def _select(pool: Sequence[Mapping[str, Any]], prediction: np.ndarray, order: np.ndarray,
            uncertainty: np.ndarray, *, seed: int) -> list[dict[str, Any]]:
    used: set[int] = set()
    chosen: list[tuple[str, int]] = []
    for index in order.tolist():
        if len(chosen) == ARM_COUNTS["best_min_margin"]:
            break
        used.add(index); chosen.append(("best_min_margin", index))
    fresh = [i for i in range(len(pool)) if i not in used and
             pool[i].get("candidate_group") == "fresh_random"]
    if len(fresh) < ARM_COUNTS["random"]:
        raise ValueError("S21 random arm lacks 20 unused fresh-random candidates")
    rng = np.random.default_rng(seed)
    random_indices = rng.permutation(fresh)[:ARM_COUNTS["random"]].tolist()
    reserved_random = set(random_indices)
    available = [i for i in range(len(pool)) if i not in used and i not in reserved_random]
    for index in sorted(available, key=lambda i: (-float(uncertainty[i]), i))[
            :ARM_COUNTS["uncertainty"]]:
        used.add(index); chosen.append(("uncertainty", index))
    for index in random_indices:
        used.add(index); chosen.append(("random", index))
    if len(chosen) != BATCH_SIZE or len(used) != BATCH_SIZE:
        raise ValueError("S21 mixed selection did not produce 60 disjoint rows")
    selected = []
    for arm, index in chosen:
        row = dict(pool[index])
        row.update({"selection_arm": arm,
                    "prediction": prediction[index].astype(float).tolist(),
                    "predicted_worst_margin_db": float(prediction[index].min()),
                    "predicted_uncertainty": float(uncertainty[index]),
                    "pool_index": index})
        selected.append(row)
    return selected


def _write_pool(folder: Path, pool: Sequence[Mapping[str, Any]], prediction: np.ndarray,
                uncertainty: np.ndarray, rank_receipt: Mapping[str, Any]) -> dict[str, Any]:
    patterns = folder / "patterns"
    patterns.mkdir(parents=True)
    rows = []
    for index, candidate in enumerate(pool):
        relative = f"patterns/{candidate['id']}.pt"
        path = folder / relative
        torch.save(torch.as_tensor(candidate["pattern"], dtype=torch.float32), path)
        rows.append({
            "id": candidate["id"], "pattern_file": relative,
            "pattern_file_sha256": _sha(path),
            "pattern_sha256": candidate["pattern_sha256"],
            "candidate_group": candidate.get("candidate_group"),
            "parent_id": candidate.get("parent_id"),
            "lineage_id": candidate.get("lineage_id"),
            "prediction": prediction[index].astype(float).tolist(),
            "predicted_worst_margin_db": float(prediction[index].min()),
            "predicted_uncertainty": float(uncertainty[index]),
        })
    pb.atomic_json(folder / "manifest.json", rows)
    pb.atomic_json(folder / "rank_receipt.json", dict(rank_receipt))
    receipt = {
        "schema_version": SCHEMA_VERSION, "candidate_count": len(rows),
        "manifest_sha256": _sha(folder / "manifest.json"),
        "rank_receipt_sha256": _sha(folder / "rank_receipt.json"),
        "candidate_set_id": rank_receipt["candidate_set_id"],
        "pattern_files_id": _json_id([{"id": row["id"],
                                       "file_sha256": row["pattern_file_sha256"]}
                                      for row in rows]),
    }
    pb.atomic_json(folder / "pool_receipt.json", receipt)
    return receipt


def _write_next_bundle(folder: Path, selected: Sequence[Mapping[str, Any]], batch: int,
                       model_signature: str, candidate_set_id: str,
                       profile_input: Path,
                       expected_profile_files: Mapping[str, str]) -> dict[str, Any]:
    canonical = folder / f"batch-{batch:03d}_input"
    canonical.mkdir(parents=True)
    for marker in ("config.yaml", "measurement.json", "score_spec.json"):
        if _sha(Path(profile_input) / marker) != expected_profile_files.get(marker):
            raise ValueError(f"source canonical profile changed before successor copy: {marker}")
        shutil.copy2(Path(profile_input) / marker, canonical / marker)
        if _sha(canonical / marker) != expected_profile_files[marker]:
            raise ValueError(f"successor profile copy differs: {marker}")
    cfg = pb.load_profile_config(canonical / "config.yaml")
    mid, sid = measurement_id(cfg.measurement), score_spec_id(cfg.score_spec)
    rows = []
    for ordinal, candidate in enumerate(selected):
        digest = candidate["pattern_sha256"]
        name = f"r81s21_b{batch:02d}_{ordinal:03d}_{digest[:8]}"
        pattern = torch.as_tensor(candidate["pattern"], dtype=torch.float32)
        torch.save(pattern, canonical / f"{name}.pt")
        rows.append({
            "id": name, "batch": batch, "pattern_file": f"{name}.pt",
            "pattern_sha256": digest, "port": "dual",
            "measurement_id": mid, "score_spec_id": sid,
            "geometry_profile": "dual25_p01_db075",
            "lineage_id": candidate.get("lineage_id", digest),
            "canonical_group_id": candidate.get("canonical_group_id") or digest,
            "selection_arm": candidate["selection_arm"],
            "candidate_id": candidate["id"], "pool_index": candidate["pool_index"],
            "model_signature": model_signature, "candidate_set_id": candidate_set_id,
            "prediction": candidate["prediction"],
            "predicted_worst_margin_db": candidate["predicted_worst_margin_db"],
            "predicted_uncertainty": candidate["predicted_uncertainty"],
        })
    pb.atomic_json(canonical / "manifest.json", rows)
    pb.validate_input(canonical, cfg)
    split = folder / f"batch-{batch:03d}_split"
    profiled_shards.split_bundle(canonical, split, shard_size=20)
    jobs = []
    for shard, machine in enumerate(base.EXPECTED_WORKERS, start=1):
        jobs.append({
            "input": f"{split.name}/shard-{shard:03d}_input",
            "store": f"dedust_r81s21provisional_b{batch:02d}_{shard:02d}",
            "prio": dispatch.LOW_TIER_PRIORITY,
            "config": f"{split.name}/shard-{shard:03d}_input/config.yaml",
            "scope": dispatch.SCOPE, "machine": machine,
            "launchable": False,
        })
    pb.atomic_json(folder / "jobs_template.json", jobs)
    return {"canonical": canonical.name, "canonical_files": _files(canonical),
            "source_profile_files": dict(expected_profile_files),
            "split": split.name,
            "split_receipt_sha256": _sha(split / "split_receipt.json"),
            "jobs_template_sha256": _sha(folder / "jobs_template.json")}


def _trained_snapshot_successes(summary: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
    snapshot_info = summary.get("snapshot")
    if not isinstance(snapshot_info, dict) or not isinstance(snapshot_info.get("snapshot"), str):
        raise ValueError("trained S21 summary lacks its frozen current snapshot")
    snapshot = Path(snapshot_info["snapshot"]).resolve()
    protocol = s21.load_protocol()
    current = s21.load_current_snapshot(snapshot, protocol)
    results = pb.read_json(snapshot / "results.json")
    samples = {row["pattern_sha256"]: results[row["id"]]["sample_sha256"]
               for row in current.rows}
    if len(samples) != len(current.rows):
        raise ValueError("trained S21 snapshot contains duplicate physical patterns")
    return samples, current.receipt


def single_cycle(*, release: Path, release_sha256: str, work_dir: Path,
                 prior_manifest: Path | None = None, prior_root: Path | None = None,
                 expected_prior_sha256: str | None = None,
                 model_output_dir: Path | None = None,
                 seed_inputs: Sequence[Path] = (),
                 candidate_pool_size: int = DEFAULT_POOL_SIZE,
                 seed: int = DEFAULT_SEED, pretrain_epochs: int = 30,
                 epochs: int = 100, batch_train_size: int = 32,
                 hidden_dims: Sequence[int] = (128, 128), dry_run: bool = False) -> dict[str, Any]:
    """Inspect once; optionally train and prepare one immutable local successor."""
    if candidate_pool_size < BATCH_SIZE or candidate_pool_size > s21.MAX_RANK_CANDIDATES:
        raise ValueError(f"candidate_pool_size must be {BATCH_SIZE}..{s21.MAX_RANK_CANDIDATES}")
    context = _release_context(release, release_sha256)
    work, ledger = _load_ledger(work_dir, context)
    if (work / PENDING).is_file():
        if dry_run:
            raise ValueError("dry-run cannot recover a pending survey transaction")
        ledger = _recover_pending(work, ledger, context)
    status = inspect(release, release_sha256, work)
    if status["status"] != "ready":
        return status
    if dry_run:
        return {**status, "status": "dry_run_ready", "would_train": True,
                "would_prepare_rows": BATCH_SIZE, "candidate_pool_size": candidate_pool_size}
    if prior_manifest is None or prior_root is None or expected_prior_sha256 is None:
        raise ValueError("single-cycle execution requires prior manifest, root, and SHA-256")
    inventory = _input_store_inventory(context)
    if inventory["source_inventory_id"] != status["source_inventory_id"]:
        raise ValueError("survey stores changed between readiness and cycle cutoff")
    if not inventory["pairs"]:
        raise ValueError("ready survey has no physical input/store pair")
    batch = int(status["next_batch"])
    profile_binding = _profile_binding(context)
    cycle = work / "cycles" / f"batch-{batch:03d}"
    if cycle.exists():
        raise FileExistsError("unledgered/final successor directory already exists")
    temp = work / "cycles" / f".batch-{batch:03d}-{uuid.uuid4().hex}"
    temp.mkdir(parents=True, exist_ok=False)
    transaction_started = False
    try:
        training_root = (_local_path(model_output_dir, must_exist=True)
                         if model_output_dir is not None else work / "models")
        paths, summary = s21.train(
            input_store_pairs=inventory["pairs"],
            split_input=Path(context["dataset"]) / dispatch.CANONICAL_INPUT,
            output_dir=training_root, prior_manifest=prior_manifest,
            prior_root=prior_root, expected_prior_sha256=expected_prior_sha256,
            min_successes=TRIGGER_NEW_SUCCESSES,
            pretrain_epochs=pretrain_epochs, epochs=epochs,
            batch_size=batch_train_size, hidden_dims=hidden_dims,
            seed=seed)
        snapshot_successes, snapshot_receipt = _trained_snapshot_successes(summary)
        trigger_successes = set(inventory["successes"])
        if not trigger_successes <= set(snapshot_successes):
            raise ValueError("trained snapshot omitted a success from the execution cutoff")
        model_dir = paths[0].parent
        completion_sha = _sha(model_dir / "completion_receipt.json")
        excluded = set(inventory["queued"]) | set(ledger["selected_pattern_sha256"])
        optional_seeds, seed_bindings = _optional_seed_rows(seed_inputs, excluded)
        generation_seeds = [*_seed_rows(inventory), *optional_seeds]
        pool = _candidate_pool(generation_seeds, excluded,
                               candidate_pool_size, seed + batch * 1009,
                               direct_candidates=optional_seeds)
        prediction, order, rank_receipt = s21.predict_and_rank(
            model_dir, pool, expected_completion_sha256=completion_sha)
        uncertainty = _ensemble_uncertainty(
            model_dir, [candidate["pattern"] for candidate in pool], prediction, summary)
        if (_sha(model_dir / "completion_receipt.json") != completion_sha or
                any(_sha(model_dir / name) != digest
                    for name, digest in summary["model_sha256"].items())):
            raise ValueError("S21 models changed during uncertainty replay")
        selected = _select(pool, prediction, order, uncertainty,
                           seed=seed + batch * 2003)
        if set(candidate["pattern_sha256"] for candidate in selected) & excluded:
            raise ValueError("successor selection overlaps a measured/queued/prior-selected pattern")
        pool_receipt = _write_pool(temp / "candidate_pool", pool, prediction,
                                   uncertainty, rank_receipt)
        bundle = _write_next_bundle(temp, selected, batch, summary["signature"],
                                    rank_receipt["candidate_set_id"],
                                    Path(context["dataset"]) / dispatch.CANONICAL_INPUT,
                                    profile_binding)
        selection_rows = [{key: candidate[key] for key in (
            "id", "pattern_sha256", "candidate_group", "selection_arm", "pool_index", "prediction",
            "predicted_worst_margin_db", "predicted_uncertainty")}
                          for candidate in selected]
        selection_receipt = {
            "schema_version": SCHEMA_VERSION, "batch": batch,
            "source_release_sha256": context["release_sha256"],
            "trigger_source_inventory_id": inventory["source_inventory_id"],
            "trigger_source_bindings": inventory["sources"],
            "trigger_new_success_pattern_sha256": status["new_success_pattern_sha256"],
            "trigger_success_pattern_sha256": sorted(inventory["successes"]),
            "trained_snapshot": {"receipt": snapshot_receipt,
                                 "successes": [{"pattern_sha256": digest,
                                                "sample_sha256": snapshot_successes[digest]}
                                               for digest in sorted(snapshot_successes)]},
            "optional_narrow_history_seeds": seed_bindings,
            "narrow_history_policy": "eligible for new 16..40 GHz truth; never treated as observed stop truth",
            "excluded_measured_or_queued_pattern_sha256": sorted(excluded),
            "model": {"signature": summary["signature"],
                      "completion_receipt_sha256": completion_sha,
                      "model_sha256": summary["model_sha256"]},
            "pool": pool_receipt,
            "ranking": rank_receipt,
            "uncertainty": "mean per-output population std across the two bound ensemble members",
            "arms": copy.deepcopy(ARM_COUNTS), "selected": selection_rows,
            "selected_id": _json_id(selection_rows),
            "bundle": bundle,
            "claims": {"provisional_only": True, "formal_pass": False,
                       "survey_complete_is_spec_pass": False,
                       "live_handoff_available": False},
        }
        pb.atomic_json(temp / "selection_receipt.json", selection_receipt)
        final_context = _release_context(release, release_sha256)
        if _profile_binding(final_context) != profile_binding:
            raise ValueError("authenticated canonical profile changed during the cycle")
        final_inventory = _input_store_inventory(final_context)
        if (not set(snapshot_successes) <= set(final_inventory["successes"]) or
                any(final_inventory["successes"][digest]["sample_sha256"] != sample_sha
                    for digest, sample_sha in snapshot_successes.items())):
            raise ValueError("a frozen snapshot success changed in the live survey stores")
        selection_receipt["post_source_inventory_id"] = final_inventory["source_inventory_id"]
        selection_receipt["post_source_bindings"] = final_inventory["sources"]
        selection_receipt["post_snapshot_new_success_pattern_sha256"] = sorted(
            set(final_inventory["successes"]) - set(snapshot_successes))
        pb.atomic_json(temp / "selection_receipt.json", selection_receipt)
        (work / "cycles").mkdir(exist_ok=True)
        new_ledger = copy.deepcopy(ledger)
        consumed = {item["pattern_sha256"]: item["sample_sha256"]
                    for item in ledger["consumed_successes"]}
        for digest, sample_sha in snapshot_successes.items():
            previous = consumed.get(digest)
            if previous not in (None, sample_sha):
                raise ValueError("frozen snapshot contradicts an earlier consumed observation")
            consumed[digest] = sample_sha
        new_ledger["consumed_successes"] = [
            {"pattern_sha256": digest, "sample_sha256": consumed[digest]}
            for digest in sorted(consumed)]
        new_ledger["selected_pattern_sha256"] = sorted(
            set(ledger["selected_pattern_sha256"]) |
            {candidate["pattern_sha256"] for candidate in selected})
        new_ledger["prepared_batches"].append({
            "batch": batch, "rows": BATCH_SIZE, "directory": str(cycle),
            "selection_receipt_sha256": _sha(temp / "selection_receipt.json"),
            "canonical_manifest_sha256": _sha(
                temp / f"batch-{batch:03d}_input" / "manifest.json"),
            "split_receipt_sha256": bundle["split_receipt_sha256"],
            "launchable": False,
        })
        pending = {
            "schema_version": SCHEMA_VERSION,
            "source_release_sha256": context["release_sha256"],
            "ledger_before_id": _json_id(ledger),
            "ledger_after": new_ledger,
            "temp_dir": str(temp), "cycle_dir": str(cycle),
            "selection_receipt_sha256": _sha(temp / "selection_receipt.json"),
        }
        pb.atomic_json(work / PENDING, pending)
        transaction_started = True
        temp.rename(cycle)
        pb.atomic_json(work / LEDGER, new_ledger)
        (work / PENDING).unlink()
        completed_receipt = cycle / "selection_receipt.json"
        return {"schema_version": SCHEMA_VERSION, "status": "prepared_local_successor",
                "batch": batch, "rows": BATCH_SIZE, "cycle_dir": str(cycle),
                "selection_receipt_sha256": _sha(completed_receipt),
                "publication": copy.deepcopy(new_ledger["publication"]),
                "handoff": copy.deepcopy(new_ledger["handoff"])}
    except Exception as exc:
        if temp.exists() and not transaction_started:
            resolved_temp = temp.resolve()
            cycles_root = (work / "cycles").resolve()
            try:
                resolved_temp.relative_to(cycles_root)
            except ValueError as safety_exc:
                raise RuntimeError("refusing to move failed temp outside controller work") from safety_exc
            failed = cycles_root / f"failed-{temp.name.lstrip('.')}-{uuid.uuid4().hex[:8]}"
            temp.rename(failed)
            pb.atomic_json(failed / "failure.json", {
                "schema_version": SCHEMA_VERSION,
                "error_type": type(exc).__name__, "error": str(exc),
                "publication": {"nas_written": False, "release_created": False,
                                "queue_changed": False, "worker_started": False,
                                "hfss_started": False},
            })
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "dry-run", "single-cycle"):
        item = sub.add_parser(name)
        item.add_argument("--release", type=Path, required=True)
        item.add_argument("--release-sha256", required=True)
        item.add_argument("--work-dir", type=Path, required=True)
        if name in ("dry-run", "single-cycle"):
            item.add_argument("--candidate-pool-size", type=int, default=DEFAULT_POOL_SIZE)
            item.add_argument("--seed", type=int, default=DEFAULT_SEED)
        if name == "single-cycle":
            item.add_argument("--prior-manifest", type=Path, required=True)
            item.add_argument("--prior-root", type=Path, required=True)
            item.add_argument("--expected-prior-sha256", required=True)
            item.add_argument("--model-output-dir", type=Path)
            item.add_argument("--seed-input", type=Path, action="append", default=[])
            item.add_argument("--pretrain-epochs", type=int, default=30)
            item.add_argument("--epochs", type=int, default=100)
            item.add_argument("--batch-train-size", type=int, default=32)
            item.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 128])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.release, args.release_sha256, args.work_dir)
    elif args.command == "dry-run":
        result = single_cycle(release=args.release, release_sha256=args.release_sha256,
                              work_dir=args.work_dir,
                              candidate_pool_size=args.candidate_pool_size,
                              seed=args.seed, dry_run=True)
    else:
        result = single_cycle(
            release=args.release, release_sha256=args.release_sha256,
            work_dir=args.work_dir, prior_manifest=args.prior_manifest,
            prior_root=args.prior_root,
            expected_prior_sha256=args.expected_prior_sha256,
            model_output_dir=args.model_output_dir,
            seed_inputs=args.seed_input,
            candidate_pool_size=args.candidate_pool_size, seed=args.seed,
            pretrain_epochs=args.pretrain_epochs, epochs=args.epochs,
            batch_train_size=args.batch_train_size, hidden_dims=args.hidden_dims)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
