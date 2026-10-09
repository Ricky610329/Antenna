from __future__ import annotations

from pathlib import Path

import pytest

from script import profiled_batch as pb
from script import symmetry_factory_cycle as cycle
from tests.test_symmetry_factory_cycle import (
    PROFILE,
    RETRY_WORKERS,
    _bundle,
    _complete,
    _dataset,
    _queued_single_error,
)


def _cutoff(dataset: Path):
    cfg = pb.load_profile_config(PROFILE)
    policy = cycle.factory.validate_factory_config(vars(cfg))
    return cfg, policy, cycle._queue_view(dataset, cfg, policy, RETRY_WORKERS)


def test_retryable_error_to_success_is_physically_validated_but_excluded_from_cutoff(
        tmp_path, monkeypatch):
    dataset = _dataset(tmp_path / "dataset")
    error_input, error_store, successful = _queued_single_error(dataset)
    stable_input = _bundle(dataset / "stable_input", 1, prefix="stable", start=901)
    stable_store = _complete(stable_input, dataset / "stable")
    jobs = pb.read_json(dataset / "jobs.json")
    cfg = pb.load_profile_config(PROFILE)
    jobs.append({"input": stable_input.name, "store": stable_store.name, "prio": 6,
                 "scope": cfg.scope, "config": f"{stable_input.name}/config.yaml"})
    pb.atomic_json(dataset / "jobs.json", jobs)
    cfg, policy, queue = _cutoff(dataset)
    error_row = pb.read_json(error_input / "manifest.json")[0]
    error_hash = error_row["pattern_sha256"]
    cutoff_order = cycle._success_order_from_proofs(queue, set())
    cutoff_valid = len(queue["successful_hashes"])
    cutoff_pending = set(queue["pending_hashes"] - queue["successful_hashes"])
    cutoff_remaining = (int(policy["target_valid_unique"]) - cutoff_valid -
                        len(cutoff_pending))
    calls = []
    original = pb.validate_store

    def validate(path, *args, **kwargs):
        calls.append(Path(path).resolve())
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pb, "validate_store", validate)
    pb.atomic_json(error_store / "results.json", successful)

    assert cycle._recheck_queue_proof(dataset, queue) == {error_hash}
    assert cycle._recheck_queue_proof(dataset, queue) == {error_hash}
    assert calls == [error_store.resolve()]
    assert len(queue["successful_hashes"]) == cutoff_valid == 1
    assert queue["pending_hashes"] - queue["successful_hashes"] == cutoff_pending == {
        error_hash}
    assert (int(policy["target_valid_unique"]) - len(queue["successful_hashes"]) -
            len(queue["pending_hashes"] - queue["successful_hashes"])) == cutoff_remaining
    assert cycle._success_order_from_proofs(queue, set()) == cutoff_order
    snapshot = cycle._snapshot_successes_from_proofs(
        queue, tmp_path / "snapshot", cutoff_order, cfg)
    snapshot_hashes = {
        row["pattern_sha256"] for row in pb.read_json(snapshot / "manifest.json")}
    assert snapshot_hashes == set(cutoff_order)
    assert error_hash not in snapshot_hashes
    with pytest.raises(ValueError, match="absent from the physical cutoff"):
        cycle._snapshot_successes_from_proofs(
            queue, tmp_path / "late-snapshot", [error_hash], cfg)


@pytest.mark.parametrize("transition", ["remove", "error_update"])
def test_retryable_error_remove_or_error_update_preserves_frozen_cutoff(
        tmp_path, transition):
    dataset = _dataset(tmp_path / "dataset")
    input_dir, store, _successful = _queued_single_error(dataset)
    _cfg, _policy, queue = _cutoff(dataset)
    row = pb.read_json(input_dir / "manifest.json")[0]
    before = {
        "successful": set(queue["successful_hashes"]),
        "pending": set(queue["pending_hashes"]),
        "order": cycle._success_order_from_proofs(queue, set()),
    }
    current = {} if transition == "remove" else {
        row["id"]: {"error": "watchdog_timeout: generated retry",
                     "error_kind": "hfss_simulation", "attempts": 4}}
    pb.atomic_json(store / "results.json", current)

    assert cycle._recheck_queue_proof(dataset, queue) == set()
    assert set(queue["successful_hashes"]) == before["successful"]
    assert set(queue["pending_hashes"]) == before["pending"]
    assert cycle._success_order_from_proofs(queue, set()) == before["order"]


@pytest.mark.parametrize("tamper", ["malformed_entry", "unknown_id"])
def test_retryable_change_rejects_malformed_or_unknown_current_results(tmp_path, tamper):
    dataset = _dataset(tmp_path / "dataset")
    input_dir, store, _successful = _queued_single_error(dataset)
    _cfg, _policy, queue = _cutoff(dataset)
    row = pb.read_json(input_dir / "manifest.json")[0]
    if tamper == "malformed_entry":
        current = {row["id"]: ["not", "an", "entry"]}
        message = "dict"
    else:
        current = {row["id"]: {"error": "retry", "attempts": 4},
                   "foreign-id": {"error": "foreign", "attempts": 4}}
        message = "unknown manifest ids"
    pb.atomic_json(store / "results.json", current)

    with pytest.raises(ValueError, match=message) as caught:
        cycle._recheck_queue_proof(dataset, queue)
    assert not isinstance(caught.value, cycle.LiveSnapshotChanged)


def test_terminal_error_transition_to_success_is_rejected(tmp_path):
    dataset = _dataset(tmp_path / "dataset")
    _input_dir, store, successful = _queued_single_error(dataset)
    pb.atomic_json(dataset / "jobs_state/failed.done", {"machine": RETRY_WORKERS[0]})
    _cfg, _policy, queue = _cutoff(dataset)
    assert queue["pair_proofs"][0]["terminal"] is True
    pb.atomic_json(store / "results.json", successful)

    with pytest.raises(ValueError, match="terminal store results changed"):
        cycle._recheck_queue_proof(dataset, queue)


def test_retryable_error_to_success_rejects_corrupt_current_raw(tmp_path):
    dataset = _dataset(tmp_path / "dataset")
    _input_dir, store, successful = _queued_single_error(dataset)
    _cfg, _policy, queue = _cutoff(dataset)
    entry = next(iter(successful.values()))
    pb.atomic_json(store / "results.json", successful)
    with (store / entry["sample_file"]).open("ab") as stream:
        stream.write(b"corrupt-late-success")

    with pytest.raises(ValueError, match="hash|內容") as caught:
        cycle._recheck_queue_proof(dataset, queue)
    assert not isinstance(caught.value, cycle.LiveSnapshotChanged)


def test_frozen_success_entry_and_raw_bytes_remain_immutable(tmp_path):
    dataset = _dataset(tmp_path / "dataset")
    input_dir = _bundle(dataset / "wave_input", 1)
    store = _complete(input_dir, dataset / "wave")
    cfg = pb.load_profile_config(PROFILE)
    pb.atomic_json(dataset / "jobs.json", [{
        "input": input_dir.name, "store": store.name, "prio": 1,
        "scope": cfg.scope, "config": f"{input_dir.name}/config.yaml"}])
    _cfg, _policy, queue = _cutoff(dataset)
    results = pb.read_json(store / "results.json")
    name = next(iter(results))
    results[name]["time_s"] += 1
    pb.atomic_json(store / "results.json", results)
    with pytest.raises(ValueError, match="frozen successful result entry"):
        cycle._recheck_queue_proof(dataset, queue)

    pb.atomic_json(store / "results.json", queue["pair_proofs"][0]["results"])
    entry = queue["pair_proofs"][0]["results"][name]
    with (store / entry["sample_file"]).open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="frozen successful sample"):
        cycle._recheck_queue_proof(dataset, queue)


def test_results_bytes_that_change_during_retry_validation_defer(tmp_path, monkeypatch):
    dataset = _dataset(tmp_path / "dataset")
    input_dir, store, successful = _queued_single_error(dataset)
    _cfg, _policy, queue = _cutoff(dataset)
    row = pb.read_json(input_dir / "manifest.json")[0]
    pb.atomic_json(store / "results.json", successful)
    original = pb.validate_store

    def validate_then_advance(path, *args, **kwargs):
        value = original(path, *args, **kwargs)
        pb.atomic_json(store / "results.json", {
            row["id"]: {"error": "watchdog_timeout: advanced during validation",
                         "error_kind": "hfss_simulation", "attempts": 4}})
        return value

    monkeypatch.setattr(pb, "validate_store", validate_then_advance)
    with pytest.raises(cycle.LiveSnapshotChanged, match="changed during appended-entry validation"):
        cycle._recheck_queue_proof(dataset, queue)

