"""Focused generated lifecycle tests for the additive R81 formal sharded entry."""
from __future__ import annotations

import copy
import shutil
from pathlib import Path

import pytest
import torch

from script import exploration as ex
from script import profiled_batch as pb
from script import profiled_shards
from script import r81_formal_sharded_entry as formal
from script import r81_worker_entry as base
from tests import test_r81_worker_entry as fixtures


@pytest.fixture(autouse=True)
def _future_committed_runtime(monkeypatch):
    monkeypatch.setattr(base, "_git_is_ancestor", lambda _commit: True)
    monkeypatch.setattr(base, "_git_path_clean", lambda _commit, _relative: True)
    monkeypatch.setattr(base, "_git_blob_sha",
                        lambda _commit, relative: pb.file_sha256(base.REPO / relative))
    fixtures._CENSUS_FIXTURES.clear()

    def bound_census(binding, _commit):
        path = fixtures._CENSUS_FIXTURES[binding["repo_path"]]
        assert pb.file_sha256(path) == binding["sha256"]
        return path, pb.read_json(path)

    monkeypatch.setattr(base, "_bound_census_status", bound_census)


def _files(folder: Path) -> dict[str, str]:
    return {path.name: pb.file_sha256(path) for path in folder.iterdir() if path.is_file()}


def _mark_done(dataset: Path, entry: dict, machine: str = base.EXPECTED_WORKERS[0]) -> None:
    state = dataset / "jobs_state"
    state.mkdir(exist_ok=True)
    store = entry["job"]["store"]
    pb.atomic_json(state / f"{store}.claim", {"machine": machine, "at": "fixture"})
    pb.atomic_json(state / f"{store}.done",
                   {"machine": machine, "at": "fixture", "errors": 0, "error_ids": []})


def _complete_shard(dataset: Path, entry: dict) -> None:
    input_dir = dataset / Path(entry["job"]["input"])
    store = dataset / entry["job"]["store"]
    cfg = pb.load_profile_config(input_dir / "config.yaml")
    rows = pb.prepare_store(input_dir, store, cfg)
    response = torch.full((3, 49), -9.0)
    results = {}
    for row in rows:
        pattern = torch.load(input_dir / row["pattern_file"], weights_only=True)
        result = pb.observation(response, None, row, pattern, cfg, elapsed=1.0)
        sample = store / result["sample_file"]
        torch.save((pattern, response), sample)
        result["sample_sha256"] = pb.file_sha256(sample)
        results[row["id"]] = result
    pb.atomic_json(store / "results.json", results)
    _mark_done(dataset, entry)


def _release(tmp_path: Path):
    path, release = fixtures._release(tmp_path, "Formal")
    dataset = Path(release["dataset_root"])
    engineering = [item for item in release["queue"]["jobs"]
                   if item["role"] in base.ENGINEERING]
    for entry in engineering:
        _mark_done(dataset, entry)
    canonical = dataset / "dedust_r81b1_input"
    split_root = dataset / "dedust_r81b1_shards"
    shards = profiled_shards.split_bundle(canonical, split_root, shard_size=20)
    formal_entries = []
    for number, shard in enumerate(shards, start=1):
        input_name = shard.relative_to(dataset).as_posix()
        store = f"dedust_r81b1s{number:02d}"
        formal_entries.append({
            "role": "formal_shard", "batch": 1, "shard": number,
            "job": {"input": input_name, "store": store, "prio": 5,
                    "config": f"{input_name}/config.yaml", "scope": base.SCOPE},
            "input_files": _files(shard),
        })
    entries = engineering + formal_entries
    pb.atomic_json(dataset / "jobs.json", [entry["job"] for entry in entries])
    release["schema_version"] = 1
    release["stage"] = formal.STAGE
    release["queue"]["jobs"] = entries
    release["queue"]["jobs_sha256"] = pb.file_sha256(dataset / "jobs.json")
    release["queue"]["formal_shards"] = {
        "1": {"canonical_input": canonical.name,
              "canonical_input_files": _files(canonical),
              "split_root": split_root.name,
              "split_receipt_sha256": pb.file_sha256(split_root / "split_receipt.json"),
              "logical_store": "dedust_r81b1"}}
    required = formal._runtime_files(release["queue"]["exploration"])
    release["runtime"]["files"] = {name: pb.file_sha256(base.REPO / name)
                                     for name in required}
    pb.atomic_json(path, release)
    return path, release, formal_entries


def _rewrite(path: Path, release: dict) -> str:
    pb.atomic_json(path, release)
    return pb.file_sha256(path)


def _reuse_release(tmp_path: Path):
    source_root = tmp_path / "source"
    sibling_root = tmp_path / "sibling"
    source_root.mkdir(parents=True)
    sibling_root.mkdir(parents=True)
    source_path, source_release = fixtures._release(source_root, "EngineeringOnly")
    source_dataset = Path(source_release["dataset_root"])
    for entry in source_release["queue"]["jobs"]:
        _mark_done(source_dataset, entry)
    source_digest = pb.file_sha256(source_path)
    source_jobs_bytes = (source_dataset / "jobs.json").read_bytes()

    path, release, shards = _release(sibling_root)
    dataset = Path(release["dataset_root"])
    engineering = [entry for entry in release["queue"]["jobs"]
                   if entry["role"] in base.ENGINEERING]
    for entry in engineering:
        store = entry["job"]["store"]
        shutil.rmtree(dataset / entry["job"]["input"])
        shutil.rmtree(dataset / store)
        (dataset / "jobs_state" / f"{store}.claim").unlink()
        (dataset / "jobs_state" / f"{store}.done").unlink()
    release["queue"]["jobs"] = shards
    pb.atomic_json(dataset / "jobs.json", [entry["job"] for entry in shards])
    release["queue"]["jobs_sha256"] = pb.file_sha256(dataset / "jobs.json")

    receipt = formal.engineering_source_binding(source_path, source_digest)
    work = Path(release["queue"]["exploration"]["work_dir"])
    receipt_path = work / "engineering_source_reuse.json"
    pb.atomic_json(receipt_path, receipt)
    release["queue"]["exploration"]["files"] = {
        item.relative_to(work).as_posix(): pb.file_sha256(item)
        for item in work.rglob("*") if item.is_file()}
    release["queue"]["engineering_reuse"] = {
        "source_release": copy.deepcopy(receipt["source_release"]),
        "receipt": {"name": receipt_path.name,
                    "sha256": pb.file_sha256(receipt_path)},
    }
    digest = _rewrite(path, release)
    return (path, release, shards, digest, source_path, source_release,
            source_jobs_bytes, receipt_path)


def test_reused_engineering_source_validates_sibling_without_dispatch_rows_or_markers(tmp_path):
    (path, release, shards, digest, source_path, source_release,
     source_jobs_bytes, receipt_path) = _reuse_release(tmp_path)
    dataset = Path(release["dataset_root"])
    report = formal.validate_release(path, digest)
    assert report["job_count"] == 3
    assert len(report["engineering_check"]["comparisons"]) == 4
    assert formal.queue_status(path, digest) == "active"
    assert [entry["role"] for entry in release["queue"]["jobs"]] == ["formal_shard"] * 3
    assert pb.read_json(dataset / "jobs.json") == [entry["job"] for entry in shards]
    assert not any((dataset / "jobs_state").glob("dedust_r81smoke.*"))
    assert not any((dataset / "jobs_state").glob("dedust_r81discrete.*"))
    assert not any((dataset / "jobs_state").glob("dedust_r81mesh.*"))
    receipt = pb.read_json(receipt_path)
    assert receipt["kind"] == "engineering_original_source_reuse"
    assert receipt["source_release"]["path"] == str(source_path.resolve())
    assert set(receipt["source_files"]) >= {
        "jobs.json", "jobs_state/dedust_r81smoke.claim",
        "jobs_state/dedust_r81smoke.done", "dedust_r81smoke/results.json"}
    assert (Path(source_release["dataset_root"]) / "jobs.json").read_bytes() == source_jobs_bytes


def test_reused_engineering_source_mutation_or_missing_receipt_fails_closed(tmp_path):
    first = tmp_path / "mutation"
    first.mkdir()
    (path, release, _shards, digest, _source_path, source_release,
     _source_jobs_bytes, _receipt_path) = _reuse_release(first)
    source_dataset = Path(source_release["dataset_root"])
    claim_path = source_dataset / "jobs_state/dedust_r81smoke.claim"
    claim = pb.read_json(claim_path)
    claim["at"] = "mutated-after-receipt"
    pb.atomic_json(claim_path, claim)
    with pytest.raises(ValueError, match="receipt differs from authenticated source closure"):
        formal.validate_release(path, digest)

    second = tmp_path / "missing"
    second.mkdir()
    (path2, _release2, _shards2, digest2, _source_path2, _source_release2,
     _source_jobs_bytes2, receipt_path2) = _reuse_release(second)
    receipt_path2.unlink()
    with pytest.raises(ValueError, match="receipt.*missing"):
        formal.validate_release(path2, digest2)

    third = tmp_path / "terminal"
    third.mkdir()
    (path3, _release3, _shards3, digest3, _source_path3, source_release3,
     _source_jobs_bytes3, _receipt_path3) = _reuse_release(third)
    source_dataset3 = Path(source_release3["dataset_root"])
    (source_dataset3 / "jobs_state/dedust_r81smoke.done").unlink()
    with pytest.raises(ValueError, match="all-success terminal|missing from the source dataset"):
        formal.validate_release(path3, digest3)


def test_reuse_queue_rejects_engineering_dispatch_rows(tmp_path):
    (path, release, _shards, _digest, _source_path, source_release,
     source_jobs_bytes, _receipt_path) = _reuse_release(tmp_path)
    dataset = Path(release["dataset_root"])
    release["queue"]["jobs"].append(copy.deepcopy(source_release["queue"]["jobs"][0]))
    pb.atomic_json(dataset / "jobs.json", [entry["job"] for entry in release["queue"]["jobs"]])
    release["queue"]["jobs_sha256"] = pb.file_sha256(dataset / "jobs.json")
    with pytest.raises(ValueError, match="must not dispatch engineering jobs"):
        formal.validate_release(path, _rewrite(path, release))
    source_dataset = Path(source_release["dataset_root"])
    assert (source_dataset / "jobs.json").read_bytes() == source_jobs_bytes


def test_reuse_rejects_overlapping_dataset_roots_and_unknown_source_markers(tmp_path):
    nested_root = tmp_path / "nested"
    nested_root.mkdir()
    (path, release, _shards, _digest, _source_path, source_release,
     _source_jobs_bytes, _receipt_path) = _reuse_release(nested_root)
    source_dataset = Path(source_release["dataset_root"])
    nested_dataset = source_dataset / "formal_nested"
    nested_dataset.mkdir()
    with pytest.raises(ValueError, match="must not overlap"):
        formal._validate_engineering_reuse(
            release["queue"]["engineering_reuse"],
            release["queue"]["exploration"], nested_dataset)

    marker_root = tmp_path / "marker"
    marker_root.mkdir()
    (path2, release2, _shards2, _digest2, _source_path2, source_release2,
     _source_jobs_bytes2, _receipt_path2) = _reuse_release(marker_root)
    source_dataset2 = Path(source_release2["dataset_root"])
    pb.atomic_json(source_dataset2 / "jobs_state/unknown_native.fail", {"status": "failed"})
    with pytest.raises(ValueError, match="unknown marker"):
        formal.validate_release(path2, _rewrite(path2, release2))


def test_reused_engineering_source_preserves_formal_batch_lifecycle(tmp_path):
    (path, release, shards, digest, _source_path, source_release,
     source_jobs_bytes, _receipt_path) = _reuse_release(tmp_path)
    dataset = Path(release["dataset_root"])
    assert formal.queue_status(path, digest) == "active"
    for entry in shards:
        _complete_shard(dataset, entry)
    assert formal.queue_status(path, digest) == "physical_complete_feedback_pending"
    receipt = formal.aggregate_batch(path, digest, 1)
    assert receipt == dataset / "dedust_r81b1" / formal.COMPLETE_NAME
    assert formal.queue_status(path, digest) == "logical_ready_feedback_pending"
    work = Path(release["queue"]["exploration"]["work_dir"])
    ex.import_feedback(work, 1, dataset / "dedust_r81b1/results.json", dataset)
    release["queue"]["exploration"]["files"] = {
        item.relative_to(work).as_posix(): pb.file_sha256(item)
        for item in work.rglob("*") if item.is_file()}
    assert formal.queue_status(path, _rewrite(path, release)) == "all_success"
    assert (Path(source_release["dataset_root"]) / "jobs.json").read_bytes() == source_jobs_bytes


def test_native_first_batch_lifecycle_stops_workers_then_imports_one_logical_feedback(tmp_path):
    path, release, shards = _release(tmp_path)
    dataset = Path(release["dataset_root"])
    digest = pb.file_sha256(path)
    report = formal.validate_release(path, digest)
    assert report["logical_batch_count"] == 1
    assert report["physical_formal_job_count"] == 3
    assert formal.queue_status(path, digest) == "active"

    for entry in shards:
        _complete_shard(dataset, entry)
    assert formal.queue_status(path, digest) == "physical_complete_feedback_pending"

    receipt = formal.aggregate_batch(path, digest, 1)
    logical = dataset / "dedust_r81b1"
    assert receipt == logical / formal.COMPLETE_NAME
    assert len(pb.validate_store(logical, require_complete=True)) == 60
    assert formal.queue_status(path, digest) == "logical_ready_feedback_pending"

    work = Path(release["queue"]["exploration"]["work_dir"])
    ex.import_feedback(work, 1, logical / "results.json", dataset)
    release["queue"]["exploration"]["files"] = {
        item.relative_to(work).as_posix(): pb.file_sha256(item)
        for item in work.rglob("*") if item.is_file()}
    digest = _rewrite(path, release)
    assert formal.queue_status(path, digest) == "all_success"
    state = pb.read_json(work / "state.json")
    audit_path = work / state["batches"]["1"]["feedback_audit"]
    audit = pb.read_json(audit_path)
    assert audit["n_imported"] == 60 and audit["truth_used_for_selection"] is False

    imported = pb.read_json(work / "dataset_manifest.json")
    imported[0]["source_manifest"] = str(
        dataset / shards[0]["job"]["store"] / "results.json")
    pb.atomic_json(work / "dataset_manifest.json", imported)
    audit["dataset_entries_sha256"] = ex.content_id(sorted(imported, key=lambda row: row["id"]))
    pb.atomic_json(audit_path, audit)
    state["batches"]["1"]["feedback_audit_sha256"] = pb.file_sha256(audit_path)
    pb.atomic_json(work / "state.json", state)
    release["queue"]["exploration"]["files"] = {
        item.relative_to(work).as_posix(): pb.file_sha256(item)
        for item in work.rglob("*") if item.is_file()}
    with pytest.raises(ValueError, match="feedback source differs from logical merge"):
        formal.queue_status(path, _rewrite(path, release))


def test_split_row_tamper_and_source_alias_are_rejected_before_worker(tmp_path):
    path, release, shards = _release(tmp_path)
    dataset = Path(release["dataset_root"])
    child = dataset / Path(shards[0]["job"]["input"])
    rows = pb.read_json(child / "manifest.json")
    rows[0]["selection_arm"] = "random"
    pb.atomic_json(child / "manifest.json", rows)
    shards[0]["input_files"] = _files(child)
    pb.atomic_json(dataset / "jobs.json", [entry["job"] for entry in release["queue"]["jobs"]])
    release["queue"]["jobs_sha256"] = pb.file_sha256(dataset / "jobs.json")
    with pytest.raises(ValueError, match="round-robin|receipt"):
        formal.validate_release(path, _rewrite(path, release))

    alias_root = tmp_path / "alias"
    alias_root.mkdir()
    path2, release2, shards2 = _release(alias_root)
    dataset2 = Path(release2["dataset_root"])
    shards2[0]["job"]["input"] = shards2[1]["job"]["input"]
    shards2[0]["job"]["config"] = shards2[1]["job"]["config"]
    shards2[0]["input_files"] = copy.deepcopy(shards2[1]["input_files"])
    pb.atomic_json(dataset2 / "jobs.json", [entry["job"] for entry in release2["queue"]["jobs"]])
    release2["queue"]["jobs_sha256"] = pb.file_sha256(dataset2 / "jobs.json")
    with pytest.raises(ValueError, match="native split shard"):
        formal.validate_release(path2, _rewrite(path2, release2))


def test_incomplete_shard_and_roster_exhaustion_never_aggregate(tmp_path):
    path, release, shards = _release(tmp_path)
    dataset = Path(release["dataset_root"])
    digest = pb.file_sha256(path)
    _complete_shard(dataset, shards[0])
    with pytest.raises(ValueError, match="all three native shards"):
        formal.aggregate_batch(path, digest, 1)
    assert not (dataset / "dedust_r81b1").exists()

    state = dataset / "jobs_state"
    failed = shards[1]["job"]["store"]
    pb.atomic_json(state / f"{failed}.claim",
                   {"machine": base.EXPECTED_WORKERS[-1], "at": "fixture",
                    "prior_fail": list(base.EXPECTED_WORKERS[:-1])})
    pb.atomic_json(state / f"{failed}.fail",
                   {"machines": list(base.EXPECTED_WORKERS), "last": "hfss", "at": "fixture"})
    assert formal.queue_status(path, digest) == "terminal_failure"


def test_existing_exact_merge_requires_explicit_recovery_and_binds_completion(tmp_path):
    path, release, shards = _release(tmp_path)
    dataset = Path(release["dataset_root"])
    digest = pb.file_sha256(path)
    for entry in shards:
        _complete_shard(dataset, entry)
    canonical = dataset / "dedust_r81b1_input"
    logical = dataset / "dedust_r81b1"
    profiled_shards.merge_stores(
        canonical, [dataset / entry["job"]["store"] for entry in shards], logical)
    with pytest.raises(FileExistsError, match="recover-existing"):
        formal.aggregate_batch(path, digest, 1)
    receipt = formal.aggregate_batch(path, digest, 1, recover_existing=True)
    assert receipt.is_file()
    assert formal.queue_status(path, digest) == "logical_ready_feedback_pending"
    with pytest.raises(FileExistsError, match="recover-existing"):
        formal.aggregate_batch(path, digest, 1)


def test_materialized_logical_store_with_missing_or_rewritten_native_terminal_fails_closed(
        tmp_path):
    path, release, shards = _release(tmp_path)
    dataset = Path(release["dataset_root"])
    digest = pb.file_sha256(path)
    for entry in shards:
        _complete_shard(dataset, entry)
    receipt_path = formal.aggregate_batch(path, digest, 1)
    receipt = pb.read_json(receipt_path)
    terminals = [item["native_terminal"] for item in receipt["physical_sources"]]
    assert [item["machine"] for item in terminals] == [base.EXPECTED_WORKERS[0]] * 3
    assert all(item["claim_sha256"] and item["done_sha256"] for item in terminals)

    claim_path = dataset / "jobs_state" / f"{shards[1]['job']['store']}.claim"
    done_path = dataset / "jobs_state" / f"{shards[1]['job']['store']}.done"
    claim_bytes, done_bytes = claim_path.read_bytes(), done_path.read_bytes()
    done_path.unlink()
    assert claim_path.is_file()
    with pytest.raises(ValueError, match="materialized logical store requires all three"):
        formal.queue_status(path, digest)

    done_path.write_bytes(done_bytes)
    claim, done = pb.read_json(claim_path), pb.read_json(done_path)
    claim["machine"] = done["machine"] = base.EXPECTED_WORKERS[1]
    pb.atomic_json(claim_path, claim)
    pb.atomic_json(done_path, done)
    with pytest.raises(ValueError, match="completion receipt.*differs"):
        formal.queue_status(path, digest)

    claim_path.write_bytes(claim_bytes)
    done_path.write_bytes(done_bytes)
    claim, done = pb.read_json(claim_path), pb.read_json(done_path)
    claim["at"] = done["at"] = "rewritten"
    pb.atomic_json(claim_path, claim)
    pb.atomic_json(done_path, done)
    with pytest.raises(ValueError, match="completion receipt.*differs"):
        formal.queue_status(path, digest)


def test_merged_source_binding_tamper_blocks_feedback_readiness(tmp_path):
    path, release, shards = _release(tmp_path)
    dataset = Path(release["dataset_root"])
    digest = pb.file_sha256(path)
    for entry in shards:
        _complete_shard(dataset, entry)
    formal.aggregate_batch(path, digest, 1)
    logical = dataset / "dedust_r81b1"
    bindings = pb.read_json(logical / "source_bindings.json")
    first = next(iter(bindings["items"]))
    bindings["items"][first]["source_store"] = str(dataset / shards[1]["job"]["store"])
    pb.atomic_json(logical / "source_bindings.json", bindings)
    with pytest.raises(ValueError, match="source binding"):
        formal.queue_status(path, digest)


def test_launcher_is_formal_only_finite_and_never_aggregates_or_writes_queue():
    text = (formal.REPO / "script/start_r81_formal_sharded_worker.ps1").read_text(
        encoding="utf-8")
    assert "script.r81_formal_sharded_entry" in text and "--queue-status" in text
    assert "--scope symmetry_filter_20261007 --selfgen 0 --poll 60 --stale 120 --once" in text
    assert "aggregate-batch" not in text and "jobs-add" not in text and "git pull" not in text
    assert "if ($queueStatus -eq 0) { return }" in text
