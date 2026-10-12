"""Focused tests for the bounded local S21 survey feedback controller."""
from __future__ import annotations

import copy
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from script import profiled_batch as pb
from script import r81_provisional_filter_dispatch as dispatch
from script import r81_s21_survey_cycle as cycle
from tests.test_filter_s21_training import _wide_response


def _fixture_pattern(number: int) -> torch.Tensor:
    pattern = np.zeros((25, 25), dtype=np.float32)
    pattern[:5, 10:15] = 1
    pattern[20:, 10:15] = 1
    for bit in range(10):
        if (number + 1) & (1 << bit):
            pattern[bit // 5, bit % 5] = 1
    return torch.from_numpy(pattern)


def _make_source(tmp_path: Path, successes: int) -> tuple[dict, Path]:
    dataset = tmp_path / "released" / "dataset"
    dataset.mkdir(parents=True)
    entries = []
    remaining = successes
    for shard in range(1, 4):
        input_dir = dataset / f"input-{shard}"
        input_dir.mkdir()
        (input_dir / "config.yaml").write_bytes(dispatch.PROFILE_CONFIG.read_bytes())
        cfg = pb.load_profile_config(input_dir / "config.yaml")
        pb.atomic_json(input_dir / "measurement.json", cfg.measurement)
        pb.atomic_json(input_dir / "score_spec.json", cfg.score_spec)
        mid = cycle.measurement_id(cfg.measurement)
        sid = cycle.score_spec_id(cfg.score_spec)
        rows = []
        for offset in range(20):
            number = (shard - 1) * 20 + offset
            pattern = _fixture_pattern(number)
            name = f"source_{number:03d}"
            torch.save(pattern, input_dir / f"{name}.pt")
            rows.append({
                "id": name, "pattern_file": f"{name}.pt",
                "pattern_sha256": pb.pattern_sha256(pattern), "port": "dual",
                "measurement_id": mid, "score_spec_id": sid,
                "geometry_profile": "dual25_p01_db075",
                "lineage_id": f"line-{number}",
                "canonical_group_id": f"family-{number}",
            })
        pb.atomic_json(input_dir / "manifest.json", rows)
        pb.validate_input(input_dir, cfg)
        store = dataset / f"store-{shard}"
        pb.prepare_store(input_dir, store, cfg)
        take = min(remaining, 20)
        results = {}
        for offset, row in enumerate(rows[:take]):
            pattern = torch.load(input_dir / row["pattern_file"], weights_only=True)
            response = _wide_response((shard - 1) * 20 + offset)
            entry = pb.observation(response, None, row, pattern, cfg, elapsed=.1)
            torch.save((pattern, response), store / entry["sample_file"])
            entry["sample_sha256"] = pb.file_sha256(store / entry["sample_file"])
            results[row["id"]] = entry
        pb.atomic_json(store / "results.json", results)
        remaining -= take
        entries.append({"job": {"input": input_dir.name, "store": store.name,
                                 "machine": dispatch.base.EXPECTED_WORKERS[shard - 1]}})
    shutil.copytree(dataset / "input-1", dataset / dispatch.CANONICAL_INPUT)
    canonical_files = cycle._files(dataset / dispatch.CANONICAL_INPUT)
    context = {
        "release": {"release_id": "fixture_release", "dataset_root": str(dataset),
                    "policy": copy.deepcopy(dispatch.POLICY),
                    "queue": {"canonical_files": canonical_files}},
        "release_sha256": "a" * 64, "dataset": dataset,
        "entries": entries, "states": ["active"] * 3,
    }
    return context, dataset


def _bind(monkeypatch, tmp_path: Path, successes: int):
    context, dataset = _make_source(tmp_path, successes)
    monkeypatch.setattr(cycle, "REPO", tmp_path)
    monkeypatch.setattr(cycle, "_release_context", lambda _path, _sha: context)
    release = tmp_path / "release.json"
    pb.atomic_json(release, {"fixture": True})
    work = tmp_path / "tmp" / "survey"
    return context, dataset, release, work


def _add_success(dataset: Path, shard: int, offset: int) -> None:
    input_dir = dataset / f"input-{shard}"
    store = dataset / f"store-{shard}"
    cfg = pb.load_profile_config(input_dir / "config.yaml")
    row = pb.read_json(input_dir / "manifest.json")[offset]
    pattern = torch.load(input_dir / row["pattern_file"], weights_only=True)
    response = _wide_response((shard - 1) * 20 + offset)
    entry = pb.observation(response, None, row, pattern, cfg, elapsed=.1)
    torch.save((pattern, response), store / entry["sample_file"])
    entry["sample_sha256"] = pb.file_sha256(store / entry["sample_file"])
    results = pb.read_json(store / "results.json")
    results[row["id"]] = entry
    pb.atomic_json(store / "results.json", results)


def test_prepare_and_exact_twenty_new_success_trigger(monkeypatch, tmp_path, capsys):
    _context, dataset, release, work = _bind(monkeypatch, tmp_path, successes=19)
    # A claimed/prepared worker store may legitimately have no results file yet.
    (dataset / "store-3" / "results.json").unlink()
    assert cycle.main(["prepare", "--release", str(release),
                       "--release-sha256", "a" * 64,
                       "--work-dir", str(work)]) == 0
    assert "prepared_local_controller" in capsys.readouterr().out
    waiting = cycle.inspect(release, "a" * 64, work)
    assert waiting["status"] == "waiting" and waiting["new_unique_successes"] == 19

    _add_success(dataset, 1, 19)
    ready = cycle.inspect(release, "a" * 64, work)
    assert ready["status"] == "ready" and ready["new_unique_successes"] == 20
    before = (work / cycle.LEDGER).read_bytes()
    assert cycle.main(["dry-run", "--release", str(release),
                       "--release-sha256", "a" * 64,
                       "--work-dir", str(work),
                       "--candidate-pool-size", "80"]) == 0
    assert "dry_run_ready" in capsys.readouterr().out
    assert (work / cycle.LEDGER).read_bytes() == before
    assert not (work / "cycles").exists()


def test_single_cycle_prepares_bound_nonoverlapping_three_by_twenty(monkeypatch, tmp_path):
    context, dataset, release, work = _bind(monkeypatch, tmp_path, successes=20)
    (dataset / "store-3" / "results.json").unlink()
    cycle.prepare(release, "a" * 64, work)

    def fake_train(**kwargs):
        model_dir = Path(kwargs["output_dir"]) / "models" / "fixture"
        model_dir.mkdir(parents=True)
        paths = []
        hashes = {}
        for member in cycle.s21.ENSEMBLE_SEEDS:
            path = model_dir / f"member-{member}.pt"
            path.write_bytes(f"member-{member}".encode())
            paths.append(path); hashes[path.name] = pb.file_sha256(path)
        pb.atomic_json(model_dir / "completion_receipt.json", {"fixture": True})
        return paths, {"signature": "b" * 64, "model_sha256": hashes,
                       "recipe": {"hidden_dims": [8]}}

    def fake_rank(_model_dir, candidates, *, expected_completion_sha256):
        prediction = np.asarray([[index / 100, index / 200, index / 300]
                                 for index in range(len(candidates))], dtype=np.float32)
        order = np.argsort(-prediction.min(axis=1), kind="stable")
        receipt = {
            "schema_version": 1, "training_protocol": cycle.s21.PROTOCOL,
            "model_signature": "b" * 64,
            "completion_receipt_sha256": expected_completion_sha256,
            "candidate_set_id": cycle._json_id([
                {"id": row["id"], "pattern_sha256": row["pattern_sha256"]}
                for row in candidates]),
            "candidate_count": len(candidates),
            "candidate_ids": [row["id"] for row in candidates],
            "ordered_ids": [candidates[index]["id"] for index in order],
            "predicted_worst_margin": [float(prediction[index].min()) for index in order],
            "ranking": "descending minimum of the three S21 margins",
            "model_sha256": {"member-0.pt": "0" * 64, "member-1.pt": "1" * 64},
        }
        return prediction, order, receipt

    monkeypatch.setattr(cycle.s21, "train", fake_train)
    monkeypatch.setattr(cycle.s21, "predict_and_rank", fake_rank)
    frozen = {digest: item["sample_sha256"]
              for digest, item in cycle._input_store_inventory(context)["successes"].items()}
    monkeypatch.setattr(cycle, "_trained_snapshot_successes",
                        lambda _summary: (frozen, {"fixture_snapshot": True}))
    monkeypatch.setattr(cycle, "_ensemble_uncertainty",
                        lambda _dir, patterns, _mean, _summary:
                        np.linspace(1, 0, len(patterns), dtype=np.float32))
    result = cycle.single_cycle(
        release=release, release_sha256="a" * 64, work_dir=work,
        prior_manifest=tmp_path / "prior.json", prior_root=tmp_path,
        expected_prior_sha256="c" * 64, candidate_pool_size=80,
        pretrain_epochs=1, epochs=1, batch_train_size=8, hidden_dims=[8])
    assert result["status"] == "prepared_local_successor"
    assert result["batch"] == 2 and result["rows"] == 60
    cycle_dir = Path(result["cycle_dir"])
    receipt = pb.read_json(cycle_dir / "selection_receipt.json")
    assert receipt["claims"] == {
        "provisional_only": True, "formal_pass": False,
        "survey_complete_is_spec_pass": False, "live_handoff_available": False}
    assert {arm: sum(row["selection_arm"] == arm for row in receipt["selected"])
            for arm in cycle.ARM_COUNTS} == cycle.ARM_COUNTS
    assert all(row["candidate_group"] == "fresh_random"
               for row in receipt["selected"] if row["selection_arm"] == "random")
    queued = set(cycle._input_store_inventory(context)["queued"])
    assert not queued.intersection(row["pattern_sha256"] for row in receipt["selected"])
    canonical = cycle_dir / "batch-002_input"
    cfg = pb.load_profile_config(canonical / "config.yaml")
    rows = pb.validate_input(canonical, cfg)
    assert len(rows) == 60 and cfg.score_spec == dispatch.S21_SPEC
    assert receipt["bundle"]["source_profile_files"] == cycle._profile_binding(context)
    split = cycle_dir / "batch-002_split"
    assert [len(pb.read_json(split / f"shard-{number:03d}_input" / "manifest.json"))
            for number in (1, 2, 3)] == [20, 20, 20]
    assert all(job["launchable"] is False
               for job in pb.read_json(cycle_dir / "jobs_template.json"))
    ledger = pb.read_json(work / cycle.LEDGER)
    assert len(ledger["consumed_successes"]) == 20
    assert len(ledger["selected_pattern_sha256"]) == 60
    assert ledger["prepared_batches"][0]["launchable"] is False
    after = cycle.inspect(release, "a" * 64, work)
    assert after["status"] == "handoff_required" and after["next_batch"] is None
    results_path = dataset / "store-1" / "results.json"
    results = pb.read_json(results_path)
    results.pop(next(iter(results)))
    pb.atomic_json(results_path, results)
    with pytest.raises(ValueError, match="contradict"):
        cycle.inspect(release, "a" * 64, work)


def test_budget_is_three_total_batches_and_never_claims_live_handoff(monkeypatch, tmp_path):
    _context, _dataset, release, work = _bind(monkeypatch, tmp_path, successes=20)
    cycle.prepare(release, "a" * 64, work)
    ledger = pb.read_json(work / cycle.LEDGER)
    prepared = []
    selected = []
    for batch in (2, 3):
        folder = work / "cycles" / f"batch-{batch:03d}"
        (folder / f"batch-{batch:03d}_input").mkdir(parents=True)
        (folder / f"batch-{batch:03d}_split").mkdir()
        hashes = [f"{batch:02x}{index:062x}" for index in range(60)]
        selected.extend(hashes)
        pb.atomic_json(folder / "selection_receipt.json", {
            "batch": batch,
            "selected": [{"pattern_sha256": digest} for digest in hashes],
        })
        pb.atomic_json(folder / f"batch-{batch:03d}_input" / "manifest.json", [])
        pb.atomic_json(folder / f"batch-{batch:03d}_split" / "split_receipt.json", {})
        prepared.append({
            "batch": batch, "rows": 60, "directory": str(folder),
            "selection_receipt_sha256": pb.file_sha256(folder / "selection_receipt.json"),
            "canonical_manifest_sha256": pb.file_sha256(
                folder / f"batch-{batch:03d}_input" / "manifest.json"),
            "split_receipt_sha256": pb.file_sha256(
                folder / f"batch-{batch:03d}_split" / "split_receipt.json"),
            "launchable": False,
        })
    ledger["prepared_batches"] = prepared
    ledger["selected_pattern_sha256"] = selected
    pb.atomic_json(work / cycle.LEDGER, ledger)
    status = cycle.inspect(release, "a" * 64, work)
    assert status["status"] == "budget_exhausted" and status["next_batch"] is None
    assert status["handoff"]["launchable"] is False
    assert status["publication"] == {
        "nas_written": False, "release_created": False, "queue_changed": False,
        "worker_started": False, "hfss_started": False}


def test_execution_rejects_readiness_inventory_race(monkeypatch, tmp_path):
    _context, _dataset, release, work = _bind(monkeypatch, tmp_path, successes=20)
    cycle.prepare(release, "a" * 64, work)
    native = cycle._input_store_inventory
    calls = 0

    def changed(context):
        nonlocal calls
        calls += 1
        value = native(context)
        if calls == 2:
            value["source_inventory_id"] = "f" * 64
        return value

    monkeypatch.setattr(cycle, "_input_store_inventory", changed)
    with pytest.raises(ValueError, match="readiness and cycle cutoff"):
        cycle.single_cycle(
            release=release, release_sha256="a" * 64, work_dir=work,
            prior_manifest=tmp_path / "prior.json", prior_root=tmp_path,
            expected_prior_sha256="c" * 64, candidate_pool_size=80)


def test_pending_transaction_is_authenticated_and_recovered(monkeypatch, tmp_path):
    context, _dataset, release, work = _bind(monkeypatch, tmp_path, successes=20)
    cycle.prepare(release, "a" * 64, work)
    ledger = pb.read_json(work / cycle.LEDGER)
    temp = work / "cycles" / ".batch-002-fixture"
    final = work / "cycles" / "batch-002"
    (temp / "batch-002_input").mkdir(parents=True)
    (temp / "batch-002_split").mkdir()
    pb.atomic_json(temp / "selection_receipt.json", {"fixture": True})
    pb.atomic_json(temp / "batch-002_input" / "manifest.json", [{"fixture": True}])
    pb.atomic_json(temp / "batch-002_split" / "split_receipt.json", {"fixture": True})
    after = copy.deepcopy(ledger)
    after["prepared_batches"].append({
        "batch": 2, "rows": 60, "directory": str(final),
        "selection_receipt_sha256": pb.file_sha256(temp / "selection_receipt.json"),
        "canonical_manifest_sha256": pb.file_sha256(
            temp / "batch-002_input" / "manifest.json"),
        "split_receipt_sha256": pb.file_sha256(
            temp / "batch-002_split" / "split_receipt.json"),
        "launchable": False,
    })
    pb.atomic_json(work / cycle.PENDING, {
        "schema_version": cycle.SCHEMA_VERSION,
        "source_release_sha256": "a" * 64,
        "ledger_before_id": cycle._json_id(ledger), "ledger_after": after,
        "temp_dir": str(temp), "cycle_dir": str(final),
        "selection_receipt_sha256": pb.file_sha256(temp / "selection_receipt.json"),
    })
    recovered = cycle._recover_pending(work, ledger, context)
    assert recovered == after and final.is_dir()
    assert not temp.exists() and not (work / cycle.PENDING).exists()
    assert pb.read_json(work / cycle.LEDGER) == after

    # A second crash point, after the ledger commit but before marker removal,
    # is also an authenticated idempotent completion rather than a dead end.
    pb.atomic_json(work / cycle.PENDING, {
        "schema_version": cycle.SCHEMA_VERSION,
        "source_release_sha256": "a" * 64,
        "ledger_before_id": cycle._json_id(ledger), "ledger_after": after,
        "temp_dir": str(temp), "cycle_dir": str(final),
        "selection_receipt_sha256": pb.file_sha256(final / "selection_receipt.json"),
    })
    recovered_again = cycle._recover_pending(work, after, context)
    assert recovered_again == after and not (work / cycle.PENDING).exists()
