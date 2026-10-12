"""Focused generated-fixture tests for the additive S21 live worker handoff."""
from __future__ import annotations

import copy
import shutil
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

from script import profiled_batch as pb
from script import profiled_shards
from script import r81_provisional_filter_dispatch as dispatch
from script import r81_s21_live_entry as live
from script import r81_worker_entry as base
from tests import test_r81_provisional_filter_dispatch as dispatch_fixtures
from tests import test_r81_worker_entry as worker_fixtures


def _files(folder: Path) -> dict[str, str]:
    return {path.name: pb.file_sha256(path)
            for path in sorted(folder.iterdir(), key=lambda item: item.name)
            if path.is_file()}


@pytest.fixture(autouse=True)
def _future_runtime(monkeypatch, tmp_path):
    monkeypatch.setattr(base, "_git_is_ancestor", lambda _commit: True)
    monkeypatch.setattr(base, "_git_path_clean", lambda _commit, _relative: True)
    monkeypatch.setattr(base, "_git_blob_sha",
                        lambda _commit, relative: pb.file_sha256(base.REPO / relative))
    worker_fixtures._CENSUS_FIXTURES.clear()

    def bound_census(binding, _commit):
        path = worker_fixtures._CENSUS_FIXTURES[binding["repo_path"]]
        return path, pb.read_json(path)

    monkeypatch.setattr(base, "_bound_census_status", bound_census)
    monkeypatch.setattr(dispatch, "REPO", tmp_path)
    # Runtime source closure is exercised by production validation and the
    # exact-set assertion below; generated fixtures do not create Git blobs.
    monkeypatch.setattr(live, "_runtime", lambda _value: None)

    def validated_model(model_dir, *, rank_ready, expected_completion_sha256):
        assert rank_ready is True
        assert pb.file_sha256(Path(model_dir) / "completion_receipt.json") == \
            expected_completion_sha256
        return pb.read_json(Path(model_dir) / "training_summary.json")

    def loaded_snapshot(snapshot, _protocol):
        root = Path(snapshot)
        rows = pb.read_json(root / "manifest.json")
        return SimpleNamespace(rows=tuple(rows), receipt=pb.read_json(root / "receipt.json"))

    monkeypatch.setattr(live.s21, "_validated_model_summary", validated_model)
    monkeypatch.setattr(live.s21, "load_protocol", lambda: {})
    monkeypatch.setattr(live.s21, "load_current_snapshot", loaded_snapshot)


def _bootstrap(tmp_path: Path):
    release_path, release, release_sha, _preparation, _source = \
        dispatch_fixtures._fixture(tmp_path)
    actor = tmp_path.resolve()
    registry = actor / "survey_loop_v001" / "registry"
    registry.mkdir(parents=True)
    bootstrap_path = actor / "survey_loop_v001" / "release" / "bootstrap.json"
    bootstrap_path.parent.mkdir()
    initial = dispatch._context(release_path, release_sha)
    document = {
        "schema_version": 1, "release_id": "r81_s21_live_fixture_v1",
        "stage": live.STAGE, "actor_root": str(actor), "shared_root": str(actor),
        "runtime": {"git_commit": "1" * 40,
                    "files": {name: "0" * 64 for name in live.RUNTIME_FILES}},
        "initial_survey": {
            "release": {"path": str(release_path.resolve()), "sha256": release_sha},
            "dataset_root": str(Path(release["dataset_root"]).resolve()),
        },
        "registry": {"directory": str(registry.resolve())},
        "budget": copy.deepcopy(live.BUDGET),
        "protocol": live._protocol(initial),
    }
    pb.atomic_json(bootstrap_path, document)
    bootstrap_sha = pb.file_sha256(bootstrap_path)
    first = {
        "schema_version": 1, "sequence": 1, "previous": None,
        "bootstrap": {"path": str(bootstrap_path.resolve()), "sha256": bootstrap_sha},
        "release": {"path": str(release_path.resolve()), "sha256": release_sha,
                    "dataset_root": str(Path(release["dataset_root"]).resolve())},
    }
    first_path = registry / "batch-001.json"
    pb.atomic_json(first_path, first)
    return bootstrap_path, bootstrap_sha, first_path


def _successor(bootstrap_path: Path, bootstrap_sha: str, first_path: Path):
    context = live.validate_bootstrap(bootstrap_path, bootstrap_sha)
    initial = context["initial_context"]
    shared = context["shared_root"]
    batch = 2
    root = shared / "survey_loop_v001" / "batches" / "batch-002"
    dataset = root / "dataset"
    dataset.mkdir(parents=True)
    source = Path(initial["dataset"]) / dispatch.CANONICAL_INPUT
    canonical = dataset / "batch-002_input"
    canonical.mkdir()
    for marker in ("config.yaml", "measurement.json", "score_spec.json"):
        shutil.copy2(source / marker, canonical / marker)
    source_rows = pb.read_json(source / "manifest.json")
    rows = []
    arms = list(live.cycle.ARM_COUNTS)
    for index, old in enumerate(source_rows):
        pattern = torch.load(source / old["pattern_file"], weights_only=True,
                             map_location="cpu")
        name = f"r81s21_b02_{index:03d}_{old['pattern_sha256'][:8]}"
        torch.save(pattern, canonical / f"{name}.pt")
        row = copy.deepcopy(old)
        row.update({
            "id": name, "pattern_file": f"{name}.pt", "batch": batch,
            "selection_arm": arms[index // 20], "candidate_id": f"candidate-{index}",
            "pool_index": index, "model_signature": "2" * 64,
            "candidate_set_id": "3" * 64, "prediction": [0.0, 0.0, 0.0],
            "predicted_worst_margin_db": 0.0, "predicted_uncertainty": 0.0,
        })
        rows.append(row)
    pb.atomic_json(canonical / "manifest.json", rows)
    cfg = pb.load_profile_config(canonical / "config.yaml")
    pb.validate_input(canonical, cfg)
    split_root = dataset / "batch-002_split"
    profiled_shards.split_bundle(canonical, split_root, shard_size=20)
    jobs = []
    entries = []
    for shard, machine in enumerate(base.EXPECTED_WORKERS, start=1):
        relative = f"batch-002_split/shard-{shard:03d}_input"
        job = {"input": relative,
               "store": f"dedust_r81s21live_b002_{shard:02d}",
               "prio": dispatch.LOW_TIER_PRIORITY,
               "config": f"{relative}/config.yaml", "scope": dispatch.SCOPE,
               "machine": machine}
        jobs.append(job)
        entries.append({"role": live.SUCCESSOR_ROLE, "shard": shard, "job": job,
                        "input_files": _files(dataset / relative)})
    pb.atomic_json(dataset / "jobs.json", jobs)

    evidence = root / "evidence"
    evidence.mkdir()
    rank = {"candidate_set_id": "3" * 64, "order": list(range(60))}
    rank_path = evidence / "rank_receipt.json"
    pb.atomic_json(rank_path, rank)
    pool_root = evidence / "candidate_pool"
    (pool_root / "patterns").mkdir(parents=True)
    pool_rows = []
    for row in rows:
        relative = f"patterns/{row['candidate_id']}.pt"
        shutil.copy2(canonical / row["pattern_file"], pool_root / relative)
        pool_rows.append({
            "id": row["candidate_id"], "pattern_file": relative,
            "pattern_file_sha256": pb.file_sha256(pool_root / relative),
            "pattern_sha256": row["pattern_sha256"], "candidate_group": "fixture",
            "parent_id": None, "lineage_id": row.get("lineage_id"),
            "prediction": row["prediction"],
            "predicted_worst_margin_db": row["predicted_worst_margin_db"],
            "predicted_uncertainty": row["predicted_uncertainty"],
        })
    pb.atomic_json(pool_root / "manifest.json", pool_rows)
    pool = {"schema_version": 1, "candidate_count": 60,
            "manifest_sha256": pb.file_sha256(pool_root / "manifest.json"),
            "rank_receipt_sha256": pb.file_sha256(rank_path),
            "candidate_set_id": "3" * 64,
            "pattern_files_id": live.cycle._json_id([
                {"id": row["id"], "file_sha256": row["pattern_file_sha256"]}
                for row in pool_rows])}
    pool_path = pool_root / "pool_receipt.json"
    pb.atomic_json(pool_path, pool)
    trigger = [row["pattern_sha256"] for row in rows[:20]]
    snapshot_root = evidence / "snapshot"
    snapshot_root.mkdir()
    snapshot_rows = copy.deepcopy(rows[:20])
    pb.atomic_json(snapshot_root / "manifest.json", snapshot_rows)
    pb.atomic_json(snapshot_root / "results.json", {
        row["id"]: {"sample_sha256": "6" * 64} for row in snapshot_rows})
    pb.atomic_json(snapshot_root / "source_bindings.json",
                   {"schema_version": 1, "partial_snapshot": True})
    snapshot_receipt = {
        "snapshot": str(snapshot_root.resolve()),
        "manifest_sha256": pb.file_sha256(snapshot_root / "manifest.json"),
        "results_sha256": pb.file_sha256(snapshot_root / "results.json"),
        "source_bindings_sha256": pb.file_sha256(snapshot_root / "source_bindings.json"),
        "samples_id": "7" * 64, "sample_count": 20,
        "source_partial_snapshot": True,
    }
    pb.atomic_json(snapshot_root / "receipt.json", snapshot_receipt)
    model_root = evidence / "model"
    model_root.mkdir()
    (model_root / "member-0.pt").write_bytes(b"member zero")
    (model_root / "member-1.pt").write_bytes(b"member one")
    model_sha = {name: pb.file_sha256(model_root / name)
                 for name in ("member-0.pt", "member-1.pt")}
    pb.atomic_json(model_root / "training_plan.json", {"fixture": True})
    summary = {"signature": "2" * 64, "model_sha256": model_sha,
               "snapshot": snapshot_receipt}
    pb.atomic_json(model_root / "training_summary.json", summary)
    completion_path = model_root / "completion_receipt.json"
    pb.atomic_json(completion_path, live.s21._completion_receipt(model_root, summary))
    selected = [{**{key: row[key] for key in (
        "pattern_sha256", "selection_arm", "pool_index", "prediction",
        "predicted_worst_margin_db", "predicted_uncertainty")},
                 "id": row["candidate_id"]} for row in rows]
    canonical_files = _files(canonical)
    split_sha = pb.file_sha256(split_root / "split_receipt.json")
    selection = {
        "schema_version": 1, "batch": batch,
        "trigger_new_success_pattern_sha256": trigger,
        "trigger": {"successes": 20, "new_pattern_sha256": trigger,
                    "required_batch": None,
                    "required_batch_new_pattern_sha256": trigger,
                    "source_inventory_id": "8" * 64},
        "trained_snapshot": {"receipt": snapshot_receipt,
                             "successes": [{"pattern_sha256": value,
                                            "sample_sha256": "6" * 64}
                                           for value in sorted(trigger)]},
        "model": {"signature": summary["signature"],
                  "completion_receipt_sha256": pb.file_sha256(completion_path),
                  "model_sha256": model_sha},
        "pool": pool, "ranking": rank, "selected": selected,
        "bundle": {"canonical_files": canonical_files,
                   "split_receipt_sha256": split_sha},
        "claims": {"provisional_only": True, "formal_pass": False,
                   "survey_complete_is_spec_pass": False,
                   "live_handoff_available": False},
    }
    selection_path = evidence / "selection_receipt.json"
    pb.atomic_json(selection_path, selection)
    previous = {"sequence": 1, "path": str(first_path.resolve()),
                "sha256": pb.file_sha256(first_path)}
    release = {
        "schema_version": 1, "release_id": "r81_s21_successor_fixture_b002",
        "stage": live.SUCCESSOR_STAGE, "sequence": batch,
        "dataset_root": str(dataset.resolve()),
        "bootstrap": {"path": str(bootstrap_path.resolve()), "sha256": bootstrap_sha},
        "previous_registry": previous,
        "cycle": {
            "batch": batch,
            "selection_receipt": {"path": str(selection_path.resolve()),
                                  "sha256": pb.file_sha256(selection_path)},
            "candidate_pool_receipt": {"path": str(pool_path.resolve()),
                                       "sha256": pb.file_sha256(pool_path)},
            "rank_receipt": {"path": str(rank_path.resolve()),
                             "sha256": pb.file_sha256(rank_path)},
            "model_completion_receipt": {"path": str(completion_path.resolve()),
                                         "sha256": pb.file_sha256(completion_path)},
            "frozen_snapshot": {"path": str(snapshot_root.resolve()),
                                "files": {path.relative_to(snapshot_root).as_posix():
                                          pb.file_sha256(path)
                                          for path in sorted(snapshot_root.rglob("*"))
                                          if path.is_file()}},
            "canonical_manifest_sha256": pb.file_sha256(canonical / "manifest.json"),
            "split_receipt_sha256": split_sha,
        },
        "queue": {"jobs_sha256": pb.file_sha256(dataset / "jobs.json"),
                  "jobs": entries, "canonical_files": canonical_files,
                  "split_receipt_sha256": split_sha},
        "policy": copy.deepcopy(live.POLICY),
    }
    release_path = root / "release.json"
    pb.atomic_json(release_path, release)
    registry_path = context["registry_dir"] / "batch-002.json"
    pb.atomic_json(registry_path, {
        "schema_version": 1, "sequence": 2, "previous": previous,
        "bootstrap": {"path": str(bootstrap_path.resolve()), "sha256": bootstrap_sha},
        "release": {"path": str(release_path.resolve()),
                    "sha256": pb.file_sha256(release_path),
                    "dataset_root": str(dataset.resolve())},
    })
    return context, release_path, registry_path, selection_path


def test_bootstrap_reuses_external_initial_release_and_exact_runtime_set(tmp_path):
    bootstrap, digest, first = _bootstrap(tmp_path)
    context = live.validate_bootstrap(bootstrap, digest)
    record = live.validate_registry_record(first, pb.file_sha256(first), context)
    assert record["dataset"] == Path(context["initial_context"]["dataset"])
    assert record["record"]["previous"] is None
    assert len(live.RUNTIME_FILES) == 56
    assert set(dispatch.RUNTIME_FILES) < set(live.RUNTIME_FILES)


def test_successor_release_authenticates_cycle_geometry_partition_and_chain(tmp_path):
    bootstrap, digest, first = _bootstrap(tmp_path)
    context, release, second, _selection = _successor(bootstrap, digest, first)
    validated = live.validate_successor_release(release, pb.file_sha256(release), context)
    record = live.validate_registry_record(second, pb.file_sha256(second), context)
    assert len(validated["entries"]) == 3 and validated["states"] == ["active"] * 3
    assert [len(pb.read_json(validated["dataset"] / item["job"]["input"] /
                             "manifest.json")) for item in validated["entries"]] == [20] * 3
    assert record["record"]["previous"]["sha256"] == pb.file_sha256(first)


@pytest.mark.parametrize("tamper", ["chain", "selection", "machine", "pattern",
                                     "model", "snapshot"])
def test_successor_rejects_chain_cycle_owner_or_pattern_drift(tmp_path, tamper):
    bootstrap, digest, first = _bootstrap(tmp_path)
    context, release_path, second, selection_path = _successor(bootstrap, digest, first)
    if tamper == "chain":
        record = pb.read_json(second)
        record["previous"]["sha256"] = "f" * 64
        pb.atomic_json(second, record)
        with pytest.raises(ValueError):
            live.validate_registry_record(second, pb.file_sha256(second), context)
        return
    release = pb.read_json(release_path)
    dataset = Path(release["dataset_root"])
    if tamper == "selection":
        selection = pb.read_json(selection_path)
        selection["selected"][0]["selection_arm"] = "random"
        pb.atomic_json(selection_path, selection)
        release["cycle"]["selection_receipt"]["sha256"] = pb.file_sha256(selection_path)
    elif tamper == "machine":
        release["queue"]["jobs"][0]["job"]["machine"] = base.EXPECTED_WORKERS[1]
        pb.atomic_json(dataset / "jobs.json",
                       [item["job"] for item in release["queue"]["jobs"]])
        release["queue"]["jobs_sha256"] = pb.file_sha256(dataset / "jobs.json")
    elif tamper == "pattern":
        row = pb.read_json(dataset / "batch-002_input" / "manifest.json")[0]
        path = dataset / "batch-002_input" / row["pattern_file"]
        path.write_bytes(path.read_bytes() + b"tamper")
        release["queue"]["canonical_files"][path.name] = pb.file_sha256(path)
    elif tamper == "model":
        completion = Path(release["cycle"]["model_completion_receipt"]["path"])
        (completion.parent / "member-0.pt").write_bytes(b"substituted model")
    else:
        snapshot = Path(release["cycle"]["frozen_snapshot"]["path"])
        results = pb.read_json(snapshot / "results.json")
        results[next(iter(results))]["sample_sha256"] = "e" * 64
        pb.atomic_json(snapshot / "results.json", results)
    pb.atomic_json(release_path, release)
    with pytest.raises((ValueError, RuntimeError, EOFError)):
        live.validate_successor_release(release_path, pb.file_sha256(release_path), context)


def test_worker_can_advance_own_terminal_batch_without_other_shards(monkeypatch, tmp_path):
    bootstrap, digest, first = _bootstrap(tmp_path)
    context, _release, _second, _selection = _successor(bootstrap, digest, first)
    context["initial_context"]["states"] = ["done", "active", "active"]
    monkeypatch.setattr(live, "validate_bootstrap", lambda _path, _sha: context)
    status = live.worker_status(bootstrap, digest, base.EXPECTED_WORKERS[0])
    waiting = live.worker_status(bootstrap, digest, base.EXPECTED_WORKERS[1])
    assert status["status"] == "ready" and status["sequence"] == 2
    assert waiting["status"] == "ready" and waiting["sequence"] == 1


def test_initial_pinned_owner_fail_is_terminal_without_cross_host_retry(tmp_path):
    bootstrap, digest, _first = _bootstrap(tmp_path)
    document = pb.read_json(bootstrap)
    dataset = Path(document["initial_survey"]["dataset_root"])
    job = pb.read_json(dataset / "jobs.json")[0]
    state = dataset / "jobs_state"
    state.mkdir(exist_ok=True)
    pb.atomic_json(state / f"{job['store']}.claim",
                   {"machine": job["machine"], "at": "fixture"})
    pb.atomic_json(state / f"{job['store']}.fail",
                   {"machines": [job["machine"]], "last": "fixture failure",
                    "at": "fixture"})
    status = live.worker_status(bootstrap, digest, job["machine"])
    assert status["status"] == "terminal_failure" and status["sequence"] == 1


@pytest.mark.parametrize("location", ["actor", "batch"])
def test_shared_stop_prevents_all_advancement_and_waiting(tmp_path, location):
    bootstrap, digest, _first = _bootstrap(tmp_path)
    document = pb.read_json(bootstrap)
    if location == "actor":
        marker = Path(document["actor_root"]) / "STOP"
    else:
        marker = Path(document["initial_survey"]["dataset_root"]) / "jobs_state" / \
            f"STOP_{dispatch.SCOPE}"
        marker.parent.mkdir(exist_ok=True)
    marker.write_text("operator stop\n", encoding="utf-8")
    status = live.worker_status(bootstrap, digest, base.EXPECTED_WORKERS[0])
    assert status["status"] == "stopped" and str(marker) in status["stop_markers"]


def test_launcher_is_continuous_once_low_priority_and_read_only_guarded():
    text = (base.REPO / "script/start_r81_s21_live_worker.ps1").read_text(encoding="utf-8")
    assert "Get-CimInstance Win32_Process" in text and "never" in text.lower()
    assert "--selfgen 0" in text and "--once" in text
    assert "while ($true)" in text and "waiting_for_publication" in text
    assert "status.status -eq 'stopped'" in text
    assert "Stop-Process" not in text and "script[.]dedust" in text
