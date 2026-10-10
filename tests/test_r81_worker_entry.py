"""Focused fail-closed tests for the future R81 worker entry."""
from __future__ import annotations

import copy
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from antenna.measurement import measurement_id, score_spec_id
from script import prepare_symmetry_filter as prep
from script import profiled_batch as pb
from script import r81_worker_entry as entry
from script import exploration as ex
from script import dedust

_REAL_BOUND_CENSUS_STATUS = entry._bound_census_status
_CENSUS_FIXTURES: dict[str, Path] = {}


@pytest.fixture(autouse=True)
def _committed_runtime_fixture(monkeypatch):
    """Model the future post-commit release while these new files are untracked."""
    monkeypatch.setattr(entry, "_git_is_ancestor", lambda _commit: True)
    monkeypatch.setattr(entry, "_git_path_clean", lambda _commit, _relative: True)
    monkeypatch.setattr(entry, "_git_blob_sha",
                        lambda _commit, relative: pb.file_sha256(entry.REPO / relative))
    _CENSUS_FIXTURES.clear()
    def bound_census(binding, _commit):
        assert set(binding) == {"repo_path", "sha256"}
        path = _CENSUS_FIXTURES[binding["repo_path"]]
        assert pb.file_sha256(path) == binding["sha256"]
        return path, pb.read_json(path)
    monkeypatch.setattr(entry, "_bound_census_status", bound_census)


def _files(folder: Path) -> dict[str, str]:
    return {path.name: pb.file_sha256(path) for path in folder.iterdir() if path.is_file()}


def _wide_fixture(dataset: Path, delta: float = .1) -> None:
    configs = prep.wide_check_configs()
    formal = dataset / "dedust_r81b1_input"
    formal.mkdir(parents=True)
    (formal / "config.yaml").write_text(yaml.safe_dump(configs["smoke"]), encoding="utf-8")
    cfg = pb.load_profile_config(formal / "config.yaml")
    rows = []
    for i in range(60):
        pattern = prep.ex.enforce_dual_geometry(np.random.default_rng(i).random((25, 25)))
        key = f"p{i}"
        torch.save(torch.tensor(pattern, dtype=torch.float32), formal / f"{key}.pt")
        rows.append({"id": key, "pattern_file": f"{key}.pt", "port": "dual",
                     "pattern_sha256": pb.pattern_sha256(pattern),
                     "measurement_id": measurement_id(cfg.measurement),
                     "score_spec_id": score_spec_id(cfg.score_spec),
                     "selection_arm": ("history", "specialist", "random")[i // 20],
                     "batch": 1, "model_hash": None})
    for name, value in (("measurement.json", cfg.measurement), ("score_spec.json", cfg.score_spec),
                        ("manifest.json", rows)):
        pb.atomic_json(formal / name, value)
    for name, offset in (("smoke", 0), ("discrete", delta), ("mesh", delta)):
        folder = dataset / f"dedust_r81{name}_input"
        indices = [0, 1, 20, 21, 40, 41] if name == "smoke" else [0, 1]
        prep.auxiliary_bundle(formal, folder, indices, config=configs[name])
        local_cfg = pb.load_profile_config(folder / "config.yaml")
        store = dataset / f"dedust_r81{name}"
        manifest = pb.prepare_store(folder, store, local_cfg)
        results = {}
        for row in manifest:
            pattern = torch.load(folder / row["pattern_file"], weights_only=True)
            response = torch.full((3, 49), -9.0) + offset
            result = pb.observation(response, None, row, pattern, local_cfg, elapsed=1.0)
            sample = store / result["sample_file"]
            torch.save((pattern, response), sample)
            result["sample_sha256"] = pb.file_sha256(sample)
            results[row["id"]] = result
        pb.atomic_json(store / "results.json", results)


def _job(role: str, store: str, *, batch: int | None = None,
         candidate: dict | None = None) -> dict:
    value = {"role": role,
             "job": {"input": store + "_input", "store": store, "prio": 5,
                     "config": store + "_input/config.yaml", "scope": prep.SCOPE}}
    if batch is not None:
        value["batch"] = batch
    if candidate is not None:
        value["candidate"] = candidate
    return value


def _positive_repeat(dataset: Path) -> tuple[dict, dict]:
    formal = dataset / "dedust_r81b1_input"
    source_rows = pb.read_json(formal / "manifest.json")
    source = dataset / "dedust_r81positive_input"
    source.mkdir()
    for name in ("config.yaml", "measurement.json", "score_spec.json"):
        shutil.copy2(formal / name, source / name)
    row = copy.deepcopy(source_rows[0])
    pb.atomic_json(source / "manifest.json", [row])
    shutil.copy2(formal / row["pattern_file"], source / row["pattern_file"])
    cfg = pb.load_profile_config(source / "config.yaml")
    store = dataset / "dedust_r81positive"
    pb.prepare_store(source, store, cfg)
    pattern = torch.load(source / row["pattern_file"], weights_only=True)
    freqs = np.linspace(16, 40, 49)
    s21 = np.full(49, -25.0, dtype=np.float32)
    s21[(freqs >= 26) & (freqs <= 30)] = -2.0
    response = torch.stack((torch.full((49,), -12.0), torch.tensor(s21), torch.full((49,), -12.0)))
    result = pb.observation(response, None, row, pattern, cfg, elapsed=1.0)
    sample = store / result["sample_file"]
    torch.save((pattern, response), sample)
    result["sample_sha256"] = pb.file_sha256(sample)
    assert result["all_passed"] and result["worst_margin"] > 0
    pb.atomic_json(store / "results.json", {row["id"]: result})

    repeat_input = dataset / "dedust_r81repeat_positive_input"
    prep.auxiliary_bundle(source, repeat_input, [0], repeat=True)
    repeat_row = pb.read_json(repeat_input / "manifest.json")[0]
    repeat_row["parent_batch_id"] = row["id"]
    pb.atomic_json(repeat_input / "manifest.json", [repeat_row])
    candidate = {"store": store.name, "id": row["id"], "files": _files(store)}
    return _job("positive_repeat", "dedust_r81repeat_positive", candidate=candidate), candidate


def _evidence(tmp_path: Path, dataset: Path) -> dict:
    root = tmp_path / "r80_evidence"
    root.mkdir()
    profile = root / "single_r80_symmetry_factory.yaml"
    shutil.copy2(entry.REPO / "configs/single_r80_symmetry_factory.yaml", profile)
    profile_producer = tmp_path / "original_r80_source" / "configs" / profile.name
    action_producer = tmp_path / "original_r80_source" / "cycles" / "action_receipt.json"
    cfg = pb.load_profile_config(profile)
    mid, sid = measurement_id(cfg.measurement), score_spec_id(cfg.score_spec)
    action = {"status": "idle", "scope": prep.SCOPE,
              "profile_sha256": pb.file_sha256(profile), "dataset_root": str(tmp_path / "r80_dataset"),
              "profile_config": str(profile_producer.resolve()),
              "valid_unique": 5000, "target_valid_unique": 5000, "pending_unique": 0,
              "planned_jobs": [], "audit": {"valid_unique_patterns": 5000}}
    action_path = root / "action.json"
    pb.atomic_json(action_path, action)
    watch = {"status": "symmetry_collection_complete", "valid_unique": 5000,
             "last_receipt": str(action_producer.resolve()),
             "last_receipt_sha256": pb.file_sha256(action_path),
             "profile_binding": {"path": str(profile_producer.resolve()),
                                 "sha256": pb.file_sha256(profile), "scope": prep.SCOPE}}
    rows_path, proofs_path = root / "rows.json", root / "pair_proofs.json"
    query_path, validation_path = root / "query_and_freeze.py", root / "record_status.py"
    pb.atomic_json(rows_path, [{"fixture": "bound saved raw rows"}])
    pb.atomic_json(proofs_path, [{"fixture": "bound pair proofs"}])
    query_path.write_text("# fixed raw query producer\n", encoding="utf-8")
    validation_path.write_text("# fixed saved-summary validator\n", encoding="utf-8")
    raw = {"status": "read_only_query_complete", "measurement_id": mid, "score_spec_id": sid,
           "unique_successful_patterns": 5000, "successful_including_repeats": 5001,
           "raw_replayed_this_query": 5001, "exact_prior_metrics_reused": 0,
           "reported_best_independently_raw_replayed": True,
           "jobs_sha256_at_scan_start": "a" * 64,
           "rows_sha256": pb.file_sha256(rows_path),
           "pair_proofs_sha256": pb.file_sha256(proofs_path),
           "query_script_sha256": pb.file_sha256(query_path),
           "controller_locks_acquired": False, "queue_or_source_mutation": False}
    raw_path = root / "report.json"
    pb.atomic_json(raw_path, raw)
    census = dict(raw)
    census.update(validation_status="passed_saved_raw_summary_validation", target_unique=5000,
                  query_report_sha256=pb.file_sha256(raw_path),
                  validation_script_sha256=pb.file_sha256(validation_path))
    watch_path, census_path = root / "watch.json", root / "census.json"
    pb.atomic_json(watch_path, watch)
    pb.atomic_json(census_path, census)
    bind = lambda path: {"path": str(path.resolve()), "sha256": pb.file_sha256(path)}
    census_repo_path = f"docs/log/assets/r80_fixture_{tmp_path.name}.json"
    _CENSUS_FIXTURES[census_repo_path] = census_path
    return {"watch_status": bind(watch_path),
            "action_receipt": {**bind(action_path), "producer_path": str(action_producer.resolve())},
            "census": {"repo_path": census_repo_path, "sha256": pb.file_sha256(census_path)},
            "profile": {**bind(profile), "producer_path": str(profile_producer.resolve())},
            "census_raw_report": bind(raw_path), "census_rows": bind(rows_path),
            "census_pair_proofs": bind(proofs_path), "census_query_producer": bind(query_path),
            "census_validation_producer": bind(validation_path),
            "measurement_id": mid, "score_spec_id": sid}


def _controller(dataset: Path, prefix: int) -> dict:
    work = dataset.parent / "r81_exploration_controller"
    (work / "batches").mkdir(parents=True)
    formal = dataset / "dedust_r81b1_input"
    for name in ("config.yaml", "measurement.json", "score_spec.json"):
        shutil.copy2(formal / name, work / name)
    cfg = ex.validate_exploration_config(ex._load_yaml(work / "config.yaml"))
    batches = {}
    for batch in range(1, prefix + 1):
        source = dataset / f"dedust_r81b{batch}_input"
        target = work / "batches" / f"batch-{batch:03d}_input"
        shutil.copytree(source, target)
        batches[str(batch)] = {"input_dir": target.relative_to(work).as_posix(),
                               "selected": 60, "model_hash": None if batch == 1 else "untrusted",
                               "feedback_audit": None, "trained_model_dir": None}
    state = {"schema_version": 1, "config_hash": ex.content_id(cfg),
             "dataset_root": str(dataset.resolve()),
             "measurement_id": measurement_id(cfg["measurement"]),
             "score_spec_id": score_spec_id(cfg["score_spec"]),
             "id_prefix": "r81_fixture", "frequencies_ghz": np.linspace(16, 40, 49).tolist(),
             "seed_inputs": [], "max_batches": 3, "measured_budget": 0,
             "next_batch": prefix + 1, "batches": batches}
    pb.atomic_json(work / "state.json", state)
    pb.atomic_json(work / "dataset_manifest.json", [])
    return {"work_dir": str(work.resolve()),
            "files": {path.relative_to(work).as_posix(): pb.file_sha256(path)
                      for path in work.rglob("*") if path.is_file()}}


def _complete_formal_batch(dataset: Path, batch: int) -> Path:
    input_dir = dataset / f"dedust_r81b{batch}_input"
    store = dataset / f"dedust_r81b{batch}"
    cfg = pb.load_profile_config(input_dir / "config.yaml")
    rows = pb.prepare_store(input_dir, store, cfg)
    results = {}
    response = torch.stack((torch.full((49,), -11.0), torch.full((49,), -4.0),
                            torch.full((49,), -11.0)))
    for row in rows:
        pattern = torch.load(input_dir / row["pattern_file"], weights_only=True)
        result = pb.observation(response, None, row, pattern, cfg, elapsed=1.0)
        sample = store / result["sample_file"]
        torch.save((pattern, response), sample)
        result["sample_sha256"] = pb.file_sha256(sample)
        results[row["id"]] = result
    pb.atomic_json(store / "results.json", results)
    return store / "results.json"


def _release(tmp_path: Path, stage: str = "EngineeringOnly", *, delta: float = .1,
             prefix: int = 1, repeat: bool = False) -> tuple[Path, dict]:
    dataset = tmp_path / "separate_r81_dataset"
    dataset.mkdir()
    _wide_fixture(dataset, delta)
    jobs = [_job(role, store) for role, (store, _count) in entry.ENGINEERING.items()]
    exploration = None
    if stage == "Formal":
        formal = dataset / "dedust_r81b1_input"
        for batch in range(2, prefix + 1):
            shutil.copytree(formal, dataset / f"dedust_r81b{batch}_input")
            rows = pb.read_json(dataset / f"dedust_r81b{batch}_input/manifest.json")
            for index, row in enumerate(rows):
                row["batch"] = batch
                row["model_hash"] = "untrusted"
                row["selection_arm"] = ("best_min_margin", "uncertainty", "random")[index // 20]
            pb.atomic_json(dataset / f"dedust_r81b{batch}_input/manifest.json", rows)
        jobs += [_job("formal_batch", f"dedust_r81b{batch}", batch=batch)
                 for batch in range(1, prefix + 1)]
        if repeat:
            repeat_job, _ = _positive_repeat(dataset)
            jobs.append(repeat_job)
        exploration = _controller(dataset, prefix)
    for item in jobs:
        item["input_files"] = _files(dataset / item["job"]["input"])
    prior_dataset = dedust.DATASET_PATH
    try:
        dedust.DATASET_PATH = dataset
        for item in jobs:
            job = item["job"]
            dedust.jobs_add(SimpleNamespace(
                input=job["input"], store=job["store"], prio=job["prio"],
                scope=job["scope"], config=str(dataset / job["input"] / "config.yaml"),
                machine=None))
        actual_jobs = pb.read_json(dataset / "jobs.json")
    finally:
        dedust.DATASET_PATH = prior_dataset
    assert actual_jobs == [item["job"] for item in jobs]
    commit = entry._git_commit()
    runtime = {"git_commit": commit,
               "files": {name: entry._git_blob_sha(commit, name) for name in entry.RUNTIME_FILES}}
    release = {"schema_version": 1, "release_id": "fixture_v1", "stage": stage,
               "dataset_root": str(dataset.resolve()), "runtime": runtime,
               "r80_completion": _evidence(tmp_path, dataset),
               "queue": {"jobs_sha256": pb.file_sha256(dataset / "jobs.json"), "jobs": jobs,
                         "expected_retry_workers": list(entry.EXPECTED_WORKERS),
                         "formal_input": {"name": "dedust_r81b1_input", "files": _files(dataset / "dedust_r81b1_input")},
                         "exploration": exploration}}
    path = tmp_path / "release.json"
    pb.atomic_json(path, release)
    return path, release


def _rewrite(path: Path, release: dict) -> str:
    pb.atomic_json(path, release)
    return pb.file_sha256(path)


def test_engineering_only_uses_actual_generated_wide_input_api(tmp_path):
    path, release = _release(tmp_path)
    report = entry.validate_release(path, pb.file_sha256(path), "EngineeringOnly")
    assert report["job_count"] == 3 and report["engineering_check"] is None
    assert report["r80"]["measurement_id"] == release["r80_completion"]["measurement_id"]


def test_archived_r80_evidence_preserves_distinct_original_producer_paths(tmp_path):
    path, release = _release(tmp_path)
    evidence = release["r80_completion"]
    assert evidence["action_receipt"]["path"] != evidence["action_receipt"]["producer_path"]
    assert evidence["profile"]["path"] != evidence["profile"]["producer_path"]
    entry.validate_release(path, pb.file_sha256(path), "EngineeringOnly")


def test_archived_action_copy_path_cannot_replace_authentic_watch_producer_path(tmp_path):
    path, release = _release(tmp_path)
    watch_path = Path(release["r80_completion"]["watch_status"]["path"])
    watch = pb.read_json(watch_path)
    watch["last_receipt"] = release["r80_completion"]["action_receipt"]["path"]
    pb.atomic_json(watch_path, watch)
    release["r80_completion"]["watch_status"]["sha256"] = pb.file_sha256(watch_path)
    with pytest.raises(ValueError, match="watcher"):
        entry.validate_release(path, _rewrite(path, release), "EngineeringOnly")


def test_tracked_census_binding_rejects_machine_absolute_path(tmp_path):
    relative = "docs/log/assets/r80_best_status_20261010_1246.json"
    tracked = entry.REPO / relative
    resolved, record = _REAL_BOUND_CENSUS_STATUS(
        {"repo_path": relative, "sha256": pb.file_sha256(tracked)}, entry._git_commit())
    assert resolved == tracked and record["validation_status"] == "passed_saved_raw_summary_validation"
    bogus = tmp_path / "r80_status.json"
    bogus.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="repo_path|tracked"):
        _REAL_BOUND_CENSUS_STATUS(
            {"repo_path": str(bogus.resolve()), "sha256": pb.file_sha256(bogus)},
            entry._git_commit())


def test_formal_batch1_without_repeat_recomputes_gate_and_is_accepted(tmp_path):
    path, _ = _release(tmp_path, "Formal")
    report = entry.validate_release(path, pb.file_sha256(path), "Formal")
    assert len(report["engineering_check"]["comparisons"]) == 4
    assert report["job_count"] == 4


def test_formal_batch2_without_previous_feedback_and_trained_model_is_rejected(tmp_path):
    path, _ = _release(tmp_path, "Formal", prefix=2)
    with pytest.raises(ValueError, match="authoritative feedback and trained model"):
        entry.validate_release(path, pb.file_sha256(path), "Formal")


@pytest.mark.skipif(os.environ.get("R81_ALLOW_SYNTHETIC_TRAINING") != "1",
                    reason="requires explicit authorization for synthetic CPU model training")
def test_standard_feedback_train_and_select_authorizes_contiguous_batch2(tmp_path):
    path, release = _release(tmp_path, "Formal")
    dataset = Path(release["dataset_root"])
    work = Path(release["queue"]["exploration"]["work_dir"])
    result_manifest = _complete_formal_batch(dataset, 1)
    ex.import_feedback(work, 1, result_manifest)
    ex.train_models(work, 1)
    selected = ex.select_batch(work, 2)
    shutil.copytree(selected, dataset / "dedust_r81b2_input")
    new_entry = _job("formal_batch", "dedust_r81b2", batch=2)
    new_entry["input_files"] = _files(dataset / "dedust_r81b2_input")
    prior_dataset = dedust.DATASET_PATH
    try:
        dedust.DATASET_PATH = dataset
        job = new_entry["job"]
        dedust.jobs_add(SimpleNamespace(
            input=job["input"], store=job["store"], prio=job["prio"], scope=job["scope"],
            config=str(dataset / job["input"] / "config.yaml"), machine=None))
    finally:
        dedust.DATASET_PATH = prior_dataset
    release["queue"]["jobs"].append(new_entry)
    release["queue"]["jobs_sha256"] = pb.file_sha256(dataset / "jobs.json")
    release["queue"]["exploration"]["files"] = {
        item.relative_to(work).as_posix(): pb.file_sha256(item)
        for item in work.rglob("*") if item.is_file()}
    report = entry.validate_release(path, _rewrite(path, release), "Formal")
    assert report["job_count"] == 5


@pytest.mark.parametrize("corruption,match", [
    ("r80_not_complete", "watcher"),
    ("same_dataset", "separate"),
    ("unknown_job", "unknown"),
    ("false_flag", "release keys"),
    ("stage_downgrade", "stage"),
    ("runtime_blob", "runtime source"),
    ("partial_engineering", ".+"),
    ("mixed_single", "instrument differs|dual profile"),
    ("engineering_disagreement", "engineering measurement check"),
])
def test_entry_rejects_unreleased_or_incomplete_state(tmp_path, corruption, match):
    stage = "Formal" if corruption in {"stage_downgrade", "partial_engineering", "engineering_disagreement"} else "EngineeringOnly"
    path, release = _release(tmp_path, stage, delta=.31 if corruption == "engineering_disagreement" else .1)
    dataset = Path(release["dataset_root"])
    if corruption == "r80_not_complete":
        watch_path = Path(release["r80_completion"]["watch_status"]["path"])
        watch = pb.read_json(watch_path); watch["valid_unique"] = 4999
        pb.atomic_json(watch_path, watch)
        release["r80_completion"]["watch_status"]["sha256"] = pb.file_sha256(watch_path)
    elif corruption == "same_dataset":
        action_path = Path(release["r80_completion"]["action_receipt"]["path"])
        action = pb.read_json(action_path); action["dataset_root"] = str(dataset)
        pb.atomic_json(action_path, action)
        digest = pb.file_sha256(action_path)
        release["r80_completion"]["action_receipt"]["sha256"] = digest
        watch_path = Path(release["r80_completion"]["watch_status"]["path"])
        watch = pb.read_json(watch_path); watch["last_receipt_sha256"] = digest
        pb.atomic_json(watch_path, watch)
        release["r80_completion"]["watch_status"]["sha256"] = pb.file_sha256(watch_path)
    elif corruption == "unknown_job":
        jobs = pb.read_json(dataset / "jobs.json")
        jobs.append({"input": "unknown_input", "store": "unknown", "prio": 1,
                     "config": "config.yaml", "scope": prep.SCOPE})
        pb.atomic_json(dataset / "jobs.json", jobs)
        release["queue"]["jobs_sha256"] = pb.file_sha256(dataset / "jobs.json")
    elif corruption == "false_flag":
        release["engineering_check_passed"] = False
    elif corruption == "stage_downgrade":
        release["stage"] = "EngineeringOnly"
    elif corruption == "runtime_blob":
        release["runtime"]["files"]["script/dedust.py"] = "0" * 64
    elif corruption == "partial_engineering":
        results = dataset / "dedust_r81mesh/results.json"
        values = pb.read_json(results); values.pop(next(iter(values)))
        pb.atomic_json(results, values)
    elif corruption == "mixed_single":
        smoke = dataset / "dedust_r81smoke_input"
        shutil.copy2(entry.REPO / "configs/single_r80_symmetry_factory.yaml", smoke / "config.yaml")
        release["queue"]["jobs"][0]["input_files"] = _files(smoke)
    with pytest.raises(ValueError, match=match):
        entry.validate_release(path, _rewrite(path, release), stage)


def test_release_sha_is_required_and_formal_input_is_unqueued_in_engineering_stage(tmp_path):
    path, release = _release(tmp_path)
    with pytest.raises(ValueError, match="release metadata hash"):
        entry.validate_release(path, "0" * 64, "EngineeringOnly")
    release["queue"]["jobs"][0]["job"]["input"] = "dedust_r81b1_input"
    release["queue"]["jobs"][0]["input_files"] = _files(
        Path(release["dataset_root"]) / "dedust_r81b1_input")
    pb.atomic_json(Path(release["dataset_root"]) / "jobs.json",
                   [item["job"] for item in release["queue"]["jobs"]])
    release["queue"]["jobs_sha256"] = pb.file_sha256(Path(release["dataset_root"]) / "jobs.json")
    with pytest.raises(ValueError, match="scoped input|engineering job|formal stays unqueued"):
        entry.validate_release(path, _rewrite(path, release), "EngineeringOnly")


def test_semantic_queue_status_keeps_partial_fail_retryable_and_exact_roster_terminal(tmp_path):
    path, release = _release(tmp_path)
    state = Path(release["dataset_root"]) / "jobs_state"
    state.mkdir(exist_ok=True)
    store = release["queue"]["jobs"][0]["job"]["store"]
    fail = state / f"{store}.fail"
    claim = state / f"{store}.claim"
    pb.atomic_json(claim, {"machine": entry.EXPECTED_WORKERS[0], "at": "now"})
    pb.atomic_json(fail, {"machines": [entry.EXPECTED_WORKERS[0]], "last": "hfss", "at": "now"})
    assert entry.queue_status(path, pb.file_sha256(path), "EngineeringOnly") == "active"
    pb.atomic_json(claim, {"machine": entry.EXPECTED_WORKERS[2], "at": "now",
                           "prior_fail": list(entry.EXPECTED_WORKERS[:2])})
    pb.atomic_json(fail, {"machines": list(entry.EXPECTED_WORKERS), "last": "hfss", "at": "now"})
    assert entry.queue_status(path, pb.file_sha256(path), "EngineeringOnly") == "active"
    for item in release["queue"]["jobs"][1:]:
        other = item["job"]["store"]
        pb.atomic_json(state / f"{other}.claim",
                       {"machine": entry.EXPECTED_WORKERS[2], "at": "now",
                        "prior_fail": list(entry.EXPECTED_WORKERS[:2])})
        pb.atomic_json(state / f"{other}.fail",
                       {"machines": list(entry.EXPECTED_WORKERS), "last": "hfss", "at": "now"})
    assert entry.queue_status(path, pb.file_sha256(path), "EngineeringOnly") == "terminal_failure"


def test_semantic_queue_status_rejects_stale_done_without_validated_store(tmp_path):
    path, release = _release(tmp_path)
    state = Path(release["dataset_root"]) / "jobs_state"
    state.mkdir(exist_ok=True)
    store = release["queue"]["jobs"][0]["job"]["store"]
    pb.atomic_json(state / f"{store}.claim",
                   {"machine": entry.EXPECTED_WORKERS[0], "at": "now"})
    pb.atomic_json(state / f"{store}.done",
                   {"machine": entry.EXPECTED_WORKERS[0], "at": "now", "errors": 0, "error_ids": []})
    shutil.rmtree(Path(release["dataset_root"]) / store)
    with pytest.raises((FileNotFoundError, ValueError)):
        entry.queue_status(path, pb.file_sha256(path), "EngineeringOnly")


def test_launcher_has_no_pull_queue_write_or_unscoped_worker():
    text = (entry.REPO / "script/start_r81_worker.ps1").read_text(encoding="utf-8")
    assert "git pull" not in text and "jobs-add" not in text
    assert "--scope symmetry_filter_20261007 --selfgen 0 --poll 60 --stale 120 --once" in text
    assert "$CheckOnly" in text and "--queue-status" in text
    assert "$anyFail" not in text and '"jobs_state\\$store.done"' not in text
