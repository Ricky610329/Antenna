"""Focused generated-fixture tests for the provisional S21 dispatch."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from script import profiled_batch as pb
from script import r81_provisional_filter_dispatch as dispatch
from script import r81_worker_entry as base
from tests import test_r81_worker_entry as fixtures


@pytest.fixture(autouse=True)
def _future_committed_runtime(monkeypatch, tmp_path):
    monkeypatch.setattr(base, "_git_is_ancestor", lambda _commit: True)
    monkeypatch.setattr(base, "_git_path_clean", lambda _commit, _relative: True)
    monkeypatch.setattr(base, "_git_blob_sha",
                        lambda _commit, relative: pb.file_sha256(base.REPO / relative))
    fixtures._CENSUS_FIXTURES.clear()

    def bound_census(binding, _commit):
        path = fixtures._CENSUS_FIXTURES[binding["repo_path"]]
        return path, pb.read_json(path)

    monkeypatch.setattr(base, "_bound_census_status", bound_census)
    monkeypatch.setattr(dispatch, "REPO", tmp_path)


def _files(folder: Path) -> dict[str, str]:
    return {path.name: pb.file_sha256(path)
            for path in sorted(folder.iterdir(), key=lambda item: item.name)
            if path.is_file()}


def _rewrite(path: Path, value: dict) -> str:
    pb.atomic_json(path, value)
    return pb.file_sha256(path)


def _fixture(tmp_path: Path):
    source_root = tmp_path / "source_engineering"
    source_root.mkdir()
    source_path, source_release = fixtures._release(source_root, "EngineeringOnly")
    source_sha = pb.file_sha256(source_path)
    source_dataset = Path(source_release["dataset_root"])
    source_files = {path.relative_to(source_dataset).as_posix(): pb.file_sha256(path)
                    for path in (source_dataset / "dedust_r81b1_input").iterdir()
                    if path.is_file()}
    prep_root = tmp_path / "tmp" / "r81_s21_provisional_fixture"
    preparation = dispatch.prepare_local(source_path, source_sha, prep_root)
    dataset = Path(preparation["dataset_root"])
    release_dir = prep_root / "release"
    release_dir.mkdir()
    commit = base._git_commit()
    release = {
        "schema_version": 1,
        "release_id": "r81_s21_provisional_fixture_v1",
        "stage": dispatch.STAGE,
        "dataset_root": str(dataset),
        "runtime": {"git_commit": commit,
                    "files": {name: pb.file_sha256(base.REPO / name)
                              for name in dispatch.RUNTIME_FILES}},
        "source": copy.deepcopy(preparation["source"]),
        "policy": copy.deepcopy(dispatch.POLICY),
        "queue": {
            "jobs_sha256": pb.file_sha256(dataset / "jobs.json"),
            "jobs": copy.deepcopy(preparation["queue"]["jobs"]),
            "canonical_files": copy.deepcopy(preparation["queue"]["canonical_files"]),
            "split_receipt_sha256": preparation["queue"]["split_receipt_sha256"],
        },
    }
    release_path = release_dir / "r81_s21_provisional_fixture_v1.json"
    digest = _rewrite(release_path, release)
    return release_path, release, digest, preparation, source_files


def test_local_preparation_is_three_pinned_discrete_s21_shards_and_preserves_source(tmp_path):
    path, release, digest, preparation, source_before = _fixture(tmp_path)
    dataset = Path(release["dataset_root"])
    report = dispatch.validate_release(path, digest)

    assert report["queue_status"] == "active"
    assert report["job_count"] == 3 and report["row_count"] == 60
    assert report["policy"] == dispatch.POLICY
    assert report["policy"]["formal_release_allowed"] is False
    assert report["policy"]["engineering_gate_satisfied"] is False
    assert report["policy"]["engineering_gate_bypassed"] is False
    assert len(dispatch.RUNTIME_FILES) == 51 and set(dispatch.probe.RUNTIME_FILES) < set(
        dispatch.RUNTIME_FILES)

    jobs = preparation["queue"]["jobs"]
    assert [entry["job"]["machine"] for entry in jobs] == list(base.EXPECTED_WORKERS)
    assert [entry["job"]["prio"] for entry in jobs] == [dispatch.LOW_TIER_PRIORITY] * 3
    assert [len(pb.read_json(dataset / Path(entry["job"]["input"]) / "manifest.json"))
            for entry in jobs] == [20, 20, 20]
    cfg = pb.load_profile_config(dataset / dispatch.CANONICAL_INPUT / "config.yaml")
    assert cfg.measurement["sweep"]["type"] == "Discrete"
    assert cfg.score_spec == dispatch.S21_SPEC
    assert {band["label"] for band in cfg.score_spec["bands"]} == {"S21"}
    assert preparation["publication"] == {
        "release_created": False, "current_pointer_changed": False,
        "nas_written": False, "worker_started": False, "hfss_started": False}

    source_dataset = Path(preparation["source"]["dataset_root"])
    source_after = {path.relative_to(source_dataset).as_posix(): pb.file_sha256(path)
                    for path in (source_dataset / "dedust_r81b1_input").iterdir()
                    if path.is_file()}
    assert source_after == source_before


@pytest.mark.parametrize("change", ["owner", "priority", "score", "geometry",
                                     "shard_duplicate", "unknown_marker"])
def test_release_rejects_machine_priority_profile_and_marker_drift(tmp_path, change):
    path, release, _digest, _preparation, _source = _fixture(tmp_path)
    dataset = Path(release["dataset_root"])
    if change == "owner":
        release["queue"]["jobs"][0]["job"]["machine"] = base.EXPECTED_WORKERS[1]
        pb.atomic_json(dataset / "jobs.json",
                       [entry["job"] for entry in release["queue"]["jobs"]])
        release["queue"]["jobs_sha256"] = pb.file_sha256(dataset / "jobs.json")
    elif change == "priority":
        release["queue"]["jobs"][0]["job"]["prio"] = 1
        pb.atomic_json(dataset / "jobs.json",
                       [entry["job"] for entry in release["queue"]["jobs"]])
        release["queue"]["jobs_sha256"] = pb.file_sha256(dataset / "jobs.json")
    elif change == "score":
        folder = dataset / dispatch.SPLIT_ROOT / "shard-001_input"
        spec = pb.read_json(folder / "score_spec.json")
        spec["bands"][0]["threshold"] = -2
        pb.atomic_json(folder / "score_spec.json", spec)
        release["queue"]["jobs"][0]["input_files"] = _files(folder)
    elif change == "geometry":
        folder = dataset / dispatch.SPLIT_ROOT / "shard-001_input"
        config = dispatch.yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
        config["measurement"]["solver"]["max_passes"] = 7
        (folder / "config.yaml").write_text(
            dispatch.yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        pb.atomic_json(folder / "measurement.json", config["measurement"])
        rows = pb.read_json(folder / "manifest.json")
        for row in rows:
            row["measurement_id"] = dispatch.measurement_id(config["measurement"])
        pb.atomic_json(folder / "manifest.json", rows)
        release["queue"]["jobs"][0]["input_files"] = _files(folder)
    elif change == "shard_duplicate":
        first = dataset / dispatch.SPLIT_ROOT / "shard-001_input"
        second = dataset / dispatch.SPLIT_ROOT / "shard-002_input"
        rows = pb.read_json(first / "manifest.json")
        duplicate = pb.read_json(second / "manifest.json")[0]
        old_file = first / rows[0]["pattern_file"]
        old_file.unlink()
        rows[0] = copy.deepcopy(duplicate)
        (first / duplicate["pattern_file"]).write_bytes(
            (second / duplicate["pattern_file"]).read_bytes())
        pb.atomic_json(first / "manifest.json", rows)
        release["queue"]["jobs"][0]["input_files"] = _files(first)
        receipt_path = dataset / dispatch.SPLIT_ROOT / "split_receipt.json"
        receipt = pb.read_json(receipt_path)
        receipt["shards"][0]["ids"] = [row["id"] for row in rows]
        receipt["shards"][0]["items"][0] = {
            "id": duplicate["id"], "pattern_file": duplicate["pattern_file"],
            "file_sha256": pb.file_sha256(first / duplicate["pattern_file"]),
            "pattern_sha256": duplicate["pattern_sha256"]}
        receipt["shards"][0]["manifest_sha256"] = pb.file_sha256(first / "manifest.json")
        pb.atomic_json(receipt_path, receipt)
        release["queue"]["split_receipt_sha256"] = pb.file_sha256(receipt_path)
    else:
        state = dataset / "jobs_state"
        state.mkdir()
        pb.atomic_json(state / "foreign.done",
                       {"machine": base.EXPECTED_WORKERS[0], "at": "fixture",
                        "errors": 0, "error_ids": []})
    digest = _rewrite(path, release)
    with pytest.raises(ValueError):
        dispatch.validate_release(path, digest)


def test_runtime_rejects_any_change_to_protected_sweepprobe48(tmp_path, monkeypatch):
    path, release, _digest, *_ = _fixture(tmp_path)
    release["runtime"]["git_commit"] = "f" * 40
    protected = next(iter(dispatch.probe.RUNTIME_FILES))
    real = base._git_blob_sha

    def changed(commit, relative):
        if commit == dispatch.PROTECTED_RUNTIME_COMMIT and relative == protected:
            return "0" * 64
        return real(commit, relative)

    monkeypatch.setattr(base, "_git_blob_sha", changed)
    digest = _rewrite(path, release)
    with pytest.raises(ValueError, match="protected SweepProbe runtime"):
        dispatch.validate_release(path, digest)


def test_launcher_requires_host_pin_and_rejects_existing_workers():
    text = (base.REPO / "script/start_r81_provisional_filter_worker.ps1").read_text(
        encoding="utf-8")
    assert "ExpectedMachine" in text and "get_local_ip" in text
    assert "Get-CimInstance Win32_Process" in text
    assert "script[.]dedust" in text and "do not double-dispatch" in text
    assert "--scope symmetry_filter_20261007 --selfgen 0" in text
