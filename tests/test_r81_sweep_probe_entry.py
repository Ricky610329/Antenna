"""Focused generated-fixture tests for the bounded R81 SweepProbe entry."""
from __future__ import annotations

import copy
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import torch
import yaml

from script import prepare_symmetry_filter as wide
from script import profiled_batch as pb
from script import r81_sweep_probe_entry as probe
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


def _mark_source_done(dataset: Path, entry: dict, machine: str) -> None:
    state = dataset / "jobs_state"
    state.mkdir(exist_ok=True)
    store = entry["job"]["store"]
    pb.atomic_json(state / f"{store}.claim", {"machine": machine, "at": "fixture"})
    pb.atomic_json(state / f"{store}.done",
                   {"machine": machine, "at": "fixture", "errors": 0, "error_ids": []})


def _rewrite(path: Path, release: dict) -> str:
    pb.atomic_json(path, release)
    return pb.file_sha256(path)


def _fixture(tmp_path: Path):
    campaign = tmp_path / "r81_campaign"
    source_root = campaign / "engineering_v001"
    source_root.mkdir(parents=True)
    source_path, source_release = fixtures._release(source_root, "EngineeringOnly")
    source_dataset = Path(source_release["dataset_root"])
    by_role = {entry["role"]: entry for entry in source_release["queue"]["jobs"]}
    owner = base.EXPECTED_WORKERS[0]
    _mark_source_done(source_dataset, by_role["engineering_smoke"], owner)
    _mark_source_done(source_dataset, by_role["engineering_discrete"],
                      base.EXPECTED_WORKERS[1])
    source_sha = pb.file_sha256(source_path)
    binding = probe.collect_source_binding(source_path, source_sha)

    root = campaign / "sweep_probe_v001"
    dataset = root / "dataset"
    release_dir = root / "release"
    dataset.mkdir(parents=True)
    release_dir.mkdir()
    canonical = source_dataset / "dedust_r81b1_input"
    discrete = source_dataset / "dedust_r81discrete_input"
    discrete_config = yaml.safe_load((discrete / "config.yaml").read_text(encoding="utf-8"))
    input_dir = dataset / probe.INPUT_NAME
    wide.auxiliary_bundle(canonical, input_dir, [0, 1],
                          config=discrete_config, repeat=True)
    job = {"input": probe.INPUT_NAME, "store": probe.STORE_NAME, "prio": 5,
           "config": f"{probe.INPUT_NAME}/config.yaml", "scope": base.SCOPE,
           "machine": owner}
    entry = {"role": "same_host_discrete", "job": job,
             "input_files": _files(input_dir)}
    pb.atomic_json(dataset / "jobs.json", [job])
    commit = base._git_commit()
    runtime = {"git_commit": commit,
               "files": {name: pb.file_sha256(base.REPO / name)
                         for name in probe.RUNTIME_FILES}}
    release = {
        "schema_version": 1, "release_id": "r81_sweep_probe_fixture_v1",
        "stage": probe.STAGE, "dataset_root": str(dataset.resolve()),
        "runtime": runtime, "source_release": copy.deepcopy(binding["source_release"]),
        "source_files": copy.deepcopy(binding["files"]),
        "queue": {"jobs_sha256": pb.file_sha256(dataset / "jobs.json"),
                  "jobs": [entry]},
    }
    path = release_dir / "r81_sweep_probe_fixture_v1.json"
    digest = _rewrite(path, release)
    return path, release, digest, source_path, source_release, binding


def _complete_probe(release: dict, *, skipped: bool = False) -> None:
    dataset = Path(release["dataset_root"])
    entry = release["queue"]["jobs"][0]
    input_dir = dataset / entry["job"]["input"]
    store = dataset / entry["job"]["store"]
    cfg = pb.load_profile_config(input_dir / "config.yaml")
    rows = pb.prepare_store(input_dir, store, cfg)
    results = {}
    for index, row in enumerate(rows):
        if skipped and index == 1:
            results[row["id"]] = {
                "error": "watchdog_timeout: generated fixture", "error_kind": "hfss_simulation",
                "attempts": 3, "status": "skipped", "skip_reason": "hfss_attempts_exhausted"}
            continue
        pattern = torch.load(input_dir / row["pattern_file"], weights_only=True)
        response = torch.full((3, 49), -9.0)
        result = pb.observation(response, None, row, pattern, cfg, elapsed=1.0)
        sample = store / result["sample_file"]
        torch.save((pattern, response), sample)
        result["sample_sha256"] = pb.file_sha256(sample)
        results[row["id"]] = result
    pb.atomic_json(store / "results.json", results)
    state = dataset / "jobs_state"
    state.mkdir(exist_ok=True)
    owner = entry["job"]["machine"]
    pb.atomic_json(state / f"{probe.STORE_NAME}.claim", {"machine": owner, "at": "fixture"})
    skipped_ids = [rows[1]["id"]] if skipped else []
    pb.atomic_json(state / f"{probe.STORE_NAME}.done",
                   {"machine": owner, "at": "fixture", "errors": len(skipped_ids),
                    "error_ids": skipped_ids})


def test_probe_source_binding_and_active_release_are_exact(tmp_path):
    path, release, digest, _source_path, _source_release, binding = _fixture(tmp_path)
    report = probe.validate_release(path, digest)
    assert set(binding) == {"source_release", "dataset_root", "files", "owner", "parent_ids"}
    assert len(binding["parent_ids"]) == 2 and len(binding["files"]) > 20
    assert set(release["runtime"]["files"]) == set(probe.RUNTIME_FILES)
    assert len(probe.RUNTIME_FILES) == 48
    assert report["job_count"] == 1 and report["row_count"] == 2
    assert report["owner"] == binding["owner"]
    assert report["queue_status"] == probe.queue_status(path, digest) == "active"
    assert report["formal_release_allowed"] is False
    assert report["engineering_gate_assessed"] is False


@pytest.mark.parametrize("corruption", ["parent", "pattern", "owner", "unknown_marker"])
def test_probe_rejects_changed_parent_pattern_owner_or_unknown_marker(tmp_path, corruption):
    path, release, _digest, _source_path, source_release, _binding = _fixture(tmp_path)
    dataset = Path(release["dataset_root"])
    entry = release["queue"]["jobs"][0]
    input_dir = dataset / probe.INPUT_NAME
    if corruption == "parent":
        rows = pb.read_json(input_dir / "manifest.json")
        rows[0]["parent_batch_id"] = "wrong_parent"
        pb.atomic_json(input_dir / "manifest.json", rows)
        entry["input_files"] = _files(input_dir)
    elif corruption == "pattern":
        rows = pb.read_json(input_dir / "manifest.json")
        source_rows = pb.read_json(Path(source_release["dataset_root"]) /
                                   "dedust_r81b1_input/manifest.json")
        replacement = source_rows[2]
        source_file = Path(source_release["dataset_root"]) / \
            "dedust_r81b1_input" / replacement["pattern_file"]
        shutil.copy2(source_file, input_dir / rows[0]["pattern_file"])
        rows[0]["pattern_sha256"] = replacement["pattern_sha256"]
        pb.atomic_json(input_dir / "manifest.json", rows)
        entry["input_files"] = _files(input_dir)
    elif corruption == "owner":
        entry["job"]["machine"] = base.EXPECTED_WORKERS[2]
        pb.atomic_json(dataset / "jobs.json", [entry["job"]])
        release["queue"]["jobs_sha256"] = pb.file_sha256(dataset / "jobs.json")
    else:
        state = dataset / "jobs_state"
        state.mkdir()
        pb.atomic_json(state / "unknown.done",
                       {"machine": entry["job"]["machine"], "at": "fixture",
                        "errors": 0, "error_ids": []})
    digest = _rewrite(path, release)
    with pytest.raises(ValueError):
        probe.validate_release(path, digest)


@pytest.mark.parametrize("skipped, expected", [(False, "all_success"),
                                                 (True, "terminal_skips")])
def test_probe_native_completion_and_skips_have_terminal_cli_zero(
        tmp_path, skipped, expected, capsys):
    path, release, digest, *_ = _fixture(tmp_path)
    _complete_probe(release, skipped=skipped)
    assert probe.queue_status(path, digest) == expected
    assert probe.main(["--release", str(path), "--release-sha256", digest,
                       "--queue-status"]) == 0
    assert capsys.readouterr().out.strip().endswith(expected)


@pytest.mark.parametrize("member", [probe.INPUT_NAME, "jobs_state", probe.STORE_NAME])
def test_probe_rejects_redirected_input_claim_and_store_directories(tmp_path, member):
    path, release, digest, *_ = _fixture(tmp_path)
    dataset = Path(release["dataset_root"])
    member_path = dataset / member
    outside = tmp_path / ("outside_" + member)
    # Both move targets are owned generated fixtures, never original user data.
    assert member_path.resolve().is_relative_to(tmp_path.resolve())
    assert outside.resolve().is_relative_to(tmp_path.resolve())
    if member_path.exists():
        shutil.move(str(member_path), str(outside))
    else:
        outside.mkdir()
    if os.name == "nt":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(member_path), str(outside)],
                       check=True, capture_output=True)
    else:
        member_path.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="redirected path|escapes its root"):
        probe.validate_release(path, digest)


def test_source_requires_original_first_two_canonical_history_rows(tmp_path):
    (_path, _release, _digest, source_path, source_release, _binding) = _fixture(tmp_path)
    dataset = Path(source_release["dataset_root"])
    canonical = dataset / source_release["queue"]["formal_input"]["name"]
    rows = pb.read_json(canonical / "manifest.json")
    rows[0], rows[2] = rows[2], rows[0]
    rows[1], rows[3] = rows[3], rows[1]
    pb.atomic_json(canonical / "manifest.json", rows)
    source_release["queue"]["formal_input"]["files"] = _files(canonical)
    source_digest = _rewrite(source_path, source_release)
    # The original wide gate permits any two shared smoke parents; this probe
    # deliberately binds ordered indices [0,1], rather than inheriting that gap.
    base.validate_release(source_path, source_digest, "EngineeringOnly")
    with pytest.raises(ValueError, match="first two canonical history rows"):
        probe.collect_source_binding(source_path, source_digest)
