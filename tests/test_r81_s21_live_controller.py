"""Focused tests for the cumulative live S21 controller/publisher."""
from __future__ import annotations

import copy
import os
from pathlib import Path

import pytest

from script import profiled_batch as pb
from script import r81_s21_live_controller as controller
from script import r81_s21_live_entry as live
from script import r81_s21_survey_cycle as cycle
from tests.test_filter_s21_training import _prior
from tests.test_r81_s21_survey_cycle import _make_source


def _prefix(tmp_path: Path, context: dict) -> list[dict]:
    registry = tmp_path / "registry" / "batch-001.json"
    registry.parent.mkdir(parents=True, exist_ok=True)
    release = tmp_path / "initial-release.json"
    pb.atomic_json(release, {"fixture": True})
    pb.atomic_json(registry, {"fixture": True})
    return [{"sequence": 1, "path": registry,
             "sha256": pb.file_sha256(registry),
             "record": {"release": {"path": str(release),
                                     "sha256": "a" * 64,
                                     "dataset_root": str(context["dataset"])}},
             "validated": {}}]


def test_real_torch_cutoff_builds_launchable_exact_three_by_twenty(monkeypatch, tmp_path):
    context, _dataset = _make_source(tmp_path, successes=20)
    root = tmp_path / controller.ACTOR_ROOT_NAME
    for name in (controller.REGISTRY_DIR, controller.CONTROLLER_DIR,
                 controller.BATCHES_DIR, controller.FAILURES_DIR):
        (root / name).mkdir(parents=True, exist_ok=True)
    prefix = _prefix(root, context)
    inventory = controller._aggregate_inventory([context])
    cutoff = {"status": "ready", "next_batch": 2, "inventory": inventory,
              "contexts": [context], "consumed": {},
              "new": sorted(inventory["successes"]),
              "required_batch": None,
              "required_batch_new": sorted(inventory["successes"])}
    prior_manifest, prior_root, prior_sha = _prior(tmp_path)
    bootstrap = root / controller.BOOTSTRAP_NAME
    bootstrap.parent.mkdir(parents=True)
    pb.atomic_json(bootstrap, {"fixture": True})
    bootstrap_sha = pb.file_sha256(bootstrap)
    model_root = root / controller.CONTROLLER_DIR / "models"
    model_root.mkdir()

    monkeypatch.setattr(live, "validate_bootstrap", lambda *_args: {"fixture": True})
    monkeypatch.setattr(controller, "_registry_prefix",
                        lambda *_args, **_kwargs: prefix)
    monkeypatch.setattr(controller, "_release_context", lambda _item: context)
    release_path, release_sha = controller._build_release(
        root=root, bootstrap_path=bootstrap, bootstrap_sha=bootstrap_sha,
        prefix=prefix, cutoff=cutoff, prior_manifest=prior_manifest,
        prior_root=prior_root, expected_prior_sha256=prior_sha,
        seed_inputs=[], candidate_pool_size=80, seed=17,
        pretrain_epochs=1, epochs=1, batch_train_size=8, hidden_dims=[8],
        model_output_root=model_root)

    validated = live.validate_successor_release(
        release_path, release_sha,
        {"shared_root": root, "bootstrap_path": bootstrap,
         "bootstrap_sha256": bootstrap_sha})
    assert pb.file_sha256(release_path) == release_sha
    release = pb.read_json(release_path)
    assert validated["release"]["sequence"] == 2
    assert release["sequence"] == 2 and release["policy"] == live.POLICY
    assert set(release["cycle"]) == {
        "batch", "selection_receipt", "candidate_pool_receipt", "rank_receipt",
        "model_completion_receipt", "frozen_snapshot", "canonical_manifest_sha256",
        "split_receipt_sha256"}
    assert [item["job"]["store"] for item in release["queue"]["jobs"]] == [
        "dedust_r81s21live_b002_01", "dedust_r81s21live_b002_02",
        "dedust_r81s21live_b002_03"]
    selection = pb.read_json(Path(release["cycle"]["selection_receipt"]["path"]))
    assert len(selection["trained_snapshot"]["successes"]) == 20
    assert len(selection["trigger_new_success_pattern_sha256"]) == 20
    assert {arm: sum(row["selection_arm"] == arm for row in selection["selected"])
            for arm in cycle.ARM_COUNTS} == cycle.ARM_COUNTS
    assert not (set(inventory["queued"]) &
                {row["pattern_sha256"] for row in selection["selected"]})
    completion = Path(release["cycle"]["model_completion_receipt"]["path"])
    assert pb.file_sha256(completion) == release["cycle"]["model_completion_receipt"]["sha256"]
    plan = pb.read_json(completion.parent / "training_plan.json")
    assert plan["recipe"]["epochs"] == 1 and len(plan["train_ids"]) + len(plan["holdout_ids"]) == 20


def test_batch_three_requires_twenty_new_batch_two_truth(monkeypatch, tmp_path):
    prefix = [{"sequence": 1}, {"sequence": 2}]
    successes = {
        f"{index:064x}": {"sample_sha256": f"{index + 1000:064x}",
                          "released_batch": 1}
        for index in range(40)
    }
    inventory = {"successes": successes, "queued": {}, "pairs": [],
                 "sources": [], "source_inventory_id": "fixture"}
    monkeypatch.setattr(controller, "_release_context", lambda item: item)
    monkeypatch.setattr(controller, "_aggregate_inventory", lambda _items: inventory)
    monkeypatch.setattr(controller, "_consumed", lambda _items: {
        key: value["sample_sha256"] for key, value in list(successes.items())[:20]})
    cutoff = controller._cutoff(prefix)
    assert cutoff["status"] == "waiting"
    assert len(cutoff["new"]) == 20 and cutoff["required_batch_new"] == []

    for index in range(40, 60):
        successes[f"{index:064x}"] = {"sample_sha256": f"{index + 1000:064x}",
                                      "released_batch": 2}
    cutoff = controller._cutoff(prefix)
    assert cutoff["status"] == "ready"
    assert cutoff["required_batch"] == 2
    assert len(cutoff["required_batch_new"]) == 20


def test_pending_registry_append_is_create_only_and_idempotent(monkeypatch, tmp_path):
    root = tmp_path / controller.ACTOR_ROOT_NAME
    (root / controller.CONTROLLER_DIR).mkdir(parents=True)
    (root / controller.REGISTRY_DIR).mkdir()
    batch = root / controller.BATCHES_DIR / "batch-002"
    batch.mkdir(parents=True)
    release = batch / "release.json"
    pb.atomic_json(release, {"fixture": True})
    release_sha = pb.file_sha256(release)
    registry = root / controller.REGISTRY_DIR / "batch-002.json"
    record = {"schema_version": 1, "sequence": 2, "previous": {"fixture": True},
              "bootstrap": {"fixture": True},
              "release": {"path": str(release), "sha256": release_sha,
                          "dataset_root": str(batch)}}
    pending = {"schema_version": 1, "recovery_expected": True,
               "registry_path": str(registry),
               "release": {"path": str(release), "sha256": release_sha},
               "record": copy.deepcopy(record)}
    pb.atomic_json(root / controller.CONTROLLER_DIR / controller.PENDING_NAME, pending)
    monkeypatch.setattr(live, "validate_successor_release", lambda *_args: {})
    monkeypatch.setattr(live, "validate_registry_record", lambda *_args: {})
    monkeypatch.setattr(live, "registry_snapshot", lambda *_args: {"fixture": True})
    monkeypatch.setattr(live, "stop_requested",
                        lambda _snapshot: {"stopped": False, "markers": []})

    fixture_context = {"bootstrap_path": root / controller.BOOTSTRAP_NAME,
                       "bootstrap_sha256": "a" * 64}
    result = controller._finish_pending(root, fixture_context)
    assert result["status"] == "published_successor" and result["recovered"] is True
    assert pb.read_json(registry) == record
    assert not (root / controller.CONTROLLER_DIR / controller.PENDING_NAME).exists()

    # Crash after the registry append but before pending cleanup: exact bytes
    # are accepted, while a different record can never overwrite the prefix.
    pb.atomic_json(root / controller.CONTROLLER_DIR / controller.PENDING_NAME, pending)
    assert controller._finish_pending(root, fixture_context)["recovered"] is True
    pb.atomic_json(root / controller.CONTROLLER_DIR / controller.PENDING_NAME, pending)
    monkeypatch.setattr(live, "stop_requested",
                        lambda _snapshot: {"stopped": True,
                                           "markers": [str(root / "STOP")]})
    stopped = controller._finish_pending(root, fixture_context)
    assert stopped["status"] == "stopped_after_publication"
    assert stopped["registry_changed"] is True and stopped["stopped"] is True
    assert not (root / controller.CONTROLLER_DIR / controller.PENDING_NAME).exists()
    monkeypatch.setattr(live, "stop_requested",
                        lambda _snapshot: {"stopped": False, "markers": []})
    changed = copy.deepcopy(pending); changed["record"]["sequence"] = 3
    pb.atomic_json(root / controller.CONTROLLER_DIR / controller.PENDING_NAME, changed)
    with pytest.raises(ValueError, match="differs"):
        controller._finish_pending(root, fixture_context)


def test_stop_preserves_prepared_release_and_pending_registry_prefix(monkeypatch, tmp_path):
    root = tmp_path / controller.ACTOR_ROOT_NAME
    (root / controller.CONTROLLER_DIR).mkdir(parents=True)
    (root / controller.REGISTRY_DIR).mkdir()
    batch = root / controller.BATCHES_DIR / "batch-002"
    batch.mkdir(parents=True)
    release = batch / "release.json"
    pb.atomic_json(release, {"prepared": True})
    release_sha = pb.file_sha256(release)
    registry = root / controller.REGISTRY_DIR / "batch-002.json"
    pending_path = root / controller.CONTROLLER_DIR / controller.PENDING_NAME
    pb.atomic_json(pending_path, {
        "schema_version": 1, "recovery_expected": True,
        "registry_path": str(registry),
        "release": {"path": str(release), "sha256": release_sha},
        "record": {"schema_version": 1, "sequence": 2}})
    monkeypatch.setattr(live, "validate_successor_release", lambda *_args: {})
    monkeypatch.setattr(live, "registry_snapshot", lambda *_args: {"fixture": True})
    marker = root / "STOP"
    monkeypatch.setattr(live, "stop_requested",
                        lambda _snapshot: {"stopped": True,
                                           "markers": [str(marker)]})
    result = controller._finish_pending(root, {
        "bootstrap_path": root / controller.BOOTSTRAP_NAME,
        "bootstrap_sha256": "a" * 64})
    assert result["status"] == "stopped_before_publication"
    assert result["markers"] == [str(marker)]
    assert pending_path.is_file() and release.is_file()
    assert not registry.exists() and result["registry_changed"] is False


def test_registry_consumer_can_observe_only_absent_or_complete_json(monkeypatch, tmp_path):
    registry = tmp_path / "registry"
    controller_dir = tmp_path / "controller"
    registry.mkdir(); controller_dir.mkdir()
    target = registry / "batch-002.json"
    value = {"schema_version": 1, "sequence": 2,
             "payload": "x" * 250_000}
    native_link = os.link
    observed = []

    def observing_link(source, destination):
        assert not Path(destination).exists()
        native_link(source, destination)
        observed.append(pb.read_json(Path(destination)))

    monkeypatch.setattr(controller.os, "link", observing_link)
    digest = controller._publish_json_atomic_noreplace(target, value, controller_dir)
    assert observed == [value]
    assert pb.read_json(target) == value and pb.file_sha256(target) == digest
    assert not list(controller_dir.glob(".registry-draft-*"))
    with pytest.raises(FileExistsError):
        controller._publish_json_atomic_noreplace(target, value, controller_dir)


def test_completed_prior_cache_is_copied_once_and_family_wait_is_nonpublication(
        monkeypatch, tmp_path):
    root = tmp_path / controller.ACTOR_ROOT_NAME
    (root / controller.CONTROLLER_DIR).mkdir(parents=True)
    source = tmp_path / "r81_s21_sm_actual_fixture"
    (source / "prior" / "signature").mkdir(parents=True)
    pb.atomic_json(source / controller.s21.SPLIT_FILE, {"assignments": {}})
    (source / "prior" / "signature" / "member-0.pt").write_bytes(b"prior")
    receipt = controller._initialize_model_cache(root, source)
    target = root / controller.CONTROLLER_DIR / "models"
    assert receipt["status"] == "copied_completed_prior"
    assert (target / controller.s21.SPLIT_FILE).is_file()
    assert (target / "prior" / "signature" / "member-0.pt").read_bytes() == b"prior"

    snapshot = target / "snapshots" / "fixture"
    snapshot.mkdir(parents=True)
    monkeypatch.setattr(controller.s21, "freeze_successes", lambda *_args: snapshot)
    monkeypatch.setattr(controller.s21, "load_protocol", lambda: {"fixture": True})
    monkeypatch.setattr(controller.s21, "load_current_snapshot",
                        lambda *_args: type("Current", (), {"rows": ({"id": "only"},)})())
    monkeypatch.setattr(controller.s21, "load_split_anchor",
                        lambda *_args: (None, [{"id": "anchor"}]))
    monkeypatch.setattr(controller.s21, "_split", lambda *_args: ([0], [], {}))
    result = controller._family_preflight(
        {"inventory": {"pairs": [(Path("input"), Path("store"))]},
         "contexts": [{"dataset": tmp_path}]}, target, 17)
    assert result == {"snapshot": str(snapshot), "sample_count": 1,
                      "train_count": 1, "holdout_count": 0, "ready": False}
    assert not (root / controller.BATCHES_DIR).exists()


def test_once_waits_on_family_split_without_creating_a_successor(monkeypatch, tmp_path):
    root = tmp_path / controller.ACTOR_ROOT_NAME
    (root / controller.CONTROLLER_DIR / "models").mkdir(parents=True)
    (root / controller.REGISTRY_DIR).mkdir()
    bootstrap = root / controller.BOOTSTRAP_NAME
    bootstrap.parent.mkdir()
    pb.atomic_json(bootstrap, {"fixture": True})
    digest = pb.file_sha256(bootstrap)
    cutoff = {"status": "ready", "next_batch": 2,
              "inventory": {"successes": {f"{index:064x}": {}
                                           for index in range(20)},
                            "source_inventory_id": "stable-cutoff"},
              "consumed": {}, "new": [f"{index:064x}" for index in range(20)],
              "required_batch": None, "required_batch_new": []}
    monkeypatch.setattr(controller, "_bootstrap_context",
                        lambda *_args: ({}, {"registry": {
                            "directory": str(root / controller.REGISTRY_DIR)}}, root))
    monkeypatch.setattr(controller, "_registry_prefix",
                        lambda *_args: [{"sequence": 1, "validated": {}}])
    monkeypatch.setattr(live, "stop_requested",
                        lambda _snapshot: {"stopped": False, "markers": []})
    monkeypatch.setattr(controller, "_cutoff", lambda _prefix: cutoff)
    monkeypatch.setattr(controller, "_family_preflight", lambda *_args: {
        "snapshot": "fixture", "sample_count": 20, "train_count": 20,
        "holdout_count": 0, "ready": False})

    result = controller.once(
        bootstrap=bootstrap, bootstrap_sha256=digest,
        prior_manifest=tmp_path / "unused-prior.json", prior_root=tmp_path,
        expected_prior_sha256="a" * 64, pretrain_epochs=1, epochs=1)
    assert result["status"] == "waiting_for_family_validation"
    assert result["source_inventory_id"] == "stable-cutoff"
    assert not (root / controller.BATCHES_DIR).exists()
