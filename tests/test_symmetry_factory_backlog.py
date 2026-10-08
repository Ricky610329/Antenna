import shutil
import threading
import time
from pathlib import Path

import pytest

from script import profiled_batch as pb
from script import symmetry_factory_backlog as keeper
from script import symmetry_factory_cycle as cycle


REPO = Path(__file__).resolve().parents[1]
PROFILE = REPO / "configs" / "single_r80_symmetry_factory.yaml"
WORKERS = ["140.123.106.216", "140.123.106.218", "140.123.106.37"]


def _dataset(path):
    (path / "jobs_state").mkdir(parents=True)
    pb.atomic_json(path / "jobs.json", [])
    return path


def _queue_add(root, job, scope):
    jobs = pb.read_json(root / "jobs.json")
    jobs.append({"input": job["input"], "store": job["store"],
                 "prio": job["prio"], "scope": scope,
                 "config": f"{job['input']}/config.yaml"})
    pb.atomic_json(root / "jobs.json", jobs)


@pytest.mark.parametrize("pending,capacity,expected", [
    (0, 5000, 96), (32, 5000, 64), (47, 5000, 48),
    (48, 5000, 0), (95, 5000, 0), (0, 47, 0), (0, 48, 48),
])
def test_refill_plan_has_hard_96_high_water_and_complete_shards(
        pending, capacity, expected):
    actual = keeper.plan_refill(pending, capacity)
    assert actual == expected and actual % 16 == 0
    assert pending + actual <= 96


@pytest.mark.parametrize("poll_seconds", [89, 91])
def test_policy_requires_exact_90_second_poll(poll_seconds):
    with pytest.raises(ValueError, match="exactly 90"):
        keeper.BacklogPolicy.from_mapping({"poll_seconds": poll_seconds})


def test_coordinator_prioritizes_mainline_after_one_active_low_turn():
    coordinator = cycle.DatasetWriteCoordinator()
    low_active = threading.Event()
    release_low = threading.Event()
    order = []

    def first_low():
        with coordinator.turn(6):
            order.append("low-1")
            low_active.set()
            release_low.wait()

    def contender(name, priority):
        with coordinator.turn(priority):
            order.append(name)

    first = threading.Thread(target=first_low)
    first.start()
    assert low_active.wait(2)
    second = threading.Thread(target=contender, args=("low-2", 6))
    main = threading.Thread(target=contender, args=("main", 1))
    second.start(); main.start()
    time.sleep(0.05)
    release_low.set()
    first.join(2); second.join(2); main.join(2)
    assert order == ["low-1", "main", "low-2"]


def test_planning_guard_yields_for_training_and_keeps_hashes_reserved():
    coordinator = cycle.DatasetWriteCoordinator()
    low_entered = threading.Event()
    release_low = threading.Event()

    with coordinator.planning_guard(1) as guard:
        def low():
            with coordinator.turn(6):
                low_entered.set()
                release_low.wait()
        thread = threading.Thread(target=low)
        thread.start()
        assert not low_entered.wait(0.05)
        guard.release_for_training()
        assert low_entered.wait(2)
        release_low.set()
        guard.reacquire_after_training()
        receipt = Path("protected-receipt.json")
        guard.reserve(receipt, ["a", "b"])
    thread.join(2)
    assert coordinator.protected_hashes() == {"a", "b"}
    coordinator.release_receipt(receipt)
    assert coordinator.protected_hashes() == set()


def test_low_wait_is_interruptible_during_main_planning():
    coordinator = cycle.DatasetWriteCoordinator()
    cancelled = threading.Event()
    finished = threading.Event()

    with coordinator.planning_guard(1):
        def low():
            with pytest.raises(cycle.CoordinatorCancelled):
                with coordinator.turn(6, cancel_event=cancelled):
                    pytest.fail("cancelled LOW turn acquired coordinator")
            finished.set()
        thread = threading.Thread(target=low)
        thread.start()
        cancelled.set()
        assert finished.wait(2)
    thread.join(2)


def test_actual_training_call_yields_planning_guard_for_low_turn(tmp_path, monkeypatch):
    coordinator = cycle.DatasetWriteCoordinator()
    order = []

    def fake_train(root, version):
        order.append(("train", version))
        with coordinator.turn(6):
            order.append(("low", version))

    monkeypatch.setattr(cycle.training, "train_version", fake_train)
    with coordinator.planning_guard(1) as guard:
        cycle._train_version_with_guard(tmp_path, 12, guard)
        order.append(("reacquired", 12))
    assert order == [("train", 12), ("low", 12), ("reacquired", 12)]


def test_dataset_lock_busy_translation_does_not_capture_body_runtime_error(tmp_path):
    dataset = _dataset(tmp_path / "dataset")
    scope = "scope"
    with cycle._dataset_exclusive_lock(dataset, scope):
        with pytest.raises(cycle.DatasetLockBusy):
            with cycle._dataset_exclusive_lock(dataset, scope):
                pass
    with pytest.raises(RuntimeError, match="another symmetry factory controller holds") as caught:
        with cycle._dataset_exclusive_lock(dataset, scope):
            raise RuntimeError("another symmetry factory controller holds text from body")
    assert type(caught.value) is RuntimeError


def test_commit_is_priority6_idempotent_and_terminal_fail_stays_reserved(tmp_path):
    dataset = _dataset(tmp_path / "dataset")
    pool = keeper.generate_reservoir(PROFILE, tmp_path / "pool", seed=17, count=64,
                                     owner="fixture")
    policy = keeper.BacklogPolicy()
    staged, metadata = keeper.prepare_low_shard(
        PROFILE, [pool], tmp_path / "staged", set(), policy)
    coordinator = cycle.DatasetWriteCoordinator()
    index = tmp_path / "index.json"
    result = keeper.commit_low_shard(
        staged, metadata, dataset=dataset, profile=PROFILE, index_path=index,
        policy=policy, expected_retry_workers=WORKERS, coordinator=coordinator,
        queue_add=_queue_add)
    again = keeper.commit_low_shard(
        staged, metadata, dataset=dataset, profile=PROFILE, index_path=index,
        policy=policy, expected_retry_workers=WORKERS, coordinator=coordinator,
        queue_add=_queue_add)
    job = pb.read_json(dataset / "jobs.json")[0]
    assert result["status"] == "committed" and again["status"] == "already_committed"
    assert job["prio"] == 6 and len(pb.read_json(dataset / job["input"] / "manifest.json")) == 16
    store = job["store"]
    pb.atomic_json(dataset / "jobs_state" / f"{store}.fail", {"machines": WORKERS[:1]})
    cfg = pb.load_profile_config(PROFILE)
    profile_sha256 = pb.file_sha256(PROFILE)
    retryable = keeper.fast_snapshot(
        dataset, cfg, index, policy, WORKERS,
        expected_profile_sha256=profile_sha256)
    assert retryable["own_pending"] == 16
    pb.atomic_json(dataset / "jobs_state" / f"{store}.fail", {"machines": WORKERS})
    terminal = keeper.fast_snapshot(
        dataset, cfg, index, policy, WORKERS,
        expected_profile_sha256=profile_sha256)
    assert terminal["own_pending"] == 0
    assert terminal["reserved_unique"] == 16


@pytest.mark.parametrize("partial_files", [(), ("backlog_job.json",),
                                            ("backlog_job.json", "config.yaml")])
def test_partial_copy_recovery_completes_exact_tree_and_publishes_atomically(
        tmp_path, partial_files):
    dataset = _dataset(tmp_path / "dataset")
    pool = keeper.generate_reservoir(PROFILE, tmp_path / "pool", seed=19, count=32,
                                     owner="fixture")
    staged, metadata = keeper.prepare_low_shard(
        PROFILE, [pool], tmp_path / "work/staged", set(), keeper.BacklogPolicy())
    store = f"dedust_r80k{metadata['cohort_id'][:16]}"
    pending = dataset / f".{store}_input.pending-{metadata['cohort_id'][:12]}"
    pending.mkdir()
    for filename in partial_files:
        shutil.copy2(staged / filename, pending / filename)
    recovered = keeper.recover_pending_inputs(
        dataset, tmp_path / "work/staged", pb.load_profile_config(PROFILE))
    destination = dataset / f"{store}_input"
    assert recovered == [destination] and destination.is_dir() and not pending.exists()
    assert cycle._tree_hashes(destination) == cycle._tree_hashes(staged)


def test_commit_reconstructs_cache_and_deterministic_receipt_after_boundary_crash(tmp_path):
    dataset = _dataset(tmp_path / "dataset")
    pool = keeper.generate_reservoir(PROFILE, tmp_path / "pool", seed=23, count=32,
                                     owner="fixture")
    policy = keeper.BacklogPolicy()
    staged, metadata = keeper.prepare_low_shard(
        PROFILE, [pool], tmp_path / "staged", set(), policy)
    coordinator = cycle.DatasetWriteCoordinator()
    index = tmp_path / "work" / "reservation_index.json"
    first = keeper.commit_low_shard(
        staged, metadata, dataset=dataset, profile=PROFILE, index_path=index,
        policy=policy, expected_retry_workers=WORKERS, coordinator=coordinator,
        queue_add=_queue_add)
    receipt = Path(first["commit_receipt"]["path"])
    expected_receipt = receipt.read_bytes()
    expected_sha256 = first["commit_receipt"]["sha256"]

    # Simulate a crash after the canonical queue/input commit but before either
    # rebuildable local cache artifact survived.
    index.unlink()
    receipt.unlink()
    recovered = keeper.commit_low_shard(
        staged, metadata, dataset=dataset, profile=PROFILE, index_path=index,
        policy=policy, expected_retry_workers=WORKERS, coordinator=coordinator,
        queue_add=_queue_add)
    assert recovered["status"] == "already_committed"
    assert receipt.read_bytes() == expected_receipt
    assert recovered["commit_receipt"]["sha256"] == expected_sha256
    assert len(pb.read_json(dataset / "jobs.json")) == 1

    payload = pb.read_json(receipt)
    payload["status"] = "tampered"
    pb.atomic_json(receipt, payload)
    with pytest.raises(ValueError, match="commit receipt differs"):
        keeper.commit_low_shard(
            staged, metadata, dataset=dataset, profile=PROFILE, index_path=index,
            policy=policy, expected_retry_workers=WORKERS, coordinator=coordinator,
            queue_add=_queue_add)


def test_restart_poll_reconciles_queued_cohort_receipt_without_duplicate_job(
        tmp_path, monkeypatch):
    dataset = _dataset(tmp_path / "dataset")
    pool = keeper.generate_reservoir(PROFILE, tmp_path / "pool", seed=27, count=32,
                                     owner="fixture")
    policy = keeper.BacklogPolicy()
    settings = {"local_workdir": str(tmp_path / "work"),
                "dataset_root": str(dataset), "profile_config": str(PROFILE),
                "prepared_blind_pool": str(pool),
                "expected_retry_workers": WORKERS}
    first_maintainer = keeper.BacklogMaintainer(
        settings=settings, policy=policy,
        coordinator=cycle.DatasetWriteCoordinator())
    staged, metadata = keeper.prepare_low_shard(
        PROFILE, [pool], first_maintainer.work / "staged", set(), policy)
    committed = keeper.commit_low_shard(
        staged, metadata, dataset=dataset, profile=PROFILE,
        index_path=first_maintainer.index_path, policy=policy,
        expected_retry_workers=WORKERS, coordinator=first_maintainer.coordinator,
        queue_add=_queue_add)
    receipt = Path(committed["commit_receipt"]["path"])
    expected_receipt = receipt.read_bytes()
    first_maintainer.index_path.unlink()
    receipt.unlink()

    restarted = keeper.BacklogMaintainer(
        settings=settings, policy=policy,
        coordinator=cycle.DatasetWriteCoordinator())
    monkeypatch.setattr(keeper, "plan_refill", lambda *_args, **_kwargs: 0)
    result = restarted.poll_once()
    assert result["status"] == "not_needed"
    assert len(pb.read_json(dataset / "jobs.json")) == 1
    assert receipt.read_bytes() == expected_receipt


def test_event_chain_rejects_prior_tamper_before_append(tmp_path):
    dataset = _dataset(tmp_path / "dataset")
    pool = keeper.generate_reservoir(PROFILE, tmp_path / "pool", seed=28, count=32,
                                     owner="fixture")
    maintainer = keeper.BacklogMaintainer(
        settings={"local_workdir": str(tmp_path / "work"),
                  "dataset_root": str(dataset), "profile_config": str(PROFILE),
                  "prepared_blind_pool": str(pool),
                  "expected_retry_workers": WORKERS},
        policy=keeper.BacklogPolicy(), coordinator=cycle.DatasetWriteCoordinator())
    first = maintainer._event("first")
    maintainer._event("second")
    payload = pb.read_json(first)
    payload["event"] = "tampered"
    pb.atomic_json(first, payload)
    with pytest.raises(ValueError, match="event chain differs"):
        maintainer._event("third")


@pytest.mark.parametrize("filename", [
    "config.yaml", "measurement.json", "score_spec.json", "manifest.json",
    "backlog_job.json",
])
def test_routine_poll_rehashes_all_indexed_immutable_metadata(tmp_path, filename):
    dataset = _dataset(tmp_path / "dataset")
    pool = keeper.generate_reservoir(PROFILE, tmp_path / "pool", seed=29, count=32,
                                     owner="fixture")
    policy = keeper.BacklogPolicy()
    staged, metadata = keeper.prepare_low_shard(
        PROFILE, [pool], tmp_path / "staged", set(), policy)
    index = tmp_path / "index.json"
    keeper.commit_low_shard(
        staged, metadata, dataset=dataset, profile=PROFILE, index_path=index,
        policy=policy, expected_retry_workers=WORKERS,
        coordinator=cycle.DatasetWriteCoordinator(), queue_add=_queue_add)
    job = pb.read_json(dataset / "jobs.json")[0]
    path = dataset / job["input"] / filename
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="metadata changed|bound profile"):
        keeper.fast_snapshot(
            dataset, pb.load_profile_config(PROFILE), index, policy, WORKERS,
            expected_profile_sha256=pb.file_sha256(PROFILE))


def test_historical_nonkeeper_policy_profile_is_reserved_and_rehashed(tmp_path):
    dataset = _dataset(tmp_path / "dataset")
    source = keeper.generate_reservoir(
        PROFILE, tmp_path / "source", seed=30, count=16, owner="historical")
    historical = dataset / "dedust_r80b1_input"
    shutil.copytree(source, historical)
    config = (historical / "config.yaml").read_text(encoding="utf-8")
    config = config.replace(
        "name: single_r80_symmetry_factory", "name: single_r80_symmetry_explore")
    config = config.replace("target_valid_unique: 5000", "target_valid_unique: 10240")
    (historical / "config.yaml").write_text(config, encoding="utf-8")
    historical_sha256 = pb.file_sha256(historical / "config.yaml")
    assert historical_sha256 != pb.file_sha256(PROFILE)

    cfg = pb.load_profile_config(PROFILE)
    index = tmp_path / "index.json"
    snapshot = keeper.fast_snapshot(
        dataset, cfg, index, keeper.BacklogPolicy(), WORKERS,
        force_rebuild=True, expected_profile_sha256=pb.file_sha256(PROFILE))
    assert snapshot["reserved_unique"] == 16
    assert snapshot["index"]["inputs"][historical.name]["metadata_sha256"][
        "config.yaml"] == historical_sha256

    # The compatible historical bytes are still immutable after indexing.
    changed = config.replace("target_valid_unique: 10240", "target_valid_unique: 10241")
    (historical / "config.yaml").write_text(changed, encoding="utf-8")
    with pytest.raises(ValueError, match="indexed physical input metadata changed"):
        keeper.fast_snapshot(
            dataset, cfg, index, keeper.BacklogPolicy(), WORKERS,
            expected_profile_sha256=pb.file_sha256(PROFILE))


@pytest.mark.parametrize("old,new", [
    ("scope: symmetry_filter_20261007", "scope: incompatible_scope"),
    ("max_passes: 6", "max_passes: 7"),
    ("name: symmetry_observation_only_v1", "name: incompatible_score"),
    ("runtime: {timeout: 1200}", "runtime: {timeout: 1199}"),
])
def test_historical_nonkeeper_rejects_identity_or_fidelity_mismatch(
        tmp_path, old, new):
    dataset = _dataset(tmp_path / "dataset")
    source = keeper.generate_reservoir(
        PROFILE, tmp_path / "source", seed=35, count=16, owner="historical")
    historical = dataset / "dedust_r80b1_input"
    shutil.copytree(source, historical)
    config_path = historical / "config.yaml"
    config = config_path.read_text(encoding="utf-8")
    assert old in config
    config_path.write_text(config.replace(old, new), encoding="utf-8")
    with pytest.raises(ValueError, match="historical physical input config is incompatible"):
        keeper.fast_snapshot(
            dataset, pb.load_profile_config(PROFILE), tmp_path / "index.json",
            keeper.BacklogPolicy(), WORKERS, force_rebuild=True,
            expected_profile_sha256=pb.file_sha256(PROFILE))


@pytest.mark.parametrize("filename", ["measurement.json", "score_spec.json"])
def test_historical_nonkeeper_rejects_physical_measurement_or_score_mismatch(
        tmp_path, filename):
    dataset = _dataset(tmp_path / "dataset")
    source = keeper.generate_reservoir(
        PROFILE, tmp_path / "source", seed=36, count=16, owner="historical")
    historical = dataset / "dedust_r80b1_input"
    shutil.copytree(source, historical)
    path = historical / filename
    payload = pb.read_json(path)
    payload["name"] = "incompatible_physical_metadata"
    pb.atomic_json(path, payload)
    with pytest.raises(ValueError, match="different measurement|different score"):
        keeper.fast_snapshot(
            dataset, pb.load_profile_config(PROFILE), tmp_path / "index.json",
            keeper.BacklogPolicy(), WORKERS, force_rebuild=True,
            expected_profile_sha256=pb.file_sha256(PROFILE))


def test_commit_rejects_changed_source_pool_and_preparation_receipt(tmp_path):
    dataset = _dataset(tmp_path / "dataset")
    pool = keeper.generate_reservoir(PROFILE, tmp_path / "pool", seed=31, count=32,
                                     owner="fixture")
    policy = keeper.BacklogPolicy()
    staged, metadata = keeper.prepare_low_shard(
        PROFILE, [pool], tmp_path / "staged", set(), policy)
    (pool / "manifest.json").write_bytes((pool / "manifest.json").read_bytes() + b"\n")
    with pytest.raises(ValueError, match="source pool metadata changed"):
        keeper.commit_low_shard(
            staged, metadata, dataset=dataset, profile=PROFILE,
            index_path=tmp_path / "index.json", policy=policy,
            expected_retry_workers=WORKERS,
            coordinator=cycle.DatasetWriteCoordinator(), queue_add=_queue_add)

    # Preparation provenance is independently bound even when all source files
    # are unchanged.
    pool2 = keeper.generate_reservoir(PROFILE, tmp_path / "pool2", seed=37, count=32,
                                      owner="fixture")
    staged2, metadata2 = keeper.prepare_low_shard(
        PROFILE, [pool2], tmp_path / "staged2", set(), policy)
    metadata2 = dict(metadata2)
    metadata2["preparation_receipt_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="preparation receipt"):
        keeper.commit_low_shard(
            staged2, metadata2, dataset=dataset, profile=PROFILE,
            index_path=tmp_path / "index2.json", policy=policy,
            expected_retry_workers=WORKERS,
            coordinator=cycle.DatasetWriteCoordinator(), queue_add=_queue_add)

    pool3 = keeper.generate_reservoir(PROFILE, tmp_path / "pool3", seed=39, count=32,
                                      owner="fixture")
    staged3, metadata3 = keeper.prepare_low_shard(
        PROFILE, [pool3], tmp_path / "staged3", set(), policy)
    source_file = pool3 / f"{metadata3['source_bindings'][0]['source_id']}.pt"
    source_file.write_bytes(source_file.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="source tensor differs"):
        keeper.commit_low_shard(
            staged3, metadata3, dataset=dataset, profile=PROFILE,
            index_path=tmp_path / "index3.json", policy=policy,
            expected_retry_workers=WORKERS,
            coordinator=cycle.DatasetWriteCoordinator(), queue_add=_queue_add)


def test_fast_snapshot_is_metadata_only_and_failed_replacements_can_exceed_target(
        tmp_path, monkeypatch):
    dataset = _dataset(tmp_path / "dataset")
    cfg = pb.load_profile_config(PROFILE)
    policy = keeper.BacklogPolicy()
    hashes = [f"h-{index}" for index in range(5001)]
    index = {"schema_version": 1, "jobs": [],
             "jobs_sha256": pb.file_sha256(dataset / "jobs.json"),
             "inputs": {"physical_input": {"input": "physical_input", "count": len(hashes),
                                               "hashes": hashes, "owner": None}}}
    monkeypatch.setattr(keeper, "refresh_index", lambda *args, **kwargs: index)
    monkeypatch.setattr(keeper, "_recheck_index_bindings", lambda *_args: None)
    monkeypatch.setattr(pb, "validate_store", lambda *_args, **_kwargs: pytest.fail(
        "routine keeper poll must not validate result stores"))
    monkeypatch.setattr(cycle, "_queue_view", lambda *_args, **_kwargs: pytest.fail(
        "routine keeper poll must not run the full queue proof"))
    snapshot = keeper.fast_snapshot(
        dataset, cfg, tmp_path / "index.json", policy, WORKERS,
        protected_hashes=["prepared-mainline"])
    assert snapshot["reserved_unique"] == 5002 and snapshot["capacity"] == 0
    assert keeper.plan_refill(snapshot["own_pending"], snapshot["capacity"], policy) == 0


def test_versioned_reservoir_is_stable_and_has_exact_profile(tmp_path):
    first = keeper.generate_reservoir(PROFILE, tmp_path / "reservoir", seed=123, count=64,
                                      owner="r80-low-backlog-v1")
    before = cycle._tree_hashes(first)
    assert keeper.generate_reservoir(PROFILE, first, seed=123, count=64,
                                     owner="r80-low-backlog-v1") == first
    assert cycle._tree_hashes(first) == before
    rows = pb.validate_input(first, pb.load_profile_config(PROFILE))
    assert len(rows) == 64 and len({row["pattern_sha256"] for row in rows}) == 64


def test_real_poll_refills_zero_to96_then_does_not_add_at_high_water(tmp_path, monkeypatch):
    dataset = _dataset(tmp_path / "dataset")
    pool = keeper.generate_reservoir(PROFILE, tmp_path / "pool", seed=321, count=128,
                                     owner="fixture")
    settings = {"local_workdir": str(tmp_path / "work"),
                "dataset_root": str(dataset), "profile_config": str(PROFILE),
                "prepared_blind_pool": str(pool),
                "expected_retry_workers": WORKERS}
    maintainer = keeper.BacklogMaintainer(
        settings=settings, policy=keeper.BacklogPolicy(),
        coordinator=cycle.DatasetWriteCoordinator())
    real_validate = pb.validate_input
    pool_validations = []
    def validate(path, cfg):
        if Path(path).resolve() == pool.resolve():
            pool_validations.append(Path(path))
        return real_validate(path, cfg)
    monkeypatch.setattr(pb, "validate_input", validate)
    first = maintainer.poll_once()
    jobs = pb.read_json(dataset / "jobs.json")
    snapshot = keeper.fast_snapshot(
        dataset, pb.load_profile_config(PROFILE), maintainer.index_path,
        maintainer.policy, WORKERS,
        expected_profile_sha256=pb.file_sha256(PROFILE))
    second = maintainer.poll_once()
    assert first["status"] == "refilled" and len(first["commits"]) == 6
    assert len(jobs) == 6 and {job["prio"] for job in jobs} == {6}
    assert snapshot["own_pending"] == 96
    assert len(pool_validations) == 1
    assert second["status"] == "not_needed"
    assert pb.read_json(dataset / "jobs.json") == jobs


def test_real_maintainer_stop_joins_an_active_transaction(tmp_path, monkeypatch):
    dataset = _dataset(tmp_path / "dataset")
    pool = keeper.generate_reservoir(PROFILE, tmp_path / "pool", seed=41, count=32,
                                     owner="fixture")
    settings = {"local_workdir": str(tmp_path / "work"),
                "dataset_root": str(dataset), "profile_config": str(PROFILE),
                "prepared_blind_pool": str(pool),
                "expected_retry_workers": WORKERS}
    maintainer = keeper.BacklogMaintainer(
        settings=settings, policy=keeper.BacklogPolicy(),
        coordinator=cycle.DatasetWriteCoordinator())
    entered = threading.Event()
    release = threading.Event()
    stopped = threading.Event()

    def active_poll():
        entered.set()
        assert release.wait(5)
        return {"status": "not_needed"}

    monkeypatch.setattr(maintainer, "poll_once", active_poll)
    maintainer.start()
    assert maintainer._thread is not None and not maintainer._thread.daemon
    assert entered.wait(2)

    stopper = threading.Thread(target=lambda: (maintainer.stop(), stopped.set()))
    stopper.start()
    assert not stopped.wait(0.05)
    assert maintainer._thread.is_alive()
    release.set()
    assert stopped.wait(2)
    stopper.join(2)
    assert not maintainer._thread.is_alive()


def test_real_maintainer_propagates_fatal_and_writes_event(tmp_path, monkeypatch):
    dataset = _dataset(tmp_path / "dataset")
    pool = keeper.generate_reservoir(PROFILE, tmp_path / "pool", seed=43, count=32,
                                     owner="fixture")
    settings = {"local_workdir": str(tmp_path / "work"),
                "dataset_root": str(dataset), "profile_config": str(PROFILE),
                "prepared_blind_pool": str(pool),
                "expected_retry_workers": WORKERS}
    maintainer = keeper.BacklogMaintainer(
        settings=settings, policy=keeper.BacklogPolicy(),
        coordinator=cycle.DatasetWriteCoordinator())
    monkeypatch.setattr(
        maintainer, "poll_once",
        lambda: (_ for _ in ()).throw(ValueError("immutable metadata changed")))
    maintainer.start()
    assert maintainer._thread is not None
    maintainer._thread.join(2)
    maintainer.stop()
    with pytest.raises(RuntimeError, match="maintainer failed") as caught:
        maintainer.raise_if_failed()
    assert isinstance(caught.value.__cause__, ValueError)
    events = [pb.read_json(path) for path in sorted(maintainer.events.glob("*.json"))]
    assert [event["event"] for event in events] == ["keeper_started", "keeper_failed"]
