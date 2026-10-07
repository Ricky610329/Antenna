from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from antenna.measurement import measurement_id, score_spec_id
from antenna.utils.store import fingerprint
from script import profiled_batch as pb
from script import symmetry_factory_cycle as cycle
from script import symmetry_training as training


REPO = Path(__file__).resolve().parents[1]
PROFILE = REPO / "configs" / "single_r80_symmetry_factory.yaml"


class ColdPredictor:
    model_role = "historical_cold_start_only"
    model_ids = [{"file": "fixture", "sha256": "0" * 64}]


def _pattern(index: int) -> torch.Tensor:
    value = torch.zeros((25, 25), dtype=torch.float32)
    value[24, 12] = 1
    for bit in range(10):
        if index & (1 << bit):
            row, col = 1 + bit * 2, 1 + bit % 11
            value[row, col] = value[row, 24 - col] = 1
    return value


def _bundle(root: Path, count: int, *, prefix: str = "item", start: int = 0) -> Path:
    root.mkdir(parents=True)
    shutil.copy2(PROFILE, root / "config.yaml")
    cfg = pb.load_profile_config(root / "config.yaml")
    pb.atomic_json(root / "measurement.json", cfg.measurement)
    pb.atomic_json(root / "score_spec.json", cfg.score_spec)
    mid, sid = measurement_id(cfg.measurement), score_spec_id(cfg.score_spec)
    rows = []
    for ordinal in range(start, start + count):
        pattern = _pattern(ordinal)
        name = f"{prefix}-{ordinal:05d}"
        torch.save(pattern, root / f"{name}.pt")
        rows.append({"id": name, "pattern_file": f"{name}.pt", "port": "single",
                     "measurement_id": mid, "score_spec_id": sid,
                     "pattern_sha256": pb.pattern_sha256(pattern),
                     "lineage_id": f"line-{ordinal}", "prediction": [float(ordinal)]})
    pb.atomic_json(root / "manifest.json", rows)
    pb.validate_input(root, cfg)
    return root


def _complete(input_dir: Path, store: Path) -> Path:
    cfg = pb.load_profile_config(input_dir / "config.yaml")
    rows = pb.prepare_store(input_dir, store, cfg)
    (store / "rad").mkdir(exist_ok=True)
    results = {}
    theta = torch.arange(-180, 181, 2, dtype=torch.float32)
    for index, row in enumerate(rows):
        pattern = torch.load(input_dir / f"{row['id']}.pt", weights_only=True)
        response = torch.stack((torch.linspace(-15, -10, 17) - index / 100,
                                torch.linspace(3, 5, 17) + index / 100))
        rad = {"theta": theta, "phi0": torch.zeros_like(theta) + index,
               "phi90": torch.zeros_like(theta) - index}
        entry = pb.observation(response, rad, row, pattern, cfg, 1.0)
        sample = store / entry["sample_file"]
        torch.save((pattern, response), sample)
        assert sample.name == fingerprint(pattern, response) + ".pt"
        rad_path = store / entry["rad_file"]
        torch.save(rad, rad_path)
        entry["sample_sha256"] = pb.file_sha256(sample)
        entry["rad_sha256"] = pb.file_sha256(rad_path)
        results[row["id"]] = entry
    pb.atomic_json(store / "results.json", results)
    pb.validate_store(store, require_complete=True)
    return store


def _attach_saved_predictions(input_dir: Path, store: Path, train: Path, *,
                              missing_gain_on_last: bool = False,
                              constant_scores: bool = False) -> None:
    protocol = training._protocol(train)
    rows = pb.read_json(input_dir / "manifest.json")
    results = pb.read_json(store / "results.json")
    theta = np.linspace(-90.0, 90.0, 91, dtype=np.float32)
    for index, row in enumerate(rows):
        entry = results[row["id"]]
        _pattern_value, target = training._target(
            store / entry["sample_file"], store / entry["rad_file"],
            protocol["frequencies_ghz"])
        response = target[:34].numpy().reshape(2, 17)
        radiation = target[34:].numpy().reshape(2, 91)
        scores = cycle.sm.score_predictions(cycle.sm.PredictionBatch(
            response[None], np.zeros(1, dtype=np.float32), radiation[None], theta))
        row.update({
            "predicted_s11": response[0].tolist(),
            "predicted_gain": response[1].tolist(),
            "predicted_radiation_phi0": radiation[0].tolist(),
            "predicted_radiation_phi90": radiation[1].tolist(),
            "predicted_radiation_theta": theta.tolist(),
            "factory_score": (1.0 if constant_scores else float(scores["factory_score"][0])),
            "navigation_score": (1.0 if constant_scores else
                                 float(scores["navigation_score"][0])),
            "selection_arm": "guided" if index % 2 == 0 else "blind",
        })
    if missing_gain_on_last:
        rows[-1].pop("predicted_gain")
    pb.atomic_json(input_dir / "manifest.json", rows)
    pb.atomic_json(store / "manifest.json", rows)


def _training(root: Path) -> Path:
    cfg = pb.load_profile_config(PROFILE)
    return training.initialize_workdir(
        root, cfg.measurement, cfg.score_spec, geometry_profile="single_v1",
        holdout_seed=7, holdout_fraction=0.2, ensemble_seeds=(1,), hidden_dims=(8,),
        legacy_epochs=0, current_epochs=1, update_min=48, update_max=96,
    )


def _dataset(root: Path) -> Path:
    root.mkdir(); (root / "jobs_state").mkdir()
    pb.atomic_json(root / "jobs.json", [])
    return root


def _dispatch_receipt(path: Path, dataset: Path, train: Path,
                      jobs: list[dict], cycle_id: str) -> Path:
    path.parent.mkdir(parents=True)
    pb.atomic_json(path, {
        "schema_version": 1, "status": "prepared", "cycle_id": cycle_id,
        "dataset_root": str(dataset), "scope": pb.load_profile_config(PROFILE).scope,
        "profile_config": str(PROFILE.resolve()), "profile_sha256": pb.file_sha256(PROFILE),
        "training_workdir": str(train.resolve()),
        "training_protocol": cycle._protocol_binding(train)[1], "planned_jobs": jobs,
    })
    return path


def _fake_pool(monkeypatch):
    calls = []

    def build(config, seed_rows, exclude_hashes, predictor=None):
        calls.append((config.selected_count, len(exclude_hashes), predictor))
        return {"count": config.selected_count, "config": config}

    def write(result, output):
        return _bundle(Path(output), result["count"], prefix=result["config"].id_prefix)

    monkeypatch.setattr(cycle.sm, "build_pool", build)
    monkeypatch.setattr(cycle.sm, "write_bundle", write)
    monkeypatch.setattr(cycle.sm, "write_audit",
                        lambda result, output, elapsed: pb.atomic_json(
                            Path(output) / "sm_pool_audit.json", {"fixture": True}))
    return calls


def test_prepare_once_is_local_idempotent_and_splits_guided_48(tmp_path, monkeypatch):
    dataset = _dataset(tmp_path / "dataset")
    train = _training(tmp_path / "training")
    seeds = _bundle(tmp_path / "seeds", 4)
    calls = _fake_pool(monkeypatch)

    receipt_path = cycle.run_once(
        local_workdir=tmp_path / "cycle", dataset_root=dataset, profile_config=PROFILE,
        training_workdir=train, seed_inputs=[seeds], cold_start_predictor=ColdPredictor())
    receipt = pb.read_json(receipt_path)

    assert receipt["status"] == "prepared" and receipt["dispatch_gate"] == "not_run"
    assert [job["count"] for job in receipt["planned_jobs"]] == [16, 16, 16]
    assert {job["prio"] for job in receipt["planned_jobs"]} == {1}
    assert pb.read_json(dataset / "jobs.json") == []
    assert cycle.run_once(local_workdir=tmp_path / "cycle", dataset_root=dataset,
                          profile_config=PROFILE, training_workdir=train,
                          seed_inputs=[seeds], cold_start_predictor=ColdPredictor()) == receipt_path
    assert len(calls) == 1


def test_prepare_once_uses_three_blind_shards_when_no_model(tmp_path):
    dataset = _dataset(tmp_path / "dataset")
    train = _training(tmp_path / "training")
    seeds = _bundle(tmp_path / "seeds", 2)
    blind = _bundle(tmp_path / "blind", 64, prefix="blind", start=100)

    receipt = pb.read_json(cycle.run_once(
        local_workdir=tmp_path / "cycle", dataset_root=dataset, profile_config=PROFILE,
        training_workdir=train, seed_inputs=[seeds], prepared_blind_pool=blind))

    assert [job["count"] for job in receipt["planned_jobs"]] == [16, 16, 16]
    assert {job["kind"] for job in receipt["planned_jobs"]} == {"b"}
    assert {job["prio"] for job in receipt["planned_jobs"]} == {6}


def test_idle_cycle_does_not_pin_active_receipt(tmp_path):
    dataset = _dataset(tmp_path / "dataset")
    train = _training(tmp_path / "training")
    seeds = _bundle(tmp_path / "seeds", 2)

    receipt_path = cycle.single_cycle(
        local_workdir=tmp_path / "cycle", dataset_root=dataset, profile_config=PROFILE,
        training_workdir=train, seed_inputs=[seeds])

    assert pb.read_json(receipt_path)["status"] == "idle"
    assert not (tmp_path / "cycle/active_cycle.json").exists()


def test_48_saved_successes_are_frozen_then_ingested_and_trained(tmp_path, monkeypatch):
    dataset = _dataset(tmp_path / "dataset")
    input_dir = _bundle(dataset / "wave_input", 48)
    _complete(input_dir, dataset / "wave")
    cfg = pb.load_profile_config(PROFILE)
    pb.atomic_json(dataset / "jobs.json", [{"input": "wave_input", "store": "wave",
                                             "prio": 6, "scope": cfg.scope,
                                             "config": "wave_input/config.yaml"}])
    pb.atomic_json(dataset / "jobs_state/wave.done", {"ok": True})
    train = _training(tmp_path / "training")
    seeds = _bundle(tmp_path / "seeds", 3, start=200)
    _fake_pool(monkeypatch)
    calls = []
    real_audit = cycle.audit_frozen_predictions

    def audit_predictions(*args, **kwargs):
        calls.append(("audit",))
        return real_audit(*args, **kwargs)

    def ingest(work, stores, *, min_new, max_new):
        assert len(pb.validate_store(stores[0], require_complete=True)) == 48
        calls.append(("ingest", min_new, max_new))
        result = Path(work) / "data/data-v001"
        result.mkdir(parents=True, exist_ok=True)
        return result

    monkeypatch.setattr(cycle.training, "ingest_profile_stores", ingest)
    monkeypatch.setattr(cycle, "audit_frozen_predictions", audit_predictions)
    monkeypatch.setattr(cycle.training, "train_version",
                        lambda work, version: calls.append(("train", version)))
    monkeypatch.setattr(cycle.training, "load_current_predictor",
                        lambda work, version: ColdPredictor())

    receipt = pb.read_json(cycle.run_once(
        local_workdir=tmp_path / "cycle", dataset_root=dataset, profile_config=PROFILE,
        training_workdir=train, seed_inputs=[seeds]))

    assert receipt["trained_this_cycle"] is True
    assert receipt["new_unique_available_before_training"] == 48
    assert calls == [("audit",), ("ingest", 48, 96), ("train", 1)]
    assert receipt["pretrain_prediction_audit"]["sha256"] == pb.file_sha256(
        Path(receipt["pretrain_prediction_audit"]["path"]))
    assert receipt["training_protocol"]["protocol_id"] == training._protocol(train)["protocol_id"]
    assert len(pb.validate_store(tmp_path / "cycle/cycles" / receipt["cycle_id"] /
                                 "training_snapshot", require_complete=True)) == 48


def test_prospective_prediction_audit_reports_error_ranks_arms_and_missing(tmp_path):
    train = _training(tmp_path / "training")
    input_dir = _bundle(tmp_path / "input", 4)
    store = _complete(input_dir, tmp_path / "store")
    _attach_saved_predictions(input_dir, store, train, missing_gain_on_last=True)
    rows = pb.read_json(input_dir / "manifest.json")
    rows[0]["prediction_valid"] = False  # Placeholder arrays must not count as predictions.
    pb.atomic_json(input_dir / "manifest.json", rows)
    pb.atomic_json(store / "manifest.json", rows)
    snapshot = cycle.snapshot_successes([(input_dir, store)], tmp_path / "snapshot")

    path = cycle.audit_frozen_predictions(snapshot, train, tmp_path / "audit.json")
    audit = pb.read_json(path)

    assert audit["order"] == "saved_predictions_audited_before_ingest_and_training"
    assert audit["quality_gate"] is None
    assert audit["all_valid_observations_retained_for_training"] is True
    assert audit["overall"]["n"] == 4
    assert audit["overall"]["prediction_complete"] == 2
    assert audit["overall"]["prediction_explicitly_invalid"] == 1
    assert audit["overall"]["prediction_missing_by_field"]["predicted_gain"] == 2
    assert audit["overall"]["components"]["s11"]["mean_absolute_error"] == pytest.approx(0)
    assert audit["overall"]["spearman"]["factory_score"]["rho"] == pytest.approx(1)
    assert set(audit["selection_arms"]) == {"blind", "guided"}
    assert cycle.audit_frozen_predictions(snapshot, train, path) == path


def test_prediction_audit_never_fabricates_rho_for_constant_predictions(tmp_path):
    train = _training(tmp_path / "training")
    input_dir = _bundle(tmp_path / "input", 3)
    store = _complete(input_dir, tmp_path / "store")
    _attach_saved_predictions(input_dir, store, train, constant_scores=True)
    snapshot = cycle.snapshot_successes([(input_dir, store)], tmp_path / "snapshot")

    audit = pb.read_json(cycle.audit_frozen_predictions(snapshot, train, tmp_path / "audit.json"))
    rank = audit["overall"]["spearman"]["factory_score"]
    assert rank["n"] == 3 and rank["rho"] is None
    assert rank["undefined_reason"] == "constant_prediction_or_measurement"


def test_source_change_during_exact_training_snapshot_rejects_before_ingest(tmp_path, monkeypatch):
    dataset = _dataset(tmp_path / "dataset")
    input_dir = _bundle(dataset / "wave_input", 48)
    store = _complete(input_dir, dataset / "wave")
    cfg = pb.load_profile_config(PROFILE)
    pb.atomic_json(dataset / "jobs.json", [{"input": "wave_input", "store": "wave",
                                             "prio": 6, "scope": cfg.scope,
                                             "config": "wave_input/config.yaml"}])
    pb.atomic_json(dataset / "jobs_state/wave.done", {"ok": True})
    train = _training(tmp_path / "training")
    seeds = _bundle(tmp_path / "seeds", 2, start=200)
    original = cycle.snapshot_successes
    called = []

    def mutate_then_snapshot(pairs, output, **kwargs):
        with (store / "results.json").open("a", encoding="utf-8") as stream:
            stream.write("\n")
        return original(pairs, output, **kwargs)

    monkeypatch.setattr(cycle, "snapshot_successes", mutate_then_snapshot)
    monkeypatch.setattr(cycle.training, "ingest_profile_stores",
                        lambda *args, **kwargs: called.append("ingest"))

    with pytest.raises(cycle.LiveSnapshotChanged, match="changed after physical audit"):
        cycle.run_once(local_workdir=tmp_path / "cycle", dataset_root=dataset,
                       profile_config=PROFILE, training_workdir=train, seed_inputs=[seeds])
    assert called == []


def test_store_manifest_change_is_fatal_not_a_retryable_results_race(tmp_path):
    dataset = _dataset(tmp_path / "dataset")
    input_dir = _bundle(dataset / "wave_input", 2)
    store = _complete(input_dir, dataset / "wave")
    cfg = pb.load_profile_config(PROFILE)
    pb.atomic_json(dataset / "jobs.json", [{"input": "wave_input", "store": "wave",
                                             "prio": 6, "scope": cfg.scope,
                                             "config": "wave_input/config.yaml"}])
    policy = cycle.factory.validate_factory_config(vars(cfg))
    bindings = cycle._queue_view(dataset, cfg, policy)["pair_bindings"]
    with (store / "manifest.json").open("a", encoding="utf-8") as stream:
        stream.write("\n")

    with pytest.raises(ValueError, match="manifest.json") as caught:
        cycle._require_pair_bindings(bindings)
    assert not isinstance(caught.value, cycle.LiveSnapshotChanged)


def test_guided_preparation_uses_only_remaining_outstanding_capacity(tmp_path, monkeypatch):
    dataset = _dataset(tmp_path / "dataset")
    train = _training(tmp_path / "training")
    seeds = _bundle(tmp_path / "seeds", 3)
    calls = _fake_pool(monkeypatch)
    pending = {f"reserved-{index}" for index in range(80)}
    jobs_hash = pb.file_sha256(dataset / "jobs.json")
    monkeypatch.setattr(cycle.factory, "snapshot", lambda *args: {"jobs_sha256": jobs_hash})
    monkeypatch.setattr(cycle, "_queue_view", lambda *args: {
        "jobs_sha256": jobs_hash, "jobs": [], "pairs": [], "pending_hashes": pending,
        "guided_pending_hashes": pending, "successful_hashes": set(),
        "exclusion_hashes": pending, "pair_bindings": [],
    })
    monkeypatch.setattr(cycle, "_trained",
                        lambda root: (1, set(), {"manifest_id": "bound", "data_version": 1}))
    monkeypatch.setattr(cycle.training, "load_current_predictor",
                        lambda root, version: ColdPredictor())

    receipt = pb.read_json(cycle.run_once(
        local_workdir=tmp_path / "cycle", dataset_root=dataset, profile_config=PROFILE,
        training_workdir=train, seed_inputs=[seeds]))

    assert calls[0][0] == 16
    assert [job["count"] for job in receipt["planned_jobs"]] == [16]
    assert receipt["limits"]["valid_plus_pending_plus_planned"] == 96


def test_regular_guided_wave_waits_when_capacity_is_below_one_shard(tmp_path, monkeypatch):
    dataset = _dataset(tmp_path / "dataset")
    train = _training(tmp_path / "training")
    seeds = _bundle(tmp_path / "seeds", 3)
    calls = _fake_pool(monkeypatch)
    pending = {f"reserved-{index}" for index in range(81)}
    jobs_hash = pb.file_sha256(dataset / "jobs.json")
    monkeypatch.setattr(cycle.factory, "snapshot", lambda *args: {"jobs_sha256": jobs_hash})
    monkeypatch.setattr(cycle, "_queue_view", lambda *args: {
        "jobs_sha256": jobs_hash, "jobs": [], "pairs": [], "pair_bindings": [],
        "pending_hashes": pending, "guided_pending_hashes": pending,
        "successful_hashes": set(), "exclusion_hashes": pending,
    })
    monkeypatch.setattr(cycle, "_trained",
                        lambda root: (1, set(), {"manifest_id": "bound", "data_version": 1}))
    monkeypatch.setattr(cycle.training, "load_current_predictor",
                        lambda root, version: ColdPredictor())

    receipt = pb.read_json(cycle.run_once(
        local_workdir=tmp_path / "cycle", dataset_root=dataset, profile_config=PROFILE,
        training_workdir=train, seed_inputs=[seeds]))

    assert receipt["status"] == "idle"
    assert receipt["planned_jobs"] == []
    assert calls == []


def test_atomic_bundle_publish_recovers_invalid_known_pending_directory(tmp_path):
    output = tmp_path / "guided_bundle"
    pending = tmp_path / ".guided_bundle.pending"
    pending.mkdir()
    (pending / "partial.txt").write_text("crash", encoding="utf-8")
    cfg = pb.load_profile_config(PROFILE)

    cycle._publish_bundle(output, cfg, lambda path: _bundle(path, 2))

    assert not pending.exists()
    assert len(pb.validate_input(output, cfg)) == 2


def test_atomic_bundle_publish_rebuilds_valid_core_missing_completion_audit(tmp_path):
    output = tmp_path / "guided_bundle"
    pending = _bundle(tmp_path / ".guided_bundle.pending", 2)
    cfg = pb.load_profile_config(PROFILE)
    calls = []

    def produce(path):
        calls.append(path)
        _bundle(path, 2)
        pb.atomic_json(path / "sm_pool_audit.json", {"complete": True})

    def validate(path):
        if pb.read_json(path / "sm_pool_audit.json") != {"complete": True}:
            raise ValueError("missing complete audit")

    cycle._publish_bundle(output, cfg, produce, validate)

    assert calls and not pending.exists()
    assert pb.read_json(output / "sm_pool_audit.json") == {"complete": True}


def test_kernel_lock_file_is_recoverable_and_retains_owner_metadata(tmp_path):
    path = tmp_path / "controller.lock"
    with cycle._exclusive_lock(path):
        assert path.is_file()
    owner = path.read_text(encoding="utf-8")
    assert "created_utc" in owner and "host" in owner and "pid" in owner
    with cycle._exclusive_lock(path):
        assert path.is_file()


def test_same_scope_wrong_measurement_is_rejected_instead_of_silently_skipped(tmp_path):
    dataset = _dataset(tmp_path / "dataset")
    input_dir = _bundle(dataset / "wrong_input", 1)
    marker = pb.read_json(input_dir / "measurement.json")
    marker["sweep"]["start_ghz"] += 0.5
    marker["sweep"]["stop_ghz"] += 0.5
    pb.atomic_json(input_dir / "measurement.json", marker)
    cfg = pb.load_profile_config(PROFILE)
    pb.atomic_json(dataset / "jobs.json", [{"input": "wrong_input", "store": "wrong",
                                             "prio": 6, "scope": cfg.scope,
                                             "config": "wrong_input/config.yaml"}])

    with pytest.raises(ValueError, match="wrong measurement"):
        cycle._queue_view(dataset, cfg, cycle.factory.validate_factory_config(vars(cfg)))


def test_commit_checks_each_shard_and_is_idempotent(tmp_path):
    dataset = _dataset(tmp_path / "dataset")
    train = _training(tmp_path / "training")
    staged = [_bundle(tmp_path / f"staged-{i}", 2, prefix=f"p{i}", start=i * 10)
              for i in range(2)]
    jobs = cycle._planned(staged, "g", 1, "a" * 64)
    receipt_path = _dispatch_receipt(
        tmp_path / "local/cycles/test/action_receipt.json", dataset, train, jobs, "a" * 64)
    checked, added = [], []

    def duplicate(input_dir, root):
        checked.append(input_dir.name)
        return len(pb.read_json(input_dir / "manifest.json"))

    def add(root, job, scope):
        queued = pb.read_json(root / "jobs.json")
        queued.append({"input": job["input"], "store": job["store"], "prio": job["prio"],
                       "scope": scope, "config": f"{job['input']}/config.yaml"})
        pb.atomic_json(root / "jobs.json", queued)
        added.append(job["store"])

    cycle.commit_dispatch(receipt_path, queue_add=add, duplicate_check=duplicate)
    cycle.commit_dispatch(receipt_path, queue_add=add, duplicate_check=duplicate)

    assert len(checked) == len(added) == 2
    assert len(pb.read_json(dataset / "jobs.json")) == 2
    assert pb.read_json(receipt_path)["status"] == "dispatched"


def test_commit_rejects_claimed_unqueued_store_without_copy_or_queue_mutation(tmp_path):
    dataset = _dataset(tmp_path / "dataset")
    train = _training(tmp_path / "training")
    staged = _bundle(tmp_path / "staged", 2)
    job = cycle._planned([staged], "g", 1, "b" * 64)[0]
    pb.atomic_json(dataset / "jobs_state" / f"{job['store']}.claim", {"machine": "other"})
    receipt_path = _dispatch_receipt(
        tmp_path / "local/cycles/test/action_receipt.json", dataset, train, [job], "b" * 64)

    with pytest.raises(ValueError, match="worker state"):
        cycle.commit_dispatch(receipt_path, queue_add=lambda *_: pytest.fail("must not queue"))
    assert not (dataset / job["input"]).exists()
    assert pb.read_json(dataset / "jobs.json") == []


def test_commit_rechecks_total_budget_before_copy(tmp_path, monkeypatch):
    dataset = _dataset(tmp_path / "dataset")
    train = _training(tmp_path / "training")
    staged = _bundle(tmp_path / "staged", 2)
    job = cycle._planned([staged], "g", 1, "c" * 64)[0]
    receipt_path = _dispatch_receipt(
        tmp_path / "local/cycles/test/action_receipt.json", dataset, train, [job], "c" * 64)
    almost_full = {f"h-{index}" for index in range(10_239)}
    monkeypatch.setattr(cycle, "_queue_view", lambda *args: {
        "jobs": [], "successful_hashes": almost_full, "pending_hashes": set(),
        "guided_pending_hashes": set(),
    })

    with pytest.raises(ValueError, match="target budget"):
        cycle.commit_dispatch(receipt_path, queue_add=lambda *_: pytest.fail("must not queue"))
    assert not (dataset / job["input"]).exists()
    assert pb.read_json(receipt_path)["status"] == "prepared"


def test_commit_rejects_changed_profile_binding_before_copy(tmp_path):
    dataset = _dataset(tmp_path / "dataset")
    train = _training(tmp_path / "training")
    staged = _bundle(tmp_path / "staged", 2)
    job = cycle._planned([staged], "g", 1, "d" * 64)[0]
    profile = tmp_path / "profile.yaml"
    shutil.copy2(PROFILE, profile)
    receipt_path = _dispatch_receipt(
        tmp_path / "local/cycles/test/action_receipt.json", dataset, train, [job], "d" * 64)
    receipt = pb.read_json(receipt_path)
    receipt["profile_config"] = str(profile.resolve())
    receipt["profile_sha256"] = pb.file_sha256(profile)
    pb.atomic_json(receipt_path, receipt)
    with profile.open("a", encoding="utf-8") as stream:
        stream.write("\n# changed\n")

    with pytest.raises(ValueError, match="profile config differs"):
        cycle.commit_dispatch(receipt_path, queue_add=lambda *_: pytest.fail("must not queue"))
    assert not (dataset / job["input"]).exists()


def test_commit_rechecks_budget_for_each_shard_under_dataset_lock(tmp_path, monkeypatch):
    dataset = _dataset(tmp_path / "dataset")
    train = _training(tmp_path / "training")
    staged = [_bundle(tmp_path / f"staged-{i}", 2, prefix=f"s{i}", start=10 * i)
              for i in range(2)]
    jobs = cycle._planned(staged, "g", 1, "e" * 64)
    receipt_path = _dispatch_receipt(
        tmp_path / "local/cycles/test/action_receipt.json", dataset, train, jobs, "e" * 64)
    calls = 0

    def queue_view(*_args):
        nonlocal calls
        calls += 1
        count = 10_238 if calls <= 2 else 10_240
        return {"jobs": [], "successful_hashes": {f"ok-{i}" for i in range(count)},
                "pending_hashes": set(), "guided_pending_hashes": set()}

    def add(root, job, scope):
        queued = pb.read_json(root / "jobs.json")
        queued.append({"input": job["input"], "store": job["store"], "prio": job["prio"],
                       "scope": scope, "config": f"{job['input']}/config.yaml"})
        pb.atomic_json(root / "jobs.json", queued)

    monkeypatch.setattr(cycle, "_queue_view", queue_view)
    with pytest.raises(ValueError, match="target budget"):
        cycle.commit_dispatch(receipt_path, queue_add=add, duplicate_check=lambda *_: 2)

    assert len(pb.read_json(dataset / "jobs.json")) == 1
    assert (dataset / jobs[0]["input"]).is_dir()
    assert not (dataset / jobs[1]["input"]).exists()
    assert pb.read_json(receipt_path)["status"] == "dispatching"
