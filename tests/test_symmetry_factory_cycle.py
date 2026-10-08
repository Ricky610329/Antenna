from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

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
RETRY_WORKERS = ["140.123.106.216", "140.123.106.218", "140.123.106.37"]


class ColdPredictor:
    model_role = "historical_cold_start_only"
    model_ids = [{"file": "fixture", "sha256": "0" * 64}]


class CurrentPredictor(ColdPredictor):
    model_role = "current_profile"
    binding = {"fixture": "current"}


def _fake_pilot_request(tmp_path, monkeypatch, *, pilot_id="a" * 64):
    request_path = tmp_path / "pilot_request.json"
    request_path.write_text('{"fixture": true}', encoding="utf-8")
    binding = {"request_path": str(request_path.resolve()),
               "request_sha256": "b" * 64, "pilot_id": pilot_id,
               "profile": {"path": str(PROFILE.resolve()),
                           "sha256": pb.file_sha256(PROFILE)},
               "protocol": {"path": "protocol", "sha256": "c" * 64},
               "addendum": {"path": "addendum", "sha256": "d" * 64},
               "training_protocol": {"path": "training", "sha256": "e" * 64}}
    request = SimpleNamespace(binding=binding, pilot_id=pilot_id, priority=1, count=16)

    def load(value, _profile):
        if value is None:
            return None, None
        return request, dict(binding)

    monkeypatch.setattr(cycle, "_load_pilot_request", load)
    return request_path, request, binding


def _fake_pilot_core(tmp_path, monkeypatch, request):
    calls = []
    ids = [f"pilot-{index:02d}" for index in range(16)]
    hashes = [f"pilot-hash-{index:02d}" for index in range(16)]
    role_counts = {cycle.shell_pilot.SHELL_ROLE: 15,
                   cycle.shell_pilot.CONTROL_ROLE: 1}
    selection = SimpleNamespace(rows=tuple(), hashes=tuple(hashes), role_counts=role_counts)
    annotated = SimpleNamespace(
        predictor_binding={"data_version": 1}, model_ids=({"sha256": "f" * 64},),
        surrogate_generation="data-v001", valid_observations_at_fit=100)

    def select(_request, exclusions):
        calls.append(("select", set(exclusions)))
        return selection

    def freeze(_pairs, _bindings, _request, out):
        calls.append(("freeze",))
        pb.atomic_json(out, {"fixture": "reference", "pilot_id": request.pilot_id,
                             "request_binding": request.binding})
        return Path(out)

    def annotate(_request, frozen, predictor):
        calls.append(("annotate", frozen, predictor.model_role))
        return annotated

    def write(_request, _annotated, out):
        calls.append(("write",))
        bundle = _bundle(Path(out), 16, prefix="pilot")
        rows = pb.read_json(bundle / "manifest.json")
        for index, row in enumerate(rows):
            row.update({"id": ids[index], "pattern_sha256": hashes[index],
                        "pilot_id": request.pilot_id, "kind": "pilot"})
        pb.atomic_json(bundle / "manifest.json", rows)
        pb.atomic_json(bundle / "pilot_audit.json",
                       {"fixture": True, "pilot_id": request.pilot_id})
        pb.atomic_json(bundle / "pilot_bundle_complete.json", {"fixture": True})
        return bundle

    def validate(_out, _request):
        return {"pilot_id": request.pilot_id, "ordered_ids": list(ids),
                "ordered_hashes": list(hashes), "role_counts": dict(role_counts),
                "predictor_binding": annotated.predictor_binding,
                "model_ids": list(annotated.model_ids),
                "surrogate_generation": annotated.surrogate_generation,
                "valid_observations_at_fit": annotated.valid_observations_at_fit}

    monkeypatch.setattr(cycle.shell_pilot, "select_rows", select)
    monkeypatch.setattr(cycle.shell_pilot, "freeze_reference", freeze)
    monkeypatch.setattr(cycle.shell_pilot, "annotate_selection", annotate)
    monkeypatch.setattr(cycle.shell_pilot, "write_pilot_bundle", write)
    monkeypatch.setattr(cycle.shell_pilot, "validate_bundle", validate)
    return calls, ids, hashes, role_counts


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


def _queued_single_error(dataset: Path, *, start: int = 900) -> tuple[Path, Path, dict]:
    input_dir = _bundle(dataset / "failed_input", 1, prefix="failed", start=start)
    store = _complete(input_dir, dataset / "failed")
    row = pb.read_json(input_dir / "manifest.json")[0]
    successful_results = pb.read_json(store / "results.json")
    pb.atomic_json(store / "results.json", {
        row["id"]: {"error": "DISP_E_EXCEPTION 0x80070223",
                    "error_kind": "hfss_simulation", "attempts": 3},
    })
    cfg = pb.load_profile_config(PROFILE)
    pb.validate_store(store, require_complete=False)
    pb.atomic_json(dataset / "jobs.json", [{
        "input": input_dir.name, "store": store.name, "prio": 1,
        "scope": cfg.scope, "config": f"{input_dir.name}/config.yaml",
    }])
    return input_dir, store, successful_results


def _add_background_truth(monkeypatch, count: int = 4_999) -> set[str]:
    background = {f"background-{index}" for index in range(count)}
    original = cycle._queue_view

    def queue_view(*args, **kwargs):
        result = original(*args, **kwargs)
        result["successful_hashes"].update(background)
        result["exclusion_hashes"].update(background)
        return result

    monkeypatch.setattr(cycle, "_queue_view", queue_view)
    monkeypatch.setattr(cycle, "_trained",
                        lambda root: (1, background, {"manifest_id": "bound",
                                                      "data_version": 1}))
    monkeypatch.setattr(cycle.training, "load_current_predictor",
                        lambda root, version: ColdPredictor())
    return background


def _dispatch_receipt(path: Path, dataset: Path, train: Path,
                      jobs: list[dict], cycle_id: str) -> Path:
    path.parent.mkdir(parents=True)
    pb.atomic_json(path, {
        "schema_version": 1, "status": "prepared", "cycle_id": cycle_id,
        "dataset_root": str(dataset), "scope": pb.load_profile_config(PROFILE).scope,
        "profile_config": str(PROFILE.resolve()), "profile_sha256": pb.file_sha256(PROFILE),
        "training_workdir": str(train.resolve()),
        "expected_retry_workers": RETRY_WORKERS,
        "training_protocol": cycle._protocol_binding(train)[1], "planned_jobs": jobs,
    })
    return path


def _pilot_dispatch_receipt(tmp_path, monkeypatch, dataset, train):
    request_path, request, binding = _fake_pilot_request(tmp_path, monkeypatch)
    staged = _bundle(tmp_path / "pilot-staged", 16, prefix="pilot")
    rows = pb.read_json(staged / "manifest.json")
    for row in rows:
        row["pilot_id"] = request.pilot_id
    pb.atomic_json(staged / "manifest.json", rows)
    job = cycle._planned([staged], "p", 1, "f" * 64)[0]
    store = f"dedust_r80p{request.pilot_id[:8]}p01"
    job.update({"input": store + "_input", "store": store,
                "pilot_id": request.pilot_id, "pilot_role": "incumbent_shell"})
    receipt_path = _dispatch_receipt(
        tmp_path / "local/cycles/test/action_receipt.json",
        dataset, train, [job], "f" * 64)
    reference = tmp_path / "pilot-reference.json"
    pb.atomic_json(reference, {"fixture": "reference"})
    proof = {"pilot_id": request.pilot_id,
             "ordered_ids": [row["id"] for row in rows],
             "ordered_hashes": [row["pattern_sha256"] for row in rows],
             "role_counts": {cycle.shell_pilot.SHELL_ROLE: 15,
                             cycle.shell_pilot.CONTROL_ROLE: 1},
             "predictor_binding": {"data_version": 1},
             "model_ids": [{"sha256": "f" * 64}],
             "surrogate_generation": "data-v001",
             "valid_observations_at_fit": 100}
    receipt = pb.read_json(receipt_path)
    receipt["pilot_request"] = binding
    receipt["pilot"] = {"state": "prepared", "request": binding,
                        "reference": cycle._path_binding(reference),
                        "bundle": {"path": str(staged.resolve()),
                                   "tree_sha256": cycle._content_id(
                                       cycle._tree_hashes(staged)),
                                   "proof": proof},
                        "selection": {key: proof[key] for key in (
                            "ordered_ids", "ordered_hashes", "role_counts")},
                        "predictor_binding": proof["predictor_binding"],
                        "model_ids": proof["model_ids"],
                        "surrogate_generation": proof["surrogate_generation"],
                        "valid_observations_at_fit": proof["valid_observations_at_fit"]}
    pb.atomic_json(receipt_path, receipt)
    monkeypatch.setattr(cycle.shell_pilot, "validate_bundle",
                        lambda path, supplied: dict(proof))
    return request_path, request, receipt_path, job, staged, reference, proof


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
    assert "pilot_request" not in receipt and "pilot" not in receipt
    assert [job["count"] for job in receipt["planned_jobs"]] == [16, 16, 16]
    assert {job["prio"] for job in receipt["planned_jobs"]} == {1}
    assert pb.read_json(dataset / "jobs.json") == []
    assert cycle.run_once(local_workdir=tmp_path / "cycle", dataset_root=dataset,
                          profile_config=PROFILE, training_workdir=train,
                          seed_inputs=[seeds], cold_start_predictor=ColdPredictor()) == receipt_path
    assert len(calls) == 1


def test_pending_pilot_uses_one_priority1_shard_and_suppresses_ordinary_planning(
        tmp_path, monkeypatch):
    dataset = _dataset(tmp_path / "dataset")
    train = _training(tmp_path / "training")
    seeds = _bundle(tmp_path / "seeds", 4)
    request_path, request, binding = _fake_pilot_request(tmp_path, monkeypatch)
    pilot_calls, ids, hashes, roles = _fake_pilot_core(tmp_path, monkeypatch, request)
    ordinary_calls = _fake_pool(monkeypatch)

    receipt = pb.read_json(cycle.run_once(
        local_workdir=tmp_path / "cycle", dataset_root=dataset, profile_config=PROFILE,
        training_workdir=train, seed_inputs=[seeds],
        cold_start_predictor=CurrentPredictor(), pilot_request=request_path,
        expected_retry_workers=RETRY_WORKERS))

    assert receipt["pilot_request"] == binding
    assert receipt["pilot"]["state"] == "prepared"
    assert receipt["pilot"]["selection"] == {
        "ordered_ids": ids, "ordered_hashes": hashes, "role_counts": roles}
    assert [(job["kind"], job["prio"], job["count"], job["pilot_id"])
            for job in receipt["planned_jobs"]] == [("p", 1, 16, request.pilot_id)]
    assert receipt["limits"]["valid_plus_pending_plus_planned"] == 16
    assert ordinary_calls == []
    assert [item[0] for item in pilot_calls] == ["select", "freeze", "annotate", "write"]


def test_completed_pilot_bundle_is_recovered_across_cycle_id_after_receipt_write_crash(
        tmp_path, monkeypatch):
    dataset = _dataset(tmp_path / "dataset")
    train = _training(tmp_path / "training")
    seeds1 = _bundle(tmp_path / "seeds-1", 3)
    seeds2 = _bundle(tmp_path / "seeds-2", 3, start=100)
    request_path, request, _binding = _fake_pilot_request(tmp_path, monkeypatch)
    calls, ids, hashes, _roles = _fake_pilot_core(tmp_path, monkeypatch, request)
    monkeypatch.setattr(cycle, "_trained",
                        lambda root: (1, set(), {"manifest_id": "bound", "data_version": 1}))
    monkeypatch.setattr(cycle.training, "load_current_predictor",
                        lambda root, version: CurrentPredictor())
    monkeypatch.setattr(cycle.shell_pilot, "recheck_reference", lambda path: None)
    original_write = cycle._write
    crashed = False

    def crash_after_complete_bundle(path, payload):
        nonlocal crashed
        if (not crashed and Path(path).name == "action_receipt.json" and
                payload.get("pilot", {}).get("state") == "prepared"):
            crashed = True
            raise RuntimeError("fixture crash after completed bundle")
        return original_write(path, payload)

    monkeypatch.setattr(cycle, "_write", crash_after_complete_bundle)
    with pytest.raises(RuntimeError, match="after completed bundle"):
        cycle.run_once(
            local_workdir=tmp_path / "cycle", dataset_root=dataset, profile_config=PROFILE,
            training_workdir=train, seed_inputs=[seeds1], pilot_request=request_path,
            expected_retry_workers=RETRY_WORKERS)
    first_bundle = next((tmp_path / "cycle/cycles").glob("*/pilot_bundle")).resolve()
    assert (first_bundle / "pilot_bundle_complete.json").is_file()

    monkeypatch.setattr(cycle.shell_pilot, "select_rows",
                        lambda *_: pytest.fail("restart must not reselect"))
    monkeypatch.setattr(cycle.shell_pilot, "annotate_selection",
                        lambda *_: pytest.fail("restart must not reannotate"))
    monkeypatch.setattr(cycle.shell_pilot, "write_pilot_bundle",
                        lambda *_: pytest.fail("restart must not republish"))
    receipt = pb.read_json(cycle.run_once(
        local_workdir=tmp_path / "cycle", dataset_root=dataset, profile_config=PROFILE,
        training_workdir=train, seed_inputs=[seeds2], pilot_request=request_path,
        expected_retry_workers=RETRY_WORKERS))

    assert receipt["pilot"]["completed_preparation_recovered"] is True
    assert Path(receipt["pilot"]["bundle"]["path"]) == first_bundle
    assert receipt["pilot"]["selection"]["ordered_ids"] == ids
    assert receipt["pilot"]["selection"]["ordered_hashes"] == hashes
    assert [item[0] for item in calls] == ["select", "freeze", "annotate", "write"]


def test_pilot_final_remaining_below16_is_durable_and_legacy_tail_is_not_starved(
        tmp_path, monkeypatch):
    dataset = _dataset(tmp_path / "dataset")
    train = _training(tmp_path / "training")
    seeds = _bundle(tmp_path / "seeds", 3)
    request_path, request, binding = _fake_pilot_request(tmp_path, monkeypatch)
    calls = _fake_pool(monkeypatch)
    successful = {f"ok-{index}" for index in range(4_999)}
    jobs_hash = pb.file_sha256(dataset / "jobs.json")
    monkeypatch.setattr(cycle.factory, "snapshot", lambda *args: {"jobs_sha256": jobs_hash})
    monkeypatch.setattr(cycle, "_queue_view", lambda *args: {
        "jobs_sha256": jobs_hash, "jobs": [], "pairs": [], "pair_bindings": [],
        "pending_hashes": set(), "guided_pending_hashes": set(),
        "successful_hashes": successful, "exclusion_hashes": successful,
    })
    monkeypatch.setattr(cycle, "_trained",
                        lambda root: (1, successful, {"manifest_id": "bound", "data_version": 1}))
    monkeypatch.setattr(cycle.training, "load_current_predictor",
                        lambda root, version: CurrentPredictor())
    monkeypatch.setattr(cycle.shell_pilot, "select_rows",
                        lambda *_: pytest.fail("permanent capacity shortage must not select"))

    receipt_path = cycle.run_once(
        local_workdir=tmp_path / "cycle", dataset_root=dataset, profile_config=PROFILE,
        training_workdir=train, seed_inputs=[seeds], pilot_request=request_path,
        expected_retry_workers=RETRY_WORKERS)
    receipt = pb.read_json(receipt_path)

    assert receipt["pilot_request"] == binding and "pilot" not in receipt
    assert calls[0][0] == 1
    assert [(job["kind"], job["count"], job["final_target_tail"])
            for job in receipt["planned_jobs"]] == [("g", 1, True)]
    decision = pb.read_json(
        tmp_path / f"cycle/pilot_decisions/{request.pilot_id}.json")
    assert decision["status"] == "terminal_unavailable"
    assert decision["code"] == "final_remaining_below_16"
    assert decision["decision_id"] == cycle._content_id({
        key: value for key, value in decision.items() if key != "decision_id"})

    def add(root, supplied, scope):
        pb.atomic_json(root / "jobs.json", [{
            "input": supplied["input"], "store": supplied["store"],
            "prio": supplied["prio"], "scope": scope,
            "config": f"{supplied['input']}/config.yaml"}])

    cycle.commit_dispatch(receipt_path, queue_add=add, duplicate_check=lambda *_: 1)
    assert pb.read_json(receipt_path)["status"] == "dispatched"
    assert len(pb.read_json(dataset / "jobs.json")) == 1

    decision["detail"] = "rewritten"
    pb.atomic_json(tmp_path / f"cycle/pilot_decisions/{request.pilot_id}.json", decision)
    with pytest.raises(ValueError, match="decision content changed"):
        cycle._existing_pilot_terminal_decision(tmp_path / "cycle", binding)


@pytest.mark.parametrize("code", ["empty_stratum", "no_eligible_blind"])
def test_permanent_pilot_selection_shortage_is_durable_and_ordinary_planning_resumes(
        tmp_path, monkeypatch, code):
    dataset = _dataset(tmp_path / "dataset")
    train = _training(tmp_path / "training")
    seeds = _bundle(tmp_path / "seeds", 3)
    request_path, request, _binding = _fake_pilot_request(tmp_path, monkeypatch)
    calls = _fake_pool(monkeypatch)
    monkeypatch.setattr(cycle.training, "load_current_predictor",
                        lambda root, version: CurrentPredictor())
    monkeypatch.setattr(cycle, "_trained",
                        lambda root: (1, set(), {"manifest_id": "bound", "data_version": 1}))

    def unavailable(*_args):
        raise cycle.shell_pilot.PilotUnavailable(code, f"fixture {code}")

    monkeypatch.setattr(cycle.shell_pilot, "select_rows", unavailable)
    receipt = pb.read_json(cycle.run_once(
        local_workdir=tmp_path / "cycle", dataset_root=dataset, profile_config=PROFILE,
        training_workdir=train, seed_inputs=[seeds], pilot_request=request_path,
        expected_retry_workers=RETRY_WORKERS))

    assert receipt["status"] == "prepared"
    assert {job["kind"] for job in receipt["planned_jobs"]} == {"g"}
    assert calls and calls[0][0] == 48
    decision = pb.read_json(
        tmp_path / f"cycle/pilot_decisions/{request.pilot_id}.json")
    assert decision["code"] == code and decision["status"] == "terminal_unavailable"


def test_pilot_waits_only_for_guided_capacity_but_historical_predictor_runs_legacy(
        tmp_path, monkeypatch):
    dataset = _dataset(tmp_path / "dataset")
    train = _training(tmp_path / "training")
    seeds = _bundle(tmp_path / "seeds", 3)
    request_path, _request, _binding = _fake_pilot_request(tmp_path, monkeypatch)
    jobs_hash = pb.file_sha256(dataset / "jobs.json")
    pending = {f"pending-{index}" for index in range(90)}
    monkeypatch.setattr(cycle.factory, "snapshot", lambda *args: {"jobs_sha256": jobs_hash})
    monkeypatch.setattr(cycle, "_queue_view", lambda *args: {
        "jobs_sha256": jobs_hash, "jobs": [], "pairs": [], "pair_bindings": [],
        "pending_hashes": pending, "guided_pending_hashes": pending,
        "successful_hashes": set(), "exclusion_hashes": pending,
    })
    monkeypatch.setattr(cycle, "_trained",
                        lambda root: (1, set(), {"manifest_id": "bound", "data_version": 1}))
    monkeypatch.setattr(cycle.training, "load_current_predictor",
                        lambda root, version: CurrentPredictor())

    waiting = pb.read_json(cycle.run_once(
        local_workdir=tmp_path / "waiting", dataset_root=dataset, profile_config=PROFILE,
        training_workdir=train, seed_inputs=[seeds], pilot_request=request_path,
        expected_retry_workers=RETRY_WORKERS))
    assert waiting["status"] == "idle" and waiting["planned_jobs"] == []
    assert waiting["pilot"]["state"] == "waiting_guided_capacity"
    assert waiting["pilot"]["guided_capacity"] == 6

    monkeypatch.setattr(cycle, "_queue_view", lambda *args: {
        "jobs_sha256": jobs_hash, "jobs": [], "pairs": [], "pair_bindings": [],
        "pending_hashes": set(), "guided_pending_hashes": set(),
        "successful_hashes": set(), "exclusion_hashes": set(),
    })
    monkeypatch.setattr(cycle, "_trained", lambda root: (0, set(), None))
    calls = _fake_pool(monkeypatch)
    legacy = pb.read_json(cycle.run_once(
        local_workdir=tmp_path / "legacy", dataset_root=dataset, profile_config=PROFILE,
        training_workdir=train, seed_inputs=[seeds], pilot_request=request_path,
        cold_start_predictor=ColdPredictor(), expected_retry_workers=RETRY_WORKERS))
    assert legacy["pilot"]["state"] == "waiting_current_profile_predictor"
    assert {job["kind"] for job in legacy["planned_jobs"]} == {"g"}
    assert calls and calls[0][0] == 48


def test_active_pre_roster_receipt_is_not_reused_for_bound_campaign(tmp_path, monkeypatch):
    dataset = _dataset(tmp_path / "dataset")
    train = _training(tmp_path / "training")
    seeds = _bundle(tmp_path / "seeds", 4)
    _fake_pool(monkeypatch)
    local = tmp_path / "cycle"
    receipt_path = cycle.run_once(
        local_workdir=local, dataset_root=dataset, profile_config=PROFILE,
        training_workdir=train, seed_inputs=[seeds], cold_start_predictor=ColdPredictor())
    receipt = pb.read_json(receipt_path)
    receipt.pop("expected_retry_workers")
    pb.atomic_json(receipt_path, receipt)

    with pytest.raises(ValueError, match="active cycle belongs to different explicit inputs"):
        cycle.run_once(local_workdir=local, dataset_root=dataset, profile_config=PROFILE,
                       training_workdir=train, seed_inputs=[seeds],
                       cold_start_predictor=ColdPredictor(),
                       expected_retry_workers=RETRY_WORKERS)


def test_active_pilot_receipt_requires_exact_present_request_binding(tmp_path, monkeypatch):
    local = tmp_path / "cycle"
    receipt_path = local / "cycles/prior/action_receipt.json"
    receipt_path.parent.mkdir(parents=True)
    dataset = (tmp_path / "dataset").resolve()
    train = (tmp_path / "training").resolve()
    request_path, request, binding = _fake_pilot_request(tmp_path, monkeypatch)
    pb.atomic_json(receipt_path, {
        "status": "prepared", "dataset_root": str(dataset),
        "profile_config": str(PROFILE.resolve()), "training_workdir": str(train),
        "expected_retry_workers": RETRY_WORKERS, "pilot_request": binding,
    })
    pb.atomic_json(local / "active_cycle.json",
                   {"cycle_id": "prior", "receipt": str(receipt_path.resolve())})

    assert cycle.run_once(
        local_workdir=local, dataset_root=dataset, profile_config=PROFILE,
        training_workdir=train, seed_inputs=[tmp_path / "seed"],
        expected_retry_workers=RETRY_WORKERS, pilot_request=request_path) == receipt_path.resolve()

    changed = dict(binding, request_sha256="0" * 64)
    monkeypatch.setattr(cycle, "_load_pilot_request",
                        lambda value, profile: (request, changed))
    with pytest.raises(ValueError, match="different pilot request"):
        cycle.run_once(local_workdir=local, dataset_root=dataset, profile_config=PROFILE,
                       training_workdir=train, seed_inputs=[tmp_path / "seed"],
                       expected_retry_workers=RETRY_WORKERS, pilot_request=request_path)


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


def test_final_5000_target_tail_reserves_only_remaining_eight(tmp_path, monkeypatch):
    dataset = _dataset(tmp_path / "dataset")
    train = _training(tmp_path / "training")
    seeds = _bundle(tmp_path / "seeds", 3)
    calls = _fake_pool(monkeypatch)
    successful = {f"ok-{index}" for index in range(4982)}
    pending = {f"pending-{index}" for index in range(10)}
    jobs_hash = pb.file_sha256(dataset / "jobs.json")
    monkeypatch.setattr(cycle.factory, "snapshot", lambda *args: {"jobs_sha256": jobs_hash})
    monkeypatch.setattr(cycle, "_queue_view", lambda *args: {
        "jobs_sha256": jobs_hash, "jobs": [], "pairs": [], "pair_bindings": [],
        "pending_hashes": pending, "guided_pending_hashes": pending,
        "successful_hashes": successful, "exclusion_hashes": successful | pending,
    })
    monkeypatch.setattr(cycle, "_trained",
                        lambda root: (1, successful, {"manifest_id": "bound", "data_version": 1}))
    monkeypatch.setattr(cycle.training, "load_current_predictor",
                        lambda root, version: ColdPredictor())

    receipt = pb.read_json(cycle.run_once(
        local_workdir=tmp_path / "cycle", dataset_root=dataset, profile_config=PROFILE,
        training_workdir=train, seed_inputs=[seeds]))

    assert calls[0][0] == 8
    assert [job["count"] for job in receipt["planned_jobs"]] == [8]
    assert receipt["planned_jobs"][0]["final_target_tail"] is True
    assert receipt["target_valid_unique"] == 5000
    assert receipt["limits"]["valid_plus_pending_plus_planned"] == 5000


def test_prepare_rejects_existing_valid_plus_pending_above_target(tmp_path, monkeypatch):
    dataset = _dataset(tmp_path / "dataset")
    train = _training(tmp_path / "training")
    seeds = _bundle(tmp_path / "seeds", 3)
    successful = {f"ok-{index}" for index in range(4_999)}
    pending = {"pending-a", "pending-b"}
    jobs_hash = pb.file_sha256(dataset / "jobs.json")
    monkeypatch.setattr(cycle.factory, "snapshot", lambda *args: {"jobs_sha256": jobs_hash})
    monkeypatch.setattr(cycle, "_queue_view", lambda *args: {
        "jobs_sha256": jobs_hash, "jobs": [], "pairs": [], "pair_bindings": [],
        "pending_hashes": pending, "guided_pending_hashes": pending,
        "successful_hashes": successful, "exclusion_hashes": successful | pending,
    })
    monkeypatch.setattr(cycle, "_trained",
                        lambda root: (1, successful, {"manifest_id": "bound",
                                                      "data_version": 1}))

    with pytest.raises(ValueError, match="valid plus pending"):
        cycle.run_once(local_workdir=tmp_path / "cycle", dataset_root=dataset,
                       profile_config=PROFILE, training_workdir=train, seed_inputs=[seeds],
                       expected_retry_workers=RETRY_WORKERS)


def test_retryable_fail_at_4999_stays_reserved_through_takeover_and_recovery(
        tmp_path, monkeypatch):
    dataset = _dataset(tmp_path / "dataset")
    train = _training(tmp_path / "training")
    seeds = _bundle(tmp_path / "seeds", 3)
    _input, store, successful_results = _queued_single_error(dataset)
    _add_background_truth(monkeypatch)
    fail = dataset / "jobs_state/failed.fail"
    claim = dataset / "jobs_state/failed.claim"
    pb.atomic_json(fail, {"machines": [RETRY_WORKERS[0]]})
    pb.atomic_json(claim, {"machine": RETRY_WORKERS[0]})

    first = pb.read_json(cycle.run_once(
        local_workdir=tmp_path / "cycle-first", dataset_root=dataset, profile_config=PROFILE,
        training_workdir=train, seed_inputs=[seeds], expected_retry_workers=RETRY_WORKERS))
    assert first["valid_unique"] == 4_999 and first["pending_unique"] == 1
    assert first["planned_jobs"] == [] and first["retryable_failed_jobs"] == ["failed"]

    fail.unlink(); claim.unlink()
    pb.atomic_json(claim, {"machine": RETRY_WORKERS[1],
                           "prior_fail": [RETRY_WORKERS[0]]})
    takeover = pb.read_json(cycle.run_once(
        local_workdir=tmp_path / "cycle-takeover", dataset_root=dataset,
        profile_config=PROFILE, training_workdir=train, seed_inputs=[seeds],
        expected_retry_workers=RETRY_WORKERS))
    assert takeover["valid_unique"] == 4_999 and takeover["pending_unique"] == 1
    assert takeover["planned_jobs"] == []

    pb.atomic_json(store / "results.json", successful_results)
    claim.unlink()
    pb.atomic_json(dataset / "jobs_state/failed.done", {"machine": RETRY_WORKERS[1],
                                                         "errors": 0})
    recovered = pb.read_json(cycle.run_once(
        local_workdir=tmp_path / "cycle-recovered", dataset_root=dataset,
        profile_config=PROFILE, training_workdir=train, seed_inputs=[seeds],
        expected_retry_workers=RETRY_WORKERS))
    assert recovered["valid_unique"] == 5_000 and recovered["pending_unique"] == 0
    assert recovered["planned_jobs"] == [] and recovered["status"] == "idle"


def test_all_retry_workers_exhausted_at_4999_releases_one_distinct_exact_tail(
        tmp_path, monkeypatch):
    dataset = _dataset(tmp_path / "dataset")
    train = _training(tmp_path / "training")
    seeds = _bundle(tmp_path / "seeds", 3)
    failed_input, _failed_store, _successful_results = _queued_single_error(dataset)
    _add_background_truth(monkeypatch)
    _fake_pool(monkeypatch)
    pb.atomic_json(dataset / "jobs_state/failed.fail", {"machines": RETRY_WORKERS})
    pb.atomic_json(dataset / "jobs_state/failed.claim", {"machine": RETRY_WORKERS[-1],
                                                          "prior_fail": RETRY_WORKERS[:-1]})

    receipt_path = cycle.run_once(
        local_workdir=tmp_path / "cycle-terminal", dataset_root=dataset,
        profile_config=PROFILE, training_workdir=train, seed_inputs=[seeds],
        expected_retry_workers=RETRY_WORKERS)
    receipt = pb.read_json(receipt_path)
    assert receipt["valid_unique"] == 4_999 and receipt["pending_unique"] == 0
    assert receipt["terminal_failed_jobs"] == ["failed"]
    assert [job["count"] for job in receipt["planned_jobs"]] == [1]
    assert receipt["limits"]["valid_plus_pending_plus_planned"] == 5_000

    def add(root, job, scope):
        queued = pb.read_json(root / "jobs.json")
        queued.append({"input": job["input"], "store": job["store"],
                       "prio": job["prio"], "scope": scope,
                       "config": f"{job['input']}/config.yaml"})
        pb.atomic_json(root / "jobs.json", queued)

    cycle.commit_dispatch(receipt_path, queue_add=add)
    tail_job = receipt["planned_jobs"][0]
    tail_input = dataset / tail_job["input"]
    failed_hash = pb.read_json(failed_input / "manifest.json")[0]["pattern_sha256"]
    tail_hash = pb.read_json(tail_input / "manifest.json")[0]["pattern_sha256"]
    assert tail_hash != failed_hash
    _complete(tail_input, dataset / tail_job["store"])
    pb.atomic_json(dataset / f"jobs_state/{tail_job['store']}.done",
                   {"machine": RETRY_WORKERS[0], "errors": 0})

    complete = pb.read_json(cycle.run_once(
        local_workdir=tmp_path / "cycle-complete", dataset_root=dataset,
        profile_config=PROFILE, training_workdir=train, seed_inputs=[seeds],
        expected_retry_workers=RETRY_WORKERS))
    assert complete["valid_unique"] == 5_000 and complete["pending_unique"] == 0
    assert complete["status"] == "idle" and complete["planned_jobs"] == []


def test_queue_view_empty_or_unknown_fail_roster_never_releases_pending(tmp_path):
    dataset = _dataset(tmp_path / "dataset")
    _queued_single_error(dataset)
    cfg = pb.load_profile_config(PROFILE)
    policy = cycle.factory.validate_factory_config(vars(cfg))
    fail = dataset / "jobs_state/failed.fail"

    for roster, machines in (([], RETRY_WORKERS),
                             (RETRY_WORKERS, RETRY_WORKERS + ["unknown-worker"]),
                             (RETRY_WORKERS, "malformed")):
        pb.atomic_json(fail, {"machines": machines})
        queue = cycle._queue_view(dataset, cfg, policy, roster)
        assert len(queue["pending_hashes"] - queue["successful_hashes"]) == 1
        assert queue["terminal_failed_jobs"] == []
        assert queue["retryable_failed_jobs"] == ["failed"]


def test_queue_view_defers_when_fail_takeover_changes_markers_mid_scan(tmp_path, monkeypatch):
    dataset = _dataset(tmp_path / "dataset")
    _queued_single_error(dataset)
    fail = dataset / "jobs_state/failed.fail"
    claim = dataset / "jobs_state/failed.claim"
    pb.atomic_json(fail, {"machines": [RETRY_WORKERS[0]]})
    pb.atomic_json(claim, {"machine": RETRY_WORKERS[0]})
    original = cycle._job_state_binding
    calls = 0

    def racing_binding(state_root, store):
        nonlocal calls
        result = original(state_root, store)
        calls += 1
        if calls == 1:
            fail.unlink(); claim.unlink()
            pb.atomic_json(claim, {"machine": RETRY_WORKERS[1],
                                   "prior_fail": [RETRY_WORKERS[0]]})
        return result

    monkeypatch.setattr(cycle, "_job_state_binding", racing_binding)
    cfg = pb.load_profile_config(PROFILE)
    policy = cycle.factory.validate_factory_config(vars(cfg))
    with pytest.raises(cycle.LiveSnapshotChanged, match="worker state changed"):
        cycle._queue_view(dataset, cfg, policy, RETRY_WORKERS)


def test_complete_queued_pilot_consumes_one_shot_across_worker_states(tmp_path, monkeypatch):
    dataset = _dataset(tmp_path / "dataset")
    request_path, request, _binding = _fake_pilot_request(tmp_path, monkeypatch)
    store_name = f"dedust_r80p{request.pilot_id[:8]}p01"
    input_name = store_name + "_input"
    input_dir = _bundle(dataset / input_name, 16, prefix="pilot")
    assert request_path.is_file()
    rows = pb.read_json(input_dir / "manifest.json")
    for row in rows:
        row["pilot_id"] = request.pilot_id
    pb.atomic_json(input_dir / "manifest.json", rows)
    cfg = pb.load_profile_config(PROFILE)
    pb.atomic_json(dataset / "jobs.json", [{
        "input": input_name, "store": store_name, "prio": 1, "scope": cfg.scope,
        "config": f"{input_name}/config.yaml"}])
    proof = {"pilot_id": request.pilot_id,
             "ordered_ids": [row["id"] for row in rows],
             "ordered_hashes": [row["pattern_sha256"] for row in rows],
             "role_counts": {"incumbent_shell": 15, "pilot_blind_control": 1}}
    monkeypatch.setattr(cycle.shell_pilot, "validate_bundle",
                        lambda path, supplied: dict(proof))
    policy = cycle.factory.validate_factory_config(vars(cfg))
    state = dataset / "jobs_state"

    for marker, payload in (
            (None, None),
            ("done", {"machine": RETRY_WORKERS[0], "errors": 0}),
            ("fail", {"machines": [RETRY_WORKERS[0]]}),
            ("fail", {"machines": RETRY_WORKERS})):
        for suffix in ("claim", "done", "fail"):
            path = state / f"{store_name}.{suffix}"
            if path.exists():
                path.unlink()
        if marker is not None:
            pb.atomic_json(state / f"{store_name}.{marker}", payload)
        queue = cycle._queue_view(dataset, cfg, policy, RETRY_WORKERS)
        evidence = cycle._queued_pilot_evidence(dataset, queue, request, cfg.scope)
        assert evidence is not None and evidence["proof"] == proof

    shutil.copytree(input_dir, dataset / "wrong_input")
    canonical = dict(queue["jobs"][0])
    for field, value in (
            ("input", "wrong_input"), ("store", "wrong_store"), ("prio", 2),
            ("scope", "wrong_scope"), ("config", "wrong/config.yaml")):
        changed_queue = dict(queue, jobs=[dict(canonical, **{field: value})])
        with pytest.raises(ValueError, match="noncanonical queue record"):
            cycle._queued_pilot_evidence(dataset, changed_queue, request, cfg.scope)


def test_consumed_pilot_request_prepares_and_commits_ordinary_wave(tmp_path, monkeypatch):
    dataset = _dataset(tmp_path / "dataset")
    request_path, request, binding = _fake_pilot_request(tmp_path, monkeypatch)
    store_name = f"dedust_r80p{request.pilot_id[:8]}p01"
    input_name = store_name + "_input"
    pilot_input = _bundle(dataset / input_name, 16, prefix="pilot", start=1_000)
    rows = pb.read_json(pilot_input / "manifest.json")
    for row in rows:
        row["pilot_id"] = request.pilot_id
    pb.atomic_json(pilot_input / "manifest.json", rows)
    cfg = pb.load_profile_config(PROFILE)
    pb.atomic_json(dataset / "jobs.json", [{
        "input": input_name, "store": store_name, "prio": 1, "scope": cfg.scope,
        "config": f"{input_name}/config.yaml"}])
    proof = {"pilot_id": request.pilot_id,
             "ordered_ids": [row["id"] for row in rows],
             "ordered_hashes": [row["pattern_sha256"] for row in rows],
             "role_counts": {cycle.shell_pilot.SHELL_ROLE: 15,
                             cycle.shell_pilot.CONTROL_ROLE: 1}}
    monkeypatch.setattr(cycle.shell_pilot, "validate_bundle",
                        lambda path, supplied: dict(proof))
    monkeypatch.setattr(cycle.shell_pilot, "select_rows",
                        lambda *_: pytest.fail("consumed pilot must not regenerate"))
    train = _training(tmp_path / "training")
    seeds = _bundle(tmp_path / "seeds", 3)
    calls = _fake_pool(monkeypatch)
    monkeypatch.setattr(cycle, "_trained",
                        lambda root: (1, set(), {"manifest_id": "bound", "data_version": 1}))
    monkeypatch.setattr(cycle.training, "load_current_predictor",
                        lambda root, version: CurrentPredictor())

    receipt_path = cycle.run_once(
        local_workdir=tmp_path / "cycle", dataset_root=dataset, profile_config=PROFILE,
        training_workdir=train, seed_inputs=[seeds],
        pilot_request=request_path, expected_retry_workers=RETRY_WORKERS)
    receipt = pb.read_json(receipt_path)
    assert receipt["pilot_request"] == binding and "pilot" not in receipt
    assert receipt["status"] == "prepared" and calls
    assert receipt["planned_jobs"] and {
        job["kind"] for job in receipt["planned_jobs"]} == {"g"}

    added = []

    def add(root, supplied, scope):
        queued = pb.read_json(root / "jobs.json")
        queued.append({"input": supplied["input"], "store": supplied["store"],
                       "prio": supplied["prio"], "scope": scope,
                       "config": f"{supplied['input']}/config.yaml"})
        pb.atomic_json(root / "jobs.json", queued)
        added.append(supplied["store"])

    cycle.commit_dispatch(
        receipt_path, queue_add=add,
        duplicate_check=lambda input_dir, root: len(pb.read_json(input_dir / "manifest.json")))
    assert pb.read_json(receipt_path)["status"] == "dispatched"
    assert len(added) == len(receipt["planned_jobs"])


def test_partial_queued_pilot_evidence_blocks_regeneration(tmp_path, monkeypatch):
    dataset = _dataset(tmp_path / "dataset")
    _request_path, request, _binding = _fake_pilot_request(tmp_path, monkeypatch)
    store_name = f"dedust_r80p{request.pilot_id[:8]}p01"
    input_name = store_name + "_input"
    input_dir = _bundle(dataset / input_name, 2, prefix="pilot")
    rows = pb.read_json(input_dir / "manifest.json")
    rows[0]["pilot_id"] = request.pilot_id
    pb.atomic_json(input_dir / "manifest.json", rows)
    cfg = pb.load_profile_config(PROFILE)
    pb.atomic_json(dataset / "jobs.json", [{
        "input": input_name, "store": store_name, "prio": 1, "scope": cfg.scope,
        "config": f"{input_name}/config.yaml"}])
    monkeypatch.setattr(cycle.shell_pilot, "validate_bundle",
                        lambda *_: (_ for _ in ()).throw(
                            cycle.shell_pilot.PilotError("partial fixture")))
    queue = cycle._queue_view(
        dataset, cfg, cycle.factory.validate_factory_config(vars(cfg)), RETRY_WORKERS)
    with pytest.raises(ValueError, match="partial or conflicting"):
        cycle._queued_pilot_evidence(dataset, queue, request, cfg.scope)


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


@pytest.mark.parametrize("recovery_point", ["post_copy", "post_queue"])
def test_pilot_commit_recovers_exact_staged_cohort_after_copy_or_queue(
        tmp_path, monkeypatch, recovery_point):
    dataset = _dataset(tmp_path / "dataset")
    train = _training(tmp_path / "training")
    (_request_path, _request, receipt_path, job, staged,
     _reference, _proof) = _pilot_dispatch_receipt(tmp_path, monkeypatch, dataset, train)
    destination = dataset / job["input"]
    shutil.copytree(staged, destination)
    expected = {"input": job["input"], "store": job["store"], "prio": job["prio"],
                "scope": pb.load_profile_config(PROFILE).scope,
                "config": f"{job['input']}/config.yaml"}
    if recovery_point == "post_queue":
        pb.atomic_json(dataset / "jobs.json", [expected])
    added, rechecked = [], []
    monkeypatch.setattr(cycle.shell_pilot, "recheck_reference",
                        lambda path: rechecked.append(Path(path)))

    def add(root, supplied, scope):
        queued = pb.read_json(root / "jobs.json")
        queued.append({"input": supplied["input"], "store": supplied["store"],
                       "prio": supplied["prio"], "scope": scope,
                       "config": f"{supplied['input']}/config.yaml"})
        pb.atomic_json(root / "jobs.json", queued)
        added.append(supplied["store"])

    cycle.commit_dispatch(receipt_path, queue_add=add, duplicate_check=lambda *_: 16)

    assert pb.read_json(dataset / "jobs.json") == [expected]
    assert added == ([] if recovery_point == "post_queue" else [job["store"]])
    assert rechecked and pb.read_json(receipt_path)["status"] == "dispatched"


def test_pilot_commit_rechecks_request_and_frozen_reference_before_queue_mutation(
        tmp_path, monkeypatch):
    dataset = _dataset(tmp_path / "dataset")
    train = _training(tmp_path / "training")
    (request_path, request, receipt_path, job, _staged,
     _reference, _proof) = _pilot_dispatch_receipt(tmp_path, monkeypatch, dataset, train)
    monkeypatch.setattr(cycle.shell_pilot, "recheck_reference",
                        lambda path: (_ for _ in ()).throw(
                            cycle.shell_pilot.PilotError("frozen old entry changed")))
    with pytest.raises(cycle.shell_pilot.PilotError, match="old entry"):
        cycle.commit_dispatch(
            receipt_path, queue_add=lambda *_: pytest.fail("must not queue"),
            duplicate_check=lambda *_: pytest.fail("must not check duplicates"))
    assert not (dataset / job["input"]).exists()

    changed = dict(request.binding, request_sha256="0" * 64)
    monkeypatch.setattr(cycle, "_load_pilot_request",
                        lambda value, profile: (request, changed))
    monkeypatch.setattr(cycle.shell_pilot, "recheck_reference",
                        lambda path: pytest.fail("request mismatch must fail first"))
    with pytest.raises(ValueError, match="request presence or exact binding changed"):
        cycle.commit_dispatch(receipt_path, queue_add=lambda *_: pytest.fail("must not queue"))
    assert request_path.is_file() and pb.read_json(dataset / "jobs.json") == []


def test_commit_rechecks_total_budget_before_copy(tmp_path, monkeypatch):
    dataset = _dataset(tmp_path / "dataset")
    train = _training(tmp_path / "training")
    staged = _bundle(tmp_path / "staged", 2)
    job = cycle._planned([staged], "g", 1, "c" * 64)[0]
    receipt_path = _dispatch_receipt(
        tmp_path / "local/cycles/test/action_receipt.json", dataset, train, [job], "c" * 64)
    almost_full = {f"h-{index}" for index in range(4_999)}
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
        count = 4_998 if calls <= 2 else 5_000
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
