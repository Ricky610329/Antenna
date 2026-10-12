import json

import pytest

from script import profiled_batch
from tests.test_scoped_worker_recovery import _write_input, _write_store


def _error(error="COM fixture failure", *, kind="hfss_simulation", attempts=3):
    return {"error": error, "error_kind": kind, "attempts": attempts}


def test_terminal_classifier_is_typed_or_known_legacy_and_requires_integer_attempts():
    assert profiled_batch.is_terminal_hfss_error(_error())
    assert profiled_batch.is_terminal_hfss_error({
        "error": "DISP_E_EXCEPTION 0x80070223", "attempts": 3,
    })
    assert not profiled_batch.is_terminal_hfss_error(_error(attempts=2))
    assert not profiled_batch.is_terminal_hfss_error(_error(attempts=True))
    assert not profiled_batch.is_terminal_hfss_error({
        "error": "DISP_E_EXCEPTION 0x80070223", "error_kind": "other", "attempts": 3,
    })
    assert not profiled_batch.is_terminal_hfss_error({"attempts": 3})


@pytest.mark.parametrize("error", [None, 7, "", " \t\r\n"])
def test_terminal_classifier_rejects_nontext_or_blank_error(error):
    assert not profiled_batch.is_terminal_hfss_error({
        "error": error, "error_kind": "hfss_simulation", "attempts": 3,
    })


@pytest.mark.parametrize("already_skipped", [False, True])
def test_hybrid_terminal_error_is_rejected_without_rewrite(tmp_path, already_skipped):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    input_root, rows, cfg = _write_input(dataset, "input", ["bad"])
    entry = _error()
    entry["sample_file"] = "successful-artifact.pt"
    entry["sample_sha256"] = "0" * 64
    entry["measurement_id"] = "forged-success-metadata"
    if already_skipped:
        entry.update(status="skipped", skip_reason="hfss_attempts_exhausted")
    store = _write_store(dataset, input_root, "store", rows, cfg, errors={"bad": entry})
    before = (store / "results.json").read_bytes()

    with pytest.raises(ValueError):
        profiled_batch.skip_exhausted_hfss_errors(store, cfg)
    assert (store / "results.json").read_bytes() == before
    with pytest.raises(ValueError):
        profiled_batch.verify_finished(store, cfg)


def test_mixed_finished_store_preserves_error_and_keeps_strict_truth(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    input_root, rows, cfg = _write_input(dataset, "input", ["good", "poison"])
    original = _error("RPC_E_SERVERFAULT exact bytes")
    store = _write_store(
        dataset, input_root, "store", rows, cfg, errors={"poison": dict(original)}
    )

    assert profiled_batch.skip_exhausted_hfss_errors(store, cfg) == {
        "success_ids": ["good"], "skipped_ids": ["poison"],
    }
    saved = json.loads((store / "results.json").read_text(encoding="utf-8"))
    assert {key: saved["poison"][key] for key in original} == original
    assert saved["poison"]["status"] == "skipped"
    assert saved["poison"]["skip_reason"] == "hfss_attempts_exhausted"
    expected = {"success_ids": ["good"], "skipped_ids": ["poison"]}
    assert profiled_batch.verify_finished(store, cfg) == expected
    assert profiled_batch.validate_finished_store(store) == expected
    with pytest.raises(ValueError):
        profiled_batch.verify_completed(store, cfg, require_complete=True)
    assert not (store.parent / "jobs_state" / "store.done").exists()
    assert not (store.parent / "jobs_state" / "store.claim").exists()


def test_skip_is_idempotent_and_does_not_rewrite_results(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    input_root, rows, cfg = _write_input(dataset, "input", ["poison"])
    store = _write_store(dataset, input_root, "store", rows, cfg,
                         errors={"poison": _error()})
    profiled_batch.skip_exhausted_hfss_errors(store, cfg)
    before = (store / "results.json").read_bytes()

    monkeypatch.setattr(
        profiled_batch, "atomic_json",
        lambda *_args, **_kwargs: pytest.fail("idempotent normalization rewrote results.json"),
    )
    assert profiled_batch.skip_exhausted_hfss_errors(store, cfg) == {
        "success_ids": [], "skipped_ids": ["poison"],
    }
    assert (store / "results.json").read_bytes() == before


@pytest.mark.parametrize("bad_entry", [
    _error(attempts=True),
    _error(attempts=0),
    {"error": "COM fixture failure", "error_kind": "hfss_simulation"},
    _error("   "),
    _error("profile_observation_invalid: missing rad",
           kind="profile_observation_invalid"),
    _error("DISP_E_EXCEPTION 0x80070223", kind="unknown"),
])
def test_invalid_raw_error_stops_normalizer_without_rewrite(tmp_path, bad_entry):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    input_root, rows, cfg = _write_input(dataset, "input", ["bad"])
    store = _write_store(dataset, input_root, "store", rows, cfg,
                         errors={"bad": bad_entry})
    before = (store / "results.json").read_bytes()

    with pytest.raises(ValueError):
        profiled_batch.skip_exhausted_hfss_errors(store, cfg)
    assert (store / "results.json").read_bytes() == before
    with pytest.raises(ValueError):
        profiled_batch.verify_finished(store, cfg)


@pytest.mark.parametrize("pending_entry", [
    _error(attempts=1),
    {"error": "RPC_E_SERVERFAULT fixture", "attempts": 2},
])
def test_known_hfss_attempts_one_or_two_remain_pending(tmp_path, pending_entry):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    input_root, rows, cfg = _write_input(dataset, "input", ["pending"])
    store = _write_store(dataset, input_root, "store", rows, cfg,
                         errors={"pending": pending_entry})
    before = (store / "results.json").read_bytes()

    assert profiled_batch.skip_exhausted_hfss_errors(store, cfg) == {
        "success_ids": [], "skipped_ids": [],
    }
    assert (store / "results.json").read_bytes() == before
    with pytest.raises(ValueError):
        profiled_batch.verify_finished(store, cfg)


@pytest.mark.parametrize("marker", [
    {"status": "skipped", "skip_reason": "wrong"},
    {"status": "skipped", "skip_reason": "hfss_attempts_exhausted", "attempts": 2},
    {"status": "ok", "skip_reason": "hfss_attempts_exhausted"},
])
def test_forged_or_contradictory_skip_marker_is_rejected(tmp_path, marker):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    input_root, rows, cfg = _write_input(dataset, "input", ["bad"])
    entry = _error()
    entry.update(marker)
    store = _write_store(dataset, input_root, "store", rows, cfg, errors={"bad": entry})

    with pytest.raises(ValueError):
        profiled_batch.skip_exhausted_hfss_errors(store, cfg)
    with pytest.raises(ValueError):
        profiled_batch.verify_finished(store, cfg)


def test_success_hash_tamper_and_unknown_result_id_fail_before_normalization(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    input_root, rows, cfg = _write_input(dataset, "input", ["good", "poison"])
    tampered = _write_store(dataset, input_root, "tampered", rows, cfg,
                            errors={"poison": _error()})
    results = json.loads((tampered / "results.json").read_text(encoding="utf-8"))
    sample = tampered / results["good"]["sample_file"]
    sample.write_bytes(sample.read_bytes() + b"tamper")
    with pytest.raises(ValueError):
        profiled_batch.skip_exhausted_hfss_errors(tampered, cfg)

    unknown = _write_store(dataset, input_root, "unknown", rows, cfg,
                           errors={"poison": _error()})
    results = json.loads((unknown / "results.json").read_text(encoding="utf-8"))
    results["not-in-manifest"] = _error()
    profiled_batch.atomic_json(unknown / "results.json", results)
    with pytest.raises(ValueError):
        profiled_batch.skip_exhausted_hfss_errors(unknown, cfg)


def test_pending_rows_are_allowed_by_normalizer_but_rejected_as_finished(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    input_root, rows, cfg = _write_input(dataset, "input", ["good", "pending"])
    store = _write_store(dataset, input_root, "store", rows, cfg)
    results = json.loads((store / "results.json").read_text(encoding="utf-8"))
    results.pop("pending")
    profiled_batch.atomic_json(store / "results.json", results)

    assert profiled_batch.skip_exhausted_hfss_errors(store, cfg) == {
        "success_ids": ["good"], "skipped_ids": [],
    }
    with pytest.raises(ValueError):
        profiled_batch.verify_finished(store, cfg)


def test_single_all_skipped_store_has_no_fake_success(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    input_root, rows, cfg = _write_input(dataset, "input", ["poison"])
    store = _write_store(dataset, input_root, "store", rows, cfg, errors={
        "poison": {"error": "watchdog_timeout: fixture", "attempts": 3},
    })

    profiled_batch.skip_exhausted_hfss_errors(store, cfg)
    assert profiled_batch.verify_finished(store, cfg) == {
        "success_ids": [], "skipped_ids": ["poison"],
    }
    assert not list(store.glob("*.pt"))
    with pytest.raises(ValueError):
        profiled_batch.verify_completed(store, cfg, require_complete=True)
