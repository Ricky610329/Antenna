# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from script.exploration import file_sha256, pattern_sha256
from script.filter_history_rescore import (OBSERVED_SCOPE_GHZ, rescore_completed_wide_store,
                                           rescore_history,
                                           wide_stores_from_worker_descriptor)


def _pattern(seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    value = rng.integers(0, 2, size=(25, 25), dtype=np.uint8)
    return torch.as_tensor(value, dtype=torch.float32)


def _response(freqs: np.ndarray, *, pass_db: float = -2.0,
              stop_db: float = -21.0, match_db: float = -11.0) -> torch.Tensor:
    value = np.full((3, len(freqs)), -5.0, dtype=np.float32)
    value[0, :] = match_db
    value[2, :] = match_db
    value[1, :] = -8.0
    value[1, (freqs >= 26.0) & (freqs <= 30.0)] = pass_db
    value[1, ((freqs >= 16.0) & (freqs <= 20.0)) |
             ((freqs >= 36.0) & (freqs <= 40.0))] = stop_db
    return torch.as_tensor(value)


def _fixture(root: Path, specs) -> tuple[Path, Path, list[dict]]:
    samples = root / "samples"
    samples.mkdir(parents=True)
    rows = []
    for index, (freqs, response) in enumerate(specs):
        pattern = _pattern(index + 1)
        sample = samples / f"sample-{index}.pt"
        torch.save((pattern, response), sample)
        rows.append({
            "id": f"sample-{index}", "sample_file": sample.name,
            "sample_sha256": file_sha256(sample),
            "pattern_sha256": pattern_sha256(pattern),
            "labels": ["S11", "S21", "S22"],
            "freqs_ghz": np.asarray(freqs).tolist(),
            "geometry": {"geom": "p01", "pixel_count": 25, "diag_bridge_w": 0.075},
            "lineage_id": f"line-{index}", "canonical_group_id": f"group-{index}",
            "provenance": {"source_store": "generated-fixture"},
        })
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(rows), encoding="utf-8")
    return manifest, samples, rows


def test_full_wide_s21_qualifies_and_match_is_separate(tmp_path):
    freqs = OBSERVED_SCOPE_GHZ.copy()
    manifest, samples, _rows = _fixture(
        tmp_path, [(freqs, _response(freqs, match_db=-8.0))])
    report = rescore_history(manifest, samples)
    record = report["records"][0]

    assert record["s21"]["qualification_covered"] is True
    assert record["s21"]["qualified"] is True
    assert record["s21"]["worst_margin_db"] == pytest.approx(1.0)
    assert record["s21"]["positive_margin_observed"] is True
    assert record["s21"]["desired_positive_and_repeat_confirmed"] is False
    assert record["s21"]["missing_coverage"] == []
    assert record["s21"]["bands"]["s21_pass"]["margin_db"] == pytest.approx(1.0)
    assert record["s21"]["bands"]["s21_low_stop"]["margin_db"] == pytest.approx(1.0)
    assert record["retained_match_reference"]["covered"] is True
    assert record["retained_match_reference"]["qualified"] is False
    assert report["summary"]["s21_qualified"] == 1
    assert report["summary"]["s21_and_retained_match_qualified"] == 0


def test_legacy_24_32_can_pass_passband_but_never_wide_standard(tmp_path):
    freqs = np.arange(24.0, 32.0 + 0.25, 0.5)
    manifest, samples, _rows = _fixture(tmp_path, [(freqs, _response(freqs))])
    report = rescore_history(manifest, samples)
    record = report["records"][0]

    assert record["s21"]["passband_only_passed"] is True
    assert record["s21"]["qualification_covered"] is False
    assert record["s21"]["qualified"] is False
    assert record["s21"]["worst_margin_db"] is None
    assert record["s21"]["missing_coverage"] == [
        "s21_low_stop", "s21_high_stop", "observed_scope_16_40"]
    assert record["retained_match_reference"]["qualified"] is True
    assert report["summary"]["full_observed_scope_16_40"] == 0
    assert report["summary"]["s21_low_stop_covered"] == 0
    assert report["summary"]["s21_high_stop_covered"] == 0
    assert report["summary"]["positive_passband_pending_stop_measurement"] == 1
    pending = report["positive_passband_pending_stop_measurement"][0]
    assert pending["id"] == "sample-0"
    assert pending["passband_margin_db"] == pytest.approx(1.0)


def test_missing_buffers_do_not_add_a_hard_requirement(tmp_path):
    freqs = np.concatenate((np.arange(16, 20.1, .5),
                            np.arange(26, 30.1, .5),
                            np.arange(36, 40.1, .5)))
    manifest, samples, _ = _fixture(tmp_path, [(freqs, _response(freqs))])
    record = rescore_history(manifest, samples)["records"][0]
    assert record["frequency_axis"]["observed_scope_16_40_complete"] is False
    assert record["s21"]["qualification_covered"] is True
    assert record["s21"]["qualified"] is True
    assert record["s21"]["worst_margin_db"] == pytest.approx(1.0)
    assert record["s21"]["missing_coverage"] == []


@pytest.mark.parametrize("freqs", [np.arange(16, 40.1, .25),
                                  np.arange(10, 50.1, .5)])
def test_denser_or_broader_observations_still_qualify(tmp_path, freqs):
    manifest, samples, _ = _fixture(tmp_path, [(freqs, _response(freqs))])
    record = rescore_history(manifest, samples)["records"][0]
    assert record["frequency_axis"]["observed_scope_16_40_complete"] is True
    assert record["s21"]["qualified"] is True
    assert record["s21"]["worst_margin_db"] == pytest.approx(1.0)


def test_covered_stop_failure_is_not_confused_with_missing_coverage(tmp_path):
    freqs = OBSERVED_SCOPE_GHZ.copy()
    manifest, samples, _rows = _fixture(
        tmp_path, [(freqs, _response(freqs, stop_db=-19.5))])
    report = rescore_history(manifest, samples)
    record = report["records"][0]

    assert record["s21"]["qualification_covered"] is True
    assert record["s21"]["qualified"] is False
    assert record["s21"]["worst_margin_db"] == pytest.approx(-0.5)
    assert record["s21"]["missing_coverage"] == []
    assert record["s21"]["bands"]["s21_low_stop"]["passed"] is False
    assert record["s21"]["bands"]["s21_low_stop"]["margin_db"] == pytest.approx(-0.5)


def test_exact_thresholds_and_endpoints_are_inclusive(tmp_path):
    freqs = OBSERVED_SCOPE_GHZ.copy()
    manifest, samples, _rows = _fixture(
        tmp_path, [(freqs, _response(freqs, pass_db=-3.0, stop_db=-20.0,
                                     match_db=-10.0))])
    record = rescore_history(manifest, samples)["records"][0]

    assert record["s21"]["qualified"] is True
    assert record["s21"]["positive_margin_observed"] is False
    assert all(item["margin_db"] == pytest.approx(0.0)
               for item in record["s21"]["bands"].values())
    assert record["retained_match_reference"]["qualified"] is True


def test_hash_tamper_is_rejected_before_scoring(tmp_path):
    freqs = OBSERVED_SCOPE_GHZ.copy()
    manifest, samples, rows = _fixture(tmp_path, [(freqs, _response(freqs))])
    torch.save((_pattern(1), _response(freqs, pass_db=-10.0)), samples / "sample-0.pt")

    with pytest.raises(ValueError, match="sample hash mismatch"):
        rescore_history(manifest, samples)

    rows[0]["sample_sha256"] = file_sha256(samples / "sample-0.pt")
    rows[0]["sample_file"] = "../outside.pt"
    manifest.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes sample root"):
        rescore_history(manifest, samples)


def test_duplicate_ids_and_response_axis_mismatch_are_rejected(tmp_path):
    freqs = OBSERVED_SCOPE_GHZ.copy()
    manifest, samples, rows = _fixture(
        tmp_path, [(freqs, _response(freqs)), (freqs, _response(freqs))])
    rows[1]["id"] = rows[0]["id"]
    manifest.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="ids must be unique"):
        rescore_history(manifest, samples)

    rows[1]["id"] = "sample-1"
    rows[1]["freqs_ghz"] = rows[1]["freqs_ghz"][:-1]
    manifest.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match=r"response must be finite with shape \(3, 48\)"):
        rescore_history(manifest, samples)


def test_completed_wide_store_uses_validator_and_keeps_engineering_role(tmp_path, monkeypatch):
    from script import profiled_batch as pb

    measurement = {
        "name": "dual25_p01_db075_wide16_40_v1", "port": "dual",
        "labels": ["S11", "S21", "S22"],
        "geometry": {"geom": "p01", "pixel_count": 25, "diag_bridge_w": 0.075},
        "sweep": {"start_ghz": 16, "stop_ghz": 40, "step_ghz": 0.5, "type": "Fast"},
        "solver": {"setup_freq_ghz": 28, "open_region_freq_ghz": 16,
                   "max_delta_s": 0.02, "max_passes": 6, "min_passes": 5,
                   "min_converged": 5},
    }
    score = {
        "name": "wide", "bands": [
            {"name": "s21_pass", "label": "S21", "start_ghz": 26,
             "stop_ghz": 30, "relation": "ge", "threshold": -3},
            {"name": "s21_low_stop", "label": "S21", "start_ghz": 16,
             "stop_ghz": 20, "relation": "le", "threshold": -20},
            {"name": "s21_high_stop", "label": "S21", "start_ghz": 36,
             "stop_ghz": 40, "relation": "le", "threshold": -20},
        ],
    }
    pattern = _pattern(90)
    response = _response(OBSERVED_SCOPE_GHZ)
    sample = tmp_path / "wide.pt"
    torch.save((pattern, response), sample)
    row = {"id": "wide", "pattern_sha256": pattern_sha256(pattern)}
    (tmp_path / "measurement.json").write_text(json.dumps(measurement), encoding="utf-8")
    (tmp_path / "score_spec.json").write_text(json.dumps(score), encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps([row]), encoding="utf-8")
    result = {
        "wide": {"status": "ok", "sample_file": sample.name,
                 "sample_sha256": file_sha256(sample),
                 "pattern_sha256": pattern_sha256(pattern),
                 "labels": ["S11", "S21", "S22"],
                 "freqs": OBSERVED_SCOPE_GHZ.tolist()}
    }
    calls = []
    monkeypatch.setattr(pb, "validate_store",
                        lambda root, require_complete: calls.append((root, require_complete)) or result)

    report = rescore_completed_wide_store(tmp_path, source_method="Fast")

    assert calls == [(tmp_path.resolve(), True)]
    assert report["observation_count"] == 1
    assert report["s21_positive_margin_observed"] == 1
    assert report["observations"][0]["source_method"] == "Fast"
    assert report["observations"][0]["s21"]["desired_positive_and_repeat_confirmed"] is False


def test_worker_descriptor_is_utf8_data_with_release_hash_binding(tmp_path):
    dataset = tmp_path / "資料集"
    release = tmp_path / "發行.json"
    release.write_text(json.dumps({
        "release_id": "engineering", "stage": "EngineeringOnly",
        "dataset_root": str(dataset),
    }, ensure_ascii=False), encoding="utf-8")
    descriptor = tmp_path / "descriptor.json"
    descriptor.write_text(json.dumps({
        "release": str(release),
        "release_sha256": hashlib.sha256(release.read_bytes()).hexdigest(),
    }, ensure_ascii=False), encoding="utf-8")

    stores, binding = wide_stores_from_worker_descriptor(descriptor)

    assert stores == [("Fast", dataset / "dedust_r81smoke"),
                      ("Discrete", dataset / "dedust_r81discrete")]
    assert binding["release_id"] == "engineering"
    release.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="release hash mismatch"):
        wide_stores_from_worker_descriptor(descriptor)
