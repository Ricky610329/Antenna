# -*- coding: utf-8 -*-
"""Generated-fixture tests for the bounded R81 historical-prior utility."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from script.exploration import file_sha256, pattern_sha256
from script.filter_prior import prepare_filter_prior


def _measurement():
    return {
        "name": "dual25_p01_db075_wide16_40_v1", "port": "dual",
        "labels": ["S11", "S21", "S22"],
        "geometry": {"geom": "p01", "pixel_count": 25, "diag_bridge_w": 0.075},
        "sweep": {"start_ghz": 16, "stop_ghz": 40, "step_ghz": 0.5, "type": "Fast"},
        "solver": {"setup_freq_ghz": 28, "open_region_freq_ghz": 16,
                   "max_delta_s": 0.02, "max_passes": 6, "min_passes": 5,
                   "min_converged": 5},
    }


def _score():
    return {
        "name": "filter_26_30_stop20_36_v1",
        "bands": [
            {"name": "s11_match", "label": "S11", "start_ghz": 26.5,
             "stop_ghz": 29.5, "relation": "le", "threshold": -10},
            {"name": "s22_match", "label": "S22", "start_ghz": 26.5,
             "stop_ghz": 29.5, "relation": "le", "threshold": -10},
            {"name": "s21_pass", "label": "S21", "start_ghz": 26,
             "stop_ghz": 30, "relation": "ge", "threshold": -3},
            {"name": "s21_low_stop", "label": "S21", "start_ghz": 16,
             "stop_ghz": 20, "relation": "le", "threshold": -20},
            {"name": "s21_high_stop", "label": "S21", "start_ghz": 36,
             "stop_ghz": 40, "relation": "le", "threshold": -20},
        ],
    }


def _pattern(index: int) -> torch.Tensor:
    rng = np.random.default_rng(index)
    pattern = rng.random((25, 25)) > 0.65
    pattern[:5, 10:15] = True
    pattern[20:25, 10:15] = True
    return torch.as_tensor(pattern, dtype=torch.float32)


def _response(offset: float = 0.0) -> torch.Tensor:
    # Covered margins are +2, +1, +1 dB at offset zero.
    return torch.stack((torch.full((17,), -12.0 - offset),
                        torch.full((17,), -2.0 + offset),
                        torch.full((17,), -11.0 - offset)))


def _write_fixture(root: Path, specs):
    samples = root / "samples"; samples.mkdir(parents=True)
    rows = []
    for sample_id, pattern, response, lineage, canonical in specs:
        path = samples / f"{sample_id}.pt"
        torch.save((pattern, response), path)
        rows.append({
            "id": sample_id, "sample_file": path.name,
            "sample_sha256": file_sha256(path),
            "pattern_sha256": pattern_sha256(pattern),
            "labels": ["S11", "S21", "S22"],
            "freqs_ghz": np.arange(24, 32.1, 0.5).tolist(),
            "geometry": {"geom": "p01", "pixel_count": 25, "diag_bridge_w": 0.075},
            "lineage_id": lineage, "canonical_group_id": canonical,
            "provenance": {"source_store": "generated-fixture", "source_id": sample_id},
        })
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(rows), encoding="utf-8")
    return manifest, samples, rows


def test_raw_3x17_yields_three_real_margins_and_two_masked_placeholders(tmp_path):
    manifest, samples, _rows = _write_fixture(
        tmp_path, [("a", _pattern(1), _response(), "line-a", "group-a")])
    prior = prepare_filter_prior(_measurement(), _score(), manifest, samples)

    assert prior.patterns.shape == (1, 25, 25)
    assert prior.raw_responses.shape == (1, 3, 17)
    assert prior.band_names == (
        "s11_match", "s22_match", "s21_pass", "s21_low_stop", "s21_high_stop")
    assert prior.margins[0].tolist() == pytest.approx([2.0, 1.0, 1.0, 0.0, 0.0])
    assert prior.margin_masks[0].tolist() == [True, True, True, False, False]
    assert prior.records[0]["role"] == "historical_prior_not_current_truth"
    assert prior.records[0]["full_current_worst_margin_available"] is False
    assert len(prior.records[0]["raw_response_sha256"]) == 64
    assert np.array_equal(prior.raw_responses[0], _response().numpy())


def test_tamper_and_wrong_historical_identity_are_rejected(tmp_path):
    manifest, samples, rows = _write_fixture(
        tmp_path, [("a", _pattern(2), _response(), "line-a", "group-a")])
    torch.save((_pattern(2), _response(0.5)), samples / "a.pt")
    with pytest.raises(ValueError, match="sample hash mismatch"):
        prepare_filter_prior(_measurement(), _score(), manifest, samples)

    torch.save((_pattern(2), _response()), samples / "a.pt")
    rows[0]["sample_sha256"] = file_sha256(samples / "a.pt")
    rows[0]["labels"] = ["S11", "S22", "S21"]
    manifest.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="labels must be exactly"):
        prepare_filter_prior(_measurement(), _score(), manifest, samples)

    rows[0]["labels"] = ["S11", "S21", "S22"]
    rows[0]["geometry"]["diag_bridge_w"] = 0.1
    manifest.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="not p01/25/0.075"):
        prepare_filter_prior(_measurement(), _score(), manifest, samples)


def test_wrong_frequency_axis_and_partially_covered_band_are_rejected(tmp_path):
    manifest, samples, rows = _write_fixture(
        tmp_path, [("a", _pattern(3), _response(), "line-a", "group-a")])
    rows[0]["freqs_ghz"][-1] = 31.5
    manifest.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="frequencies must be exactly"):
        prepare_filter_prior(_measurement(), _score(), manifest, samples)

    rows[0]["freqs_ghz"] = np.arange(24, 32.1, 0.5).tolist()
    manifest.write_text(json.dumps(rows), encoding="utf-8")
    score = copy.deepcopy(_score())
    score["bands"][2]["start_ghz"] = 23.5
    with pytest.raises(ValueError, match="partially overlaps"):
        prepare_filter_prior(_measurement(), score, manifest, samples)


def test_holdout_pattern_and_groups_are_excluded_then_physical_pattern_is_deduplicated(tmp_path):
    shared = _pattern(4)
    pattern_b, pattern_c, pattern_d = _pattern(5), _pattern(6), _pattern(7)
    manifest, samples, _rows = _write_fixture(tmp_path, [
        ("keep", shared, _response(), "train-line", "train-group"),
        ("duplicate", shared, _response(0.2), "other-line", "other-group"),
        ("line-hold", pattern_b, _response(), "held-line", "b-group"),
        ("line-alias", pattern_b, _response(0.1), "renamed-line", "renamed-group"),
        ("group-hold", pattern_c, _response(), "c-line", "held-group"),
        ("pattern-hold", pattern_d, _response(), "d-line", "d-group"),
    ])
    holdout = [
        {"lineage_id": "held-line"},
        {"canonical_group_id": "held-group"},
        {"pattern": pattern_d, "pattern_sha256": pattern_sha256(pattern_d)},
    ]
    prior = prepare_filter_prior(_measurement(), _score(), manifest, samples,
                                 current_holdout=holdout)

    assert [record["id"] for record in prior.records] == ["keep"]
    reasons = {row["id"]: row["reasons"] for row in prior.exclusions}
    assert reasons == {
        "duplicate": ["duplicate_physical_pattern"],
        "line-hold": ["current_holdout_lineage"],
        "line-alias": ["current_holdout_physical_alias"],
        "group-hold": ["current_holdout_canonical_group"],
        "pattern-hold": ["current_holdout_pattern"],
    }
    assert prior.audit["historical_rows_validated"] == 6
    assert prior.audit["records_kept"] == 1
    assert prior.audit["records_excluded"] == 5


def test_current_profile_must_remain_exact_r81_identity(tmp_path):
    manifest, samples, _rows = _write_fixture(
        tmp_path, [("a", _pattern(8), _response(), "line-a", "group-a")])
    measurement = _measurement()
    measurement["geometry"]["geom"] = "p00"
    with pytest.raises(ValueError, match="current geometry"):
        prepare_filter_prior(measurement, _score(), manifest, samples)


def test_holdout_lineage_and_canonical_names_share_one_exclusion_namespace(tmp_path):
    manifest, samples, _rows = _write_fixture(tmp_path, [
        ("line-matches-canonical", _pattern(9), _response(), "shared-a", "train-a"),
        ("canonical-matches-line", _pattern(10), _response(), "train-b", "shared-b"),
        ("keep", _pattern(11), _response(), "train-c", "train-c"),
    ])
    prior = prepare_filter_prior(
        _measurement(), _score(), manifest, samples,
        current_holdout=[{"canonical_group_id": "shared-a"}, {"lineage_id": "shared-b"}],
    )

    assert [record["id"] for record in prior.records] == ["keep"]
    reasons = {row["id"]: row["reasons"] for row in prior.exclusions}
    assert reasons == {
        "line-matches-canonical": ["current_holdout_lineage"],
        "canonical-matches-line": ["current_holdout_canonical_group"],
    }


@pytest.mark.parametrize("bad_sha", [123, [], "not-a-sha"])
def test_malformed_holdout_pattern_sha_has_controlled_validation_error(tmp_path, bad_sha):
    manifest, samples, _rows = _write_fixture(
        tmp_path, [("a", _pattern(12), _response(), "line-a", "group-a")])
    holdout = {"pattern_sha256": bad_sha}
    with pytest.raises(ValueError, match="invalid pattern_sha256"):
        prepare_filter_prior(_measurement(), _score(), manifest, samples,
                             current_holdout=[holdout])


def test_manifest_and_sample_are_parsed_from_the_bytes_that_were_hashed(tmp_path, monkeypatch):
    manifest, samples, _rows = _write_fixture(
        tmp_path, [("a", _pattern(13), _response(), "line-a", "group-a")])
    manifest_bytes = manifest.read_bytes()
    sample_bytes = (samples / "a.pt").read_bytes()
    real_read_bytes = Path.read_bytes
    reads = {manifest.resolve(): 0, (samples / "a.pt").resolve(): 0}

    def counted_read_bytes(path):
        resolved = path.resolve()
        if resolved in reads:
            reads[resolved] += 1
        return real_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", counted_read_bytes)
    prior = prepare_filter_prior(_measurement(), _score(), manifest, samples)

    assert reads == {manifest.resolve(): 1, (samples / "a.pt").resolve(): 1}
    assert prior.audit["manifest_sha256"] == __import__("hashlib").sha256(manifest_bytes).hexdigest()
    assert prior.records[0]["source_sample_sha256"] == __import__("hashlib").sha256(sample_bytes).hexdigest()
