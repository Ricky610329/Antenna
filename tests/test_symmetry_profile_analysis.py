from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from antenna.measurement import measurement_id, score_spec_id
from script import profiled_batch as pb
from script.symmetry_analysis import analyze_profile_stores


def _measurement(name="profile-analysis-test"):
    return {
        "name": name,
        "port": "single",
        "labels": ["S11", "Gain"],
        "geometry": {"geom": "single_v1", "pixel_count": 25, "diag_bridge_w": 0.1},
        "sweep": {"start_ghz": 24.0, "stop_ghz": 32.0, "step_ghz": 0.5,
                  "type": "Interpolating"},
        "solver": {"setup_freq_ghz": 28.0, "open_region_freq_ghz": 28.0,
                   "max_delta_s": 0.02, "max_passes": 6, "min_passes": 5,
                   "min_converged": 5},
    }


def _pattern(index):
    pattern = torch.zeros((25, 25), dtype=torch.float32)
    pattern[24, 12] = 1
    for bit in range(4):
        if index & (1 << bit):
            row, col = 2 + bit * 4, 1 + bit
            pattern[row, col] = pattern[row, 24 - col] = 1
    return pattern


def _content_id(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _write_store(root: Path, specs, *, measurement=None, bindings_schema=1):
    root.mkdir(parents=True)
    (root / "rad").mkdir()
    measurement = deepcopy(measurement or _measurement())
    score = {"name": "observation-only", "bands": []}
    cfg = SimpleNamespace(port="single", measurement=measurement, score_spec=score)
    mid, sid = measurement_id(measurement), score_spec_id(score)
    rows, results = [], {}
    theta = torch.arange(-180, 181, 2, dtype=torch.float32)
    for ordinal, (name, pattern_index, kind, arm) in enumerate(specs):
        pattern = _pattern(pattern_index)
        response = torch.stack((
            torch.linspace(-16.0, -8.0, 17) + ordinal,
            torch.linspace(3.0, 6.2, 17) - ordinal * 0.25,
        ))
        rad = {
            "theta": theta,
            "phi0": -torch.abs(theta) / 18.0 + ordinal * 0.1,
            "phi90": -torch.abs(theta) / 24.0 - ordinal * 0.1,
        }
        row = {
            "id": name,
            "port": "single",
            "measurement_id": mid,
            "score_spec_id": sid,
            "pattern_sha256": pb.pattern_sha256(pattern),
            "lineage_id": f"family-{pattern_index}",
            "kind": kind,
            "selection_arm": arm,
        }
        entry = pb.observation(response, rad, row, pattern, cfg, elapsed=1.0 + ordinal)
        sample_path, rad_path = root / entry["sample_file"], root / entry["rad_file"]
        torch.save((pattern, response), sample_path)
        torch.save(rad, rad_path)
        entry["sample_sha256"] = pb.file_sha256(sample_path)
        entry["rad_sha256"] = pb.file_sha256(rad_path)
        rows.append(row)
        results[name] = entry
    for filename, value in (
            ("measurement.json", measurement), ("score_spec.json", score),
            ("manifest.json", rows), ("results.json", results)):
        pb.atomic_json(root / filename, value)
    if bindings_schema == 1:
        source_bindings = {
            "schema_version": 1,
            "partial_snapshot": True,
            "source_batches_may_be_incomplete": True,
            "measurement_id": mid,
            "score_spec_id": sid,
            "counts": {"unique_selected_rows": len(rows)},
            "items": {
                name: {
                    "source_store": f"fixture-source-{name}",
                    "source_sample_sha256": entry["sample_sha256"],
                    "source_rad_sha256": entry["rad_sha256"],
                }
                for name, entry in results.items()
            },
        }
    elif bindings_schema == 2:
        source = {
            "input": str(root / "producer-input"),
            "store": str(root / "producer-store"),
            "input_metadata_sha256": {
                "manifest.json": pb.file_sha256(root / "manifest.json"),
                "measurement.json": pb.file_sha256(root / "measurement.json"),
                "score_spec.json": pb.file_sha256(root / "score_spec.json"),
            },
            "store_metadata_sha256": {
                "manifest.json": pb.file_sha256(root / "manifest.json"),
                "results.json": pb.file_sha256(root / "results.json"),
            },
            "partial_source_results_sha256": pb.file_sha256(root / "results.json"),
            "cutoff_manifest_count": len(rows),
            "cutoff_successful_count": len(rows),
        }
        source_bindings = {
            "schema_version": 2,
            "partial_snapshot": True,
            "source_batches_may_be_incomplete": True,
            "cutoff_kind": "same_cycle_ephemeral_physical_proof",
            "cutoff_id": _content_id([source]),
            "measurement_id": mid,
            "score_spec_id": sid,
            "selected_pattern_sha256": [row["pattern_sha256"] for row in rows],
            "sources": [source],
            "items": {
                name: {
                    "source_input": source["input"],
                    "source_store": source["store"],
                    "source_manifest_sha256": source["store_metadata_sha256"]["manifest.json"],
                    "partial_source_results_sha256": source["store_metadata_sha256"]["results.json"],
                    "cutoff_result_entry": entry,
                    "cutoff_result_entry_id": _content_id(entry),
                    "source_sample_file": entry["sample_file"],
                    "source_sample_sha256": entry["sample_sha256"],
                    "source_rad_file": entry["rad_file"],
                    "source_rad_sha256": entry["rad_sha256"],
                }
                for name, entry in results.items()
            },
        }
    else:
        raise AssertionError("fixture supports only source_bindings schema 1 or 2")
    pb.atomic_json(root / "source_bindings.json", source_bindings)
    assert len(pb.validate_store(root, require_complete=True)) == len(specs)
    return root


def test_profile_analysis_prefers_nonrepeat_and_binds_complete_arrays(tmp_path):
    repeat_store = _write_store(tmp_path / "repeat", [
        ("repeat-copy", 1, "repeat", "notarize"),
        ("only-here", 2, "factory", "random"),
    ])
    fresh_store = _write_store(tmp_path / "fresh", [
        ("fresh-physical", 1, "factory", "geometry"),
    ])
    out_json, out_npz = tmp_path / "analysis.json", tmp_path / "analysis.npz"

    returned = analyze_profile_stores([repeat_store, fresh_store], out_json, out_npz)
    saved = json.loads(out_json.read_text(encoding="utf-8"))

    assert returned["counts"]["validated_complete_source_rows"] == 3
    assert saved["counts"] == {
        "source_stores": 2,
        "validated_complete_source_rows": 3,
        "valid_response_rows": 3,
        "valid_radiation_rows": 3,
        "unique_selected_rows": 2,
        "duplicate_rows_removed": 1,
        "repeat_or_notarize_candidates": 1,
        "repeat_or_notarize_selected": 0,
        "symmetry_class": {"exact": 2},
    }
    assert [row["id"] for row in saved["rows"]] == ["only-here", "fresh-physical"]
    assert saved["duplicate_rows"][0]["id"] == "repeat-copy"
    assert saved["duplicate_rows"][0]["preferred_id"] == "fresh-physical"
    assert saved["duplicate_rows"][0]["raw_result"]["status"] == "ok"
    assert saved["arms"]["selected_counts"] == {"geometry": 1, "random": 1}
    assert saved["scope"]["performance_filter_applied"] is False
    assert saved["scope"]["causal_claim"] is False
    assert saved["counts"]["symmetry_class"] == {"exact": 2}
    assert saved["rows"][0]["metal_fraction"] == pytest.approx(3 / 625)
    assert saved["rows"][1]["geometry_mismatch_fraction"] == 0

    with np.load(out_npz, allow_pickle=False) as arrays:
        assert arrays["ids"].tolist() == ["only-here", "fresh-physical"]
        assert arrays["patterns"].shape == (2, 25, 25)
        assert arrays["responses"].shape == (2, 2, 17)
        assert arrays["s11_db"].shape == arrays["gain_db"].shape == (2, 17)
        assert arrays["radiation"].shape == (2, 2, 181)
        assert arrays["phi0_db"].shape == arrays["phi90_db"].shape == (2, 181)
        assert arrays["response_freqs_ghz"].tolist() == pytest.approx(
            np.arange(24.0, 32.1, 0.5).tolist())
        assert arrays["radiation_theta_deg"].tolist() == list(range(-180, 181, 2))
        fresh_index = arrays["ids"].tolist().index("fresh-physical")
        # The non-repeat observation wins even though its store was supplied later.
        assert arrays["s11_db"][fresh_index, 0] == pytest.approx(-16.0)
        assert arrays["s11_band_margin_db"].shape == (2,)
        assert arrays["phi0_mirror_power_45"].shape == (2,)
    assert saved["artifacts"]["npz"]["sha256"] == pb.file_sha256(out_npz)
    assert saved["quantiles"]["metal_fraction"]["finite_count"] == 2
    assert [source["source_bindings_provenance"]["schema_version"]
            for source in saved["sources"]] == [1, 1]


def test_profile_analysis_accepts_producer_shaped_v2_and_preserves_cutoff(tmp_path):
    store = _write_store(
        tmp_path / "v2", [("first", 6, "factory", "geometry"),
                          ("second", 7, "factory", "random")],
        bindings_schema=2,
    )
    source_files = [path for path in store.rglob("*") if path.is_file()]
    hashes_before = {path: pb.file_sha256(path) for path in source_files}

    summary = analyze_profile_stores(
        [store], tmp_path / "analysis.json", tmp_path / "analysis.npz")

    provenance = summary["sources"][0]["source_bindings_provenance"]
    bindings = pb.read_json(store / "source_bindings.json")
    assert provenance["schema_version"] == 2
    assert provenance["cutoff_kind"] == "same_cycle_ephemeral_physical_proof"
    assert provenance["cutoff_id"] == bindings["cutoff_id"]
    assert provenance["selected_pattern_sha256"] == bindings["selected_pattern_sha256"]
    assert provenance["sources"] == bindings["sources"]
    assert provenance["cutoff_id_reconstruction"] == "unavailable_original_pair_proof_absent"
    assert any("original pair-proof is absent" in item for item in summary["limitations"])
    assert {path: pb.file_sha256(path) for path in source_files} == hashes_before


@pytest.mark.parametrize("tamper", ["entry", "content_id", "selection", "source"])
def test_profile_analysis_rejects_mutated_v2_binding(tmp_path, tamper):
    store = _write_store(
        tmp_path / tamper, [("first", 8, "factory", "geometry"),
                            ("second", 9, "factory", "random")],
        bindings_schema=2,
    )
    bindings = pb.read_json(store / "source_bindings.json")
    first = bindings["items"]["first"]
    if tamper == "entry":
        first["cutoff_result_entry"]["time_s"] += 1
    elif tamper == "content_id":
        first["cutoff_result_entry_id"] = "0" * 64
    elif tamper == "selection":
        bindings["selected_pattern_sha256"].reverse()
    else:
        first["source_store"] += "-different"
    pb.atomic_json(store / "source_bindings.json", bindings)

    with pytest.raises(ValueError, match="schema-v2"):
        analyze_profile_stores(
            [store], tmp_path / "analysis.json", tmp_path / "analysis.npz")
    assert not (tmp_path / "analysis.json").exists()
    assert not (tmp_path / "analysis.npz").exists()


def test_profile_analysis_rejects_unknown_binding_schema(tmp_path):
    store = _write_store(tmp_path / "store", [("row", 10, "factory", "random")])
    bindings = pb.read_json(store / "source_bindings.json")
    bindings["schema_version"] = 3
    pb.atomic_json(store / "source_bindings.json", bindings)

    with pytest.raises(ValueError, match="unsupported snapshot_successes"):
        analyze_profile_stores(
            [store], tmp_path / "analysis.json", tmp_path / "analysis.npz")


def test_profile_analysis_rejects_corrupt_completed_store_before_outputs(tmp_path):
    store = _write_store(tmp_path / "store", [("row", 3, "factory", "random")])
    result = pb.read_json(store / "results.json")["row"]
    (store / result["sample_file"]).write_bytes(b"corrupt")

    with pytest.raises((ValueError, RuntimeError, EOFError)):
        analyze_profile_stores([store], tmp_path / "analysis.json", tmp_path / "analysis.npz")
    assert not (tmp_path / "analysis.json").exists()
    assert not (tmp_path / "analysis.npz").exists()


def test_profile_analysis_rejects_mixed_measurement_identity(tmp_path):
    first = _write_store(tmp_path / "first", [("a", 4, "factory", "random")])
    second_measurement = _measurement("different-profile")
    second = _write_store(
        tmp_path / "second", [("b", 5, "factory", "geometry")],
        measurement=second_measurement,
    )

    with pytest.raises(ValueError, match="do not share measurement_id and score_spec_id"):
        analyze_profile_stores([first, second], tmp_path / "analysis.json", tmp_path / "analysis.npz")
