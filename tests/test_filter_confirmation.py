from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from antenna.measurement import measurement_id, score_spec_id
from script import profiled_batch as pb
from script.filter_confirmation import analyze_filter_confirmation, write_fresh_json


def _config(tmp_path: Path) -> Path:
    config = {
        "name": "r81-confirmation-test",
        "port": "dual",
        "scope": "r81_confirmation_test",
        "measurement": {
            "name": "dual25-confirmation-test",
            "port": "dual",
            "labels": ["S11", "S21", "S22"],
            "geometry": {"geom": "p01", "pixel_count": 25, "diag_bridge_w": 0.075},
            "sweep": {"start_ghz": 16.0, "stop_ghz": 40.0, "step_ghz": 0.5,
                      "type": "Fast"},
            "solver": {"setup_freq_ghz": 28.0, "open_region_freq_ghz": 16.0,
                       "max_delta_s": 0.02, "max_passes": 6, "min_passes": 5,
                       "min_converged": 5},
        },
        "score_spec": {
            "name": "five-band-test",
            "bands": [
                {"name": "s11_match", "label": "S11", "start_ghz": 26.5,
                 "stop_ghz": 29.5, "relation": "le", "threshold": -10.0},
                {"name": "s22_match", "label": "S22", "start_ghz": 26.5,
                 "stop_ghz": 29.5, "relation": "le", "threshold": -10.0},
                {"name": "s21_pass", "label": "S21", "start_ghz": 26.0,
                 "stop_ghz": 30.0, "relation": "ge", "threshold": -3.0},
                {"name": "s21_low_stop", "label": "S21", "start_ghz": 16.0,
                 "stop_ghz": 20.0, "relation": "le", "threshold": -20.0},
                {"name": "s21_high_stop", "label": "S21", "start_ghz": 36.0,
                 "stop_ghz": 40.0, "relation": "le", "threshold": -20.0},
            ],
        },
    }
    path = tmp_path / "r81.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _pattern(extra=False):
    value = torch.zeros((25, 25), dtype=torch.float32)
    value[:5, 10:15] = 1
    value[20:, 10:15] = 1
    if extra:
        value[10, 10] = 1
    return value


def _response(margin):
    freqs = np.arange(16.0, 40.1, 0.5)
    result = np.full((3, 49), -30.0, dtype=np.float32)
    result[0] = -10.0 - margin
    result[2] = -10.0 - margin
    passing = (freqs >= 26.0) & (freqs <= 30.0)
    result[1, passing] = -3.0 + margin
    result[1, (freqs >= 16.0) & (freqs <= 20.0)] = -20.0 - margin
    result[1, (freqs >= 36.0) & (freqs <= 40.0)] = -20.0 - margin
    return torch.tensor(result)


def _store(root, config_path, row_id, margin, *, repeat=False, parent=None,
           pattern=None, source=None):
    root.mkdir()
    cfg = pb.load_profile_config(config_path)
    pattern = _pattern() if pattern is None else pattern
    row = {
        "id": row_id,
        "port": "dual",
        "kind": "repeat" if repeat else "profile",
        "lineage_id": "shared-lineage",
        "measurement_id": measurement_id(cfg.measurement),
        "score_spec_id": score_spec_id(cfg.score_spec),
        "pattern_sha256": pb.pattern_sha256(pattern),
    }
    if repeat:
        row.update(parent_batch_id=parent, repeat=True, repeat_reason="repeatability",
                   selection_arm="repeat")
    response = _response(margin)
    entry = pb.observation(response, None, row, pattern, cfg, elapsed=1.0)
    sample = root / entry["sample_file"]
    torch.save((pattern, response), sample)
    entry["sample_sha256"] = pb.file_sha256(sample)
    for name, value in (("measurement.json", cfg.measurement),
                        ("score_spec.json", cfg.score_spec),
                        ("manifest.json", [row]), ("results.json", {row_id: entry})):
        pb.atomic_json(root / name, value)
    if source is not None:
        pb.atomic_json(root / "source_bindings.json", {
            "schema_version": 1, "partial_snapshot": True,
            "measurement_id": measurement_id(cfg.measurement),
            "score_spec_id": score_spec_id(cfg.score_spec),
            "items": {row_id: {"source_store": source,
                               "source_sample_file": entry["sample_file"],
                               "source_sample_sha256": entry["sample_sha256"]}},
        })
    assert len(pb.validate_store(root, require_complete=True)) == 1
    return root


@pytest.mark.parametrize("candidate_margin,repeat_margin,confirmed,reason", [
    (0.25, 0.1, True, "both_measured_worst_margins_strictly_positive"),
    (0.0, 0.1, False, "candidate_worst_margin_not_strictly_positive"),
    (0.2, -0.01, False, "repeat_worst_margin_not_strictly_positive"),
])
def test_strict_positive_confirmation_and_complete_evidence(
        tmp_path, candidate_margin, repeat_margin, confirmed, reason):
    config = _config(tmp_path)
    candidate = _store(tmp_path / "candidate", config, "original", candidate_margin)
    repeat = _store(tmp_path / "repeat", config, "retest", repeat_margin,
                    repeat=True, parent="original")

    result = analyze_filter_confirmation(config, candidate, "original", repeat, "retest")

    assert result["performance_target_confirmed"] is confirmed
    assert result["confirmation_reason"] == reason
    assert result["pair_min_worst_margin_db"] == pytest.approx(
        min(candidate_margin, repeat_margin), abs=2e-6)
    assert len(result["candidate"]["margins_db"]) == 5
    assert len(result["repeat"]["worst_frequencies_ghz"]) == 5
    assert result["candidate"]["generic_all_passed_ge_zero"] is (candidate_margin >= 0)
    assert result["engineering_checks_not_assessed"] is True
    assert result["campaign_complete"] is False
    assert result["expected_profile"]["response_shape"] == [3, 49]


def test_rejects_repeat_metadata_profile_pattern_and_same_observation(tmp_path):
    config = _config(tmp_path)
    candidate = _store(tmp_path / "candidate", config, "original", 0.2)
    wrong_parent = _store(tmp_path / "wrong-parent", config, "retest", 0.2,
                          repeat=True, parent="other")
    with pytest.raises(ValueError, match="parent_batch_id"):
        analyze_filter_confirmation(config, candidate, "original", wrong_parent, "retest")

    nonrepeat = _store(tmp_path / "nonrepeat", config, "retest", 0.2)
    with pytest.raises(ValueError, match="repeat row"):
        analyze_filter_confirmation(config, candidate, "original", nonrepeat, "retest")

    different = _store(tmp_path / "different", config, "retest", 0.2,
                       repeat=True, parent="original", pattern=_pattern(extra=True))
    with pytest.raises(ValueError, match="patterns differ"):
        analyze_filter_confirmation(config, candidate, "original", different, "retest")

    bad_config = yaml.safe_load(config.read_text(encoding="utf-8"))
    bad_config["measurement"]["name"] = "different-profile"
    bad_path = tmp_path / "wrong.yaml"
    bad_path.write_text(yaml.safe_dump(bad_config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="expected measurement"):
        analyze_filter_confirmation(bad_path, candidate, "original", different, "retest")

    with pytest.raises(ValueError, match="distinct observations"):
        analyze_filter_confirmation(config, candidate, "original", candidate, "original")


def test_source_binding_rejects_hash_mismatch_and_same_source_copy(tmp_path):
    config = _config(tmp_path)
    candidate = _store(tmp_path / "candidate", config, "original", 0.2,
                       source=str(tmp_path / "worker-a"))
    repeat = _store(tmp_path / "repeat", config, "retest", 0.2, repeat=True,
                    parent="original", source=str(tmp_path / "worker-b"))
    assert analyze_filter_confirmation(
        config, candidate, "original", repeat, "retest")["performance_target_confirmed"]

    document = pb.read_json(repeat / "source_bindings.json")
    document["items"]["retest"]["source_sample_sha256"] = "0" * 64
    pb.atomic_json(repeat / "source_bindings.json", document)
    with pytest.raises(ValueError, match="binding/hash differs"):
        analyze_filter_confirmation(config, candidate, "original", repeat, "retest")

    # Same response bytes are valid across distinct worker observations; only
    # matching trusted source-store + row identity is rejected as a clone.
    clone_candidate = _store(tmp_path / "clone-candidate", config, "shared", 0.2,
                             source=str(tmp_path / "one-worker"))
    clone_repeat = _store(tmp_path / "clone-repeat", config, "shared", 0.2, repeat=True,
                          parent="shared", source=str(tmp_path / "one-worker"))
    with pytest.raises(ValueError, match="same source observation"):
        analyze_filter_confirmation(
            config, clone_candidate, "shared", clone_repeat, "shared")


def test_hash_corruption_and_fresh_output_are_enforced(tmp_path):
    config = _config(tmp_path)
    candidate = _store(tmp_path / "candidate", config, "original", 0.2)
    repeat = _store(tmp_path / "repeat", config, "retest", 0.2,
                    repeat=True, parent="original")
    result = analyze_filter_confirmation(config, candidate, "original", repeat, "retest")
    receipt = write_fresh_json(tmp_path / "receipt.json", result)
    with pytest.raises(FileExistsError):
        write_fresh_json(receipt, result)
    forbidden_parent = candidate / "new-output-directory"
    with pytest.raises(ValueError, match="candidate input store"):
        write_fresh_json(forbidden_parent / "receipt.json", result)
    assert not forbidden_parent.exists()

    entry = pb.read_json(repeat / "results.json")["retest"]
    (repeat / entry["sample_file"]).write_bytes(b"corrupt")
    with pytest.raises((ValueError, RuntimeError, EOFError)):
        analyze_filter_confirmation(config, candidate, "original", repeat, "retest")


def test_rejects_wrong_49_point_frequency_grid(tmp_path):
    config = _config(tmp_path)
    wrong = yaml.safe_load(config.read_text(encoding="utf-8"))
    wrong["measurement"]["sweep"].update(start_ghz=15.0, stop_ghz=39.0)
    wrong["score_spec"]["bands"][-1].update(start_ghz=35.0, stop_ghz=39.0)
    wrong_path = tmp_path / "wrong-grid.yaml"
    wrong_path.write_text(yaml.safe_dump(wrong, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="dual R81 3x49"):
        analyze_filter_confirmation(
            wrong_path, tmp_path / "unused-candidate", "original",
            tmp_path / "unused-repeat", "retest")
