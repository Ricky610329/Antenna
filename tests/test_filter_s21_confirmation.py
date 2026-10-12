"""Focused generated native-store tests for three-band S21 confirmation."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from antenna.measurement import measurement_id, score_spec_id
from script import filter_s21_confirmation as confirmation
from script import profiled_batch as pb
from script.filter_s21_confirmation import (
    CANONICAL_PROFILE,
    analyze_filter_s21_confirmation,
    write_fresh_json,
)


def _canonical_copy(tmp_path: Path) -> Path:
    path = tmp_path / "dual_r81_s21_discrete.yaml"
    shutil.copy2(CANONICAL_PROFILE, path)
    return path


def _pattern(extra: bool = False) -> torch.Tensor:
    value = torch.zeros((25, 25), dtype=torch.float32)
    value[:5, 10:15] = 1
    value[20:, 10:15] = 1
    if extra:
        value[10, 10] = 1
    return value


def _response(margin: float) -> torch.Tensor:
    """S11/S22 deliberately fail any plausible match gate; only S21 scores."""
    freqs = np.arange(16.0, 40.1, 0.5)
    result = np.full((3, 49), 42.0, dtype=np.float32)
    result[1] = -10.0
    result[1, (freqs >= 26.0) & (freqs <= 30.0)] = -3.0 + margin
    result[1, (freqs >= 16.0) & (freqs <= 20.0)] = -20.0 - margin
    result[1, (freqs >= 36.0) & (freqs <= 40.0)] = -20.0 - margin
    return torch.from_numpy(result)


def _store(root: Path, config: Path, row_id: str, margin: float, *,
           repeat: bool = False, parent: str | None = None,
           pattern: torch.Tensor | None = None) -> Path:
    root.mkdir()
    cfg = pb.load_profile_config(config)
    pattern = _pattern() if pattern is None else pattern
    row = {
        "id": row_id,
        "port": "dual",
        "kind": "repeat" if repeat else "profile",
        "lineage_id": "fixture-lineage",
        "measurement_id": measurement_id(cfg.measurement),
        "score_spec_id": score_spec_id(cfg.score_spec),
        "pattern_sha256": pb.pattern_sha256(pattern),
    }
    if repeat:
        row.update(parent_batch_id=parent, repeat=True,
                   repeat_reason="repeatability", selection_arm="repeat")
    response = _response(margin)
    entry = pb.observation(response, None, row, pattern, cfg, elapsed=1.0)
    sample = root / entry["sample_file"]
    torch.save((pattern, response), sample)
    entry["sample_sha256"] = pb.file_sha256(sample)
    for name, value in (("measurement.json", cfg.measurement),
                        ("score_spec.json", cfg.score_spec),
                        ("manifest.json", [row]),
                        ("results.json", {row_id: entry})):
        pb.atomic_json(root / name, value)
    assert len(pb.validate_store(root, require_complete=True)) == 1
    return root


def _snapshot(root: Path, source: Path) -> tuple[Path, str]:
    root.mkdir()
    rows = pb.read_json(source / "manifest.json")
    results = pb.read_json(source / "results.json")
    assert len(rows) == len(results) == 1
    row_id = rows[0]["id"]
    entry = results[row_id]
    for name in ("measurement.json", "score_spec.json", "manifest.json", "results.json"):
        shutil.copy2(source / name, root / name)
    shutil.copy2(source / entry["sample_file"], root / entry["sample_file"])
    bindings = root / "source_bindings.json"
    pb.atomic_json(bindings, {
        "schema_version": 1,
        "partial_snapshot": True,
        "measurement_id": entry["measurement_id"],
        "score_spec_id": entry["score_spec_id"],
        "items": {row_id: {
            "source_store": str(source.resolve()),
            "source_sample_file": entry["sample_file"],
            "source_sample_sha256": entry["sample_sha256"],
        }},
    })
    assert len(pb.validate_store(root, require_complete=True)) == 1
    return root, pb.file_sha256(bindings)


def _pair(tmp_path: Path, config: Path, candidate_margin: float,
          repeat_margin: float, *, repeat_parent: str = "original") -> dict:
    candidate_source = _store(
        tmp_path / "candidate-source", config, "original", candidate_margin)
    repeat_source = _store(
        tmp_path / "repeat-source", config, "retest", repeat_margin,
        repeat=True, parent=repeat_parent)
    candidate, candidate_pin = _snapshot(tmp_path / "candidate", candidate_source)
    repeat, repeat_pin = _snapshot(tmp_path / "repeat", repeat_source)
    return {
        "candidate": candidate, "candidate_source": candidate_source,
        "candidate_pin": candidate_pin, "repeat": repeat,
        "repeat_source": repeat_source, "repeat_pin": repeat_pin,
    }


def _analyze(config: Path, pair: dict):
    return analyze_filter_s21_confirmation(
        config, pair["candidate"], "original", pair["repeat"], "retest",
        candidate_source_bindings_sha256=pair["candidate_pin"],
        repeat_source_bindings_sha256=pair["repeat_pin"])


def _append_unrelated_native_observation(source: Path, config: Path) -> None:
    cfg = pb.load_profile_config(config)
    rows = pb.read_json(source / "manifest.json")
    results = pb.read_json(source / "results.json")
    pattern = _pattern(extra=True)
    row = {
        "id": "unrelated-append", "port": "dual", "kind": "profile",
        "lineage_id": "unrelated-lineage",
        "measurement_id": measurement_id(cfg.measurement),
        "score_spec_id": score_spec_id(cfg.score_spec),
        "pattern_sha256": pb.pattern_sha256(pattern),
    }
    response = _response(-0.5)
    entry = pb.observation(response, None, row, pattern, cfg, elapsed=1.0)
    sample = source / entry["sample_file"]
    torch.save((pattern, response), sample)
    entry["sample_sha256"] = pb.file_sha256(sample)
    rows.append(row)
    results[row["id"]] = entry
    pb.atomic_json(source / "manifest.json", rows)
    pb.atomic_json(source / "results.json", results)
    assert len(pb.validate_store(source, require_complete=True)) == 2


@pytest.mark.parametrize("candidate_margin,repeat_margin,confirmed,reason", [
    (0.25, 0.10, True, "both_measured_s21_worst_margins_strictly_positive"),
    (-0.01, 0.10, False, "candidate_worst_margin_not_strictly_positive"),
    (0.00, 0.10, False, "candidate_worst_margin_not_strictly_positive"),
])
def test_positive_negative_and_zero_use_only_three_s21_bands(
        tmp_path, candidate_margin, repeat_margin, confirmed, reason):
    config = _canonical_copy(tmp_path)
    pair = _pair(tmp_path, config, candidate_margin, repeat_margin)
    # A later unrelated append to the native source is allowed; the selected
    # sample and the caller-pinned snapshot binding remain unchanged.
    _append_unrelated_native_observation(pair["candidate_source"], config)

    result = _analyze(config, pair)

    assert result["s21_positive_margin_repeat_confirmed"] is confirmed
    assert result["confirmation_reason"] == reason
    assert result["pair_min_worst_margin_db"] == pytest.approx(
        min(candidate_margin, repeat_margin), abs=2e-6)
    assert result["expected_profile"]["response_labels"] == ["S11", "S21", "S22"]
    assert result["expected_profile"]["native_profile_documents_exact"] is True
    assert [fact["name"] for fact in result["source_band_facts"]] == [
        "s21_pass", "s21_low_stop", "s21_high_stop"]
    assert len(result["candidate"]["margins_db"]) == 3
    assert result["candidate"]["generic_all_passed_ge_zero"] is (
        candidate_margin >= 0.0)
    assert result["formal_gate_satisfied"] is False
    assert result["engineering_gate_satisfied"] is False
    assert result["campaign_complete"] is False
    assert result["pair_binding"]["distinct_bound_source_observations"] is True
    assert result["provenance_assurance"][
        "independent_hfss_execution_cryptographically_proven"] is False
    assert result["pair_binding"]["candidate_source_bindings_sha256"] == pair[
        "candidate_pin"]
    assert any("S11 and S22" in item for item in result["limitations"])
    assert any("outside 16--40" in item for item in result["limitations"])


@pytest.mark.parametrize("change", ["profile", "geometry"])
def test_rejects_noncanonical_profile_or_native_geometry(tmp_path, change):
    config = _canonical_copy(tmp_path)
    candidate_config = config
    analysis_config = config
    modified = yaml.safe_load(config.read_text(encoding="utf-8"))
    if change == "profile":
        modified["name"] = "lookalike-but-not-canonical"
        analysis_config = tmp_path / "wrong-profile.yaml"
        analysis_config.write_text(yaml.safe_dump(modified, sort_keys=False), encoding="utf-8")
    else:
        modified["measurement"]["geometry"]["diag_bridge_w"] = 0.08
        candidate_config = tmp_path / "wrong-geometry.yaml"
        candidate_config.write_text(yaml.safe_dump(modified, sort_keys=False), encoding="utf-8")

    pair = _pair(tmp_path, candidate_config, 0.2, 0.2)
    if change == "geometry":
        # The repeat side should remain canonical so the candidate's native
        # geometry is the specific rejection under test.
        canonical_repeat_source = _store(
            tmp_path / "canonical-repeat-source", config, "retest", 0.2,
            repeat=True, parent="original")
        pair["repeat"], pair["repeat_pin"] = _snapshot(
            tmp_path / "canonical-repeat", canonical_repeat_source)
        pair["repeat_source"] = canonical_repeat_source
    expected = "exact copy" if change == "profile" else "native profile documents"
    with pytest.raises(ValueError, match=expected):
        _analyze(analysis_config, pair)


def test_rejects_wrong_parent_and_snapshot_copy_of_same_source(tmp_path):
    config = _canonical_copy(tmp_path)
    pair = _pair(tmp_path, config, 0.2, 0.2, repeat_parent="other")
    with pytest.raises(ValueError, match="parent_batch_id"):
        _analyze(config, pair)

    pattern_root = tmp_path / "pattern"
    pattern_root.mkdir()
    pattern_pair = _pair(pattern_root, config, 0.2, 0.2)
    wrong_pattern_source = _store(
        tmp_path / "wrong-pattern-source", config, "retest", 0.2,
        repeat=True, parent="original", pattern=_pattern(extra=True))
    pattern_pair["repeat"], pattern_pair["repeat_pin"] = _snapshot(
        tmp_path / "wrong-pattern", wrong_pattern_source)
    pattern_pair["repeat_source"] = wrong_pattern_source
    with pytest.raises(ValueError, match="patterns differ"):
        _analyze(config, pattern_pair)

    shared_source = _store(tmp_path / "shared-source", config, "shared", 0.2)
    copied_candidate, candidate_pin = _snapshot(
        tmp_path / "copied-candidate", shared_source)
    copied_repeat, repeat_pin = _snapshot(tmp_path / "copied-repeat", shared_source)
    with pytest.raises(ValueError, match="same actual source observation"):
        analyze_filter_s21_confirmation(
            config, copied_candidate, "shared", copied_repeat, "shared",
            candidate_source_bindings_sha256=candidate_pin,
            repeat_source_bindings_sha256=repeat_pin)


def test_requires_pinned_bindings_and_rejects_modified_binding(tmp_path):
    config = _canonical_copy(tmp_path)
    pair = _pair(tmp_path, config, 0.2, 0.2)

    unbound = tmp_path / "unbound-copy"
    shutil.copytree(pair["candidate_source"], unbound)
    with pytest.raises(ValueError, match="requires source_bindings"):
        analyze_filter_s21_confirmation(
            config, unbound, "original", pair["repeat"], "retest",
            candidate_source_bindings_sha256="0" * 64,
            repeat_source_bindings_sha256=pair["repeat_pin"])

    binding_path = pair["candidate"] / "source_bindings.json"
    binding = pb.read_json(binding_path)
    binding["items"]["unrelated"] = dict(binding["items"]["original"])
    pb.atomic_json(binding_path, binding)
    with pytest.raises(ValueError, match="differs from caller pin"):
        _analyze(config, pair)


def test_fresh_output_is_atomic_exclusive_and_outside_all_evidence(
        tmp_path, monkeypatch):
    config = _canonical_copy(tmp_path)
    pair = _pair(tmp_path, config, 0.2, 0.2)
    result = _analyze(config, pair)

    output = write_fresh_json(tmp_path / "receipts" / "confirmation.json", result)
    assert output.is_file()
    assert not list(output.parent.glob(".filter-s21-confirmation-draft-*"))
    with pytest.raises(FileExistsError):
        write_fresh_json(output, result)
    for source in (pair["candidate"], pair["repeat"],
                   pair["candidate_source"], pair["repeat_source"]):
        forbidden = source / "new" / "confirmation.json"
        with pytest.raises(ValueError, match="evidence root"):
            write_fresh_json(forbidden, result)
        assert not forbidden.parent.exists()

    bad = dict(result)
    bad["not_json"] = float("nan")
    invalid = tmp_path / "serialization-failure" / "receipt.json"
    with pytest.raises(ValueError, match="Out of range float"):
        write_fresh_json(invalid, bad)
    assert not invalid.exists() and not invalid.parent.exists()

    crash = tmp_path / "crash" / "receipt.json"

    def fail_link(_source, _destination):
        raise OSError("generated publication failure")

    monkeypatch.setattr(confirmation.os, "link", fail_link)
    with pytest.raises(OSError, match="generated publication failure"):
        write_fresh_json(crash, result)
    assert not crash.exists()
    assert not list(crash.parent.glob(".filter-s21-confirmation-draft-*"))

    existing = tmp_path / "existing.json"
    existing.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        write_fresh_json(existing, result)
    assert existing.read_bytes() == b"existing"


def test_cli_help():
    completed = subprocess.run(
        [sys.executable, "-m", "script.filter_s21_confirmation", "--help"],
        cwd=CANONICAL_PROFILE.parents[1], text=True, capture_output=True,
        check=False)
    assert completed.returncode == 0, completed.stderr
    assert "--candidate-store" in completed.stdout
    assert "--candidate-source-bindings-sha256" in completed.stdout
    assert "--repeat-store" in completed.stdout
    assert "--repeat-source-bindings-sha256" in completed.stdout
    assert "--out" in completed.stdout
