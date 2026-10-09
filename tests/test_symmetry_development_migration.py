from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from antenna.measurement import measurement_id, score_spec_id
from script.exploration import content_id, file_sha256, pattern_sha256
from script.profiled_batch import observation
from script import symmetry_development_migration as migration
from script.symmetry_training import (
    fixed_group_split,
    ingest_profile_stores,
    initialize_workdir,
    register_legacy_prior,
)


def _measurement():
    return {
        "name": "development-migration-test",
        "port": "single",
        "labels": ["S11_dB", "Gain_dBi"],
        "geometry": {"geom": "single_v1", "pixel_count": 25, "diag_bridge_w": 0.1},
        "sweep": {"start_ghz": 24.0, "stop_ghz": 32.0, "step_ghz": 0.5,
                  "type": "Interpolating"},
        "solver": {"setup_freq_ghz": 28.0, "open_region_freq_ghz": 32.0,
                   "max_delta_s": 0.02, "max_passes": 15, "min_passes": 2,
                   "min_converged": 2},
    }


def _score():
    return {"name": "development-migration-score", "bands": [
        {"name": "return", "label": "S11_dB", "start_ghz": 27.5,
         "stop_ghz": 28.5, "relation": "le", "threshold": -10.0},
        {"name": "gain", "label": "Gain_dBi", "start_ghz": 27.5,
         "stop_ghz": 28.5, "relation": "ge", "threshold": 0.0},
    ]}


def _pattern(index: int) -> torch.Tensor:
    value = torch.zeros((25, 25), dtype=torch.float32)
    value[24, 12] = 1
    row = 2 + index % 20
    column = 1 + index % 11
    value[row, column] = value[row, 24 - column] = 1
    return value


def _response(index: int) -> torch.Tensor:
    frequency = torch.arange(17, dtype=torch.float32)
    return torch.stack((-15.0 + 0.05 * frequency + index,
                        2.0 + 0.1 * frequency - index * 0.1))


def _radiation(index: int):
    theta = torch.arange(-180, 181, 2, dtype=torch.float32)
    return {"theta": theta,
            "phi0": torch.cos(torch.deg2rad(theta)) + index * 0.01,
            "phi90": torch.sin(torch.deg2rad(theta)) + index * 0.01}


def _write_store(root: Path, specs: list[tuple[str, str, int]]) -> Path:
    root.mkdir(parents=True)
    (root / "rad").mkdir()
    measurement, score = _measurement(), _score()
    cfg = SimpleNamespace(port="single", measurement=measurement, score_spec=score)
    rows, results = [], {}
    for name, lineage, index in specs:
        pattern, response, rad = _pattern(index), _response(index), _radiation(index)
        row = {"id": name, "port": "single", "measurement_id": measurement_id(measurement),
               "score_spec_id": score_spec_id(score), "pattern_sha256": pattern_sha256(pattern),
               "lineage_id": lineage}
        entry = observation(response, rad, row, pattern, cfg, 1.0 + index)
        sample_path, rad_path = root / entry["sample_file"], root / entry["rad_file"]
        torch.save((pattern, response), sample_path)
        torch.save(rad, rad_path)
        entry["sample_sha256"] = file_sha256(sample_path)
        entry["rad_sha256"] = file_sha256(rad_path)
        rows.append(row)
        results[name] = entry
    for name, value in (("measurement.json", measurement), ("score_spec.json", score),
                        ("manifest.json", rows), ("results.json", results)):
        (root / name).write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return root


def _groups(seed: int, fraction: float) -> dict[str, list[str]]:
    found = {"train": [], "holdout": []}
    index = 0
    while min(len(values) for values in found.values()) < 3:
        name = f"family-{index}"
        found[fixed_group_split(name, fraction, seed)].append(name)
        index += 1
    return found


def _register_legacy(parent: Path, root: Path, lineages: list[str]) -> None:
    source = root / "legacy-source"
    (source / "samples").mkdir(parents=True)
    (source / "rad").mkdir()
    rows = []
    for index, lineage in enumerate(lineages):
        pattern = torch.zeros((25, 25), dtype=torch.float32)
        pattern[index + 1, 3] = 1
        sample, rad = source / "samples" / f"{index}.pt", source / "rad" / f"{index}.pt"
        torch.save((pattern, _response(index + 10)), sample)
        torch.save(_radiation(index + 10), rad)
        rows.append({"id": f"legacy-{index}", "lineage_id": lineage,
                     "sample_file": sample.relative_to(source).as_posix(),
                     "rad_file": rad.relative_to(source).as_posix()})
    manifest = source / "manifest.json"
    manifest.write_text(json.dumps(rows), encoding="utf-8")
    register_legacy_prior(parent, manifest, source, {
        "source_name": "generated-legacy",
        "source_geometry_profile": "legacy-generated",
        "compatibility_note": "shape-compatible generated fixture only",
    })


def _make_parent(root: Path, *, duplicate_pattern: bool = False,
                 add_second_version: bool = True) -> tuple[Path, str]:
    seed, fraction = 71, 0.5
    groups = _groups(seed, fraction)
    development = groups["holdout"][0]
    parent = root / "parent"
    initialize_workdir(
        parent, _measurement(), _score(), geometry_profile="single_v1",
        holdout_seed=seed, holdout_fraction=fraction, ensemble_seeds=(3,), hidden_dims=(8,),
        learning_rate=1e-3, batch_size=2, legacy_epochs=1, current_epochs=2,
        update_min=2, update_max=8,
        initial_lineage_aliases={"development-alias": development},
        alias_source_sha256="ab" * 32,
    )
    specs = [
        ("development-alias-row", "development-alias", 1),
        ("development-direct-row", development, 2),
        ("other-train-row", groups["train"][0], 3),
        ("other-holdout-row", groups["holdout"][1], 4),
    ]
    if duplicate_pattern:
        specs[-1] = ("repeated-pattern-row", groups["holdout"][1], 1)
    first = _write_store(root / "profile-v1", specs)
    ingest_profile_stores(parent, [first], min_new=3 if duplicate_pattern else 4,
                          max_new=4)
    _register_legacy(parent, root, [development, groups["train"][1]])
    (parent / "models").mkdir()
    (parent / "models" / "must-not-copy.bin").write_bytes(b"parent-model-sentinel")
    if add_second_version:
        second = _write_store(root / "profile-v2", [
            ("later-a", groups["train"][2], 7),
            ("later-b", groups["holdout"][2], 8),
        ])
        ingest_profile_stores(parent, [second], min_new=2, max_new=2)
    return parent, development


def _tree_hashes(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): file_sha256(path)
            for path in sorted(root.rglob("*")) if path.is_file()}


def _write_manifest_and_receipt(parent: Path, rows: list[dict], version: int = 1) -> None:
    version_dir = parent / "data" / f"data-v{version:03d}"
    receipt_path = version_dir / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["manifest_id"] = content_id(rows)
    receipt["cumulative_observations"] = len(rows)
    receipt["cumulative_valid_unique"] = len({row["pattern_sha256"] for row in rows})
    (version_dir / "manifest.json").write_text(json.dumps(rows, sort_keys=True), encoding="utf-8")
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")


def _write_protocol(parent: Path, protocol: dict) -> None:
    protocol = dict(protocol)
    protocol["protocol_id"] = content_id({
        key: value for key, value in protocol.items() if key != "protocol_id"})
    (parent / "protocol.json").write_text(
        json.dumps(protocol, sort_keys=True), encoding="utf-8")


def test_migrates_explicit_frozen_version_with_whole_alias_group_and_zero_new_counts(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    parent, development = _make_parent(allowed)
    before = _tree_hashes(parent)
    child = allowed / "child"
    child.mkdir()  # A caller-provided empty destination is an allowed fresh target.

    result = migration.migrate_development_workdir(
        parent, 1, child, [development], allowed_root=allowed)

    assert result == child
    assert _tree_hashes(parent) == before
    assert json.loads((parent / "state.json").read_text())["latest_data_version"] == 2
    parent_rows = json.loads((parent / "data/data-v001/manifest.json").read_text())
    child_rows = json.loads((child / "data/data-v001/manifest.json").read_text())
    assert [row["id"] for row in child_rows] == sorted(row["id"] for row in parent_rows)
    parent_by_id = {row["id"]: row for row in parent_rows}
    for row in child_rows:
        source = parent_by_id[row["id"]]
        for key in ("id", "lineage_id", "canonical_group_id", "pattern_sha256",
                    "source_sample_sha256", "source_rad_sha256", "sample_file", "rad_file"):
            assert row[key] == source[key]
        assert row["reference_split"] == source["split"]
        if row["canonical_group_id"] == development:
            assert row["split"] == "profile_train"
            assert row["reference_split"] == "profile_holdout"
        else:
            assert row["split"] == source["split"]
    assert {row["lineage_id"] for row in child_rows
            if row["canonical_group_id"] == development} == {development, "development-alias"}

    assert (child / "legacy/manifest.json").read_bytes() == (parent / "legacy/manifest.json").read_bytes()
    assert (child / "legacy/provenance.json").read_bytes() == (parent / "legacy/provenance.json").read_bytes()
    legacy = json.loads((child / "legacy/manifest.json").read_text())
    assert next(row for row in legacy if row["lineage_id"] == development)["split"] == "legacy_holdout"
    assert not (child / "models").exists()
    assert json.loads((child / "state.json").read_text())["latest_data_version"] == 1
    receipt = json.loads((child / "data/data-v001/receipt.json").read_text())
    assert receipt["new_valid_unique"] == receipt["new_observations"] == 0
    assert receipt["collection_delta_valid_unique"] == receipt["collection_delta_observations"] == 0
    assert receipt["bootstrap_valid_unique"] == len(parent_rows)
    assert receipt["cumulative_valid_unique"] == len(parent_rows)
    proof = migration.verify_migrated_workdir(child, allowed_root=allowed)
    assert proof["repeat_nonunique_exclusion"]["manifest_rows"] == len(parent_rows)
    assert "do not retain the upstream source kind" in proof["repeat_nonunique_exclusion"]["limitation"]
    parent_protocol = json.loads((parent / "protocol.json").read_text())
    child_protocol = json.loads((child / "protocol.json").read_text())
    assert child_protocol["updates"] == parent_protocol["updates"]
    assert child_protocol["model"] == parent_protocol["model"]
    assert child_protocol["reference_protocol_id"] == parent_protocol["protocol_id"]
    assert child_protocol["development_train_groups"] == [development]
    child_profile_payloads = {row["sample_file"] for row in child_rows} | {
        row["rad_file"] for row in child_rows}
    assert {path.relative_to(child).as_posix() for path in (child / "profile").rglob("*.pt")} == child_profile_payloads


def test_ignores_mutable_parent_state_alias_retarget_and_uses_frozen_sources(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    parent, development = _make_parent(allowed, add_second_version=False)
    state_path = parent / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["lineage_aliases"][development] = "retargeted-by-mutable-state"
    state["lineage_aliases"]["development-alias"] = "retargeted-by-mutable-state"
    state_path.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
    before = _tree_hashes(parent)

    child = migration.migrate_development_workdir(
        parent, 1, allowed / "child", [development], allowed_root=allowed)

    assert _tree_hashes(parent) == before
    rows = json.loads((child / "data/data-v001/manifest.json").read_text())
    members = [row for row in rows if row["canonical_group_id"] == development]
    assert {row["lineage_id"] for row in members} == {
        development, "development-alias"}
    assert {row["split"] for row in members} == {"profile_train"}


@pytest.mark.parametrize("field", ["id", "lineage_id", "canonical_group_id"])
def test_rejects_non_string_profile_identity_without_coercion(tmp_path, field):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    parent, development = _make_parent(allowed, add_second_version=False)
    rows = json.loads((parent / "data/data-v001/manifest.json").read_text())
    rows[0][field] = 17
    _write_manifest_and_receipt(parent, rows)

    with pytest.raises(ValueError, match="normalized string"):
        migration.migrate_development_workdir(
            parent, 1, allowed / "child", [development], allowed_root=allowed)


def test_rejects_non_string_protocol_alias_target_and_canonical_retarget(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    parent, development = _make_parent(allowed, add_second_version=False)
    protocol = json.loads((parent / "protocol.json").read_text())
    protocol["lineage_aliases"]["mapping"]["development-alias"] = 17
    _write_protocol(parent, protocol)
    with pytest.raises(ValueError, match="normalized string"):
        migration.migrate_development_workdir(
            parent, 1, allowed / "child-a", [development], allowed_root=allowed)

    parent, development = _make_parent(allowed / "second", add_second_version=False)
    protocol = json.loads((parent / "protocol.json").read_text())
    protocol["lineage_aliases"]["mapping"][development] = "retargeted-root"
    _write_protocol(parent, protocol)
    with pytest.raises(ValueError, match="canonical roots|alias root"):
        migration.migrate_development_workdir(
            parent, 1, allowed / "child-b", [development], allowed_root=allowed)


@pytest.mark.parametrize("groups", [[17], [" normalized-with-space "]])
def test_rejects_non_string_or_unnormalized_development_group(tmp_path, groups):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    parent, _development = _make_parent(allowed, add_second_version=False)
    with pytest.raises(ValueError, match="normalized string"):
        migration.migrate_development_workdir(
            parent, 1, allowed / "child", groups, allowed_root=allowed)


@pytest.mark.parametrize("tamper", ["role", "raw_hash", "canonical_identity", "truth_shape"])
def test_rejects_parent_role_identity_hash_and_truth_tamper(tmp_path, tamper):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    parent, development = _make_parent(allowed, add_second_version=False)
    manifest_path = parent / "data/data-v001/manifest.json"
    rows = json.loads(manifest_path.read_text())
    if tamper == "role":
        rows[0]["split"] = "profile_train" if rows[0]["split"] == "profile_holdout" else "profile_holdout"
    elif tamper == "raw_hash":
        rows[0]["source_sample_sha256"] = "0" * 64
    elif tamper == "canonical_identity":
        rows[1]["lineage_id"] = rows[0]["lineage_id"]
        rows[1]["canonical_group_id"] = "different-canonical-root"
        rows[1]["split"] = "profile_" + fixed_group_split("different-canonical-root", 0.5, 71)
    else:
        sample = parent / rows[0]["sample_file"]
        pattern, _response_value = torch.load(sample, weights_only=True)
        torch.save((pattern, torch.zeros((2, 16))), sample)
        rows[0]["source_sample_sha256"] = file_sha256(sample)
    _write_manifest_and_receipt(parent, rows)

    with pytest.raises(ValueError):
        migration.migrate_development_workdir(
            parent, 1, allowed / "child", [development], allowed_root=allowed)


def test_rejects_repeated_parent_pattern_and_reports_no_silent_dedup(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    parent, development = _make_parent(
        allowed, duplicate_pattern=True, add_second_version=False)
    with pytest.raises(ValueError, match="repeated/nonunique patterns"):
        migration.migrate_development_workdir(
            parent, 1, allowed / "child", [development], allowed_root=allowed)


def test_rejects_external_overlap_and_nonempty_destination(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    parent, development = _make_parent(allowed, add_second_version=False)
    nonempty = allowed / "nonempty"
    nonempty.mkdir()
    (nonempty / "sentinel").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="empty fresh"):
        migration.migrate_development_workdir(
            parent, 1, nonempty, [development], allowed_root=allowed)
    with pytest.raises(ValueError, match="must not overlap"):
        migration.migrate_development_workdir(
            parent, 1, parent / "child", [development], allowed_root=allowed)
    external = tmp_path / "external-parent"
    external.mkdir()
    with pytest.raises(ValueError, match="allowed root"):
        migration.migrate_development_workdir(
            external, 1, allowed / "child", [development], allowed_root=allowed)
    assert (nonempty / "sentinel").read_text(encoding="utf-8") == "keep"


def test_partial_failure_stays_unpublished_and_requires_manual_disposition(tmp_path, monkeypatch):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    parent, development = _make_parent(allowed, add_second_version=False)
    child = allowed / "child"
    original = migration.training._atomic_copy
    calls = 0

    def fail_during_copy(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("generated copy interruption")
        return original(source, destination)

    monkeypatch.setattr(migration.training, "_atomic_copy", fail_during_copy)
    with pytest.raises(OSError, match="generated copy interruption"):
        migration.migrate_development_workdir(
            parent, 1, child, [development], allowed_root=allowed)


@pytest.mark.parametrize("tamper", ["alias", "protocol_id"])
def test_rejects_initialized_staging_state_identity_tamper_before_publication(
        tmp_path, monkeypatch, tamper):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    parent, development = _make_parent(allowed, add_second_version=False)
    child = allowed / "child"
    original = migration.training.initialize_workdir

    def initialize_then_retarget(*args, **kwargs):
        result = original(*args, **kwargs)
        state_path = result / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if tamper == "alias":
            state["lineage_aliases"][development] = "retargeted-staging-root"
        else:
            state["protocol_id"] = "0" * 64
        state_path.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
        return result

    monkeypatch.setattr(
        migration.training, "initialize_workdir", initialize_then_retarget)
    with pytest.raises(ValueError, match="state protocol or alias closure"):
        migration.migrate_development_workdir(
            parent, 1, child, [development], allowed_root=allowed)
    assert not child.exists()
    assert not child.exists()
    pending = list(allowed.glob(".child.migration-pending-*"))
    assert len(pending) == 1
    assert not (pending[0] / "migration_complete.json").exists()
    assert json.loads((pending[0] / "state.json").read_text())["latest_data_version"] == 0

    monkeypatch.setattr(migration.training, "_atomic_copy", original)
    with pytest.raises(FileExistsError, match="will not be overwritten"):
        migration.migrate_development_workdir(
            parent, 1, child, [development], allowed_root=allowed)


def test_verifier_rejects_postpublication_manifest_and_payload_tamper(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    parent, development = _make_parent(allowed, add_second_version=False)
    child = migration.migrate_development_workdir(
        parent, 1, allowed / "child", [development], allowed_root=allowed)
    rows = json.loads((child / "data/data-v001/manifest.json").read_text())
    payload = child / rows[0]["sample_file"]
    payload.write_bytes(payload.read_bytes() + b"tamper")
    with pytest.raises(ValueError):
        migration.verify_migrated_workdir(child, allowed_root=allowed)


def test_verifier_rejects_resealed_published_state_alias_retarget(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    parent, development = _make_parent(allowed, add_second_version=False)
    child = migration.migrate_development_workdir(
        parent, 1, allowed / "child", [development], allowed_root=allowed)
    state_path = child / "state.json"
    marker_path = child / "migration_complete.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["lineage_aliases"][development] = "retargeted-published-root"
    state_path.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["state_sha256"] = file_sha256(state_path)
    marker_path.write_text(json.dumps(marker, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="alias/protocol binding"):
        migration.verify_migrated_workdir(child, allowed_root=allowed)
