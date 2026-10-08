import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from antenna.measurement import measurement_id, score_spec_id
from script import exploration as ex
from script import profiled_batch as pb
from script import symmetry_incumbent_shell_pilot as pilot
from script.symmetry_sm_pool import PredictionBatch


REPO = Path(__file__).resolve().parents[1]
PROFILE = REPO / "configs" / "single_r80_symmetry_factory.yaml"


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pb.atomic_json(path, value)


def _tree_hash(root):
    hashes = {p.relative_to(root).as_posix(): pb.file_sha256(p)
              for p in sorted(root.rglob("*")) if p.is_file()}
    return pilot._content_id(hashes)


def _random_unique(rng, seen):
    while True:
        pattern = ex.random_single_pattern(rng)
        digest = pb.pattern_sha256(pattern)
        if digest not in seen:
            seen.add(digest)
            return pattern, digest


def _request_fixture(tmp_path, blind_count=8):
    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg = pb.load_profile_config(PROFILE)
    rng = np.random.default_rng(1708)
    seen = set()
    anchor_pattern, anchor_hash = _random_unique(rng, seen)
    anchor_id = "fixture_anchor"
    ancestry = "fixture_ancestry"
    source_id = "fixture_source"
    anchor_root = tmp_path / "anchor"
    anchor_root.mkdir()
    anchor_tensor = anchor_root / f"{anchor_id}.pt"
    torch.save(torch.as_tensor(anchor_pattern, dtype=torch.float32), anchor_tensor)
    anchor_manifest = anchor_root / "manifest.json"
    _write(anchor_manifest, [{"id": anchor_id, "pattern_sha256": anchor_hash,
                              "ancestry_id": ancestry, "source_id": source_id}])

    blind = tmp_path / "blind"
    blind.mkdir()
    shutil.copy2(PROFILE, blind / "config.yaml")
    _write(blind / "measurement.json", cfg.measurement)
    _write(blind / "score_spec.json", cfg.score_spec)
    blind_rows = []
    for index in range(blind_count):
        pattern, digest = _random_unique(rng, seen)
        name = f"blind_{index:03d}"
        torch.save(torch.as_tensor(pattern, dtype=torch.float32), blind / f"{name}.pt")
        blind_rows.append({
            "id": name, "pattern_sha256": digest, "port": "single",
            "measurement_id": measurement_id(cfg.measurement),
            "score_spec_id": score_spec_id(cfg.score_spec),
            "kind": "symmetry_factory", "selection_arm": "blind_buffer",
            "candidate_group": "fresh_random", "source": "fixture_blind",
            "lineage_id": f"blind-lineage-{index}",
        })
    _write(blind / "manifest.json", blind_rows)
    assert len(pb.validate_input(blind, cfg)) == blind_count

    training = tmp_path / "training_protocol.json"
    _write(training, {"schema_version": 1, "fixture": True})
    addendum = tmp_path / "addendum.md"
    addendum.write_text("fixture addendum\n", encoding="utf-8")
    protocol = tmp_path / "protocol.json"
    _write(protocol, {
        "schema_version": 1, "algorithm_id": pilot.ALGORITHM_ID,
        "scope": cfg.scope,
        "fixed_anchor": {"id": anchor_id, "pattern_sha256": anchor_hash,
                         "ancestry_id": ancestry},
        "unchanged_campaign": {
            "target_valid_unique": 5000,
            "profile_sha256": pb.file_sha256(PROFILE),
            "measurement_id": measurement_id(cfg.measurement),
            "score_spec_id": score_spec_id(cfg.score_spec),
            "training_protocol_sha256": pb.file_sha256(training),
        },
        "dispatch": {"rows_per_job": 16, "priority": 1, "planned_kind": "p"},
    })
    request_path = tmp_path / "request.json"
    request = {
        "schema_version": 1, "kind": pilot.REQUEST_KIND,
        "algorithm_id": pilot.ALGORITHM_ID,
        "protocol_binding": {"path": str(protocol.resolve()),
                             "sha256": pb.file_sha256(protocol)},
        "addendum_binding": {"path": str(addendum.resolve()),
                             "sha256": pb.file_sha256(addendum)},
        "profile_binding": {"path": str(PROFILE.resolve()),
                            "sha256": pb.file_sha256(PROFILE)},
        "training_protocol_binding": {"path": str(training.resolve()),
                                      "sha256": pb.file_sha256(training)},
        "anchor": {
            "id": anchor_id, "pattern_sha256": anchor_hash,
            "ancestry_id": ancestry, "source_id": source_id,
            "manifest_path": str(anchor_manifest.resolve()),
            "manifest_sha256": pb.file_sha256(anchor_manifest),
            "tensor_path": str(anchor_tensor.resolve()),
            "tensor_sha256": pb.file_sha256(anchor_tensor),
        },
        "blind_pool_binding": {"path": str(blind.resolve()), "sha256": _tree_hash(blind)},
        "campaign": {
            "target_valid_unique": 5000, "max_guided_outstanding": 96,
            "shard_size": 16, "priority": 1, "planned_kind": "p",
            "measurement_id": measurement_id(cfg.measurement),
            "score_spec_id": score_spec_id(cfg.score_spec),
            "profile_sha256": pb.file_sha256(PROFILE),
            "training_protocol_sha256": pb.file_sha256(training),
        },
        "selection_salt": pilot.SELECTION_SALT, "one_shot": True,
    }
    _write(request_path, request)
    return pilot.load_request(request_path, PROFILE), blind_rows


class FakePredictor:
    model_role = "current_profile"

    def __init__(self, cfg, offset=0.0):
        self.offset = offset
        self.calls = 0
        self.binding = {
            "measurement_id": measurement_id(cfg.measurement),
            "score_spec_id": score_spec_id(cfg.score_spec),
            "protocol_id": "fixture-training", "data_version": 12,
            "manifest_id": "fixture-manifest", "cumulative_valid_unique": 639,
        }
        self.model_ids = [{"file": "fixture-model", "sha256": "a" * 64,
                           "protocol_id": "fixture-training", "data_version": 12,
                           "manifest_id": "fixture-manifest"}]

    def predict(self, patterns, *, batch_size):
        self.calls += 1
        n = len(patterns)
        response = np.empty((n, 2, 17), dtype=np.float32)
        response[:, 0] = -12.0 - self.offset
        response[:, 1] = 5.0 + self.offset
        theta = np.linspace(-90, 90, 91, dtype=np.float32)
        radiation = np.zeros((n, 2, 91), dtype=np.float32) + self.offset
        members = np.stack((response - 0.1, response + 0.1))
        return PredictionBatch(response, np.full(n, 0.2, dtype=np.float32),
                               radiation, theta, members)


def test_request_and_selection_are_deterministic_exact_d1_and_preserve_lineages(tmp_path):
    request, blind_rows = _request_fixture(tmp_path)
    first = pilot.select_rows(request, {request.anchor["pattern_sha256"]})
    second = pilot.select_rows(request, {request.anchor["pattern_sha256"]})

    assert first.hashes == second.hashes and first.role_counts == {
        pilot.SHELL_ROLE: 15, pilot.CONTROL_ROLE: 1}
    shell = first.rows[:15]
    assert {(row["row_stratum"], row["column_stratum"]) for row in shell} == {
        (r, c) for r in range(5) for c in range(3)}
    assert all(row["independent_hamming_distance"] == 1 for row in shell)
    assert all(row["physical_hamming_distance"] in (1, 2) for row in shell)
    assert all(row["lineage_id"] == request.anchor["ancestry_id"] ==
               row["ancestry_id"] for row in shell)
    assert first.rows[-1]["lineage_id"] == blind_rows[0]["lineage_id"]
    assert first.rows[-1]["blind_source_id"] == blind_rows[0]["id"]


def test_selection_is_frozen_before_the_single_predictor_call(tmp_path):
    request, _ = _request_fixture(tmp_path)
    selection = pilot.select_rows(request, {request.anchor["pattern_sha256"]})
    p1, p2 = FakePredictor(request.profile, 0.0), FakePredictor(request.profile, 2.0)
    a1 = pilot.annotate_selection(request, selection, p1)
    a2 = pilot.annotate_selection(request, selection, p2)

    assert p1.calls == p2.calls == 1
    assert a1.hashes == a2.hashes == selection.hashes
    assert a1.rows[0]["predicted_s11"] != a2.rows[0]["predicted_s11"]
    assert all(row["predictor_model_ids"] == list(a1.model_ids) for row in a1.rows)


def test_stratum_and_blind_unavailability_have_only_stable_permanent_codes(tmp_path):
    request, blind_rows = _request_fixture(tmp_path)
    anchor = np.asarray(torch.load(request.anchor["tensor_path"], weights_only=True), dtype=bool)
    excluded = {request.anchor["pattern_sha256"]}
    for row in range(0, 5):
        for col in range(0, 4):
            pattern = anchor.copy(); pattern[row, col] = ~pattern[row, col]
            excluded.add(pb.pattern_sha256(ex.enforce_single_geometry(pattern)))
    with pytest.raises(pilot.PilotUnavailable) as caught:
        pilot.select_rows(request, excluded)
    assert caught.value.code == "empty_stratum"

    all_blind = {row["pattern_sha256"] for row in blind_rows}
    with pytest.raises(pilot.PilotUnavailable) as caught:
        pilot.select_rows(request, all_blind | {request.anchor["pattern_sha256"]})
    assert caught.value.code == "no_eligible_blind"


def test_request_and_bound_source_tamper_fail_before_prediction(tmp_path):
    request, _ = _request_fixture(tmp_path)
    raw = json.loads(request.path.read_text(encoding="utf-8"))
    raw["campaign"]["priority"] = 6
    _write(request.path, raw)
    with pytest.raises(pilot.PilotError, match="request changed"):
        pilot.select_rows(request, set())

    request2, _ = _request_fixture(tmp_path / "second")
    Path(request2.addendum_binding["path"]).write_text("changed\n", encoding="utf-8")
    with pytest.raises(pilot.PilotError, match="addendum"):
        pilot.select_rows(request2, set())

    request3, _ = _request_fixture(tmp_path / "third")
    raw3 = json.loads(request3.path.read_text(encoding="utf-8"))
    protocol_path = Path(raw3["protocol_binding"]["path"])
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["unchanged_campaign"]["measurement_id"] = "f" * 64
    _write(protocol_path, protocol)
    raw3["protocol_binding"]["sha256"] = pb.file_sha256(protocol_path)
    _write(request3.path, raw3)
    with pytest.raises(pilot.PilotError, match="unchanged campaign identity"):
        pilot.load_request(request3.path, PROFILE)
    protocol["unchanged_campaign"]["measurement_id"] = measurement_id(request3.profile.measurement)
    protocol["dispatch"] = []
    _write(protocol_path, protocol)
    raw3["protocol_binding"]["sha256"] = pb.file_sha256(protocol_path)
    _write(request3.path, raw3)
    with pytest.raises(pilot.PilotError, match="campaign or dispatch schema"):
        pilot.load_request(request3.path, PROFILE)


def test_worker_bundle_roundtrip_and_manifest_tamper_rejection(tmp_path):
    request, _ = _request_fixture(tmp_path)
    selection = pilot.select_rows(request, {request.anchor["pattern_sha256"]})
    annotated = pilot.annotate_selection(request, selection, FakePredictor(request.profile))
    bundle = pilot.write_pilot_bundle(request, annotated, tmp_path / "bundle")
    checked = pilot.validate_bundle(bundle, request)

    assert checked["ordered_hashes"] == list(selection.hashes)
    assert checked["role_counts"] == {pilot.SHELL_ROLE: 15, pilot.CONTROL_ROLE: 1}
    assert len(pb.validate_input(bundle, request.profile)) == 16
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    manifest[0]["pilot_id"] = "wrong"
    _write(bundle / "manifest.json", manifest)
    with pytest.raises(pilot.PilotError):
        pilot.validate_bundle(bundle, request)
    manifest[0]["pilot_id"] = request.pilot_id
    manifest[0]["factory_score"] += 1.0
    _write(bundle / "manifest.json", manifest)
    with pytest.raises(pilot.PilotError, match="predictor scalar"):
        pilot.validate_bundle(bundle, request)
    manifest[0]["factory_score"] -= 1.0
    original_coordinate = manifest[0]["independent_coordinate"]
    manifest[0]["independent_coordinate"] = [24, 12]
    _write(bundle / "manifest.json", manifest)
    with pytest.raises(pilot.PilotError, match="deterministic selection field"):
        pilot.validate_bundle(bundle, request)
    manifest[0]["independent_coordinate"] = original_coordinate
    manifest[-1]["blind_source_id"] = "not-the-first-eligible-row"
    _write(bundle / "manifest.json", manifest)
    with pytest.raises(pilot.PilotError, match="deterministic selection field"):
        pilot.validate_bundle(bundle, request)


def _observation_entry(store, row, pattern, response, rad, cfg, elapsed=1.0):
    pattern_tensor = torch.as_tensor(pattern, dtype=torch.float32)
    response_tensor = torch.as_tensor(response, dtype=torch.float32)
    entry = pb.observation(response_tensor, rad, row, pattern_tensor, cfg, elapsed)
    sample = store / entry["sample_file"]
    rad_path = store / entry["rad_file"]
    rad_path.parent.mkdir(exist_ok=True)
    torch.save((pattern_tensor, response_tensor), sample)
    torch.save(rad, rad_path)
    entry["sample_sha256"] = pb.file_sha256(sample)
    entry["rad_sha256"] = pb.file_sha256(rad_path)
    return entry


def _pair_fixture(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg = pb.load_profile_config(PROFILE)
    rng = np.random.default_rng(992)
    seen = set()
    pattern_a, hash_a = _random_unique(rng, seen)
    pattern_b, hash_b = _random_unique(rng, seen)
    rows = [
        {"id": "repeat-first", "pattern_sha256": hash_a, "kind": "repeat"},
        {"id": "normal-second", "pattern_sha256": hash_a, "kind": "pilot-source"},
        {"id": "pending", "pattern_sha256": hash_b, "kind": "pilot-source"},
    ]
    for row in rows:
        row.update(port="single", measurement_id=measurement_id(cfg.measurement),
                   score_spec_id=score_spec_id(cfg.score_spec))
    input_dir, store = tmp_path / "input", tmp_path / "store"
    input_dir.mkdir(); store.mkdir()
    shutil.copy2(PROFILE, input_dir / "config.yaml")
    for root in (input_dir, store):
        _write(root / "measurement.json", cfg.measurement)
        _write(root / "score_spec.json", cfg.score_spec)
        _write(root / "manifest.json", rows)
    for row, pattern in zip(rows, (pattern_a, pattern_a, pattern_b)):
        torch.save(torch.as_tensor(pattern, dtype=torch.float32), input_dir / f"{row['id']}.pt")
    response_a = np.stack((np.full(17, -11.0), np.full(17, 4.5)))
    response_b = np.stack((np.full(17, -12.0), np.full(17, 5.0)))
    theta = np.arange(-180, 181, 2, dtype=float)
    rad = {"theta": torch.as_tensor(theta, dtype=torch.float32),
           "phi0": torch.zeros(len(theta), dtype=torch.float32),
           "phi90": torch.zeros(len(theta), dtype=torch.float32)}
    results = {
        rows[0]["id"]: _observation_entry(store, rows[0], pattern_a, response_a, rad, cfg),
        rows[1]["id"]: _observation_entry(store, rows[1], pattern_a, response_a, rad, cfg),
    }
    _write(store / "results.json", results)
    before = pilot._metadata_hashes(input_dir, store)
    binding = {"input": str(input_dir.resolve()), "store": str(store.resolve()),
               "input_metadata_sha256": before["input"],
               "store_metadata_sha256": before["store"]}
    return (input_dir, store), binding, rows, pattern_b, response_b, rad, results


def _validated_cutoff(pair, binding, rows, results):
    input_dir, store = pair
    return {
        "input": str(input_dir.resolve()), "store": str(store.resolve()),
        "rows": list(rows), "results": dict(results), "terminal": False,
        "input_metadata_sha256": dict(binding["input_metadata_sha256"]),
        "input_pattern_file_sha256": {
            row["id"]: pb.file_sha256(input_dir / f"{row['id']}.pt") for row in rows},
        "store_metadata_sha256": dict(binding["store_metadata_sha256"]),
    }


def test_reference_prefers_normal_and_accepts_append_only_success(tmp_path):
    request, _ = _request_fixture(tmp_path / "request")
    pair, binding, rows, pattern_b, response_b, rad, results = _pair_fixture(tmp_path / "pair")
    reference_path = pilot.freeze_reference([pair], [binding], request,
                                            tmp_path / "reference.json")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    assert reference["valid_unique"] == 1
    assert reference["observations"][0]["id"] == "normal-second"

    results["pending"] = _observation_entry(pair[1], rows[2], pattern_b, response_b, rad,
                                             request.profile)
    _write(pair[1] / "results.json", results)
    pilot.recheck_reference(reference_path)


def test_reference_validated_cutoff_accepts_append_during_construction_and_excludes_it(
        tmp_path, monkeypatch):
    request, _ = _request_fixture(tmp_path / "request")
    pair, binding, rows, pattern_b, response_b, rad, results = _pair_fixture(
        tmp_path / "pair")
    cutoff = _validated_cutoff(pair, binding, rows, results)
    original = pilot._metric_from_observation
    appended = False

    def metric(store, row, entry, cfg):
        nonlocal appended
        value = original(store, row, entry, cfg)
        if not appended:
            appended = True
            current = dict(results)
            current["pending"] = _observation_entry(
                pair[1], rows[2], pattern_b, response_b, rad, request.profile)
            _write(pair[1] / "results.json", current)
        return value

    monkeypatch.setattr(pilot, "_metric_from_observation", metric)
    reference_path = pilot.freeze_reference(
        [pair], [binding], request, tmp_path / "reference.json",
        validated_cutoff=[cutoff])
    reference = json.loads(reference_path.read_text(encoding="utf-8"))

    assert appended
    assert reference["valid_unique"] == 1
    assert reference["B_factory_margin_db"] == pytest.approx(0.5)
    assert reference["representative_order"] == [rows[1]["pattern_sha256"]]
    assert [row["id"] for row in reference["observations"]] == ["normal-second"]
    assert reference["pairs"][0]["metadata_before"]["store"]["results.json"] != (
        reference["pairs"][0]["metadata_after"]["store"]["results.json"])
    assert reference["validated_cutoff_id"] == pilot._content_id([cutoff])
    pilot.recheck_reference(reference_path)


@pytest.mark.parametrize("mutation", ["schema", "path", "manifest", "profile"])
def test_reference_validated_cutoff_rejects_identity_mismatch(
        tmp_path, monkeypatch, mutation):
    request, _ = _request_fixture(tmp_path / "request")
    pair, binding, rows, _pattern_b, _response_b, _rad, results = _pair_fixture(
        tmp_path / "pair")
    cutoff = _validated_cutoff(pair, binding, rows, results)
    expected = ""
    if mutation == "schema":
        cutoff["unexpected"] = True
        expected = "proof schema"
    elif mutation == "path":
        cutoff["store"] = str((tmp_path / "other-store").resolve())
        expected = "pair path"
    elif mutation == "manifest":
        cutoff["rows"] = [dict(rows[0], id="different-id"), *rows[1:]]
        expected = "manifest differs"
    else:
        original = pb.load_profile_config(pair[0] / "config.yaml")
        wrong = type("WrongProfile", (), vars(original).copy())()
        wrong.scope = "different-scope"
        monkeypatch.setattr(pilot.pb, "load_profile_config", lambda _path: wrong)
        expected = "profile differs"

    with pytest.raises(pilot.PilotError, match=expected):
        pilot.freeze_reference(
            [pair], [binding], request, tmp_path / f"reference-{mutation}.json",
            validated_cutoff=[cutoff])


@pytest.mark.parametrize("mutation", ["entry", "raw"])
def test_reference_validated_cutoff_rejects_frozen_observation_tamper(tmp_path, mutation):
    request, _ = _request_fixture(tmp_path / "request")
    pair, binding, rows, _pattern_b, _response_b, _rad, results = _pair_fixture(
        tmp_path / "pair")
    cutoff = _validated_cutoff(pair, binding, rows, results)
    if mutation == "entry":
        changed = dict(results)
        changed["normal-second"] = {**changed["normal-second"], "time_s": 9.0}
        _write(pair[1] / "results.json", changed)
        expected = "frozen observation entry changed"
    else:
        sample = pair[1] / results["normal-second"]["sample_file"]
        sample.write_bytes(sample.read_bytes() + b"tamper")
        expected = "raw hash changed"

    with pytest.raises((pilot.PilotError, pilot.PilotSnapshotChanged), match=expected):
        pilot.freeze_reference(
            [pair], [binding], request, tmp_path / f"reference-{mutation}.json",
            validated_cutoff=[cutoff])


def test_reference_rejects_frozen_entry_and_raw_mutation(tmp_path):
    request, _ = _request_fixture(tmp_path / "request")
    pair, binding, _rows, _pattern_b, _response_b, _rad, results = _pair_fixture(
        tmp_path / "pair")
    reference_path = pilot.freeze_reference([pair], [binding], request,
                                            tmp_path / "reference.json")
    results["normal-second"] = {**results["normal-second"], "time_s": 9.0}
    _write(pair[1] / "results.json", results)
    with pytest.raises(pilot.PilotError, match="entry changed"):
        pilot.recheck_reference(reference_path)

    pair2, binding2, *_ = _pair_fixture(tmp_path / "pair2")
    reference2 = pilot.freeze_reference([pair2], [binding2], request,
                                        tmp_path / "reference2.json")
    value = json.loads(reference2.read_text(encoding="utf-8"))
    Path(value["observations"][0]["sample_path"]).write_bytes(b"tampered")
    with pytest.raises(pilot.PilotError, match="raw observation changed"):
        pilot.recheck_reference(reference2)


def test_reference_construction_race_has_domain_exception(tmp_path, monkeypatch):
    request, _ = _request_fixture(tmp_path / "request")
    pair, binding, *_ = _pair_fixture(tmp_path / "pair")
    original = pilot._metadata_hashes
    calls = 0

    def changed(input_dir, store):
        nonlocal calls
        calls += 1
        value = original(input_dir, store)
        if calls >= 2:
            value["store"]["results.json"] = "f" * 64
        return value

    monkeypatch.setattr(pilot, "_metadata_hashes", changed)
    with pytest.raises(pilot.PilotSnapshotChanged):
        pilot.freeze_reference([pair], [binding], request, tmp_path / "reference.json")
    assert not (tmp_path / "reference.json").exists()


def _truth_rows(B, bundle_rows, material_shell=False):
    rows = []
    for index, source in enumerate(bundle_rows):
        value = (B + 0.4 if material_shell and index == 0 and
                 source["pilot_role"] == pilot.SHELL_ROLE else B - 4.0)
        if source["pilot_role"] == pilot.CONTROL_ROLE:
            value = B + 1.0
        rows.append({
            "id": source["id"], "pattern_sha256": source["pattern_sha256"],
            "pilot_role": source["pilot_role"], "s11_margin_db": value,
            "gain_margin_db": value, "factory_margin_db": value,
            "radiation_margin_db": 0.0, "radiation_margin_clipped_db": 0.0,
        })
    return rows


def test_readout_is_reference_only_requires_complete_and_excludes_control(tmp_path):
    request, _ = _request_fixture(tmp_path / "request")
    selection = pilot.select_rows(request, {request.anchor["pattern_sha256"]})
    annotated = pilot.annotate_selection(request, selection, FakePredictor(request.profile))
    bundle = pilot.write_pilot_bundle(request, annotated, tmp_path / "bundle")
    bundle_validation = pilot.validate_bundle(bundle, request)
    bundle_rows = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    pair, binding, *_ = _pair_fixture(tmp_path / "pair")
    reference_path = pilot.freeze_reference([pair], [binding], request,
                                            tmp_path / "reference.json")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    B = reference["B_factory_margin_db"]
    incomplete = pilot.readout({"pilot_id": request.pilot_id,
                                "bundle_validation": bundle_validation,
                                "rows": [], "all_terminal": False,
                                "marker_errors": 0, "result_errors": 0}, reference)
    assert incomplete["status"] == "incomplete"

    red = pilot.readout({"pilot_id": request.pilot_id,
                         "bundle_validation": bundle_validation,
                         "rows": _truth_rows(B, bundle_rows),
                         "all_terminal": True,
                         "marker_errors": 0, "result_errors": 0}, reference)
    assert red["status"] == "shell_red"
    assert red["blind_control"]["material"] is True
    assert red["blind_control_can_make_shell_positive"] is False
    assert red["shell_zero_relevant_is_red"] is True

    positive = pilot.readout({"pilot_id": request.pilot_id,
                              "bundle_validation": bundle_validation,
                              "rows": _truth_rows(B, bundle_rows, material_shell=True),
                              "all_terminal": True, "marker_errors": 0,
                              "result_errors": 0}, reference)
    assert positive["status"] == "shell_positive"
    assert positive["reference_only"] is True and positive["campaign_first_claim"] is False

    wrong = _truth_rows(B, bundle_rows)
    wrong[0]["pattern_sha256"] = "f" * 64
    with pytest.raises(pilot.PilotError, match="validated pilot bundle"):
        pilot.readout({"pilot_id": request.pilot_id,
                       "bundle_validation": bundle_validation,
                       "rows": wrong, "all_terminal": True,
                       "marker_errors": 0, "result_errors": 0}, reference)
