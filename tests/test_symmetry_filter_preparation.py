"""Bounded measurement gate: successful files must still agree numerically."""
import json
import numpy as np
import pytest
import torch
import yaml

from antenna.measurement import measurement_id, score_spec_id

from script import prepare_symmetry_filter as prep


def _fixture(tmp_path, delta):
    configs = prep.wide_check_configs()
    formal = tmp_path / "dedust_r81b1_input"
    formal.mkdir()
    (formal / "config.yaml").write_text(yaml.safe_dump(configs["smoke"]), encoding="utf-8")
    cfg = prep.pb.load_profile_config(formal / "config.yaml")
    rows = []
    for i in range(60):
        pattern = prep.ex.enforce_dual_geometry(np.random.default_rng(i).random((25, 25)))
        key = f"p{i}"
        torch.save(torch.tensor(pattern, dtype=torch.float32), formal / (key + ".pt"))
        rows.append({"id": key, "pattern_file": key + ".pt", "port": "dual",
                     "pattern_sha256": prep.pb.pattern_sha256(pattern),
                     "measurement_id": measurement_id(cfg.measurement),
                     "score_spec_id": score_spec_id(cfg.score_spec),
                     "selection_arm": ("history", "specialist", "random")[i // 20]})
    for name, value in (("measurement.json", cfg.measurement), ("score_spec.json", cfg.score_spec),
                        ("manifest.json", rows)):
        prep.pb.atomic_json(formal / name, value)
    for name, offset in (("smoke", 0), ("discrete", delta), ("mesh", delta)):
        folder = tmp_path / ("dedust_r81" + name + "_input")
        indices = [0, 1, 20, 21, 40, 41] if name == "smoke" else [0, 1]
        prep.auxiliary_bundle(formal, folder, indices, config=configs[name])
        cfg = prep.pb.load_profile_config(folder / "config.yaml")
        store = tmp_path / ("dedust_r81" + name)
        manifest = prep.pb.prepare_store(folder, store, cfg)
        results = {}
        for row in manifest:
            pattern = torch.load(folder / row["pattern_file"], weights_only=True)
            response = torch.full((3, 49), -9.0) + offset
            entry = prep.pb.observation(response, None, row, pattern, cfg, elapsed=1.0)
            sample = store / entry["sample_file"]
            torch.save((pattern, response), sample)
            entry["sample_sha256"] = prep.pb.file_sha256(sample)
            results[row["id"]] = entry
        prep.pb.atomic_json(store / "results.json", results)


def test_gate_rejects_numerical_disagreement_even_after_artifact_validation(tmp_path):
    _fixture(tmp_path, .31)
    out = prep.measurement_check(tmp_path)
    assert out["full_batch_release_allowed"] is False
    assert len(out["comparisons"]) == 4


def test_gate_accepts_bounded_agreement_and_rejects_missing_sample(tmp_path):
    _fixture(tmp_path, .1)
    report = prep.measurement_check(tmp_path)
    assert report["engineering_check_passed"]
    assert not report["performance_target_assessed"] and not report["campaign_complete"]
    assert len(report["metadata_sha256"]) == 29
    path = tmp_path / "dedust_r81mesh/results.json"
    values = json.loads(path.read_text(encoding="utf-8"))
    values[next(iter(values))] = {"error": "timeout"}
    path.write_text(json.dumps(values), encoding="utf-8")
    with pytest.raises(ValueError, match="未完整成功"):
        prep.measurement_check(tmp_path)


@pytest.mark.parametrize("instrument", ["discrete", "mesh", "timeout"])
def test_gate_rejects_same_instrument_disguised_as_independent_check(tmp_path, instrument):
    _fixture(tmp_path, .1)
    name = "mesh" if instrument == "timeout" else instrument
    path = tmp_path / f"dedust_r81{name}_input/config.yaml"
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if instrument == "discrete":
        value["measurement"]["sweep"]["type"] = "Fast"
    elif instrument == "mesh":
        value["measurement"]["solver"]["max_delta_s"] = .02
    else:
        value["runtime"]["timeout"] = 1800
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    with pytest.raises(ValueError, match="instrument differs"):
        prep.measurement_check(tmp_path)


def test_gate_rejects_duplicate_parent_instead_of_collapsing_evidence(tmp_path):
    _fixture(tmp_path, .1)
    path = tmp_path / "dedust_r81mesh_input/manifest.json"
    rows = prep.pb.read_json(path)
    rows[1]["parent_batch_id"] = rows[0]["parent_batch_id"]
    prep.pb.atomic_json(path, rows)
    with pytest.raises(ValueError, match="distinct known parent"):
        prep.measurement_check(tmp_path)


def test_preflight_rejects_missing_smoke_stratum_before_hfss(tmp_path):
    _fixture(tmp_path, .1)
    folder = tmp_path / "dedust_r81smoke_input"
    rows = prep.pb.read_json(folder / "manifest.json")
    formal = tmp_path / "dedust_r81b1_input"
    # Replace one random representative with a third history representative.
    parent = prep.pb.read_json(formal / "manifest.json")[2]
    rows[-1].update(parent_batch_id=parent["id"], pattern_sha256=parent["pattern_sha256"])
    import shutil
    shutil.copyfile(formal / parent["pattern_file"], folder / rows[-1]["pattern_file"])
    prep.pb.atomic_json(folder / "manifest.json", rows)
    with pytest.raises(ValueError, match="two representatives"):
        prep.validate_wide_inputs(tmp_path)


@pytest.mark.parametrize("corruption", ["profile", "curve"])
def test_gate_rejects_tampered_curve_and_store_profile(tmp_path, corruption):
    _fixture(tmp_path, .1)
    if corruption == "profile":
        path = tmp_path / "dedust_r81mesh/measurement.json"
        measurement = prep.pb.read_json(path)
        measurement["solver"]["max_delta_s"] = .02
        prep.pb.atomic_json(path, measurement)
    else:
        store = tmp_path / "dedust_r81mesh"
        entry = next(iter(prep.pb.read_json(store / "results.json").values()))
        (store / entry["sample_file"]).write_bytes(b"damaged")
    with pytest.raises(ValueError, match="measurement_id|hash"):
        prep.measurement_check(tmp_path)


def test_preflight_requires_mesh_and_discrete_to_remeasure_same_smoke_parents(tmp_path):
    _fixture(tmp_path, .1)
    folder = tmp_path / "dedust_r81mesh_input"
    rows = prep.pb.read_json(folder / "manifest.json")
    formal = tmp_path / "dedust_r81b1_input"
    parent = prep.pb.read_json(formal / "manifest.json")[20]
    rows[-1].update(parent_batch_id=parent["id"], pattern_sha256=parent["pattern_sha256"])
    import shutil
    shutil.copyfile(formal / parent["pattern_file"], folder / rows[-1]["pattern_file"])
    prep.pb.atomic_json(folder / "manifest.json", rows)
    with pytest.raises(ValueError, match="share two smoke parents"):
        prep.validate_wide_inputs(tmp_path)


def test_single_phase_reads_only_r55_and_prepares_three_jobs_52_measurements(tmp_path, monkeypatch):
    history = tmp_path / "history"
    source = history / "dedust_r55sym_input"
    source.mkdir(parents=True)
    source_rows = []
    for i in range(34):
        key = f"r55_{i:02d}"
        source_rows.append({"id": key})
        torch.save(torch.zeros(25, 25), source / f"{key}.pt")
    (source / "manifest.json").write_text(json.dumps(source_rows), encoding="utf-8")

    calls = []

    def fake_prepare_run(config, dataset, work, seed_inputs):
        calls.append((config.name, work.name))
        assert len(prep.pb.read_json(seed_inputs[0] / "manifest.json")) == 34
        bundle = work / "batch_001_input"
        bundle.mkdir(parents=True)
        (bundle / "config.yaml").write_text("name: fixture\n", encoding="utf-8")
        rows = []
        for i in range(48):
            name = f"single_{i:02d}.pt"
            torch.save(torch.zeros(25, 25), bundle / name)
            rows.append({"id": f"single_{i:02d}", "pattern_file": name,
                         "pattern_sha256": f"{i:064x}"})
        (bundle / "manifest.json").write_text(json.dumps(rows), encoding="utf-8")
        return bundle

    def fake_auxiliary(source_bundle, output, indices, **_kwargs):
        output.mkdir(parents=True)
        rows = prep.pb.read_json(source_bundle / "manifest.json")
        selected = [{"id": f"repeat_{i}", "parent_batch_id": rows[idx]["id"]}
                    for i, idx in enumerate(indices)]
        (output / "manifest.json").write_text(json.dumps(selected), encoding="utf-8")
        return output

    monkeypatch.setattr(prep.ex, "prepare_run", fake_prepare_run)
    monkeypatch.setattr(prep.ex, "maxmin_hamming_indices", lambda _patterns, _n: [0, 47])
    monkeypatch.setattr(prep.pb, "load_profile_config", lambda _path: object())
    monkeypatch.setattr(prep.pb, "validate_input", lambda *_args: None)
    monkeypatch.setattr(prep, "auxiliary_bundle", fake_auxiliary)
    monkeypatch.setattr(
        prep, "dual_history_seeds",
        lambda _root: (_ for _ in ()).throw(AssertionError("single phase read dual history")))

    out = prep.REPO / "tmp" / f"pytest_single_{tmp_path.name}"
    try:
        prep.prepare(history, out)
        receipt = prep.pb.read_json(out / "preparation.json")
        assert calls == [("single_r80_symmetry_explore.yaml", "single")]
        assert receipt["phase"] == "single"
        assert receipt["prepared_jobs"] == [
            "dedust_r80b1", "dedust_r80repeat1", "dedust_r80repeat2"]
        assert receipt["prepared_measurements"] == 52
        assert receipt["dual_history_read"] is False
        assert receipt["dual_selected"] == 0
        assert receipt["deferred"] == ["R81 wide-filter profile"]
        assert not (out / "dual").exists()
        assert not (out / "dual_seeds").exists()
        assert not any((out / "dataset").glob("dedust_r81*_input"))
    finally:
        if out.exists():
            import shutil
            out.resolve().relative_to((prep.REPO / "tmp").resolve())
            shutil.rmtree(out)


def test_package_inputs_are_limited_to_receipt_list():
    receipt = {
        "prepared_jobs": ["dedust_r80b1", "dedust_r80repeat1", "dedust_r80repeat2"],
        "prepared_inputs": [
            "dedust_r80b1_input", "dedust_r80repeat1_input", "dedust_r80repeat2_input"],
    }
    jobs = [
        {"store": name.removesuffix("_input"), "input": name}
        for name in receipt["prepared_inputs"]
    ]
    assert prep.package_input_names(receipt, jobs) == sorted(receipt["prepared_inputs"])
    jobs[0]["input"] = "dedust_r81b1_input"
    with pytest.raises(ValueError, match="prepared input list"):
        prep.package_input_names(receipt, jobs)
