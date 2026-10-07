"""Bounded measurement gate: successful files must still agree numerically."""
import json
import pytest
import torch

from script import prepare_symmetry_filter as prep


def _fixture(tmp_path, monkeypatch, delta):
    monkeypatch.setattr(prep.pb, "load_profile_config", lambda p: object())
    monkeypatch.setattr(prep.pb, "verify_completed", lambda *args: None)
    for name, offset in (("smoke", 0), ("discrete", delta), ("mesh", delta)):
        store = tmp_path / ("dedust_r81" + name)
        store.mkdir()
        manifest, results = [], {}
        for i in range(6 if name == "smoke" else 2):
            key = f"p{i}"
            manifest.append({"id": key, "parent_batch_id": key})
            torch.save((torch.ones(25, 25), torch.zeros(3, 49) + offset), store / (key + ".pt"))
            results[key] = {"status": "ok", "sample_file": key + ".pt",
                            "margins": {f"m{j}": offset for j in range(5)}}
        (store / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (store / "results.json").write_text(json.dumps(results), encoding="utf-8")


def test_gate_rejects_numerical_disagreement_even_after_artifact_validation(tmp_path, monkeypatch):
    _fixture(tmp_path, monkeypatch, .31)
    out = prep.measurement_check(tmp_path)
    assert out["full_batch_release_allowed"] is False
    assert len(out["comparisons"]) == 4


def test_gate_accepts_bounded_agreement_and_rejects_missing_sample(tmp_path, monkeypatch):
    _fixture(tmp_path, monkeypatch, .1)
    assert prep.measurement_check(tmp_path)["engineering_check_passed"]
    path = tmp_path / "dedust_r81mesh/results.json"
    values = json.loads(path.read_text(encoding="utf-8"))
    values["p0"] = {"error": "timeout"}
    path.write_text(json.dumps(values), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        prep.measurement_check(tmp_path)


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
