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
