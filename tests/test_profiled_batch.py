"""具名 measurement 批次沿用 dedust lifecycle 的離線／假 COM 整合測試。"""
import argparse
import json
from copy import deepcopy
from pathlib import Path as StdPath

import numpy as np
import pytest
import torch
import yaml

from antenna.measurement import measurement_id, score_spec_id
from antenna.utils.utils import Path
from script import dedust, profiled_batch


REPO = StdPath(__file__).resolve().parents[1]
CONFIGS = {
    "single": REPO / "configs" / "single_r80_symmetry_explore.yaml",
    "dual": REPO / "configs" / "dual_r81_wide_filter.yaml",
}


def _config(port):
    return yaml.safe_load(CONFIGS[port].read_text(encoding="utf-8"))


def _pattern(port):
    pattern = np.zeros((25, 25), dtype=np.float32)
    if port == "single":
        pattern[24, 12] = 1
    else:
        pattern[:5, 10:15] = 1
        pattern[20:, 10:15] = 1
    return torch.tensor(pattern)


def _write_input(dataset_root, name, cfg_data, *, pid="p0"):
    folder = StdPath(dataset_root) / name
    folder.mkdir(parents=True)
    measurement = profiled_batch.validate_measurement(cfg_data["measurement"])
    score = profiled_batch.validate_score_spec(cfg_data["score_spec"], measurement=measurement)
    pattern = _pattern(cfg_data["port"])
    row = {
        "id": pid,
        "port": cfg_data["port"],
        "kind": "profile",
        "selection_arm": "random",
        "lineage_id": pid,
        "measurement_id": measurement_id(measurement),
        "score_spec_id": score_spec_id(score),
        "pattern_sha256": profiled_batch.pattern_sha256(pattern),
    }
    profiled_batch.atomic_json(folder / "measurement.json", measurement)
    profiled_batch.atomic_json(folder / "score_spec.json", score)
    profiled_batch.atomic_json(folder / "manifest.json", [row])
    torch.save(pattern, folder / f"{pid}.pt")
    return folder


def _run_args(input_name, store, config, out):
    return argparse.Namespace(
        input=input_name, store=store, config=str(config), out=str(out), sweep=None,
        timeout=17, max_fail=2, cooldown=0, max_blowout=1, retry_pass=0,
        job_prio=9, tier2_prio=8, scope="symmetry_filter_20261007",
        claim_path=None, claim_me=None,
    )


class _FakeBase:
    opens = 0
    calls = 0
    kwargs_seen = []

    @classmethod
    def reset(cls):
        cls.opens = cls.calls = 0
        cls.kwargs_seen = []

    def __init__(self, record_path, sweep_type, **kwargs):
        type(self).kwargs_seen.append((record_path, sweep_type, kwargs))
        self.last_radiation = None

    def open(self):
        type(self).opens += 1

    def start(self, number):
        self.number = number

    def end(self):
        return 2.5

    def quit(self):
        return None


class _FakeDual(_FakeBase):
    def __call__(self, _pattern_tensor):
        type(self).calls += 1
        freqs = np.linspace(16, 40, 49)
        s21 = np.full(49, -25.0, dtype=np.float32)
        s21[(freqs >= 26) & (freqs <= 30)] = -2.0
        return {"S11": torch.full((49,), -12.0),
                "S21": torch.tensor(s21),
                "S22": torch.full((49,), -12.0)}


class _FakeSingle(_FakeBase):
    def __call__(self, _pattern_tensor):
        type(self).calls += 1
        theta = np.arange(-180, 181, 2, dtype=float)
        self.last_radiation = {"theta": theta, "phi0": np.zeros(181), "phi90": np.ones(181)}
        return {"S11": torch.full((17,), -12.0), "Gain": torch.full((17,), 5.0)}


class _FakeSingleMissingRad(_FakeBase):
    def __call__(self, _pattern_tensor):
        type(self).calls += 1
        return {"S11": torch.full((17,), -12.0), "Gain": torch.full((17,), 5.0)}


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    root.mkdir()
    monkeypatch.setattr(dedust, "DATASET_PATH", Path(str(root)))
    return root


def test_dual_profile_full_run_resume_scope_filter_and_rescore(dataset, tmp_path, monkeypatch):
    cfg_data = _config("dual")
    _write_input(dataset, "wide_input", cfg_data)
    _FakeDual.reset()
    monkeypatch.setattr("antenna.patch.DualPortSimulator", _FakeDual)
    # 不同 scope 的 tier-1 不得讓本 scope 的 tier-2 誤讓位。
    (dataset / "jobs.json").write_text(json.dumps([
        {"input": "foreign_input", "store": "foreign", "prio": 1, "scope": "foreign_scope"}
    ]), encoding="utf-8")
    args = _run_args("wide_input", "wide_store", CONFIGS["dual"], tmp_path / "work-dual")
    assert dedust.run(args) is None
    assert _FakeDual.opens == 1 and _FakeDual.calls == 1
    _record, sweep_type, kwargs = _FakeDual.kwargs_seen[0]
    assert sweep_type == "Fast"
    assert (kwargs["sweep_start"], kwargs["sweep_end"], kwargs["sweep_step"]) == (16.0, 40.0, 0.5)
    assert (kwargs["setup_frequency"], kwargs["open_region_frequency"]) == (28.0, 16.0)

    store = dataset / "wide_store"
    results = json.loads((store / "results.json").read_text(encoding="utf-8"))
    entry = results["p0"]
    assert entry["status"] == "ok" and "wm" not in entry
    assert entry["all_passed"] is True and entry["worst_margin"] == pytest.approx(1.0)
    assert profiled_batch.file_sha256(store / entry["sample_file"]) == entry["sample_sha256"]
    pattern, response = torch.load(store / entry["sample_file"], weights_only=True)
    assert pattern.shape == (25, 25) and response.shape == (3, 49)
    assert len(profiled_batch.validate_store(store)) == 1

    # 完整續跑只做離線 hash/replay 驗證，不開 fake HFSS。
    assert dedust.run(args) is None
    assert _FakeDual.opens == 1 and _FakeDual.calls == 1

    changed = deepcopy(cfg_data["score_spec"])
    changed["bands"][2]["threshold"] = -1.0
    changed_path = tmp_path / "changed_score.json"
    profiled_batch.atomic_json(changed_path, changed)
    original_results = (store / "results.json").read_bytes()
    original_score = (store / "score_spec.json").read_bytes()
    summary = profiled_batch.rescore_store(store, changed_path, tmp_path / "rescore.json")
    assert summary["rows"]["p0"]["all_passed"] is False
    assert (store / "results.json").read_bytes() == original_results
    assert (store / "score_spec.json").read_bytes() == original_score
    with pytest.raises(ValueError, match="store 外"):
        profiled_batch.rescore_store(store, changed_path, store / "new_summary.json")
    occupied = tmp_path / "occupied.json"
    occupied.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="不存在"):
        profiled_batch.rescore_store(store, changed_path, occupied)
    assert occupied.read_text(encoding="utf-8") == "keep"


def test_single_profile_never_loses_radiation_sidecar(dataset, tmp_path, monkeypatch):
    cfg_data = _config("single")
    _write_input(dataset, "single_input", cfg_data)
    _FakeSingle.reset()
    monkeypatch.setattr("antenna.patch.SinglePortRadSimulator", _FakeSingle)
    args = _run_args("single_input", "single_store", CONFIGS["single"], tmp_path / "work-single")
    dedust.run(args)
    store = dataset / "single_store"
    entry = json.loads((store / "results.json").read_text(encoding="utf-8"))["p0"]
    assert entry["rad_file"] == "rad/p0.pt"
    assert profiled_batch.file_sha256(store / entry["rad_file"]) == entry["rad_sha256"]
    assert entry["performance_gate"] is False and "wm" not in entry
    assert len(profiled_batch.validate_store(store)) == 1
    (store / entry["rad_file"]).unlink()
    with pytest.raises((ValueError, FileNotFoundError), match="場型|找不到|No such file"):
        profiled_batch.validate_store(store)


def test_profile_cli_sweep_conflict_fails_before_store_or_hfss(dataset, tmp_path, monkeypatch):
    _write_input(dataset, "wide_input", _config("dual"))
    _FakeDual.reset()
    monkeypatch.setattr("antenna.patch.DualPortSimulator", _FakeDual)
    args = _run_args("wide_input", "never_store", CONFIGS["dual"], tmp_path / "never-work")
    args.sweep = "Discrete"
    with pytest.raises(SystemExit, match="不可用 CLI"):
        dedust.run(args)
    assert not (dataset / "never_store").exists() and _FakeDual.opens == 0


def test_invalid_profile_observation_is_recorded_and_never_completes(dataset, tmp_path, monkeypatch):
    _write_input(dataset, "single_input", _config("single"))
    _FakeSingleMissingRad.reset()
    monkeypatch.setattr("antenna.patch.SinglePortRadSimulator", _FakeSingleMissingRad)
    args = _run_args("single_input", "bad_store", CONFIGS["single"], tmp_path / "bad-work")
    with pytest.raises(SystemExit, match="完整性驗證失敗"):
        dedust.run(args)
    entry = json.loads((dataset / "bad_store" / "results.json").read_text(encoding="utf-8"))["p0"]
    assert entry["attempts"] == 1 and entry["error"].startswith("profile_observation_invalid:")
    assert "status" not in entry


def test_jobs_add_snapshots_portable_profile_and_validates_scope(dataset):
    _write_input(dataset, "wide_input", _config("dual"))
    args = argparse.Namespace(input="wide_input", store="wide_store", prio=3, machine=None,
                              config=str(CONFIGS["dual"]), scope=None)
    dedust.jobs_add(args)
    jobs = json.loads((dataset / "jobs.json").read_text(encoding="utf-8"))
    assert jobs == [{"input": "wide_input", "store": "wide_store", "prio": 3,
                     "scope": "symmetry_filter_20261007", "config": "wide_input/config.yaml"}]
    assert (dataset / "wide_input" / "config.yaml").read_bytes() == CONFIGS["dual"].read_bytes()
    bad = argparse.Namespace(input="wide_input", store="other", prio=3, machine=None,
                             config=str(CONFIGS["dual"]), scope="wrong_scope")
    with pytest.raises(ValueError, match="scope"):
        dedust.jobs_add(bad)


def test_worker_never_marks_profile_done_when_results_missing(dataset, monkeypatch):
    _write_input(dataset, "wide_input", _config("dual"))
    dedust.jobs_add(argparse.Namespace(input="wide_input", store="wide_store", prio=3, machine=None,
                                       config=str(CONFIGS["dual"]), scope=None))
    captured = {}
    def fake_run(run_args):
        captured["args"] = run_args
    monkeypatch.setattr(dedust, "run", fake_run)
    monkeypatch.setattr("antenna.utils.web.get_local_ip", lambda: "127.0.0.81")
    args = argparse.Namespace(scope="symmetry_filter_20261007", selfgen=0, poll=0, timeout=10,
                              stale=120, once=True, config=str(CONFIGS["single"]), sweep=None,
                              max_fail=1, cooldown=0, max_blowout=1, retry_pass=0, tier2_prio=8,
                              selfgen_port="dual", selfgen_config=None)
    with pytest.raises(SystemExit, match="禁止 .done"):
        dedust.worker(args)
    state = dataset / "jobs_state"
    assert not (state / "wide_store.done").exists()
    assert (state / "wide_store.fail").exists()
    assert captured["args"].scope == "symmetry_filter_20261007"
    assert captured["args"].config == str(dataset / "wide_input" / "config.yaml")


def test_profile_checkdup_is_identity_aware_and_legacy_ignores_marker(dataset):
    cfg = _config("dual")
    _write_input(dataset, "a_input", cfg, pid="a")
    _write_input(dataset, "b_input", cfg, pid="b")
    with pytest.raises(ValueError, match="同量測條件下有重複"):
        dedust.check_dup(argparse.Namespace(input="b_input"))

    different = deepcopy(cfg)
    different["measurement"]["name"] += "_other_identity"
    _write_input(dataset, "c_input", different, pid="c")
    dedust.check_dup(argparse.Namespace(input="c_input"))

    legacy = dataset / "legacy_input"
    legacy.mkdir()
    torch.save(_pattern("dual"), legacy / "legacy.pt")
    profiled_batch.atomic_json(legacy / "manifest.json", [{"id": "legacy", "port": "dual", "kind": "normal"}])
    dedust.check_dup(argparse.Namespace(input="legacy_input"))


def test_jobs_ls_and_report_understand_profile_status(dataset, capsys):
    cfg = _config("dual")
    folder = _write_input(dataset, "wide_input", cfg)
    store = dataset / "wide_store"
    store.mkdir()
    for name in ("measurement.json", "score_spec.json", "manifest.json"):
        (store / name).write_bytes((folder / name).read_bytes())
    profiled_batch.atomic_json(store / "results.json", {"p0": {"status": "ok", "margins": {"x": 1.0}}})
    profiled_batch.atomic_json(dataset / "jobs.json", [{
        "input": "wide_input", "store": "wide_store", "prio": 3, "scope": "symmetry_filter_20261007"}])
    dedust.jobs_ls(argparse.Namespace(scope="symmetry_filter_20261007", all=True))
    assert "1/1" in capsys.readouterr().out
    dedust.report(argparse.Namespace(input="ignored", store="wide_store"))
    output = capsys.readouterr().out
    assert "observations" in output and "p0" in output and "{'x': 1.0}" in output


def test_legacy_dual_loader_explicitly_skips_measurement_store(dataset, monkeypatch):
    store = dataset / "profile_store"
    store.mkdir()
    (store / "measurement.json").write_text("{}", encoding="utf-8")
    torch.save((_pattern("dual"), torch.zeros(3, 49)), store / "sample.pt")
    from script import sm_dual
    monkeypatch.setattr(sm_dual, "DATASET_PATH", Path(str(dataset)))
    x, y, source, raw = sm_dual._scan_stores(["profile_store"], workers=1)
    assert x.shape == (0, 625) and y.shape == (0, 3, 17) and source == [] and raw == 0
