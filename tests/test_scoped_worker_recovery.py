import argparse
import json
from pathlib import Path as StdPath

import numpy as np
import pytest
import torch

from antenna.measurement import measurement_id, score_spec_id
from antenna.utils.store import SampleStore
from antenna.utils.utils import Path
from script import dedust, profiled_batch


REPO = StdPath(__file__).resolve().parents[1]
PROFILE = REPO / "tests" / "fixtures" / "profile_single.yaml"
SCOPE = "symmetry_filter_20261007"


def _pattern(index):
    pattern = torch.zeros((25, 25), dtype=torch.float32)
    pattern[24, 12] = 1
    if index:
        pattern[20 - index, 12 - index] = 1
        pattern[20 - index, 12 + index] = 1
    return pattern


def _write_input(dataset, name, ids):
    cfg = profiled_batch.load_profile_config(PROFILE)
    root = StdPath(dataset) / name
    root.mkdir(parents=True)
    rows = []
    for index, sample_id in enumerate(ids):
        pattern = _pattern(index)
        rows.append({
            "id": sample_id,
            "port": "single",
            "kind": "profile",
            "selection_arm": "random",
            "lineage_id": sample_id,
            "measurement_id": measurement_id(cfg.measurement),
            "score_spec_id": score_spec_id(cfg.score_spec),
            "pattern_sha256": profiled_batch.pattern_sha256(pattern),
        })
        torch.save(pattern, root / f"{sample_id}.pt")
    profiled_batch.atomic_json(root / "measurement.json", cfg.measurement)
    profiled_batch.atomic_json(root / "score_spec.json", cfg.score_spec)
    profiled_batch.atomic_json(root / "manifest.json", rows)
    (root / "config.yaml").write_bytes(PROFILE.read_bytes())
    return root, rows, cfg


def _ok_entry(store_root, row, pattern, cfg, value=0.0):
    response = torch.stack((torch.full((17,), -12.0 + value), torch.full((17,), 5.0 + value)))
    theta = torch.arange(-180, 181, 2, dtype=torch.float64)
    rad = {"theta": theta, "phi0": torch.zeros(181), "phi90": torch.ones(181)}
    entry = profiled_batch.observation(response, rad, row, pattern, cfg, 2.5)
    SampleStore(Path(str(store_root))).add(pattern, response)
    entry["sample_sha256"] = profiled_batch.file_sha256(StdPath(store_root) / entry["sample_file"])
    rad_dir = StdPath(store_root) / "rad"
    rad_dir.mkdir(exist_ok=True)
    torch.save(rad, rad_dir / f"{row['id']}.pt")
    entry["rad_sha256"] = profiled_batch.file_sha256(rad_dir / f"{row['id']}.pt")
    return entry


def _write_store(dataset, input_root, store_name, rows, cfg, *, errors=None):
    store_root = StdPath(dataset) / store_name
    profiled_batch.prepare_store(input_root, store_root, cfg)
    errors = errors or {}
    results = {}
    for index, row in enumerate(rows):
        if row["id"] in errors:
            results[row["id"]] = errors[row["id"]]
        else:
            results[row["id"]] = _ok_entry(store_root, row, _pattern(index), cfg, float(index))
    profiled_batch.atomic_json(store_root / "results.json", results)
    return store_root


def _worker_args(scope=SCOPE, *, once=False):
    return argparse.Namespace(
        scope=scope, selfgen=0, once=once, poll=0, timeout=900, stale=120,
        config=str(PROFILE), sweep=None, max_fail=5, cooldown=0, max_blowout=1,
        retry_pass=2, tier2_prio=8, selfgen_port="dual", selfgen_config=None,
    )


class _OneGoodOneBadSimulator:
    def __init__(self, record_path, sweep_type, **kwargs):
        self.last_radiation = None

    def open(self):
        return None

    def start(self, number):
        self.number = number

    def __call__(self, _pattern_tensor):
        if self.number == 1:
            raise RuntimeError("COM fixture failure")
        self.last_radiation = {
            "theta": np.arange(-180, 181, 2, dtype=float),
            "phi0": np.zeros(181),
            "phi90": np.ones(181),
        }
        return {"S11": torch.full((17,), -12.0), "Gain": torch.full((17,), 5.0)}

    def end(self):
        return 2.5

    def quit(self):
        return None


class _SuccessfulSimulator(_OneGoodOneBadSimulator):
    def __call__(self, _pattern_tensor):
        self.last_radiation = {
            "theta": np.arange(-180, 181, 2, dtype=float),
            "phi0": np.zeros(181),
            "phi90": np.ones(181),
        }
        return {"S11": torch.full((17,), -12.0), "Gain": torch.full((17,), 5.0)}


class _WorkdirOneGoodOneBadSimulator(_OneGoodOneBadSimulator):
    def __init__(self, record_path, sweep_type, **kwargs):
        super().__init__(record_path, sweep_type, **kwargs)
        StdPath(record_path).mkdir(parents=True, exist_ok=True)
        (StdPath(record_path) / "solver.tmp").write_text("current job", encoding="utf-8")


def test_run_skips_exhausted_hfss_row_and_keeps_strict_truth_incomplete(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    monkeypatch.setattr(dedust, "DATASET_PATH", Path(str(dataset)))
    monkeypatch.setattr("antenna.patch.SinglePortRadSimulator", _OneGoodOneBadSimulator)
    monkeypatch.setattr("script.kill.kill", lambda: None)
    _write_input(dataset, "partial_input", ["ok", "bad"])
    args = argparse.Namespace(
        input="partial_input", store="partial", config=str(PROFILE), out=str(tmp_path / "work"),
        sweep=None, timeout=10, max_fail=5, cooldown=0, max_blowout=1, retry_pass=2,
        job_prio=1, tier2_prio=8, scope=SCOPE, claim_path=None, claim_me=None,
    )

    assert dedust.run(args) is None

    results = json.loads((dataset / "partial" / "results.json").read_text(encoding="utf-8"))
    assert results["bad"] == {
        "error": "COM fixture failure", "error_kind": "hfss_simulation", "attempts": 3,
        "status": "skipped", "skip_reason": "hfss_attempts_exhausted",
    }
    assert profiled_batch.verify_finished(dataset / "partial",
                                          profiled_batch.load_profile_config(PROFILE)) == {
        "success_ids": ["ok"], "skipped_ids": ["bad"],
    }
    with pytest.raises(ValueError, match="未完整成功"):
        profiled_batch.validate_store(dataset / "partial")


def test_partial_run_cleans_only_current_default_workdir_after_raw_verification(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dedust, "DATASET_PATH", Path(str(dataset)))
    monkeypatch.setattr("antenna.patch.SinglePortRadSimulator", _WorkdirOneGoodOneBadSimulator)
    monkeypatch.setattr("script.kill.kill", lambda: None)
    _write_input(dataset, "partial_input", ["ok", "bad"])
    unrelated = tmp_path / "_dedust_other"
    unrelated.mkdir()
    (unrelated / "keep.txt").write_text("unrelated", encoding="utf-8")
    args = argparse.Namespace(
        input="partial_input", store="partial", config=str(PROFILE), out=None,
        sweep=None, timeout=10, max_fail=5, cooldown=0, max_blowout=1, retry_pass=2,
        job_prio=1, tier2_prio=8, scope=SCOPE, claim_path=None, claim_me=None,
    )

    assert dedust.run(args) is None

    assert not (tmp_path / "_dedust_partial").exists()
    assert (unrelated / "keep.txt").read_text(encoding="utf-8") == "unrelated"
    assert profiled_batch.verify_finished(dataset / "partial",
                                          profiled_batch.load_profile_config(PROFILE)) == {
        "success_ids": ["ok"], "skipped_ids": ["bad"],
    }

    custom_input, _, _ = _write_input(dataset, "custom_input", ["ok2", "bad2"])
    assert custom_input.exists()
    custom_out = tmp_path / "user-output"
    args.input, args.store, args.out = "custom_input", "custom", str(custom_out)
    assert dedust.run(args) is None
    assert (custom_out / "solver.tmp").read_text(encoding="utf-8") == "current job"


def test_fail_takeover_skips_attempt_three_without_opening_hfss(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dedust, "DATASET_PATH", Path(str(dataset)))
    monkeypatch.setattr("antenna.utils.web.get_local_ip", lambda: "140.123.106.216")
    monkeypatch.setattr("antenna.patch.SinglePortRadSimulator",
                        lambda **_kw: pytest.fail("exhausted row must not open HFSS"))
    input_root, rows, cfg = _write_input(dataset, "retry_input", ["retry"])
    store = _write_store(dataset, input_root, "retry_store", rows, cfg, errors={
        "retry": {"error": "DISP_E_EXCEPTION 0x80070223", "attempts": 3},
    })
    profiled_batch.atomic_json(dataset / "jobs.json", [{
        "input": "retry_input", "store": "retry_store", "prio": 1, "scope": SCOPE,
        "config": "retry_input/config.yaml",
    }])
    state = dataset / "jobs_state"
    state.mkdir()
    profiled_batch.atomic_json(state / "retry_store.fail", {
        "machines": ["140.123.106.37"], "last": "old partial",
    })
    profiled_batch.atomic_json(state / "retry_store.claim", {"machine": "140.123.106.37"})

    dedust.worker(_worker_args(once=True))

    results = json.loads((store / "results.json").read_text(encoding="utf-8"))
    claim = json.loads((state / "retry_store.claim").read_text(encoding="utf-8"))
    assert results["retry"]["status"] == "skipped" and results["retry"]["attempts"] == 3
    assert claim["machine"] == "140.123.106.216" and claim["prior_fail"] == ["140.123.106.37"]
    assert not (state / "retry_store.fail").exists()
    done = json.loads((state / "retry_store.done").read_text(encoding="utf-8"))
    assert done["errors"] == 1 and done["error_ids"] == ["retry"]


def test_scoped_worker_rejects_wrong_profile_scope_before_claim(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dedust, "DATASET_PATH", Path(str(dataset)))
    monkeypatch.setattr("antenna.utils.web.get_local_ip", lambda: "140.123.106.37")
    input_root, _, _ = _write_input(dataset, "input", ["sample"])
    config_path = input_root / "config.yaml"
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            f"scope: {SCOPE}", "scope: another_experiment"
        ),
        encoding="utf-8",
    )
    profiled_batch.atomic_json(dataset / "jobs.json", [{
        "input": "input", "store": "store", "prio": 1, "scope": SCOPE,
        "config": "input/config.yaml",
    }])
    monkeypatch.setattr(dedust, "run", lambda _args: pytest.fail("scope mismatch must not run"))

    with pytest.raises(ValueError, match="profile config scope"):
        dedust.worker(_worker_args())

    state = dataset / "jobs_state"
    assert not (state / "store.claim").exists()
    assert not (state / "store.fail").exists()
    assert not (state / "store.done").exists()
