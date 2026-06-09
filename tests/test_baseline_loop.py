"""
端到端行為基準：讀 YAML → antenna.training.run_training(mock) → 比對 golden。

驗證「YAML 驅動的 production 訓練核心」在單/雙埠都重現既有 golden (行為不變)：
- single → golden_loop.json (重構前就鎖定的單埠基準)
- dual   → golden_loop_dual.json (config 驅動版鎖定的雙埠基準)

simulator 以 mock 注入：實作完整 COM 生命週期 (open/start/end/clean) 並記錄呼叫，
測試斷言生命週期被正確呼叫。高 patience → 不觸發 rollback (確定性)。
"""
import os
import json

import pytest
import torch

from antenna.training import load_config, run_training, setup_responses

EPOCHS = 6
FIX = os.path.join(os.path.dirname(__file__), "fixtures")


class _MockSim:
    """確定性假模擬器 + 記錄 COM 生命週期呼叫。labels 決定回傳哪些響應。"""
    def __init__(self, labels):
        self.labels = labels
        self.calls = {"open": 0, "start": 0, "end": 0, "clean": 0}

    def open(self): self.calls["open"] += 1
    def start(self, num): self.calls["start"] += 1
    def end(self, *a, **k): self.calls["end"] += 1; return 0
    def clean(self, *a, **k): self.calls["clean"] += 1
    def restart(self, **k): pass

    def __call__(self, pattern, **kw):
        fr = pattern.float().mean()
        x = torch.linspace(0, 1, 17)
        bump = torch.exp(-((x - 0.5) ** 2) / 0.05)
        out = {}
        for lbl in self.labels:
            if lbl in ("S11", "S22"):
                out[lbl] = -12.0 * bump * fr
            else:                          # S21 / Gain
                out[lbl] = 4.0 * bump * fr - 19.0 * (1.0 - fr)
        return out


def _run(yaml_name, labels, tmpdir):
    cfg = load_config(os.path.join(FIX, yaml_name))
    sim = _MockSim(labels)
    series = {k: [] for k in ["real_loss", "min_loss", "fake_loss", "r_feed", "tau", "lr"]}

    def capture(epoch, m):
        for k in series:
            series[k].append(m[k])

    run_training(cfg, simulator=sim, record_path=tmpdir, seed=0, on_epoch=capture)
    return series, sim.calls


def _compare_golden(series, golden_file):
    path = os.path.join(os.path.dirname(__file__), golden_file)
    assert len(series["real_loss"]) == EPOCHS
    for k, vals in series.items():
        assert all(v == v for v in vals), f"{k} 含 NaN"
    if os.path.exists(path):
        golden = json.load(open(path, encoding="utf-8"))
        for key, gvals in golden.items():
            assert key in series and len(series[key]) == len(gvals), f"{golden_file} {key} 長度不符"
            for i, (a, b) in enumerate(zip(series[key], gvals)):
                assert abs(a - b) <= 1e-4, (
                    f"[golden drift] {golden_file} {key}[{i}]: 現在={a:.8g} vs 基準={b:.8g}"
                )
    else:
        json.dump(series, open(path, "w", encoding="utf-8"), indent=2)


@pytest.fixture
def _restore_single():
    """dual 測試後把全域響應還原成 single，避免污染後續 (單埠) 測試。"""
    yield
    setup_responses(load_config(os.path.join(FIX, "single_test.yaml")))


def test_baseline_single(tmp_path):
    series, calls = _run("single_test.yaml", ("S11", "Gain"), tmp_path)
    _compare_golden(series, "golden_loop.json")
    assert calls["open"] == 1
    assert calls["start"] == EPOCHS
    assert calls["end"] == EPOCHS
    assert calls["clean"] == EPOCHS


def test_baseline_dual(tmp_path, _restore_single):
    series, calls = _run("dual_test.yaml", ("S11", "S21", "S22"), tmp_path)
    _compare_golden(series, "golden_loop_dual.json")
    assert calls["open"] == 1
    assert calls["start"] == EPOCHS
    assert calls["end"] == EPOCHS
