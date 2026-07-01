"""
tests/test_guided_configs.py — 3 個新主線 config (generator-free 階梯) 的 smoke：
載入真實 configs/single_guided*.yaml → 用 mock 跑幾 epoch → 確認鍵合法 + 端到端 wiring 不破、loss 有限。

(config 的 surrogate.pretrained / offline_dataset 是給 entry point 解析的 metadata，build_surrogate 忽略；
 故此處用 mock 不帶預訓練檔即可跑 → SM 從隨機起步，仍能驗證 direct/ensemble/trust 全鏈路。)
"""
import csv
import os

import pytest
import torch

from antenna.training import load_config, run_training, setup_responses

CONFIGS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")
FIX = os.path.join(os.path.dirname(__file__), "fixtures")
GUIDED = ["single_guided_harvest.yaml", "single_guided_ens_harvest.yaml",
          "single_guided_ens_adapt_harvest.yaml", "single_guided_dlffit_harvest.yaml",
          "single_guided_refit_harvest.yaml",
          # Round 2 (ensemble+trust 治本,n_basis=8)
          "single_r2_ens_harvest.yaml", "single_r2_enstrust_harvest.yaml",
          "single_r2_refit_enstrust_harvest.yaml",
          # Round 3 (探索/DIP factorial;E=lr↑, D=sigmoid, E+D=兩者)
          "single_r3_explore.yaml", "single_r3_dip.yaml", "single_r3_dip_explore.yaml"]


class _CountSim:
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
            out[lbl] = (-12.0 * bump * fr) if lbl in ("S11", "S22") else (4.0 * bump * fr - 19.0 * (1.0 - fr))
        return out


@pytest.fixture(autouse=True)
def _restore_single():
    """每個 smoke 跑完還原 single spec，避免污染其他測試 (config 安裝了自己的 spec)。"""
    yield
    setup_responses(load_config(os.path.join(FIX, "single_test.yaml")))


@pytest.mark.parametrize("name", GUIDED)
def test_guided_config_loads_and_runs(name, tmp_path):
    """3 個新 config 都能載入 (鍵白名單通過) 且 mock 跑得完、gen_loss 有限。"""
    cfg = load_config(os.path.join(CONFIGS, name))
    losses = []
    run_training(cfg, simulator=_CountSim(("S11", "Gain")), record_path=tmp_path, seed=0,
                 max_epochs=3, on_epoch=lambda e, m: losses.append(m["gen_loss"]), verbose=False)
    assert len(losses) == 3 and all(l == l for l in losses)
    with open(tmp_path / "metrics.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    # direct 多候選 → score_spread/cand_similarity 有值;sigmoid 單候選 → 空(走單張路徑)
    if cfg.generator.get("name", "sigmoid") == "direct":
        assert rows[-1]["score_spread"] != "" and rows[-1]["cand_similarity"] != ""
    else:
        assert rows[-1]["score_spread"] == ""


def test_debug_signals_logged(tmp_path):
    """新追蹤訊號落 csv:worst_margin/metal_frac/grad_norm 每 epoch 有;sm_gap/sm_fit_* fresh epoch 有。"""
    cfg = load_config(os.path.join(CONFIGS, "single_guided_harvest.yaml"))
    run_training(cfg, simulator=_CountSim(("S11", "Gain")), record_path=tmp_path, seed=0,
                 max_epochs=3, verbose=False)
    with open(tmp_path / "metrics.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    last = rows[-1]
    assert last["worst_margin"] != "" and last["metal_frac"] != "" and last["grad_norm"] != ""
    assert any(r["sm_gap"] != "" for r in rows)            # generalization 訊號 (fresh epoch)
    assert any(r["sm_fit_loss"] != "" and r["sm_fit_epochs"] != "" for r in rows)   # SM 重訓 (dlf elite)


def test_guided_radiation_weight_lowered():
    """確認 3 個新 config 的 radiation.weight 已調低 (≤0.1，重點是 S11/Gain)。"""
    for name in GUIDED:
        cfg = load_config(os.path.join(CONFIGS, name))
        assert cfg.radiation["weight"] <= 0.1, f"{name} radiation.weight 應調低"


def test_guided_ladder_isolates_one_variable():
    """階梯 = 每級只多一招：Exp1 direct；Exp2 +ensemble+uncertainty；Exp3 +trust.enable。"""
    e1 = load_config(os.path.join(CONFIGS, "single_guided_harvest.yaml"))
    e2 = load_config(os.path.join(CONFIGS, "single_guided_ens_harvest.yaml"))
    e3 = load_config(os.path.join(CONFIGS, "single_guided_ens_adapt_harvest.yaml"))
    assert e1.generator["name"] == "direct" and e1.surrogate.get("name", "mlp") == "mlp" and not e1.trust
    assert e2.surrogate["name"] == "ensemble" and e2.loss.get("uncertainty", 0) > 0 and not e2.trust
    assert e3.surrogate["name"] == "ensemble" and e3.trust.get("enable") is True
