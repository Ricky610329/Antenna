"""
tests/test_ensemble_trust.py — EnsembleSurrogate (集成 SM) + TrustController (閉迴路信任控制) 單元，
以及 generator-free + ensemble + trust 的端到端整合 (Exp2/Exp3)。

驗證：
- TrustController：enable=False → 靜態 (tau_mult≡1、λ_trust/κ=base、update no-op)；
  enable=True → gap 小→t 高、gap 大→t 低 (clamp)、EMA 平滑、三致動器隨 t 動。
- EnsembleSurrogate：成員數/檔名、uncertainty (成員分歧、可微、相同權重→0)、save/load、pre_load 擾動。
- 整合：direct + ensemble + trust 跑得完、metrics.csv 有 sm_unc/trust_t 欄；ensemble 但 trust off →
  有 sm_unc、無 trust_t；單一 SM → 無 sm_unc (不錯位)。
"""
import csv
import os

import pytest
import torch

from antenna.models import EnsembleMLPSurrogate, MLPSurrogate
from antenna.training import TrustController, load_config, run_training

FIX = os.path.join(os.path.dirname(__file__), "fixtures")


class _CountSim:
    def __init__(self, labels):
        self.labels = labels
        self.eval_count = 0
        self.calls = {"open": 0, "start": 0, "end": 0, "clean": 0}

    def open(self): self.calls["open"] += 1
    def start(self, num): self.calls["start"] += 1
    def end(self, *a, **k): self.calls["end"] += 1; return 0
    def clean(self, *a, **k): self.calls["clean"] += 1
    def restart(self, **k): pass

    def __call__(self, pattern, **kw):
        self.eval_count += 1
        fr = pattern.float().mean()
        x = torch.linspace(0, 1, 17)
        bump = torch.exp(-((x - 0.5) ** 2) / 0.05)
        out = {}
        for lbl in self.labels:
            out[lbl] = (-12.0 * bump * fr) if lbl in ("S11", "S22") else (4.0 * bump * fr - 19.0 * (1.0 - fr))
        return out


# ── TrustController 單元 ───────────────────────────────────────────────────
def test_trust_disabled_is_static():
    """enable=False：t 恆 1、tau_mult≡1、λ_trust/κ = 靜態 base、update 為 no-op。"""
    tc = TrustController(enable=False, lambda_trust_base=0.2, kappa_base=0.1)
    tc.update(5.0, 0.0)                       # 大 gap 但停用 → 不更新
    assert tc.t == 1.0
    assert tc.tau_mult() == 1.0
    assert tc.lambda_trust() == pytest.approx(0.2)
    assert tc.kappa() == pytest.approx(0.1)


def test_trust_small_gap_high_trust():
    """gap=0 → t=exp(0)=1 → clamp t_max；tau 幾乎不放軟、λ_trust/κ 幾乎 0 (利用)。"""
    tc = TrustController(enable=True, lambda_trust_base=0.2, kappa_base=0.1,
                         g0=1.0, ema=1.0, t_max=0.95, tau_inflate=3.0)
    tc.update(1.0, 1.0)                       # gap=0
    assert tc.t == pytest.approx(0.95)
    assert tc.tau_mult() == pytest.approx(1.0 + 2.0 * 0.05)
    assert tc.lambda_trust() == pytest.approx(0.2 * 0.05)
    assert tc.kappa() == pytest.approx(0.1 * 0.05)


def test_trust_large_gap_low_trust():
    """gap 很大 → t→0 → clamp t_min；tau 放軟近上限、λ_trust/κ 近 base (探索/拉回)。"""
    tc = TrustController(enable=True, lambda_trust_base=0.2, kappa_base=0.1,
                         g0=1.0, ema=1.0, t_min=0.05, tau_inflate=3.0)
    tc.update(10.0, 0.0)                      # gap=10 → exp(-10)≈0
    assert tc.t == pytest.approx(0.05)
    assert tc.tau_mult() == pytest.approx(1.0 + 2.0 * 0.95)
    assert tc.lambda_trust() == pytest.approx(0.2 * 0.95)


def test_trust_ema_smoothing():
    """gap_ema 用 EMA 平滑：第一筆=gap；之後 = ema·gap + (1−ema)·prev。"""
    tc = TrustController(enable=True, lambda_trust_base=0.0, kappa_base=0.0, g0=1.0, ema=0.5)
    tc.update(2.0, 0.0)
    assert tc.gap_ema == pytest.approx(2.0)          # 首筆
    tc.update(0.0, 0.0)
    assert tc.gap_ema == pytest.approx(1.0)          # 0.5·0 + 0.5·2


def test_trust_t_clamped_both_ends():
    tc = TrustController(enable=True, lambda_trust_base=0, kappa_base=0, g0=1.0, ema=1.0, t_min=0.1, t_max=0.9)
    tc.update(0.0, 0.0); assert tc.t == pytest.approx(0.9)     # 上限
    tc.update(50.0, 0.0); assert tc.t == pytest.approx(0.1)    # 下限


def test_trust_nan_gap_max_exploration_not_sticky():
    """SM 預測 NaN/inf → 直接最大探索 (t_min)，且「不」把 NaN 折進 gap_ema (否則永久卡高信任、方向相反)。"""
    tc = TrustController(enable=True, lambda_trust_base=0.2, kappa_base=0.1, g0=1.0, ema=0.5, t_min=0.05, t_max=0.95)
    tc.update(0.5, 0.5)                       # 正常 gap=0 → gap_ema=0
    good = tc.gap_ema
    tc.update(float("inf"), 0.0)             # inf gap → t_min、gap_ema 不汙染
    assert tc.t == pytest.approx(0.05)
    assert tc.gap_ema == pytest.approx(good)
    tc.update(0.5, 0.5)                       # 下一筆正常 → EMA 從上次好值續算 (沒被 NaN 卡死)
    assert tc.t == pytest.approx(0.95)


def test_trust_invalid_config_raises():
    """誤設護欄：g0≤0 或 tau_inflate<1 (會反向銳化) → 建構即報錯。"""
    with pytest.raises(ValueError):
        TrustController(enable=True, lambda_trust_base=0, kappa_base=0, g0=0.0)
    with pytest.raises(ValueError):
        TrustController(enable=True, lambda_trust_base=0, kappa_base=0, tau_inflate=0.5)


# ── EnsembleSurrogate 單元 ─────────────────────────────────────────────────
def test_ensemble_build_and_member_names(tmp_path):
    ens = EnsembleMLPSurrogate(str(tmp_path), 625, (2, 17), hidden=(16, 16), ensemble_size=3)
    assert len(ens.members) == 3
    assert [m.name for m in ens.members] == ["sm0", "sm1", "sm2"]


def test_ensemble_requires_min_two(tmp_path):
    with pytest.raises(ValueError):
        EnsembleMLPSurrogate(str(tmp_path), 625, (2, 17), hidden=(16, 16), ensemble_size=1)


def test_ensemble_uncertainty_positive_and_differentiable(tmp_path):
    """不同 init 成員 → 預測有分歧 (uncertainty>0)；可微 (供信任懲罰反傳)。"""
    torch.manual_seed(0)
    ens = EnsembleMLPSurrogate(str(tmp_path), 625, (2, 17), hidden=(16, 16), ensemble_size=4)
    x = torch.rand(625, requires_grad=True)
    u = ens.uncertainty(x)
    assert u.item() > 0
    u.backward()
    assert torch.isfinite(x.grad).all()


def test_ensemble_uncertainty_zero_when_members_identical(tmp_path):
    """成員權重相同 → 預測一致 → uncertainty=0 (機制正確性的鎖)。"""
    ens = EnsembleMLPSurrogate(str(tmp_path), 625, (2, 17), hidden=(16, 16), ensemble_size=3)
    sd = ens.members[0].model.state_dict()
    for m in ens.members[1:]:
        m.model.load_state_dict(sd)
    assert ens.uncertainty(torch.rand(625)).item() == pytest.approx(0.0, abs=1e-6)


def test_ensemble_call_criterion_finite(tmp_path):
    """__call__ 回成員平均 → MultiResponses.criterion() 有限 (guidance 主項可用)。"""
    ens = EnsembleMLPSurrogate(str(tmp_path), 625, (2, 17), hidden=(16, 16), ensemble_size=3)
    assert torch.isfinite(ens(torch.rand(625)).criterion())


def test_ensemble_save_load_roundtrip(tmp_path):
    """save → sm0..smK-1.pth；load 還原各成員權重 (逐成員)。"""
    ens = EnsembleMLPSurrogate(str(tmp_path), 625, (2, 17), hidden=(16, 16), ensemble_size=3)
    ens.save()
    assert (tmp_path / "sm0.pth").exists() and (tmp_path / "sm1.pth").exists() and (tmp_path / "sm2.pth").exists()
    before = ens.members[1].model.fc_patch[0].weight.detach().clone()
    ens2 = EnsembleMLPSurrogate(str(tmp_path), 625, (2, 17), hidden=(16, 16), ensemble_size=3)
    assert not torch.allclose(ens2.members[1].model.fc_patch[0].weight, before)   # 新 init 不同
    ens2.load()
    assert torch.allclose(ens2.members[1].model.fc_patch[0].weight, before)       # load 後一致


def test_ensemble_pre_load_anchors_member0_perturbs_rest(tmp_path):
    """暖啟動：成員0 = 精確預訓練 (作錨)；成員1+ = 預訓練 + 擾動 (製造多樣性)。"""
    single = MLPSurrogate(str(tmp_path), 625, (2, 17), hidden=(16, 16))
    pre = tmp_path / "pre.pth"
    single.save_as(pre)
    ens = EnsembleMLPSurrogate(str(tmp_path), 625, (2, 17), hidden=(16, 16), ensemble_size=3, init_perturb=0.05)
    ens.pre_load_model(str(pre))
    ref = single.model.fc_patch[0].weight
    assert torch.allclose(ens.members[0].model.fc_patch[0].weight, ref)            # 成員0 精確
    assert not torch.allclose(ens.members[1].model.fc_patch[0].weight, ref)        # 成員1 擾動


def test_ensemble_preload_syncs_lookahead_anchor(tmp_path):
    """A6:擾動成員權重時同步 Ranger slow_buffer (Lookahead 錨),否則每 k 步把權重拉回未擾動的
    共同錨、吃掉 init_perturb 的多樣性 → uncertainty 邊跑邊塌。"""
    single = MLPSurrogate(str(tmp_path), 625, (2, 17), hidden=(16, 16))
    out = single.model(torch.rand(625))               # 跑一步 → Ranger lazy-init slow_buffer
    (out ** 2).mean().backward()
    single.optimizer.step()
    pre = tmp_path / "pre.pth"
    single.save_as(pre)
    ens = EnsembleMLPSurrogate(str(tmp_path), 625, (2, 17), hidden=(16, 16), ensemble_size=3, init_perturb=0.05)
    ens.pre_load_model(str(pre))
    m = ens.members[1]
    p = m.model.fc_patch[0].weight                     # 有 optimizer state 的 trunk 參數 (numel>1)
    st = m.optimizer.state.get(p)
    assert st is not None and "slow_buffer" in st
    assert torch.allclose(st["slow_buffer"], p.detach())   # 錨已同步成擾動後權重 (不會被拉回共同錨)


def test_reset_online_lr_restores_construction_lr(tmp_path):
    """offline 預訓練 / 暖身會把 lr 砍到地板;reset_online_lr 把 lr 拉回建構值 (保留動量)。"""
    sm = MLPSurrogate(str(tmp_path), 625, (2, 17), hidden=(16, 16), lr=0.001)
    for g in sm.optimizer.param_groups:               # 模擬 ReduceLROnPlateau 把 lr 砍到 floor
        g["lr"] = 1e-6
    sm.reset_online_lr()
    assert sm.optimizer.param_groups[0]["lr"] == pytest.approx(0.001)


# ── 整合 (Exp2/Exp3)：direct + ensemble + trust ──────────────────────────────
def test_guided_ensemble_trust_loop_runs(tmp_path):
    """Exp3：direct + ensemble + 信任懲罰 + acquisition + 閉迴路控制 → 跑得完、metrics 有 sm_unc/trust_t。"""
    cfg = load_config(os.path.join(FIX, "single_test.yaml"))
    cfg.generator = {"name": "direct", "num_candidates": 4}
    cfg.surrogate = {"name": "ensemble", "ensemble_size": 3}
    cfg.loss = {"spectral_connectivity": 0.0005, "uncertainty": 0.1}
    cfg.selection = {"uncertainty_weight": 0.05}
    cfg.trust = {"enable": True, "g0": 0.5}
    losses = []
    run_training(cfg, simulator=_CountSim(("S11", "Gain")), record_path=tmp_path, seed=0,
                 max_epochs=4, on_epoch=lambda e, m: losses.append(m["gen_loss"]), verbose=False)
    assert len(losses) == 4 and all(l == l for l in losses)
    with open(tmp_path / "metrics.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[-1]["sm_unc"] != ""                 # ensemble → 有不確定性
    assert rows[-1]["trust_t"] != ""                # 閉迴路 → 有信任標量 t
    assert rows[-1]["gap_ema"] != ""                # 閉迴路 → 有 gap_ema (調 g0 稽核訊號)


def test_ensemble_static_trust_off_has_unc_no_t(tmp_path):
    """Exp2：ensemble 但 trust off → sm_unc 有值、trust_t 留空 (靜態 λ_trust)。"""
    cfg = load_config(os.path.join(FIX, "single_test.yaml"))
    cfg.generator = {"name": "direct", "num_candidates": 3}
    cfg.surrogate = {"name": "ensemble", "ensemble_size": 3}
    cfg.loss = {"spectral_connectivity": 0.0005, "uncertainty": 0.1}
    run_training(cfg, simulator=_CountSim(("S11", "Gain")), record_path=tmp_path, seed=0,
                 max_epochs=2, verbose=False)
    with open(tmp_path / "metrics.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[-1]["sm_unc"] != ""
    assert rows[-1]["trust_t"] == ""                # 未開閉迴路 → 留空
    assert rows[-1]["gap_ema"] == ""               # 未開閉迴路 → gap_ema 也留空


def test_single_sm_has_no_uncertainty_column(tmp_path):
    """單一 SM (非 ensemble) → sm_unc/trust_t 留空 (不錯位、golden 風格相容)。"""
    cfg = load_config(os.path.join(FIX, "single_test.yaml"))
    cfg.generator = {"name": "direct", "num_candidates": 3}
    run_training(cfg, simulator=_CountSim(("S11", "Gain")), record_path=tmp_path, seed=0,
                 max_epochs=2, verbose=False)
    with open(tmp_path / "metrics.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[-1]["sm_unc"] == ""
    assert rows[-1]["trust_t"] == ""
