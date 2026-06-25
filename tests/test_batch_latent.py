"""
tests/test_batch_latent.py — BatchLatentGenerator (γ 多候選) 單元 + 多候選訓練 loop 整合測試。

驗證 reparam 高斯雲 / σ 退火 / forward (K,out_dim)；以及 run_training 的多候選分支
(同批生成 K → SM 評分選最佳 → 只把選中那張送 HFSS) 能跑完、loss 有限、每 epoch ≤ 1 次 HFSS。
"""
import os

import pytest
import torch

from antenna.models.generators import BatchLatentGenerator, MultiScaleGenerator, SigmoidGenerator
from antenna.training import load_config, run_training

FIX = os.path.join(os.path.dirname(__file__), "fixtures")


# ── BatchLatentGenerator 單元 ──────────────────────────────────────────────
def test_forward_shape_k_outdim():
    g = BatchLatentGenerator(34, 625, hidden=(16, 16), num_candidates=5)
    assert g().shape == (5, 625)


def test_sigma_zero_collapses_to_center():
    """σ=0 → 所有候選 = fc_patch(z_center)、彼此完全相同 (無探索)。"""
    g = BatchLatentGenerator(34, 625, hidden=(16, 16), num_candidates=4, sigma=0.0)
    out = g()
    assert torch.allclose(out[0], out[1]) and torch.allclose(out[0], out[3])


def test_sigma_positive_diversifies():
    """σ>0 → 候選彼此不同 (高斯雲探索)。"""
    torch.manual_seed(0)
    g = BatchLatentGenerator(34, 625, hidden=(16, 16), num_candidates=4, sigma=0.5)
    out = g()
    assert not torch.allclose(out[0], out[1])


def test_anneal_sigma_interpolates_and_clamps():
    g = BatchLatentGenerator(8, 16, hidden=(8,), num_candidates=2, sigma=0.5, sigma_min=0.05)
    g.anneal_sigma(0.0); assert g.sigma == pytest.approx(0.5)
    g.anneal_sigma(1.0); assert g.sigma == pytest.approx(0.05)
    g.anneal_sigma(0.5); assert g.sigma == pytest.approx(0.275)
    g.anneal_sigma(2.0); assert g.sigma == pytest.approx(0.05)    # 超過 1 → clamp


def test_z_center_learnable_and_gets_grad():
    g = BatchLatentGenerator(8, 16, hidden=(8,), num_candidates=3, sigma=0.3)
    assert any(p is g.z_center for p in g.parameters())          # z_center 進 optimizer
    g().sum().backward()
    assert g.z_center.grad is not None and torch.isfinite(g.z_center.grad).all()


# ── MultiScaleGenerator 單元 ──────────────────────────────────────────────
def test_multiscale_forward_shape_and_fewer_params():
    g = MultiScaleGenerator(34, 625, scales=(1, 5, 13))
    out = g(torch.randn(34))
    assert out.shape == (625,) and torch.isfinite(out).all()
    n_ms = sum(p.numel() for p in g.parameters())
    n_sig = sum(p.numel() for p in SigmoidGenerator(34, 625).parameters())
    assert n_ms < n_sig                                  # 淺層多尺度 → 參數遠少於主 MLP


def test_multiscale_batch_equals_stacked():
    """(B,in) 批次 ≡ 逐筆 forward 疊起來 (per-row BiScaleNorm + 各筆獨立上採樣)。"""
    g = MultiScaleGenerator(8, 625, scales=(1, 5, 13))
    x = torch.randn(4, 8)
    batch = g(x)
    stacked = torch.stack([g(row) for row in x])
    assert batch.shape == (4, 625)
    assert torch.allclose(batch, stacked, atol=1e-5)


def test_multiscale_single_scale_is_uniform():
    """scales=(1,)：1×1 上採樣成均勻場 → BiScaleNorm 後整張同值 (多尺度機制的極端證明)。"""
    g = MultiScaleGenerator(8, 625, scales=(1,))
    out = g(torch.randn(8))
    assert torch.allclose(out, out[0].expand_as(out), atol=1e-5)


def test_multiscale_invalid_outdim_and_scales_raise():
    """建構時早 fail：out_dim 非完全平方、scale≤0、scale>side 都應 raise ValueError (防呆)。"""
    with pytest.raises(ValueError):
        MultiScaleGenerator(8, 624)                       # 非完全平方
    with pytest.raises(ValueError):
        MultiScaleGenerator(8, 625, scales=(0, 5))        # s≤0
    with pytest.raises(ValueError):
        MultiScaleGenerator(8, 625, scales=(30,))         # s>side(25)


def test_multiscale_3d_lead_equals_stacked():
    """3-D lead (B1,B2,in) ≡ 攤平逐筆 (lead 對位無錯，通用 reshape 守門)。"""
    g = MultiScaleGenerator(8, 625, scales=(1, 5, 13))
    x = torch.randn(2, 3, 8)
    out = g(x)
    assert out.shape == (2, 3, 625)
    assert torch.allclose(out.reshape(-1, 625), g(x.reshape(-1, 8)), atol=1e-5)


def test_multiscale_zero_params_degenerate_grad_finite():
    """退化路徑穿過 multiscale：所有頭參數歸零 → acc 全 0 → forward 全 0、backward 梯度有限
    (BiScaleNorm 的 clamp_min(eps) 在 multiscale 上也防住 0/0 反向 NaN)。"""
    g = MultiScaleGenerator(8, 625, scales=(1, 5, 13))
    for p in g.parameters():
        torch.nn.init.zeros_(p)
    x = torch.randn(8, requires_grad=True)
    out = g(x)
    assert torch.allclose(out, torch.zeros_like(out))
    out.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in g.parameters())


# ── 多候選訓練 loop 整合 ───────────────────────────────────────────────────
class _CountSim:
    """確定性假模擬器 + 計數 HFSS 評估次數 (__call__) 與 COM 生命週期。"""
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


def test_multi_candidate_loop_runs_one_hfss_per_epoch(tmp_path):
    """batch_latent：跑得完、gen_loss 有限、每 epoch ≤ 1 次 HFSS (best-of-K，而非 K 次)。"""
    cfg = load_config(os.path.join(FIX, "single_test.yaml"))
    cfg.generator = {"name": "batch_latent", "num_candidates": 4}
    sim = _CountSim(("S11", "Gain"))
    losses = []
    run_training(cfg, simulator=sim, record_path=tmp_path, seed=0, max_epochs=4,
                 on_epoch=lambda e, m: losses.append(m["gen_loss"]), verbose=False)
    assert len(losses) == 4
    assert all(l == l for l in losses)                  # 無 NaN
    assert sim.eval_count <= 4                          # 每 epoch ≤ 1 次 HFSS (K=4 卻不到 16)
    assert sim.calls["start"] == 4


def test_metrics_csv_multi_without_rad_partial_columns(tmp_path):
    """batch_latent 但不開 rad：metrics.csv 的 sigma/score_* 有值、rad_loss 留空，讀回不錯位。"""
    import csv
    cfg = load_config(os.path.join(FIX, "single_test.yaml"))      # 無 radiation 區段 → rad_on=False
    cfg.generator = {"name": "batch_latent", "num_candidates": 3}
    sim = _CountSim(("S11", "Gain"))
    run_training(cfg, simulator=sim, record_path=tmp_path, seed=0, max_epochs=2, verbose=False)
    with open(tmp_path / "metrics.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[-1]["sigma"] != ""                       # multi → sigma 有值
    assert rows[-1]["score_spread"] != ""                # multi → select 有值
    assert rows[-1]["rad_loss"] == ""                    # 非 rad → rad_loss 留空 (不錯位)
