"""
tests/test_direct_generator.py — DirectPatternGenerator (generator-free) 單元 + 整合測試。

驗證「pattern logits 本身即可學參數」(無 MLP)：forward 形狀 / BiScaleNorm 同尺度 / K 候選獨立可微 /
is_multi_candidate 旗標；以及 run_training 走多候選路徑 (generator=direct) 能跑完、loss 有限、
每 epoch ≤ 1 次 HFSS，且 K=1 退化成單張路徑。
"""
import csv
import os

import pytest
import torch

from antenna.models.generators import DirectPatternGenerator
from antenna.training import load_config, run_training

FIX = os.path.join(os.path.dirname(__file__), "fixtures")


class _CountSim:
    """確定性假模擬器 + 計數 HFSS 評估次數 (沿用 test_batch_latent 的 mock)。"""
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


# ── DirectPatternGenerator 單元 ────────────────────────────────────────────
def test_forward_shape_k_outdim():
    g = DirectPatternGenerator(34, 625, num_candidates=5)
    assert g().shape == (5, 625)
    assert g.is_multi_candidate is True


def test_forward_k1_collapses_to_single_vector():
    """K=1 → forward 回 (out_dim,) 與 SigmoidGenerator 同形 → 走單張路徑；旗標 False。"""
    g = DirectPatternGenerator(34, 625, num_candidates=1)
    assert g().shape == (625,)
    assert g.is_multi_candidate is False


def test_logits_are_leaf_parameter_and_get_grad():
    """優化變數就是 logits 本身 (leaf Parameter)，且可微 (梯度經 BiScaleNorm 流回)。"""
    g = DirectPatternGenerator(8, 16, num_candidates=3)
    assert any(p is g.logits for p in g.parameters())     # logits 進 optimizer
    g().sum().backward()
    assert g.logits.grad is not None and torch.isfinite(g.logits.grad).all()


def test_biscalenorm_range_and_per_row():
    """輸出經 BiScaleNorm → 落 [-1, 1]；各列 (候選) 各自正規化 (per-row，互不耦合)。"""
    torch.manual_seed(0)
    g = DirectPatternGenerator(8, 20, num_candidates=4)
    out = g()
    assert out.max() <= 1.0 + 1e-6 and out.min() >= -1.0 - 1e-6
    assert float(out.max(dim=-1).values.min()) == pytest.approx(1.0, abs=1e-5)   # 每列都有 +1 (正半邊最大)


def test_candidates_diverse_at_init():
    """randn 初始化 → 各候選 logits 列彼此不同 (pattern 空間天生多樣，非塌縮)。"""
    torch.manual_seed(0)
    g = DirectPatternGenerator(8, 625, num_candidates=4)
    out = g()
    assert not torch.allclose(out[0], out[1])
    assert not torch.allclose(out[0], out[3])


def test_init_scale_zero_is_uniform_then_normalized():
    """init_scale=0 → logits 全 0 → BiScaleNorm 後仍全 0 (退化路徑可微、不 NaN)。"""
    g = DirectPatternGenerator(8, 16, num_candidates=2, init_scale=0.0)
    out = g()
    assert torch.allclose(out, torch.zeros_like(out))
    out.sum().backward()
    assert torch.isfinite(g.logits.grad).all()           # BiScaleNorm clamp_min(eps) 防 0/0 反向 NaN


# ── generator-free 訓練 loop 整合 ──────────────────────────────────────────
def test_direct_multi_loop_runs_one_hfss_per_epoch(tmp_path):
    """direct(K=4)：走多候選路徑、跑得完、gen_loss 有限、每 epoch ≤ 1 次 HFSS (best-of-K)。"""
    cfg = load_config(os.path.join(FIX, "single_test.yaml"))
    cfg.generator = {"name": "direct", "num_candidates": 4}
    sim = _CountSim(("S11", "Gain"))
    losses = []
    run_training(cfg, simulator=sim, record_path=tmp_path, seed=0, max_epochs=4,
                 on_epoch=lambda e, m: losses.append(m["gen_loss"]), verbose=False)
    assert len(losses) == 4
    assert all(l == l for l in losses)                    # 無 NaN
    assert sim.eval_count <= 4                            # 每 epoch ≤ 1 次 HFSS (K=4 卻不到 16)
    assert sim.calls["start"] == 4


def test_direct_k1_runs_single_path(tmp_path):
    """direct(K=1)：退化成單張路徑 (非多候選)，照樣跑完、loss 有限。"""
    cfg = load_config(os.path.join(FIX, "single_test.yaml"))
    cfg.generator = {"name": "direct", "num_candidates": 1}
    sim = _CountSim(("S11", "Gain"))
    losses = []
    run_training(cfg, simulator=sim, record_path=tmp_path, seed=0, max_epochs=3,
                 on_epoch=lambda e, m: losses.append(m["gen_loss"]), verbose=False)
    assert len(losses) == 3 and all(l == l for l in losses)


def test_direct_metrics_csv_no_sigma_column_but_has_select(tmp_path):
    """direct 無 σ → metrics.csv 不應有 sigma 值 (留空)；但多候選的 score_*/cand_similarity 有值。"""
    cfg = load_config(os.path.join(FIX, "single_test.yaml"))
    cfg.generator = {"name": "direct", "num_candidates": 3}
    run_training(cfg, simulator=_CountSim(("S11", "Gain")), record_path=tmp_path, seed=0,
                 max_epochs=2, verbose=False)
    with open(tmp_path / "metrics.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[-1].get("sigma", "") == ""                # direct 無 σ → 留空 (不錯位)
    assert rows[-1]["score_spread"] != ""                 # 多候選 select 有值
    assert rows[-1]["cand_similarity"] != ""              # 候選相似度有值
