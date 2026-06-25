"""
tests/test_acp_boundary_gate.py — boundary-gated ACP (opt-in) 單元 + loop 整合測試。

boundary 當「探索/固化」依據：boundary≥τ_b (衝出 SM 可信區) → 抑制 plateau warm restart
(不加熱、冷卻就地固化)；boundary<τ_b (區內卡住) → 放行往外探。boundary=None → 現行 ACP (golden)。
"""
import os

import pytest
import torch

from antenna.training import load_config, run_training

FIX = os.path.join(os.path.dirname(__file__), "fixtures")


def _make_sched(patience=3, **kw):
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=0.01)
    from antenna.optim.scheduler import AdaptiveCyclicalScheduler
    return AdaptiveCyclicalScheduler(opt, T_0=10, lr_max=0.01, lr_min=1e-4,
                                     temp_max=4.0, temp_min=0.1, warmup_ratio=0.2,
                                     patience=patience, factor=0.5, on_plateau="linear", **kw)


# ── 單元：閘門邏輯 ─────────────────────────────────────────────────────────
def test_boundary_none_matches_not_passing():
    """boundary=None 與「完全不傳 boundary」逐步一致 (golden 安全的內部一致性)。"""
    metrics = [5, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1]
    s1, s2 = _make_sched(), _make_sched()
    for m in metrics:
        s1.step(m)
        s2.step(m, boundary=None, boundary_threshold=None)
        assert (s1.T_cur, s1.T_i, round(s1.get_temp(), 8)) == (s2.T_cur, s2.T_i, round(s2.get_temp(), 8))


def test_gate_suppresses_when_out_fires_when_in():
    """plateau 時：boundary≥τ_b → 抑制 (T_i 不縮)；boundary<τ_b → warm restart (T_i 縮)。"""
    # out-of-region → 抑制
    s = _make_sched(boundary_suppress_cap=99)
    s.step(5.0)
    ti0 = s.T_i
    for _ in range(3):
        s.step(5.0, boundary=10.0, boundary_threshold=1.0)
    assert s._last_restart_suppressed is True
    assert s.T_i == ti0                                   # 抑制 → 週期未縮

    # in-region → 放行 restart
    s2 = _make_sched()
    s2.step(5.0)
    ti1 = s2.T_i
    for _ in range(3):
        s2.step(5.0, boundary=0.1, boundary_threshold=1.0)
    assert s2._last_restart_suppressed is False
    assert s2.T_i < ti1                                   # restart → 週期縮 (factor=0.5)


def test_suppress_cap_forces_restart_through():
    """連續抑制達 cap → 強制放行一次 (防餓死)。patience=1 → 每步觸發 plateau。"""
    s = _make_sched(patience=1, boundary_suppress_cap=2)
    s.step(5.0)
    flags = []
    for _ in range(5):
        s.step(5.0, boundary=10.0, boundary_threshold=1.0)   # 每步 plateau + 出界
        flags.append(s._last_restart_suppressed)
    assert flags == [True, True, False, True, True]          # 抑制,抑制,放行,抑制,抑制


# ── τ_b (boundary_threshold) 單元 ───────────────────────────────────────────
def test_boundary_threshold_dedup_and_degenerate():
    """τ_b = κ·中位 NN 間距：排除重複 (零距離不污染)、全相同→inf (閘門退回現行 ACP)、M<2→inf。"""
    from antenna.losses import boundary_threshold
    P = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])   # 3 個不同 pattern
    assert 0.0 < boundary_threshold(P, 1.0) < float("inf")
    Pd = torch.tensor([[0., 0., 0.], [0., 0., 0.], [3., 0., 0.]])  # row0==row1 重複
    assert boundary_threshold(Pd, 1.0) == pytest.approx(3.0)       # 重複的 0 距離被排除 → 取 3 (非 0)
    assert boundary_threshold(torch.zeros(4, 5), 1.0) == float("inf")  # 全相同 → inf
    assert boundary_threshold(torch.zeros(1, 5)) == float("inf")       # M<2 → inf


# ── loop 整合 ─────────────────────────────────────────────────────────────
class _Sim:
    def __init__(self, labels): self.labels = labels; self.calls = {"open":0,"start":0,"end":0,"clean":0}
    def open(self): self.calls["open"] += 1
    def start(self, n): self.calls["start"] += 1
    def end(self, *a, **k): self.calls["end"] += 1; return 0
    def clean(self, *a, **k): self.calls["clean"] += 1
    def restart(self, **k): pass
    def __call__(self, pattern, **kw):
        fr = pattern.float().mean()
        x = torch.linspace(0, 1, 17)
        bump = torch.exp(-((x - 0.5) ** 2) / 0.05)
        return {lbl: (-12.0 * bump * fr if lbl in ("S11", "S22") else 4.0 * bump * fr - 19.0 * (1 - fr))
                for lbl in self.labels}


def test_boundary_gate_loop_runs_and_emits_diagnostics(tmp_path):
    """boundary_gate + replay 緩衝：跑得完、gen_loss 無 NaN；replay 夠大後 snap 出現 boundary 診斷。"""
    cfg = load_config(os.path.join(FIX, "single_test.yaml"))
    cfg.sm_train = {"min_loss": 0.5, "max_epoch": 3, "mode": "replay", "newest_steps": 2, "replay_size": 64}
    cfg.scheduler = {"on_plateau": "linear", "boundary_gate": True,
                     "boundary_kappa": 1.5, "boundary_recompute_every": 1}
    snaps = []
    run_training(cfg, simulator=_Sim(("S11", "Gain")), record_path=tmp_path, seed=0,
                 max_epochs=6, patience=2, on_epoch=lambda e, m: snaps.append(m), verbose=False)
    assert len(snaps) == 6
    assert all(s["gen_loss"] == s["gen_loss"] for s in snaps)        # 無 NaN
    assert any("boundary" in s and "boundary_threshold" in s for s in snaps)  # replay>2 後有診斷
