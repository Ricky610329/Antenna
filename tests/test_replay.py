"""
ReplayBuffer + SM 經驗回放模式 (sm_train.mode: replay) 的測試。

replay 是 opt-in 改良：把「最新一筆擬到收斂」反模式換成「最新少數步 + 回放緩衝一遍」。
預設 mode=single → 行為與原樣相同 (golden 由 test_baseline_loop 把關)。
"""
import os

import torch

from antenna.training import TrainConfig, load_config, run_training
from antenna.utils.replay import ReplayBuffer

FIX = os.path.join(os.path.dirname(__file__), "fixtures")

SINGLE_TARGETS = {
    "S11":  {"side": 0,   "center": -10, "width": [5, 0, 7, 0, 5], "method": "low"},
    "Gain": {"side": -19, "center": 4,   "width": [5, 0, 7, 0, 5], "method": "high"},
}


class _Mock:
    """確定性假模擬器 (S11/Gain)，含 COM 生命週期。"""
    def __init__(self, labels): self.labels = labels
    def open(self): pass
    def start(self, num): pass
    def end(self, *a, **k): return 0
    def clean(self, *a, **k): pass
    def restart(self, **k): pass
    def __call__(self, pattern, **kw):
        fr = pattern.float().mean()
        x = torch.linspace(0, 1, 17)
        bump = torch.exp(-((x - 0.5) ** 2) / 0.05)
        out = {}
        for lbl in self.labels:
            out[lbl] = (-12.0 * bump * fr) if lbl in ("S11", "S22") else (4.0 * bump * fr - 19.0 * (1.0 - fr))
        return out


# ── ReplayBuffer ───────────────────────────────────────────────────────────
def test_replay_buffer_fifo_cap():
    """定容 FIFO：滿了丟最舊，留最近 maxlen 筆。"""
    rb = ReplayBuffer(maxlen=3)
    for i in range(5):
        rb.add(torch.full((2, 2), float(i)), torch.tensor([float(i)]))
    assert len(rb) == 3
    firsts = [rb[i][0][0, 0].item() for i in range(3)]
    assert firsts == [2.0, 3.0, 4.0]                  # 最舊 0,1 被淘汰


def test_replay_buffer_detaches():
    """落地時 detach → 不抓著計算圖。"""
    rb = ReplayBuffer(maxlen=2)
    rb.add(torch.ones(2, requires_grad=True), torch.zeros(1))
    assert not rb[0][0].requires_grad


# ── replay 模式端到端 ───────────────────────────────────────────────────────
def test_run_training_replay_mode_runs(tmp_path):
    """mode=replay：閉迴路跑得起來、SM 走回放分支、無 NaN。"""
    cfg = load_config(os.path.join(FIX, "single_test.yaml"))
    cfg.sm_train.update(mode="replay", newest_steps=5, replay_size=16)
    state = run_training(cfg, simulator=_Mock(("S11", "Gain")),
                         record_path=tmp_path, seed=0, verbose=False)
    last = state.last("sim_loss")
    assert last == last                                # 非 NaN
    assert int(state.last("epoch")) >= 1


def test_config_accepts_replay_and_rad_cap_keys():
    """白名單接受 sm_train.{mode,newest_steps,replay_size} 與 radiation.{sm_max_epoch,sm_min_loss}。"""
    cfg = TrainConfig(
        name="x", port="single", targets=SINGLE_TARGETS,
        sm_train={"mode": "replay", "newest_steps": 5, "replay_size": 16},
        radiation={"enable": True, "n_theta": 9, "sm_max_epoch": 1000, "sm_min_loss": 1.0},
    )
    assert cfg.sm_train["mode"] == "replay"
    assert cfg.radiation["sm_max_epoch"] == 1000
