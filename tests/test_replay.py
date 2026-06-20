"""
ReplayBuffer + SM 經驗回放模式 (sm_train.mode: replay) 的測試。

replay 是 opt-in 改良：把「最新一筆擬到收斂」反模式換成「最新少數步 + 回放緩衝一遍」。
預設 mode=single → 行為與原樣相同 (golden 由 test_baseline_loop 把關)。
"""
import os

import torch

from antenna.training import TrainConfig, load_config, run_training
from antenna.losses import boundary_loss
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
        rb.add(torch.full((2, 2), float(i)), torch.tensor([float(i)]), loss=float(i))
    assert len(rb) == 3
    firsts = [rb[i][0][0, 0].item() for i in range(3)]
    assert firsts == [2.0, 3.0, 4.0]                  # 最舊 0,1 被淘汰


def test_replay_buffer_detaches():
    """落地時 detach → 不抓著計算圖。"""
    rb = ReplayBuffer(maxlen=2)
    rb.add(torch.ones(2, requires_grad=True), torch.zeros(1), loss=0.0)
    assert not rb[0][0].requires_grad


def test_replay_buffer_elite_filters_by_loss():
    """DLF：elite(threshold) 只回傳 loss ≤ threshold 的菁英子集。"""
    rb = ReplayBuffer(maxlen=10)
    for i, lo in enumerate([5.0, 1.0, 8.0, 2.0, 9.0]):    # loss 各異
        rb.add(torch.full((2, 2), float(i)), torch.zeros(1), loss=lo)
    elite = rb.elite(threshold=3.0)                       # 留 loss ≤ 3 → 第 1(1.0)、3(2.0) 筆
    assert len(elite) == 2
    kept = sorted(s[0][0, 0].item() for s in elite)       # pattern 值 = 原 index
    assert kept == [1.0, 3.0]


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


def test_run_training_dlf_mode_runs(tmp_path):
    """mode=dlf：閉迴路跑得起來、走 DLF 菁英過濾分支、無 NaN。"""
    cfg = load_config(os.path.join(FIX, "single_test.yaml"))
    cfg.sm_train.update(mode="dlf", newest_steps=5, replay_size=16)
    state = run_training(cfg, simulator=_Mock(("S11", "Gain")),
                         record_path=tmp_path, seed=0, verbose=False)
    last = state.last("sim_loss")
    assert last == last and int(state.last("epoch")) >= 1


# ── boundary loss ───────────────────────────────────────────────────────────
def test_boundary_loss_distance_and_self_exclusion():
    """近已見→低、遠→高；且排除「與自己相同」那筆 (current 可能已在緩衝)。"""
    seen = torch.tensor([[0., 0., 0.], [1., 1., 1.]])
    near = torch.tensor([0.1, 0.0, 0.0])
    far = torch.tensor([0.5, 0.5, 0.5])
    assert boundary_loss(near, seen) < boundary_loss(far, seen)        # 越近越低
    same = torch.tensor([0., 0., 0.])                                  # == seen[0]
    assert boundary_loss(same, seen).item() > 0                        # 排除自己 → 取次近 (seen[1])


def test_boundary_loss_differentiable():
    p = torch.rand(9, requires_grad=True)
    boundary_loss(p, torch.rand(4, 9)).backward()                      # 不該報錯
    assert p.grad is not None


def test_run_training_boundary_runs(tmp_path):
    """dlf + boundary：閉迴路跑得起來、boundary 項有進 GEN loss、無 NaN。"""
    cfg = load_config(os.path.join(FIX, "single_test.yaml"))
    cfg.sm_train.update(mode="dlf", newest_steps=5, replay_size=16)
    cfg.loss["boundary"] = 0.1
    state = run_training(cfg, simulator=_Mock(("S11", "Gain")),
                         record_path=tmp_path, seed=0, verbose=False)
    gl = state.last("gen_loss")
    assert gl == gl and int(state.last("epoch")) >= 1                  # 非 NaN


def test_config_accepts_replay_and_rad_cap_keys():
    """白名單接受 sm_train.{mode,newest_steps,replay_size} 與 radiation.{sm_max_epoch,sm_min_loss}。"""
    cfg = TrainConfig(
        name="x", port="single", targets=SINGLE_TARGETS,
        sm_train={"mode": "replay", "newest_steps": 5, "replay_size": 16},
        radiation={"enable": True, "n_theta": 9, "sm_max_epoch": 1000, "sm_min_loss": 1.0},
    )
    assert cfg.sm_train["mode"] == "replay"
    assert cfg.radiation["sm_max_epoch"] == 1000
