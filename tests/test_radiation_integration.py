"""
tests/test_radiation_integration.py — 方向圖接進閉迴路 (Stage 2) 的整合測試，全程無 HFSS。

涵蓋：
- HFSSNet 方向圖頭：forward(freq) 不變、forward_rad 形狀對、無頭時報錯。
- **golden 安全的硬證明**：同 seed 下「有/無 rad head」的 freq 參數 (fc_patch) 逐一相同
  → 加方向圖頭不會擾動既有 S11/Gain 數值。
- TrainConfig 的 radiation 區段白名單。
- 端到端：用 mock 方向圖模擬器把 run_training 跑完，確認方向圖頭被訓練、rad_loss 有記錄。
"""
import pytest
import torch

from antenna.models.surrogates import HFSSNet
from antenna.training import TrainConfig, run_training


SINGLE_TARGETS = {
    "S11":  {"side": 0,   "center": -10, "width": [5, 0, 7, 0, 5], "method": "low"},
    "Gain": {"side": -19, "center": 4,   "width": [5, 0, 7, 0, 5], "method": "high"},
}


# ── HFSSNet 方向圖頭 ────────────────────────────────────────────────────────
def test_forward_unchanged_and_forward_rad_shape():
    net = HFSSNet(625, (2, 17), rad_response=(2, 9))
    x = torch.randn(625)
    assert tuple(net(x).shape) == (2, 17)              # freq 路徑形狀不變
    assert tuple(net.forward_rad(x).shape) == (2, 9)   # 方向圖頭 (phi0/phi90, n_theta)


def test_forward_rad_raises_without_head():
    net = HFSSNet(625, (2, 17))                         # 不給 rad_response → 無方向圖頭
    assert net.head_rad is None
    with pytest.raises(RuntimeError):
        net.forward_rad(torch.randn(625))


def test_rad_head_does_not_perturb_freq_params():
    """golden 安全的硬證明：同 seed 下，加不加 rad head，fc_patch (freq) 參數逐一相同。"""
    torch.manual_seed(0)
    plain = HFSSNet(625, (2, 17))
    torch.manual_seed(0)
    withrad = HFSSNet(625, (2, 17), rad_response=(2, 181))
    pp = list(plain.fc_patch.parameters())
    pr = list(withrad.fc_patch.parameters())
    assert len(pp) == len(pr)
    for a, b in zip(pp, pr):
        assert torch.equal(a, b)                       # freq backbone+head 參數完全相同
    assert plain.head_rad is None and withrad.head_rad is not None


# ── TrainConfig radiation 白名單 ────────────────────────────────────────────
def test_config_accepts_radiation_section():
    cfg = TrainConfig(name="x", port="single", targets=SINGLE_TARGETS,
                      radiation={"enable": True, "n_theta": 9, "window_deg": 55})
    assert cfg.radiation["enable"] is True


def test_config_rejects_unknown_radiation_key():
    with pytest.raises(ValueError):
        TrainConfig(name="x", port="single", targets=SINGLE_TARGETS,
                    radiation={"enabel": True})        # 打錯鍵 → 報錯，不默默吃


def test_config_accepts_freeze_trunk():
    cfg = TrainConfig(name="x", port="single", targets=SINGLE_TARGETS,
                      radiation={"enable": True, "n_theta": 9, "freeze_trunk": False})
    assert cfg.radiation["freeze_trunk"] is False


# ── 凍 trunk：方向圖訓練不污染 S11/Gain backbone（修 NaN 爆炸的核心）──────────
def _split_params(model):
    trunk = {n: p for n, p in model.named_parameters() if not n.startswith("head_rad")}
    head = {n: p for n, p in model.named_parameters() if n.startswith("head_rad")}
    return trunk, head


def test_rad_freeze_trunk_protects_backbone(tmp_path):
    """freeze_trunk=True：train_one_data_rad 只動 head_rad，fc_patch（共用 backbone）逐一不變。"""
    from antenna.models.surrogates import MLPSurrogate
    sm = MLPSurrogate(tmp_path / "ck", 625, (2, 17), rad_response=(2, 9), max_epoch=5)
    trunk0, head0 = _split_params(sm.model)
    trunk0 = {n: p.detach().clone() for n, p in trunk0.items()}
    head0 = {n: p.detach().clone() for n, p in head0.items()}

    target = torch.rand(2, 9) * 10 + 5      # 量級夠大 → loss>min_loss，確實會訓練
    sm.train_one_data_rad(torch.rand(625), target, max_epoch=5, freeze_trunk=True, verbose=False)

    trunk1, head1 = _split_params(sm.model)
    for n, p in trunk1.items():
        assert torch.equal(trunk0[n], p), f"凍 trunk 失敗：{n} 被改動"
    assert any(not torch.equal(head0[n], p) for n, p in head1.items()), "head_rad 沒被訓練"


def test_rad_unfrozen_updates_trunk(tmp_path):
    """freeze_trunk=False：梯度會回流共用 backbone → fc_patch[:-1] 至少一個參數改變（證明旗標有效）。"""
    from antenna.models.surrogates import MLPSurrogate
    sm = MLPSurrogate(tmp_path / "ck", 625, (2, 17), rad_response=(2, 9), max_epoch=5)
    trunk0, _ = _split_params(sm.model)
    trunk0 = {n: p.detach().clone() for n, p in trunk0.items()}

    target = torch.rand(2, 9) * 10 + 5
    sm.train_one_data_rad(torch.rand(625), target, max_epoch=5, freeze_trunk=False, verbose=False)

    trunk1, _ = _split_params(sm.model)
    assert any(not torch.equal(trunk0[n], p) for n, p in trunk1.items()), "不凍時 backbone 應被更新"


# ── 端到端 (mock 方向圖模擬器，無 HFSS) ─────────────────────────────────────
class _MockRadSim:
    """假的方向圖模擬器：回傳 S11/Gain dict (同 baseline mock) + 設 last_radiation。"""
    def __init__(self, labels, n_theta):
        self.labels = labels
        self.theta = torch.linspace(-90, 90, n_theta)
        self.last_radiation = None
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
            else:
                out[lbl] = 4.0 * bump * fr - 19.0 * (1.0 - fr)
        #? 方向圖：以 theta 為軸的鐘形，boresight(0°) 最高 (確定性、可微標籤)
        g = 4.0 * fr * torch.exp(-(self.theta ** 2) / (2 * 40.0 ** 2)) - 19.0 * (1.0 - fr)
        self.last_radiation = {"theta": self.theta, "phi0": g, "phi90": g * 0.95}
        return out


def test_radiation_loop_runs_and_logs_rad_loss(tmp_path):
    cfg = TrainConfig(
        name="test_rad", port="single", epochs=3, patience=50,
        sm_train={"min_loss": 0.5, "max_epoch": 5},
        targets=SINGLE_TARGETS,
        radiation={"enable": True, "weight": 1.0, "window_deg": 55,
                   "floor_db": 3, "warmup_epochs": 0, "n_theta": 9},
    )
    sim = _MockRadSim(("S11", "Gain"), n_theta=9)
    state = run_training(cfg, simulator=sim, record_path=tmp_path, seed=0, verbose=False)

    assert int(state.last("epoch")) == 3
    assert sim.calls["start"] == 3
    rl = state.last("rad_loss", None)
    assert rl is not None, "rad_loss 沒被記錄 → 方向圖分支沒跑"
    assert rl == rl and rl >= 0.0, "rad_loss 應為非負且非 NaN"


def test_radiation_loop_with_mirror_generator(tmp_path):
    """mirror + rad 整合：對稱生成器在方向圖閉迴路裡跑得起來、rad_loss 有記、pattern 左右對稱。"""
    cfg = TrainConfig(
        name="x", port="single", epochs=2, patience=50,
        sm_train={"min_loss": 0.5, "max_epoch": 3},
        generator={"name": "mirror"},
        targets=SINGLE_TARGETS,
        radiation={"enable": True, "n_theta": 9, "warmup_epochs": 0},
    )
    sim = _MockRadSim(("S11", "Gain"), n_theta=9)
    state = run_training(cfg, simulator=sim, record_path=tmp_path, seed=0, verbose=False)
    assert int(state.last("epoch")) == 2
    assert state.last("rad_loss", None) is not None                 # rad 分支有跑
    patt, _ = state.pattern_at(state.best_epoch("sim_loss"))         # merged 25x25 (含對稱 feed)
    assert torch.equal(patt, patt.flip(dims=[1]))                    # 整張左右對稱（撐過二值化+feed）


# ── 部分載入：舊 sm.pth (無 head_rad) → strict=False 暖啟動 rad 版 SM ──────────
#    這修的是「HFSS 還沒開始收集資料就卡住」的根因：rad 版 SM 不能用 pretrained 時，
#    會退回 offline_dataset 在數萬筆上從零預訓練 → 卡死。strict=False 讓它能秒載暖啟動。
def test_pre_load_partial_warm_starts_rad_sm(tmp_path):
    from antenna.models.surrogates import MLPSurrogate
    plain = MLPSurrogate(tmp_path / "plain", 625, (2, 17))            # 舊 SM (無方向圖頭)
    ckpt = plain.save_as(tmp_path / "old_sm.pth")
    rad = MLPSurrogate(tmp_path / "rad", 625, (2, 17), rad_response=(2, 9))   # rad 版 (多 head_rad)

    rad.pre_load_model(ckpt, strict=False)                           # 部分載入：不該報錯

    # 共用 trunk/freq head (fc_patch) 權重應被灌成舊 SM 的值
    for a, b in zip(plain.model.fc_patch.parameters(), rad.model.fc_patch.parameters()):
        assert torch.equal(a.detach().cpu(), b.detach().cpu())
    # head_rad 仍在、維持有限的隨機初始 (沒被汙染)
    assert rad.model.head_rad is not None
    for p in rad.model.head_rad.parameters():
        assert torch.all(torch.isfinite(p))


def test_pre_load_partial_warm_starts_optimizer(tmp_path):
    """strict=False 也要暖啟動 optimizer 狀態：冷 optimizer + 暖權重會讓 train_one_data
    第一步過衝、3 步發散爆 NaN（實測 2136→90→4.9e6→inf）。trunk 的 per-param 狀態要對位灌入。"""
    from antenna.models.surrogates import MLPSurrogate
    plain = MLPSurrogate(tmp_path / "p", 625, (2, 17))
    plain.train_one_data(torch.rand(625), torch.rand(2, 17), max_epoch=3, verbose=False)  # 累積 optimizer 狀態
    n_saved = len(plain.optimizer.state)
    assert n_saved > 0
    ckpt = plain.save_as(tmp_path / "p.pth")

    rad = MLPSurrogate(tmp_path / "r", 625, (2, 17), rad_response=(2, 9))
    assert len(rad.optimizer.state) == 0                 # 載入前 optimizer 全冷
    rad.pre_load_model(ckpt, strict=False)
    assert len(rad.optimizer.state) == n_saved           # trunk 狀態被對位暖啟動 (head_rad 保持新鮮)


def test_pre_load_strict_still_rejects_subset(tmp_path):
    """回歸保護：strict=True (預設) 載缺 head_rad 的檔 → 報錯，不默默吃 (其他實驗行為不變)。"""
    from antenna.models.surrogates import MLPSurrogate
    plain = MLPSurrogate(tmp_path / "plain2", 625, (2, 17))
    ckpt = plain.save_as(tmp_path / "old_sm2.pth")
    rad = MLPSurrogate(tmp_path / "rad2", 625, (2, 17), rad_response=(2, 9))
    with pytest.raises(RuntimeError):
        rad.pre_load_model(ckpt)                                     # 預設 strict=True → 缺鍵報錯


# ── 端到端：極端方向圖 target（深零點 -inf/巨大負值）不該炸掉整個 run ──────────
class _ExtremeRadSim(_MockRadSim):
    """模擬 HFSS 深零點：phi0 注入 -inf / -1e9（dB 深 null），驗證清理 + 防護網扛得住。"""
    def __call__(self, pattern, **kw):
        out = super().__call__(pattern, **kw)
        bad = self.last_radiation["phi0"].clone()
        bad[0] = float("-inf"); bad[1] = -1e9
        self.last_radiation["phi0"] = bad
        return out


def test_radiation_loop_survives_extreme_targets(tmp_path):
    cfg = TrainConfig(
        name="x", port="single", epochs=2, patience=50,
        sm_train={"min_loss": 0.5, "max_epoch": 3}, targets=SINGLE_TARGETS,
        radiation={"enable": True, "n_theta": 9, "warmup_epochs": 0},
    )
    sim = _ExtremeRadSim(("S11", "Gain"), n_theta=9)
    state = run_training(cfg, simulator=sim, record_path=tmp_path, seed=0, verbose=False)
    assert int(state.last("epoch")) == 2          # 沒被 -inf rad target 炸掉
    assert torch.isfinite(torch.tensor(state.last("gen_loss")))
