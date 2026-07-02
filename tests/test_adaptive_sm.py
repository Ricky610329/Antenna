"""
tests/test_adaptive_sm.py — AdaptiveSMTrainController 純邏輯 + config 驗證測試（不碰 NAS/模型）。

自適應 SM 訓練量控制器：以 held-out fresh 點量泛化、自調每輪 SM 重訓 epoch 數（見 docs/discuss/decisions.md）。
"""
import pytest

from antenna.training import AdaptiveSMTrainController, TrainConfig


def _ctrl(**kw):
    base = dict(enable=True, snapshots=5, epoch_min=1, epoch_max=64, ema=0.3)
    base.update(kw)
    return AdaptiveSMTrainController(**base)


def _targets():
    return {"S11": {"side": 0, "center": -10, "width": [5, 0, 7, 0, 5], "method": "low"},
            "Gain": {"side": -19, "center": 4, "width": [5, 0, 7, 0, 5], "method": "high"}}


def test_disabled_is_inert():
    """enable=False → schedule 空、target=fallback、observe 不改變任何東西（golden 安全）。"""
    c = AdaptiveSMTrainController(enable=False, fallback_epochs=1)
    assert c.schedule() == []
    assert c.target_epochs() == 1
    c.observe({1: 5.0, 64: 2.0})
    assert c.target_epochs() == 1


def test_schedule_log_spaced_in_range():
    c = _ctrl()
    s = c.schedule()
    assert 2 <= len(s) <= 5                     # 去重後可能 <5
    #? target 內部存 float (死鎖修復)：頂點快照要對齊「實際會訓的整數 epoch 數」= target_epochs()。
    assert s == sorted(s) and s[0] >= 1 and s[-1] <= c.target_epochs()
    assert all(isinstance(x, int) for x in s)


def test_target_init_geomid():
    c = _ctrl(epoch_min=1, epoch_max=100)
    assert 1 < c.target_epochs() < 100          # 幾何中點


def test_observe_boundary_increases_target():
    """最多訓的快照泛化最好(邊界) → target 加碼探更多。"""
    c = _ctrl(ema=1.0)                           # ema=1 → 立即反應、好斷言
    t0 = c.target_epochs()
    errs = {ep: (100.0 - ep) for ep in c.schedule()}   # 越多訓誤差越低 → argmin=最大點(邊界)
    c.observe(errs)
    assert c.target_epochs() > t0


def test_observe_inside_moves_toward_argmin():
    """中間某訓練量泛化最好(U 型) → target 移向該點、不一味加碼。"""
    c = _ctrl(ema=1.0)
    sched = c.schedule()
    mid = sched[len(sched) // 2]
    errs = {ep: abs(ep - mid) + 1.0 for ep in sched}   # argmin=mid
    c.observe(errs)
    assert abs(c.target_epochs() - mid) <= 1


def test_observe_recovers_from_low_target_deadlock():
    """回歸 (2026-07-02)：早期雜訊把 target 壓低後，「多訓比較好」的真相要能把 target 拉回來。
    舊實作兩個成因鎖死在 2：(1) 整數 target 上做 EMA、1.3× 加碼被 round() 吃掉 (target≤5 吸收態)；
    (2) argmin 在歷史所有桶選、範圍外的過期低值永遠投票。修法：target 存 float + 只有本輪觀測的桶投票。"""
    c = _ctrl(epoch_min=1, epoch_max=32)
    for _ in range(5):                                   # 5 輪誤導：越少訓誤差越低 → target 被壓到底
        c.observe({ep: 1.0 + 0.1 * ep for ep in c.schedule()})
    assert c.target_epochs() <= 3                        # 確認真的掉下去了 (前置條件)
    for _ in range(40):                                  # 之後 40 輪真相：越多訓誤差越低
        c.observe({ep: 10.0 - 0.1 * ep for ep in c.schedule()})
    assert c.target_epochs() >= 8                        # 舊實作卡死在 2；修後要爬得回來


def test_observe_pure_noise_stays_conservative():
    """探測完全沒資訊 (誤差與訓練量無關) → target 保守往低走、不暴走 (=退回 dlf 現狀的安全失效模式)。"""
    import itertools
    c = _ctrl(epoch_min=1, epoch_max=32)
    noise = itertools.cycle([1.3, 0.9, 1.1, 1.0, 1.2, 0.8, 1.05])
    for _ in range(40):
        c.observe({ep: next(noise) for ep in c.schedule()})
        assert c.epoch_min <= c.target_epochs() <= c.epoch_max   # 全程不出界
    assert c.target_epochs() <= 8                        # 不因雜訊暴衝到上界白燒訓練


def test_seed_target():
    """斷點續跑：seed_target 續上次 target (夾進 [epoch_min, epoch_max])；None/非有限/disable → 不動。"""
    c = _ctrl(epoch_min=1, epoch_max=32)
    c.seed_target(17.0)
    assert c.target_epochs() == 17
    c.seed_target(999)                                   # 出界 (如 config 改小了範圍) → 夾回
    assert c.target_epochs() == 32
    c.seed_target(None)
    assert c.target_epochs() == 32                       # 不動
    c.seed_target(float("nan"))
    assert c.target_epochs() == 32                       # 不動
    off = AdaptiveSMTrainController(enable=False, fallback_epochs=1)
    off.seed_target(17.0)
    assert off.target_epochs() == 1                      # disable → 惰性


def test_observe_ignores_empty_and_nonfinite():
    c = _ctrl(ema=1.0)
    t0 = c.target_epochs()
    c.observe({})
    assert c.target_epochs() == t0
    c.observe({1: float("nan"), 5: float("inf")})
    assert c.target_epochs() == t0


def test_probe_stats():
    c = _ctrl()
    st = c.probe_stats({2: 3.0, 8: 1.0, 30: 5.0})
    assert st["probe_argmin"] == 8.0 and st["probe_min_err"] == 1.0 and st["probe_max_err"] == 5.0
    assert c.probe_stats({}) == {}


def test_bad_ema_rejected():
    with pytest.raises(ValueError, match="ema"):
        AdaptiveSMTrainController(enable=True, ema=0.0)
    with pytest.raises(ValueError, match="ema"):
        AdaptiveSMTrainController(enable=True, ema=1.5)


def test_adaptive_run_completes_and_logs(tmp_path):
    """mock 整合：adaptive run 跑得完、每 epoch 記 sm_train_epochs、held-out 探測有作用 (probe_argmin 出現)、無 NaN。"""
    import csv
    import math
    from antenna.training import run_training
    from test_baseline_loop import _MockSim
    cfg = TrainConfig(name="t_adaptive", port="single", targets=_targets(),
                      lr=0.005, sm_train={"mode": "adaptive", "lr": 0.001},
                      adaptive={"snapshots": 3, "epoch_min": 1, "epoch_max": 4, "ema": 0.5})
    gl = []
    run_training(cfg, simulator=_MockSim(("S11", "Gain")), record_path=tmp_path, seed=0,
                 max_epochs=5, on_epoch=lambda e, m: gl.append(m["gen_loss"]), verbose=False)
    assert len(gl) == 5 and all(math.isfinite(x) for x in gl)     # 跑完、gen_loss 有限 (無 NaN/爆炸)
    with open(tmp_path / "metrics.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert all(r["sm_train_epochs"] != "" for r in rows)          # 每 epoch 都記自適應訓練量
    assert any(r["probe_argmin"] != "" for r in rows)             # held-out 探測有跑到 (第 ≥2 個 fresh epoch)
    assert any(r["elite_n"] != "" for r in rows)                  # elite 集大小有落欄 (成本解讀用)
    #? hfss_calls = 累計真實模擬次數 (mock 每 epoch 都 fresh → 等於 epoch 序)、dense 每列都有
    calls = [float(r["hfss_calls"]) for r in rows]
    assert calls == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_config_adaptive_section_requires_mode():
    """設了 adaptive 區段但 mode 非 adaptive → fail-fast（比照 island_suppression 靜默沒開的教訓）。"""
    with pytest.raises(ValueError, match="adaptive"):
        TrainConfig(name="x", port="single", targets=_targets(), adaptive={"snapshots": 5})
    # mode: adaptive + adaptive 區段 → OK；adaptive 進了 SM_MODES
    cfg = TrainConfig(name="x", port="single", targets=_targets(),
                      sm_train={"mode": "adaptive"}, adaptive={"snapshots": 5, "epoch_max": 32})
    assert cfg.sm_train["mode"] == "adaptive"
    # adaptive 區段打錯鍵 → 白名單 raise
    with pytest.raises(ValueError, match="adaptive"):
        TrainConfig(name="x", port="single", targets=_targets(),
                    sm_train={"mode": "adaptive"}, adaptive={"snapshotss": 5})
