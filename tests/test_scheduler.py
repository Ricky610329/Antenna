"""AdaptiveCyclicalScheduler 單元測試。

涵蓋：
- warmup 線性上升與 cosine 衰減
- temperature schedule 形狀
- `_is_metric_better` 在 min / max 模式下的比較
- `state_dict` / `load_state_dict` round-trip
- `on_plateau` 三種模式（reset / peak / linear）
- tau callback 可以注入自訂 callable（解耦驗證）
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from antenna.schedulers import AdaptiveCyclicalScheduler

# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------


def _make_optimizer():
    """建立一個極簡的 optimizer 以便測試 scheduler。"""
    return torch.optim.SGD(nn.Linear(2, 2).parameters(), lr=0.0)


def _noop(_tau: float) -> None:
    """no-op tau callback，避免測試觸碰全域 AntennaPattern.tau 狀態。"""


def _make_scheduler(tau_callback=None, **kwargs):
    """建立 scheduler，預設參數簡化讓驗證較直觀 (T_0=20, warmup_ratio=0.25 → 5 步 warmup)。"""
    defaults = dict(
        T_0=20,
        T_mult=1,
        lr_max=1.0,
        lr_min=0.0,
        temp_max=10.0,
        temp_min=1.0,
        warmup_ratio=0.25,
        mode="min",
        factor=0.5,
        patience=3,
        on_plateau="peak",
    )
    defaults.update(kwargs)
    optimizer = _make_optimizer()
    scheduler = AdaptiveCyclicalScheduler(optimizer, tau_callback=tau_callback or _noop, **defaults)
    return scheduler, optimizer


# ---------------------------------------------------------------------------
# Warmup & cosine 形狀
# ---------------------------------------------------------------------------


def test_warmup_lr_linearly_increases():
    """warmup 階段 lr 應該從 lr_min 線性上升至 lr_max。"""
    scheduler, optimizer = _make_scheduler(T_0=20, warmup_ratio=0.25, lr_min=0.0, lr_max=1.0)
    # warmup_steps = int(20 * 0.25) = 5；T_cur 從 0 開始至 5 即結束 warmup
    lrs = []
    # 不傳 metric 會觸發 warnings，但不影響邏輯。包起來避免 noise。
    for _ in range(6):
        with pytest.warns(UserWarning):
            scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    # 前 5 步（T_cur = 1..5）處於 warmup 結束前，lr 線性遞增
    # T_cur=1 → 1/5=0.2, T_cur=2 → 0.4, ... T_cur=4 → 0.8, T_cur=5 → cosine 起點 (=lr_max)
    for i in range(4):
        assert lrs[i + 1] > lrs[i], f"warmup step {i} lr 未遞增: {lrs}"
    # 接近線性：每一步增量大致相等
    diffs = [lrs[i + 1] - lrs[i] for i in range(4)]
    for d in diffs:
        assert math.isclose(d, diffs[0], rel_tol=1e-6), f"warmup 非線性: {diffs}"


def test_cosine_phase_decays():
    """cosine 退火階段 lr 應該從 lr_max 衰減至 lr_min。"""
    scheduler, optimizer = _make_scheduler(T_0=20, warmup_ratio=0.25, lr_min=0.0, lr_max=1.0)
    lrs = []
    # 跑完整個週期（T_0 = 20 步）
    for _ in range(20):
        with pytest.warns(UserWarning):
            scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    # warmup_steps=5，T_cur=5 時應在峰值附近（cosine_progress=0 → lr=lr_max）
    peak_idx = 4  # lrs[4] 對應 T_cur=5
    assert math.isclose(lrs[peak_idx], 1.0, rel_tol=1e-6)

    # cosine 階段：後半應該遞減
    cosine_lrs = lrs[peak_idx:]
    for i in range(len(cosine_lrs) - 1):
        assert cosine_lrs[i] >= cosine_lrs[i + 1] - 1e-9, (
            f"cosine 階段未單調遞減: step {i}: {cosine_lrs[i]} -> {cosine_lrs[i + 1]}"
        )


def test_temperature_schedule_shape():
    """溫度應該跟隨 lr 做相同形狀的排程：warmup 上升 + cosine 衰減。"""
    scheduler, _ = _make_scheduler(T_0=20, warmup_ratio=0.25, temp_min=1.0, temp_max=10.0)
    temps = []
    for _ in range(20):
        with pytest.warns(UserWarning):
            scheduler.step()
        temps.append(scheduler.get_temp())

    # warmup 結束點（T_cur=5 → index 4）為峰值
    assert math.isclose(temps[4], 10.0, rel_tol=1e-6)
    # warmup 階段遞增
    for i in range(4):
        assert temps[i + 1] > temps[i]
    # cosine 階段最後一個應該很接近 temp_min
    assert temps[-1] < 2.0  # cosine progress 接近 1，值接近 temp_min


# ---------------------------------------------------------------------------
# _is_metric_better
# ---------------------------------------------------------------------------


def test_is_metric_better_min_mode():
    scheduler, _ = _make_scheduler(mode="min", threshold=0.0)
    scheduler.best_metric = 1.0
    assert scheduler._is_metric_better(0.5) is True
    assert scheduler._is_metric_better(1.0) is False
    assert scheduler._is_metric_better(1.5) is False


def test_is_metric_better_max_mode():
    scheduler, _ = _make_scheduler(mode="max", threshold=0.0)
    scheduler.best_metric = 1.0
    assert scheduler._is_metric_better(1.5) is True
    assert scheduler._is_metric_better(1.0) is False
    assert scheduler._is_metric_better(0.5) is False


def test_is_metric_better_with_threshold():
    """threshold 應該讓 `更好` 判定變嚴格。"""
    # min 模式
    scheduler, _ = _make_scheduler(mode="min", threshold=0.1)
    scheduler.best_metric = 1.0
    # metric < 1.0 - 0.1 = 0.9 才算更好
    assert scheduler._is_metric_better(0.89) is True
    assert scheduler._is_metric_better(0.91) is False

    # max 模式
    scheduler, _ = _make_scheduler(mode="max", threshold=0.1)
    scheduler.best_metric = 1.0
    # metric > 1.0 + 0.1 = 1.1 才算更好
    assert scheduler._is_metric_better(1.11) is True
    assert scheduler._is_metric_better(1.09) is False


# ---------------------------------------------------------------------------
# state_dict round-trip
# ---------------------------------------------------------------------------


def test_state_dict_round_trip():
    scheduler, _ = _make_scheduler()
    # 跑幾步累積狀態
    for metric in [0.9, 0.8, 0.85, 0.9, 0.95]:
        scheduler.step(metric)

    snapshot = scheduler.state_dict()

    # 建立一個全新 scheduler 並載入狀態
    new_scheduler, _ = _make_scheduler()
    new_scheduler.load_state_dict(snapshot)

    # 關鍵狀態應該保留
    assert new_scheduler.T_i == scheduler.T_i
    assert new_scheduler.T_cur == scheduler.T_cur
    assert new_scheduler.current_temp == scheduler.current_temp
    assert new_scheduler.patience_counter == scheduler.patience_counter
    assert new_scheduler.best_metric == scheduler.best_metric


def test_state_dict_contains_expected_keys():
    scheduler, _ = _make_scheduler()
    for metric in [0.9, 0.8]:
        scheduler.step(metric)
    state = scheduler.state_dict()
    for key in ("T_i", "T_cur", "current_temp", "patience_counter", "best_metric"):
        assert key in state, f"state_dict 缺少 key: {key}"


# ---------------------------------------------------------------------------
# on_plateau 三種模式
# ---------------------------------------------------------------------------


def _trigger_plateau(scheduler, n_steps=None):
    """連續送入變差的 metric 以觸發 plateau 邏輯。"""
    # patience 預設 3，送 3 次沒有改善的 metric 就會觸發
    n_steps = n_steps or scheduler.patience
    for _ in range(n_steps):
        # 明顯更差，保證 is_metric_better 為 False
        scheduler.step(metric=1e9)


def test_on_plateau_reset_mode():
    """`reset` 模式：觸發後應該回到週期起點 (T_cur == 0 after step)。"""
    scheduler, _ = _make_scheduler(on_plateau="reset", patience=3, T_0=20, factor=0.5)
    # 先讓 best_metric 有意義
    scheduler.step(metric=0.0)  # best_metric = 0.0
    _trigger_plateau(scheduler)

    # 觸發後 T_cur 會被設為 -1，再經過 step 尾端 += 1 → 0
    assert scheduler.T_cur == 0
    # T_i 應該被縮減（factor=0.5），但不能小於 T_0 // 2
    assert scheduler.T_i == max(int(20 * 0.5), 20 // 2)


def test_on_plateau_peak_mode():
    """`peak` 模式：觸發後應該跳到 warmup 結束點（最大值）。"""
    scheduler, _ = _make_scheduler(on_plateau="peak", patience=3, T_0=20, warmup_ratio=0.25, factor=1.0)
    scheduler.step(metric=0.0)
    _trigger_plateau(scheduler)

    # T_i 保持 20 (factor=1.0 不會縮減，但會與 T_0//2 取 max)
    expected_T_i = max(int(20 * 1.0), 20 // 2)
    assert scheduler.T_i == expected_T_i
    # warmup_steps = int(T_i * 0.25)
    expected_warmup = int(expected_T_i * 0.25)
    # step 尾端 T_cur += 1，所以應該正好 == warmup_steps
    assert scheduler.T_cur == expected_warmup


def test_on_plateau_linear_mode():
    """`linear` 模式：從當前 lr 回推 warmup 步數，T_cur 應在 [0, warmup_steps] 之間。"""
    scheduler, optimizer = _make_scheduler(
        on_plateau="linear", patience=3, T_0=20, warmup_ratio=0.25, factor=1.0, lr_min=0.0, lr_max=1.0
    )
    scheduler.step(metric=0.0)
    _trigger_plateau(scheduler)

    # 應該落在 [0, warmup_steps] 範圍，且 +1 後不會超過 warmup_steps + 1
    warmup_steps = int(scheduler.T_i * 0.25)
    # T_cur 已 +1，所以合法範圍是 [1, warmup_steps + 1]
    assert 0 <= scheduler.T_cur <= warmup_steps + 1
    assert optimizer.param_groups[0]["lr"] >= 0.0


def test_on_plateau_invalid_mode_rejected():
    """建構時若指定未知的 on_plateau 模式，應該 raise ValueError。"""
    with pytest.raises(ValueError):
        AdaptiveCyclicalScheduler(
            _make_optimizer(),
            T_0=10,
            on_plateau="nonsense",  # type: ignore[arg-type]
            tau_callback=_noop,
        )


# ---------------------------------------------------------------------------
# tau callback 注入（解耦驗證）
# ---------------------------------------------------------------------------


def test_custom_tau_callback_is_invoked():
    """自訂 tau_callback 每次 step 都應該被呼叫，並收到當前 temperature。"""
    captured: list[float] = []

    def custom_cb(tau: float):
        captured.append(tau)

    scheduler, _ = _make_scheduler(tau_callback=custom_cb)
    # _LRScheduler 在 __init__ 中會呼叫一次 step()，所以先把基線清掉
    baseline = len(captured)
    for _ in range(5):
        with pytest.warns(UserWarning):
            scheduler.step()

    assert len(captured) - baseline == 5
    # 最後一次 captured 應與當時的 get_temp() 一致
    assert captured[-1] == scheduler.get_temp()


def test_default_tau_callback_sets_antenna_pattern_tau():
    """未傳入 tau_callback 時，應該使用預設 callback 並更新 AntennaPattern.tau。"""
    from antenna import AntennaPattern

    # 記錄原始值以便還原
    original_tau = getattr(AntennaPattern, "tau", None)
    try:
        optimizer = _make_optimizer()
        scheduler = AdaptiveCyclicalScheduler(
            optimizer,
            T_0=10,
            lr_max=1.0,
            lr_min=0.0,
            temp_max=5.0,
            temp_min=0.5,
            warmup_ratio=0.2,
        )
        with pytest.warns(UserWarning):
            scheduler.step()
        assert AntennaPattern.tau == scheduler.get_temp()
    finally:
        if original_tau is not None:
            AntennaPattern.tau = original_tau


# ---------------------------------------------------------------------------
# 建構參數驗證
# ---------------------------------------------------------------------------


def test_invalid_T_0_raises():
    with pytest.raises(ValueError):
        AdaptiveCyclicalScheduler(_make_optimizer(), T_0=0, tau_callback=_noop)


def test_invalid_T_mult_raises():
    with pytest.raises(ValueError):
        AdaptiveCyclicalScheduler(_make_optimizer(), T_0=10, T_mult=0, tau_callback=_noop)


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        AdaptiveCyclicalScheduler(_make_optimizer(), T_0=10, mode="weird", tau_callback=_noop)
