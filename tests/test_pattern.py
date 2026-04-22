"""AntennaPattern 核心類別的純單元測試。

不依賴 HFSS 模擬器，僅覆蓋：
- fill_rate：全 0、全 1、半滿、以及空 pattern 的填充率計算
- merge：多個 sub-pattern 合併後的 shape 與值、空 patterns 應 raise
- mutate：in-place 翻轉 pixel 的行為
- binarize / binarization：硬性與 STE 可微分二值化
- setDefaultCoordinate：class-level 狀態 round-trip 與型別檢查
- copy / __add__ / __getitem__ / __len__ / __str__ / __invert__ 等容器行為
- input_dim / size_converter / getRandomPattern 等公開 helper
- simulate 在未註冊 simulator 時應 raise
"""

from __future__ import annotations

import pytest
import torch

from antenna.core.pattern import AntennaPattern


@pytest.fixture
def coordinate_4x4():
    """設定 4x4 的 default coordinate，並在測試結束後還原。"""
    prev = getattr(AntennaPattern, "_antenna_pattern_coordinate", None)
    AntennaPattern.setDefaultCoordinate((0, 4, 0, 4))
    yield (0, 4, 0, 4)
    if prev is None:
        if hasattr(AntennaPattern, "_antenna_pattern_coordinate"):
            delattr(AntennaPattern, "_antenna_pattern_coordinate")
    else:
        AntennaPattern._antenna_pattern_coordinate = prev  # type: ignore[attr-defined]


# --- fill_rate ---------------------------------------------------------------


def test_fill_rate_all_zeros(coordinate_4x4):
    pattern = AntennaPattern(torch.zeros(4, 4), coordinate_4x4)
    assert pattern.fill_rate == pytest.approx(0.0)


def test_fill_rate_all_ones(coordinate_4x4):
    pattern = AntennaPattern(torch.ones(4, 4), coordinate_4x4)
    assert pattern.fill_rate == pytest.approx(1.0)


def test_fill_rate_half_filled(coordinate_4x4):
    tensor = torch.zeros(4, 4)
    tensor[:2, :] = 1.0  # 上半部為 1
    pattern = AntennaPattern(tensor, coordinate_4x4)
    assert pattern.fill_rate == pytest.approx(0.5)


# --- merge -------------------------------------------------------------------


def test_merge_single_pattern_shape_and_value(coordinate_4x4):
    tensor = torch.ones(4, 4)
    pattern = AntennaPattern(tensor, coordinate_4x4)
    merged = pattern.merge()
    assert merged.shape == (4, 4)
    assert torch.allclose(merged.cpu(), tensor)


def test_merge_two_patterns_overlay():
    """後加入的 pattern 會覆蓋前面的 pattern。"""
    base = AntennaPattern(torch.zeros(4, 4), (0, 4, 0, 4))
    overlay = AntennaPattern(torch.ones(2, 2), (0, 2, 0, 2))
    combined = base + overlay

    merged = combined.merge()
    assert merged.shape == (4, 4)
    # 左上 2x2 區塊應被覆蓋為 1
    assert torch.allclose(merged[:2, :2].cpu(), torch.ones(2, 2))
    # 其餘仍為 0
    assert merged[2:, :].sum().item() == 0
    assert merged[:, 2:].sum().item() == 0


def test_merge_empty_patterns_raises():
    """空 patterns 呼叫 merge 應 raise ValueError。"""
    empty = AntennaPattern([])
    with pytest.raises(ValueError, match="No patterns to merge"):
        empty.merge()


# --- mutate ------------------------------------------------------------------


def test_mutate_flips_pixels_in_place(coordinate_4x4):
    torch.manual_seed(42)
    original = torch.zeros(4, 4)
    pattern = AntennaPattern(original.clone(), coordinate_4x4)

    mutated = pattern.mutate(rate=0.5)
    merged = mutated.merge().cpu()

    # 翻轉後應為 {0, 1} 的二值張量
    unique = torch.unique(merged)
    assert set(unique.tolist()).issubset({0.0, 1.0})

    # rate=0.5 於 16 pixel 上應翻轉約 8 個（int(16*0.5) = 8）
    assert int(merged.sum().item()) == 8


def test_mutate_rate_zero_keeps_values(coordinate_4x4):
    original = torch.zeros(4, 4)
    original[0, 0] = 1.0
    pattern = AntennaPattern(original.clone(), coordinate_4x4)

    mutated = pattern.mutate(rate=0.0)
    assert torch.allclose(mutated.merge().cpu(), original)


# --- binarize (hard, gradient-free) ------------------------------------------


def test_binarize_hard_is_binary(coordinate_4x4):
    tensor = torch.tensor([[0.1, 0.6, 0.4, 0.9], [0.5, 0.2, 0.7, 0.3], [0.0, 1.0, 0.5, 0.5], [0.8, 0.5, 0.49, 0.51]])
    pattern = AntennaPattern(tensor, coordinate_4x4)
    hard = pattern.binarize(threshold=0.5)
    merged = hard.merge().cpu()
    # 結果應為 {0, 1}
    assert set(torch.unique(merged).tolist()).issubset({0.0, 1.0})
    # 門檻 0.5 為 >=，故 0.5 的位置應為 1
    assert merged[0, 1].item() == 1.0
    assert merged[0, 0].item() == 0.0


def test_binarize_preserves_non_square_shape():
    """binarize 必須正確處理非正方形 pattern（H 不等於 W）。"""
    # 2x3 (H=2, W=3)
    AntennaPattern.setDefaultCoordinate((0, 3, 0, 2))
    try:
        tensor = torch.tensor([[0.1, 0.9, 0.4], [0.6, 0.2, 0.8]])
        pattern = AntennaPattern(tensor, (0, 3, 0, 2))
        hard = pattern.binarize(threshold=0.5)
        merged = hard.merge().cpu()
        assert merged.shape == (2, 3)
    finally:
        delattr(AntennaPattern, "_antenna_pattern_coordinate")


# --- binarization (STE, soft + hard) -----------------------------------------


def test_binarization_outputs_are_binary(coordinate_4x4):
    logits = torch.randn(4, 4, requires_grad=True)
    binary = AntennaPattern.binarization(logits, tau=0.5)

    # forward 值應為 {0, 1}
    unique_vals = torch.unique(binary.detach().cpu())
    assert set(unique_vals.tolist()).issubset({0.0, 1.0})


def test_binarization_gradient_flows_through_ste(coordinate_4x4):
    logits = torch.randn(4, 4, requires_grad=True)
    binary = AntennaPattern.binarization(logits, tau=0.5)

    loss = binary.sum()
    loss.backward()

    # STE 應讓梯度可回傳至 logits
    assert logits.grad is not None
    assert not torch.all(logits.grad == 0)


def test_binarization_only_soft_is_differentiable(coordinate_4x4):
    logits = torch.randn(4, 4, requires_grad=True)
    soft = AntennaPattern.binarization(logits, tau=0.5, only_soft=True)
    # 軟輸出應介於 0 與 1 之間且 requires_grad
    assert soft.min().item() >= 0.0
    assert soft.max().item() <= 1.0
    assert soft.requires_grad


def test_binarization_tau_below_min_is_clamped(coordinate_4x4):
    """tau 過小 (< 1e-4) 應被 clamp 避免 steepness 爆炸。"""
    logits = torch.randn(4, 4, requires_grad=True)
    AntennaPattern.binarization(logits, tau=1e-10)
    assert AntennaPattern.tau >= 1e-4


def test_binarization_from_1d_logits(coordinate_4x4):
    """1D logits 應被 reshape 為 size() 所指定的形狀。"""
    logits = torch.randn(16, requires_grad=True)
    binary = AntennaPattern.binarization(logits, tau=0.5)
    assert binary.shape == (4, 4)


def test_binarization_uses_explicit_threshold(coordinate_4x4):
    """顯式指定 threshold 時應使用該值而非 mean。"""
    logits = torch.full((4, 4), 0.0, requires_grad=True)
    # threshold 設很負，邏輯值幾乎全應為 1
    binary = AntennaPattern.binarization(logits, tau=0.1, threshold=-5.0)
    assert binary.mean().item() > 0.9


# --- setDefaultCoordinate class state round-trip ----------------------------


def test_set_default_coordinate_round_trip():
    prev = getattr(AntennaPattern, "_antenna_pattern_coordinate", None)
    try:
        coord = (0, 8, 0, 8)
        AntennaPattern.setDefaultCoordinate(coord)
        assert AntennaPattern._antenna_pattern_coordinate == coord  # type: ignore[attr-defined]
        assert AntennaPattern.size() == (8, 8)
        assert AntennaPattern.size(flatten=True) == 64
        assert AntennaPattern.getAllPixel() == 64
    finally:
        if prev is None:
            if hasattr(AntennaPattern, "_antenna_pattern_coordinate"):
                delattr(AntennaPattern, "_antenna_pattern_coordinate")
        else:
            AntennaPattern._antenna_pattern_coordinate = prev  # type: ignore[attr-defined]


def test_set_default_coordinate_rejects_non_tuple():
    with pytest.raises(TypeError):
        AntennaPattern.setDefaultCoordinate([0, 4, 0, 4])  # type: ignore[arg-type]


def test_set_default_coordinate_rejects_wrong_length():
    with pytest.raises(ValueError):
        AntennaPattern.setDefaultCoordinate((0, 4, 0))  # type: ignore[arg-type]


def test_size_without_coordinate_raises():
    """未設定 coordinate 直接呼叫 size() 應 raise RuntimeError。"""
    prev = getattr(AntennaPattern, "_antenna_pattern_coordinate", None)
    if hasattr(AntennaPattern, "_antenna_pattern_coordinate"):
        delattr(AntennaPattern, "_antenna_pattern_coordinate")
    try:
        with pytest.raises(RuntimeError, match="setDefaultCoordinate"):
            AntennaPattern.size()
    finally:
        if prev is not None:
            AntennaPattern._antenna_pattern_coordinate = prev  # type: ignore[attr-defined]


def test_get_all_pixel_without_coordinate_returns_zero():
    """未設定 coordinate 時 getAllPixel 應回傳 0，不 raise。"""
    prev = getattr(AntennaPattern, "_antenna_pattern_coordinate", None)
    if hasattr(AntennaPattern, "_antenna_pattern_coordinate"):
        delattr(AntennaPattern, "_antenna_pattern_coordinate")
    try:
        assert AntennaPattern.getAllPixel() == 0
    finally:
        if prev is not None:
            AntennaPattern._antenna_pattern_coordinate = prev  # type: ignore[attr-defined]


# --- 容器 / dunder 方法 ------------------------------------------------------


def test_init_with_non_tensor_non_list_raises():
    with pytest.raises(TypeError, match="Expected type for pattern"):
        AntennaPattern("not a tensor")  # type: ignore[arg-type]


def test_init_with_antenna_pattern_returns_same_object(coordinate_4x4):
    pattern = AntennaPattern(torch.zeros(4, 4), coordinate_4x4)
    wrapped = AntennaPattern(pattern)  # type: ignore[arg-type]
    assert wrapped is pattern


def test_init_without_coordinate_raises():
    """未提供 coordinate 且未設 default 應 raise ValueError。"""
    prev = getattr(AntennaPattern, "_antenna_pattern_coordinate", None)
    if hasattr(AntennaPattern, "_antenna_pattern_coordinate"):
        delattr(AntennaPattern, "_antenna_pattern_coordinate")
    try:
        with pytest.raises(ValueError, match="coordinate"):
            AntennaPattern(torch.zeros(4, 4))
    finally:
        if prev is not None:
            AntennaPattern._antenna_pattern_coordinate = prev  # type: ignore[attr-defined]


def test_len_equals_number_of_sub_patterns(coordinate_4x4):
    base = AntennaPattern(torch.zeros(4, 4), coordinate_4x4)
    overlay = AntennaPattern(torch.ones(2, 2), (0, 2, 0, 2))
    combined = base + overlay
    assert len(combined) == 2


def test_str_contains_shape_and_size(coordinate_4x4):
    pattern = AntennaPattern(torch.zeros(4, 4), coordinate_4x4)
    s = str(pattern)
    assert "AntennaPattern" in s
    assert "4" in s


def test_getitem_out_of_range_raises(coordinate_4x4):
    pattern = AntennaPattern(torch.zeros(4, 4), coordinate_4x4)
    with pytest.raises(IndexError):
        _ = pattern[99]


def test_getitem_returns_antenna_pattern(coordinate_4x4):
    base = AntennaPattern(torch.zeros(4, 4), coordinate_4x4)
    overlay = AntennaPattern(torch.ones(2, 2), (0, 2, 0, 2))
    combined = base + overlay
    second = combined[1]
    assert isinstance(second, AntennaPattern)
    assert second.merge().shape == (2, 2)


def test_add_with_non_pattern_raises(coordinate_4x4):
    pattern = AntennaPattern(torch.zeros(4, 4), coordinate_4x4)
    with pytest.raises(TypeError, match="Unsupported operand"):
        _ = pattern + 1  # type: ignore[operator]


def test_invert_returns_detached_cpu_tensor(coordinate_4x4):
    pattern = AntennaPattern(torch.ones(4, 4), coordinate_4x4)
    result = ~pattern
    assert not result.requires_grad
    assert result.device.type == "cpu"


def test_copy_creates_new_instance(coordinate_4x4):
    pattern = AntennaPattern(torch.ones(4, 4), coordinate_4x4)
    cloned = pattern.copy()
    assert cloned is not pattern
    assert torch.allclose(cloned.merge().cpu(), pattern.merge().cpu())


def test_series_is_flat_1d(coordinate_4x4):
    pattern = AntennaPattern(torch.ones(4, 4), coordinate_4x4)
    series = pattern.series
    assert series.dim() == 1
    assert series.numel() == 16


# --- input_dim / size_converter / getRandomPattern --------------------------


def test_input_dim_2d(coordinate_4x4):
    pattern = AntennaPattern(torch.zeros(4, 4), coordinate_4x4)
    assert pattern.input_dim() == 2


def test_input_dim_1d(coordinate_4x4):
    pattern = AntennaPattern(torch.zeros(16), coordinate_4x4)
    assert pattern.input_dim() == 1


def test_input_dim_multilayer_raises(coordinate_4x4):
    """多層 (list 初始化) 無 input_tensor，input_dim 應 raise。"""
    base = AntennaPattern(torch.zeros(4, 4), coordinate_4x4)
    overlay = AntennaPattern(torch.ones(2, 2), (0, 2, 0, 2))
    combined = base + overlay
    with pytest.raises(RuntimeError, match="multilayer"):
        combined.input_dim()


def test_get_random_pattern_fill_rate():
    torch.manual_seed(0)
    import numpy as np

    np.random.seed(0)
    pattern = AntennaPattern.getRandomPattern((10, 10), fill_rate=0.3)
    merged = pattern.merge().cpu()
    # 30 / 100 = 0.3
    assert int(merged.sum().item()) == 30
    assert set(torch.unique(merged).tolist()).issubset({0.0, 1.0})


# --- simulate 錯誤路徑 -------------------------------------------------------


def test_simulate_without_simulator_raises(coordinate_4x4):
    pattern = AntennaPattern(torch.zeros(4, 4), coordinate_4x4)
    # 清除可能由其他測試設置的 _simulator
    if "_simulator" in AntennaPattern.__dict__:
        delattr(AntennaPattern, "_simulator")
    with pytest.raises(RuntimeError, match="register_simulator"):
        pattern.simulate()
