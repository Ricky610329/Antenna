"""AntennaPattern 核心類別的純單元測試。

不依賴 HFSS 模擬器，僅覆蓋：
- fill_rate：全 0、全 1、半滿的填充率計算
- merge：多個 sub-pattern 合併後的 shape 與值
- mutate：in-place 翻轉 pixel 的行為
- binarization：輸出 {0, 1} 且梯度可經由 STE 回傳
- setDefaultCoordinate：class-level 狀態 round-trip
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


# --- binarization (STE) ------------------------------------------------------


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
