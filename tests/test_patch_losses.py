"""Patch 損失函數單元測試。

涵蓋 ``custom_loss_r`` / ``custom_loss_g`` / ``custom_loss_minmax`` 在以下面向的行為：

- 正常輸入的輸出 shape / dtype（回傳為 0 維 float32 張量）。
- 已知輸入的數值驗證。
- 空 mask / 無越界邊界情形（應回傳可微分的零張量）。
- backward 梯度可傳（不越界情形下梯度為 0，但可呼叫）。
- ``custom_loss_r`` / ``custom_loss_g`` 行為一致（thin wrapper）。
"""

from __future__ import annotations

import pytest
import torch

from antenna.losses.patch_losses import (
    custom_loss_boundary,
    custom_loss_g,
    custom_loss_minmax,
    custom_loss_r,
)

# ----- fixtures（inline 定義，不使用 conftest.py） -----


@pytest.fixture
def simple_target() -> torch.Tensor:
    """包含 high (4.0) 與 low (-10.0) 以及中間值的 1D target。"""
    return torch.tensor([4.0, -10.0, 0.0, 4.0, -10.0, 1.0])


@pytest.fixture
def violating_prediction() -> torch.Tensor:
    """在 target 對應位置上全部越界的 prediction：
    high 位置低於 high、low 位置高於 low。
    """
    return torch.tensor([2.0, -5.0, 0.0, 3.0, -8.0, 1.0], requires_grad=True)


@pytest.fixture
def exact_prediction() -> torch.Tensor:
    """與 target 完全相同的 prediction（無越界）。"""
    return torch.tensor([4.0, -10.0, 0.0, 4.0, -10.0, 1.0], requires_grad=True)


# ----- 形狀與型別測試 -----


def test_custom_loss_r_returns_scalar_float32(simple_target, violating_prediction):
    loss = custom_loss_r(violating_prediction, simple_target)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss.dtype == torch.float32


def test_custom_loss_g_returns_scalar_float32(simple_target, violating_prediction):
    loss = custom_loss_g(violating_prediction, simple_target)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss.dtype == torch.float32


def test_custom_loss_minmax_returns_scalar_float32(simple_target, violating_prediction):
    loss_high = custom_loss_minmax(violating_prediction, simple_target, method="high")
    loss_low = custom_loss_minmax(violating_prediction, simple_target, method="low")
    for loss in (loss_high, loss_low):
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.dtype == torch.float32


# ----- 數值測試（已知輸入） -----


def test_custom_loss_r_known_positive_when_violating(simple_target, violating_prediction):
    """當 prediction 同時在 high、low 邊界越界時，損失必為正。"""
    loss = custom_loss_r(violating_prediction, simple_target)
    assert loss.item() > 0


def test_custom_loss_r_equals_g_when_input_identical(simple_target, violating_prediction):
    """r 與 g 在數學上等價（僅語意不同），給相同輸入應得相同結果。"""
    loss_r = custom_loss_r(violating_prediction, simple_target)
    loss_g = custom_loss_g(violating_prediction, simple_target)
    assert torch.allclose(loss_r, loss_g)


def test_custom_loss_boundary_matches_wrappers(simple_target, violating_prediction):
    """底層 custom_loss_boundary 應與 r/g wrapper 結果相同。"""
    expected = custom_loss_r(violating_prediction, simple_target)
    got_r = custom_loss_boundary(violating_prediction, simple_target, side="r")
    got_g = custom_loss_boundary(violating_prediction, simple_target, side="g")
    assert torch.allclose(got_r, expected)
    assert torch.allclose(got_g, expected)


def test_custom_loss_minmax_high_low_sum_equals_boundary(simple_target, violating_prediction):
    """minmax(high) + minmax(low) 應等於 custom_loss_boundary。"""
    loss_high = custom_loss_minmax(violating_prediction, simple_target, method="high")
    loss_low = custom_loss_minmax(violating_prediction, simple_target, method="low")
    loss_boundary = custom_loss_boundary(violating_prediction, simple_target)
    assert torch.allclose(loss_high + loss_low, loss_boundary)


def test_custom_loss_minmax_high_smoothl1_manual():
    """手算驗證 SmoothL1 計算正確：target.max()=4.0，有一個越界元素 pred=2.0 => |4-2|=2 > 1 => 2-0.5=1.5。"""
    pred = torch.tensor([2.0, 0.0, 0.0, 4.0], requires_grad=True)
    target = torch.tensor([4.0, 0.0, 0.0, 4.0])
    loss = custom_loss_minmax(pred, target, method="high")
    # 只有 index=0 越界，SmoothL1(2.0, 4.0) = |2-4| - 0.5 = 1.5
    assert torch.allclose(loss, torch.tensor(1.5))


def test_custom_loss_minmax_mseloss_type():
    """驗證 loss_type='MSELoss' 分支正確。"""
    pred = torch.tensor([2.0, 0.0, 0.0, 4.0], requires_grad=True)
    target = torch.tensor([4.0, 0.0, 0.0, 4.0])
    loss = custom_loss_minmax(pred, target, method="high", loss_type="MSELoss")
    # MSE(2.0, 4.0) = (2-4)^2 / 1 = 4.0
    assert torch.allclose(loss, torch.tensor(4.0))


# ----- 邊界情形測試（空 mask / 無越界） -----


def test_custom_loss_r_zero_when_no_violation(simple_target, exact_prediction):
    """prediction 與 target 相等時，不越界，損失應為 0。"""
    loss = custom_loss_r(exact_prediction, simple_target)
    assert torch.allclose(loss, torch.tensor(0.0))


def test_custom_loss_g_zero_when_no_violation(simple_target, exact_prediction):
    loss = custom_loss_g(exact_prediction, simple_target)
    assert torch.allclose(loss, torch.tensor(0.0))


def test_custom_loss_minmax_zero_when_no_violation(simple_target, exact_prediction):
    loss_high = custom_loss_minmax(exact_prediction, simple_target, method="high")
    loss_low = custom_loss_minmax(exact_prediction, simple_target, method="low")
    assert torch.allclose(loss_high, torch.tensor(0.0))
    assert torch.allclose(loss_low, torch.tensor(0.0))


def test_custom_loss_minmax_zero_returns_grad_tensor(simple_target, exact_prediction):
    """無越界時回傳的零張量必須可微分（requires_grad=True）。"""
    loss = custom_loss_minmax(exact_prediction, simple_target, method="high")
    assert loss.requires_grad is True


def test_custom_loss_minmax_single_bound_mask():
    """target 中不存在某個邊界（例如僅有 low）時仍應正常運作。"""
    # target 全為同一值，max == min，violation 條件兩側都無
    pred = torch.tensor([1.0, 1.0], requires_grad=True)
    target = torch.tensor([1.0, 1.0])
    loss_high = custom_loss_minmax(pred, target, method="high")
    loss_low = custom_loss_minmax(pred, target, method="low")
    assert torch.allclose(loss_high, torch.tensor(0.0))
    assert torch.allclose(loss_low, torch.tensor(0.0))


# ----- 錯誤參數測試 -----


def test_custom_loss_minmax_invalid_method(simple_target, violating_prediction):
    with pytest.raises(ValueError, match="method"):
        custom_loss_minmax(violating_prediction, simple_target, method="middle")  # type: ignore[arg-type]


def test_custom_loss_boundary_invalid_side(simple_target, violating_prediction):
    with pytest.raises(ValueError, match="side"):
        custom_loss_boundary(violating_prediction, simple_target, side="x")  # type: ignore[arg-type]


def test_custom_loss_invalid_loss_type(simple_target, violating_prediction):
    with pytest.raises(ValueError, match="Unsupported loss_type"):
        custom_loss_r(violating_prediction, simple_target, loss_type="L1Loss")


# ----- 梯度可傳測試 -----


def test_custom_loss_r_backward_produces_gradient(simple_target, violating_prediction):
    """越界情形下，backward 應在越界位置產生非零梯度。"""
    loss = custom_loss_r(violating_prediction, simple_target)
    loss.backward()
    assert violating_prediction.grad is not None
    # 越界位置 (index 0, 1, 3, 4) 的梯度應非零
    grad = violating_prediction.grad
    assert grad.shape == violating_prediction.shape
    # 至少應有非零梯度
    assert (grad != 0).any()


def test_custom_loss_g_backward_produces_gradient(simple_target, violating_prediction):
    loss = custom_loss_g(violating_prediction, simple_target)
    loss.backward()
    assert violating_prediction.grad is not None
    assert (violating_prediction.grad != 0).any()


def test_custom_loss_minmax_backward_produces_gradient(simple_target, violating_prediction):
    loss = custom_loss_minmax(violating_prediction, simple_target, method="high")
    loss.backward()
    assert violating_prediction.grad is not None
    assert (violating_prediction.grad != 0).any()


def test_custom_loss_minmax_backward_when_no_violation(simple_target, exact_prediction):
    """不越界時 backward 仍應可被呼叫（梯度為 0 但不報錯）。"""
    loss = custom_loss_minmax(exact_prediction, simple_target, method="high")
    # 即便回傳的是 _zero_loss()，也必須能 backward
    loss.backward()
    # grad 可能為 None（因為無計算圖連結），也可能為全零；兩者皆合法
    if exact_prediction.grad is not None:
        assert torch.allclose(exact_prediction.grad, torch.zeros_like(exact_prediction))


# ----- 邊界情形：空頻段 / 全零 / 全一 -----


def test_custom_loss_boundary_empty_tensor():
    """空頻段（numel == 0）輸入應回傳零張量且不拋出例外。"""
    pred = torch.empty(0, requires_grad=True)
    target = torch.empty(0)
    loss_r = custom_loss_r(pred, target)
    loss_g = custom_loss_g(pred, target)
    loss_high = custom_loss_minmax(pred, target, method="high")
    loss_low = custom_loss_minmax(pred, target, method="low")
    for loss in (loss_r, loss_g, loss_high, loss_low):
        assert torch.allclose(loss, torch.tensor(0.0))


def test_custom_loss_boundary_all_zero_target():
    """target 全為 0：max == min == 0，mask 涵蓋全部元素；
    pred 任一側越界都應得到對應的單側損失。"""
    pred = torch.tensor([-1.0, 0.0, 2.0], requires_grad=True)
    target = torch.zeros(3)
    # method="high": bound=0, 越界條件 pred < 0 => index 0 越界
    loss_high = custom_loss_minmax(pred, target, method="high")
    # method="low" : bound=0, 越界條件 pred > 0 => index 2 越界
    loss_low = custom_loss_minmax(pred, target, method="low")
    assert loss_high.item() > 0
    assert loss_low.item() > 0


def test_custom_loss_boundary_all_ones_target():
    """target 全為 1：high == low 邊界同值，雙邊界損失等於兩單側之和。"""
    pred = torch.tensor([0.5, 1.0, 1.5], requires_grad=True)
    target = torch.ones(3)
    loss_boundary = custom_loss_boundary(pred, target)
    loss_high = custom_loss_minmax(pred, target, method="high")
    loss_low = custom_loss_minmax(pred, target, method="low")
    assert torch.allclose(loss_boundary, loss_high + loss_low)


def test_custom_loss_boundary_does_not_mutate_inputs(simple_target, violating_prediction):
    """損失計算不應修改輸入張量（無 in-place 操作）。"""
    pred_before = violating_prediction.detach().clone()
    target_before = simple_target.detach().clone()
    _ = custom_loss_boundary(violating_prediction, simple_target)
    _ = custom_loss_minmax(violating_prediction, simple_target, method="high")
    assert torch.equal(violating_prediction.detach(), pred_before)
    assert torch.equal(simple_target, target_before)
