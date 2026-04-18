"""``antenna.losses.interval`` 與 ``antenna.losses.patch_losses.interval_loss`` 的單元測試。"""

import pytest
import torch

from antenna.losses.interval import custom_loss_interval
from antenna.losses.patch_losses import interval_loss

# ---------------------------------------------------------------------------
# custom_loss_interval
# ---------------------------------------------------------------------------


def test_custom_loss_interval_zero_when_all_inside():
    # prediction 完全在 [low, high] 區間內
    low = torch.full((5,), -1.0)
    high = torch.full((5,), 1.0)
    pred = torch.zeros(5)

    loss = custom_loss_interval(pred, low, high)
    assert torch.allclose(loss, torch.tensor(0.0))


def test_custom_loss_interval_positive_when_above_upper():
    low = torch.full((3,), 0.0)
    high = torch.full((3,), 1.0)
    pred = torch.tensor([2.0, 3.0, 4.0])

    loss = custom_loss_interval(pred, low, high)
    assert loss > 0


def test_custom_loss_interval_positive_when_below_lower():
    low = torch.full((3,), 0.0)
    high = torch.full((3,), 1.0)
    pred = torch.tensor([-2.0, -3.0, -4.0])

    loss = custom_loss_interval(pred, low, high)
    assert loss > 0


def test_custom_loss_interval_gradient_flows_outside_interval():
    low = torch.full((3,), 0.0)
    high = torch.full((3,), 1.0)
    pred = torch.tensor([2.0, -2.0, 0.5], requires_grad=True)

    loss = custom_loss_interval(pred, low, high)
    loss.backward()

    # 區間內元素 (0.5) 的梯度應為 0
    assert pred.grad[2].abs().item() == pytest.approx(0.0, abs=1e-6)
    # 區間外元素應有非零梯度
    assert pred.grad[0].abs().item() > 0
    assert pred.grad[1].abs().item() > 0


def test_custom_loss_interval_supports_mse():
    low = torch.full((3,), 0.0)
    high = torch.full((3,), 1.0)
    pred = torch.tensor([2.0, 2.0, 2.0])

    loss_smooth = custom_loss_interval(pred, low, high, loss_type="SmoothL1Loss")
    loss_mse = custom_loss_interval(pred, low, high, loss_type="MSELoss")
    # MSE 對距離 >1 的誤差懲罰更大
    assert loss_mse > loss_smooth


# ---------------------------------------------------------------------------
# interval_loss
# ---------------------------------------------------------------------------


def test_interval_loss_absolute_bounds_zero_when_inside():
    lower = torch.full((4,), -1.0)
    upper = torch.full((4,), 1.0)
    pred = torch.tensor([-0.5, 0.0, 0.5, 0.9])

    loss = interval_loss(pred, lower, upper)
    assert torch.allclose(loss, torch.tensor(0.0))


def test_interval_loss_absolute_bounds_positive_when_outside():
    lower = torch.full((3,), -1.0)
    upper = torch.full((3,), 1.0)
    pred = torch.tensor([-5.0, 5.0, 0.0])

    loss = interval_loss(pred, lower, upper)
    assert loss > 0


def test_interval_loss_relative_mode_requires_target():
    pred = torch.tensor([0.0, 1.0])
    with pytest.raises(ValueError):
        interval_loss(pred, -0.5, 0.5)  # 沒給 target


def test_interval_loss_relative_mode_uses_target_offsets():
    target = torch.tensor([0.0, 0.0, 0.0])
    pred_inside = torch.tensor([-0.4, 0.0, 0.4])
    pred_outside = torch.tensor([-2.0, 0.0, 2.0])

    loss_inside = interval_loss(pred_inside, -0.5, 0.5, target=target)
    loss_outside = interval_loss(pred_outside, -0.5, 0.5, target=target)

    assert torch.allclose(loss_inside, torch.tensor(0.0))
    assert loss_outside > 0


def test_interval_loss_invalid_loss_type_raises():
    pred = torch.zeros(3)
    lower = torch.full((3,), -1.0)
    upper = torch.full((3,), 1.0)

    with pytest.raises(ValueError):
        interval_loss(pred, lower, upper, loss_type="HuberLossX")


def test_interval_loss_gradient_flows_outside_interval():
    lower = torch.full((3,), -1.0)
    upper = torch.full((3,), 1.0)
    pred = torch.tensor([2.0, 0.0, -2.0], requires_grad=True)

    loss = interval_loss(pred, lower, upper)
    loss.backward()

    assert pred.grad is not None
    # 區間內元素梯度為 0，區間外梯度非零
    assert pred.grad[1].abs().item() == pytest.approx(0.0, abs=1e-6)
    assert pred.grad[0].abs().item() > 0
    assert pred.grad[2].abs().item() > 0


def test_interval_loss_supports_mse_loss_type():
    lower = torch.full((3,), -1.0)
    upper = torch.full((3,), 1.0)
    pred = torch.tensor([3.0, 3.0, 3.0])

    loss_smooth = interval_loss(pred, lower, upper, loss_type="SmoothL1Loss")
    loss_mse = interval_loss(pred, lower, upper, loss_type="MSELoss")
    assert loss_mse > loss_smooth
