"""Patch 天線專用損失函數。"""

from typing import Literal, Union

import torch
from torch import Tensor, nn

from antenna.types import *


def custom_loss_r(prediciton, target, loss_type="SmoothL1Loss"):

    criterion_r = nn.SmoothL1Loss() if loss_type == "SmoothL1Loss" else nn.MSELoss()

    high_response = target.max()
    low_response = target.min()

    mask_25 = target == high_response  # mask == -2.5 index
    mask_b_25 = prediciton[mask_25] < high_response

    if mask_b_25.sum() == 0:
        loss_25 = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    else:
        loss_25 = criterion_r(prediciton[mask_25][mask_b_25], target[mask_25][mask_b_25])

    mask_10 = target == low_response  # mask == -2.5 index
    mask_b_10 = prediciton[mask_10] > low_response

    if mask_b_10.sum() == 0:
        loss_10 = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    else:
        loss_10 = criterion_r(prediciton[mask_10][mask_b_10], target[mask_10][mask_b_10])

    loss = loss_25 + loss_10
    return loss


def custom_loss_g(prediciton, target, loss_type="SmoothL1Loss"):

    criterion_g = nn.SmoothL1Loss() if loss_type == "SmoothL1Loss" else nn.MSELoss()

    high_gain = target.max()
    low_gain = target.min()

    mask_10 = target == low_gain  # mask == -10 index
    mask_b_10 = prediciton[mask_10] > low_gain

    if mask_b_10.sum() == 0:
        loss_10 = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    else:
        loss_10 = criterion_g(prediciton[mask_10][mask_b_10], target[mask_10][mask_b_10])

    mask_4 = target == high_gain  # mask == 4 index
    mask_b_4 = prediciton[mask_4] < high_gain

    if mask_b_4.sum() == 0:
        loss_4 = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    else:
        loss_4 = criterion_g(prediciton[mask_4][mask_b_4], target[mask_4][mask_b_4])

    loss = loss_10 + loss_4
    return loss


def custom_loss_minmax(prediciton: Tensor, target: Tensor, method: Literal["low", "high"], loss_type="SmoothL1Loss"):

    criterion = nn.SmoothL1Loss() if loss_type == "SmoothL1Loss" else nn.MSELoss()
    loss_zero = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

    match method:
        case "high":
            target_high = target.max()
            mask_high = target == target_high
            mask_b_high = prediciton[mask_high] < target_high
            return (
                loss_zero
                if mask_b_high.sum() == 0
                else criterion(prediciton[mask_high][mask_b_high], target[mask_high][mask_b_high])
            )

        case "low":
            target_low = target.min()
            mask_low = target == target_low
            mask_b_low = prediciton[mask_low] > target_low
            return (
                loss_zero
                if mask_b_low.sum() == 0
                else criterion(prediciton[mask_low][mask_b_low], target[mask_low][mask_b_low])
            )

        case _:
            raise ValueError("The method must be `low` or `high`.")


@overload
def interval_loss(
    prediction: Tensor,
    lower_response: float,
    upper_response: float,
    target: Tensor = None,
    *,
    loss_type: str = "SmoothL1Loss",
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Interval Loss: 視為相對於 Target 的誤差容許值[target + lower, target + upper], 限制 prediction 必須在此動態邊界內。
    """
    ...


@overload
def interval_loss(
    prediction: Tensor,
    lower_response: Tensor,
    upper_response: Tensor,
    *,
    loss_type: str = "SmoothL1Loss",
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Interval Loss: 限制 prediction 必須在 [lower, upper] 之間。
    """
    ...


def interval_loss(
    prediction: Tensor,
    lower_response: float | Tensor,
    upper_response: float | Tensor,
    target: Tensor = None,
    *,
    loss_type: str = "SmoothL1Loss",
    reduction: str = "mean",
) -> Tensor:
    """
    區間損失 (Interval Loss) 的核心運算函數。
    """
    if loss_type == "SmoothL1Loss":
        loss_fn = nn.SmoothL1Loss(reduction=reduction)
    elif loss_type == "MSELoss":
        loss_fn = nn.MSELoss(reduction=reduction)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    if isinstance(lower_response, Tensor) and isinstance(upper_response, Tensor):
        min_bound = lower_response
        max_bound = upper_response

    else:  # * Target + Offset
        if target is None:
            raise ValueError("使用 Float (相對偏移模式) 時，必須傳入 target。")

        min_bound = target + lower_response
        max_bound = target + upper_response

    # * Universal Clamp Logic
    target_clamped = torch.clamp(prediction, min=min_bound, max=max_bound).detach()
    loss = loss_fn(prediction, target_clamped)

    return loss
