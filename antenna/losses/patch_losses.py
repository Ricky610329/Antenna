"""Patch 天線專用損失函數。

本模組提供三個主要的損失函數與一個底層共用邏輯：

- :func:`custom_loss_boundary` — 通用雙邊界懲罰（同時處理 ``target == max`` 與 ``target == min``）。
- :func:`custom_loss_r` / :func:`custom_loss_g` — 向後相容的 thin wrapper，語意上與 ``boundary`` 相同。
- :func:`custom_loss_minmax` — 單邊界懲罰（僅處理 ``method="low"`` 或 ``method="high"`` 其中一側）。
- :func:`interval_loss` — 區間損失，限制預測落在指定區間內。
"""

from typing import Literal

import torch
from torch import Tensor, nn

__all__ = [
    "custom_loss_boundary",
    "custom_loss_r",
    "custom_loss_g",
    "custom_loss_minmax",
    "interval_loss",
]


def _make_criterion(loss_type: str, reduction: str = "mean") -> nn.Module:
    """根據字串回傳對應的 PyTorch 損失函數實例。"""
    if loss_type == "SmoothL1Loss":
        return nn.SmoothL1Loss(reduction=reduction)
    if loss_type == "MSELoss":
        return nn.MSELoss(reduction=reduction)
    raise ValueError(f"Unsupported loss_type: {loss_type!r}（僅支援 'SmoothL1Loss' 或 'MSELoss'）")


def _zero_loss() -> Tensor:
    """回傳可微分的零張量（用於空 mask 情形）。"""
    return torch.tensor(0.0, dtype=torch.float32, requires_grad=True)


def _one_sided_penalty(
    prediction: Tensor,
    target: Tensor,
    method: Literal["low", "high"],
    criterion: nn.Module,
) -> Tensor:
    """單邊界懲罰：僅對「越界側」的預測計算損失。

    - ``method="high"``：對 ``target == target.max()`` 位置中、``prediction`` **低於** target 的元素懲罰。
    - ``method="low"`` ：對 ``target == target.min()`` 位置中、``prediction`` **高於** target 的元素懲罰。

    若沒有任何元素越界，回傳可微分的零張量。
    """
    if method == "high":
        bound = target.max()
        mask = target == bound
        violation = prediction[mask] < bound
    elif method == "low":
        bound = target.min()
        mask = target == bound
        violation = prediction[mask] > bound
    else:
        raise ValueError(f"method 必須為 'low' 或 'high'，收到 {method!r}")

    if violation.sum() == 0:
        return _zero_loss()

    return criterion(prediction[mask][violation], target[mask][violation])


def custom_loss_boundary(
    prediction: Tensor,
    target: Tensor,
    side: Literal["r", "g"] = "r",
    loss_type: str = "SmoothL1Loss",
) -> Tensor:
    """雙邊界懲罰：同時對 ``target`` 的最大值與最小值邊界計算越界損失。

    此函數統一了原本 :func:`custom_loss_r`（反射係數/return loss）與
    :func:`custom_loss_g`（增益/gain）的重複邏輯。兩者在數學上相同，
    皆為「high 邊界越界（pred < high）」與「low 邊界越界（pred > low）」的損失和。

    Parameters
    ----------
    prediction : Tensor
        模型預測張量。
    target : Tensor
        目標張量；會從中取 ``max`` 與 ``min`` 作為雙邊界。
    side : Literal["r", "g"]
        語意標籤，目前僅為區分用途（r=return loss、g=gain），不影響計算。
    loss_type : str
        內部使用的基礎損失函數名稱，可為 ``"SmoothL1Loss"`` 或 ``"MSELoss"``。

    Returns
    -------
    Tensor
        兩個邊界的損失和（若無越界元素則為 0）。
    """
    if side not in ("r", "g"):
        raise ValueError(f"side 必須為 'r' 或 'g'，收到 {side!r}")

    criterion = _make_criterion(loss_type)
    loss_high = _one_sided_penalty(prediction, target, "high", criterion)
    loss_low = _one_sided_penalty(prediction, target, "low", criterion)
    return loss_high + loss_low


def custom_loss_r(prediciton: Tensor, target: Tensor, loss_type: str = "SmoothL1Loss") -> Tensor:
    """反射係數專用損失（thin wrapper，向後相容）。

    等價於 ``custom_loss_boundary(prediciton, target, side="r", loss_type=loss_type)``。
    """
    return custom_loss_boundary(prediciton, target, side="r", loss_type=loss_type)


def custom_loss_g(prediciton: Tensor, target: Tensor, loss_type: str = "SmoothL1Loss") -> Tensor:
    """增益專用損失（thin wrapper，向後相容）。

    等價於 ``custom_loss_boundary(prediciton, target, side="g", loss_type=loss_type)``。
    """
    return custom_loss_boundary(prediciton, target, side="g", loss_type=loss_type)


def custom_loss_minmax(
    prediciton: Tensor,
    target: Tensor,
    method: Literal["low", "high"],
    loss_type: str = "SmoothL1Loss",
) -> Tensor:
    """單邊界懲罰：依 ``method`` 只對 high 或 low 其中一側計算越界損失。"""
    criterion = _make_criterion(loss_type)
    return _one_sided_penalty(prediciton, target, method, criterion)


def interval_loss(
    prediction: Tensor,
    lower_response: float | Tensor,
    upper_response: float | Tensor,
    target: Tensor | None = None,
    *,
    loss_type: str = "SmoothL1Loss",
    reduction: str = "mean",
) -> Tensor:
    """區間損失 (Interval Loss)。

    支援兩種模式：

    - **絕對邊界**：當 ``lower_response`` 與 ``upper_response`` 皆為 :class:`Tensor` 時，
      將 prediction clamp 至 ``[lower_response, upper_response]`` 計算損失。
    - **相對偏移**：當 ``lower_response`` / ``upper_response`` 為純量，需另外傳入
      ``target``，則邊界為 ``[target + lower, target + upper]``。
    """
    loss_fn = _make_criterion(loss_type, reduction=reduction)

    if isinstance(lower_response, Tensor) and isinstance(upper_response, Tensor):
        min_bound = lower_response
        max_bound = upper_response
    else:
        # 相對偏移模式必須提供 target
        if target is None:
            raise ValueError("使用純量 (相對偏移模式) 時，必須傳入 target。")
        min_bound = target + lower_response
        max_bound = target + upper_response

    target_clamped = torch.clamp(prediction, min=min_bound, max=max_bound).detach()
    return loss_fn(prediction, target_clamped)
