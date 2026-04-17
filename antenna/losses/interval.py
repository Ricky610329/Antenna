"""區間型損失函數：若 prediction 落在目標區間內則損失為 0。"""

import torch
from torch import Tensor, nn


def custom_loss_interval(
    prediction: Tensor,
    target_low: Tensor,
    target_high: Tensor,
    loss_type: str = "SmoothL1Loss",
) -> Tensor:
    """依目標區間計算損失。

    - 若 ``target_low <= prediction <= target_high``，對應元素的損失為 0。
    - 否則回傳 prediction 與最近邊界之間的 SmoothL1 / MSE 損失。

    Args:
        prediction: 模型預測值。
        target_low: 區間下界，形狀可 broadcast 至 ``prediction``。
        target_high: 區間上界，形狀可 broadcast 至 ``prediction``。
        loss_type: ``"SmoothL1Loss"`` 或 ``"MSELoss"``。

    Returns:
        對所有元素取平均後的純量損失。
    """
    criterion = nn.SmoothL1Loss(reduction="none") if loss_type == "SmoothL1Loss" else nn.MSELoss(reduction="none")

    losses = torch.zeros_like(prediction)

    # 1. prediction 高於上界
    mask_above = prediction > target_high
    if mask_above.any():
        losses[mask_above] = criterion(prediction[mask_above], target_high.expand_as(prediction)[mask_above])

    # 2. prediction 低於下界
    mask_below = prediction < target_low
    if mask_below.any():
        losses[mask_below] = criterion(prediction[mask_below], target_low.expand_as(prediction)[mask_below])

    # 3. 區間內的元素保持 0，不需額外處理

    return losses.mean()
