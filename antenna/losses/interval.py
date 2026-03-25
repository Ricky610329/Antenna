import torch
from torch import Tensor, nn


def custom_loss_interval(prediction: Tensor, target_low: Tensor, target_high: Tensor, loss_type="SmoothL1Loss"):
    """
    計算基於目標區間的自定義 loss。
    如果 prediction 在 [target_low, target_high] 區間內，則 loss 為 0。
    否則，計算 prediction 與最近的區間邊界之間的 loss。
    """
    criterion = nn.SmoothL1Loss(reduction="none") if loss_type == "SmoothL1Loss" else nn.MSELoss(reduction="none")

    # 初始化 loss tensor
    losses = torch.zeros_like(prediction)

    # 1. 處理 prediction > target_high 的情況
    mask_above = prediction > target_high
    if mask_above.sum() > 0:
        losses[mask_above] = criterion(prediction[mask_above], target_high.expand_as(prediction)[mask_above])

    # 2. 處理 prediction < target_low 的情況
    mask_below = prediction < target_low
    if mask_below.sum() > 0:
        losses[mask_below] = criterion(prediction[mask_below], target_low.expand_as(prediction)[mask_below])

    # 3. prediction 在區間內的情況 (target_low <= prediction <= target_high)
    #    此時 losses[mask_in_interval] 仍然是 0，不需要額外處理

    return losses.mean()  # 返回平均 loss
