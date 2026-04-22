"""RIS 模擬與專屬損失函數之公開入口。"""

from torch import Tensor
from torch.nn import functional as F

from .simulate_ris import RISSimulator

__all__ = ["RISSimulator", "custom_loss"]


def custom_loss(prediction, target: Tensor, loss_type: str = "SmoothL1Loss"):
    """RIS 遠場響應之遮罩式 SmoothL1 損失。

    僅針對「預測偏離可接受範圍」的取樣點計算損失：
    - 目標為 low_response 之處，若預測值大於 low_response 則計入
    - 目標為 high_response 之處，若預測值小於 low_response 則計入
    不滿足條件時退回一個小係數的 MSE，確保仍有梯度可回傳。
    """
    high_response = target.max()
    low_response = target.min()

    mask_low = target == low_response
    mask_low_violated = prediction[mask_low] > low_response

    mask_high = target == high_response
    mask_high_violated = prediction[mask_high] < low_response

    # 為避免條件不成立時失去梯度，退回一個小係數的 MSE 作為 dummy loss
    if mask_low_violated.sum() > 0:
        loss_low = F.smooth_l1_loss(
            prediction[mask_low][mask_low_violated],
            target[mask_low][mask_low_violated],
        )
    else:
        loss_low = 0.01 * F.mse_loss(prediction, target)

    if mask_high_violated.sum() > 0:
        loss_high = F.smooth_l1_loss(
            prediction[mask_high][mask_high_violated],
            target[mask_high][mask_high_violated],
        )
    else:
        loss_high = 0.01 * F.mse_loss(prediction, target)

    return loss_low + loss_high
