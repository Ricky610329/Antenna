"""RIS 模擬與專屬損失函數之公開入口。"""

import torch
from torch import Tensor
from torch.nn import functional as F

from .simulate_ris import RISSimulator

__all__ = ["RISSimulator", "custom_loss", "custom_loss_directivity"]


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


def custom_loss_directivity(
    prediction,
    target: Tensor,
    sidelobe_threshold: float | None = None,
    main_beam_weight: float = 0.1,
):
    """Tolerance + reward 風格的 RIS 損失，用於追求 main beam 增益最大化。

    跟 :func:`custom_loss` 相比的關鍵差異：**main beam 區域額外加上「響應越高
    loss 越低」的獎勵項**，而不只是檢查「不要跌到 sidelobe 之下」。直接給
    generator 把能量集中到 main beam 的梯度訊號。

    Args:
        prediction: 模型預測響應（dB）。
        target: 目標響應 mask（梯形：center 值代表 main beam，min 值代表 sidelobe）。
        sidelobe_threshold: sidelobe 區域的響應上限（dB）；超過會懲罰。
            None 時用 ``target.min()``，跟原 ``custom_loss`` 一致。
        main_beam_weight: main beam reward 項的權重；越大越鼓勵峰值升高。

    Returns:
        ``side_penalty + main_beam_weight * (-mean(prediction[main_beam]))``
        Sidelobe 項是平方越界 penalty；main beam 項是負平均（最小化即最大化）。
    """
    high_response = target.max()
    if sidelobe_threshold is None:
        sidelobe_threshold = float(target.min().item())

    mask_main = target == high_response
    mask_side = target == target.min()

    # Sidelobe penalty：超過 threshold 才計入，平方放大
    if mask_side.any():
        side_excess = (prediction[mask_side] - sidelobe_threshold).clamp(min=0)
        side_loss = side_excess.pow(2).mean()
    else:
        side_loss = torch.tensor(0.0, device=prediction.device)

    # Main beam reward：響應越高 loss 越低（負 mean）
    if mask_main.any():
        main_reward = -prediction[mask_main].mean()
    else:
        main_reward = torch.tensor(0.0, device=prediction.device)

    return side_loss + main_beam_weight * main_reward
