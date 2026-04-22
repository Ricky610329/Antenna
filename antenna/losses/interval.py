"""區間型損失函數：若 prediction 落在目標區間內則損失為 0。"""

import torch
from torch import Tensor

from antenna.losses.patch_losses import _make_criterion


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
    criterion = _make_criterion(loss_type, reduction="mean")
    # 將 prediction 夾到區間內作為目標值；未越界處 diff=0 ⇒ 損失為 0，
    # detach() 確保梯度只沿 prediction 路徑回傳。
    clamped = torch.clamp(prediction, min=target_low, max=target_high).detach()
    return criterion(prediction, clamped)
