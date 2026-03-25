from torch import Tensor
from torch.functional import F

from .simulate_ris import RISSimulator


def custom_loss(prediction, target: Tensor, loss_type="SmoothL1Loss"):

    high_response = target.max()
    low_response = target.min()

    # 基本條件
    mask_20 = target == low_response
    mask_b_20 = prediction[mask_20] > low_response

    mask_0 = target == high_response
    mask_s_0 = prediction[mask_0] < low_response

    # 為了確保有梯度，設定條件不滿足時也會計入一個 dummy loss
    if mask_b_20.sum() > 0:
        loss_20 = F.smooth_l1_loss(prediction[mask_20][mask_b_20], target[mask_20][mask_b_20])
    else:
        # 使用全體 prediction 的一小部分作 dummy loss，保證梯度
        loss_20 = 0.01 * F.mse_loss(prediction, target)

    if mask_s_0.sum() > 0:
        loss_0 = F.smooth_l1_loss(prediction[mask_0][mask_s_0], target[mask_0][mask_s_0])
    else:
        loss_0 = 0.01 * F.mse_loss(prediction, target)

    loss = loss_20 + loss_0

    return loss
