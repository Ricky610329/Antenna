import torch
from torch import nn


class BiScaleNorm(nn.Module):
    """雙向縮放正規化：將正值除以 ``max``、負值除以 ``|min|``，零值保持為零。

    等同於將輸入分別壓至 ``[0, 1]``（正半）與 ``[-1, 0]``（負半），
    使正、負兩側的動態範圍獨立縮放。
    """

    def forward(self, input_vector):
        zero = torch.zeros((), dtype=input_vector.dtype, device=input_vector.device)
        # 正值正規化：x > 0 -> x / max(x)
        max_val = torch.max(input_vector)
        positive_normalized = torch.where(input_vector > 0, input_vector / max_val, zero)

        # 負值正規化：x < 0 -> x / |min(x)|
        min_val = torch.min(input_vector)
        negative_normalized = torch.where(input_vector < 0, input_vector / torch.abs(min_val), zero)

        return positive_normalized + negative_normalized
