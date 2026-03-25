import torch
from torch import nn


class BiScaleNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_vector):
        # 大於 0 的值的正規化
        max_val = torch.max(input_vector)
        positive_normalized = torch.where(
            input_vector > 0, input_vector / max_val, torch.tensor(0.0, device=input_vector.device)
        )

        # 小於 0 的值的正規化
        min_val = torch.min(input_vector)
        negative_normalized = torch.where(
            input_vector < 0, input_vector / torch.abs(min_val), torch.tensor(0.0, device=input_vector.device)
        )

        # 合併正規化結果
        normalized_vector = positive_normalized + negative_normalized
        return normalized_vector
