"""擴大 logits clamp 範圍的 GumbelSigmoidGEN 變體。

用於驗證 §8.4d 的假說：標準版 clamp([-5,5]) 太窄，sigmoid(±5/tau=0.1)
仍接近飽和但不夠極端，導致 generator 收斂到 logits ≈ 0、threshold 後
全亮。擴大到 ±20 後，logits 有更大空間表達正/負偏好。
"""

import torch
from torch import nn

from antenna.core.pattern import AntennaPattern
from antenna.core.response import AntennaResponse
from antenna.models.autograd import GumbelSigmoid
from antenna.models.generators.base import _build_fc_patch, _kaiming_init_
from antenna.utils.config import config


class WideGumbelSigmoidGEN(nn.Module):
    """與 GumbelSigmoidGEN 相同但擴大 logits clamp 範圍至 [-20, 20]。

    其他結構（FC stack、tau parameter、Kaiming init）與標準版一致，
    只差 forward 的 clamp 邊界。
    """

    LOGITS_CLAMP = 20.0  # 標準版是 5.0

    def __init__(self):
        super().__init__()
        pattern_size = AntennaPattern.size(flatten=True)
        self.fc_patch = _build_fc_patch(
            input_dim=AntennaResponse.size(flatten=True),
            hidden_dims=(pattern_size, pattern_size * 2, pattern_size),
            output_dim=pattern_size,
        )
        _kaiming_init_(self.fc_patch)

        self.tau = nn.Parameter(torch.tensor(5.0, requires_grad=True))
        self.tau_history: list[float] = []

        self.to(config.device)

    def forward(self, input):
        self.logits = torch.clamp(self.fc_patch(input), min=-self.LOGITS_CLAMP, max=self.LOGITS_CLAMP)
        x = GumbelSigmoid.apply(self.logits, self.tau)
        self.tau_history.append(self.tau.detach().cpu().item())
        return x

    def anneal_tau(self, min_tau: float = 0.1) -> None:
        self.tau = torch.clamp(self.tau, min=min_tau)
        self.tau_history.append(self.tau.detach().cpu().item())

    def binarize(self, threshold: float = 0.5) -> AntennaPattern:
        binarized_output = (torch.sigmoid(self.logits) > threshold).float()
        return AntennaPattern(binarized_output)
