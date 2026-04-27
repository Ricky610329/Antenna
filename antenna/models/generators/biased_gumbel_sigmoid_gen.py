"""帶反 collapse 初始化的 GumbelSigmoidGEN 變體。

§8.4d / V12 觀察：generator 動力學偏好 logits ≈ 0，最終 hard-binarized
pattern 是 100% 全亮。為打破此 trivial 解：把最後一層 fc_patch 的 bias 用
較大的隨機初值（uniform 在 ±BIAS_INIT_RANGE），這樣 epoch 1 forward 已是
有結構（半 on 半 off）的 pattern，generator 必須學「修正」而非「從 0 開始」。
"""

import torch
from torch import nn

from antenna.core.pattern import AntennaPattern
from antenna.core.response import AntennaResponse
from antenna.models.autograd import GumbelSigmoid
from antenna.models.generators.base import _build_fc_patch, _kaiming_init_
from antenna.utils.config import config


class BiasedGumbelSigmoidGEN(nn.Module):
    """GumbelSigmoidGEN + 最後一層 bias 用大範圍隨機初始化以打破 collapse。"""

    LOGITS_CLAMP = 5.0
    BIAS_INIT_RANGE = 3.0  # 最後一層 bias 在 [-3, 3] 隨機（sigmoid(3) ≈ 0.95 → 強偏向二極化）

    def __init__(self):
        super().__init__()
        pattern_size = AntennaPattern.size(flatten=True)
        self.fc_patch = _build_fc_patch(
            input_dim=AntennaResponse.size(flatten=True),
            hidden_dims=(pattern_size, pattern_size * 2, pattern_size),
            output_dim=pattern_size,
        )
        _kaiming_init_(self.fc_patch)
        # 找最後一個帶 bias 的 Linear（fc_patch 是 Sequential），把 bias 換成大範圍隨機
        last_linear = None
        for module in self.fc_patch.modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                last_linear = module
        if last_linear is not None:
            with torch.no_grad():
                last_linear.bias.uniform_(-self.BIAS_INIT_RANGE, self.BIAS_INIT_RANGE)

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
