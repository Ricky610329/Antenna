import torch
from torch import nn

from antenna.core.pattern import AntennaPattern
from antenna.core.response import AntennaResponse
from antenna.models.autograd import GumbelSigmoid
from antenna.models.generators.base import _build_fc_patch, _kaiming_init_
from antenna.utils.config import config


class GumbelSigmoidGEN(nn.Module):
    """以 Gumbel-Sigmoid 採樣產生可微分近似離散輸出的生成器。

    與 :class:`SigmoidGEN` 相比使用較深且隨 pattern size 縮放的 MLP，
    並對權重做 Kaiming 初始化。`tau` 為可學習參數。
    """

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
        """輸出經過 Gumbel-Sigmoid 的可微分近似離散張量。"""
        # clamp 至 [-5, 5] 以防 GumbelSigmoid 反傳時 tau 梯度爆炸
        self.logits = torch.clamp(self.fc_patch(input), min=-5.0, max=5.0)
        x = GumbelSigmoid.apply(self.logits, self.tau)
        self.tau_history.append(self.tau.detach().cpu().item())
        return x

    def anneal_tau(self, min_tau: float = 0.1) -> None:
        """Annealing — 訓練初期較大的 tau 鼓勵探索，後期收斂至離散解。

        目前僅以 ``clamp(min=min_tau)`` 作為下限保護；完整 annealing schedule
        尚未實作（若要加入可以在此處乘以衰減率）。
        """
        self.tau = torch.clamp(self.tau, min=min_tau)
        self.tau_history.append(self.tau.detach().cpu().item())

    def binarize(self, threshold: float = 0.5) -> AntennaPattern:
        """以 sigmoid(logits) > threshold 做硬二值化（不可微）。"""
        binarized_output = (torch.sigmoid(self.logits) > threshold).float()
        return AntennaPattern(binarized_output)
