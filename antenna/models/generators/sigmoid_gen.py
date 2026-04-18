from torch import Tensor, nn

from antenna.core.pattern import AntennaPattern
from antenna.core.response import AntennaResponse
from antenna.models.generators.base import _build_fc_patch
from antenna.utils.config import config


class SigmoidGEN(nn.Module):
    """以 STE 可微分二值化 (AntennaPattern.binarization) 為輸出的生成器。"""

    def __init__(self):
        super().__init__()
        self.fc_patch = _build_fc_patch(
            input_dim=AntennaResponse.size(flatten=True),
            hidden_dims=(1024, 1024),
            output_dim=AntennaPattern.size(flatten=True),
        )
        self.to(config.device)

    def forward(self, input, tau: float | None = None) -> Tensor:
        x = self.fc_patch(input)
        return AntennaPattern.binarization(x, tau)
