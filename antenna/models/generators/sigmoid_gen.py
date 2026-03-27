from typing import Optional

from torch import Tensor, nn

from antenna.core.pattern import AntennaPattern
from antenna.core.response import AntennaResponse
from antenna.models.components import BiScaleNorm
from antenna.utils.config import config


class SigmoidGEN(nn.Module):
    """
    Generator Model
    """

    def __init__(self):
        super().__init__()
        self.fc_patch = nn.Sequential(  # Can use BiScaleNorm or nn.PReLU, except the last layer.
            nn.Linear(AntennaResponse.size(flatten=True), 1024),
            nn.PReLU(),
            nn.Linear(1024, 1024),
            nn.PReLU(),
            nn.Linear(1024, AntennaPattern.size(flatten=True)),
            BiScaleNorm(),
        )
        self.to(config.device)

    def forward(self, input, tau: float | None = None) -> Tensor:
        x = self.fc_patch(input)
        return AntennaPattern.binarization(x, tau)
