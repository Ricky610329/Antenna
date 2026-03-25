from torch import nn
from torch.types import Tensor

from antenna import *
from antenna.models.components import BiScaleNorm
from antenna.types import *
from antenna.utils import *


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

    def forward(self, input, tau: Optional[float] = None) -> Tensor:
        x = self.fc_patch(input)
        return AntennaPattern.binarization(x, tau)
