from torch import nn
from torch.types import Tensor

from antenna import *
from antenna.models.autograd import sign_f
from antenna.models.components import BiScaleNorm
from antenna.types import *
from antenna.utils import *


class OldGEN(nn.Module):
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

        self.r = sign_f.apply
        self.to(config.device)

    def forward(self, input) -> Tensor:
        x = self.fc_patch(input)
        x = self.r(x) / 2 + 0.5  # type: ignore
        return x
