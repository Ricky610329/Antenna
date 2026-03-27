from torch import Tensor, nn

from antenna.core.response import AntennaResponse
from antenna.utils.config import config


class GradientEstimator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.PReLU(),
            nn.Linear(2048, 1024),
            nn.PReLU(),
            nn.Linear(1024, 512),
            nn.PReLU(),
            nn.Linear(512, 512),
            nn.PReLU(),
            nn.Linear(512, output_dim),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Flatten(),
        )
        self.to(config.device)

    def forward(self, A: Tensor):
        A = A.unsqueeze(0)  # ? [batch, W, H]
        # print(A.shape)
        output = self.conv(A)
        output = self.net(output)
        return AntennaResponse(output)
