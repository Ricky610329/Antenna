"""全連接型 HFSS 代理網路 (HFSSNet)。

將 pattern 像素映射到 AntennaResponse，為學長最早期的代理模型實作。
"""

from torch import Tensor, nn

from antenna.utils.config import config


class HFSSNet(nn.Module):
    """Pattern -> Response 的全連接代理模型。

    Args:
        num_pattern_pixel: 輸入 pattern 的像素總數（flatten 後）。
        num_response: 輸出響應的形狀，例如 ``(3, 17)``。
    """

    def __init__(self, num_pattern_pixel: int = 625, num_response: tuple = (3, 17)):
        super().__init__()
        self.num_response = num_response
        self.num_pattern_pixel = num_pattern_pixel

        self.fc_patch = nn.Sequential(
            nn.Linear(num_pattern_pixel, 2048),
            nn.PReLU(),
            nn.Linear(2048, 1024),
            nn.PReLU(),
            nn.Linear(1024, 512),
            nn.PReLU(),
            nn.Linear(512, 128),
            nn.PReLU(),
            nn.Linear(128, 64),
            nn.PReLU(),
            nn.Linear(64, num_response[0] * num_response[1]),
        )
        self.to(config.device)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_pattern_pixel={self.num_pattern_pixel}, num_response={self.num_response})"
        )

    def forward(self, input: Tensor) -> Tensor:
        x = self.fc_patch(input)
        x = x.reshape(self.num_response)
        return x
