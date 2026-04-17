from torch import Tensor, nn

from antenna.core.pattern import AntennaPattern
from antenna.core.response import AntennaResponse
from antenna.models.autograd import sign_f
from antenna.models.generators.base import _build_fc_patch
from antenna.utils.config import config


class OldGEN(nn.Module):
    """舊版生成器：以 sign_f (STE) 將輸出離散化至 {0, 1}。

    目前仍被 `train_single.py` / `train_dual.py` / `train_ris.py` /
    `train_single_mirror.py` 等舊腳本使用，故保留。
    """

    def __init__(self):
        super().__init__()
        self.fc_patch = _build_fc_patch(
            input_dim=AntennaResponse.size(flatten=True),
            hidden_dims=(1024, 1024),
            output_dim=AntennaPattern.size(flatten=True),
        )
        self.r = sign_f.apply
        self.to(config.device)

    def forward(self, input) -> Tensor:
        x = self.fc_patch(input)
        # sign_f 輸出 {-1, +1}，線性映射到 {0, 1}
        x = self.r(x) / 2 + 0.5  # type: ignore[operator]
        return x
