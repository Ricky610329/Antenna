"""MirrorCVAE — 結合鏡像與 CVAE 的生成器。"""

import torch
from torch import Tensor, nn

from antenna import AntennaPattern, AntennaResponse, MultiResponses, config
from antenna.functions import mirror
from antenna.types import *


class MirrorCVAE(nn.Module, Generic[CustomSModel]):
    """
    結合了 CVAE 解碼器 (生成器) 與鏡像/評估/選擇機制的模組。
    """

    def __init__(
        self,
        latent_dim: int,
        smodel: CustomSModel,
        lower_pattern: AntennaPattern = None,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        condition_dim = AntennaResponse.size(flatten=True)
        pattern_dim = AntennaPattern.size(flatten=True)

        if not callable(smodel):
            raise TypeError("smodel 必須是一個可呼叫的 nn.Module")

        self.smodel = smodel
        self.smodel.requires_grad(False)

        self.lower = lower_pattern

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 1024),
            nn.PReLU(),
            nn.Linear(1024, 1024),
            nn.PReLU(),
            nn.Linear(1024, pattern_dim),
            nn.Sigmoid(),
        )

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([z, c], dim=0)
        base_pattern_tensor = self.decoder(inputs)
        return base_pattern_tensor

    def forward(self, c: torch.Tensor, z: torch.Tensor | None = None) -> list[ResultType]:
        if z is None:
            z = torch.randn(self.latent_dim, device=c.device)

        base_pattern_tensor = self.decode(z, c)
        base_pattern_obj = AntennaPattern(base_pattern_tensor)

        mirrored_tensors = mirror(base_pattern_obj.merge(), mode="-|*")
        mirrored_patterns = [
            AntennaPattern(t) + self.lower if self.lower else AntennaPattern(t) for t in mirrored_tensors
        ]

        all_losses: list[torch.Tensor] = []
        all_results: list[MultiResponses] = []
        self.smodel.model.eval()

        with torch.enable_grad():
            for pattern in mirrored_patterns:
                output_result = self.smodel(pattern.series)
                fake_loss = output_result.criterion()
                all_losses.append(fake_loss)
                all_results.append(output_result)

        losses_tensor_detached = torch.stack([l.detach() for l in all_losses])
        best_loss_index = torch.argmin(losses_tensor_detached)

        results: list[ResultType] = []
        for i, pattern in enumerate(mirrored_patterns):
            result_dict: ResultType = {
                "pattern": pattern,
                "real_result": None,
                "fake_result": all_results[i],
                "real_loss": None,
                "fake_loss": all_losses[i],
                "sm_loss": [],
                "time": 0,
                "sort_key": all_losses[i].item(),
                "is_best": (i == best_loss_index.item()),
            }
            results.append(result_dict)

            results_sorted: list[ResultType] = sorted(results, key=lambda x: x["sort_key"])

        return results_sorted
