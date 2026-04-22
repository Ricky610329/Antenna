"""結合鏡像生成與 surrogate 選擇的 CVAE 風格生成器。"""

from typing import Generic

import torch
from torch import nn

from antenna.core.pattern import AntennaPattern
from antenna.core.response import AntennaResponse, MultiResponses
from antenna.losses.mirror import mirror
from antenna.types import CustomSModel, ResultType


class MirrorCVAE(nn.Module, Generic[CustomSModel]):
    """結合 CVAE 解碼器 (生成器) 與鏡像/評估/選擇機制的模組。

    ``forward()`` 會依序執行：

    1. 以條件 ``c`` 與潛在向量 ``z`` 解碼出基礎 pattern。
    2. 透過 :func:`antenna.losses.mirror.mirror` 產生多組鏡像 pattern。
    3. 以傳入的 surrogate model (``smodel``) 評估所有鏡像 pattern。
    4. 依 smodel 損失排序並標記最佳解。
    5. 回傳排序後的結果（含梯度），可直接用於反向傳播。
    """

    # Decoder hidden layer 維度（與 OldGEN 類似但不含 BiScaleNorm）
    _DECODER_HIDDEN_DIM = 1024

    def __init__(
        self,
        latent_dim: int,
        smodel: CustomSModel,
        lower_pattern: AntennaPattern | None = None,
    ):
        """初始化 MirrorCVAE 生成器。

        Args:
            latent_dim: CVAE 的潛在向量維度 (例如 128)。
            smodel: 預先訓練好的 surrogate model，用以評估鏡像 pattern 的好壞。
            lower_pattern: 要疊加到每個 pattern 上的靜態 'lower' 部分；可為 None。
        """
        super().__init__()

        if not callable(smodel):
            raise TypeError("smodel 必須是一個可呼叫的 nn.Module")

        self.latent_dim = latent_dim
        condition_dim = AntennaResponse.size(flatten=True)
        pattern_dim = AntennaPattern.size(flatten=True)

        # 儲存 smodel 並凍結其參數（假設 smodel 在別處訓練）
        self.smodel = smodel
        self.smodel.requires_grad(False)

        self.lower = lower_pattern

        # CVAE 解碼器：與 OldGEN 結構相近（Linear+PReLU x2 + Linear），最後加 Sigmoid 約束至 [0, 1]
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, self._DECODER_HIDDEN_DIM),
            nn.PReLU(),
            nn.Linear(self._DECODER_HIDDEN_DIM, self._DECODER_HIDDEN_DIM),
            nn.PReLU(),
            nn.Linear(self._DECODER_HIDDEN_DIM, pattern_dim),
            nn.Sigmoid(),
        )

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """將潛在向量 ``z`` 與條件 ``c`` (皆為 1D) 拼接後解碼為基礎 pattern 張量。"""
        inputs = torch.cat([z, c], dim=0)
        return self.decoder(inputs)

    def forward(self, c: torch.Tensor, z: torch.Tensor | None = None) -> list[ResultType]:
        """執行「產生 → 鏡像 → 評估 → 選擇」的前向傳播。

        Args:
            c: 條件向量 (例如 ``AntennaResponse.target.concat()``)。
            z: 固定的潛在向量（用於可重現生成）；若為 None 則隨機採樣。

        Returns:
            依 ``fake_loss`` 升序排列的 :class:`ResultType` 清單，
            其中 ``is_best=True`` 的項為 smodel 評估最佳的鏡像 pattern。
        """
        if z is None:
            z = torch.randn(self.latent_dim, device=c.device)

        # 1. 解碼出基礎 pattern（保留 decoder 梯度）
        base_pattern_tensor = self.decode(z, c)
        base_pattern_obj = AntennaPattern(base_pattern_tensor)

        # 2. 產生鏡像 patterns（cat/flip 皆可微分）
        mirrored_tensors = mirror(base_pattern_obj.merge(), mode="-|*")
        mirrored_patterns = [
            AntennaPattern(t) + self.lower if self.lower else AntennaPattern(t) for t in mirrored_tensors
        ]

        # 3. 以 smodel 評估所有鏡像 patterns
        all_losses: list[torch.Tensor] = []
        all_results: list[MultiResponses] = []
        self.smodel.model.eval()

        with torch.enable_grad():  # 確保梯度流經 smodel
            for pattern in mirrored_patterns:
                output_result = self.smodel(pattern.series)
                fake_loss = output_result.criterion()
                all_losses.append(fake_loss)
                all_results.append(output_result)

        # 4. 在 detach 後的張量上取 argmin，使「選擇」操作本身不接收梯度
        losses_tensor_detached = torch.stack([loss.detach() for loss in all_losses])
        best_loss_index = torch.argmin(losses_tensor_detached)

        # 5. 打包結果（sm_loss / time 於 HFSS 模擬後才會被更新）
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

        return sorted(results, key=lambda x: x["sort_key"])
