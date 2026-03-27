from typing import Generic

import torch
from torch import nn

from antenna.core.pattern import AntennaPattern
from antenna.core.response import AntennaResponse, MultiResponses
from antenna.losses.mirror import mirror
from antenna.types import CallableParam, CustomSModel, ResultType


class MirrorCVAE(nn.Module, Generic[CustomSModel]):
    """
    結合了 CVAE 解碼器 (生成器) 與鏡像/評估/選擇機制的模組。

    forward() 方法會執行以下操作：
    1. 根據輸入的條件 (c) 和一個隨機採樣的潛在向量 (z) 生成一個 "基礎 pattern"。
    2. 使用 mirror() 函數 [cite: 331-332] 產生多個鏡像版本的 pattern 。
    3. 使用傳入的 surrogate model (smodel) 評估所有鏡像 pattern 。
    4. 找出 "最佳" (smodel 損失最低) 的 pattern 。
    5. 回傳這個最佳的 pattern 及其對應的 smodel 損失，兩者都帶有梯度，
       可直接用於反向傳播 。
    """

    def __init__(
        self,
        latent_dim: int,
        smodel: CustomSModel,  # 您預先訓練好的 surrogate model
        lower_pattern: AntennaPattern = None,  # 靜態的 'lower' pattern
    ):
        """
        初始化 MirrorCVAE 生成器。

        Args:
            latent_dim (int): CVAE 的潛在向量維度 (例如: 128)。
            smodel (nn.Module): 一個預先訓練好的代理模型 (Surrogate Model)。
                             這個模型將被用來 "評估" 鏡像 pattern 的好壞。
            lower_pattern (AntennaPattern): 要添加到每個 pattern 上的靜態 'lower' 部分。
        """
        super().__init__()

        self.latent_dim = latent_dim
        condition_dim = AntennaResponse.size(flatten=True)
        pattern_dim = AntennaPattern.size(flatten=True)

        # --- 儲存外部模組 ---
        if not callable(smodel):
            raise TypeError("smodel 必須是一個可呼叫的 nn.Module")

        # 儲存 smodel，並凍結其參數 (如果 smodel 在別處訓練)
        self.smodel = smodel
        self.smodel.requires_grad(False)

        self.lower = lower_pattern

        # --- CVAE 解碼器 (生成器) ---
        # !!
        # !! 請將這個 self.decoder 替換為您自己的 CVAE 解碼器架構
        # !!
        # 這裡使用一個基於 OldGEN [cite: 405-407] 的範例架構
        self.decoder = nn.Sequential(
            # 輸入維度 = 潛在向量 + 條件
            nn.Linear(latent_dim + condition_dim, 1024),
            nn.PReLU(),
            nn.Linear(1024, 1024),
            nn.PReLU(),
            nn.Linear(1024, pattern_dim),
            nn.Sigmoid(),  # 確保輸出在 0 到 1 之間
        )

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        解碼器：將潛在向量 z 和條件 c 轉換為基礎 pattern 張量。
        """
        inputs = torch.cat([z, c], dim=0)
        base_pattern_tensor = self.decoder(inputs)
        return base_pattern_tensor

    def forward(self, c: torch.Tensor, z: torch.Tensor | None = None) -> list[ResultType]:
        """
                執行 "產生-鏡像-評估-選擇" 的前向傳播。

                Args:
                    c (torch.Tensor):
        條件向量 (例如 AntennaResponse.target.concat())。
                    z (Optional[torch.Tensor], optional): 一個固定的潛在向量 (用於可重現的生成)。
                                                          如果為 None，將隨機採樣。

                Returns:
                    Tuple[AntennaPattern, torch.Tensor]:
                    - best_pattern (AntennaPattern):
        評估後最佳的鏡像 pattern 物件。
                                                   梯度會連結到這個物件。
                    - best_fake_loss (torch.Tensor):
        來自 smodel 對 best_pattern 的評估損失。
                                                    這是您應該呼叫 .backward() 的損失張量。
        """
        if z is None:
            # 如果未提供 z，則隨機採樣一個
            z = torch.randn(self.latent_dim, device=c.device)

        # 1. 解碼 (生成) 基礎 pattern 張量
        # 這個張量帶有來自解碼器的梯度
        base_pattern_tensor = self.decode(z, c)
        base_pattern_obj = AntennaPattern(base_pattern_tensor)

        # 2. 產生鏡像 patterns
        # 這個鏡像操作 (cat, flip) 是可微分的 [cite: 331-343]
        mirrored_tensors = mirror(base_pattern_obj.merge(), mode="-|*")
        mirrored_patterns = [
            AntennaPattern(t) + self.lower if self.lower else AntennaPattern(t) for t in mirrored_tensors
        ]
        # 3. 評估所有鏡像 patterns
        all_losses: list[torch.Tensor] = []
        all_results: list[MultiResponses] = []
        self.smodel.model.eval()  # 確保 smodel 處於評估模式

        with torch.enable_grad():  # 確保在 smodel 內部計算時保留梯度
            for pattern in mirrored_patterns:
                # 讓梯度流經 smodel

                output_result = self.smodel(pattern.series)
                fake_loss = output_result.criterion()
                all_losses.append(fake_loss)
                all_results.append(output_result)

        # 4. 找出最佳損失的索引
        # 我們在 .detach() 後的張量上執行 argmin，
        # 這樣 "選擇" 操作本身 (argmin) 就不會接收梯度。
        losses_tensor_detached = torch.stack([l.detach() for l in all_losses])
        best_loss_index = torch.argmin(losses_tensor_detached)

        # (可選) 填充您在 `train_single_mirror.py`  中使用的 results 列表，用於繪圖
        results: list[ResultType] = []
        for i, pattern in enumerate(mirrored_patterns):
            result_dict: ResultType = {
                "pattern": pattern,
                "real_result": None,
                "fake_result": all_results[i],
                "real_loss": None,
                "fake_loss": all_losses[i],
                "sm_loss": [],  # sm_loss 在 HFSS 模擬後才更新
                "time": 0,  # time 在 HFSS 模擬後才更新
                "sort_key": all_losses[i].item(),
                "is_best": (i == best_loss_index.item()),
            }
            results.append(result_dict)

            results_sorted: list[ResultType] = sorted(results, key=lambda x: x["sort_key"])

        return results_sorted
