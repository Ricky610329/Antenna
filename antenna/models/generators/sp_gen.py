"""Small Pattern GEN：以子 pattern table 拼合出大尺寸天線圖的生成器。"""

from functools import partial
from typing import Generic

import numpy as np
import torch
from torch import Tensor, nn

from antenna.losses.mirror import gumbel_sinkhorn_rectangular
from antenna.types import Callable, CallableParam, Tuple
from antenna.utils.figure import Axes, Figure


class SPGEN(nn.Module, Generic[CallableParam]):
    """以 Gumbel-Sinkhorn 指派子 pattern 拼合成完整天線圖的生成器。

    設計上假定所有子 pattern 皆為同尺寸的正方形（``small_h == small_w``），
    並以 ``grid_size = size // small_size`` 做網格排列。
    """

    def __init__(
        self,
        pattern_table: Tuple,
        size: int = 40,
        gumbel_fn: Callable[CallableParam, Tensor] = gumbel_sinkhorn_rectangular,
        **gumbel_fn_kwargs,
    ):
        super().__init__()

        self.pattern_table = pattern_table
        self.pattern_table_tensor = self._to_tensor()
        self.num_patterns = len(pattern_table)  # [Channels] 子 pattern 數量
        # [big_h, big_w] 子 pattern 在大圖中的排列網格
        self.grid_size = size // self.pattern_size
        self.logits = nn.Parameter(
            # [batch, big_h, big_w, Channels]
            torch.randn(1, self.grid_size, self.grid_size, self.num_patterns),
            requires_grad=True,
        )
        self.gumbel_fn: Callable[CallableParam, Tensor] = partial(gumbel_fn, **gumbel_fn_kwargs)

    def __str__(self):
        return f"SPGEN(total={self.pattern_size * self.grid_size}(small[{self.pattern_size}]xbig[{self.grid_size}))"

    def _to_tensor(self) -> Tensor:
        """將 ``pattern_table`` 轉為 ``[Channels, small_h * small_w]`` 的張量。

        同時於首次呼叫時記錄子 pattern 的邊長為 ``self.pattern_size``。
        """
        flattened = [np.array(pattern, dtype=np.int16).reshape(-1) for pattern in self.pattern_table]
        # 假定所有子 pattern 為同尺寸正方形（使用第一個的邊長）
        self.pattern_size: int = len(self.pattern_table[0])
        return torch.tensor(np.stack(flattened), dtype=torch.float32)

    def forward(self, tau: float = 1.0, hard: bool = True):
        # 原始 logits 形狀: [1, grid_h, grid_w, num_patterns]
        batch_size, grid_h, grid_w, num_patterns = self.logits.shape

        # 1. 將 grid_h, grid_w 合併為一個「位置」維度，以符合 gumbel_sinkhorn_rectangular 輸入
        #    新形狀: [1, grid_h * grid_w, num_patterns]
        num_positions = grid_h * grid_w
        reshaped_logits = self.logits.view(batch_size, num_positions, num_patterns)

        # 2. Gumbel-Sinkhorn：訓練時 hard=False、推論時可設 True
        #    (n_iters 由 __init__ 的 gumbel_fn_kwargs 透過 partial 綁定；未指定則沿用 gumbel_fn 預設)
        #    輸出 assignment_matrix 形狀: [1, grid_h * grid_w, num_patterns]
        assignment_matrix = self.gumbel_fn(reshaped_logits, tau=tau, hard=hard)

        # 3. 以矩陣乘法挑選子 pattern
        #    [1, K, M] @ [M, S*S] -> [1, K, S*S]（K=num_positions, M=num_patterns, S=pattern_size）
        selected_patterns = torch.matmul(assignment_matrix, self.pattern_table_tensor)

        # 4. 重塑回最終圖像形狀
        soft_output = (
            selected_patterns.view(
                batch_size,
                self.grid_size,
                self.grid_size,  # batch, grid_h, grid_w
                self.pattern_size,
                self.pattern_size,  # small_h, small_w
            )
            .permute(0, 1, 3, 2, 4)  # -> (batch, grid_h, small_h, grid_w, small_w)
            .reshape(batch_size, self.grid_size * self.pattern_size, self.grid_size * self.pattern_size)
        )

        self.output_image = soft_output
        return self.output_image

    def save(self, nrowcol: tuple, result_path, pattern_dict: dict | None = None):
        """將子 pattern 表格繪製出來並存檔。"""
        with Figure(
            "SPGEN Small Pattern",
            nrowcol,
            save=True,
            rootdir=result_path,
            size=(18, 12),
            default_axes_title_size=10,
            default_tick_size=6,
        ) as fig:
            if pattern_dict:
                for name, pattern in pattern_dict.items():
                    ax: Axes = fig.index(-1)
                    ax.axis("off")
                    ax.set_title(f"{name}")
                    ax.imshow(pattern, cmap="viridis")
            else:
                for n in range(len(self)):
                    ax: Axes = fig.index(-1)
                    ax.axis("off")
                    ax.set_title(f"Small Pattern {n + 1}")
                    ax.imshow(self[n], cmap="viridis")

    def __getitem__(self, idx):
        return self.pattern_table[idx]

    def __len__(self):
        return len(self.pattern_table)
