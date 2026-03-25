"""SPGEN — Gumbel-Sinkhorn Pattern 選擇生成器。"""

from functools import partial

import numpy as np
import torch
from torch import Tensor, nn

from antenna import AntennaPattern
from antenna.functions import gumbel_sinkhorn_rectangular
from antenna.types import *
from antenna.utils import Axes, Figure


class SPGEN(nn.Module, Generic[CallableParam]):
    def __init__(
        self,
        pattern_table: Tuple,
        size=40,
        gumbel_fn: Callable[CallableParam, Tensor] = gumbel_sinkhorn_rectangular,
        **gumbel_fn_kwargs,
    ):
        super().__init__()

        self.pattern_table = pattern_table
        self.pattern_table_tensor = self._to_tensor()
        self.num_patterns = len(pattern_table)
        self.grid_size = size // self.patern_size
        self.logits = nn.Parameter(
            torch.randn(1, self.grid_size, self.grid_size, self.num_patterns),
            requires_grad=True,
        )
        self.gumbel_fn: Callable[CallableParam, Tensor] = partial(gumbel_fn, **gumbel_fn_kwargs)

    def __str__(self):
        return f"SPGEN(total={self.patern_size * self.grid_size}(small[{self.patern_size}]xbig[{self.grid_size}))"

    def _to_tensor(self):
        _reselt = []
        for pattern in self.pattern_table:
            self.patern_size: int = len(pattern)
            _reselt.append(np.array(pattern, dtype=np.int16).reshape(-1))
        return torch.tensor(np.stack(_reselt), dtype=torch.float32)

    def forward(self, tau: float = 1.0, n_iters: int = 20, hard: bool = True):
        batch_size, grid_h, grid_w, num_patterns = self.logits.shape
        num_positions = grid_h * grid_w
        reshaped_logits = self.logits.view(batch_size, num_positions, num_patterns)
        assignment_matrix = self.gumbel_fn(reshaped_logits, tau=tau, hard=hard)
        selected_patterns = torch.matmul(assignment_matrix, self.pattern_table_tensor)
        soft_output = (
            selected_patterns.view(
                batch_size,
                self.grid_size,
                self.grid_size,
                self.patern_size,
                self.patern_size,
            )
            .permute(0, 1, 3, 2, 4)
            .reshape(batch_size, self.grid_size * self.patern_size, self.grid_size * self.patern_size)
        )
        self.output_image = soft_output
        return self.output_image

    def save(self, nrowcol: tuple, result_path, pattern_dict: dict = None):
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
