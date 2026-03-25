"""AntennaPattern 核心類別。"""

from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from antenna.types import *
from antenna.utils.config import config
from antenna.utils.data import size_converter


class AntennaPattern:
    _history_datas: List[List[torch.Tensor]] = []
    _best_loss = float("inf")

    tau: float = 1.0
    """
    The temperature parameter controls the steepness of the Sigmoid.
    - A smaller tau (e.g., 0.1) makes the approximation closer to hard binarization.
    - It must be > 0.
    """

    def __new__(cls, pattern: "AntennaPattern", *args) -> "AntennaPattern":
        if isinstance(pattern, AntennaPattern):
            return pattern
        else:
            return super().__new__(cls)

    @overload
    def __init__(self, pattern: Tensor, coordinate: Optional[Tuple[int, int, int, int]] = None):
        """
        Example:
        ```
        AntennaPattern.setCoordinate((0, 25, 0, 25))
        ```
        """

    @overload
    def __init__(self, patterns: List[Tuple[Tensor, int, int, int, int]]):
        """
        Args:
            pattern: [(pattern, x1, x2, y1, y2), ...] >>> pattern is 2D
        """

    def __init__(self, pattern: Union[Tensor, List], coordinate: Optional[Tuple[int, int, int, int]] = None):

        if isinstance(pattern, AntennaPattern):
            return

        # * The core of this class.
        # ? [(pattern, x1, x2, y1, y2), ...] >>> pattern is 2D
        self.patterns: List[Tuple[Tensor, int, int, int, int]] = []

        if isinstance(pattern, Tensor):
            self.input_tensor = torch.clamp(pattern.to(config.device), min=0.0, max=1.0)
            self.coordinate: Union[Tuple[int, int, int, int], Tuple] = coordinate or getattr(
                self, "_antenna_pattern_coordinate", None
            )

            self._check_input()

        elif isinstance(pattern, List):
            self.patterns = pattern

        else:
            raise TypeError(f"Expected type for pattern is Tensor or List, but got {type(pattern)}")

    def _check_input(self):
        _dim = self.input_dim()
        _c = self.coordinate
        _input_tensor = self.input_tensor

        if not _c:
            raise ValueError(
                "Please enter the `coordinate` parameter or use `setDefaultCoordinate()` to set the default value."
            )
        if _dim == 1:
            _input_tensor = _input_tensor.reshape((_c[1] - _c[0], _c[3] - _c[2]))
        elif _dim == 2:
            pass
        else:
            raise ValueError(f"Input pattern expected >1 dimension, but got {_dim} dimension")

        self.patterns.append((_input_tensor, _c[0], _c[1], _c[2], _c[3]))

    @property
    def series(self):
        """One-dimensional array after merge."""
        return self.merge().reshape(-1)

    @property
    def fill_rate(self) -> float:
        """計算並返回天線 pattern 的金屬填充率。"""
        merged_pattern = self.merge()
        if merged_pattern.numel() == 0:
            return 0.0
        return (torch.sum(merged_pattern) / merged_pattern.numel()).item()

    @classmethod
    def register_simulator(cls, simulator: Callable[[Tensor], Dict[str, Tensor]]):
        cls._simulator = simulator

    @classmethod
    def getAllPixel(cls):
        """
        TODO: 目前是取回所有的像素點，但實際上是取得大圖的像素點
        """
        x1, x2, y1, y2 = cast(Tuple[int, int, int, int], getattr(cls, "_antenna_pattern_coordinate", (0, 0, 0, 0)))
        return (x2 - x1) * (y2 - y1)

    @overload
    @classmethod
    def size(cls, flatten: Literal[True]) -> int: ...
    @overload
    @classmethod
    def size(cls, flatten: Literal[False]) -> Tuple[int, int]: ...
    @overload
    @classmethod
    def size(cls) -> Tuple[int, int]: ...
    @classmethod
    def size(cls, flatten: bool = False):
        """The number of labels used to calculate loss and the number of points in their labels."""
        if not hasattr(cls, "_antenna_pattern_coordinate"):
            raise RuntimeError("Please use `setDefaultCoordinate()` first.")
        x1, x2, y1, y2 = cast(Tuple[int, int, int, int], getattr(cls, "_antenna_pattern_coordinate", (0, 0, 0, 0)))

        return (x2 - x1) * (y2 - y1) if flatten else ((x2 - x1), (y2 - y1))

    def size_converter(self, flatten: bool = False, batch: bool = False, output_shape=None) -> torch.Tensor:
        """:param output_shape: Priority use. (B, H, W, N) EX: "B, 1, H, W" or "B, N, 1" """
        return size_converter(self, self.merge(), flatten=flatten, batch=batch, output_shape=output_shape)

    @classmethod
    def _getRandomPattern(cls, w=40, h=40):
        patterns = torch.randn(w, h, dtype=torch.float32, device=config.device)
        binaries = (patterns > 0.5).float()
        return cls(binaries, (0, w, 0, h))

    @classmethod
    def getRandomPattern(cls, shape: tuple, fill_rate: float = 0.5) -> Self:
        """
        根據指定的填充率 (fill rate) 生成一個二元 (0/1) 的 pattern。

        Args:
            shape (tuple): Pattern 的形狀, 例如 (w, h)。
            fill_rate (float): 金屬填充的比例, 範圍在 0.0 到 1.0 之間。

        Returns:
            生成的二元 pattern。

        Exapmple::

            AntennaPattern.getRandomPattern((25, 25), fill_rate = np.random.uniform(0.1, 0.9))
        """
        w = shape[0]
        h = shape[1]
        total_pixels = w * h
        num_ones = int(total_pixels * fill_rate)

        # 生成一個扁平化的一維數組
        pattern_flat = np.zeros(total_pixels)
        pattern_flat[:num_ones] = 1

        # 隨機打亂
        np.random.shuffle(pattern_flat)

        # 重塑為目標形狀並轉換為 PyTorch Tensor
        pattern_tensor = torch.tensor(pattern_flat.reshape(shape), dtype=torch.float32, device=config.device)
        return cls(pattern_tensor, (0, w, 0, h))

    def __str__(self):
        _shape = self.merge().shape
        return f"AntennaPattern(Pattern_num={self.__len__()} Shape=[{_shape[0]}, {_shape[1]}] Size=[{_shape.numel()}])"

    def __getitem__(self, key) -> "AntennaPattern":
        if key >= self.__len__():
            raise IndexError(f"Expected size {self.__len__()} but got size {key}")
        pattern, x1, x2, y1, y2 = self.patterns[key]
        return AntennaPattern(pattern, (x1, x2, y1, y2))

    def __add__(self, other):
        if isinstance(other, AntennaPattern):
            antenna_pattern = self.copy()
            antenna_pattern.patterns = self.patterns + other.patterns
            antenna_pattern.coordinate = None
            antenna_pattern.input_tensor = None

            return antenna_pattern
        else:
            raise TypeError(f"Unsupported operand type for +: 'AntennaPattern' and '{type(other)}'")

    def __len__(self):
        return len(self.patterns)

    def __invert__(self):
        """Detach the response"""
        return self.merge().detach().cpu()

    def input_dim(self) -> int:
        if self.input_tensor is None:
            raise RuntimeError("This function is not for multilayer boards.")

        if len(self.input_tensor.shape) == 1 or self.input_tensor.shape[0] == 1:
            return 1
        else:
            return self.input_tensor.dim()

    def copy(self):
        return AntennaPattern(self.patterns)

    @classmethod
    def setDefaultCoordinate(cls, _coordinate: Tuple[int, int, int, int]):
        """
        Coordinate Design.

        """
        if not isinstance(_coordinate, tuple):
            raise TypeError(f"Expected tuple, but got {type(_coordinate)}")

        if not len(_coordinate) == 4:
            raise ValueError(f"Expected tuple of length 4, but got {len(_coordinate)}")

        setattr(cls, "_antenna_pattern_coordinate", _coordinate)

    def binarize(self, threshold=0.5):
        """Binarize and become gradient-free."""
        bi = (self.merge() >= threshold).float()
        return AntennaPattern(bi, (0, len(bi), 0, len(bi)))

    @classmethod
    def binarization(cls, pattern: Tensor, tau: Optional[float] = None, threshold=None, *, only_soft: bool = False):
        """
        Perform differentiable binarization using the STE technique.

        Args:
            pattern: The pattern to be binarized. Its requires_grad will be set to True.
            tau:   The temperature parameter controls the steepness of the Sigmoid.
                    A smaller tau (e.g., 0.1) makes the approximation closer to hard binarization.
                    It must be > 0.

        Returns:
            torch.Tensor: Binarized tensor
        """
        # * Gradient is required
        pattern.requires_grad_(True)
        cls.tau: float = tau or getattr(cls, "tau", 1.0)
        if cls.tau < 1e-4:
            cls.tau = 1e-4

        if len(pattern.shape) == 1:
            pattern = pattern.reshape(*cls.size())

        # 將 logits 限制在 [-10, 10] 之間，Sigmoid(-10) 已極趨近 0，Sigmoid(10) 極趨近 1
        pattern = torch.clamp(pattern, min=-10.0, max=10.0)

        # * Calculate threshold and steepness
        threshold = threshold or pattern.mean().detach()  # avg
        steepness = 1 / cls.tau

        # * Produces a "soft" approximation
        #  This is to provide a smooth gradient during "backward" propagation.
        soft_pattern = torch.sigmoid(steepness * (pattern - threshold))
        if torch.isnan(soft_pattern).any():
            soft_pattern = torch.nan_to_num(soft_pattern, nan=0.5)

        if only_soft is True:
            return soft_pattern

        # * Produces a "hard" binarization result (0/1, not differentiable).
        #  This is to get the 0/1 result you want during "forward" propagation.
        hard_pattern = torch.round(soft_pattern)

        # * STE
        #  Forward(hard):   (hard - soft) + soft
        #  Backward(soft)： `.detach()` will block the gradient of hard_pattern
        binary_pattern = (hard_pattern - soft_pattern).detach() + soft_pattern

        return binary_pattern

    def binarization_(self, tau: Optional[float] = None, threshold=None):
        pattern = self.merge().clone()
        shape = pattern.shape
        self.patterns = [(AntennaPattern.binarization(pattern, tau=tau, threshold=threshold), 0, shape[1], 0, shape[0])]

    def merge(self) -> torch.Tensor:
        """
        將所有 pattern 合併成一個大的底層 pattern
        - 後加入的 pattern 會覆蓋前面的 pattern
        - 返回合併後的二維 tensor
        """
        if not self.patterns:
            raise ValueError("No patterns to merge")

        max_x = max(x2 for _, _, x2, _, _ in self.patterns)
        min_x = min(x1 for _, x1, _, _, _ in self.patterns)

        max_y = max(y2 for _, _, _, _, y2 in self.patterns)
        min_y = min(y1 for _, _, _, y1, _ in self.patterns)

        base_pattern = torch.zeros((max_y, max_x))
        for pattern, x1, x2, y1, y2 in self.patterns:
            base_pattern[y1:y2, x1:x2] = pattern  # 後面的 pattern 覆蓋前面的

        return base_pattern.to(config.device)[min_y:max_y, min_x:max_x]

    def simulate(self, no_grad: bool = True, **param):
        from antenna.core.response import AntennaResponse

        pattern = self.merge()
        result_response = {}

        if hasattr(self, "_simulator"):
            if no_grad:
                with torch.no_grad():
                    result: Dict[str, Tensor] = self._simulator(pattern.detach(), **param)
            else:
                result: Dict[str, Tensor] = self._simulator(pattern, **param)
        else:
            raise RuntimeError("Please use `register_simulator()` to register the simulator.")

        for key, value in result.items():
            result_response[key] = AntennaResponse(value)

        # TODO
        # if not any([pattern.equal(p) for p, _ in self._history_datas]):
        AntennaPattern._history_datas.append([pattern, result_response])

        return AntennaResponse(result_response)

    def plot(self, axes: Optional[Axes] = None, show: bool = False, title: str = "Antenna Pattern {shape}"):
        ax: Axes = plt.axes(axes)  # type: ignore
        ax.set_title(title.format(shape=self.size()))
        ax.imshow(self.merge().cpu().detach(), cmap="viridis")
        ax.axis("off")
        if show:
            plt.show()
        return ax

    def plot_individual(self, axes: Optional[Axes] = None, show: bool = False):
        if not self.patterns:
            raise ValueError("No patterns to merge")

        max_x = max(x2 for _, _, x2, _, _ in self.patterns)
        max_y = max(y2 for _, _, _, _, y2 in self.patterns)
        base_pattern = torch.zeros((max_x, max_y), dtype=self.input_tensor.dtype)
        _result = []
        for pattern, x1, x2, y1, y2 in self.patterns:
            _pattern = base_pattern.clone()
            _pattern[x1:x2, y1:y2] = pattern
            _result.append(_pattern)

        ax: Axes = plt.axes(axes)  # type: ignore
        ax.set_title("Antenna Pattern Individual")
        ax.imshow(torch.cat(_result, dim=1).cpu().detach(), cmap="viridis")
        if show:
            plt.show()
        return ax

    def mutate(self, rate):
        matrix = self.merge()
        total = matrix.numel()
        n = int(total * rate)
        indices = torch.randperm(total).tolist()
        selected_indices = indices[:n]

        for idx in selected_indices:
            i, j = divmod(idx, matrix.size(1))
            matrix[i, j] = 1 - matrix[i, j]
        return AntennaPattern(matrix)

    def total_variation_loss(self, weight=0.01):
        """計算 Total Variation Loss 以抑制過度破碎的圖樣"""
        img = self.merge()
        h_img, w_img = img.size()

        tv_h = torch.pow(img[1:, :] - img[:-1, :], 2).sum()
        tv_w = torch.pow(img[:, 1:] - img[:, :-1], 2).sum()

        return weight * (tv_h + tv_w) / (h_img * w_img)

    def island_suppression_loss(self, weight: float = 1.0, kernel_size: int = 5) -> torch.Tensor:
        """
        孤島抑制損失 (Island Suppression Loss)。
        專為「沒有參考圖樣」的情況設計。
        透過計算像素與其「局部鄰域平均值」的差異，來抑制孤立的噪點 (孤島) 或孔洞。
        這類似於一種局部平滑約束。

        Args:
            weight: 權重。
            kernel_size: 鄰域視窗大小，建議使用奇數 (如 3 或 5)。較大的 kernel 會促成更大的連通區塊。
        """
        img = self.merge()

        # 確保為浮點數
        if not img.is_floating_point():
            img = img.float()

        # 準備進行 2D Pooling: 需要 (Batch, Channel, Height, Width)
        # 這裡假設 img 為 (H, W)，擴展為 (1, 1, H, W)
        img_input = img.unsqueeze(0).unsqueeze(0)

        # 計算局部平均 (Local Average)
        # Padding 設為 kernel_size // 2 以保持輸出尺寸不變
        avg_img = F.avg_pool2d(img_input, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

        # 去掉多餘維度回歸 (H, W)
        avg_img = avg_img.squeeze(0).squeeze(0)

        # 計算像素與局部平均的 L1 差異
        loss = torch.abs(img - avg_img).sum()

        return weight * loss / img.numel()
