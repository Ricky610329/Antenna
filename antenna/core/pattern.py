"""AntennaPattern 核心類別。"""

from typing import Callable, Literal, Self, cast, overload

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.axes import Axes
from torch import Tensor

from antenna.utils.config import config
from antenna.utils.data import size_converter

# * Sigmoid logits 的截斷範圍
# Sigmoid(-10) 已極趨近 0，Sigmoid(10) 極趨近 1，超出範圍只會造成數值不穩
_SIGMOID_LOGIT_CLAMP: float = 10.0

# * Tau 下限：避免除以 0 或 steepness 爆炸
_TAU_MIN: float = 1e-4

# * 二值化預設門檻
_DEFAULT_BINARIZE_THRESHOLD: float = 0.5

# * Coordinate 的類別屬性名稱（存放在 class 上）
_COORDINATE_ATTR: str = "_antenna_pattern_coordinate"


class AntennaPattern:
    """
    天線 pattern 的核心容器。

    以「多個帶有座標的 2D Tensor」為內部表示，支援：
    - 合併 (merge) 為單一底層 pattern
    - 可微分二值化 (STE binarization)
    - 隨機生成 / 突變 / 填充率計算
    - 與註冊的 simulator 搭配進行 HFSS 模擬

    Coordinate 格式：``(x1, x2, y1, y2)``，其中 ``x`` 為寬度方向、``y`` 為高度方向。
    """

    _history_datas: list[list[torch.Tensor]] = []
    _best_loss = float("inf")

    tau: float = 1.0
    """Sigmoid 溫度參數，控制 soft binarization 的陡峭度。
    - 越小 (例如 0.1) 越接近硬性二值化。
    - 必須 > 0。
    """

    def __new__(cls, pattern: "AntennaPattern", *args) -> "AntennaPattern":
        # 若傳入的已是 AntennaPattern，直接回傳原物件；__init__ 也會 early return 避免重複初始化
        if isinstance(pattern, AntennaPattern):
            return pattern
        return super().__new__(cls)

    @overload
    def __init__(self, pattern: Tensor, coordinate: tuple[int, int, int, int] | None = None):
        """
        Example::

            AntennaPattern.setDefaultCoordinate((0, 25, 0, 25))
        """

    @overload
    def __init__(self, patterns: list[tuple[Tensor, int, int, int, int]]):
        """
        Args:
            pattern: ``[(pattern, x1, x2, y1, y2), ...]`` 其中 pattern 為 2D。
        """

    def __init__(self, pattern: Tensor | list, coordinate: tuple[int, int, int, int] | None = None):
        # 對應 __new__ 回傳既有物件的情境，避免重複初始化覆蓋原狀態
        if isinstance(pattern, AntennaPattern):
            return

        # * 核心資料結構：[(pattern, x1, x2, y1, y2), ...]，其中 pattern 為 2D
        self.patterns: list[tuple[Tensor, int, int, int, int]] = []

        if isinstance(pattern, Tensor):
            self.input_tensor = torch.clamp(pattern.to(config.device), min=0.0, max=1.0)
            self.coordinate: tuple[int, int, int, int] | tuple = coordinate or getattr(self, _COORDINATE_ATTR, None)
            self._check_input()

        elif isinstance(pattern, list):
            self.patterns = pattern

        else:
            raise TypeError(f"Expected type for pattern is Tensor or List, but got {type(pattern)}")

    def _check_input(self):
        """驗證 `input_tensor` 與 `coordinate` 並加入 `patterns`。"""
        _dim = self.input_dim()
        _c = self.coordinate
        _input_tensor = self.input_tensor

        if not _c:
            raise ValueError(
                "Please enter the `coordinate` parameter or use `setDefaultCoordinate()` to set the default value."
            )
        if _dim == 1:
            _input_tensor = _input_tensor.reshape((_c[1] - _c[0], _c[3] - _c[2]))
        elif _dim != 2:
            raise ValueError(f"Input pattern expected >1 dimension, but got {_dim} dimension")

        self.patterns.append((_input_tensor, _c[0], _c[1], _c[2], _c[3]))

    @property
    def series(self):
        """合併後攤平為 1D 的張量。"""
        return self.merge().reshape(-1)

    @property
    def fill_rate(self) -> float:
        """計算並返回天線 pattern 的金屬填充率。"""
        merged_pattern = self.merge()
        if merged_pattern.numel() == 0:
            return 0.0
        return (torch.sum(merged_pattern) / merged_pattern.numel()).item()

    @classmethod
    def register_simulator(cls, simulator: Callable[[Tensor], dict[str, Tensor]]):
        """註冊 HFSS 或其他 simulator callable。"""
        cls._simulator = simulator

    @classmethod
    def getAllPixel(cls):
        """取得預設 coordinate 所定義的像素總數。

        等同於 ``size(flatten=True)``，但在未設定 coordinate 時回傳 0 而非 raise。
        """
        x1, x2, y1, y2 = cast(tuple[int, int, int, int], getattr(cls, _COORDINATE_ATTR, (0, 0, 0, 0)))
        return (x2 - x1) * (y2 - y1)

    @overload
    @classmethod
    def size(cls, flatten: Literal[True]) -> int: ...
    @overload
    @classmethod
    def size(cls, flatten: Literal[False]) -> tuple[int, int]: ...
    @overload
    @classmethod
    def size(cls) -> tuple[int, int]: ...
    @classmethod
    def size(cls, flatten: bool = False):
        """取得預設 coordinate 的寬高 (W, H) 或攤平後總像素數。"""
        if not hasattr(cls, _COORDINATE_ATTR):
            raise RuntimeError("Please use `setDefaultCoordinate()` first.")
        x1, x2, y1, y2 = cast(tuple[int, int, int, int], getattr(cls, _COORDINATE_ATTR, (0, 0, 0, 0)))

        return (x2 - x1) * (y2 - y1) if flatten else ((x2 - x1), (y2 - y1))

    def size_converter(self, flatten: bool = False, batch: bool = False, output_shape=None) -> torch.Tensor:
        """依 ``output_shape`` 或 flatten/batch 參數調整 merge 後張量的形狀。

        :param output_shape: 優先使用 (B, H, W, N)，例如 ``"B, 1, H, W"`` 或 ``"B, N, 1"``。
        """
        return size_converter(self, self.merge(), flatten=flatten, batch=batch, output_shape=output_shape)

    @classmethod
    def getRandomPattern(cls, shape: tuple, fill_rate: float = 0.5) -> Self:
        """
        根據指定的填充率 (fill rate) 生成一個二元 (0/1) 的 pattern。

        Args:
            shape (tuple): Pattern 的形狀, 例如 (w, h)。
            fill_rate (float): 金屬填充的比例, 範圍在 0.0 到 1.0 之間。

        Returns:
            生成的二元 pattern。

        Example::

            AntennaPattern.getRandomPattern((25, 25), fill_rate=np.random.uniform(0.1, 0.9))
        """
        w, h = shape[0], shape[1]
        total_pixels = w * h
        num_ones = int(total_pixels * fill_rate)

        # 生成扁平化的一維數組（前 num_ones 個為 1）後隨機打亂
        pattern_flat = np.zeros(total_pixels)
        pattern_flat[:num_ones] = 1
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
        raise TypeError(f"Unsupported operand type for +: 'AntennaPattern' and '{type(other)}'")

    def __len__(self):
        return len(self.patterns)

    def __invert__(self):
        """Detach 並搬回 CPU 的合併結果。"""
        return self.merge().detach().cpu()

    def input_dim(self) -> int:
        """回傳 `input_tensor` 的維度（1 或 2）。僅適用於單層 pattern。"""
        if self.input_tensor is None:
            raise RuntimeError("This function is not for multilayer boards.")

        if len(self.input_tensor.shape) == 1 or self.input_tensor.shape[0] == 1:
            return 1
        return self.input_tensor.dim()

    def copy(self):
        """淺拷貝一個新的 AntennaPattern（共享底層 patterns list 之元素）。"""
        return AntennaPattern(self.patterns)

    @classmethod
    def setDefaultCoordinate(cls, _coordinate: tuple[int, int, int, int]):
        """設定 class-level 預設 coordinate。

        Args:
            _coordinate: ``(x1, x2, y1, y2)`` 4 元素 tuple。
        """
        if not isinstance(_coordinate, tuple):
            raise TypeError(f"Expected tuple, but got {type(_coordinate)}")

        if len(_coordinate) != 4:
            raise ValueError(f"Expected tuple of length 4, but got {len(_coordinate)}")

        setattr(cls, _COORDINATE_ATTR, _coordinate)

    def binarize(self, threshold: float = _DEFAULT_BINARIZE_THRESHOLD):
        """硬性二值化 (gradient-free)。回傳新的 `AntennaPattern`。"""
        bi = (self.merge() >= threshold).float()
        # bi 的形狀為 (H, W)，coordinate 對應 (x1, x2, y1, y2) 即 (W_start, W_end, H_start, H_end)
        h, w = bi.shape[0], bi.shape[1]
        return AntennaPattern(bi, (0, w, 0, h))

    @classmethod
    def binarization(
        cls,
        pattern: Tensor,
        tau: float | None = None,
        threshold=None,
        *,
        only_soft: bool = False,
    ):
        """
        使用 STE (Straight-Through Estimator) 技術的可微分二值化。

        Args:
            pattern: 要二值化的 tensor；會被設定 ``requires_grad=True``。
            tau: Sigmoid 溫度參數；越小越接近硬性二值化，必須 > 0。
            threshold: 二值化門檻；None 時使用 ``pattern.mean()``。
            only_soft: 若為 True，只回傳 soft sigmoid 輸出（完全可微）。

        Returns:
            torch.Tensor: 二值化後的張量（forward 為 0/1，backward 走 soft sigmoid 的梯度）。
        """
        # * Gradient is required
        pattern.requires_grad_(True)
        cls.tau = float(tau) if tau is not None else getattr(cls, "tau", 1.0)
        if cls.tau < _TAU_MIN:
            cls.tau = _TAU_MIN

        if len(pattern.shape) == 1:
            pattern = pattern.reshape(*cls.size())

        # 將 logits 限制在 [-CLAMP, CLAMP] 之間以避免 Sigmoid 數值飽和
        pattern = torch.clamp(pattern, min=-_SIGMOID_LOGIT_CLAMP, max=_SIGMOID_LOGIT_CLAMP)

        # * 計算 threshold 與 steepness
        threshold = threshold if threshold is not None else pattern.mean().detach()
        steepness = 1 / cls.tau

        # * 產生 soft 近似 (backward 時提供平滑梯度)
        soft_pattern = torch.sigmoid(steepness * (pattern - threshold))
        if torch.isnan(soft_pattern).any():
            soft_pattern = torch.nan_to_num(soft_pattern, nan=0.5)

        if only_soft:
            return soft_pattern

        # * 產生 hard 二值化結果 (0/1，不可微)，forward 時實際使用
        hard_pattern = torch.round(soft_pattern)

        # * STE：Forward 為 hard，Backward 透過 soft 的梯度回傳
        binary_pattern = (hard_pattern - soft_pattern).detach() + soft_pattern

        return binary_pattern

    def binarization_(self, tau: float | None = None, threshold=None):
        """原地 (in-place) 將 patterns 替換為可微分二值化後的單一 pattern。"""
        pattern = self.merge().clone()
        shape = pattern.shape
        self.patterns = [(AntennaPattern.binarization(pattern, tau=tau, threshold=threshold), 0, shape[1], 0, shape[0])]

    def _bounding_box(self) -> tuple[int, int, int, int]:
        """計算所有 sub-pattern 的 bounding box: ``(min_x, max_x, min_y, max_y)``。"""
        if not self.patterns:
            raise ValueError("No patterns to merge")

        max_x = max(x2 for _, _, x2, _, _ in self.patterns)
        min_x = min(x1 for _, x1, _, _, _ in self.patterns)
        max_y = max(y2 for _, _, _, _, y2 in self.patterns)
        min_y = min(y1 for _, _, _, y1, _ in self.patterns)
        return min_x, max_x, min_y, max_y

    def merge(self) -> torch.Tensor:
        """
        將所有 pattern 合併成一個大的底層 pattern。

        - 後加入的 pattern 會覆蓋前面的 pattern
        - 返回合併後的二維 tensor (H, W)
        """
        min_x, max_x, min_y, max_y = self._bounding_box()

        base_pattern = torch.zeros((max_y, max_x))
        for pattern, x1, x2, y1, y2 in self.patterns:
            base_pattern[y1:y2, x1:x2] = pattern  # 後面的 pattern 覆蓋前面的

        return base_pattern.to(config.device)[min_y:max_y, min_x:max_x]

    def simulate(self, no_grad: bool = True, **param):
        """呼叫已註冊的 simulator 並包裝成 `AntennaResponse`。"""
        from antenna.core.response import AntennaResponse

        pattern = self.merge()
        result_response = {}

        if not hasattr(self, "_simulator"):
            raise RuntimeError("Please use `register_simulator()` to register the simulator.")

        if no_grad:
            with torch.no_grad():
                result: dict[str, Tensor] = self._simulator(pattern.detach(), **param)
        else:
            result: dict[str, Tensor] = self._simulator(pattern, **param)

        for key, value in result.items():
            result_response[key] = AntennaResponse(value)

        AntennaPattern._history_datas.append([pattern, result_response])

        return AntennaResponse(result_response)

    def plot(self, axes: Axes | None = None, show: bool = False, title: str = "Antenna Pattern {shape}"):
        """繪製合併後的 pattern。"""
        ax: Axes = plt.axes(axes)  # type: ignore
        ax.set_title(title.format(shape=self.size()))
        ax.imshow(self.merge().cpu().detach(), cmap="viridis")
        ax.axis("off")
        if show:
            plt.show()
        return ax

    def plot_individual(self, axes: Axes | None = None, show: bool = False):
        """逐一繪製各個 sub-pattern (水平串接)。"""
        if not self.patterns:
            raise ValueError("No patterns to merge")

        _, max_x, _, max_y = self._bounding_box()
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
        """以指定比例隨機翻轉像素值 (0↔1)，回傳新的 `AntennaPattern`。"""
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
        """計算 Total Variation Loss 以抑制過度破碎的圖樣。"""
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
