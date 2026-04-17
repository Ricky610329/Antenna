"""正則化損失：Total Variation、連通性、間隙封閉與饋電點連通指標。"""

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch import Tensor, nn

from antenna.types import *
from antenna.utils.utils import Figure, plt


def total_variation_loss(img, weight: float = 0.01) -> Tensor:
    """計算 Total Variation Loss 以抑制過度破碎的圖樣。

    Args:
        img: 可為 ``AntennaPattern`` 或任意維度的張量，會被轉為 ``(B, 1, H, W)``。
        weight: 損失權重。
    """
    from antenna import AntennaPattern
    from antenna.utils.data import size_converter

    img = size_converter(AntennaPattern, img, output_shape="B, 1, H, W")
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    return weight * (tv_h + tv_w) / (bs_img * c_img * h_img * w_img)


class SpectralConnectivityLoss(nn.Module):
    """頻譜圖論法連通性損失。

    將 Pattern 視為圖 (Graph)：每個像素為節點，相鄰像素間有一條邊，
    邊的權重為兩端像素值的乘積；當兩端都接近 1 時權重才大。

    目標：最大化拉普拉斯矩陣 (Laplacian) 的第二小特徵值 (Fiedler Value)。
    若 $\\lambda_2 > 0$：圖上的金屬部分連通；若 $\\lambda_2 \\approx 0$：圖斷成多塊。
    """

    def __init__(self, height: int = 25, width: int = 25):
        super().__init__()

        self.H = height
        self.W = width
        self.num_nodes = height * width

        # 預先建立 4-連通 (上下左右) 的鄰接索引
        src: list[int] = []
        dst: list[int] = []
        for r in range(height):
            for c in range(width):
                idx = r * width + c
                if r + 1 < height:  # 下方鄰居
                    src.append(idx)
                    dst.append((r + 1) * width + c)
                if c + 1 < width:  # 右方鄰居
                    src.append(idx)
                    dst.append(r * width + (c + 1))

        # 使用 register_buffer 讓 .to(device) 自動搬運
        self.register_buffer("src", torch.as_tensor(src, dtype=torch.long), persistent=False)
        self.register_buffer("dst", torch.as_tensor(dst, dtype=torch.long), persistent=False)

    def forward(self, antenna_map: Tensor) -> Tensor:
        """
        Args:
            antenna_map: ``(Batch, 1, H, W)``，數值範圍 0~1。

        Returns:
            Loss 標量。若批次內的樣本都已足夠連通則為 0。
        """
        batch_size = antenna_map.shape[0]
        flat_map = antenna_map.reshape(batch_size, -1)
        device = antenna_map.device

        # register_buffer 已經會隨模組搬移，但保險起見做一次 to()
        src = self.src.to(device)
        dst = self.dst.to(device)

        losses = []
        for b in range(batch_size):
            # 1. 建構邊的權重：兩端像素值的乘積 (加 eps 避免全 0)
            node_vals = flat_map[b] + 1e-4
            weights = node_vals[src] * node_vals[dst]

            # 2. 建構拉普拉斯矩陣 L = D - A
            A = torch.zeros(self.num_nodes, self.num_nodes, device=device)
            A[src, dst] = weights
            A[dst, src] = weights  # 對稱

            degree = torch.sum(A, dim=1)
            L = torch.diag(degree) - A

            # 3. 實對稱矩陣的特徵值
            eigvals = torch.linalg.eigvalsh(L)

            # 4. eigvals[0] 理論上為 0，取第二小特徵值 (Fiedler Value)
            lambda_2 = eigvals[1]

            # 希望 lambda_2 越大越好；超過門檻後損失為 0
            losses.append(torch.relu(0.5 - lambda_2))

        return torch.mean(torch.stack(losses))


class GapClosingLoss(nn.Module):
    """間隙封閉損失：Closing = Dilation(膨脹) + Erosion(侵蝕)。

    若原圖有裂縫，closing 會將裂縫填滿；我們希望原圖本身就沒有裂縫，
    故以 ``||closed - original||^2`` 作為懲罰項。
    """

    def forward(self, antenna_map: Tensor) -> Tensor:
        # 1. Soft Dilation (以 MaxPool 近似)
        dilated = F.max_pool2d(antenna_map, kernel_size=3, stride=1, padding=1)
        # 2. Soft Erosion：Erosion(x) = -Max(-x)
        closed = -F.max_pool2d(-dilated, kernel_size=3, stride=1, padding=1)
        return torch.mean((closed - antenna_map) ** 2)


class FeedReachability:  # R_feed
    """共同連通指標 (Mutual Feed Connectivity Index)。

    只有當所有饋電點都落在同一塊連通金屬上時，才計算該塊的像素佔比；
    否則視為失敗並回傳 0。
    """

    #: 預設使用 4-連通 (十字形)；若想改 8-連通可改 ``np.ones((3, 3))``。
    DEFAULT_STRUCTURE = np.array(
        [[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]],
    )

    def __init__(self, feed_positions: list[tuple[int, int]]):
        """
        Args:
            feed_positions: 饋電點座標列表 ``[(r1, c1), (r2, c2), ...]``。
        """
        from scipy.ndimage import label

        assert len(feed_positions) > 0, "feed_positions 不可為空"

        self.feed_positions = feed_positions
        """饋電點座標。"""
        self.rate: float | None = None
        """最近一次計算的電流導通率。"""
        self.mask: np.ndarray | None = None
        """最近一次計算的電流導通遮罩。"""
        self.pattern: np.ndarray | None = None
        """最近一次評估的 pattern。"""
        self.title: str = ""
        """最近一次繪圖用的標題。"""
        self.structure = self.DEFAULT_STRUCTURE
        """連通性結構元素。"""
        self.record: list[FeedReachabilityDictType] = []
        self.r_feed_str = "$R_{{feed}}$"

        self._label = label

    @classmethod
    def single_feed(cls) -> "FeedReachability":
        shape = (25, 25)
        return cls([(shape[0] - 1, shape[1] // 2)])

    @classmethod
    def dual_feed(cls) -> "FeedReachability":
        shape = (25, 25)
        return cls([(shape[0] - 1, shape[1] // 2), (0, shape[1] // 2)])

    def __call__(
        self,
        pattern: Union[Tensor, np.ndarray],
        *,
        record: bool = False,
        title: str = "Pattern ($R_{{feed}}$={rate:.2%})",
    ) -> float:
        """
        Args:
            pattern: 2D array (1=金屬, 0=介質)；Tensor 會自動轉為 ndarray。
            record: 是否把本次結果附加到 ``self.record``。
            title: 繪圖用標題模板，可含 ``{rate}`` 佔位符。

        Returns:
            Feed Reachability Rate，範圍 [0, 1]。
        """
        if isinstance(pattern, Tensor):
            pattern = pattern.detach().cpu().numpy()

        labeled_array, _ = self._label(pattern, structure=self.structure)

        # 1. 取得每個饋電點所在的連通塊 label
        feed_labels: list[int] = []
        for pos in self.feed_positions:
            if not (0 <= pos[0] < pattern.shape[0] and 0 <= pos[1] < pattern.shape[1]):
                logger.error("饋入點座標越界")
                return self._fail(pattern, title)
            lbl = labeled_array[pos]
            if lbl <= 0:
                logger.warning("其中一個饋入點沒金屬，直接失敗")
                return self._fail(pattern, title)
            feed_labels.append(lbl)

        # 2. 「AND」邏輯：所有饋電點須落在同一塊
        unique_labels = set(feed_labels)
        if len(unique_labels) == 1:
            shared_label = next(iter(unique_labels))
            shared_mask = (labeled_array == shared_label).astype(pattern.dtype)
            total_metal_pixels = np.sum(pattern)
            mutual_index = float(np.sum(shared_mask) / total_metal_pixels) if total_metal_pixels > 0 else 0.0
        else:
            mutual_index = 0.0
            shared_mask = np.zeros_like(pattern)

        self._store(pattern, shared_mask, mutual_index, title, record=record)
        return mutual_index

    def _fail(self, pattern: np.ndarray, title: str) -> float:
        """饋電點無效時的 fallback：回傳 0 並清空遮罩。"""
        self._store(pattern, np.zeros_like(pattern), 0.0, title, record=False)
        return 0.0

    def _store(
        self,
        pattern: np.ndarray,
        mask: np.ndarray,
        rate: float,
        title: str,
        *,
        record: bool,
    ) -> None:
        self.rate = rate
        self.mask = mask
        self.pattern = pattern
        self.title = title.format(rate=rate)
        if record:
            self.record.append(
                {
                    "pattern": pattern,
                    "feed_positions": self.feed_positions,
                    "rate": rate,
                    "mask": mask,
                    "title": self.title,
                }
            )

    @property
    def r_feed_dict(self) -> dict:
        result = defaultdict(list)
        for entry in self.record:
            result[entry["title"]].append(entry["rate"])
        return result

    @property
    def r_feed_list(self) -> list[float]:
        return [entry["rate"] for entry in self.record]

    @property
    def rate_list(self) -> list[float]:
        return [entry["rate"] * 100 for entry in self.record]

    @property
    def r_feed_avg(self) -> float:
        if not self.record:
            return 0.0
        return float(np.mean(self.r_feed_list))

    def plot(self, axes=None, show: bool = False, data: FeedReachabilityDictType = None):
        pattern = data["pattern"] if data else self.pattern
        mask = data["mask"] if data else self.mask
        title = data["title"] if data else self.title
        feed_positions = data["feed_positions"] if data else self.feed_positions

        ax: Axes = axes if axes else plt.axes(axes)  # type: ignore
        ax.set_title(title)

        # 底圖：淺冷灰色
        display_img = np.full((pattern.shape[0], pattern.shape[1], 3), [0.96, 0.96, 0.97])
        # 原始金屬區域：中灰色
        display_img[pattern == 1] = [0.74, 0.76, 0.78]
        # 有效連通區域：綠色
        if mask is not None:
            display_img[mask == 1] = [0.1, 0.7, 0.1]

        ax.imshow(display_img, interpolation="nearest")

        # 標註饋電點
        for feed_pos in feed_positions:
            ax.plot(feed_pos[1], feed_pos[0], "ro", markersize=8, markeredgecolor="yellow")

        ax.axis("off")
        if show:
            plt.show()
        return ax

    def plot_records(self, cols: int = 4, show: bool = True):
        record_n = len(self.record)
        with Figure("FeedReachability", ncols=(record_n, cols), show=show) as fig:
            for r_feed in self.record:
                ax = fig.index(-1)
                self.plot(ax, show=False, data=r_feed)

    def plot_records_rate(self, axes=None, show: bool = False):
        plt.rcParams.update({"font.size": 16})
        ax: plt.Axes = axes if axes else plt.axes(axes)  # type: ignore

        for key, rate in self.r_feed_dict.items():
            ax.plot(rate, label=f"{key} (Avg. = {np.mean(rate):.2%})")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("$R_{{feed}}$")
        ax.set_ylim(0, 1)
        plt.legend()
        if show:
            plt.show()
        return ax

    def plot_one_record_rate(self, axes=None, show: bool = False):
        plt.rcParams.update({"font.size": 16})
        ax: plt.Axes = axes if axes else plt.axes(axes)  # type: ignore
        ax.set_title(f"Feed Reachability (Avg. = {self.r_feed_avg:.2%})")

        ax.plot(self.rate_list)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("$R_{{feed}}$ (%)")
        ax.set_ylim(0, 100)
        if show:
            plt.show()
        return ax
