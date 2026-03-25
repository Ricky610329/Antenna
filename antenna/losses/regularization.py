from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch import Tensor, nn

from antenna.types import *
from antenna.utils.utils import Figure, plt


def total_variation_loss(img, weight=0.01):
    """計算 Total Variation Loss 以抑制過度破碎的圖樣"""
    from antenna import AntennaPattern
    from antenna.utils.data import size_converter

    img = size_converter(AntennaPattern, img, output_shape="B, 1, H, W")
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    return weight * (tv_h + tv_w) / (bs_img * c_img * h_img * w_img)


class SpectralConnectivityLoss(nn.Module):
    def __init__(self, height=25, width=25):
        """
        頻譜圖論法: 將 Pattern 看作一張圖 (Graph)

        - 節點 (Nodes)：每個像素是一個節點。
        - 邊 (Edges)：如果兩個像素相鄰，且都有金屬 (數值高)，則邊的權重很大；如果其中一個沒金屬，邊的權重接近 0。

        目標：最大化拉普拉斯矩陣 (Laplacian Matrix) 的第二小特徵值 ($\\lambda_2$)，又稱為 Fiedler Value。$\\lambda_2 > 0$：整張圖的金屬部分是連通的。$\\lambda_2 \approx 0$：圖斷成了兩塊或更多塊。
        """
        super().__init__()

        self.H = height
        self.W = width
        self.num_nodes = height * width

        # 預先建立鄰接關係索引 (Adjacency Indices)
        # 這裡建立 4-連通 (上下左右) 的關係
        self.adj_indices = []
        for r in range(height):
            for c in range(width):
                idx = r * width + c
                # 下
                if r + 1 < height:
                    self.adj_indices.append((idx, (r + 1) * width + c))
                # 右
                if c + 1 < width:
                    self.adj_indices.append((idx, r * width + (c + 1)))

        self.src = torch.tensor([x[0] for x in self.adj_indices]).long()
        self.dst = torch.tensor([x[1] for x in self.adj_indices]).long()

    def forward(self, antenna_map):
        """
        antenna_map: (Batch, 1, H, W) 數值 0~1
        """
        batch_size = antenna_map.shape[0]
        # 攤平成 (Batch, N)
        flat_map = antenna_map.view(batch_size, -1)

        device = antenna_map.device
        self.src = self.src.to(device)
        self.dst = self.dst.to(device)

        losses = []

        for b in range(batch_size):
            # 1. 建構邊的權重 (Edge Weights)
            # 邊的權重 = 兩個連接像素值的幾何平均 (或乘積)
            # 只有當兩個像素都是 1 時，邊才是 1
            node_vals = flat_map[b] + 1e-4  # 加小數值避免全 0
            weights = node_vals[self.src] * node_vals[self.dst]

            # 2. 建構拉普拉斯矩陣 L = D - A
            # A: 鄰接矩陣 (Adjacency Matrix)
            A = torch.zeros(self.num_nodes, self.num_nodes, device=device)
            A[self.src, self.dst] = weights
            A[self.dst, self.src] = weights  # 對稱

            # D: 度矩陣 (Degree Matrix) - 對角線是 A 的列總和
            degree = torch.sum(A, dim=1)
            D = torch.diag(degree)

            L = D - A

            # 3. 計算特徵值 (Eigenvalues)
            # 因為 L 是實對稱矩陣，使用 symeig 或 linalg.eigh
            eigvals = torch.linalg.eigvalsh(L)

            # 4. 取第二小特徵值 (Fiedler Value)
            # eigvals[0] 理論上是 0 (對應全 1 向量)，我們關注 eigvals[1]
            lambda_2 = eigvals[1]

            # Loss 設計：我們希望 lambda_2 越大越好 (越連通)
            # 如果 lambda_2 已經夠大 (例如 > 0.1)，Loss 就可以是 0
            losses.append(torch.relu(0.5 - lambda_2))

        return torch.mean(torch.stack(losses))


class GapClosingLoss(nn.Module):
    def __init__(self):
        """
        Closing = Dilation(膨脹) + Erosion(侵蝕)
        """
        super().__init__()

    def forward(self, antenna_map):
        # 1. Soft Dilation (膨脹) - 填補裂縫
        # 使用 MaxPool 模擬
        dilated = F.max_pool2d(antenna_map, kernel_size=3, stride=1, padding=1)

        # 2. Soft Erosion (腐蝕) - 恢復外形
        # Erosion(x) = -Max(-x)
        closed = -F.max_pool2d(-dilated, kernel_size=3, stride=1, padding=1)

        # 3. 計算 Loss
        # 如果 antenna_map 有裂縫，closed 會把裂縫填滿 (數值變大)
        # 我們希望 antenna_map 本身就沒有裂縫，即 antenna_map 應該接近 closed
        # Loss = || Closed - Original ||

        loss = torch.mean((closed - antenna_map) ** 2)
        return loss


class FeedReachability:  # R_feed
    def __init__(self, feed_positions: list[tuple[int, int]]):
        """
        計算共同連通指標 (Mutual Feed Connectivity Index)
        只有當所有饋電點都連通在同一個金屬塊上時，才計算該塊的佔比。

        :param feed_positions: 座標列表 [(r1, c1), (r2, c2), ...]
        """
        from scipy.ndimage import label

        assert len(feed_positions) > 0, ""

        self.feed_positions = feed_positions
        """潰入點"""
        self.rate = None
        """電流導通率"""
        self.mask = 0
        """電流導通的遮罩"""
        # self.structure = np.ones((3, 3))
        self.structure = np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]])
        """連通性, 預設採用 8-連通 (8-connectivity), 若要4連通可以是十字架"""
        self.record: list[FeedReachabilityDictType] = []
        self.r_feed_str = "$R_{{feed}}$"

        self._label = label

    @classmethod
    def single_feed(cls):
        shape = (25, 25)
        return cls([(shape[0] - 1, int((shape[1]) / 2))])

    @classmethod
    def dual_feed(cls):
        shape = (25, 25)
        return cls([(shape[0] - 1, int((shape[1]) / 2)), (0, int((shape[1]) / 2))])

    def __call__(
        self,
        pattern: Union[Tensor, np.ndarray],
        *,
        record: bool = False,
        title: str = "Pattern ($R_{{feed}}$={rate:.2%})",
    ):
        """
        :param pattern: 2D array (1=金屬, 0=介質)
        :return: Feed Reachability Rate
        """

        if isinstance(pattern, Tensor):
            pattern = pattern.numpy()

        labeled_array, _ = self._label(pattern, structure=self.structure)

        # 1. 取得所有饋電點所在的 Label IDs
        feed_labels = []

        for pos in self.feed_positions:
            # 檢查座標是否越界或該處無金屬
            if 0 <= pos[0] < pattern.shape[0] and 0 <= pos[1] < pattern.shape[1]:
                lbl = labeled_array[pos]
                if lbl > 0:
                    feed_labels.append(lbl)
                else:
                    logger.warning("其中一個潰入點沒金屬，直接失敗")
                    return 0.0, np.zeros_like(pattern)  # 其中一個點沒金屬，直接失敗
            else:
                logger.error("潰入點座標是否越界")
                return 0.0, np.zeros_like(pattern)

        # 2. 「AND」邏輯檢查：判斷所有饋電點的 Label 是否完全相同
        unique_labels = set(feed_labels)

        if len(unique_labels) == 1:
            # 所有饋電點都在同一個連通塊上
            shared_label = list(unique_labels)[0]
            shared_mask = labeled_array == shared_label

            total_metal_pixels = np.sum(pattern)
            connected_pixels = np.sum(shared_mask)
            mutual_index = connected_pixels / total_metal_pixels

        else:
            # 饋電點分布在不同的連通塊上，或彼此斷開
            mutual_index = 0.0
            shared_mask = np.zeros_like(pattern)

        self.rate = mutual_index
        self.mask = shared_mask
        self.pattern = pattern
        self.title = title.format(rate=mutual_index)

        if record:
            self.record.append(
                {
                    "pattern": pattern,
                    "feed_positions": self.feed_positions,
                    "rate": mutual_index,
                    "mask": shared_mask,
                    "title": self.title,
                }
            )
        return mutual_index

    @property
    def r_feed_dict(self):
        result = defaultdict(list)
        for entry in self.record:
            result[entry["title"]].append(entry["rate"])

        return result

    @property
    def r_feed_list(self):
        return [_["rate"] for _ in self.record]

    @property
    def rate_list(self):
        return [_["rate"] * 100 for _ in self.record]

    @property
    def r_feed_avg(self):
        return np.mean(self.r_feed_list)

    def plot(self, axes=None, show=False, data: FeedReachabilityDictType = None):
        pattern = self.pattern

        pattern = data["pattern"] if data else self.pattern
        mask = data["mask"] if data else self.mask
        data["rate"] if data else self.rate
        title = data["title"] if data else self.title
        feed_positions = data["feed_positions"] if data else self.feed_positions

        ax: Axes = axes if axes else plt.axes(axes)  # type: ignore
        ax.set_title(title)

        # * 初始化底圖
        display_img = np.full((pattern.shape[0], pattern.shape[1], 3), [0.96, 0.96, 0.97])  # 淺冷灰色

        # * 標示所有原始金屬區域
        display_img[pattern == 1] = [0.74, 0.76, 0.78]  # 中灰色

        # * 疊加有效連通區域
        display_img[mask == 1] = [0.1, 0.7, 0.1]

        # * 繪製影像
        ax.imshow(display_img, interpolation="nearest")

        # * 標註饋電點
        for feed_pos in feed_positions:
            ax.plot(feed_pos[1], feed_pos[0], "ro", markersize=8, markeredgecolor="yellow")

        ax.axis("off")  # on/off
        # plt.grid(False)
        if show:
            plt.show()
        return ax

    def plot_records(self, cols: int = 4, show: bool = True):
        record_n = len(self.record)
        with Figure("FeedReachability", ncols=(record_n, cols), show=show) as fig:
            for n, r_feed in enumerate(self.record):
                ax = fig.index(-1)
                self.plot(ax, show=False, data=r_feed)

    def plot_records_rate(self, axes=None, show=False):

        plt.rcParams.update(
            {
                "font.size": 16,
            }
        )

        ax: plt.Axes = axes if axes else plt.axes(axes)  # type: ignore
        # ax.set_title(f'Feed Reachability (Avg. = {self.r_feed_avg:.2%})')

        for key, rate in self.r_feed_dict.items():
            ax.plot(rate, label=f"{key} (Avg. = {np.mean(rate):.2%})")

        ax.set_xlabel("Epoch")  # x 軸名稱
        ax.set_ylabel("$R_{{feed}}$")  # y 軸名稱
        ax.set_ylim(0, 1)

        # plt.grid(False)
        plt.legend()
        if show:
            plt.show()
        return ax

    def plot_one_record_rate(self, axes=None, show=False):

        plt.rcParams.update(
            {
                "font.size": 16,
            }
        )

        ax: plt.Axes = axes if axes else plt.axes(axes)  # type: ignore
        ax.set_title(f"Feed Reachability (Avg. = {self.r_feed_avg:.2%})")

        ax.plot(self.rate_list)

        ax.set_xlabel("Epoch")  # x 軸名稱
        ax.set_ylabel("$R_{{feed}}$ (%)")  # y 軸名稱
        ax.set_ylim(0, 100)

        # plt.grid(False)
        if show:
            plt.show()
        return ax
