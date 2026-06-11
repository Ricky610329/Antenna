###* ============================================================================
###* antenna/losses.py
###* ----------------------------------------------------------------------------
###* 本檔集中「反向設計閉迴路」所需的可微分損失(loss)、對稱性工具與排程器。
###* 角色定位 (對應專案 pipeline)：
###*   GEN(SigmoidGEN)：目標響應 → 25x25 二元 pattern（STE 可微分二值化）
###*   SM (OldSM/HFSSNet)：pattern → 預測響應，是 HFSS 的可微分替身
###*   SIM(Single/DualPortSimulator)：以 COM 驅動 Ansys HFSS 取得真實響應
###* 閉迴路：GEN 生 pattern → HFSS 得真實 loss → 線上訓練 SM → loss 經 SM 反傳
###*         更新 GEN → 套用 pattern 正則化 → early-stop rollback。
###*
###* 為何需要本檔的正則化 / 排程器？
###*   單純讓 SM-loss 變小，往往會產生「破碎、不連通、無法製造」的金屬圖樣
###*   （像隨機點陣的孤島）。本檔的 loss 把「可製造性 / 連通性」這類物理與工程
###*   約束注入梯度，逼 GEN 收斂到實際做得出來、且電流能流通的天線結構：
###*     - total_variation_loss      ：抑制過度破碎（鼓勵大塊連續區域）
###*     - SpectralConnectivityLoss  ：用圖拉普拉斯 Fiedler 值衡量整體金屬連通性
###*     - GapClosingLoss            ：形態學閉運算填補細小裂縫
###*     - FeedReachability(R_feed)  ：檢核饋電點是否落在同一連通金屬塊上
###*   AdaptiveCyclicalScheduler 則同步調整 GEN 的學習率 lr 與二值化溫度 tau，
###*   在停滯時強制重啟，避免卡在破碎的局部極小值。
###* ----------------------------------------------------------------------------
###* tau 與 STE 的關係（全檔關鍵概念，後續多處引用）：
###*   GEN 的二值化是「sigmoid(logits / tau) → 前向取硬門檻 0/1，反向用軟性梯度」
###*   的 Straight-Through Estimator(STE)。
###*     - tau 大 → sigmoid 平緩，輸出接近 0.5 的灰階，梯度順暢但 pattern 模糊
###*               （適合暖身期廣域探索）。
###*     - tau 小 → sigmoid 陡峭，輸出趨近乾淨的 0/1，pattern 銳利可製造，但梯度
###*               稀疏易卡住（適合退火後期收斂定形）。
###*   因此排程器「lr 與 tau 同步退火」：高溫探索 → 低溫定形，與閉迴路的
###*   early-stop rollback 一起把 GEN 推向可製造解。
###* ============================================================================
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from typing import Union
from antenna.types import Axes, FeedReachabilityDictType
from antenna.utils.figure import Figure
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from loguru import logger


def total_variation_loss(img, weight=0.01):
    """計算 Total Variation Loss 以抑制過度破碎的圖樣"""
    #* 為何重要：TV loss 懲罰相鄰像素的差異，鼓勵大塊「連續同值」區域、抑制
    #* 棋盤格般的高頻雜訊。對天線而言，破碎像素＝無法蝕刻製造、且電氣行為不可控；
    #* TV 把 GEN 推向平滑連續的金屬塊，是「可製造性」最基本的正則化。
    #! 與 STE 的互動：在較大 tau(輸出偏灰階)時 TV 梯度最有意義；tau 太小、pattern
    #! 已硬二值化後，相鄰差只剩 0/1，TV 主要懲罰邊界周長(鼓勵更圓潤、更少邊界)。
    from .utils.data import size_converter
    from . import AntennaPattern
    img = size_converter(AntennaPattern, img, output_shape="B, 1, H, W")  #* 統一成 (B,1,H,W) 才能做維度差分
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:,:,1:,:] - img[:,:,:-1,:], 2).sum()   #* 垂直方向相鄰列差平方和(高頻能量)
    tv_w = torch.pow(img[:,:,:,1:] - img[:,:,:,:-1], 2).sum()   #* 水平方向相鄰行差平方和
    return weight * (tv_h + tv_w) / (bs_img * c_img * h_img * w_img)   #* 除以元素數做正規化，使量級不隨解析度漂移



class SpectralConnectivityLoss(nn.Module):
    """以圖拉普拉斯(graph Laplacian)的代數連通度衡量金屬連通性的可微分 loss。

    原理(譜圖理論)：把 25x25 pattern 視為網格圖，金屬像素間的邊權重高、介質間低。
    拉普拉斯矩陣 L=D-A 的第二小特徵值 λ2(Fiedler value)是「代數連通度」：
        λ2 ≈ 0 → 圖近乎斷裂(存在孤島)；λ2 越大 → 整體越連通、越難被切開。
    天線需電流能在整塊金屬上流動，故希望 λ2 大；本 loss 取 1/λ2，最小化它即等於
    鼓勵 λ2 增大、消除孤島。相較只看局部鄰接的 TV/GapClosing，這是「全域連通」約束。
    """
    #! 計算成本高：對 num_nodes×num_nodes(625×625)做 eigvalsh，且 forward 內逐 batch
    #! for 迴圈跑特徵分解，屬重型正則化，通常以較小權重、較低頻率使用。
    def __init__(self, height=25, width=25, epsilon=1e-2):
        super().__init__()
        self.H, self.W = height, width
        self.num_nodes = height * width
        self.epsilon = epsilon  # 基礎連通性，防止 lambda_2 鎖死在 0
        #? epsilon 的妙用：給每條邊一個微小底權重，使圖永遠是「連通」的(λ2>0)，
        #? 避免 λ2≡0 造成 1/λ2 爆炸或梯度消失；金屬-金屬邊則在此底值上再加 1.0。

        # 預建鄰接索引
        #* 只連「右」與「下」鄰居即可涵蓋所有 4-鄰接無向邊(避免重複建邊)，下方 forward
        #* 再對稱填入 A[src,dst] 與 A[dst,src]。索引預建並註冊為 buffer，隨模型搬到對的 device。
        src, dst = [], []
        for r in range(height):
            for c in range(width):
                idx = r * width + c
                if r + 1 < height: src.append(idx); dst.append((r + 1) * width + c)   #* 與下方鄰居連邊
                if c + 1 < width: src.append(idx); dst.append(r * width + (c + 1))    #* 與右方鄰居連邊
        self.register_buffer('src', torch.tensor(src).long())
        self.register_buffer('dst', torch.tensor(dst).long())

    def forward(self, antenna_map):
        # 確保輸入是 (B, 1, H, W)
        if antenna_map.dim() == 2: antenna_map = antenna_map.unsqueeze(0).unsqueeze(0)
        batch_size = antenna_map.shape[0]
        flat_map = antenna_map.view(batch_size, -1)   #* 攤平成節點向量，索引對應 r*W+c
        losses = []

        for b in range(batch_size):
            # 改進權重邏輯：金屬與金屬=1.0, 其他部分至少有 epsilon 的連通性
            node_vals = flat_map[b]
            # 邊權重 = 節點相乘的加權 + 基礎連通性
            #? 乘積讓「兩端皆金屬(≈1*1)」的邊權重高、「一端為介質(≈0)」的邊權重趨近 epsilon。
            #? 因 pattern 此時多為連續(可微分)值，梯度可經此乘積回傳到 GEN，是可微分的關鍵。
            w = (node_vals[self.src] * node_vals[self.dst]) + self.epsilon

            # 建立對稱拉普拉斯
            A = torch.zeros(self.num_nodes, self.num_nodes, device=antenna_map.device)
            A[self.src, self.dst] = w   #* 無向圖 → 對稱填入兩個方向
            A[self.dst, self.src] = w
            D = torch.diag(torch.sum(A, dim=1))   #* 度矩陣(對角=各節點邊權重和)
            L = D - A                             #* 拉普拉斯 L=D-A，半正定，特徵值升序

            # 計算特徵值
            eigvals = torch.linalg.eigvalsh(L)   #* 對稱矩陣用 eigvalsh：較快且數值穩定
            lambda_2 = eigvals[1]  # Fiedler Value
            #? λ1 必為 0(常數向量)，λ2 才反映連通強度，故取索引 1。

            # 目標：讓 lambda_2 越大越好（金屬連通後 lambda_2 會顯著增加）
            # 使用負對數或反比，讓差異更明顯
            losses.append(1/(lambda_2))   #* 反比：λ2 小(快斷裂)→loss 大，強力推連通
            # losses.append(-torch.log(lambda_2+1e6))
            # losses.append(torch.exp(-lambda_2))   #? 上兩行為替代懲罰式(已停用)，曲線形狀不同

        return torch.mean(torch.stack(losses))   #* 對 batch 取平均

class GapClosingLoss(nn.Module):
    """形態學「閉運算(Closing)」的可微分版，用來懲罰金屬中的細小裂縫/孔洞。

    閉運算 = 先膨脹(Dilation)再侵蝕(Erosion)：膨脹會把細縫填滿、侵蝕再縮回原本外形，
    淨效果是「補洞但幾乎不改變大塊外輪廓」。若 pattern 本來就沒裂縫，closed≈原圖、
    loss≈0；有裂縫時 closed 與原圖差異大、loss 上升，逼 GEN 自己把縫補起來。
    與 SpectralConnectivityLoss 互補：此 loss 處理「局部、小尺度」斷點，連通 loss 顧
    「全域」連通；兩者都服務於「電流可流通且可製造」的最終目標。
    """
    def __init__(self):
        """
        Closing = Dilation(膨脹) + Erosion(侵蝕)
        """
        super().__init__()

    def forward(self, antenna_map):
        # R = (kernel_size-1)/2   #* kernel=3 → 影響半徑 R=1，只補 1 像素級的細縫

        # 1. Soft Dilation (膨脹) - 填補裂縫
        # 使用 MaxPool 模擬
        #? MaxPool 取鄰域最大值＝形態學膨脹；對連續值是「可微分」的軟膨脹(梯度走最大值處)。
        dilated = F.max_pool2d(antenna_map, kernel_size=3, stride=1, padding=1)

        # 2. Soft Erosion (腐蝕) - 恢復外形
        # Erosion(x) = -Max(-x)
        #? 對偶關係：取負→MaxPool(=膨脹)→再取負，即等價於鄰域最小值＝侵蝕。
        closed = -F.max_pool2d(-dilated, kernel_size=3, stride=1, padding=1)

        # 3. 計算 Loss
        # 如果 antenna_map 有裂縫，closed 會把裂縫填滿 (數值變大)
        # 我們希望 antenna_map 本身就沒有裂縫，即 antenna_map 應該接近 closed
        # Loss = || Closed - Original ||
        #! 注意：此 loss 只在「裂縫處」非零，故不會強迫填滿大空洞，僅整治細縫，
        #! 不致過度肥大化金屬而破壞天線設計意圖。
        loss = torch.mean((closed - antenna_map) ** 2)
        return loss

class FeedReachability: # R_feed
    """饋電連通度指標 R_feed：評估「所有饋電點是否落在同一塊連通金屬上」。

    物理意義：天線必須由饋電點(feed)把訊號送進金屬輻射體；若某饋電點下沒有金屬、
    或多個饋電點各自連到不同的金屬孤島，天線就無法正常工作。R_feed 因此是「pattern
    在工程上是否可用」的硬性檢核。
    與本檔其他 loss 的差別：R_feed 走 scipy 連通元件標記(label)，是「不可微分的評估
    指標(metric)」，用於監看/early-stop/視覺化，而非直接回傳梯度的訓練 loss。
    回傳值定義為「共同連通塊像素 / 全部金屬像素」的佔比：全部饋電點同塊時介於 0~1，
    否則為 0。single_feed/dual_feed 為 25x25 預設饋電佈局的便捷建構。
    """
    def __init__(self, feed_positions:list[tuple[int, int]]):
        """
        計算共同連通指標 (Mutual Feed Connectivity Index)
        只有當所有饋電點都連通在同一個金屬塊上時，才計算該塊的佔比。

        :param feed_positions: 座標列表 [(r1, c1), (r2, c2), ...]
        """
        from scipy.ndimage import label
        assert len(feed_positions)>0, ""

        self.feed_positions = feed_positions
        """潰入點"""
        self.rate = None
        """電流導通率"""
        self.mask = 0
        """電流導通的遮罩"""
        # self.structure = np.ones((3, 3))   #* 註解掉的此版才是真 8-連通(含對角)
        #! docstring 與實作不符的陷阱：下方十字結構其實是 4-連通(只連上下左右)，
        #! 與下方字串敘述的「8-連通」相反；要改 8-連通需改回上一行的 np.ones((3,3))。
        self.structure = np.array([
            [0.0,1.0,0.0],
            [1.0,1.0,1.0],
            [0.0,1.0,0.0]])
        """連通性, 預設採用 8-連通 (8-connectivity), 若要4連通可以是十字架"""
        self.record:list[FeedReachabilityDictType] = []   #* 累積各次評估結果，供逐 epoch 繪圖
        self.r_feed_str = "$R_{{feed}}$"   #* matplotlib LaTeX 標籤用字串

        self._label = label   #* 綁定 scipy.ndimage.label(連通元件標記)以利重複呼叫
    
    @classmethod
    def single_feed(cls):
        """單埠佈局：饋電點在底邊中央 (對應 SinglePortSimulator)。"""
        shape = (25, 25)
        return cls([(shape[0]-1, int((shape[1])/2))])   #* (最底列, 中間行)

    @classmethod
    def dual_feed(cls):
        """雙埠佈局：饋電點在底邊中央與頂邊中央 (對應 DualPortSimulator)。"""
        shape = (25, 25)
        return cls([(shape[0]-1, int((shape[1])/2)),(0, int((shape[1])/2))])   #* 底邊中央 + 頂邊中央
    
    def __call__(self, pattern: Union[Tensor, np.ndarray], *, record:bool=False, title:str = "Pattern ($R_{{feed}}$={rate:.2%})"):
        """
        :param pattern: 2D array (1=金屬, 0=介質)
        :return: Feed Reachability Rate
        """
        
        if isinstance(pattern, Tensor):
            pattern = pattern.numpy()   #* scipy 只吃 ndarray；此處等同切斷梯度(本類為評估指標)

        #* label：把 pattern 中相連的金屬像素標成同一整數 ID，0 代表背景(介質)
        labeled_array, _ = self._label(pattern, structure=self.structure)

        # 1. 取得所有饋電點所在的 Label IDs
        feed_labels = []

        for pos in self.feed_positions:

            # 檢查座標是否越界或該處無金屬
            if 0 <= pos[0] < pattern.shape[0] and 0 <= pos[1] < pattern.shape[1]:
                lbl = labeled_array[pos]   #* 取該饋電點所屬的連通塊 ID
                if lbl > 0:
                    feed_labels.append(lbl)
                else:
                    #! 任一饋電點下方無金屬(lbl==0)即整體失敗：天線根本無法被饋電
                    logger.warning("其中一個潰入點沒金屬，直接失敗")
                    return 0.0, np.zeros_like(pattern) # 其中一個點沒金屬，直接失敗
            else:
                logger.error("潰入點座標是否越界")
                return 0.0, np.zeros_like(pattern)

        # 2. 「AND」邏輯檢查：判斷所有饋電點的 Label 是否完全相同
        #?   只有所有饋電點的塊 ID 相同(set 長度為 1)才算「共同連通」；只要有人在
        #?   不同塊上就算失敗，因為各埠的電流路徑沒有真正接在一起。
        unique_labels = set(feed_labels)

        if len(unique_labels) == 1:
            # 所有饋電點都在同一個連通塊上
            shared_label = list(unique_labels)[0]
            shared_mask = (labeled_array == shared_label)   #* 此共同塊的布林遮罩

            total_metal_pixels = np.sum(pattern)            #* 全部金屬像素
            connected_pixels = np.sum(shared_mask)          #* 落在共同塊上的金屬像素
            #* 佔比越高代表越多金屬實際參與饋電路徑、孤島越少(理想接近 1.0)
            mutual_index = connected_pixels / total_metal_pixels

        else:
            # 饋電點分布在不同的連通塊上，或彼此斷開
            mutual_index = 0.0
            shared_mask = np.zeros_like(pattern)
        
        self.rate = mutual_index
        self.mask = shared_mask
        self.pattern = pattern
        self.title = title.format(rate=mutual_index)   #* 把 rate 填入標題模板，供繪圖顯示

        if record:   #* 需逐 epoch 追蹤趨勢時才存檔，避免無謂記憶體累積
            self.record.append(
                {
                    'pattern': pattern,
                    'feed_positions': self.feed_positions,
                    'rate': mutual_index,
                    'mask': shared_mask,
                    "title": self.title
                }
            ) 
        return mutual_index
    
    @property
    def r_feed_dict(self):
        #* 依 title 分組的 rate 序列：同一標題(同一條曲線)的歷次 rate 彙整成 list，
        #* 方便在同一張圖上畫多條(例如不同饋電佈局)。
        result = defaultdict(list)
        for entry in self.record:
            result[entry['title']].append(entry['rate'])

        return result

    @property
    def r_feed_list(self):
        return [_['rate'] for _ in self.record]   #* 所有歷史 rate(0~1)

    @property
    def rate_list(self):
        return [_['rate']*100 for _ in self.record]   #* 同上但換算成百分比(0~100)

    @property
    def r_feed_avg(self):
        return np.mean(self.r_feed_list)   #* 整段訓練的平均 R_feed(單一總結指標)
    
    def plot(self, axes = None, show=False, data:FeedReachabilityDictType=None):
        """視覺化單筆結果：底圖灰、金屬中灰、共同連通塊綠、饋電點紅(黃框)。"""
        #* 傳入 data(來自 record 的某筆)即可重畫歷史結果；不傳則畫最近一次 __call__ 的狀態。
        pattern = self.pattern

        pattern = data['pattern'] if data else self.pattern
        mask = data['mask'] if data else self.mask
        rate = data['rate'] if data else self.rate
        title = data['title'] if data else self.title
        feed_positions = data['feed_positions'] if data else self.feed_positions

        ax:Axes = axes if axes else plt.axes(axes) # type: ignore
        ax.set_title(title)

        #* 初始化底圖
        display_img = np.full((pattern.shape[0], pattern.shape[1], 3), [0.96, 0.96, 0.97]) # 淺冷灰色
        
        #* 標示所有原始金屬區域
        display_img[pattern == 1] = [0.74, 0.76, 0.78] # 中灰色
        
        #* 疊加有效連通區域
        display_img[mask == 1] = [0.1, 0.7, 0.1]
        
        #* 繪製影像
        ax.imshow(display_img, interpolation='nearest')
        
        #* 標註饋電點
        for feed_pos in feed_positions:
            ax.plot(feed_pos[1], feed_pos[0], 'ro', markersize=8, markeredgecolor='yellow')

        ax.axis('off') # on/off
        # plt.grid(False)
        if show: plt.show()
        return ax

    def plot_records(self, cols: int = 4, show: bool = True):
        record_n = len(self.record)
        with Figure("FeedReachability", ncols=(record_n,cols), show=show) as fig:
            for n, r_feed in enumerate(self.record):
                ax = fig.index(-1)
                self.plot(ax, show=False, data=r_feed)

    def plot_records_rate(self, axes = None, show=False):

        plt.rcParams.update({
            'font.size': 16,
        })

        ax:plt.Axes = axes if axes else plt.axes(axes) # type: ignore
        # ax.set_title(f'Feed Reachability (Avg. = {self.r_feed_avg:.2%})')

        for key, rate in self.r_feed_dict.items():
            ax.plot(rate, label=f"{key} (Avg. = {np.mean(rate):.2%})")

        ax.set_xlabel('Epoch')  # x 軸名稱
        ax.set_ylabel('$R_{{feed}}$')  # y 軸名稱
        ax.set_ylim(0, 1)

        # plt.grid(False)
        plt.legend()
        if show: plt.show()
        return ax

    def plot_one_record_rate(self, axes = None, show=False):

        plt.rcParams.update({
            'font.size': 16,
        })

        ax:plt.Axes = axes if axes else plt.axes(axes) # type: ignore
        ax.set_title(f'Feed Reachability (Avg. = {self.r_feed_avg:.2%})')

        ax.plot(self.rate_list)

        ax.set_xlabel('Epoch')  # x 軸名稱
        ax.set_ylabel('$R_{{feed}}$ (%)')  # y 軸名稱
        ax.set_ylim(0, 100)

        # plt.grid(False)
        if show: plt.show()
        return ax