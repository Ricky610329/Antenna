###* ============================================================================
###* antenna/losses.py
###* ----------------------------------------------------------------------------
###* 本檔集中「反向設計閉迴路」所需的可微分損失(loss)、對稱性工具與排程器。
###* 角色定位 (對應專案 pipeline)：
###*   GEN(SigmoidGenerator)：目標響應 → 25x25 二元 pattern（STE 可微分二值化）
###*   SM (MLPSurrogate/HFSSNet)：pattern → 預測響應，是 HFSS 的可微分替身
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
from typing import Literal, Union
from antenna.utils.types import Axes, FeedReachabilityDictType
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
    from .utils.torch_utils import size_converter
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

###* ============================================================================
###* 響應損失 (響應 vs 目標)：單埠 custom_loss_minmax / 雙埠 interval_loss
###* 由 setup_responses 經 spec.register_loss_fn(...) 綁進響應規格。
###* (從 antenna/patch/__init__.py 歸位 —— 它們是損失，不是模擬器。)
###* ============================================================================

def custom_loss_minmax(prediciton:Tensor, target:Tensor, method:Literal['low', 'high'], loss_type='SmoothL1Loss'):
    """
    單邊極值損失 (Min/Max One-sided Loss)：單埠 train_single 的主損失函數。

    設計意圖：
        相對於同時管目標「最高點與最低點」兩端的寫法，本函數「只挑一個極值點」
        並只做單一方向的懲罰，把「達標即可」的不等式規格表達得最乾淨：
          - method='high'：只看「目標最高點」(如 Gain 中央 +4dB)，只罰「預測偏低」
                           (預測 < 目標)；預測更高視為更好，不罰。
          - method='low' ：只看「目標最低點」(如 S11 中央凹陷 -10dB)，只罰「預測偏高」
                           (預測 > 目標)；預測更低 (匹配更好) 不罰。
        故 train_single 中：S11 用 method='low' (反射夠低即可)、Gain 用 method='high'
        (增益夠高即可)。核心精神同為「夠好就不罰、過頭也不罰」,
        只在「未滿足規格」的方向產生梯度。

    :param prediciton: SM 對該條響應的預測值。
    :param target: 目標響應曲線。
    :param method: 'low' (罰目標最低點處預測偏高) 或 'high' (罰目標最高點處預測偏低)。
    :param loss_type: 'SmoothL1Loss' 或 'MSELoss'。
    """
    criterion = nn.SmoothL1Loss() if loss_type=='SmoothL1Loss' else nn.MSELoss()
    loss_zero = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)  # 達標時回傳的零損失 (仍保留梯度需求)

    match method:
        case 'high':
            #* 高點規格：要求「預測 >= 目標最高值」(如增益要夠高)
            target_high = target.max()
            mask_high = target == target_high           # 目標最高點所在的頻點
            mask_b_high = prediciton[mask_high] < target_high  # 其中「預測偏低 (未達標)」者才罰
            # 全部達標 → 0；否則只對未達標頻點計誤差，不懲罰「更高」的預測。
            return loss_zero if mask_b_high.sum() == 0 else criterion(
                prediciton[mask_high][mask_b_high], target[mask_high][mask_b_high]
            )

        case 'low':
            #* 低點規格：要求「預測 <= 目標最低值」(如反射要夠低)
            target_low = target.min()
            mask_low = target == target_low             # 目標最低點所在的頻點
            mask_b_low = prediciton[mask_low] > target_low     # 其中「預測偏高 (未達標)」者才罰
            # 全部達標 → 0；否則只對未達標頻點計誤差，不懲罰「更低」的預測。
            return loss_zero if mask_b_low.sum() == 0 else criterion(
                prediciton[mask_low][mask_b_low], target[mask_low][mask_b_low]
            )

        case _:
            # 防呆：method 僅允許 'low' / 'high'。
            raise ValueError('The method must be `low` or `high`.')

# interval_loss 兩種呼叫模式 (由 lower/upper 的型別在函式內以 isinstance 分派)：
#   (1) 相對模式：lower/upper 為 float 偏移，邊界 = target + 偏移 (需傳 target)。
#   (2) 絕對模式：lower/upper 為 Tensor，直接當成上下界 (不需 target)。
# 設計意圖：天線規格常以「目標 ± 容差」表達 (如 [target-1, target+1])，比 minmax 更柔性 ──
#   允許預測在容差帶內自由浮動而不受罰，只懲罰「超出帶外」的部分。
#! 原兩個 @overload 型別 stub 已移除：核心慣例不用 @overload/TypeVar (見 CLAUDE.md 型別紀律)；
#  兩模式改在下方單一函式 docstring 說清楚 (執行語意不變、byte-identical、golden 安全)。
def interval_loss(
    prediction: Tensor,  lower_response: Union[float, Tensor],  upper_response: Union[float, Tensor], 
    target: Tensor = None,* , loss_type: str = 'SmoothL1Loss', reduction: str = 'mean'
) -> Tensor:
    """
    區間損失 (Interval Loss) 的核心運算函數。
    
    :param prediction: 預測值。
    :param lower_response: 
        - Float: 相對於 Target 的下限偏移 (如 -0.5)。
        - Tensor: 絕對下限值。
    :param upper_response: 
        - Float: 相對於 Target 的上限偏移 (如 0.5)。
        - Tensor: 絕對上限值。
    :param target (Tensor, optional): 真實標籤。若使用 float 模式 (相對偏移) 則為必填。
    :param loss_type: 'SmoothL1Loss' 或 'MSELoss'。
    :param reduction: 'mean' 或 'sum'。
    """
    # 底層距離度量：超出容差帶後，依超出量大小計罰 (SmoothL1 對大偏差較穩健)。
    if loss_type == 'SmoothL1Loss':
        loss_fn = nn.SmoothL1Loss(reduction=reduction)
    elif loss_type == 'MSELoss':
        loss_fn = nn.MSELoss(reduction=reduction)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    if isinstance(lower_response, Tensor) and isinstance(upper_response, Tensor):
        #* 絕對模式：直接以傳入的 Tensor 當上下界，不依賴 target。
        min_bound = lower_response
        max_bound = upper_response

    else:   #* Target + Offset
        #* 相對模式：邊界隨目標曲線逐點平移，形成「目標 ± 容差」的動態容差帶。
        if target is None:
            raise ValueError("使用 Float (相對偏移模式) 時，必須傳入 target。")

        min_bound = target + lower_response   # 逐頻點下界 = 目標 + 下限偏移 (如 target + (-1))
        max_bound = target + upper_response   # 逐頻點上界 = 目標 + 上限偏移 (如 target + (+1))

    #* Universal Clamp Logic
    # 我們將 Prediction 限制在 [min_bound, max_bound] 範圍內，得到一個「參考目標 (Reference Target)」。
    # - 若 Prediction 在範圍內：Ref = Prediction。 Loss = 0。
    # - 若 Prediction 超出範圍：Ref = 邊界值。 Loss = |Pred - 邊界值|。
    # clamp 把 prediction 夾進 [min_bound, max_bound]：帶內者夾後等於自己 (loss=0)，
    # 帶外者夾到最近邊界；.detach() 讓此「參考目標」不帶梯度、被當成常數。
    target_clamped = torch.clamp(prediction, min=min_bound, max=max_bound).detach() # 確保參考目標被視為常數，讓梯度正確指向 Prediction
    # 以「prediction vs 夾後參考目標」算損失：等價於只懲罰超出容差帶的部分，
    # 梯度方向把帶外預測往最近邊界拉、帶內預測則完全自由 (零梯度)。
    loss = loss_fn(prediction, target_clamped)

    return loss   # 帶內 → 0；帶外 → 與最近邊界的距離


###* ============================================================================
###* 方向圖損失 (radiation pattern)：beam_coverage_loss
###* ----------------------------------------------------------------------------
###* 角色：把「方向圖 gain vs 角度」(固定 28GHz、phi 0°/90°) 塑成相對 boresight 的
###*       「平頂 + 中央峰」形狀。吃 SM rad head 的預測 rad_pred，可微，供 GEN 反傳。
###* 與 S11/Gain 主路徑解耦：方向圖走另一條 x 軸(角度)、另一顆 rad head，預設 off。
###*
###* 分工(關鍵)：
###*   - 「boresight 絕對增益要高」── 由既有 Gain target(method='high') 負責，本函式不碰。
###*   - 「角度上要平、窗內不掉超過 floor_db、0° 最高」── 本函式負責(純塑形)。
###* 故本函式刻意「相對」：一切錨在預測的 G0=rad_pred[θ≈0]，不寫絕對 dB target
###* (方向圖因此不需要在 targets: 寫梯形曲線)。自我歸一化，不受天線絕對增益高低影響。
###* ============================================================================

def beam_coverage_loss(
    rad_pred: Tensor,
    theta: Tensor,
    *,
    window_deg: float = 55.0,
    floor_db: float = 3.0,
    boresight_weight: float = 1.0,
    flatness_weight: float = 0.0,
    reduction: str = "mean",
) -> Tensor:
    """
    方向圖覆蓋損失 (相對 boresight 的「平頂 + 中央峰」形狀)。

    由兩個單邊(relu)項 + 一個選用平坦項組成，全部相對預測的 boresight 增益 G0=rad_pred[θ≈0]：
      ① floor 項   ：逼窗內(|θ|≤window_deg)每個角度 gain ≥ G0 − floor_db。
                     低於才罰；高於不罰 ──「越高越好」(比照 custom_loss_minmax 的單邊精神)。
      ② boresight 項：罰窗內任何角度 gain 超過 G0，逼 0° 成為窗內最高點。
      ③ flatness 項 (選用，flatness_weight>0 才作用)：罰窗內每個角度「對 G0 的偏差平方」，
                     對任何偏離都給梯度 → 主動把波形壓平到 boresight 準位 (順帶拉高窗內)。
                     與 ① 的差別：① 是 hinge(只罰掉超過 floor 的、帶內 0 梯度＝容忍 ripple)；
                     ③ 連帶內小起伏也壓。flatness_weight=0 → 此項完全不影響 (golden 安全)。
    總損失 = floor_loss + boresight_weight · boresight_loss + flatness_weight · flatness_loss。
    (整體權重 w_rad 由訓練端在加進 GEN loss 時再乘，不在此函式內。)

    :param rad_pred: SM rad head 的預測，dB gain。shape (n_phi, n_theta) 或 (n_theta,)。
    :param theta:    角度取樣點 (度)，shape (n_theta,)。boresight 取 |θ| 最小的那點。
    :param window_deg: 主波束覆蓋窗的半角 (度)。學長規格 55。
    :param floor_db:   窗內允許比 boresight 低的量 (dB)。學長規格 3。
    :param boresight_weight: ② 相對 ① 的權重。
    :param reduction:  'mean' (對窗內元素平均) 或 'sum'。
    :return: 純量 loss (可微；梯度會流過 rad_pred，含 G0)。
    """
    if reduction not in ("mean", "sum"):
        raise ValueError(f"reduction 必須是 'mean' 或 'sum'，但得到 {reduction!r}")

    if rad_pred.dim() == 1:
        rad_pred = rad_pred.unsqueeze(0)            # (n_theta,) → (1, n_theta) 單一切面
    if rad_pred.dim() != 2:
        raise ValueError(f"rad_pred 需為 (n_phi, n_theta) 或 (n_theta,)，但得到 shape {tuple(rad_pred.shape)}")

    n_theta = rad_pred.shape[1]
    theta = theta.reshape(-1).to(device=rad_pred.device, dtype=rad_pred.dtype)
    if theta.numel() != n_theta:
        raise ValueError(f"theta 長度 {theta.numel()} 與 rad_pred 的 n_theta {n_theta} 不符")

    #? 錨點：boresight = |θ| 最小的取樣點 (每個 phi 切面共用同一個 θ≈0 的欄)。
    #! argmin/abs 作用在固定的 θ 網格上，不在梯度路徑；G0 是 rad_pred 的切片，可微。
    bore_idx = int(theta.abs().argmin())
    g0 = rad_pred[:, bore_idx:bore_idx + 1]         # (n_phi, 1)，保留維度供逐角度廣播

    #? 窗：只在 |θ| ≤ window_deg 的角度上算 loss (主波束覆蓋區)。
    in_window = theta.abs() <= window_deg           # (n_theta,) bool
    if not bool(in_window.any()):
        raise ValueError(
            f"window_deg={window_deg} 內沒有任何 θ 取樣點 (θ 範圍 [{float(theta.min())}, {float(theta.max())}])"
        )
    pred_w = rad_pred[:, in_window]                 # (n_phi, n_window)

    #? ① floor：低於 G0−floor_db 才罰 (單邊)；窗內等於/高於 floor → 0。
    floor_deficit = torch.relu((g0 - floor_db) - pred_w)
    #? ② boresight-max：超過 G0 才罰 (單邊)；逼 0° 最高。
    boresight_excess = torch.relu(pred_w - g0)

    #? ③ flatness (選用)：對 G0 的偏差平方 (雙邊、處處有梯度)，主動把窗內壓平到 boresight 準位。
    #! flatness_weight=0 時整項短路、根本不算 (pred_w-g0)^2：
    #    - 對有限輸入逐位元同原樣 (loss + 0 == loss)，golden 安全；
    #    - 且避免 rad_pred 含 inf 時 0*inf=NaN 把 loss 從 inf 污染成更難 debug 的 NaN (防 NaN 退化)。
    #  注意 ③ 是平方項，量級與 ①② (線性 relu) 不同 → flatness_weight 的語意非線性、隨 ripple 振幅放大。
    if reduction == "mean":
        loss = floor_deficit.mean() + boresight_weight * boresight_excess.mean()
        if flatness_weight:
            loss = loss + flatness_weight * ((pred_w - g0) ** 2).mean()
        return loss
    loss = floor_deficit.sum() + boresight_weight * boresight_excess.sum()
    if flatness_weight:
        loss = loss + flatness_weight * ((pred_w - g0) ** 2).sum()
    return loss


def boundary_loss(pattern: Tensor, seen: Tensor) -> Tensor:
    """
    邊界損失 (trust region)：懲罰「生成 pattern 偏離 SM 已見訓練分布」的距離。

    動機 (Neural Adjoint 文獻最大改進)：G 靠「下降凍住的 SM 預測 loss」優化 pattern；SM 只在
    見過資料的地方準。沒有此項，G 會鑽到 SM 從沒見過的區域 (SM 外推、瞎掰好分數)，
    產生「SM 說好、HFSS 說爛」的設計 → 每個這種設計都白燒一次昂貴 HFSS 評估。此項把 G 拉回
    SM 真的懂的鄰域 → 少浪費評估 (＝加速)。NA 原版是連續參數的 box；我們是二元 pattern，
    改用「與最近一個已見 pattern 的距離」當邊界訊號。

    :param pattern: 當前生成的 pattern (攤平，帶梯度)，shape (N,)。
    :param seen:    SM 已見的 pattern (攤平、detach)，shape (M, N)。通常＝replay 緩衝。
    :return: 純量 loss＝與「最近一個『不同的』已見 pattern」的均方距離 (可微，梯度拉 pattern 回鄰域)。
    """
    if seen is None or seen.numel() == 0:
        return pattern.new_zeros(())
    seen = seen.to(device=pattern.device, dtype=pattern.dtype)
    d = ((pattern.unsqueeze(0) - seen) ** 2).mean(dim=1)   # (M,) 與每個已見 pattern 的 MSE
    d = d[d > 1e-9]                                         # 排除與自己相同的那筆 (距離 0；current 可能已在緩衝)
    if d.numel() == 0:
        return pattern.new_zeros(())
    return d.min()                                         # 與最近一個已見的距離


def candidate_repulsion(logits: Tensor) -> Tensor:
    """候選排斥 (有界 RBF)：罰一批 (K, D) logits 兩兩太像 → 逼生成器用 latent、防 batch_latent 候選崩塌。

    動機：batch_latent 同批 K 候選會塌縮 (MLP 學會忽略 z → score_spread→0、best-of-K 失效)。
    在聚合 loss 加此項，逼「不同 z 解出不同 pattern」。

    :param logits: 一批候選的連續 logits (二值化前)，shape (K, D)，帶梯度。
    :return: 平均 off-diagonal RBF 相似度 ∈ (0,1] (高=塌縮)；最小化它 → 候選分散。
             有界 (不像「最大化距離」會炸 logits)；h=兩兩平方距離中位數 (detach、SVGD median heuristic)，
             作用在連續 logits → 梯度平滑。K<2 → 0。
    ⚠ 已知療效取捨 (非 bug)：median 帶寬下，「少數候選散開、少數塌在一起」時，散開候選把 median 拉高 →
      塌縮 pair 落在 RBF 飽和區、推力被削弱 (梯度∝exp(-d2/h)/h)；值有 exp(-1)≈0.37 的軟下限。
      若 TB 看到 cand_similarity 壓不下去 / score_spread 仍塌，根因多半在此 → 升級路徑：min/per-point
      帶寬 或 改 hinge(margin) 排斥。先用此版量測，good SM 下塌縮未必嚴重，再決定要不要升級。
    """
    K = logits.shape[0]
    if K < 2:
        return logits.new_zeros(())
    d2 = torch.cdist(logits, logits) ** 2                  # (K, K) 平方 L2
    off = ~torch.eye(K, dtype=torch.bool, device=logits.device)
    h = d2.detach()[off].median().clamp_min(1e-12)
    return torch.exp(-d2 / h)[off].mean()


def boundary_threshold(seen: Tensor, kappa: float = 1.5) -> float:
    """boundary-gated ACP 的「出界」門檻 τ_b = κ · replay 典型 NN 間距。

    典型間距 = 各 pattern 到「最近的另一個」pattern 的 per-element MSE 的中位數
    (排除自身 + 重複/近重複，閾值 1e-9，與 boundary_loss 的 d>1e-9 一致)。boundary≥τ_b 視為「衝出
    SM 可信區」。全相同/M<2 → inf (無從定義間距 → 閘門永不判出界 → 退回現行 ACP)。

    :param seen:  (M, N) 已見 pattern (攤平、通常＝replay 緩衝)。
    :param kappa: 門檻係數 (越大越寬鬆、越不易判出界)。
    """
    if seen.shape[0] < 2:
        return float("inf")
    D = torch.cdist(seen, seen)
    mse = (D * D) / seen.shape[1]                          # per-element MSE (對齊 boundary_loss 單位)
    mse[mse <= 1e-9] = float("inf")                        # 排除自身 + 重複/近重複
    return kappa * float(mse.min(dim=1).values.median())


def worst_margin(response, labels, targets) -> tuple:
    """in-band(中央平台)對 spec 的「最差餘裕」(dB)：正＝達標、越高越好。客觀判讀指標(非 loss)。

    定義與 custom_loss_minmax 的嚴格點一致(亦＝論文 in-band spec)：對每個 label 取「中央平台」＝
    width[0]+width[1] : +width[2] 的頻點(n257 width [5,0,7,0,5] → 索引 5:12 ≈ 26.5-29.5GHz)，
      method=low  (S11) ：margin = center − max(band)   (帶內都低於 center → 達標)
      method=high (Gain)：margin = min(band) − center   (帶內都高於 center → 達標)
    worst-margin = min over labels。response 與 benchmark 共用同一定義(`script/benchmark_vs_random.py`)。

    :param response: (n_labels, n_points) 或攤平,列序＝labels。
    :param labels:   label 順序 (例 ['S11','Gain'])。
    :param targets:  {label: {center, width, method, ...}} (＝ TrainConfig.targets)。
    :return: (worst_margin, {label: margin})。
    """
    labels = list(labels)
    response = torch.as_tensor(response).float().reshape(len(labels), -1)
    margins = {}
    for i, label in enumerate(labels):
        t = targets[label]
        if "method" not in t:   # 只支援 single-port 的 method(low/high) target;dual 的 interval 餘裕定義不同、未實作
            raise ValueError(f"worst_margin 目前只支援 single-port (method) target;{label} 無 'method' "
                             f"(dual interval 未實作)。")
        w = t["width"]
        band_start, band_end = w[0] + w[1], w[0] + w[1] + w[2]   # 中央平台 = in-band = 嚴格 spec 區
        n_points = response[i].shape[0]
        #! 界線檢查：width 總和超過響應點數 → band 空/截斷，band.max()/min() 會丟難懂錯。
        #  現行 n257 width [5,0,7,0,5] (band 5:12, 17 點) 合法、不觸發；防的是 width 配錯或換 x 網格沒改 width。
        if band_end > n_points or w[2] <= 0:
            raise ValueError(f"worst_margin: {label} 的中央平台切片 [{band_start}:{band_end}] "
                             f"越界或為空 (響應僅 {n_points} 點，width={list(w)})。請檢查 targets 的 width。")
        band = response[i][band_start:band_end]
        c = float(t["center"])
        margins[label] = (c - float(band.max())) if t["method"] == "low" else (float(band.min()) - c)
    return min(margins.values()), margins


###* ============================================================================
###* dual-port (二埠濾波天線) 判準：worst_margin_dual + dual_energy_max
###* 與 single 的 worst_margin **完全獨立**（single 是 golden 基準,一個 byte 都不動）；
###* 呼叫端自行依 port 選尺。規格出處＝docs/reference/notes/senior-thesis-dual-port.md §4.1/§6.8。
###* ============================================================================

def _dual_target_curve(t, n_points: int, label: str) -> np.ndarray:
    """把 targets 的 (side, center, width 五段) 展開成目標曲線 —— 與 `TargetResponse.__call__` 同一條公式。

    五段＝左平台(side) / 左斜邊(side→center 線性) / 中央平台(center) / 右斜邊 / 右平台
    （見 `antenna/response.py:165-202`）。此處只用 numpy 重算一次（不建 TargetResponse，
    避免判準函式牽進全域 spec/裝置狀態）；長度不符即 fail-fast。
    """
    w = list(t["width"])
    if len(w) != 5:
        raise ValueError(f"worst_margin_dual: {label} 的 width 需 5 段，得到 {len(w)} 段 ({w})。")
    side, center = float(t["side"]), float(t["center"])
    curve = np.concatenate([
        np.ones(w[0]) * side,
        np.linspace(side, center, w[1]),
        np.ones(w[2]) * center,
        np.linspace(center, side, w[3]),
        np.ones(w[4]) * side,
    ])
    if len(curve) != n_points:
        raise ValueError(f"worst_margin_dual: {label} 的 width {w} 展開成 {len(curve)} 點，"
                         f"但響應有 {n_points} 點。請檢查 targets 的 width。")
    return curve


def worst_margin_dual(response, labels, targets) -> tuple:
    """dual-port 的「最差餘裕」(dB)：正＝達標、越高越好。客觀判讀指標(非 loss)，與 single 的
    `worst_margin` 是**兩把獨立的尺**（single 路徑一個 byte 都沒動）。

    六項單側 margin（門檻值全部從 targets 的 side/center 讀，不硬編）:

    ====  =========================  ==================================  ===========================
    項    定義                        頻點集合 (dual_base.yaml 的 idx)      進 wm?
    ====  =========================  ==================================  ===========================
    m1    center − max(S11[帶內])     S11 target==min → idx 5-11           ✔
    m2    center − max(S22[帶內])     S22 target==min → idx 5-11           ✔
    m3    min(S21[通帶]) − center     S21 target==max → idx 3-13           ✔
    m4    side − max(S21[阻帶])       S21 target==min → idx 0-2 ∪ 14-16    ✔
    m5    min(S11[帶外]) − side       S11 target==max → idx 0-4 ∪ 12-16    ✘ (只記帳)
    m6    min(S22[帶外]) − side       S22 target==max → idx 0-4 ∪ 12-16    ✘ (只記帳)
    ====  =========================  ==================================  ===========================

    **wm_dual = min(m1..m4)；m5/m6 只入 per 不入 min**（判準定案 2026-08-10）：帶外反射由能量守恆
    `|Sii|²+|S21|²+P_rad+P_loss=1` 幾乎被 S21 阻帶規格蘊含、資訊量低，塞進 min 只會讓判準對一個
    近乎「白送」的量敏感（比照 single 把帶外拆成 usable_lo/hi 獨立紀錄的做法）。

    #! **頻點集合一律走 mask，不用 width 切片算術**（single 的 `worst_margin` 用切片是因為它的
    #  width 斜邊恆為 0）。dual 的 S11/S22 width=[4,2,5,2,4] 有斜邊，而 `np.linspace(side,center,2)`
    #  只產出端點 [side, center] → 斜邊的兩個點實際上一個等於 side、一個等於 center。
    #  切片算術 `w[0]+w[1] : +w[2]` 會切出 idx 6:11，**漏掉 idx 5 與 idx 11 這兩個真的等於 center
    #  的頻點＝靜默切錯帶**。mask (`target==target.min()/max()`) 與 `custom_loss_minmax` 同一把尺，
    #  斜邊怎麼配都不會漏。

    :param response: (n_labels, n_points) 或攤平,列序＝labels。
    :param labels:   label 順序 (dual 為 ['S11','S21','S22'])。
    :param targets:  {label: {side, center, width, ...}} (＝ TrainConfig.targets)。
    :return: (wm, per)。wm=min(m1..m4)；per 含 'm1'..'m6' 六項 + 每 label 的主 margin
             ('S11'=m1、'S22'=m2、'S21'=min(m3,m4))。
    """
    labels = list(labels)
    need = {"S11", "S21", "S22"}
    if not need.issubset(labels):
        raise ValueError(f"worst_margin_dual 需要 labels 含 {sorted(need)}，得到 {labels}。"
                         f"(single-port 請用 worst_margin)")
    response = torch.as_tensor(response).float().reshape(len(labels), -1).cpu()   # mask 走 CPU bool，統一裝置
    n_points = response.shape[1]

    curves, rows, sides, centers = {}, {}, {}, {}
    for label in ("S11", "S21", "S22"):
        t = targets[label]
        curves[label] = _dual_target_curve(t, n_points, label)
        rows[label] = response[labels.index(label)]
        sides[label], centers[label] = float(t["side"]), float(t["center"])

    #! 方向自證：S11/S22 是「帶內要更低」(center < side)、S21 是「通帶要更高」(center > side)。
    #  兩者若被寫反，min/max mask 會整組互換而 margin 靜默反號 → 這裡 fail-fast。
    for label in ("S11", "S22"):
        if not centers[label] < sides[label]:
            raise ValueError(f"worst_margin_dual: {label} 應為帶內壓低型 (center < side)，"
                             f"但 center={centers[label]}、side={sides[label]}。")
    if not centers["S21"] > sides["S21"]:
        raise ValueError(f"worst_margin_dual: S21 應為通帶抬高型 (center > side)，"
                         f"但 center={centers['S21']}、side={sides['S21']}。")

    def _lo(label):     # 該 label 目標曲線的低平台 (== curve.min()) 所在頻點的響應值
        c = curves[label]
        return rows[label][torch.as_tensor(c == c.min())]

    def _hi(label):     # 該 label 目標曲線的高平台 (== curve.max()) 所在頻點的響應值
        c = curves[label]
        return rows[label][torch.as_tensor(c == c.max())]

    per = {
        "m1": centers["S11"] - float(_lo("S11").max()),      # 帶內 S11 反射夠低
        "m2": centers["S22"] - float(_lo("S22").max()),      # 帶內 S22 反射夠低
        "m3": float(_hi("S21").min()) - centers["S21"],      # 通帶 S21 傳得過去
        "m4": sides["S21"] - float(_lo("S21").max()),        # 阻帶 S21 抑制得夠
        "m5": float(_hi("S11").min()) - sides["S11"],        # 帶外 S11 反射夠強 (記帳)
        "m6": float(_hi("S22").min()) - sides["S22"],        # 帶外 S22 反射夠強 (記帳)
    }
    wm = min(per["m1"], per["m2"], per["m3"], per["m4"])     #! m5/m6 刻意不進 min
    per["S11"], per["S22"] = per["m1"], per["m2"]
    per["S21"] = min(per["m3"], per["m4"])
    return wm, per


def dual_energy_max(response) -> float:
    """dual 每筆的能量自證：`max` over 頻點 of (|S11|²+|S21|², |S22|²+|S21|²)。

    被動網路單埠激發時 `|Sii|² + |S21|² + P_rad + P_loss = 1` → 本值**必 ≤ 1**，>1 即代表
    這筆資料/模擬壞了（免費的儀器自證；single 沒有這種閉合檢查）。harvest_dual 一萬筆實測
    max≈0.91，缺的 ~46% 就是輻射——這結構是「二埠濾波天線」而非無輻射濾波器的實錘。

    :param response: (3, n) 或攤平，**列序固定為 dual 的 labels 序 (S11, S21, S22)**，單位 dB。
    :return: 最大能量和 (無因次，線性功率)。
    """
    r = torch.as_tensor(response).float().reshape(3, -1)
    p = torch.pow(10.0, r / 10.0)                            # dB → 線性功率
    return float(torch.maximum(p[0] + p[1], p[2] + p[1]).max())