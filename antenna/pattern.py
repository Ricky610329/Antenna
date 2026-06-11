"""
antenna/pattern.py — 像素化天線圖樣的資料抽象 (閉迴路的「物件」本體)。

AntennaPattern：座標系統、多塊疊合 (merge)、STE 可微分二值化 (binarization)、
呼叫 SIM 模擬 (simulate)、可製造性正則化 (total_variation / island_suppression)。
從 antenna/__init__.py 拆出 (純搬家)。
"""
from typing import Callable, Dict, List, Optional, Self, Tuple, Union, cast

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
from loguru import logger

from antenna.utils.types import Axes
from antenna.utils import config
from antenna.utils.torch_utils import size_converter
from antenna.patch.patch_simulator import PatchSimulator
from antenna.response import AntennaResponse

class AntennaPattern:
    """
    天線像素圖樣的核心抽象 (例如 25x25 的二元金屬/空白佈局)。

    #? 角色: 它是 GEN→SM→SIM 之間流動的「pattern 載體」。GEN 生成器吐出連續張量 (logits),
    #?       經 binarization() 用 STE 變成可微分的 0/1 圖樣; merge() 把多塊子圖疊成一張大圖;
    #?       simulate() 把圖樣丟給 SIM(HFSS) 取得真實響應; 另提供 total_variation_loss /
    #?       island_suppression_loss 等正則化, 引導 GEN 產生連通、可製造的圖樣。
    #! 多處狀態掛在「類別層級」(座標、模擬器、歷史、tau), 因此整個 pipeline 共用同一份設定/歷史。
    """
    #? 訓練過程的 (pattern, 響應) 歷史紀錄; simulate() 每次模擬都會 append, 供線上重訓 SM / 分析使用
    _history_datas:List[List[torch.Tensor]] = []
    _best_loss = float('inf')           #? 記錄迄今最佳 loss (供 early-stop / rollback 邏輯參考)

    #? tau(二值化溫度) 不再是「會被動態改寫的全域類別屬性」：改由排程器
    #? (AdaptiveCyclicalScheduler) 產生 → 訓練迴圈讀取 → 顯式傳入 binarization()。
    #? 控制 sigmoid 陡峭度 (越小越接近硬 0/1，須 > 0)；binarization 未提供時預設 1.0。

    def __new__(cls, pattern:"AntennaPattern", *args) -> "AntennaPattern":
        #? 冪等工廠: 傳入的已是 AntennaPattern 就原樣返回, 避免重複包裝
        if isinstance(pattern, AntennaPattern):
            return pattern
        else:
            return super(AntennaPattern, cls).__new__(cls)
    
    def __init__(self, pattern:Union[Tensor, List], coordinate:Optional[Tuple[int,int, int, int]] = None):
        """
        pattern 可以是：單張 2D Tensor (搭配 coordinate 或類別預設座標)，
        或多塊清單 [(pattern, x1, x2, y1, y2), ...] (各自帶座標)。
        """

        #! __new__ 已把既有實例原樣返回, 這裡再判一次以免重複初始化清空 patterns
        if isinstance(pattern, AntennaPattern):
            return

        #* The core of this class.
        #? patterns 是本類別的核心資料結構: 一串 (子圖, x1, x2, y1, y2), 各自帶座標, merge 時依序疊圖
        #? [(pattern, x1, x2, y1, y2), ...] >>> pattern is 2D
        self.patterns:List[Tuple[Tensor, int, int, int, int]] = []

        if isinstance(pattern, Tensor):
            #? 單張圖樣: 先 clamp 到 [0,1] (像素值合法範圍), 座標優先用傳入值, 否則回退到類別預設座標
            self.input_tensor = torch.clamp(pattern.to(config.device), min=0.0, max=1.0)
            self.coordinate:Union[Tuple[int,int, int, int], Tuple] = coordinate or getattr(self, '_antenna_pattern_coordinate', None)

            self._check_input()     #? 驗證維度並把這張圖放進 patterns

        elif isinstance(pattern, List):
            #? 已是 [(子圖, 座標…), …] 的多塊清單: 直接採用 (用於 copy / __add__ 疊圖結果)
            self.patterns = pattern

        else:
            raise TypeError(
                f"Expected type for pattern is Tensor or List, but got {type(pattern)}"
            )
    
    def _check_input(self):
        """驗證輸入圖樣維度, 並依座標把一維向量還原成二維, 再登記進 patterns。"""
        _dim = self.input_dim()
        _c = self.coordinate            #? (x1, x2, y1, y2): 這張子圖在大圖中的擺放範圍
        _input_tensor = self.input_tensor

        #! 沒有座標就無從擺放/還原形狀, 直接報錯提示先 setDefaultCoordinate()
        if not _c: raise ValueError(
            'Please enter the `coordinate` parameter or use `setDefaultCoordinate()` to set the default value.'
        )
        if _dim == 1:
            #? GEN 常吐出攤平的一維 logits, 依座標寬高 reshape 回 (高, 寬) 二維圖
            _input_tensor = _input_tensor.reshape((_c[1]-_c[0], _c[3]-_c[2]))
        elif _dim == 2:
            pass        #? 已是二維就直接使用
        else:
            raise ValueError(f"Input pattern expected >1 dimension, but got {_dim} dimension")

        #? 登記成一塊帶座標的子圖, 之後 merge() 會依此座標把它貼到大圖上
        self.patterns.append(
            (
                _input_tensor, _c[0], _c[1], _c[2], _c[3]
            )
        )

    @property
    def series(self):
        """One-dimensional array after merge."""
        #? merge 成大圖後攤平成一維; 供需要向量輸入的場合 (例如比對/餵 SM)
        return self.merge().reshape(-1)
    
    @property
    def fill_rate(self) -> float:
        """計算並返回天線 pattern 的金屬填充率。"""
        #? 填充率 = 為 1 的像素數 / 總像素數; 常作為監控指標或正則化目標 (避免全空/全滿)
        merged_pattern = self.merge()
        if merged_pattern.numel() == 0:
            return 0.0
        return (torch.sum(merged_pattern) / merged_pattern.numel()).item()
    
    @classmethod
    def register_simulator(cls, simulator:Union[PatchSimulator, Callable[[Tensor],Dict[str, Tensor]]]):
        #! 註冊「類別層級」的模擬器 (SIM, 即 COM 驅動的 HFSS); 之後所有實例 simulate() 都共用這顆
        cls._simulator = simulator

    @classmethod
    def getAllPixel(cls):
        """
        TODO: 目前是取回所有的像素點，但實際上是取得大圖的像素點
        """
        #? 依預設座標算出整張大圖的像素總數 (寬 x 高)
        x1, x2, y1, y2 = cast(Tuple[int,int, int, int], getattr(cls, '_antenna_pattern_coordinate', (0,0,0,0)))
        return (x2-x1)*(y2-y1)
    
    @classmethod
    def size(cls, flatten:bool = False):
        #? flatten=True 回傳總元素數 (int)；否則回傳 (列數, 行數) tuple
        """The number of labels used to calculate loss and the number of points in their labels."""
        #! 尺寸取自類別預設座標, 因此必須先 setDefaultCoordinate(); flatten=True 給出 PATTERN_SIZE(總像素數)
        if not hasattr(cls, '_antenna_pattern_coordinate'):
            raise RuntimeError("Please use `setDefaultCoordinate()` first.")
        x1, x2, y1, y2 = cast(Tuple[int,int, int, int], getattr(cls, '_antenna_pattern_coordinate', (0,0,0,0)))

        #? flatten -> 寬*高(攤平長度); 否則 -> (寬, 高) 二維形狀, 供 GEN 輸出/reshape 對齊
        return (x2-x1)*(y2-y1) if flatten else ((x2-x1), (y2-y1))

    def size_converter(self,flatten: bool = False, batch: bool = False, output_shape = None) -> torch.Tensor:
        """:param output_shape: Priority use. (B, H, W, N) EX: "B, 1, H, W" or "B, N, 1" """
        return size_converter(
            self, self.merge(),
            flatten = flatten, batch = batch, output_shape = output_shape
        )

    @classmethod
    def _getRandomPattern(cls, w=40, h=40):
        #? (內部用) 以常態分布隨機數 >0.5 門檻產生二元圖; 注意填充率不可控 (近似 ~31%), 故另有 getRandomPattern
        patterns = torch.randn(
            w, h,
            dtype = torch.float32,
            device = config.device
        )
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
        #? 做法: 先放定量的 1 再隨機洗牌, 因此填充率「精確可控」(優於 _getRandomPattern 的機率式門檻)
        w = shape[0]
        h = shape[1]
        total_pixels = w * h
        num_ones = int(total_pixels * fill_rate)        #? 依目標填充率算出要放幾個 1

        # 生成一個扁平化的一維數組
        pattern_flat = np.zeros(total_pixels)
        pattern_flat[:num_ones] = 1                     #? 前 num_ones 個設 1, 其餘為 0

        # 隨機打亂
        np.random.shuffle(pattern_flat)                 #? 洗牌讓 1 隨機散布, 但總數不變 -> 填充率精準

        # 重塑為目標形狀並轉換為 PyTorch Tensor
        pattern_tensor = torch.tensor(pattern_flat.reshape(shape), dtype=torch.float32, device = config.device)
        return cls(pattern_tensor, (0, w, 0, h))

    def __str__(self):
        _shape = self.merge().shape
        return f"AntennaPattern(Pattern_num={self.__len__()} Shape=[{_shape[0]}, {_shape[1]}] Size=[{_shape.numel()}])"
    
    def __getitem__(self, key) -> "AntennaPattern":
        #? 取出第 key 塊子圖, 重新包成獨立的單塊 AntennaPattern (保留其原座標)
        if key >= self.__len__():
            raise IndexError(f"Expected size {self.__len__()} but got size {key}")
        pattern, x1, x2, y1, y2 = self.patterns[key]
        return AntennaPattern(pattern, (x1, x2, y1, y2))

    def __add__(self, other):
        #? `+` 語意: 把兩者的子圖清單串接成「多塊疊合」圖樣 (merge 時後者會蓋過前者)
        if isinstance(other, AntennaPattern):
            antenna_pattern = self.copy()
            antenna_pattern.patterns = self.patterns + other.patterns
            #! 疊合後已無單一輸入/單一座標的概念, 故把 coordinate/input_tensor 清為 None
            antenna_pattern.coordinate = None
            antenna_pattern.input_tensor = None

            return antenna_pattern
        else:
            raise TypeError(
                "Unsupported operand type for +: 'AntennaPattern' and '{}'".format(type(other))
            )

    def __len__(self):
        #? 子圖塊數 (不是像素數)
        return len(self.patterns)

    def __invert__(self):
        """Detach the response"""
        #? `~pattern` 語法糖: 取出 merge 後已 detach 並搬回 CPU 的大圖, 方便畫圖/存檔
        return self.merge().detach().cpu()

    def input_dim(self) -> int:
        #! 僅適用「單塊」圖樣; 多層疊合後 input_tensor 為 None 會直接報錯
        if self.input_tensor is None:
            raise RuntimeError("This function is not for multilayer boards.")

        #? 一維向量, 或第 0 維為 1 的 (1, N) 都視為「邏輯一維」(需 reshape 還原成二維)
        if len(self.input_tensor.shape) == 1 or self.input_tensor.shape[0] == 1:
            return 1
        else:
            return self.input_tensor.dim()

    def copy(self):
        #? 以現有 patterns 清單複製出新實例 (淺層: 共用底層子圖張量)
        return AntennaPattern(self.patterns)

    @classmethod
    def setDefaultCoordinate(cls, _coordinate:Tuple[int, int, int, int]):
        """
        Coordinate Design.

        """
        #! 設定「類別層級」預設座標 (x1, x2, y1, y2), 即整個 pipeline 共用的 pattern 畫布尺寸,
        #! size()/binarization() 還原形狀都依賴它; 通常在程式啟動時呼叫一次 (見模組頂端 Example)
        if not isinstance(_coordinate, tuple):
            raise TypeError(f"Expected tuple, but got {type(_coordinate)}")

        if not len(_coordinate) == 4:
            raise ValueError(f"Expected tuple of length 4, but got {len(_coordinate)}")

        setattr(cls, '_antenna_pattern_coordinate', _coordinate)

    def binarize(self, threshold = 0.5):
        """Binarize and become gradient-free."""
        #! 「硬」二值化: 直接門檻成 0/1 並切斷梯度, 用於最終推論/送 HFSS, 不可用於需反傳的訓練步驟
        #? 與下方 binarization()(STE 可微分) 不同, 這裡刻意不保留梯度
        bi = (self.merge() >= threshold).float()
        return AntennaPattern(bi, (0, len(bi), 0, len(bi)))
    
    @classmethod
    def binarization(cls, pattern:Tensor, tau:Optional[float] = None, threshold = None, *, only_soft:bool=False):
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
        #! 核心: 這是讓「不可微分的 0/1 二值化」可以反傳的關鍵, 是 GEN 能被梯度優化的前提。
        #*  Gradient is required
        pattern.requires_grad_(True)
        #? tau(溫度) 控制 sigmoid 陡峭度: 越小越接近硬階梯, 但梯度越尖; 設下限 1e-4 避免除零/數值爆炸。
        #? tau 為顯式參數 (由排程器產生、迴圈傳入); 未提供時預設 1.0, 不再讀寫全域。
        _tau = tau or 1.0
        if _tau < 1e-4: _tau = 1e-4

        if len(pattern.shape) == 1:
            pattern = pattern.reshape(*cls.size())      #? GEN 輸出常是攤平向量, 依預設座標還原成二維圖

        # 將 logits 限制在 [-10, 10] 之間，Sigmoid(-10) 已極趨近 0，Sigmoid(10) 極趨近 1
        pattern = torch.clamp(pattern, min=-10.0, max=10.0)     #? 夾住 logits 防止 sigmoid 飽和導致梯度消失/NaN

        #* Calculate threshold and steepness
        #? 門檻預設取整張圖均值 (detach 不讓門檻參與梯度), 等效自適應「中位線」; steepness=1/tau 即陡峭度
        threshold = threshold or pattern.mean().detach() # avg
        steepness = 1/_tau

        #* Produces a "soft" approximation
        #  This is to provide a smooth gradient during "backward" propagation.
        #? soft 是平滑的 sigmoid 近似, 介於 0~1, 提供「backward 時可用的軟梯度」
        soft_pattern = torch.sigmoid(steepness * (pattern - threshold))
        if torch.isnan(soft_pattern).any():
            soft_pattern = torch.nan_to_num(soft_pattern, nan=0.5)      #? 數值保險: 萬一出現 NaN 退回中性 0.5

        if only_soft is True: return soft_pattern       #? 只要軟值時直接回傳 (例如想看連續機率圖)

        #* Produces a "hard" binarization result (0/1, not differentiable).
        #  This is to get the 0/1 result you want during "forward" propagation.
        #? hard 是真正要送進 SM/SIM 的 0/1 結果 (對 soft 四捨五入), 但 round 本身無梯度
        hard_pattern = torch.round(soft_pattern)

        #* STE
        #  Forward(hard):   (hard - soft) + soft
        #  Backward(soft)： `.detach()` will block the gradient of hard_pattern
        #! Straight-Through Estimator 訣竅:
        #!   forward 數值 = (hard - soft).detach() + soft = hard (因 detach 部分視為常數, 前向就是硬 0/1);
        #!   backward 梯度 = 只有最後那個 soft 帶梯度 -> d(binary)/d(pattern) 等同 soft 的梯度。
        #!   => 前向用硬值(符合物理), 反向走軟梯度(可優化 GEN), 兩全其美。
        binary_pattern = (hard_pattern - soft_pattern).detach() + soft_pattern

        return binary_pattern

    def binarization_(self, tau:Optional[float] = None, threshold = None):
        #? 原地版 (尾底線慣例): 先 merge 成大圖, 經 STE 二值化後「取代」自身 patterns 為單一塊
        #! 注意座標為 (0, 寬, 0, 高) -> 元組順序是 (x1, x2=shape[1], y1, y2=shape[0]), 與 merge 的 [y, x] 對應
        pattern = self.merge().clone()
        shape = pattern.shape
        self.patterns = [(
            AntennaPattern.binarization(pattern, tau=tau, threshold=threshold),
            0, shape[1], 0, shape[0]
        )]
        

    def merge(self) -> torch.Tensor:
        """
        將所有 pattern 合併成一個大的底層 pattern
        - 後加入的 pattern 會覆蓋前面的 pattern
        - 返回合併後的二維 tensor
        """
        if not self.patterns:
            raise ValueError("No patterns to merge")

        #? 先求所有子圖座標的外接框 (min/max), 作為大畫布範圍
        max_x = max(x2 for _, _, x2, _, _ in self.patterns)
        min_x = min(x1 for _, x1, _, _, _ in self.patterns)

        max_y = max(y2 for _, _, _, _, y2 in self.patterns)
        min_y = min(y1 for _, _, _, y1, _ in self.patterns)

        #! 索引是 [y, x] (列在前): base_pattern 形狀 (max_y, max_x), 貼圖時用 [y1:y2, x1:x2]
        base_pattern = torch.zeros((max_y, max_x))
        for pattern, x1, x2, y1, y2 in self.patterns:
            #! 疊圖語意: 直接賦值覆蓋 -> 清單中「後加入的子圖會蓋過先前的」(__add__ 順序決定優先權)
            base_pattern[y1:y2, x1:x2] = pattern  # 後面的 pattern 覆蓋前面的

        #? 最後裁回實際外接框 [min_y:max_y, min_x:max_x] 並搬到 config.device
        return base_pattern.to(config.device)[min_y:max_y, min_x:max_x]
    

    def simulate(self, no_grad:bool = True, **param):
        """把目前圖樣送進已註冊的模擬器, 取回頻率響應 (預設不追蹤梯度)。

        #? 角色: 這是接 SIM(真實 HFSS) 取得 ground truth 的入口; 也可接 SM 代理當可微分替身。
        #? no_grad=True(預設): 對應呼叫不可微分的 HFSS(SIM), 純取真值;
        #? no_grad=False:      保留計算圖, 供透過可微分代理 SM 反傳更新 GEN。
        """
        pattern = self.merge()      #? 先疊合成最終大圖再送模擬
        result_response = {}

        if hasattr(self, "_simulator"):
            if no_grad:
                with torch.no_grad():
                    try:
                        #? 送 detach 後的圖樣 (HFSS 不需梯度); SIM 經 COM 驅動 HFSS, 偶發崩潰故包 try
                        result:Dict[str, Tensor] = self._simulator(pattern.detach(), **param)
                    except Exception as e:
                        #! HFSS/COM 不穩時的自癒: 砍掉重啟模擬器再跑一次, 避免整輪訓練中斷
                        logger.warning(f'模擬器發生錯誤, 將重新啟動並執行: {e}')
                        self._simulator.restart(kill=True)
                        result:Dict[str, Tensor] = self._simulator(pattern.detach(), **param)
            else:
                #? 保留梯度路徑 (走可微分 SM 代理時用), 故不 detach、不進 no_grad
                result:Dict[str, Tensor]  = self._simulator(pattern, **param)
        else:
            raise RuntimeError(
                "Please use `register_simulator()` to register the simulator."
            )

        #? 模擬器回傳 {label: 響應張量}; 逐一包成 AntennaResponse
        for key, value in result.items():
            result_response[key] = AntennaResponse(value)

        #! 把 (pattern, 響應) 存進類別層級歷史: 這是「線上重訓 SM」的資料來源 (閉迴路關鍵)
        # TODO
        # if not any([pattern.equal(p) for p, _ in self._history_datas]):
        AntennaPattern._history_datas.append(
            [pattern, result_response]
        )

        #? 用 dict 建構會走 __new__ 分派成 MultiResponses, 對外提供統一的多響應介面
        return AntennaResponse(result_response)

    
    def plot(self, axes:Optional[Axes] = None, show:bool = False, title:str = "Antenna Pattern {shape}"):
        #? 視覺化 merge 後的整張圖樣 (detach 不影響梯度)
        ax:Axes = plt.axes(axes) # type: ignore
        ax.set_title(title.format(shape=self.size()))
        ax.imshow(self.merge().cpu().detach(), cmap='viridis')
        ax.axis('off')
        if show: plt.show()
        return ax

    def plot_individual(self, axes:Optional[Axes] = None, show:bool = False):
        #? 把各子圖「分別」貼到空白底圖後橫向拼接, 用於檢視疊合前每塊的內容
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

        ax:Axes = plt.axes(axes) # type: ignore
        ax.set_title("Antenna Pattern Individual")
        ax.imshow(torch.cat(_result, dim=1).cpu().detach(), cmap='viridis')
        if show: plt.show()
        return ax

    def mutate(self, rate):
        #? 隨機翻轉一定比例的像素 (0<->1), 類似演化演算法的突變算子, 用於擾動/多樣化探索
        matrix = self.merge()
        total = matrix.numel()
        n = int(total * rate)                       #? 依比例算出要翻轉的像素數
        indices = torch.randperm(total).tolist()    #? 全像素隨機排列, 取前 n 個當突變點
        selected_indices = indices[:n]

        for idx in selected_indices:
            i, j = divmod(idx, matrix.size(1))      #? 一維索引換回 (列, 欄)
            matrix[i, j] = 1 - matrix[i, j]         #? 0<->1 翻轉
        return AntennaPattern(matrix)
    
    def total_variation_loss(self, weight=0.01):
        """計算 Total Variation Loss 以抑制過度破碎的圖樣"""
        #? 角色: 加在 GEN 損失上的圖樣正則化, 懲罰相鄰像素的差異 -> 鼓勵大塊連通、好製造的金屬區。
        #? 因為 STE 使二值化可微, 這個對 merge 後圖樣的 loss 也能反傳回 GEN。
        img = self.merge()
        h_img, w_img = img.size()

        #? 分別計算垂直(上下相鄰)與水平(左右相鄰)的平方差總和; 差異越多代表越破碎 -> loss 越大
        tv_h = torch.pow(img[1:, :] - img[:-1, :], 2).sum()
        tv_w = torch.pow(img[:, 1:] - img[:, :-1], 2).sum()

        #? 除以像素數做正規化, 再乘權重控制正則化強度
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
        #? 角色: 與 total_variation_loss 互補的圖樣正則化, 專打「孤立噪點/孔洞」, 同樣可反傳回 GEN。
        img = self.merge()

        # 確保為浮點數
        if not img.is_floating_point():
            img = img.float()
            
        # 準備進行 2D Pooling: 需要 (Batch, Channel, Height, Width)
        # 這裡假設 img 為 (H, W)，擴展為 (1, 1, H, W)
        img_input = img.unsqueeze(0).unsqueeze(0)
        
        # 計算局部平均 (Local Average)
        # Padding 設為 kernel_size // 2 以保持輸出尺寸不變
        avg_img = F.avg_pool2d(
            img_input, 
            kernel_size=kernel_size, 
            stride=1, 
            padding=kernel_size // 2
        )
        
        # 去掉多餘維度回歸 (H, W)
        avg_img = avg_img.squeeze(0).squeeze(0)
        
        # 計算像素與局部平均的 L1 差異
        # 若某點是孤島 (值為 1，周圍全 0)，平均值很低 (如 0.1)，差異大 (0.9) -> Loss 高
        # 若某點在內部 (值為 1，周圍全 1)，平均值很高 (如 1.0)，差異小 (0.0) -> Loss 低
        loss = torch.abs(img - avg_img).sum()
        
        return weight * loss / img.numel()
    
