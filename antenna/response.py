"""
antenna/response.py — 頻率響應的資料抽象 (GEN→SM 反傳鏈的損失端)。

- MultiResponses ：多條響應的容器 (label -> AntennaResponse)，stack/concat/criterion。
- TargetResponse ：一組實驗的「響應規格」實例 (labels/x 軸/目標曲線/loss hooks)，
                   以 AntennaResponse.use(spec) 原子安裝。
- AntennaResponse：單條響應 + criterion (依安裝的規格比對目標)。
從 antenna/__init__.py 拆出 (純搬家)。
"""
from collections import defaultdict
from functools import partial
from types import FunctionType
from typing import Callable, Dict, Iterable, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor, concat, stack
import matplotlib.pyplot as plt

from antenna.utils.types import Axes, Tensor_N, Tensor_W_H
from antenna.utils import config, tensor
from antenna.utils.torch_utils import size_converter

class MultiResponses:
    """
    多筆頻率響應的容器 (label -> AntennaResponse)。

    #? 角色: SM/SIM 一次模擬可能吐出多個 port/多種響應 (例如 S11、增益曲線…),
    #?       每一條都用一個 label 命名。MultiResponses 把這些 AntennaResponse 收進一個有序 dict,
    #?       方便後續以 stack/concat 攤平成模型輸入, 或用 criterion() 逐一累加 loss。
    #! 注意: criterion()/concat() 的 label 對齊是依 `AntennaResponse.labels` 的「註冊順序」,
    #!       因此這裡 dict 的鍵順序必須與 registerLabels() 一致, 否則 loss 會對錯目標。
    """
    def __init__(self, responses:Union[dict, Tensor] = None):
        #? 支援三種輸入: dict(label->tensor)、單一攤平 Tensor(依 size() 還原成多筆)、或 None(空容器)
        _responses:Dict[str, AntennaResponse] = {}
        if isinstance(responses, Dict):
            #? dict 形式: 鍵即 label, 值逐一包成 AntennaResponse
            for key, response in responses.items():
                _responses[key] = AntennaResponse(response)
        elif isinstance(responses, Tensor):
            #? Tensor 形式: 用 AntennaResponse.size()=(label數, 每條長度) 還原成多列, 再依註冊 labels 命名
            for n, response in enumerate(responses.reshape(AntennaResponse.size())):
                _responses[AntennaResponse.labels[n]] = AntennaResponse(response)
        elif responses is None:
            pass        #? 空容器: 通常給 TargetResponse 之後再逐項填入目標
        else:
            raise TypeError(f"Expected type `dict or Tensor`, but got type {type(responses)}")
        self.responses = _responses

    def __len__(self) -> int:
        return len(self.responses)

    def __str__(self):
        responses_str = " ".join([f"{k}[{v.response.shape.numel()}]" for k, v in self.responses.items()])
        return f"MultiResponses(num={self.__len__()}, key={responses_str})"

    def __getitem__(self, key) -> "AntennaResponse":
        #? 同時支援用整數索引 (依插入順序) 或字串 label 取值
        if isinstance(key, int):
            if key >= self.__len__():
                raise IndexError(f"Expected size {self.__len__()} but got size {key}")
            key = list(self.responses.keys())[key]

        return self.responses[key]

    def __setitem__(self, key, value):
        self.responses[key] = AntennaResponse(value)

    def __delitem__(self, key):
        del self.responses[key]


    def __invert__(self):
        """Detach the response"""
        #? `~obj` 語法糖: 取出已 detach 並搬回 CPU 的張量, 方便畫圖/存檔而不污染計算圖
        return self.stack().detach().cpu()

    def to_list(self):
        #? 取出底層原始張量 (保留計算圖), 給 stack/concat/criterion 使用
        return [n.response for n in self.responses.values()]

    def stack(self) -> Tensor_W_H:
        #? 沿新維度堆疊 -> (label數, 每條長度), 保留二維結構
        return stack(self.to_list())

    def concat(self) -> Tensor_N:
        #? 串接成一維 -> (label數 * 每條長度), 用於攤平成模型輸入/輸出向量
        return concat(self.to_list())
    
    def size_converter(self, flatten: bool = False, batch: bool = False, output_shape = None) -> torch.Tensor:
        """:param output_shape: Priority use. (B, H, W, N) EX: "B, 1, H, W" or "B, N, 1" """
        return size_converter(
            AntennaResponse, self.to_list(),
            flatten = flatten, batch = batch, output_shape = output_shape
        )
    
    def criterion(self):
        """The loss will be calculated from the registered labels."""
        #? 把這批響應依「註冊順序 AntennaResponse.labels」重新對齊到各自的 label
        #! 用 zip 對齊: 依賴 to_list() 的順序與 labels 順序一致 (見類別說明的對齊陷阱)
        responses = {}
        for label, res in zip(AntennaResponse.labels, self.to_list()):
            responses[label] = res

        #? 從 0 起算並逐 label 累加 loss; 每個 label 各自查它註冊的 loss hook (見 AntennaResponse.criterion)
        #? 角色: 這就是 GEN→SM 反傳鏈的「總損失」, 對應每個 port/響應目標的偏差總和
        loss = tensor(0.0, requires_grad=True)
        for key, value in responses.items():
            loss = loss + AntennaResponse(value).criterion(key)
        return loss

class TargetResponse(MultiResponses):
    """
    一組實驗的「響應規格」實例：labels 順序、x 軸、目標曲線 (期望響應)、loss hooks。

    #? 角色: 描述「我們希望天線長成什麼樣子」的理想頻率響應 (例如某頻段要低 S11),
    #?       同時保管每個 label 的「目標曲線」與「對應的 loss 函式 (loss hook)」,
    #?       因此 criterion 計算時才能依 label 找到正確的目標與比較方式。
    #! 用法: 建好一個自包含的 spec 實例後, 以 AntennaResponse.use(spec) 整組安裝 (原子切換)。
    #!       建構過程不碰任何全域狀態; 兩組 spec 可在同一 process 共存, 安裝哪組由 use() 決定。
    #! 順序的雙軌制 (dual 的既有行為, 勿改):
    #!   - criterion/labels 對齊順序 = labels 參數順序 (metadata 槽位預建)。
    #!   - concat() (GEN 輸入排列) = 目標「加入」順序 (responses dict 插入序), 可與 labels 順序不同。
    """
    def __init__(self, labels=None, x=None):
        super().__init__(None)                              #? 以空容器起始, 目標之後逐項加入
        self._note = {}                                     #? label -> 人類可讀的設定字串 (供除錯/列印)
        self.metadata:dict[str, dict] = defaultdict(dict)   #? label -> {response, loss_fn, side/center/width…} 完整中繼資料
        #? x 可給代號 ('ris'/'n257') 或自訂 (start, stop, total); total 即每條響應的點數
        match x:
            case 'ris':   self.x_range = (0, 360, 361)
            case 'n257':  self.x_range = (24, 32, 17)   #? 26.5 - 28 - 29.5
            case _:       self.x_range = x              #? tuple 或 None (之後再補)
        if labels:
            self.labels = labels                        #? 走 setter: 預建 metadata 槽位, 鎖定對齊順序

    def x(self):
        """x 軸取樣點 (np.linspace 展開)。"""
        if self.x_range is None:
            raise RuntimeError("此 spec 未設定 x。建構時傳入 x= ('ris'/'n257'/(start, stop, total))。")
        return np.linspace(*self.x_range)

    def size(self, flatten:bool = False):
        """(label數, 每條響應點數)；flatten=True 回傳兩者相乘的總長度。"""
        if not self.labels or self.x_range is None:
            raise RuntimeError("此 spec 尚未設定 labels 或 x。")
        _ = (len(self.labels), self.x_range[2])
        return _[0] * _[1] if flatten else _

    def __getitem__(self, key):
        """
        Target Response Design.

        Use `setTargetResponse()` before use, otherwise use the default value
        """
        #! 取目標前必須先註冊, 否則直接報錯 (避免拿到未定義的目標曲線而靜默算錯 loss)
        if key not in self._note.keys():
            raise RuntimeError(
                f"The {key} of TargetResponse is not registered. " \
                "Please use `registerTargetResponse()` first."
            )
        return super().__getitem__(key)
    
    def __call__(self, side:float, center:float, width:Tuple[int,int,int,int,int], label:str = "response", add:bool = False) -> Tensor:
        """
        Target Response Design.

        :param side: The Y value at both ends of the response.
        :param center: The y value of the center point of the response.

        :return: AntennaResponse
        
        """
        #! width 必須剛好 5 段, 分別對應下面「左平台→左斜邊→中央平台→右斜邊→右平台」五段
        if len(width) != 5:
            raise ValueError(f"Expected 5 width, but got {len(width)}")
        #? 用五段拼出一條「梯形/凹槽」目標曲線 (常見於指定某頻段要壓低的設計):
        #?   兩端維持 side 值, 中段平滑過渡到 center 值, 中央維持 center 值
        mask_up = np.concatenate([
            np.ones(width[0]) * side,               #? 左側平台 (side 高度)
            np.linspace(side, center, width[1]),    #? 左斜邊 (side → center 線性過渡)
            np.ones(width[2]) * center,             #? 中央平台 (center 高度, 即目標凹陷/峰值處)
            np.linspace(center, side, width[3]),    #? 右斜邊 (center → side 線性過渡)
            np.ones(width[4]) * side                #? 右側平台 (side 高度)
        ])
        # expected_response = np.array(mask_up)#.reshape(-1, sum(_width))
        expected_response = tensor(np.array(mask_up), dtype=torch.float32, device=config.device)

        #? add=True 才真正「註冊」進目標表; 否則只回傳曲線供預覽 (不污染既有目標)
        if add:
            self[label] = expected_response                 #? 寫入目標容器 (走 MultiResponses.__setitem__)
            self._note[label] = f"side={side}, center={center}, width={width}"
            self.metadata[label].update({                   #? 同步完整參數到中繼資料, 供列印/重現
                'response': expected_response,
                'side': side,
                'center': center,
                'width': width,
                'note': f"side={side}, center={center}, width={width}",
            })

        return expected_response
    
    def register_loss_fn(self, label, loss_fn, **loss_fn_param):
        #? 把該 label 的 loss 函式連同固定參數用 partial 綁定後存進 metadata;
        #? criterion() 之後就用這個被綁好的函式比較「模擬響應 vs 目標響應」
        self.metadata[label].update({
            'loss_fn': partial(loss_fn, **loss_fn_param),
            #? 另存一個可讀名稱 (函式取 __name__, 類別 callable 取 class 名), 純供列印/紀錄
            'loss_fn_name': loss_fn.__name__ if isinstance(loss_fn, FunctionType) else loss_fn.__class__.__name__
        })

    def loss_fn(self, label) -> Callable[..., Tensor]:
        #? 取回該 label 已綁定參數的 loss 函式
        return self.metadata[label]['loss_fn']

    @property
    def labels(self):
        #? labels 直接以 metadata 的鍵序為準 (註冊先後即為對齊順序)
        return list(self.metadata.keys())

    @labels.setter
    def labels(self, labels:Iterable[str]):
        #! 用「讀取 defaultdict[label]」這個副作用來「預先建立」各 label 的空 metadata 槽位,
        #! 確保 labels 順序固定下來; `_ =` 只是觸發 defaultdict 自動建鍵, 並非真的要值
        for label in labels:
            _ = self.metadata[label]
    
    def concat(self):
        #? 把所有目標曲線串成一維, 並順手「驗證」總長度是否等於 spec 宣告的 (label數 x 點數)
        _result = super().concat()
        #! 尺寸不符通常代表 labels 宣告與實際加入的目標對不上, 直接報錯避免後續靜默錯誤
        if _result.size(0) != self.size(flatten = True):
            raise RuntimeError(
                'The concat size does not match the set size. ' \
                'Please check the spec labels.' \
                f'\n{_result.size(0)} != {self.size(flatten = True)}{self.size()}'
            )
        return _result
    
    def __str__(self):
       
        _ = " ".join(
            [
                f"{key}({value['note']}, loss={value.get('loss_fn_name', None)})" 
                for key, value in self.metadata.items()
            ]
        )
        return f"TargetResponse({_})"
    
class AntennaResponse:
    """
    Antenna Response Design.

    Attributes:
        response (Tensor): response
        target (TargetResponse): target response
    """
    #? 角色: 包裝「單一條」頻率響應 (SM 預測或 SIM 實測), 並對接共用的目標 target 與 loss hook,
    #?       criterion() 即在此把這條響應與其目標比對出 loss, 是 GEN→SM 反傳的損失末端。
    #? 兩種預設 x 軸 (橫軸取樣點): RIS 用角度 0~360, patch(n257 頻段) 用 24~32 GHz 共 17 點
    x_patch_n257 = np.linspace(24, 32, 17) #? 26.5 - 28 - 29.5
    x_ris = np.linspace(0, 360, 361)

    #! 類別層級的「當前規格」: 由 use(spec) 整組安裝 (唯一的寫入點), 深層內部
    #! (MultiResponses 的 Tensor 還原 / criterion 對齊 / SM 推論包裝) 透過它解析。
    #! 訓練端的「讀取」(維度/GEN 輸入) 請直接拿著 spec 實例, 不要讀類別狀態。
    target = TargetResponse()

    def __new__(cls, response):
        #? 工廠式分派: 傳 dict 會「改建」成 MultiResponses; 傳已是 AntennaResponse 則原樣返回 (冪等)
        if isinstance(response, cls):
            return response
        elif isinstance(response, Dict):
            return MultiResponses(response)
        else:
            return super(AntennaResponse, cls).__new__(cls)

    def __init__(self, response:Union[Tensor, Dict]):
        """
        Antenna Response Design.

        Args:
            response: Response of the antenna.
        
        Raises:
            TypeError: If the response is not a tensor.
        
        """
        #! 若 __new__ 已分派成既有實例或 MultiResponses, 這裡直接跳過, 避免重複初始化覆蓋資料
        if isinstance(response,(AntennaResponse, Dict) ):
            return
        elif isinstance(response, Tensor):
            response = response.to(config.device)       #? 統一搬到 config.device (CPU/CUDA), 保證後續運算同裝置
        else:
            raise TypeError("Expected Tensor, but got {}".format(type(response)))

        #? 同時保存兩種視角: response 為一維 (給 loss/串接), vertical 為 (1, N) 二維 (給某些畫圖/模型)
        if len(response.shape) == 1:
            self.response = response
            self.vertical = self._reshape2vertical()
        else:
            self.response = response.reshape(-1)
            self.vertical = response

    def __str__(self):
        return f"AntennaResponse(size={self.response.size().numel()})"

    def __invert__(self):
        """Detach the response"""
        return self.response.detach().cpu()
    
    def _reshape2vertical(self):
        #? 把一維響應轉成 (1, N) 並開啟梯度追蹤, 供 batch 維度=1 的下游使用
        assert len(self.response.shape) == 1
        _v = self.response.reshape(1, self.response.shape[0])
        _v.requires_grad_(True)
        return _v

    def plot(self, label, axes:Optional[Axes] = None, show:bool = False):
        #? 把「目標曲線(紅)」與「本次模擬響應(藍)」疊在同一張圖, 直觀檢視兩者差距 (即 loss 來源)
        ax:Axes = plt.axes(axes) # type: ignore
        ax.set_title(f'Antenna Response')
        ax.plot(self.target[label].response.cpu().detach(), color='red', label='Target')
        ax.plot(self.response.cpu().detach(), color='blue', label='Simulation')
        ax.legend()
        if show: plt.show()
        return ax

    @classmethod
    def use(cls, spec:TargetResponse) -> TargetResponse:
        """安裝一組響應規格 (原子切換)。這是類別層級狀態「唯一」的寫入點。

        spec 需自包含 (labels + x)；安裝後深層內部 (MultiResponses 還原/criterion)
        即依此 spec 解析。回傳 spec 本身, 方便呼叫端繼續持有實例使用。
        """
        if not spec.labels or spec.x_range is None:
            raise ValueError("spec 必須自包含 labels 與 x (TargetResponse(labels=..., x=...))。")
        cls.target = spec
        cls.labels = tuple(spec.labels)
        cls._x = spec.x_range
        return spec

    @classmethod
    def x(cls):
        """Get the x-axis value of this response."""
        if not hasattr(cls, '_x'):
            raise RuntimeError("No spec installed. Please use `AntennaResponse.use(spec)` first.")
        return np.linspace(*cls._x)

    @classmethod
    def size(cls, flatten:bool = False):
        #? flatten=True 回傳總元素數 (int)；否則回傳 (列數, 行數) tuple
        """The number of labels used to calculate loss and the number of points in their labels."""
        if not cls.target.labels:
            raise RuntimeError("No spec installed. Please use `AntennaResponse.use(spec)` first.")
        #? 尺寸 = (label數, 每條響應點數); flatten=True 回傳兩者相乘的總長度 (給 RESPONSE_SIZE)
        _ = (len(cls.target.labels), cls._x[2])
        return _[0] * _[1] if flatten else _

    @classmethod
    def to_str(cls):
        """Get response information and default values."""
        return f"AntennaResponse(size={cls.size()}, x={cls._x}, target={cls.target})"

    def criterion(self, label:str = "response", **param) -> Tensor:
        """[Loss Function] spec 須先以 register_loss_fn 綁好該 label 的 loss hook。"""
        #! 未註冊 loss hook 就呼叫會直接報錯, 避免靜默回傳無意義的 loss
        if label not in self.target.labels:
            raise RuntimeError(f"The {label} of LossHook is not registered in the installed spec.")

        #? 取出該 label 已綁好參數的 loss 函式, 餵入「本條響應 self.response」算出純量 loss
        #! loss 函式通常會內部存取 target[label] 取得目標曲線; 此處只負責把模擬響應傳進去
        return self.target.loss_fn(label)(
            self.response, **param
        )

