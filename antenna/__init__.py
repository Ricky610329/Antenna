"""
It includes a microstrip patch antenna and a reconfigurable intelligent surface (RIS).

Example::

    from antenna import AntennaPattern, AntennaResponse, get_result_path
    from antenna.utils import config, connect_network_drive, ROOTDIR
    config.device = "cuda:0"

    from antenna.models import ...
    from antenna.smodels import ...

    #* Select according to actual application.
    from antenna.patch import ...

    #* Basic Config
    connect_network_drive("T:", r"\\140.123.106.219\temp", "user", "ailab120")
    RESULT_PATH, is_connect_run = get_result_path('[...][{device}] ...', rootdir=ROOTDIR)
    
    #* Set Antemma Pattern
    AntennaPattern.setDefaultCoordinate((0, n, 0, n))
    PATTERN_SIZE = AntennaPattern.size(flatten=True)
    simulator = ...
    AntennaPattern.register_simulator(simulator)

    #* Set Antenna Response (建 spec → 一次安裝)
    spec = TargetResponse(labels=('S11', ...), x='n257')
    spec(side, center, width, label='S11', add=True)          # 加目標曲線
    spec.register_loss_fn('S11', loss_fn, **params)           # 綁 loss hook
    AntennaResponse.use(spec)
    RESPONSE_SIZE = spec.size(flatten=True)

"""
###* 套件匯入 ###
#? 本檔是整個閉迴路 pipeline 的「資料抽象核心」：
#?   - AntennaPattern：把 GEN 生成器吐出的張量包裝成「25x25 像素圖樣」, 並負責座標系統、
#?     多塊疊合 (merge)、STE 可微分二值化 (binarization)、呼叫 SIM 模擬器 (simulate)、
#?     以及給 GEN 用的圖樣正則化損失 (total_variation_loss / island_suppression_loss)。
#?   - AntennaResponse / TargetResponse / MultiResponses：把 SM/SIM 吐出的頻率響應包裝起來,
#?     並依「註冊的 loss hook」計算 loss, 是 GEN→SM 反傳鏈的損失來源。
from collections import defaultdict
from types import FunctionType
from typing import (
    Callable, Dict, Iterable, List, Literal,
    Optional, Self, Tuple, Union, cast,
)

from antenna.utils import Path, config, get_local_ip, tensor   #? config(全域設定)、自訂 Path、自訂 tensor
from antenna.utils.utils import TID, get_shake_128, global_exception_handler
from antenna.types import Axes, Tensor_N, Tensor_W_H   #? 形狀語意別名
from antenna.utils.data import size_converter   #? 統一的形狀轉換器 (flatten / batch / 自訂 output_shape)
from antenna.patch.patch_simulator import PatchSimulator   #? SIM 模擬器基底 (COM 驅動 HFSS), 僅作型別標註用

import numpy as np
import torch
from torch import Tensor, concat, stack
import torch.nn.functional as F      #? island_suppression_loss 用 avg_pool2d 算局部平均
import matplotlib.pyplot as plt
from loguru import logger #? pip3 install loguru
from functools import partial         #? register_loss_fn 用 partial 把 loss_fn 的固定參數預先綁定
from os.path import normpath
from time import time

def get_result_path(
    name:str = "{id}-{device}", *, 
    rootdir = None, 
    set_logger:bool = True, 
    generate_code:Optional[str] = None,
    excepthook_mode:Union[bool, Literal['only_hfss']] = 'only_hfss',
    enable_exception_handler: bool = False,
):
    """
    Args:
        name: Folder and log name, support {device}, {hash_id}, {tid}, {id}.
        set_logger: Whether to set the logger.
            EX: XXX.log
        generate_code: 
            EX: __file__
        excepthook_mode:
            If True, an email will be sent for any exception; if False, nothing will happen.
            [global_exception_handler(mode)]
        enable_exception_handler:
            config.enable_exception_handler = enable_exception_handler

    Examples: (equivalence)
    ```
    RESULT_PATH, EXISTS = get_result_path()
    RESULT_PATH, CONTINUE_RUN = get_result_path(
        "{device}-{hash_id}", # device, hash_id, tid, id
        rootdir = ROOTDIR, generate_code = __file__, enable_exception_handler = True
    )

    NAME = RESULT_PATH.stem
    ```
    """
    from script.process_files import FileProcessor
    #* 組出本次實驗的唯一識別資訊, 供資料夾/log 命名與斷點續跑判斷
    _now = int(time())                              #? 秒級 timestamp, 對應 name 中的 {id}
    _device = get_local_ip().split('.')[-1]         #? 取本機 IP 末段當作「機器代號」, 對應 {device}
    _hash_id = get_shake_128(name, length=6)        #? 由 name 雜湊出短碼, 對應 {hash_id} (相同 name 會得到相同碼)
    rootdir = Path(str(normpath(rootdir))) if rootdir else  Path(__file__).parent.parent
    result_path = rootdir.joinpath(
        "result", str(name.format(id = _now, device = _device, tid = TID.generate(), hash_id = _hash_id))
    )
    #! exists 用來判斷是否「續跑」: 若資料夾已存在代表先前已建立過同名實驗
    exists  = result_path.exists()
    result_path.not_exist_create()                  #? 不存在才建立, 避免覆蓋既有結果

    if set_logger:
        logger.add(
            result_path.joinpath(f"{result_path.stem}.log"),
            format = "{time:YYYY-MM-DD HH:mm:ss} {level} {message}",
            level = "INFO",
        )
    if generate_code:
        FileProcessor(
            output_dir = result_path,
            project_name=result_path.stem,
            generated_by=generate_code,
            verbose = False
        ).run()

    # from .utils.utils import global_exception_handler
    #* 把實驗層級的狀態寫回全域 config, 讓 pipeline 各處 (GEN/SM/SIM/trainer) 都能讀到同一份設定
    config.excepthook = global_exception_handler(excepthook_mode)  #? HFSS 例外時可寄信通知 (only_hfss)
    config.enable_exception_handler = enable_exception_handler

    config.NAME = result_path.stem
    config.RESULT_PATH = result_path
    config.ID = _hash_id
    config.CONTINUE_RUN = exists            #? 供主程式判斷要不要載入 checkpoint 接續訓練
    config.MAIN_PROGRAM = generate_code
    
    logger.info(f"The results will be saved in {result_path.absolute()} (Continue: {exists}, CUDA: {torch.cuda.is_available()})")
    return result_path, exists

def mult(_ob):
    """將可迭代物件中的所有元素連乘 (例如把 shape (H, W) 換算成總元素數)。"""
    _result = 1
    for i in _ob:
        _result *= i
    return _result

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
    
def reshape(_tensor:torch.Tensor):
    #? 小工具: 一維 -> 列向量 (1, N); 其他 -> 行向量 (N, 1), 方便對齊不同模型的維度需求
    _shape = _tensor.shape
    if len(_shape) == 1:
        return _tensor.reshape(1, _shape[0])
    else:
        return _tensor.reshape(_shape[0], 1)

#? 直接執行本檔時的簡易自測: 用隨機 361 點響應畫一張圖, 驗證 AntennaResponse.plot 可運作
if __name__ == "__main__":
    config.device = 'cpu'
    # ap = tensor(np.random.rand(40*40), dtype=torch.float32)
    # binary_ap = (ap >= 0.5).float()
    # response = AntennaResponse.getTargetResponse()
    # pattern = AntennaPattern(binary_ap, [(0, 40, 0,40)])
    # pattern.plot()
    # response.plot()
    response = AntennaResponse(torch.randn(361))
    response.plot()





