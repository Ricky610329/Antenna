"""
antenna/types.py — 全專案共用型別定義模組

本模組集中定義整個反向設計閉迴路系統所需的：
  - TypeVar / ParamSpec：供 Models、SurrogateModel、AdaptiveCyclicalScheduler 等泛型類別使用
  - Protocol：描述結構性介面契約（如 Sizable、CallableModule）
  - TypedDict：描述固定鍵名的字典結構（如 Checkpoint、ResultType、RecordStateDict）
  - TypeAlias：為 Tensor 形狀賦予語意名稱（如 Tensor_B_N、Tensor_W_H）

所有可執行模組皆透過 `from antenna.types import *`（經由 antenna/__init__.py）取得此處定義。
型別 import 順序對循環引用的解析至關重要，請勿任意調整。
"""
# Defer type annotation evaluation to resolve forward reference and circular import issues.
from __future__ import annotations

from numbers import Number
from types import FunctionType
# Can use the built-in.
from typing import (
    Tuple,
    List,
    Dict,
    Deque
)
# Function
from typing import (
    cast,
    overload
)
from typing import (
    TYPE_CHECKING,
    TypeAlias,
    Protocol,
    TypeVar,
    #* TypeVar Param
    # bound: Must be of the specified type or a subclass
    # covariant: Can contain subclasses
    # contravariant: Can contain parent class
    TypeVarTuple,   #? *TypeVarTuple
    ParamSpec,
    Callable,       #? Callable[..., Any]
    Any,
    Optional,
    Union,
    Sequence, # list, tuple, str, range
    Literal,
    TypedDict,
    Generic,
    Hashable as _Hashable,
    Iterable        #? yield
)
from typing_extensions import (
    Self
)

###* Torch ###
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.types import Device, Tensor

#* Figure
from matplotlib.figure import Figure as _Figure
from matplotlib.axes._axes import Axes  # type: ignore

#* Conditional import
# The content of this block is only executed during static type checking (such as mypy) and is ignored at runtime.
# It is used to resolve circular imports or speed up startup.
if TYPE_CHECKING:
    from antenna import AntennaPattern, MultiResponses
    from antenna.smodels import SurrogateModel
    from numpy import ndarray

###* ─────────────────────────────────────────────────────
###* TypeVar：神經網路元件的泛型佔位符
###* 供 Models / SurrogateModel / AdaptiveCyclicalScheduler 等泛型類別的型別參數使用。
###* covariant=True 表示可接受子類別（協變），bound 限定最低父類型。
###* ─────────────────────────────────────────────────────

# 代表任何繼承自 torch.nn.Module 的神經網路模型；
# 用於 Models[CustomModule, ...] 及 MirrorCVAE 等泛型簽章，
# 讓 GEN / SM 等不同網路架構可共用同一套訓練框架。
CustomModule = TypeVar('CustomModule', bound=Module, covariant=True)

# 代表任何繼承自 SurrogateModel 的代理模型（SM）；
# 用於 MirrorCVAE(Generic[CustomSModel])，允許將預先訓練好的 SM 嵌入 GEN 推理流程。
CustomSModel = TypeVar('CustomSModel', bound="SurrogateModel", covariant=True)

# 代表任何繼承自 torch Optimizer 的優化器（如 Adam、SGD）；
# 用於 Models.__init__(optimizer: CustomOptimizer) 及 AdaptiveCyclicalScheduler 泛型。
CustomOptimizer = TypeVar('CustomOptimizer', bound=Optimizer, covariant=True)

# 代表任何繼承自 LRScheduler 的學習率排程器；
# 用於 Models / SurrogateModel 的泛型簽章，為可選元件（Optional[CustomScheduler]）。
CustomScheduler = TypeVar('CustomScheduler', bound=LRScheduler, covariant=True)

###* ─────────────────────────────────────────────────────
###* ParamSpec / TypeVar：模型前向傳播與損失函數的參數規格
###* ─────────────────────────────────────────────────────

# 捕捉模型 forward() 的完整參數規格（位置 + 關鍵字）；
# 用於 Models.__call__ 及 CallableModule.forward 的型別對齊，
# 確保呼叫端傳入的參數與模型簽章一致。
ModelParams = ParamSpec('ModelParams')

# 捕捉損失函數（criterion）的完整參數規格；
# 用於 AntennaResponse.registerLossHook / criterion，
# 讓不同響應指標（S11、增益等）可掛載任意簽章的損失鉤子。
LossParams = ParamSpec('LossParams')

# 代表模型前向傳播的回傳型別；
# 搭配 ModelParams 用於 CallableModule Protocol 及 Models.__call__，
# 確保呼叫端能靜態推斷 forward 的輸出型別。
ReturnType = TypeVar('ReturnType', covariant=True)

###* ─────────────────────────────────────────────────────
###* Protocol：結構性介面定義（鴨子型別）
###* ─────────────────────────────────────────────────────

# 描述「可被當成神經網路模型呼叫」的最小介面；
# 同時要求實作 forward()（PyTorch 風格）與 __call__()（Python 可呼叫物件）。
# 用於 Models.__init__ 的 model 參數，允許傳入不繼承 Module 但符合介面的自定義模組。
class CallableModule(Protocol[ModelParams, ReturnType]):
    def forward(self, *args: ModelParams.args, **kwargs: ModelParams.kwargs) -> ReturnType:
        ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        ...

#* General Callable
#? fn: Callable[CallableParam, CallableReturn]

# 通用可呼叫物件的參數規格，用於非模型場景（如 SPGEN 的 gumbel_fn、size_converter 的鉤子）。
CallableParam = ParamSpec('CallableParam')

# 通用可呼叫物件的回傳型別，與 CallableParam 搭配標註任意函數。
CallableReturn = TypeVar('CallableReturn')

###* ─────────────────────────────────────────────────────
###* TypedDict：固定結構的字典型別定義
###* ─────────────────────────────────────────────────────

#* In antenna.utils.utils.Record
# 描述 Record 物件的序列化快照結構；
# _data 存放當前批次數據，_history 存放歷史累積數據。
# 作為 Checkpoint.record_state_dict 的欄位型別，用於模型存檔與還原訓練記錄。
class RecordStateDict(TypedDict):
    _data: dict[str, list]
    _history: dict[str, list]


# 描述一次完整「生成→模擬→評估」循環的結果快照；
# 同時保存 GEN 生成的 pattern、SM 預測響應（fake）與 HFSS 模擬響應（real），
# 以及對應的損失值與排序鍵，供 MirrorCVAE 多候選排序與最佳解選取使用。
class ResultType(TypedDict):
    """
    Real: The results after simulation

    Fake: The result after model calculation
    """
    pattern: "AntennaPattern"
    real_result: "MultiResponses"
    fake_result: "MultiResponses"
    real_loss: Tensor    #? Simulator Loss: There is not usually a gradient.
    fake_loss: Tensor    #? Model Loss: There is usually a gradient.
    sm_loss: list        #? Surrogate Model Loss: Values ​​will only be available after HFSS simulation.
    time: int
    sort_key: Number    #? results: List[ResultType] = sorted(List[ResultType], key=lambda x: x["sort_key"])
    is_best: bool

#* In antenna.models and antenna.smodels
# 描述模型存檔（.pt 檔）的完整內容結構；
# 用於 Models.checkpoint() 方法的回傳值與 load 邏輯（models.py:173/189），
# 統一儲存網路權重、優化器/排程器狀態、訓練記錄與執行裝置資訊。
class Checkpoint(TypedDict):
    title: str
    model_state_dict: Optional[dict[str, Any]]      # Module
    optimizer_state_dict: Optional[dict[str, Any]]  # Optimizer
    scheduler_state_dict: Optional[dict[str, Any]]  # LRScheduler
    record_state_dict: RecordStateDict
    device: Device

###* ─────────────────────────────────────────────────────
###* TypeVar：資料容器與可雜湊鍵的泛型佔位符
###* ─────────────────────────────────────────────────────

#* In antenna.utils.data.Data
# 代表 Data 泛型容器所裝載的任意資料型別（如 AntennaPattern 陣列、dict 等）；
# 讓 Data[DataType].data 的靜態型別能被呼叫端正確推斷。
DataType = TypeVar('DataType')

# 代表可作為 dict 鍵的可雜湊型別（bound=Hashable 確保支援 __hash__）；
# 用於 make_hashable 函數的型別簽章，讓輸入與輸出型別對應。
Hashable = TypeVar('Hashable', bound=_Hashable)

###* ─────────────────────────────────────────────────────
###* Protocol：形狀查詢介面
###* ─────────────────────────────────────────────────────

#* In antenna.utils.data.size_converter()
# 描述「能回報自身尺寸」的物件介面（如 AntennaPattern、AntennaResponse）；
# size_converter 以此為第一參數，在執行時期動態查詢資料的空間維度，
# 再將輸入 Tensor 重塑為對應形狀（攤平/影像/批次）。
class Sizable(Protocol):
    """
    The object or category must provide a `.size()` method.
    """

    @overload
    def size(self, flatten: Literal[True]) -> int: ...
    @overload
    def size(self, flatten: Literal[False]) -> Tuple[int, ...]: ...
    @overload
    def size(self) -> Tuple[int, ...]: ...

    def size(self, flatten: bool = False) -> Union[int, Tuple[int, ...]]:
        ...

###* ─────────────────────────────────────────────────────
###* TypeAlias：Tensor 形狀語意別名
###* 底層皆為 torch.Tensor；別名僅為靜態分析提供形狀提示，不影響執行期行為。
###* B=Batch, N=攤平特徵數, W=寬, H=高
###* ─────────────────────────────────────────────────────

# 批次 + 攤平形狀 (B, N)；用於 size_converter flatten=True, batch=True 的回傳值，
# 例如送入 SM / GEN 前，用來統一「批次攤平」表示的張量形狀。
Tensor_B_N:TypeAlias = Tensor

# 批次 + 影像形狀 (B, W, H) 或 (B, 1, H, W)；
# 用於 size_converter flatten=False, batch=True 的回傳值，
# 以及需要卷積運算的影像型天線 pattern 表示。
Tensor_B_W_H:TypeAlias = Tensor

# 單樣本攤平形狀 (N,)；
# 用於 size_converter flatten=True, batch=False 的回傳值，
# 以及 AntennaPattern.concat()（__init__.py:198）的回傳型別標註。
Tensor_N:TypeAlias = Tensor

# 單樣本影像形狀 (W, H)；
# 用於 size_converter flatten=False, batch=False 的回傳值，
# 以及 AntennaPattern.stack()（__init__.py:194）的回傳型別標註。
Tensor_W_H:TypeAlias = Tensor

###* ─────────────────────────────────────────────────────
###* TypedDict：天線饋入點可達性評估結果
###* ─────────────────────────────────────────────────────

# 描述單次天線饋入點可達性（FeedReachability）評估的結果字典；
# 用於 FeedReachabilityTracker.record 串列（functions.py:710）與 plot() 方法（functions.py:816），
# 記錄饋入點座標、電流導通率、遮罩矩陣及對應的天線 pattern，供逐 epoch 視覺化分析。
class FeedReachabilityDictType(TypedDict):
    feed_positions: list
    """潰入點"""
    rate:float
    """電流導通率"""
    mask:ndarray
    """電流導通的遮罩"""
    pattern: ndarray
    title: str
