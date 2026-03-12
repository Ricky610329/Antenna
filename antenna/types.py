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

CustomModule = TypeVar('CustomModule', bound=Module, covariant=True)
CustomSModel = TypeVar('CustomSModel', bound="SurrogateModel", covariant=True)
CustomOptimizer = TypeVar('CustomOptimizer', bound=Optimizer, covariant=True)
CustomScheduler = TypeVar('CustomScheduler', bound=LRScheduler, covariant=True)

ModelParams = ParamSpec('ModelParams')
LossParams = ParamSpec('LossParams')
ReturnType = TypeVar('ReturnType', covariant=True)

class CallableModule(Protocol[ModelParams, ReturnType]):
    def forward(self, *args: ModelParams.args, **kwargs: ModelParams.kwargs) -> ReturnType:
        ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        ...

#* General Callable
#? fn: Callable[CallableParam, CallableReturn]
CallableParam = ParamSpec('CallableParam')
CallableReturn = TypeVar('CallableReturn')

#* In antenna.utils.utils.Record
class RecordStateDict(TypedDict):
    _data: dict[str, list]
    _history: dict[str, list]


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
class Checkpoint(TypedDict):
    title: str
    model_state_dict: Optional[dict[str, Any]]      # Module
    optimizer_state_dict: Optional[dict[str, Any]]  # Optimizer
    scheduler_state_dict: Optional[dict[str, Any]]  # LRScheduler
    record_state_dict: RecordStateDict
    device: Device

#* In antenna.utils.data.Data
DataType = TypeVar('DataType') 
Hashable = TypeVar('Hashable', bound=_Hashable)

#* In antenna.utils.data.size_converter()
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

Tensor_B_N:TypeAlias = Tensor
Tensor_B_W_H:TypeAlias = Tensor
Tensor_N:TypeAlias = Tensor
Tensor_W_H:TypeAlias = Tensor

class FeedReachabilityDictType(TypedDict):
    feed_positions: list
    """潰入點"""
    rate:float
    """電流導通率"""
    mask:ndarray
    """電流導通的遮罩"""
    pattern: ndarray
    title: str