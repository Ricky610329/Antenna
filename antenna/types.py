# Defer type annotation evaluation to resolve forward reference and circular import issues.
from __future__ import annotations 

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
    Hashable as _Hashable
)
from typing_extensions import (
    Self
)

###* Torch ###
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.types import Device, Tensor

#* Conditional import
# The content of this block is only executed during static type checking (such as mypy) and is ignored at runtime. 
# It is used to resolve circular imports or speed up startup.
if TYPE_CHECKING:
    from antenna import AntennaPattern, MultiResponses
    from antenna.smodels import SurrogateModel

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

#* General
CallableParam = ParamSpec('CallableParam')

#* In antenna.utils.utils.Record
class RecordStateDict(TypedDict):
    _data: dict[str, list]
    _history: dict[str, list]


class ResultType(TypedDict):
    pattern: "AntennaPattern"
    result:"MultiResponses"
    loss:Tensor
    sm_loss:list
    time:int
    is_best:bool

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

