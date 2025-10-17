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

CustomModule = TypeVar('CustomModule', bound=Module, covariant=True)
CustomOptimizer = TypeVar('CustomOptimizer', bound=Optimizer, covariant=True)
CustomScheduler = TypeVar('CustomScheduler', bound=LRScheduler, covariant=True)

ModelParams = ParamSpec('ModelParams')
ReturnType = TypeVar('ReturnType', covariant=True)

class CallableModule(Protocol[ModelParams, ReturnType]):
    def forward(self, *args: ModelParams.args, **kwargs: ModelParams.kwargs) -> ReturnType:
        ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        ...

#* General
CallableParam = ParamSpec('CallableParam')

#* In antenna.models and antenna.smodels
class Checkpoint(TypedDict):
    title: str
    mode_state_dictl: Optional[Module]
    optimizer_state_dict: Optional[Optimizer]
    scheduler_state_dict: Optional[LRScheduler]
    device: Any



#* In antenna.utils.data.Data
DataType = TypeVar('DataType') 
Hashable = TypeVar('Hashable', bound=_Hashable)