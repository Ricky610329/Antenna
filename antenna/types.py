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
    TypeVar, 
    TypeVarTuple,
    ParamSpec,
    Callable, 
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

#* General
CallableParam = ParamSpec('CallableParam')

#* In antenna.models and antenna.smodels
class Checkpoint(TypedDict):
    title: str
    mode_state_dictl: Optional[Module]
    optimizer_state_dict: Optional[Optimizer]
    scheduler_state_dict: Optional[LRScheduler]
    device: Any

CustomModule = TypeVar('CustomModule', bound=Module)
CustomOptimizer = TypeVar('CustomOptimizer', bound=Optimizer)
CustomScheduler = TypeVar('CustomScheduler', bound=LRScheduler)

#* In antenna.utils.data.Data
DataType = TypeVar('DataType') 
Hashable = TypeVar('Hashable', bound=_Hashable)