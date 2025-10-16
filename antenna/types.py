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
    Callable, 
    Any, 
    Optional, 
    Union, 
    Sequence, 
    Literal,
    TypedDict,
    Generic
)
from typing_extensions import (
    Self
)

###* Torch ###
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class Checkpoint(TypedDict):
    title: str
    mode_state_dictl: Optional[Module]
    optimizer_state_dict: Optional[Optimizer]
    scheduler_state_dict: Optional[LRScheduler]
    device: Any

CustomModule = TypeVar('CustomModule', bound=Module)
CustomOptimizer = TypeVar('CustomOptimizer', bound=Optimizer)
CustomScheduler = TypeVar('CustomScheduler', bound=LRScheduler)