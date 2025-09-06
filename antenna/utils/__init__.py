from .utils import errorCallback
from .utils import Path
from .utils import plot
from .utils import Config
from .utils import config
from .utils import Figure
from .utils import Axes
from .utils import Record
from .utils import json
from .utils import *

from .web import connect_network_drive
from .web import get_local_ip
from .web import Email

from torch import nn
from torch import Tensor
from .torch_utils import tensor
from .torch_utils import cTensor
from .torch_utils import *

from typing import (
    Tuple, List, Dict, Deque, # Can use the built-in.
    TypeVar, cast, Callable, Any, Optional, overload, Union, Sequence, Literal
)
from typing_extensions import Self

ROOTDIR = Path(r"T:\碩二_吳維文's\Patch Antenna\Experiment")
DATASET_PATH = ROOTDIR.joinpath('dataset')