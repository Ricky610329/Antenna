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

from .data import size_converter

from torch import nn
from torch import Tensor
from .torch_utils import tensor
from .torch_utils import cTensor
from .torch_utils import *

from antenna.types import *

ROOTDIR = Path(r"T:\碩二_吳維文's\Patch Antenna\Experiment")
DATASET_PATH = ROOTDIR.joinpath('dataset')