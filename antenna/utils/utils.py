"""
Re-export hub: imports everything from the split submodules so that
``from antenna.utils.utils import *`` continues to work unchanged.
"""

# --- path ---
# Re-export symbols that the old module-level code made available via its imports
from collections.abc import Callable, Sequence  # noqa: F401
from copy import deepcopy  # noqa: F401
from typing import (  # noqa: F401
    Any,
    Deque,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)

import matplotlib.pyplot as plt  # noqa: F401
import numpy as np  # noqa: F401
import torch  # noqa: F401
from loguru import logger  # noqa: F401
from matplotlib.figure import Figure as _Figure  # noqa: F401
from numpy import ndarray, random  # noqa: F401
from pandas import DataFrame  # noqa: F401
from torch import Tensor, cuda, is_grad_enabled, set_default_device, set_grad_enabled  # noqa: F401
from torch import device as _torch_device  # noqa: F401
from torch import load as _torch_load  # noqa: F401
from torch import manual_seed as _manual_seed  # noqa: F401
from torch import save as _torch_save  # noqa: F401
from tqdm import tqdm, trange  # noqa: F401

# --- config ---
from antenna.utils.config import (  # noqa: F401
    Complete,
    Config,
    MultiConfig,
    config,
    errorCallback,
    global_exception_handler,
)

# --- figure ---
from antenna.utils.figure import (  # noqa: F401
    FIG_CONFIG,
    TQDM_BAR_SIMPLE,
    TQDM_CONFIG,
    Axes,
    Figure,
    plot,
)

# --- hashing ---
from antenna.utils.hashing import TID, get_shake_128  # noqa: F401

# --- json_utils ---
from antenna.utils.json_utils import json  # noqa: F401
from antenna.utils.path import Path  # noqa: F401

# --- record ---
from antenna.utils.record import Record  # noqa: F401

ReturnType = TypeVar("ReturnType")


if __name__ == "__main__":
    # print(Path("./checkpoint").manage_file_count("*.pth", keep_latest=1))
    # print(Path("./checkpoint/GEN_model_0.pth").load_torch())
    config.device = "cpu"
    r = Record("Temp", load=True, rootdir=r"D:\patch_result\1750340068")

    print(r)
    print(r.history)
    # r.save()
    # loss = LossFunction(CustomLoss())
    # for i in range(10):
    #     loss(tensor([i]), tensor([i + i]))
    # loss.plot(show=True)
