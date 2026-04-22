from typing import Any

import numpy as np
from numpy import ndarray, random
from torch import (
    Tensor,
    __version__,
    concat,
    cuda,
    float64,
    is_grad_enabled,  # with no_grad():...
    nn,
    # get_default_device,
    set_default_device,
    set_grad_enabled,
    stack,
)
from torch import (
    device as _torch_device,
)
from torch import (
    load as _torch_load,
)
from torch import (
    manual_seed as _manual_seed,
)
from torch import (
    save as _torch_save,
)
from torch import (
    tensor as _tensor,
)

from antenna.utils.config import config

try:
    from torch.utils.tensorboard import SummaryWriter  # type:ignore pip install tensorboard

    def getTensorBoardWriter(log_dir: str = "./runs") -> SummaryWriter:
        """

        ## Usage
        ```bash
        tensorboard --logdir=runs
        ```

        ## Example
        ```
        tbwriter = getTensorBoardWriter()
        for n_iter in range(100):
            tbwriter.add_scalar('Loss/train', np.random.random(), n_iter)
            tbwriter.add_scalar('Loss/test', np.random.random(), n_iter)
            tbwriter.add_scalar('Accuracy/train', np.random.random(), n_iter)
            tbwriter.add_scalar('Accuracy/test', np.random.random(), n_iter)
        ```
        """
        return SummaryWriter(log_dir)
except ModuleNotFoundError:
    pass


def tensor(data: Any, dtype=None, device=None, requires_grad: bool = False):
    return _tensor(data, dtype=dtype, device=device or config.device, requires_grad=requires_grad)


def cTensor(data: Any, requires_calculate: bool, *, device=None, dtype=None):
    """
    Creating Tensors.

    Args:
        requires_calculate (bool):
            - If True: device=device or config.device, requires_grad=True
            - If Fasle: device='cpu', requires_grad=False
        device:
            Not applicable when requires_calculate is Fasle.

    Example:
        ```
        cTensor([1, 2, 3], requires_calculate=True)     # tensor([1., 2., 3.], dtype=torch.float64, requires_grad=True)
        cTensor([1, 2, 3], requires_calculate=False)    # tensor([1, 2, 3])
        ```

    """
    t = _tensor(data)
    if requires_calculate:
        t = t if t.dtype.is_floating_point else t.type(dtype or float64)
        t = t.to(device or config.device)
        t = t.requires_grad_(True)
    else:
        t = t.cpu()
    return t
