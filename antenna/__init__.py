"""
It includes a microstrip patch antenna and a reconfigurable intelligent surface (RIS).

Example::

    from antenna import *
    config.device = "cuda:0"

    from antenna.utils import *
    from antenna.models import ...
    from antenna.smodels import ...

    #* Select according to actual application.
    from antenna.ris import ...
    from antenna.patch import ...

    #* Basic Config
    connect_default_drive()
    RESULT_PATH, is_connect_run = get_result_path('[...][{device}] ...', rootdir=ROOTDIR)

    #* Set Antemma Pattern
    AntennaPattern.setDefaultCoordinate((0, n, 0, n))
    PATTERN_SIZE = AntennaPattern.size(flatten=True)
    simulator = ...
    AntennaPattern.register_simulator(simulator)

    #* Set Antenna Response
    AntennaResponse.registerLabels('response', ..., x = '...')
    x = AntennaResponse.x()
    RESPONSE_SIZE = AntennaResponse.size(flatten=True)

"""

import sys
from functools import partial
from os.path import normpath
from time import time

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from loguru import logger

# Re-export 核心類別（從 core/ 拆分出來，保持向後相容）
from antenna.core.pattern import AntennaPattern
from antenna.core.response import AntennaResponse, MultiResponses, TargetResponse
from antenna.types import *
from antenna.utils import *
from antenna.utils.data import size_converter


def get_result_path(
    name: str = "{id}-{device}",
    *,
    rootdir=None,
    set_logger: bool = True,
    generate_code: Optional[str] = None,
    excepthook_mode: Union[bool, Literal["only_hfss"]] = "only_hfss",
    enable_exception_handler: bool = False,
):
    """
    Args:
        name: Folder and log name, support {device}, {hash_id}, {tid}, {id}.
        set_logger: Whether to set the logger.
            EX: XXX.log
        generate_code:
            EX: __file__
        excepthook_mode:
            If True, an email will be sent for any exception; if False, nothing will happen.
            [global_exception_handler(mode)]
        enable_exception_handler:
            config.enable_exception_handler = enable_exception_handler

    Examples: (equivalence)
    ```
    RESULT_PATH, EXISTS = get_result_path()
    RESULT_PATH, CONTINUE_RUN = get_result_path(
        "{device}-{hash_id}", # device, hash_id, tid, id
        rootdir = ROOTDIR, generate_code = __file__, enable_exception_handler = True
    )

    NAME = RESULT_PATH.stem
    ```
    """
    from script.process_files import FileProcessor

    _now = int(time())
    _device = get_local_ip().split(".")[-1]
    _hash_id = get_shake_128(name, length=6)
    rootdir = Path(str(normpath(rootdir))) if rootdir else Path(__file__).parent.parent
    result_path = rootdir.joinpath(
        "result", str(name.format(id=_now, device=_device, tid=TID.generate(), hash_id=_hash_id))
    )
    exists = result_path.exists()
    result_path.not_exist_create()

    if set_logger:
        logger.add(
            result_path.joinpath(f"{result_path.stem}.log"),
            format="{time:YYYY-MM-DD HH:mm:ss} {level} {message}",
            level="INFO",
        )
    if generate_code:
        FileProcessor(
            output_dir=result_path, project_name=result_path.stem, generated_by=generate_code, verbose=False
        ).run()

    config.excepthook = global_exception_handler(excepthook_mode)
    config.enable_exception_handler = enable_exception_handler

    config.NAME = result_path.stem
    config.RESULT_PATH = result_path
    config.ID = _hash_id
    config.CONTINUE_RUN = exists
    config.MAIN_PROGRAM = generate_code

    logger.info(
        f"The results will be saved in {result_path.absolute()} (Continue: {exists}, CUDA: {torch.cuda.is_available()})"
    )
    return result_path, exists


def mult(_ob):
    _result = 1
    for i in _ob:
        _result *= i
    return _result


def reshape(_tensor: torch.Tensor):
    _shape = _tensor.shape
    if len(_shape) == 1:
        return _tensor.reshape(1, _shape[0])
    else:
        return _tensor.reshape(_shape[0], 1)


if __name__ == "__main__":
    config.device = "cpu"
    response = AntennaResponse(torch.randn(361))
    response.plot()
