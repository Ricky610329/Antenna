"""Antenna 套件頂層命名空間。

整合微帶貼片天線 (microstrip patch antenna) 與可重構智慧表面 (RIS) 的深度學習工具。

使用範例::

    from antenna import *
    config.device = "cuda:0"

    from antenna.utils import *
    from antenna.models import ...
    from antenna.smodels import ...

    # 依實際應用擇一
    from antenna.ris import ...
    from antenna.patch import ...

    # 基本設定
    connect_default_drive()
    RESULT_PATH, is_connect_run = get_result_path('[...][{device}] ...', rootdir=ROOTDIR)

    # 設定天線 Pattern
    AntennaPattern.setDefaultCoordinate((0, n, 0, n))
    PATTERN_SIZE = AntennaPattern.size(flatten=True)
    simulator = ...
    AntennaPattern.register_simulator(simulator)

    # 設定 Antenna Response
    AntennaResponse.registerLabels('response', ..., x = '...')
    x = AntennaResponse.x()
    RESPONSE_SIZE = AntennaResponse.size(flatten=True)
"""

from os.path import normpath
from time import time

import torch
from loguru import logger

# Re-export 核心類別（從 core/ 拆分出來，保持向後相容）
from antenna.core.pattern import AntennaPattern
from antenna.core.response import AntennaResponse, MultiResponses, TargetResponse
from antenna.types import *
from antenna.utils import *


def get_result_path(
    name: str = "{id}-{device}",
    *,
    rootdir=None,
    set_logger: bool = True,
    generate_code: Optional[str] = None,
    excepthook_mode: Union[bool, Literal["only_hfss"]] = "only_hfss",
    enable_exception_handler: bool = False,
):
    """建立結果資料夾並設定 logger / 例外處理。

    Args:
        name: 資料夾與 log 檔名，支援 {device}, {hash_id}, {tid}, {id} 格式標記。
        set_logger: 是否建立對應的 `<name>.log` 檔。
        generate_code: 若提供（通常為 `__file__`），會把原始碼快照存進結果資料夾。
        excepthook_mode: 例外自動通知模式；True 時任何例外皆寄信，False 則不處理。
        enable_exception_handler: 是否啟用 config.enable_exception_handler。

    使用範例::

        RESULT_PATH, EXISTS = get_result_path()
        RESULT_PATH, CONTINUE_RUN = get_result_path(
            "{device}-{hash_id}",
            rootdir=ROOTDIR, generate_code=__file__, enable_exception_handler=True,
        )
        NAME = RESULT_PATH.stem
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
