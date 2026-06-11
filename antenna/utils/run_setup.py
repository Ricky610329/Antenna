"""
antenna/utils/run_setup.py — 實驗 run 的 production 殼層設定。

get_result_path：建結果夾 (含續跑偵測)、掛檔案 log、複製執行程式碼快照、
設定例外處理 (HFSS 錯誤可寄信)。從 antenna/__init__.py 拆出 (純搬家)——
它是「殼層」職責，不屬於資料抽象核心。
"""
from os.path import normpath
from time import time
from typing import Literal, Optional, Union

import torch
from loguru import logger

from .utils import TID, Path, config, get_shake_128, global_exception_handler
from .web import get_local_ip

def get_result_path(
    name:str = "{id}-{device}", *, 
    rootdir = None, 
    set_logger:bool = True, 
    generate_code:Optional[str] = None,
    excepthook_mode:Union[bool, Literal['only_hfss']] = 'only_hfss',
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
    #* 組出本次實驗的唯一識別資訊, 供資料夾/log 命名與斷點續跑判斷
    _now = int(time())                              #? 秒級 timestamp, 對應 name 中的 {id}
    _device = get_local_ip().split('.')[-1]         #? 取本機 IP 末段當作「機器代號」, 對應 {device}
    _hash_id = get_shake_128(name, length=6)        #? 由 name 雜湊出短碼, 對應 {hash_id} (相同 name 會得到相同碼)
    rootdir = Path(str(normpath(rootdir))) if rootdir else  Path(__file__).parent.parent
    result_path = rootdir.joinpath(
        "result", str(name.format(id = _now, device = _device, tid = TID.generate(), hash_id = _hash_id))
    )
    #! exists 用來判斷是否「續跑」: 若資料夾已存在代表先前已建立過同名實驗
    exists  = result_path.exists()
    result_path.not_exist_create()                  #? 不存在才建立, 避免覆蓋既有結果

    if set_logger:
        logger.add(
            result_path.joinpath(f"{result_path.stem}.log"),
            format = "{time:YYYY-MM-DD HH:mm:ss} {level} {message}",
            level = "INFO",
        )
    if generate_code:
        FileProcessor(
            output_dir = result_path,
            project_name=result_path.stem,
            generated_by=generate_code,
            verbose = False
        ).run()

    # from .utils.utils import global_exception_handler
    #* 把實驗層級的狀態寫回全域 config, 讓 pipeline 各處 (GEN/SM/SIM/trainer) 都能讀到同一份設定
    config.excepthook = global_exception_handler(excepthook_mode)  #? HFSS 例外時可寄信通知 (only_hfss)
    config.enable_exception_handler = enable_exception_handler

    config.NAME = result_path.stem
    config.RESULT_PATH = result_path
    config.ID = _hash_id
    config.CONTINUE_RUN = exists            #? 供主程式判斷要不要載入 checkpoint 接續訓練
    config.MAIN_PROGRAM = generate_code
    
    logger.info(f"The results will be saved in {result_path.absolute()} (Continue: {exists}, CUDA: {torch.cuda.is_available()})")
    return result_path, exists

