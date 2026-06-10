# ==============================================================================
# utils.py — 反向設計閉迴路系統的「工具核心」
# ------------------------------------------------------------------------------
# 本檔是整個 pipeline (GEN/SM/SIM 三角色、train_single.py / train_dual.py 訓練腳本)
# 共用的基礎設施。重點元件一覽：
#   - errorCallback / global_exception_handler：例外處理；HFSS COM 錯誤會自動寄 email。
#   - Complete：訓練完成通知 (可寄信)。
#   - Path：pathlib 擴充 (自動建目錄、舊檔輪替、load_torch 載入 checkpoint 等)。
#   - Config + 全域 config：dict 子類，集中管理 device / checkpoint 路徑 / 例外處理開關等。
#   - Figure：matplotlib 的 with-context 包裝 (可存 GIF/MP4)。
#   - Record (★)：訓練狀態的「時序記錄器」，也就是 run_training 裡的 TEMP；
#                 是斷點續跑與 rollback 的核心。
#   - json / TID / get_shake_128：JSON 點路徑存取、時間 ID、短雜湊 ID。
# ==============================================================================
from typing import (
    Tuple, List, Dict, Deque, # Can use the built-in.
    TypeVar, cast, Callable, Any, Optional, overload, Union, Sequence, Literal
)
from loguru import logger  # 專案統一使用 loguru 而非標準 logging
import traceback
from types import TracebackType
import torch
from torch import (
    __version__,
    Tensor,
    cuda,
    manual_seed as _manual_seed,
    load as _torch_load,
    save as _torch_save,
    device as _torch_device,
    # get_default_device,
    set_default_device,

    set_grad_enabled, is_grad_enabled # with no_grad():... 用來在 Figure context 內暫時關閉/開啟梯度
)
from numpy import (
    ndarray,
    random
)
from  pickle import (
    dump as _pickle_dump,
    load as _pickle_load
)
from json import (
    load as _json_load, 
    dump as _json_dump
)
from pandas import DataFrame
from tqdm import trange
from collections import defaultdict
from warnings import filterwarnings

from pathlib import Path as _Path
from os.path import getctime, exists

import sys

import numpy as np
from copy import deepcopy
from datetime import datetime
from shutil import rmtree as _rmtree
from time import time

#* Figure
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.figure import Figure as _Figure
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.axes._axes import Axes  # type: ignore
from tqdm import tqdm

# 泛型回傳型別：給 save(update_hook=...)、Figure.convert_to、Record.custom 等
# 「呼叫使用者函式並原樣回傳其結果」的方法做型別標註，讓 IDE 能推斷回傳型別。
ReturnType = TypeVar('ReturnType')

# matplotlib 存圖的預設參數：300 dpi、去白邊、透明背景 (方便貼進報告/論文)。
FIG_CONFIG = {
    "format": 'png',
    "bbox_inches": "tight",
    "pad_inches": 0.1,
    "dpi": 300,
    "transparent": True,
    "facecolor": "none", # white
    "edgecolor": "none",
}
# tqdm 進度條預設：以 epoch 為單位，最少每 1 秒刷新一次 (避免頻繁重繪拖慢訓練)。
TQDM_CONFIG = {
    'unit': 'epoch',
    'unit_scale': True,
    'mininterval': 1.0,
    'dynamic_ncols': True
}
# 精簡版進度條格式：左側標籤 + bar + 進度數字 + postfix (可掛即時 loss 等資訊)。
TQDM_BAR_SIMPLE = '{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}'

def errorCallback(errorCallback:Optional[Callable[[str],Any]]=None, *errorCallbackArgs, **errorCallbackKwargs):
    """
    Error callback function.

    ## Usage
    ```python
    @errorCallback()
    def func():
        raise Exception("Error")
    ```
    """
    # 裝飾器工廠：用 errorCallback() 包住函式後，函式內任何例外都會被攔下，
    # 不讓單一步驟的錯誤中斷整個訓練流程 (例如某個繪圖/存檔失敗時仍能繼續跑)。
    def decorator(func:Callable):
        def wrap(*args, **kwargs):
            try:
                return func(*args, **kwargs)   # print(func.__name__)
            except Exception as e:
                # 有指定 callback 就把錯誤訊息字串交給它處理 (例如寄信/通知)，
                # 否則只用 logger 記下完整 traceback，函式回傳 None 繼續執行。
                if errorCallback:
                    errorCallback(str(e), *errorCallbackArgs, **errorCallbackKwargs)
                else:
                    logger.exception(e)
        return wrap
    return decorator

def global_exception_handler(mode:Union[bool, Literal["only_hfss"]] = True) -> Callable[[type[BaseException], BaseException, TracebackType], None]:
    """
    建立一個可指派給 `sys.excepthook` 的「全域未捕捉例外處理器」。

    為什麼需要：HFSS 模擬透過 COM 跑在 Windows 端，整段訓練常常一跑就是幾小時，
    若在無人看顧時崩潰 (尤其是 HFSS COM 端的 com_error)，需要「主動把錯誤連同
    完整 traceback 寄 email」通知，而不是讓行程默默死掉。

    :param mode:
        - True：任何未捕捉例外都記 log 並寄信。
        - "only_hfss"：所有例外都記 log，但「只有 HFSS COM 錯誤」才寄信
          (避免一般程式 bug 也洗信箱)。
        - False：不做任何加工，直接回傳 Python 預設的 excepthook。

    透過 Config.enable_exception_handler = True 來把回傳的 hook 掛上 sys.excepthook。
    """
    # 嘗試取得 HFSS 的 COM 例外類別；若環境裡沒有 (例如純 CPU 測試環境)，
    # 退化為 None，後續就不會把任何例外判定成 HFSS 錯誤。
    try:
        from antenna.patch import com_error  # type: ignore
    except Exception:
        com_error = None  # type: ignore

    original_hook = sys.__excepthook__  # 保留 Python 原生 hook 當後路

    # mode=False：使用者明確關閉，直接還原成預設行為。
    if mode == False:
        return original_hook

    def excepthook(exc_type:type[BaseException], exc_value: BaseException, exc_traceback:TracebackType):
        """
        Extract global variables from the logger.
        ```
        import sys
        sys.excepthook = global_exception_handler
        ```
        """

        # 先把完整 traceback 轉成字串，待會要塞進 email 內文方便遠端排查。
        # Full traceback string
        tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))

        # 不論是否寄信，都先用 loguru 完整記錄這個例外 (含 exc_info)。
        # Log the exception with full exc_info
        logger.opt(
            exception = (exc_type, exc_value, exc_traceback)
        ).error(
            f"[{exc_type.__name__}] {exc_value}",
            exc_info=(exc_type, exc_value, exc_traceback)
        )

        # 判斷是否為 HFSS COM 錯誤，並依 mode 決定要不要寄信。
        is_com_error = (com_error is not None) and issubclass(exc_type, com_error)
        send_email = (mode is True) or (mode == "only_hfss" and is_com_error)
        if issubclass(exc_type, KeyboardInterrupt):
            # 使用者按 Ctrl+C 主動中止：這是正常停止，不算錯誤、不寄信，直接返回。
            logger.info("Ctrl + C: Manually stop program execution.")
            # original_hook(exc_type, exc_value, exc_traceback)
            return
        elif is_com_error:
            # HFSS COM 錯誤：盡量抽出 HRESULT 與 strerror 組成易讀訊息 (利於對照 Ansys 文件)。
            hresult = getattr(exc_value, "hresult", "N/A")
            strerror = getattr(exc_value, "strerror", str(exc_value))
            text = f"HFSS {exc_type.__name__}: HRESULT={hresult} — {strerror}"
        else:
            # 一般 Python 例外。
            text = f"{exc_type.__name__}: {exc_value}"

        if send_email:
            # 延後匯入 (避免在 import utils 時就拉進網路/SMTP 相依)；get_local_ip 讓信件
            # 標題帶上是哪台機器出錯 (實驗室常有多台機跑模擬)。
            from antenna.utils.web import Email, get_local_ip
            with Email("weiwen@alum.ccu.edu.tw") as email:
                msg = email.getText(
                    f'Antanna Error ({get_local_ip()})',
                    f"{text}\n\nTraceback:\n{tb_text}"
                )

                # SMTP 慣例：sendmail 回傳空 dict {} 表示所有收件者都成功。
                status = email.sendMessage(msg.as_string())

                if status == {}:
                    logger.success("Email sent successfully!")
                else:
                    logger.error('Email send failed!')
        # original_hook(exc_type, exc_value, exc_traceback)
    return excepthook  # 回傳閉包，由呼叫端掛到 sys.excepthook

def Complete(message="Process completed.", send_email:bool=False, **results):
    """
    訓練/流程「完成通知」。把結果摘要 (任意關鍵字參數) 整理成多行文字後記 log，
    並可選擇寄一封成功通知信 — 對照 global_exception_handler 的「失敗通知」。

    train_single/dual 末尾即以此回報，例如帶上 Min Loss、整份 config 等。

    Example ::

        _dict = {}
        Complete(
            "Training Finished!", send_email=True,
            **_dict, **{"Min Loss": ...}, a='a'
        )
    """
    full_message = f"Completed: {message}"
    # 把所有額外傳入的具名結果逐行展開成 "key: value" 附在訊息後面。
    if results and isinstance(results, dict):
        result_lines = [f"{key}: {value}" for key, value in results.items()]
        formatted_result = "\n".join(result_lines)

        full_message += f"\n\n{formatted_result}"

    if send_email:
        # 與例外處理器相同：延後匯入 Email，標題帶上本機 IP 以區分機器。
        from antenna.utils.web import Email, get_local_ip
        with Email("weiwen@alum.ccu.edu.tw") as email:
            msg = email.getText(
                f'Antanna Success ({get_local_ip()})',
                full_message
            )

            status = email.sendMessage(msg.as_string())  # 空 dict 代表寄送成功

            if status == {}:
                logger.success("Email sent successfully!")
            else:
                logger.error('Email send failed!')

    logger.success(full_message)  # 不論是否寄信，都在本機 log 留一筆成功訊息

class Path(type(_Path())): # type: ignore
    # 繼承「pathlib 在當前 OS 上的具體類別」(type(_Path()) 在 Windows 為 WindowsPath、
    # 在 POSIX 為 PosixPath)，所以這個 Path 仍是真正的 pathlib 路徑、可直接用所有原生方法，
    # 只是額外掛上 not_exist_create / manage_file_count / load_torch 等便利工具。
    def __new__(cls, *args, **kwargs):
        # pathlib 的 __new__ 不認得自訂的 create 參數，先在這裡剔除避免 TypeError；
        # 實際的 create 行為留到 __init__ 處理。
        kwargs.pop('create', None)
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, *args, create:bool=False, **kwargs):
        """
        Path model.
       
        ## Usage
        ```python
        path = Path("./path/to/file.ext")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
        path.unlink()
        path.del_from_glob('*.pth')
        path.manage_file_count('*.pth', keep_latest=3)
        path.load_torch()
        path.not_exist_create(create_file=True)
        ```
        """
        # self.path = path

        # create=True 時，建構物件當下就把對應目錄/檔案準備好，省去呼叫端再手動 mkdir。
        if create: self.not_exist_create()

    def __reduce__(self):
        # 讓自訂 Path 能被 pickle：還原時只用字串路徑重建 (不帶 create 等暫態旗標)。
        # 重要 — Record 會把含路徑的狀態 pickle 存檔，少了這個會序列化失敗。
        return (self.__class__, (str(self),))

    def rmtree(self) -> bool:
        # 若自身是目錄就遞迴刪除整棵樹並回傳 True；不是目錄則回傳 False (交給呼叫端改用 unlink)。
        if self.is_dir():
            _rmtree(self)
            return True
        else:
            return False
    
    def not_exist_create(self, create_file:bool = False):
        """
        Create the path if it does not exist.

        :param create_file: Whether to create the file.
        :return: Whether the path does not exist.
        """
        # 以「有無副檔名」來推斷這個路徑是檔案還是目錄：
        if self.suffix:
            # 有副檔名 → 視為檔案：先確保父目錄存在，必要時再建立空檔。
            self.parent.mkdir(parents=True, exist_ok=True)
            if create_file: self.touch(exist_ok=True)
        else:  # No file extension, treated as a directory.
            # 無副檔名 → 視為目錄：直接遞迴建立。
            self.mkdir(parents=True, exist_ok=True)
        return self  # 回傳 self 以便鏈式呼叫，如 Path(...).not_exist_create().joinpath(...)

    def del_from_glob(self, pattern:str):
        """
        Delete all files matching the pattern.

        :param pattern: Patterns matching files, E.g., '*.pth'
        """
        # 無副檔名 → 自身是目錄：刪掉目錄下所有符合 pattern 的檔案。
        if not self.suffix:
            paths = list(self.glob(pattern))
            for path in paths:
                path.unlink()
        else:
            # 有副檔名 → 自身就是檔案：直接刪除 (此時 pattern 被忽略)。
            self.unlink()

    def manage_file_count(self, file:str, keep_latest:Optional[int] = 3):
        """
        Manage the number of archives and only keep the latest specified number.

        :param file: Patterns matching archives, E.g., '*.pth'
        :param keep_latest: Latest quantity to keep.
        """
        # 訓練常每 epoch 存一份 checkpoint，這個方法用來「只保留最新 N 份」避免硬碟爆掉。
        # keep_latest=None → 視為不限制，直接放行不刪。
        if keep_latest is None:
            return False

        # Confirm that the target directory exists.
        if not self.exists():
            raise FileNotFoundError(f"The destination directory ({self.absolute()}) does not exist.")

        # 依「建立時間 (ctime)」由舊到新排序所有符合 pattern 的檔案。
        # Get all files matching the pattern.
        files_sorted = sorted(self.glob(file), key=getctime)

        # 超量時，刪掉最前面 (最舊) 的那幾份，保留尾端最新的 keep_latest 份。
        # If the file exceeds the limit, delete the oldest file.
        if len(files_sorted) > keep_latest:
            for old_backup in files_sorted[:len(files_sorted)-keep_latest]:
                # 先試著當目錄刪 (rmtree)；若不是目錄 (回傳 False) 再當檔案 unlink。
                if not old_backup.rmtree():
                    old_backup.unlink()
            return True
        else:
            return False
    
    def load_torch(self, device = None):
        """以此路徑載入 torch checkpoint，預設搬到全域 config.device。"""
        from antenna.models import config
        # PyTorch 2.6 起 torch.load 預設 weights_only=True，會擋下含自訂類別 (如本專案的
        # AntennaPattern / Path) 的 checkpoint；這裡偵測版本後顯式關掉，確保舊存檔能載回。
        if __version__ >= "2.6.0":
            return _torch_load(self, weights_only=False, map_location=device or config.device)
        else:
            return _torch_load(self, map_location=device or config.device)

def plot(x,file_name:Optional[str] = None) -> None:
        """
        Plot the weight matrix on a 3D graph
        """
        # 簡易折線繪圖小工具 (debug 用)：清掉當前 figure → 畫 x → 顯示，
        # 傳了 file_name 才順手存檔。正式訓練繪圖請改用下方 Figure context manager。
        # This part is for plotting the graph
        plt.clf()
        # plt.figure(figsize=(20, 10))
        plt.title(f'')
        plt.plot(x)
        plt.legend()

        plt.show()

        if file_name: plt.savefig(file_name, **FIG_CONFIG)

class Config(dict):
    """
    全域設定容器 (dict 子類)。同時支援屬性存取與字典存取：
    `config.device` 與 `config['device']` 等價 (見 __getattr__/__setattr__)。

    設計重點：
      - 繼承 dict → 可直接 `**config` 展開傳入 Complete()、可被 JSON 序列化。
      - device / checkpoint_save_path / enable_exception_handler 用 property 包裝，
        在賦值的同時做副作用 (設定 torch 預設裝置、掛 excepthook 等)。
      - save()/load() 只保存可 JSON 化的型別，方便每次實驗把設定快照存進結果資料夾。
    全域單例為檔案最後建立的 `config`，整個專案共用同一份。
    """
    NAME:str = None
    """Project name."""
    MAIN_PROGRAM = None
    """Executed by this file."""
    EPOCHS:int = None
    """ """
    LR:float = None
    """Learning Rate For Main Training Loop."""

    RESULT_PATH:Path = None
    CONTINUE_RUN:bool = None

    excepthook: Callable[[type[BaseException], BaseException, TracebackType | None], Any]
    """
    sys.excepthook = excepthook

    Use ::
        
        Config.enable_exception_handler = True
    
    Example (excepthook) ::
    
        def global_exception_handler(
            exc_type:type[BaseException] | None, 
            exc_value: BaseException | None, 
            exc_traceback
        ):...
        Config.excepthook = global_exception_handler
    """

    def __init__(self):
        # 預設值。注意這裡用屬性賦值 (self.epochs=...) 經 __setattr__ 轉存進 dict，
        # 之後即可同時用 config.epochs 或 config['epochs'] 取得。
        self.ID:str = str(int(time()))  # 以建立當下的 Unix 秒數當作此次執行的預設 ID
        self.epochs = 10
        self.lr = 1e-3 # Learning Rate For Main Training Loop
        self.element_num = 40
        self['checkpoint_save_path'] = Path("./checkpoint")
        self['device'] = _torch_device(type='cpu')  # 預設 CPU；之後常被腳本改成實際裝置

        # 預先準備好例外處理 hook (預設模式)，但尚未掛上 sys.excepthook；
        # 要等 enable_exception_handler = True 才真正生效。
        self.excepthook = global_exception_handler()

    def __getattr__(self, name):
        # 該Class沒有此變數(name)會執行。
        # 找不到的屬性一律回退到字典查找，達成「屬性 == 鍵」的存取體驗 (查無則 KeyError)。
        return self[name]

    def __setattr__(self, name, value):
        cls_attr = getattr(type(self), name, None)

        # 檢查是否是 property 實例，或者是一個方法
        # 關鍵分流：若 name 對應到 property/方法 (如 device、checkpoint_save_path)，
        # 走 object.__setattr__ 以正確觸發 setter 的副作用；否則只是普通設定值，存進 dict。
        if isinstance(cls_attr, (property, type(lambda:0))):
            object.__setattr__(self, name, value)
        else:
            self[name] = value

    def check_keys(self, *keys:str, only_warning:bool = False ):
        # 前置檢查：確認必要的設定鍵都已就緒。only_warning=True 只警告不中斷，
        # 否則缺鍵就丟 KeyError，及早讓使用者補上而非跑到一半才爆。
        for key in keys:
            if key not in self:
                if only_warning:
                    logger.warning(f'{key} not set.')
                else:
                    raise KeyError(f'{key} not set.')
    @property
    def enable_exception_handler(self) -> bool:
        """Whether exception handling is enabled or not."""
        # 以「目前 sys.excepthook 是否還是 Python 原生 hook」反推是否已啟用自訂處理器。
        if sys.excepthook is sys.__excepthook__:
            return False
        else:
            return True
    
    @enable_exception_handler.setter
    def enable_exception_handler(self, mode:bool):
        # 開 → 掛上 __init__ 準備好的 self.excepthook (HFSS 錯誤會寄信)；關 → 還原原生 hook。
        sys.excepthook = self.excepthook if mode else sys.__excepthook__

    @property
    def device(self):
        return self['device']
    
    @device.setter
    def device(self, device):
        # 賦值 device 不只是存值，還連帶設定 torch 全域行為：
        # 傳 None/空 → 自動選 cuda:0 (有 GPU) 或 cpu。
        device = device or _torch_device("cuda:0" if cuda.is_available() else "cpu")
        set_default_device(device)        # 之後新建張量預設落在此裝置 (省去到處 .to(device))
        if device != "cpu":
            cuda.set_device(device)       # 多卡時固定目前 CUDA 裝置
        self['device'] = device
    
    @property
    def checkpoint_save_path(self) -> Path:
        """
        Default: ./checkpoint
        ```
        config.checkpoint_save_path.not_exist_create()
        ```
        """
        # 取 checkpoint 路徑；若尚未設定就退而求其次用 ./checkpoint 並順手建好目錄。
        _checkpoint_save_path:Path = self.get(
            'checkpoint_save_path',
            Path('checkpoint').not_exist_create()
        )
        return _checkpoint_save_path.absolute()  # 一律回傳絕對路徑，避免 cwd 變動造成存錯位置

    @checkpoint_save_path.setter
    def checkpoint_save_path(self, path):
        # 一律包成自訂 Path，讓後續可用 not_exist_create / manage_file_count 等擴充方法。
        self['checkpoint_save_path'] = Path(path)

    def setRandomSeeds(self, seed = 0):
        # 一次固定 torch(CPU/GPU) 與 numpy 三處亂數種子，確保實驗可重現。
        _manual_seed(seed)
        cuda.manual_seed(seed)
        random.seed(seed)
    
    def setWarning(self, warning_type:str = "ignore"):
        # 統一控制 warnings 過濾；訓練時常設 "ignore" 壓掉第三方套件的雜訊警告。
        return filterwarnings(warning_type) # type: ignore
    
    @overload
    def save(self, name:str = 'config', rootdir:Optional[str] = None, *, update_hook:Callable[[dict], ReturnType]) -> ReturnType:...
    @overload
    def save(self, name:str = 'config', rootdir:Optional[str] = None) -> None:...

    def save(self, name:str = 'config', rootdir:Optional[str] = None, *, update_hook:Optional[Callable[[dict], ReturnType]]=None):
        """
        Only save the following types
        ```
        dict, list, tuple, str, int, float, bool, None
        ```
        If it is other, it will be automatically converted to a string using `str()`

        :param update_hook: def update_hook(config)..., Ex: wandb.config.update
        """
        # 把當次實驗的設定快照存成 JSON (預設 ./config.json)，方便日後追溯/續跑核對。
        path = Path(rootdir or "./", f"{name}.json")
        _save = {}
        self.update(vars(self))  # 把以 object 屬性形式存的欄位 (如 ID) 也併進 dict 一起存
        for key, value in self.items():
            # 可 JSON 化的型別原樣保留；其餘 (如 device、Path 物件) 一律 str() 化，
            # 保證不會因不可序列化而存檔失敗 (代價是載回後是字串而非原物件)。
            if isinstance(value, (dict, list, tuple, str, int, float, bool)) or value is None:
                _save[key] = value
            else:
                _save[key] = str(value)
        with open(path,'w', encoding='utf-8') as f:
            _json_dump(_save, f, indent = 4, ensure_ascii = False)  # 中文不轉義，方便人讀

        # 可選的 update_hook：把設定同步給外部追蹤工具，例如 wandb.config.update。
        if update_hook: return update_hook(self)

    def load(self, name:str = 'config', rootdir:Optional[str] = None):
        """
        Only load the following types
        ```
        dict, list, tuple, str, int, float, bool, None
        ```
        """
        # TODO
        # 從 JSON 載回設定並 update 進自身 (注意：device/Path 等存檔時被字串化，載回仍是字串)。
        path = Path(rootdir or "./", f"{name}.json")
        with open(path, 'r', encoding = 'utf-8') as f:
            self.update(_json_load(f))

    def __str__(self):
        # 把所有設定鍵值串成 "Config(k=v, ...)" 方便 log 時一眼看完整組設定。
        _str = ", ".join(f"{k}={v}" for k, v in self.items())
        return f"{self.__class__.__name__}({_str})"

# 全域單例設定：整個專案 (各模組、訓練腳本) 透過 import 共用這一份 config。
config = Config()

class Figure:
    """
    matplotlib Figure 的 with-context 包裝。

    解決兩個訓練常見痛點：
      1. 樣板繁瑣：自動處理 figure 尺寸、字型大小、子圖網格 (index/addAll)，
         離開 with 區塊時依 save/show 自動存圖或顯示、並 plt.close() 釋放記憶體
         (長訓練若不關 figure 會記憶體洩漏)。
      2. 梯度誤算：進入 context 時依 requires_grad 暫停 autograd，離開時還原；
         避免「畫圖時順手做的張量運算」誤入計算圖、污染 GEN/SM 的梯度。

    另提供 saveGIF / saveMP4 把每個 epoch 的圖串成動畫，呈現訓練演進。
    """
    def __init__(
            self,
            name:str,
            nrowcol:tuple = (1, 1),
            ncols:tuple = (0, 0),
            save:bool = False, show:bool = False,
            rootdir:Optional[str] = None,
            size:tuple = (18, 12),
            default_font_size = 12,
            default_axes_title_size = 20,
            default_tick_size:int = 18,
            requires_grad:bool = False,
            **kwargs
        ):
        """
        :param size: Example: (18, 12), (18 * 2, 9 * 2)
        :param kwargs: All plt.figure() arguments
        :param ncols: (total, cols) -> nrowcol=(total/cols, cols)

        ## Example
        ```
        with Figure("test_3_2", nrowcol=(2,2), save=True) as fig:
    
            ax1 = fig.index(1)
            ax1.set_title("test")
            ax1.plot([1, 2, 3, 4])

            fig.addAll()
            fig[2].set_title("test")
            fig[2].plot([1, 2, 3, 4])
        ```
        
        ## Set
        ```
        class:
            ...
            def plot(self, axes:Axes|None = None):
                ax:Axes = plt.axes(axes) # type: ignore
                ax.set_title("test")
                ax.plot([1, 2, 3, 4])
        ```

        ## 動畫
        ```
        line = {}
        epochs = 1500
        line = np.random.random((epochs))
            
        with Figure("line", rootdir=r'./') as fig:
            fig.addAll()
            def update(frame):
                fig[0].clear()
                fig[0].set_title("line")
                fig[0].set_xlim(0, epochs)
                fig[0].plot(line[:frame+1])

                fig.fig.tight_layout(pad=0.1)
                
                return fig
            fig.saveMP4(update, epochs, video_time=5)
        ```
        """
        from math import ceil
        fig = plt.figure(name, **kwargs)  # 以 name 當 figure 識別字 (相同 name 會復用同一張)
        fig.set_size_inches(*size)
        fig.tight_layout(pad=0.1)
        # 統一字型/標題/刻度大小，讓多張輸出圖風格一致 (便於並排比較或放進論文)。
        plt.rcParams.update({
            'font.size': default_font_size,
            'axes.titlesize': default_axes_title_size,
            'xtick.labelsize': default_tick_size,
            'ytick.labelsize': default_tick_size,
            'axes.labelsize': default_tick_size,
        })
        # fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)

        self.fig = fig
        self.save = save                 # 離開 context 時是否存圖
        self.show = show                 # 離開 context 時是否 plt.show()
        self.name = name
        # 子圖網格：若用 ncols=(總數, 欄數) 指定，則自動換算列數 (ceil)；否則直接用 nrowcol。
        self.nrowcol = (ceil(ncols[0] / ncols[1]), ncols[1]) if ncols[0] > 0 else nrowcol
        self.current_index = 1           # index() 自動遞增的子圖游標
        self.rootdir = Path(rootdir or "./")  # 圖檔輸出目錄
        self.requires_grad = requires_grad    # context 內是否保留梯度 (預設關閉)

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, nrowcol={self.nrowcol}, save={self.save}, show={self.show}, rootdir={self.rootdir.absolute()}, size={self.fig.get_size_inches()})"


    def index(self, index:int = 1, title:Optional[str] = None):
        """
        :param index: Support -1
        """
        # 在網格上取得/新增一個子圖 Axes 並回傳。傳 index=-1 表「沿用目前游標」，
        # 因此 train_single/dual 內常連續呼叫 fig.index(-1) 一格一格往下擺子圖。
        self.current_index = self.current_index if index == -1 else index
        ax = self.fig.add_subplot(self.nrowcol[0], self.nrowcol[1], self.current_index)
        self.current_index += 1  # 取完即前進，下一次 -1 會落到下一格

        if title is not None:
            ax.set_title(title)
        return ax

    def addAll(self):
        # 一次把整個網格 (nrow*ncol) 的子圖全部建出來，之後即可用 fig[i] 索引存取。
        for i in range(self.__len__()) :
            self.index(i+1)
            
    def convert_to(self, fn:Callable[[_Figure], ReturnType]) -> ReturnType:
        """
        Convert to the specified type.

        :param fn: Convert function. Ex: wandb.Image

        Example::

            fig.conver_to(wandb.Image)
        """
        # 把內部 matplotlib Figure 交給轉換函式 (例如 wandb.Image) 並回傳其結果，
        # 方便把圖直接上傳到實驗追蹤平台。
        return fn(self.fig)

    def saveGIF(self, update:Callable, epochs:int = 10, dpi = 150):
        # 用 PillowWriter 把 update(frame) 逐格畫出的內容串成 GIF，存到 rootdir/name.gif。
        # progress_callback 掛 tqdm，讓動畫輸出也有進度條。
        writer = PillowWriter(fps=30, metadata={"artist": "WeiWen Wu"})
        tqdm_iter = trange(epochs, desc="Plotting")
        ani = FuncAnimation(self.fig, update, frames=epochs)
        ani.save(f"{self.rootdir.joinpath(self.name)}.gif", writer=writer, dpi=dpi, progress_callback=lambda i, n: tqdm_iter.update())

    def saveMP4(self, update:Callable[[int], "Figure"], epochs:int = 10, dpi = 150, video_time = None, del_temp = False):
        # 產出 MP4 影片。策略是「先把每一格畫成 PNG 暫存，再把 PNG 串成影片」——
        # 比直接用 FuncAnimation 重畫複雜圖更穩 (避免狀態殘留)，代價是多一次磁碟暫存。
        from imageio_ffmpeg import get_ffmpeg_exe #? pip install imageio-ffmpeg
        metadata = {
            'title': f'{self.name}',
            "artist": "WeiWen Wu",
            'comment': "Provided by WeiWen's kit"
        }
        rcParams['animation.ffmpeg_path'] = get_ffmpeg_exe()  # 指向 imageio 內附的 ffmpeg

        # 第一階段：逐 epoch 呼叫 update(n) 重畫，並把該格存成 PNG 暫存檔。
        path_video_temp = self.rootdir.joinpath('video_temp').not_exist_create()
        path_merges:list[Path] = []
        for n in trange(epochs, desc='Creating'):
            self.fig.clear()
            path_merges.append(
                update(n).saveIMG(
                    path_video_temp.joinpath(f'{n}.png')
                )
            )
        # 第二階段：動畫實際只是「把第 frame 張 PNG 讀進來貼滿畫面」。
        def _update(frame):
            plt.clf()
            plt.imshow(
                plt.imread(path_merges[frame])
            )
            plt.axis('off')
            plt.tight_layout(pad=0)
            return self

        # 由 video_time(秒) 反推 fps，並夾在 1~120 之間避免極端值；video_time 未給則用 30。
        fps = int(epochs/video_time) if video_time else 30
        writer = FFMpegWriter(fps=max(1, min(fps, 120)), metadata=metadata) # , bitrate=1800
        filename = self._ani_save(_update, epochs, writer, dpi)
        writer.finish()
        logger.info(f'Video creation completed. ({filename.absolute()}, fps: {fps})')
        if del_temp: path_video_temp.rmtree()  # 視需要清掉中途的 PNG 暫存目錄


    def _ani_save(self, update: Callable[[int], Any], epochs, writer, dpi):
        # saveGIF/saveMP4 共用的底層存檔：FuncAnimation 逐格呼叫 update 並寫出，
        # progress_callback 掛 tqdm 顯示輸出進度。回傳最終檔名供記 log。
        tqdm_iter = trange(epochs, desc="Plotting")
        ani = FuncAnimation(self.fig, update, frames=epochs)
        filename = self.rootdir.joinpath(f"{self.name}.mp4")
        ani.save(
            filename, writer=writer, dpi=dpi,
            progress_callback=lambda i, n: tqdm_iter.update(),
        )
        return filename
        
    
    def saveIMG(self, path = None):
        # 把目前 figure 存成 PNG。注意這裡的 FIG_CONFIG 是區域變數，刻意用白底
        # (facecolor/edgecolor='white') 覆蓋模組頂端那份透明設定 —— 因為這些圖會被
        # saveMP4 讀回貼進影片，透明背景會變黑，故統一改白底。
        FIG_CONFIG = {
            "format": 'png',
            "bbox_inches": "tight",
            "pad_inches": 0.1,
            "dpi": 300,
            "transparent": True,
            "facecolor": "white", # white or none
            "edgecolor": "white", # white or none
        }
        # self.fig.set_size_inches(18, 12)
        path = path or self.rootdir.joinpath(f"{self.name}.png")  # 未指定就用 rootdir/name.png
        plt.savefig(path, **FIG_CONFIG)
        return path  # 回傳實際存檔路徑 (saveMP4 靠它收集每格 PNG)

    def __getitem__(self, index:int) -> Axes:
        """
        Use first
        ```
        fig.addAll()
        ```
        """
        # fig[i] 取第 i 個已建立的子圖 Axes；前提是已先 addAll() 把子圖都建出來。
        return self.fig.get_axes()[index]

    def __len__(self) -> int:
        # 子圖總數 = 列 × 欄；addAll() 與 fig[i] 邊界都以此為準。
        return self.nrowcol[0] * self.nrowcol[1]

    def __enter__(self):
        # 進入 with：記下目前的 autograd 開關，再切到本 Figure 指定的 requires_grad。
        # 預設關閉梯度，確保畫圖時的張量運算不會被記進計算圖。
        self.prev = is_grad_enabled()
        set_grad_enabled(self.requires_grad)

        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback):
        # 離開 with：唯有「沒有發生例外」時才 show/save (避免存出半成品的錯誤圖)。
        if not exc_type:
            if self.show: plt.show()
            if self.save: self.saveIMG()
        plt.close()                  # 一律關閉 figure，釋放記憶體 (長訓練必要)
        set_grad_enabled(self.prev)  # 還原進入前的 autograd 狀態

class Record:
    """
    訓練狀態的「時序記錄器」—— 即 train_single.py / train_dual.py 裡的 ★TEMP★。

    核心心智模型：它是一個「每個 key 都對應一條歷史序列 (list)」的容器。
        record['loss'] = 0.5   # 不是覆寫，而是 append 到 'loss' 這條序列尾端
        record('loss')         # 取 'loss' 序列的「最後一筆」(目前值)
        record['loss']         # 取整條 'loss' 序列 (歷史全紀錄)

    為什麼這樣設計 / 在訓練迴圈中的角色：
      - 斷點續跑：每個 epoch 都把 loss / epoch / pattern / 結果等 append 進來，
        再 save() 成 .record (pickle)；重啟時 load(=True) 即可從上次 epoch 接著跑。
      - 繪圖：因為每個 key 是完整序列，直接丟給 matplotlib 就是整段訓練曲線。
      - rollback / early stop：
          * early_stop('real_loss', patience) → 連續 patience 次沒進步就回傳 True，
            觸發把 GEN 回滾到歷史最佳 epoch。
          * find('real_loss', min_loss, 'epoch') → 反查「最佳 loss 出現在哪個 epoch」，
            據此 generator.change(該 epoch, load=True) 把權重載回最佳狀態。
      - pattern 去重：index('patch_pattern_buf', 某 pattern) 找這張 pattern 是否模擬過，
        命中就跳過昂貴的 HFSS、直接用 find 取回先前結果 (省時的關鍵)。

    _data 存「時序資料」；_history 存「每次 save 的中繼資料 (時間/描述/長度)」當存檔日誌。
    """
    def __init__(self, name:str = "record", rootdir:Optional[str] = None, load:bool = False):
        # defaultdict(list)：存取任一新 key 時自動建立空序列，因此 record['x']=v 永遠是 append。
        self._data:dict[str, list] = defaultdict(list)
        self._history = defaultdict(list)  # 存檔日誌：每次 save() 追加一筆 (time/description/len)
        self.name = name
        self.path = Path(rootdir or "./").joinpath(
            f"{name}.record"  # 存成 <name>.record (pickle 檔)
        )

        if load: self.load()  # load=True：建構時就從磁碟載回上次狀態 (斷點續跑入口)

    def __call__(self, key, default = None, *, append = False):
        """Get the last value of key."""
        # record(key)：取該序列「最後一筆」= 目前值。這是訓練迴圈最常用的讀法。
        # append=True 時，若 key 還沒有值就先把 default 寫進去再回傳 (確保曲線從第一個 epoch 起就有點)。
        return self.end(key, default, append = append)

    def __setitem__(self, key, value):
        # record[key] = value：把 value「追加」到該序列尾端 (不是覆寫!)，形成時序。
        self._data[key].append(value)

    def __getitem__(self, key):
        """Get the complete array of keys."""
        # record[key]：取「整條序列」(歷史全紀錄)，常直接餵給 matplotlib 畫曲線。
        if self.__contains__(key):
            return self._data[key]
        else:
            # 故意不回空 list 而是報錯並列出現有 key，避免打錯字導致畫出空圖卻無感。
            _keys = ', '.join(self._data.keys())
            raise KeyError(f"{key} does not exist. (Current key: {_keys})")

    def __delitem__(self, key):
        del self._data[key]

    def  __contains__(self, item:str):
        # 'key' in record：判斷該序列是否存在 (去重時先確認 buffer 鍵已建立)。
        return item in self._data

    def state_dict(self) -> dict[str, dict[str, list]]:
        """Return the state of the Record as a dict."""
        # 仿 PyTorch state_dict 介面：把可序列化的完整狀態打包成普通 dict 供 save() pickle。
        return {    # Convert to a normal dict.
            '_data': dict(self._data),
            '_history': dict(self._history)
        }

    def load_state_dict(self, state_dict: dict[str, dict[str, list]]):
        """Load the Record state."""
        # 從 state_dict 還原；用 .get 容錯舊檔缺欄位，並包回 defaultdict(list) 維持 append 語義。
        loaded_data = state_dict.get('_data', {})
        loaded_history = state_dict.get('_history', {})

        self._data = defaultdict(list, loaded_data)
        self._history = defaultdict(list, loaded_history)

    def end(self, key, default = None, *, append = False):
        # 取某序列的最後一筆 (即「目前值」)。__call__ 就是轉呼叫這裡。
        if self.__contains__(key) and len(self.__getitem__(key)) > 0:
            return self.__getitem__(key)[-1]
        else:
            # 序列還空：append=True → 先把 default 寫入再回傳 (遞迴一次取出)；否則僅回傳 default。
            if append:
                self.__setitem__(key, default)
                return self.end(key)
            else:
                return default

    def add(self, key, num, default = None):
        """
        add('a', 1):
        a += 1
        """
        # 累加器：以「目前值 + num」作為新的一筆 append 進去 (仍保留每一步的歷史)。
        # 訓練裡的 de (距上次刷新最佳的 epoch 數) 就靠 TEMP.add('de', 1) 累加。
        self.__setitem__(
            key, self.end(key, default) + num
        )
        

    
    def save(self, description:Optional[str] = None):
        # 訓練每個 epoch 結尾呼叫 (TEMP.save(f"{epoch} times"))：這就是斷點續跑的「存檔點」。
        # 先在 _history 追加一筆存檔日誌 (時間/描述/當前長度)，再把整個狀態 pickle 寫檔。
        self._history["time"].append(str(datetime.now()).split(".")[0])  # 去掉微秒，只留到秒
        self._history["description"].append(description or "No description")
        self._history["len"].append(len(self))

        current_state = self.state_dict()
        with open(str(self.path), "wb") as f:
            _pickle_dump(
                current_state,
                file = f
            )

    def load(self):
        # 從 .record 載回 (建構時 load=True 會走這裡)。
        if not self.path.exists():
            self.save()  # 首次執行還沒有存檔 → 先存一份空的，避免讀檔失敗
        with open(str(self.path), "rb") as f:
            loaded_state = _pickle_load(f)
        self.load_state_dict(loaded_state)

        return self._data

    def average(self, key:str):
        # 回傳整條序列的平均值 (空序列回 None)。
        # 訓練用它做兩件事：(1) 判斷本筆 real_loss 是否優於歷史平均 → 決定是否收進線上資料集；
        #                  (2) 在圖上標 r_feed / time 的平均值。
        _key_datas = self._data[key]
        _key_datas_len = len(_key_datas)
        if _key_datas_len > 0:
            return sum(_key_datas) / _key_datas_len
        else:
            return None
        
    def index(self, key:str, value, *, start:int = 0, stop:int = sys.maxsize) -> Optional[int]:
        """
        Find the index of `value` in `key`.
        
        Returns:
            Returns the index value, starting from 0. 

            If `value` is not in `key`, returns `None`.

        Example:
            ```
            temp = Record('temp')
            for epoch in range(1, 10+1):
                temp['epoch'] = epoch
            print(temp.index('epoch', 0)) # None
            print(temp.index('epoch', 1)) # 0
            ```
        """
        # 在某序列中找出 value 第一次出現的位置 (找不到回 None)。
        # ★ pattern 去重的核心：訓練用 index('patch_pattern_buf', 這張 pattern) 判斷
        #   這張圖樣是否模擬過 —— 非 None 代表命中快取，可省下一次昂貴的 HFSS 模擬。
        if key not in self._data:
            return None

        # numpy/torch 張量不能用 in / list.index (== 會逐元素比較、語義不對)，
        # 必須改用 array_equal / torch.equal 逐筆做「整體相等」比對。
        if isinstance(value, ndarray):
            _result = [
                np.array_equal(value, x)
                for x in self[key][start:stop]
            ]
        elif isinstance(value, Tensor):
            import torch
            _result = [
                torch.equal(value, x)
                for x in self[key][start:stop]
            ]
        else:
            # 一般可雜湊/可比較的值：直接用 list.index (含 start/stop 範圍)。
            if value in self[key]:
                return self[key].index(value, start, stop)
            else:
                return None

        # 張量情形：回傳第一個 True 的位置 (即第一筆相等的索引)。
        if True in _result:
            return _result.index(True)
        else:
            return None
    
    @overload
    def find(self, key, value, other_keys:str, *, start=0, stop=sys.maxsize) -> Optional[Any]:...
    @overload
    def find(self, key, value, other_keys:Tuple[str, ...] , *, start=0, stop=sys.maxsize) -> Optional[List[Any]]:...

    def find(self, key, value, other_keys, *, start=0, stop=sys.maxsize):
        """
        Find the `value` in `key` that corresponds to `other keys`.

        Returns:
            Returns the `value` corresponding to the `other key`.

            If `value` is not in `key`, returns `None`.

        Examples:
            ```
            temp = Record('temp')
            for epoch, (a, b) in enumerate(zip(
                ['a1', 'a2', 'a3'], ['b1', 'b2', 'b3']
            ), start = 1):
                temp['epoch'] = epoch
                temp['a'] = a
                temp['b'] = b

            print(temp.find('a', 'a1', "epoch"))    # 1
            print(temp.find('epoch', 3, ('a','b'))) # ['a3', 'b3']
            ```
        """
        # 「同一時間步、跨序列查表」：先用 index 在 key 序列找到 value 的位置 _index，
        # 再回傳同一位置上 other_keys 序列的值 (因所有序列以 epoch 同步對齊)。
        # 訓練兩大用途：
        #   (1) rollback：find('real_loss', min_loss, 'epoch') → 反查最佳 loss 是哪個 epoch。
        #   (2) 去重命中：find('patch_pattern_buf', 此 pattern, ('patch_result_buf','real_loss'))
        #       → 取回先前同一張 pattern 的模擬結果與 loss，免再跑 HFSS。
        _index = self.index(key, value, start=start, stop=stop)
        if _index is None:
            return None
        elif isinstance(other_keys, str):
            return self[other_keys][_index]            # 單一 key → 回傳單值
        else:
            _result = []
            for other_key in other_keys:
                _result.append(self[other_key][_index])  # 多個 key → 回傳對齊的值清單
            return _result

    def best(self, mode: Callable = min, key:str = "real_loss", output_keys:list[str] = ['epoch', 'patch_pattern_buf', 'patch_result_buf']) -> list:
        # 找出某指標的「最佳那一筆」並一次帶回對應欄位。
        # 預設 mode=min、key='real_loss'：即「真實 loss 最小的那個 epoch」的 epoch/pattern/結果。
        if key not in self._data or not self._data[key]:
            return None

        # 取得目標 key 中的最佳數值 (Best value)
        best_value = mode(self._data[key])

        # 呼叫現有的 find 方法回傳對應的 output_keys
        return self.find(key, best_value, output_keys)

    def early_stop(self, key: str, patience: int = 10, is_maximize: bool = False) -> bool:
        """
        根據指定 key 的歷史資料，決定是否應該 early stop。
        若最近 `patience` 次都沒有改善，回傳 True。
        Args:
            is_maximize: 若為 True, 則尋找最大值, 否則尋找最小值。
        """
        # ★ 注意：在本專案訓練迴圈中，回傳 True 並非「停止訓練」，而是「觸發 rollback」——
        #   即把 GEN 載回歷史最佳 epoch、並用線上資料集重訓 SM，藉此跳出停滯/局部最佳。
        values = self._data[key]
        if len(values) < patience + 1:
            return False  # 數據不足，不應該停止

        # 根據是最大化還是最小化來決定如何判斷最佳值
        if is_maximize:
            best_func = max
            comparison_op = lambda current, best: current <= best # 對於最大化，如果當前值小於最佳值則視為退步
        else:
            best_func = min
            comparison_op = lambda current, best: current >= best # 對於最小化，如果當前值大於最佳值則視為退步

        # 'best_so_far' 應該是到目前為止，在 patience 視窗之前所見的整體最佳值
        best_so_far = best_func(values[:len(values) - patience])
        recent_values = values[len(values) - patience:]

        # 檢查所有最近的數值是否都比 best_so_far 差
        if all(comparison_op(v, best_so_far) for v in recent_values):
            return True
        return False

    def reset(self, key:Optional[str]=None, delete:bool = False):
        # 清空紀錄：給 key → 只清該序列 (delete=True 連鍵一起移除，否則清成空 list)；
        # 不給 key → 整個 _data 重置 (重新開始記錄)。
        if key is not None:
            if delete:
                self._data.pop(key, None)
            else:
                self._data[key] = []
        else:
            self._data = defaultdict(list)


    def custom(self, key:str, fn:Callable[[list], ReturnType], *, default = None) -> Optional[ReturnType]:
        # 對整條序列套用任意彙總函式並回傳結果 (空序列回 default)。
        # 例：訓練結尾 TEMP.custom('real_loss', min) 取整段最小 real_loss 當最終戰績。
        _key_data = self._data[key]
        if _key_data:
            return fn(_key_data)
        return default

    @property
    def dataframe(self):
        # 把所有時序資料轉成 pandas DataFrame (每個 key 一欄)，方便檢視/匯出/印出。
        processed_data = {}
        for key, values in self._data.items():
            processed_values = []
            for item in values:
                if isinstance(item, torch.Tensor):
                    # 張量先搬回 CPU、detach 脫離計算圖再轉成原生 list/數值，
                    # 否則 DataFrame 無法妥善容納帶梯度的 GPU 張量。
                    # Move to CPU and detach to convert to a standard Python list/number
                    processed_values.append(item.cpu().detach().tolist())
                else:
                    processed_values.append(item)

            processed_data[key] = processed_values
        try:
            return DataFrame(processed_data)
        except ValueError as e:
            # DataFrame 要求各欄等長；長度不一時把目前各序列長度 (repr) 一併拋出，方便定位哪一欄漏記。
            raise ValueError(f"{e}\n{repr(self)}")

    @property
    def history(self):
        # 以 DataFrame 呈現存檔日誌 (每次 save 的時間/描述/長度)，可快速回顧續跑歷程。
        return DataFrame(self._history)

    def __str__(self):
        # print(record) 直接顯示整張資料表。
        return str(self.dataframe)

    def __repr__(self):
        # 精簡摘要：列出每個 key 及其序列長度，例如 Record(temp: epoch[10] real_loss[10] ...)。
        _str = ''
        for key, value in self._data.items():
            _str += f"{key}[{len(value)}] "

        return f"Record({self.name}: {_str})"

    def __len__(self):
        # len(record)：以 DataFrame 列數為準 = 已記錄的 epoch 數 (各序列同步成長)。
        return len(self.dataframe)

class json:
    """
    ### Example
    ```
    from utils.utils import json
    _json = json('static/config.json')
    print(_json('base/UPLOAD_FOLDER'))
    _json_data = _json.load()
    print(_json_data['success'])
    _json_data['success'] = False
    _json.dump(_json_data)
    ```
    """
    def __init__(self, path:str, create:bool = True) -> None:
        # 以「'/' 分隔的點路徑」操作 JSON 檔的小工具 (例如 'base/UPLOAD_FOLDER' 表巢狀鍵)。
        # 每次讀寫都直接落地到檔案，適合當輕量設定/狀態檔，而非高頻寫入。
        self.path = Path(path)

        # 檔案不存在：create=True → 建空檔並初始化成 {}；否則直接報錯。
        if not self.path.exists():
            if create:
                self.path.touch()
                self.dump({})
            else:
                raise FileNotFoundError(f"JSON file '{path}' does not exist.")

    @overload
    def __call__(self, key:str) -> Any: 
        """
        Get the value of the specified key in the JSON file.

        ### Example
        >>> _json('base/UPLOAD_FOLDER')
        """
    ...
    @overload
    def __call__(self, key:str, value:Any) -> dict: 
        """
        Set the value of the specified key in the JSON file.

        ### Example
        >>> _json('base/UPLOAD_FOLDER', 'new/path')
        """
    ...
    def __call__(self, key:str, value = None):
        # 單一入口：給 value → 寫入並回寫檔案；不給 value → 讀取。key 以 '/' 表巢狀路徑。
        keys = key.split('/')
        if value is not None:
            # 從命令列/網頁表單來的值常是字串，這裡把 "null"/"true"/"false" 還原成真正型別。
            if value == "null": value = None
            if value in ["True", "true"]: value = True
            if value in ["False", "false"]: value = False
            result = self._set(keys, value)
            self.dump(result)
            return result
        else:
            return self._get(keys)
    def __getitem__(self, key):
        return self.__call__(key, value = None)   # _json[key] → 讀
    def __setitem__(self, key, value):
        return self.__call__(key, value)          # _json[key] = value → 寫

    def get(self, key:str, default = None):
        # 類似 dict.get：讀不到時把 default 寫進檔案並回傳 (順手補上預設值)。
        keys = key.split('/')
        try:
            return self._get(keys)
        except KeyError:
            result = self._set(keys, default)
            self.dump(result)
            return default

    def _set(self, keys:list, value:Any) -> dict:
        # 依路徑逐層下探設值。注意：實作以 exec 動態組字串賦值來處理任意深度的巢狀，
        # 中途缺少的層級會自動補成空 dict。⚠ 因為用了 exec，請勿傳入不可信的鍵名 (有注入風險)。
        temp =  self.load().copy()
        _ = "temp"
        for i, k in enumerate(keys):
            if k == '': continue
            _ += f"['{k}']"

            if i == len(keys) - 1:
                exec(f"{_} = value")          # 最後一層：實際寫入 value
            else:
                if k not in temp:
                    exec(f"{_} = {{}}")       # 中間層不存在：先建一個空 dict 再往下

        return temp

    def _get(self, keys:list) -> Any:
        # 依路徑逐層下探取值；任一層不存在會自然丟 KeyError (由 get() 接住補預設)。
        self.data = self.load()
        result = self.data.copy()
        for k in keys:
            if k == '': continue
            result = result[k]
        return result
    def load(self) -> dict:
        # 讀整份 JSON 成 dict (每次操作前都重讀，確保拿到磁碟最新內容)。
        with open(self.path, 'r', encoding='utf-8') as f:
            return _json_load(f)

    def dump(self, data:dict) -> bool:
        # 整份覆寫回磁碟；ensure_ascii=False 讓中文不被轉義、indent=4 便於人讀。
        with open(self.path, 'w', encoding='utf-8') as f:
            _json_dump(data, f, ensure_ascii=False, indent=4)
        return True

    def delete(self, key:str) -> bool:
        # 刪除某路徑的鍵：先走到父層，再刪掉最後一段。任一層或目標不存在則回 False。
        keys = key.split('/')
        data = self.load()

        # Traversing through the keys
        temp = data
        for k in keys[:-1]:  # Get to the parent of the key to delete
            if k in temp:
                temp = temp[k]
            else:
                return False  # If the key doesn't exist, return False

        # Deleting the key
        if keys[-1] in temp:
            del temp[keys[-1]]
            self.dump(data)  # Save the updated data back to the file
            return True
        else:
            return False

class TID:
    """
    TID (Time-based ID) Generator
    支援輸出格式：Base62 字串 或 Integer (偏移數值)
    """
    # 用途：產生「短、依時間遞增、可反解回時間」的 ID 當實驗/結果資料夾名稱。
    # 相對於直接用 Unix 秒數，先減去自訂基準再 Base62 編碼可得到更短的字串。
    import string
    # 設定基準時間 (Epoch): 2001-09-28 00:00:00 UTC
    CUSTOM_EPOCH = 1001635200  # 以此為起點計算偏移，縮短 ID 長度
    
    # Base62 字元集
    ALPHABET = string.digits + string.ascii_lowercase + string.ascii_uppercase
    BASE = len(ALPHABET)

    @classmethod
    @overload
    def generate(cls, timestamp: Optional[int] = None, as_int: Literal[False] = False) -> str:
        ...
    @classmethod
    @overload
    def generate(cls, timestamp: Optional[int] = None, as_int: Literal[True] = ...) -> int:
        ...
    
    @classmethod
    def generate(cls, timestamp: int = None, as_int: bool = False) -> Union[str, int]:
        """
        產生 TID。
        :param timestamp: 指定時間戳，若無則使用當前時間
        :param as_int: True 回傳整數 (偏移值); False 回傳 Base62 字串 (預設)
        """
        if timestamp is None:
            timestamp = int(time())
            
        # 計算偏移量 (ID 本體)
        delta = timestamp - cls.CUSTOM_EPOCH
        
        if delta < 0:
            raise ValueError(f"時間早於基準點 2001-09-28，無法產生 ID")
            
        # 若使用者想要 Int，直接回傳偏移後的數值
        if as_int:
            return delta

        # 若為 0 的邊界情況
        if delta == 0:
            return cls.ALPHABET[0]

        # 進行 Base62 編碼
        arr = []
        num = delta
        while num:
            num, rem = divmod(num, cls.BASE)
            arr.append(cls.ALPHABET[rem])
        
        arr.reverse()
        return ''.join(arr)

    @classmethod
    def decode(cls, tid: Union[str, int]) -> int:
        """
        將 TID (字串或整數) 還原為原始 Unix Timestamp
        """
        # 如果傳入的是整數 (Offset Int)，直接加回 Epoch
        if isinstance(tid, int):
            return tid + cls.CUSTOM_EPOCH
            
        # 如果是字串，先解 Base62
        num = 0
        for char in tid:
            if char not in cls.ALPHABET:
                raise ValueError(f"非法字元: {char}")
            num = num * cls.BASE + cls.ALPHABET.index(char)
            
        return num + cls.CUSTOM_EPOCH

def get_shake_128(text: str, length: int = 6) -> str:
    """Generate a shake_128 ID."""
    # 由任意字串產生定長 (預設 6) 的可變長度雜湊 ID。
    # 訓練腳本用它把實驗名稱壓成短 hash_id 嵌進結果資料夾/檔名，避免路徑過長且具識別度。
    from hashlib import shake_128
    return shake_128(text.encode()).hexdigest(length // 2 + 1)[:length]

# 以下為本檔的手動測試/示範區，僅在直接執行 `python utils.py` 時運行，被 import 時不會觸發。
if __name__ == "__main__":
    # print(Path("./checkpoint").manage_file_count("*.pth", keep_latest=1))
    # print(Path("./checkpoint/GEN_model_0.pth").load_torch())
    config.device = 'cpu'
    # 載入某次既有訓練結果的 Record，印出資料表與存檔歷程，驗證續跑載入是否正常。
    r = Record('Temp', load=True, rootdir=r"D:\patch_result\1750340068")

    print(r)
    print(r.history)
    # r.save()
    # loss = LossFunction(CustomLoss())
    # for i in range(10):
    #     loss(tensor([i]), tensor([i + i]))
    # loss.plot(show=True)
