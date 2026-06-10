# ==============================================================================
# utils.py — 反向設計閉迴路系統的「基礎底座」
# ------------------------------------------------------------------------------
# 本檔是 utils 子套件的最底層 (其他工具模組都可能 import 它)。重點元件：
#   - errorCallback / global_exception_handler：例外處理；HFSS COM 錯誤會自動寄 email。
#   - Complete：訓練完成通知 (可寄信)。
#   - Path：pathlib 擴充 (自動建目錄、舊檔輪替、load_torch 載入 checkpoint 等)。
#     ★ 序列化注意：Path 實例會被 pickle 進舊結果檔 (有 __reduce__)，本類「不可搬離本檔」，
#       否則舊 checkpoint / record 反序列化會找不到 antenna.utils.utils.Path。
#   - Config + 全域 config：dict 子類，集中管理 device / checkpoint 路徑 / 例外處理開關等。
#   - TID / get_shake_128：時間 ID、短雜湊 ID。
# 已拆出的同套件模組 (經 antenna.utils facade 取用)：
#   figure.py — Figure (matplotlib with-context 包裝)；record.py — Record (=TEMP)；
#   store.py — SampleStore (一筆一檔樣本庫)；data.py — 舊 DataManager (學長 code 保留)。
# ==============================================================================
from typing import TypeVar, Callable, Any, Optional, overload, Union, Literal
from loguru import logger  # 專案統一使用 loguru 而非標準 logging
import traceback
from types import TracebackType
from torch import (
    __version__,
    cuda,
    manual_seed as _manual_seed,
    load as _torch_load,
    device as _torch_device,
    set_default_device,
)
from numpy import random
from json import (
    load as _json_load,
    dump as _json_dump
)
from warnings import filterwarnings

from pathlib import Path as _Path
from os.path import getctime

import sys
from shutil import rmtree as _rmtree
from time import time

import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes  # type: ignore  (facade 對外 re-export，本檔僅轉手)

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

