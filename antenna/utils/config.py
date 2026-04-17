import sys
import traceback
from collections.abc import Callable
from json import dump as _json_dump
from json import load as _json_load
from pathlib import Path as _StdPath
from time import time
from types import TracebackType
from typing import (
    Any,
    Literal,
    TypeVar,
    overload,
)
from warnings import filterwarnings

from loguru import logger
from numpy import random
from torch import (
    cuda,
    set_default_device,
)
from torch import (
    device as _torch_device,
)
from torch import (
    manual_seed as _manual_seed,
)

from antenna.utils.path import Path

ReturnType = TypeVar("ReturnType")


def errorCallback(errorCallback: Callable[[str], Any] | None = None, *errorCallbackArgs, **errorCallbackKwargs):
    """
    Error callback function.

    ## Usage
    ```python
    @errorCallback()
    def func():
        raise Exception("Error")
    ```
    """

    def decorator(func: Callable):
        def wrap(*args, **kwargs):
            try:
                return func(*args, **kwargs)  # print(func.__name__)
            except Exception as e:
                if errorCallback:
                    errorCallback(str(e), *errorCallbackArgs, **errorCallbackKwargs)
                else:
                    logger.exception(e)

        return wrap

    return decorator


def global_exception_handler(
    mode: bool | Literal["only_hfss"] = True,
) -> Callable[[type[BaseException], BaseException, TracebackType], None]:
    try:
        from antenna.patch import com_error  # type: ignore
    except Exception:
        com_error = None  # type: ignore

    original_hook = sys.__excepthook__

    if not mode:
        return original_hook

    def excepthook(exc_type: type[BaseException], exc_value: BaseException, exc_traceback: TracebackType):
        """
        全域例外處理 hook。

        ```
        import sys
        sys.excepthook = global_exception_handler
        ```
        """
        # 完整 traceback 字串
        tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))

        # 以 loguru 記錄例外
        logger.opt(exception=(exc_type, exc_value, exc_traceback)).error(
            f"[{exc_type.__name__}] {exc_value}", exc_info=(exc_type, exc_value, exc_traceback)
        )

        is_com_error = (com_error is not None) and issubclass(exc_type, com_error)
        send_email = (mode is True) or (mode == "only_hfss" and is_com_error)
        if issubclass(exc_type, KeyboardInterrupt):
            logger.info("Ctrl + C: Manually stop program execution.")
            return
        elif is_com_error:
            hresult = getattr(exc_value, "hresult", "N/A")
            strerror = getattr(exc_value, "strerror", str(exc_value))
            text = f"HFSS {exc_type.__name__}: HRESULT={hresult} — {strerror}"
        else:
            text = f"{exc_type.__name__}: {exc_value}"

        if send_email:
            from antenna.utils.web import Email, get_local_ip

            with Email("weiwen@alum.ccu.edu.tw") as email:
                msg = email.getText(f"Antanna Error ({get_local_ip()})", f"{text}\n\nTraceback:\n{tb_text}")

                status = email.sendMessage(msg.as_string())

                if status == {}:
                    logger.success("Email sent successfully!")
                else:
                    logger.error("Email send failed!")

    return excepthook


def Complete(message="Process completed.", send_email: bool = False, **results):
    """

    Example ::

        _dict = {}
        Complete(
            "Training Finished!", send_email=True,
            **_dict, **{"Min Loss": ...}, a='a'
        )
    """
    full_message = f"Completed: {message}"
    if results and isinstance(results, dict):
        result_lines = [f"{key}: {value}" for key, value in results.items()]
        formatted_result = "\n".join(result_lines)

        full_message += f"\n\n{formatted_result}"

    if send_email:
        from antenna.utils.web import Email, get_local_ip

        with Email("weiwen@alum.ccu.edu.tw") as email:
            msg = email.getText(f"Antanna Success ({get_local_ip()})", full_message)

            status = email.sendMessage(msg.as_string())

            if status == {}:
                logger.success("Email sent successfully!")
            else:
                logger.error("Email send failed!")

    logger.success(full_message)


class Config(dict):
    NAME: str = None
    """Project name."""
    MAIN_PROGRAM = None
    """Executed by this file."""
    EPOCHS: int = None
    """ """
    LR: float = None
    """Learning Rate For Main Training Loop."""

    RESULT_PATH: Path = None
    CONTINUE_RUN: bool = None

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
        self.ID: str = str(int(time()))
        self.epochs = 10
        self.lr = 1e-3  # Learning Rate For Main Training Loop
        self.element_num = 40
        self["checkpoint_save_path"] = Path("./checkpoint")
        self["device"] = _torch_device(type="cpu")

        self.excepthook = global_exception_handler()

    def __getattr__(self, name):
        # 該Class沒有此變數(name)會執行。
        return self[name]

    def __setattr__(self, name, value):
        cls_attr = getattr(type(self), name, None)

        # 檢查是否是 property 實例，或者是一個方法
        if isinstance(cls_attr, (property, type(lambda: 0))):
            object.__setattr__(self, name, value)
        else:
            self[name] = value

    def check_keys(self, *keys: str, only_warning: bool = False):
        for key in keys:
            if key not in self:
                if only_warning:
                    logger.warning(f"{key} not set.")
                else:
                    raise KeyError(f"{key} not set.")

    @property
    def enable_exception_handler(self) -> bool:
        """Whether exception handling is enabled or not."""
        if sys.excepthook is sys.__excepthook__:
            return False
        else:
            return True

    @enable_exception_handler.setter
    def enable_exception_handler(self, mode: bool):
        sys.excepthook = self.excepthook if mode else sys.__excepthook__

    @property
    def device(self):
        return self["device"]

    @device.setter
    def device(self, device):
        device = device or _torch_device("cuda:0" if cuda.is_available() else "cpu")
        set_default_device(device)
        if device != "cpu":
            cuda.set_device(device)
        self["device"] = device

    @property
    def checkpoint_save_path(self) -> Path:
        """
        Default: ./checkpoint
        ```
        config.checkpoint_save_path.not_exist_create()
        ```
        """
        _checkpoint_save_path: Path = self.get("checkpoint_save_path", Path("checkpoint").not_exist_create())
        return _checkpoint_save_path.absolute()

    @checkpoint_save_path.setter
    def checkpoint_save_path(self, path):
        self["checkpoint_save_path"] = Path(path)

    def setRandomSeeds(self, seed=0):
        _manual_seed(seed)
        cuda.manual_seed(seed)
        random.seed(seed)

    def setWarning(self, warning_type: str = "ignore"):
        return filterwarnings(warning_type)  # type: ignore

    @overload
    def save(
        self, name: str = "config", rootdir: str | None = None, *, update_hook: Callable[[dict], ReturnType]
    ) -> ReturnType: ...
    @overload
    def save(self, name: str = "config", rootdir: str | None = None) -> None: ...

    def save(
        self,
        name: str = "config",
        rootdir: str | None = None,
        *,
        update_hook: Callable[[dict], ReturnType] | None = None,
    ):
        """
        將設定序列化為 JSON 儲存。

        僅支援以下型別：
        ```
        dict, list, tuple, str, int, float, bool, None
        ```
        其餘型別會透過 `str()` 轉為字串。

        :param update_hook: def update_hook(config)..., Ex: wandb.config.update

        .. note::
            Legacy API — 新 code 請改用 Hydra DictConfig / OmegaConf.save。
            此方法僅供 legacy `train_*.py` 腳本使用。
        """
        path = _StdPath(rootdir or "./") / f"{name}.json"
        _save = {}
        self.update(vars(self))
        for key, value in self.items():
            if isinstance(value, (dict, list, tuple, str, int, float, bool)) or value is None:
                _save[key] = value
            else:
                _save[key] = str(value)
        with open(path, "w", encoding="utf-8") as f:
            _json_dump(_save, f, indent=4, ensure_ascii=False)

        if update_hook:
            return update_hook(self)

    def load(self, name: str = "config", rootdir: str | None = None):
        """
        從 JSON 檔載入設定至目前 Config 實例。

        僅能讀入 JSON 原生支援的型別（dict, list, str, int, float, bool, None）。

        .. note::
            Legacy API — 新 code 請改用 Hydra DictConfig / OmegaConf.load。
        """
        path = _StdPath(rootdir or "./") / f"{name}.json"
        with open(path, encoding="utf-8") as f:
            self.update(_json_load(f))

    def __str__(self):
        _str = ", ".join(f"{k}={v}" for k, v in self.items())
        return f"{self.__class__.__name__}({_str})"


config = Config()


class MultiConfig:
    """
    多標籤設定容器。

    .. note::
        Legacy API — 新 code 請改用 Hydra multi-run 或 structured configs。
    """

    def __init__(self, config: dict[str, dict[str, Any]] | None = None, label: str | None = None):
        """
        Example ::

            MULTICONFIG = MultiConfig(
                {
                    'default': {
                        ...
                    }
                },
                label = 'default'
            )

        :param config: 標籤到設定字典的映射。
        :param label: 要使用的標籤；若為 None 則讀取 `sys.argv[1]`。
        """
        self.metadata: dict[str, dict] = config if config is not None else {}

        if label is None and len(sys.argv) < 2:
            raise ValueError(
                "Must provide a configuration label either as a command-line argument "
                "or directly to the MultiConfig constructor."
            )
        self.config_label = str(label if label is not None else sys.argv[1])

    @property
    def label(self):
        return self.config_label

    @label.setter
    def label(self, value):
        self.config_label = str(value)

    def get_label_data(self, label: str | None = None):
        return self.metadata[label or self.config_label]

    def __setitem__(self, key, value):
        self.metadata[self.config_label][key] = value

    def __getitem__(self, key):
        return self.metadata[self.config_label][key]

    def __call__(self, key: str, default=None):
        if key in self.metadata[self.config_label]:
            return self.metadata[self.config_label][key]
        else:
            return default
