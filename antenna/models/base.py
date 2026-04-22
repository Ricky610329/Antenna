from types import FunctionType
from typing import Callable, Generic

import torch
from loguru import logger
from torch import Tensor

from antenna.types import (
    CallableModule,
    Checkpoint,
    CustomModule,
    CustomOptimizer,
    CustomScheduler,
    LossParams,
    ModelParams,
    ReturnType,
)
from antenna.utils.config import config
from antenna.utils.path import Path
from antenna.utils.record import Record


class Models(Generic[CustomModule, ModelParams, ReturnType, CustomOptimizer, CustomScheduler, LossParams]):
    def __init__(
        self,
        name: str = "models_{label}",
        rootdir: str | Path | None = None,
        model: CustomModule | CallableModule[ModelParams, ReturnType] | None = None,
        optimizer: CustomOptimizer | None = None,
        scheduler: CustomScheduler | None = None,
        criterion: Callable[LossParams, Tensor] | None = None,
        *,
        load: bool = False,
        device=config.device,
    ):
        has_placeholder = "{label}" in name
        self._name = name if has_placeholder else None
        self.name = None if has_placeholder else name

        self._rootdir = rootdir or config.checkpoint_save_path

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.record = Record(self.__class__.__name__, rootdir=self._rootdir, load=load and self.model_file.exists())

        # 快取 device；避免每次透過 next(model.parameters()) 取得（昂貴）。
        self.device = device
        if load:
            self.load()

    def __call__(self, *args: ModelParams.args, **kwargs: ModelParams.kwargs) -> ReturnType:
        return self.model(*args, **kwargs)

    def __str__(self):
        criterion_name = (
            self.criterion.__name__ if isinstance(self.criterion, FunctionType) else self.criterion.__class__.__name__
        )
        return (
            f"{self.__class__.__name__}("
            f"Model={self.model.__class__.__name__}, "
            f"Optimizer={self.optimizer.__class__.__name__}, "
            f"Scheduler={self.scheduler.__class__.__name__}, "
            f"Criterion={criterion_name})"
        )

    @property
    def model_file(self) -> Path:
        """model checkpoint 檔案的完整路徑。"""
        assert self.name, "Please use `Models.change()` first."
        return Path(self._rootdir).joinpath(f"{self.name}.pth")

    @property
    def device(self):
        """快取的 model device。"""
        return self._device

    @device.setter
    def device(self, device):
        if self.model is not None:
            self.model.to(device=device)
            self._device = next(self.model.parameters()).device
        else:
            self._device = torch.device(device) if not isinstance(device, torch.device) else device

    @property
    def FloatTensor(self):
        return torch.FloatTensor if str(self.device) == "cpu" else torch.cuda.FloatTensor  # type: ignore

    def change(self, label: str, *, load: bool = False, save: bool = False):
        """變更 model 的 label（以 ``name`` 中的 ``{label}`` 佔位符為準）。

        :param label: 欲套用的 label；若 ``name`` 中不含 ``{label}`` 佔位符則不會變更檔名。
        :param load: 套用新 label 後是否立即讀取對應的 checkpoint。
        :param save: 變更 label 之前是否先存檔舊的 checkpoint。
        """
        if save:
            self.save()
        if self._name:
            self.name = self._name.format(label=label)
        if load:
            self.load()
        return self.name

    def load(self, force: bool = False):
        """從 :pyattr:`model_file` 讀取 checkpoint 並還原 model/optimizer/scheduler/record 狀態。

        :param force: 若為 ``True``，即使 checkpoint 的 ``title`` 與目前實例不符也直接套用。
        """
        checkpoint_loaded = self._load_checkpoint_from_disk()
        if not force and checkpoint_loaded["title"] != self.__str__():
            raise RuntimeError(
                f"Please use the correct model file.\nFile: {checkpoint_loaded['title']}\nCurrent: {self.__str__()}"
            )
        self.device = checkpoint_loaded["device"]
        self.model.load_state_dict(checkpoint_loaded["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint_loaded["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint_loaded["scheduler_state_dict"])
        self.record.load_state_dict(checkpoint_loaded["record_state_dict"])

    def save(self) -> Path:
        """將目前的 checkpoint 存入 :pyattr:`model_file`。"""
        return self.save_as(self.model_file)

    def save_as(self, filename: str | Path) -> Path:
        """以原子寫入（透過 ``.tmp`` 暫存檔）方式將 checkpoint 存到指定路徑。

        :param filename: 檔案完整路徑，含副檔名 (suffix)。
        """
        filename = Path(filename)
        temp_file = filename.with_suffix(filename.suffix + ".tmp")
        try:
            torch.save(self._build_checkpoint(), temp_file)
            temp_file.replace(filename)
        except Exception:
            if temp_file.exists():
                temp_file.unlink()
            raise
        return filename

    def pre_load_model(self, path: str | Path):
        """從指定 checkpoint 預載 model/optimizer 權重，並驗證參數不含 NaN/inf。"""
        path = Path(path)
        checkpoint_loaded: Checkpoint = path.load_torch()
        self.model.load_state_dict(checkpoint_loaded["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint_loaded["optimizer_state_dict"])
        for name, param in self.model.state_dict().items():
            if not torch.all(torch.isfinite(param)):
                raise RuntimeError(f"!!! 在參數 '{name}' 中發現無效值 (NaN 或 inf) !!!")
        logger.success(f"Successfully loaded the pre-trained model. ({path})")

    def step(self, optimizer_param=None, scheduler_param=None):
        """執行一次 optimizer step；若有 scheduler，同時觸發 scheduler step。"""
        self.optimizer.step(optimizer_param)
        if self.scheduler:
            self.scheduler.step(scheduler_param)

    def checkpoint(self, load: bool = False) -> Checkpoint:
        """保留舊 public 介面；load=True 讀檔，False 產生當前 state checkpoint。"""
        return self._load_checkpoint_from_disk() if load else self._build_checkpoint()

    def _build_checkpoint(self) -> Checkpoint:
        return {
            "title": self.__str__(),
            "model_state_dict": None if not self.model else self.model.state_dict(),
            "optimizer_state_dict": None if not self.optimizer else self.optimizer.state_dict(),
            "scheduler_state_dict": None if not self.scheduler else self.scheduler.state_dict(),
            "device": self.device,
            "record_state_dict": self.record.state_dict(),
        }

    def _load_checkpoint_from_disk(self) -> Checkpoint:
        return self.model_file.load_torch()

    def requires_grad(self, mode: bool = True, train: bool | None = None):
        """設定 model 所有參數的 ``requires_grad``，並可選擇切換 train/eval 模式。

        :param mode: 是否對參數啟用梯度追蹤。
        :param train: ``True`` 切換至 ``train()``、``False`` 切換至 ``eval()``、``None`` 保持現狀。
        :return: 參數的 ``requires_grad`` 實際值（以第一個參數為準）。
        """
        for param in self.model.parameters():
            param.requires_grad = mode

        match train:
            case True:
                self.model.train()
            case False:
                self.model.eval()
            case _:
                pass

        return next(self.model.parameters()).requires_grad
