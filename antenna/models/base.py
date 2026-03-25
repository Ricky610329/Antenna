from types import FunctionType

import torch
from torch.types import Tensor

from antenna import *
from antenna.types import *
from antenna.utils import *


class Models(Generic[CustomModule, ModelParams, ReturnType, CustomOptimizer, CustomScheduler, LossParams]):
    def __init__(
        self,
        name: str = "models_{label}",
        rootdir: Optional[Union[str, Path]] = None,
        model: Optional[CustomModule | CallableModule[ModelParams, ReturnType]] = None,
        optimizer: Optional[CustomOptimizer] = None,
        scheduler: Optional[CustomScheduler] = None,
        criterion: Optional[Callable[LossParams, Tensor]] = None,
        *,
        load: bool = False,
        device=config.device,
    ):
        if "{label}" in name:
            self._name = name
            self.name = None
        else:
            self._name = None
            self.name = name

        self._rootdir = rootdir or config.checkpoint_save_path

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.record = Record(self.__class__.__name__, rootdir=self._rootdir, load=load and self.model_file.exists())

        self.device = device
        if load:
            self.load()

    def __call__(self, *args: ModelParams.args, **kwargs: ModelParams.kwargs) -> ReturnType:
        return self.model(*args, **kwargs)

    def __str__(self):
        _str = "{class_name}(Model={model}, Optimizer={optimizer}, Scheduler={scheduler}, Criterion={criterion})"
        return _str.format(
            class_name=self.__class__.__name__,
            model=self.model.__class__.__name__,
            optimizer=self.optimizer.__class__.__name__,
            scheduler=self.scheduler.__class__.__name__,
            criterion=self.criterion.__name__
            if isinstance(self.criterion, FunctionType)
            else self.criterion.__class__.__name__,
        )

    @property
    def model_file(self) -> Path:
        """The full path to the model archive."""
        assert self.name, "Please use `Models.change()` first."
        return Path(self._rootdir).joinpath(f"{self.name}.pth")

    @property
    def device(self):
        """Model parameters of the device."""
        return next(self.model.parameters()).device

    @device.setter
    def device(self, device):
        self.model.to(device=device)

    @property
    def FloatTensor(self):
        return torch.FloatTensor if str(self.device) == "cpu" else torch.cuda.FloatTensor  # type: ignore

    def change(self, label: str, *, load: bool = False, save: bool = False):
        """
        Change model label.

        :param label: models label. You can enter `{label}` in name.
        :param load: Load the changed models.
        :param save: Save the models before the change.
        """
        if save:
            self.save()
        if self._name:
            self.name = self._name.format(label=label)
        if load:
            self.load()
        return self.name

    def load(self, force: bool = False):
        checkpoint_loaded = self.checkpoint(load=True)
        if checkpoint_loaded["title"] == self.__str__() or force:
            self.device = checkpoint_loaded["device"]

            self.model.load_state_dict(checkpoint_loaded["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint_loaded["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint_loaded["scheduler_state_dict"])
            self.record.load_state_dict(checkpoint_loaded["record_state_dict"])

        else:
            raise RuntimeError(
                f"Please use the correct model file.\nFile: {checkpoint_loaded['title']}\nCurrent: {self.__str__()}"
            )

    def save(self) -> Path:
        return self.save_as(self.model_file)

    def save_as(self, filename: Union[str, Path]) -> Path:
        """
        :param filename: 檔案完整路徑，含副檔名(suffix)
        """

        filename = Path(filename)
        temp_file = filename.with_suffix(filename.suffix + ".tmp")
        try:
            torch.save(self.checkpoint(load=False), temp_file)
            temp_file.replace(filename)
        except Exception:
            if temp_file.exists():
                temp_file.unlink()
            raise
        return filename

    def pre_load_model(self, path: Union[str, Path]):
        path = Path(path)
        checkpoint_loaded: Checkpoint = path.load_torch()
        self.model.load_state_dict(checkpoint_loaded["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint_loaded["optimizer_state_dict"])
        for name, param in self.model.state_dict().items():
            if not torch.all(torch.isfinite(param)):
                raise RuntimeError(f"!!! 在參數 '{name}' 中發現無效值 (NaN 或 inf) !!!")
        logger.success(f"Successfully loaded the pre-trained model. ({path})")

    def step(self, optimizer_param=None, scheduler_param=None):
        self.optimizer.step(optimizer_param)
        if self.scheduler:
            self.scheduler.step(scheduler_param)

    def checkpoint(self, load: bool = False) -> Checkpoint:
        if load:
            checkpoint: dict = self.model_file.load_torch()
        else:
            checkpoint = {
                "title": self.__str__(),
                "model_state_dict": None if not self.model else self.model.state_dict(),
                "optimizer_state_dict": None if not self.optimizer else self.optimizer.state_dict(),
                "scheduler_state_dict": None if not self.scheduler else self.scheduler.state_dict(),
                "device": self.device,
                "record_state_dict": self.record.state_dict(),
            }
        return checkpoint

    def requires_grad(self, mode: bool = True, train: Optional[bool] = None):
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
