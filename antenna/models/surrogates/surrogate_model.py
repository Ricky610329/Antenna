"""代理模型 (Surrogate Model) 的訓練封裝器與工廠函數。

- :class:`SurrogateModel`：繼承 :class:`antenna.models.base.Models` 的代理模型訓練封裝。
- :func:`OldSM`：以 :class:`HFSSNet` 建立代理模型（學長的原始做法）。
- :func:`UNetSM`：以 :class:`EnhancedHFSSUNet` 建立代理模型。
"""

from typing import Callable, Generic, cast

import torch
from loguru import logger
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from antenna.core.pattern import AntennaPattern
from antenna.core.response import AntennaResponse, MultiResponses
from antenna.models.base import Models
from antenna.models.surrogates.hfss_net import HFSSNet
from antenna.models.surrogates.unet import EnhancedHFSSUNet
from antenna.ranger import Ranger
from antenna.schedulers.adaptive_cyclical import AdaptiveCyclicalScheduler
from antenna.types import (
    CallableParam,
    CustomModule,
    CustomOptimizer,
    CustomScheduler,
    LossParams,
    ModelParams,
    ReturnType,
)
from antenna.utils.config import config
from antenna.utils.data import DataManager, size_converter
from antenna.utils.figure import TQDM_BAR_SIMPLE, TQDM_CONFIG
from antenna.utils.torch_utils import tensor


class SurrogateModel(
    Models[CustomModule, ModelParams, ReturnType, CustomOptimizer, CustomScheduler, LossParams],
    Generic[CustomModule, ModelParams, ReturnType, CustomOptimizer, CustomScheduler, LossParams],
):
    """代理模型訓練封裝。

    Global Variable
    ---------------
    ``config['HFSS.min_loss']``, ``config['HFSS.max_epoch']`` 於 ``train_one_data`` 中使用。
    """

    def __init__(
        self,
        model: CustomModule,
        criterion: Callable[CallableParam, Tensor],
        optimizer: CustomOptimizer,
        scheduler: CustomScheduler | None = None,
        *,
        rootdir=None,
    ):
        super().__init__(
            name="sm",
            rootdir=rootdir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            load=False,  # 避免呼叫父類別未覆寫的 load
        )

        self.epoch = 0

    def __call__(self, pattern) -> MultiResponses:
        self.epoch += 1
        return MultiResponses(self.model(pattern))

    def train_by_datas(
        self,
        dataset: DataManager,
        epochs: int = 100,
        batch_size: int | None = None,
        *,
        verbose: bool = True,
    ) -> list[float]:
        """使用 ``dataset`` 訓練代理模型。

        Args:
            dataset: 用於訓練的資料集。
            epochs: 總訓練 epoch 數。
            batch_size: 每個 batch 的大小。
            verbose: 是否顯示進度條。

        Returns:
            每個 epoch 的平均 loss 列表。
        """
        self.requires_grad(True, train=True)
        self.record.reset()

        if dataset is None or len(dataset) <= 0:
            return []
        if batch_size is not None:
            batch_size = min(len(dataset), batch_size)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator(device=config.device),
        )

        epoch_bar = tqdm(range(epochs), desc="Training...", disable=not verbose, **TQDM_CONFIG)
        for epoch in epoch_bar:
            for patterns, real_responses in cast(tuple[Tensor, Tensor], dataloader):
                patterns = size_converter(AntennaPattern, patterns, flatten=True, batch=True)
                real_responses = size_converter(AntennaResponse, real_responses, flatten=False, batch=True)

                inputs: Tensor = patterns.flatten(start_dim=1).to(config.device)
                labels: Tensor = real_responses.to(config.device)

                self.optimizer.zero_grad()
                outputs: Tensor = self.model(inputs)
                loss: Tensor = self.criterion(outputs, labels)

                loss.backward()
                self.step(scheduler_param=loss)

                self.record["loss"] = loss.item()

            avg_epoch_loss = self.record.average("loss")
            self.record.reset("loss", delete=True)
            self.record["epoch_loss"] = avg_epoch_loss

            epoch_bar.set_postfix({"Loss": f"{avg_epoch_loss:.4e}"})

            if self.record.early_stop("epoch_loss", int(epochs / 2)):
                logger.success(f"Early Stopping triggered at epoch {epoch + 1}!")
                break

        self.model.eval()
        return self.record["epoch_loss"]

    def train_one_data(
        self,
        pattern: Tensor,
        real_response: Tensor,
        min_loss=None,
        max_epoch=None,
        *,
        verbose: bool = True,
    ):
        """用單筆資料訓練模型。

        Args:
            pattern: 真實天線 pattern。
            real_response: ``pattern`` 對應的真實響應。
            min_loss: loss 下限（達到即停止）。
            max_epoch: epoch 上限。
            verbose: 是否顯示進度條。

        Returns:
            最終的 loss 值。
        """
        self.requires_grad(True, train=True)
        self.record.reset()

        self.record["loss"] = float("inf")
        self.record["epoch"] = 0

        input = tensor(pattern, requires_grad=True)
        label = tensor(real_response, requires_grad=True)

        min_loss = min_loss or config["HFSS.min_loss"]
        max_epoch = max_epoch or config["HFSS.max_epoch"]

        epoch_bar = tqdm(
            total=max_epoch,
            desc="Training one data",
            bar_format=TQDM_BAR_SIMPLE,
            disable=not verbose,
            **TQDM_CONFIG,
        )
        while self.record("loss", 0) > min_loss and self.record("epoch", float("inf")) < max_epoch:
            self.optimizer.zero_grad()

            outputs_result: Tensor = self.model(input)

            loss: Tensor = self.criterion(
                outputs_result.reshape(-1, *AntennaResponse.size()),
                label.reshape(-1, *AntennaResponse.size()),
            )

            loss.backward()
            self.step(scheduler_param=loss)

            self.record["loss"] = loss.item()
            self.record.add("epoch", 1)

            epoch_bar.update()
            epoch_bar.set_postfix({"loss": f"{self.record('loss'):.2f}/{min_loss}"})

        self.model.eval()
        return self.record["loss"]


def OldSM(checkpoint) -> SurrogateModel:
    """以全連接 :class:`HFSSNet` 建立的代理模型（學長的原始做法）。"""
    model_ge = HFSSNet(  # Pattern -> Response
        AntennaPattern.size(flatten=True), AntennaResponse.size()
    )
    criterion_ge = nn.MSELoss()
    optimizer_ge = Ranger(params=model_ge.parameters(), lr=config["HFSS.lr"])
    scheduler_ge = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_ge, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )
    return SurrogateModel(model_ge, criterion_ge, optimizer_ge, scheduler_ge, rootdir=checkpoint)


def UNetSM(
    checkpoint,
    base_channels: int = 64,
    dropout_prob: float = 0.15,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    loss_type: str = "L1",
) -> SurrogateModel:
    """以 :class:`EnhancedHFSSUNet` 建立代理模型。

    Args:
        checkpoint: 模型權重的儲存/載入路徑。
        base_channels: U-Net 第一層的基礎通道數。
        dropout_prob: 應用於 DoubleConv 層的 Dropout 概率。
        learning_rate: 優化器學習率。
        weight_decay: 優化器權重衰減。
        loss_type: 損失函數類型，``"L1"`` 或 ``"MSE"``。
    """
    model_ge = EnhancedHFSSUNet(base_channels=base_channels, dropout_prob=dropout_prob)

    if loss_type == "L1":
        criterion_ge = nn.L1Loss()
    elif loss_type == "MSE":
        criterion_ge = nn.MSELoss()
    else:
        raise ValueError("loss_type 必須是 'L1' 或 'MSE'")

    optimizer_ge = Ranger(
        params=model_ge.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler_ge = AdaptiveCyclicalScheduler(optimizer_ge)
    return SurrogateModel(model_ge, criterion_ge, optimizer_ge, scheduler_ge, rootdir=checkpoint)
