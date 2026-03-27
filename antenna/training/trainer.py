"""
統一訓練迴圈，取代原始 train_*.py 腳本。

使用方式::

    from antenna.training.trainer import Trainer
    trainer = Trainer(cfg)
    trainer.run()
"""

from time import time

import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from antenna.core.pattern import AntennaPattern
from antenna.core.response import AntennaResponse
from antenna.losses.patch_losses import custom_loss_g, custom_loss_minmax, custom_loss_r
from antenna.losses.regularization import FeedReachability, GapClosingLoss, SpectralConnectivityLoss
from antenna.models.base import Models
from antenna.models.generators.gumbel_sigmoid_gen import GumbelSigmoidGEN
from antenna.models.generators.sigmoid_gen import SigmoidGEN
from antenna.schedulers.adaptive_cyclical import AdaptiveCyclicalScheduler
from antenna.smodels import OldSM
from antenna.utils.config import config
from antenna.utils.data import DataManager
from antenna.utils.figure import Figure
from antenna.utils.path import Path
from antenna.utils.record import Record
from antenna.utils.web import connect_default_drive

# 損失函數對應表
LOSS_FN_REGISTRY = {
    "custom_loss_minmax": custom_loss_minmax,
    "custom_loss_r": custom_loss_r,
    "custom_loss_g": custom_loss_g,
}

# 模型對應表
MODEL_REGISTRY = {
    "sigmoid_gen": SigmoidGEN,
    "gumbel_sigmoid_gen": GumbelSigmoidGEN,
}


class Trainer:
    """統一訓練器，從 Hydra DictConfig 驅動。"""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self._setup_environment()
        self._setup_paths()
        self._setup_antenna()
        self._setup_models()
        self._setup_tracking()

    # ── 初始化階段 ──────────────────────────────────

    def _setup_environment(self):
        """設定裝置、網路磁碟。"""
        config.device = self.cfg.environment.device
        connect_default_drive(self.cfg.environment.network_drive_letter)

    def _setup_paths(self):
        """建立結果目錄。"""
        from antenna import get_result_path
        from antenna.utils import ROOTDIR

        rootdir = self.cfg.environment.rootdir or str(ROOTDIR)
        self.result_path, self.continue_run = get_result_path(
            self.cfg.experiment_name,
            rootdir=rootdir,
            enable_exception_handler=True,
        )
        self.path_pic = self.result_path.joinpath("pic").not_exist_create()
        self.path_checkpoint = self.result_path.joinpath("checkpoint").not_exist_create()

        config.epochs = self.cfg.epochs
        config.lr = self.cfg.optimizer.lr
        config.checkpoint_save_path = self.path_checkpoint

    def _setup_antenna(self):
        """設定 AntennaPattern 與 AntennaResponse。"""
        coord = tuple(self.cfg.pattern.coordinate)
        AntennaPattern.setDefaultCoordinate(coord)

        # 設定 Response
        resp_cfg = self.cfg.response
        AntennaResponse.registerLabels(*resp_cfg.labels, x=resp_cfg.x)

        # 註冊每個 label 的 target 與 loss hook
        self._targets = {}
        for label, label_cfg in resp_cfg.label_configs.items():
            target_cfg = label_cfg.target
            target_tensor = AntennaResponse.registerTargetResponse(
                target_cfg.side, target_cfg.center, target_cfg.width, label=label
            )
            self._targets[label] = target_tensor

            loss_fn = LOSS_FN_REGISTRY.get(label_cfg.loss_fn)
            if loss_fn is None:
                raise ValueError(f"未知的損失函數: {label_cfg.loss_fn}")
            AntennaResponse.registerLossHook(loss_fn, label=label, target=target_tensor, **label_cfg.loss_params)

    def _setup_models(self):
        """建立生成器、優化器、排程器、代理模型。"""
        # 生成器
        model_cls = MODEL_REGISTRY.get(self.cfg.model)
        if model_cls is None:
            raise ValueError(f"未知的模型: {self.cfg.model}")
        self.model = model_cls()

        # 優化器
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.cfg.optimizer.lr,
            betas=tuple(self.cfg.optimizer.betas),
        )

        # 排程器
        sched_cfg = self.cfg.scheduler
        if sched_cfg._target_ == "none":
            self.scheduler = None
        elif "AdaptiveCyclicalScheduler" in sched_cfg._target_:
            self.scheduler = AdaptiveCyclicalScheduler(
                self.optimizer,
                T_0=sched_cfg.get("T_0", 100),
                T_mult=sched_cfg.get("T_mult", 1),
                lr_max=sched_cfg.get("lr_max", 0.005),
                lr_min=sched_cfg.get("lr_min", 1e-6),
                temp_max=sched_cfg.get("temp_max", 4.0),
                temp_min=sched_cfg.get("temp_min", 0.1),
                warmup_ratio=sched_cfg.get("warmup_ratio", 0.2),
                patience=sched_cfg.get("patience", 25),
                factor=sched_cfg.get("factor", 0.7),
                mode=sched_cfg.get("mode", "min"),
                on_plateau=sched_cfg.get("on_plateau", "linear"),
            )
        elif "ReduceLROnPlateau" in sched_cfg._target_:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                factor=sched_cfg.get("factor", 0.5),
                patience=sched_cfg.get("patience", 10),
                min_lr=sched_cfg.get("min_lr", 1e-6),
            )
        else:
            self.scheduler = None

        # Models 封裝
        self.generator = Models(
            name="generator_{label}",
            rootdir=self.path_checkpoint,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            criterion=custom_loss_minmax,
        )

        # 代理模型
        config["HFSS.lr"] = self.cfg.surrogate.hfss_lr
        config["HFSS.min_loss"] = self.cfg.surrogate.hfss_min_loss
        config["HFSS.max_epoch"] = self.cfg.surrogate.hfss_max_epoch
        self.smodel = OldSM(checkpoint=self.path_checkpoint)

        # 載入檢查點
        if self.continue_run and hasattr(self, "record") and ("epoch" in self.record):
            self.generator.change(self.record("epoch"), load=True)
            self.smodel.load()

    def _setup_tracking(self):
        """初始化記錄、資料集、正則化。"""
        from antenna.utils import DATASET_PATH

        self.record = Record("temp", rootdir=self.result_path, load=True)
        self.online_dataset = DataManager("online", rootdir=self.result_path)

        # 正則化損失
        self.sc_loss = SpectralConnectivityLoss() if self.cfg.spectral_connectivity_loss_weight > 0 else None
        self.gc_loss = GapClosingLoss() if self.cfg.gap_closing_loss_weight > 0 else None
        self.r_feed = FeedReachability.single_feed()

    # ── 訓練迴圈 ──────────────────────────────────

    def run(self):
        """主訓練迴圈。"""
        epoch = self.record("epoch", 0)
        jump = 0

        logger.info(f"開始訓練: {self.cfg.experiment_name}")
        logger.info(f"模型: {self.cfg.model}, 設備: {self.cfg.environment.device}, Epochs: {self.cfg.epochs}")

        while epoch < self.cfg.epochs + 1:
            start = time()
            epoch += 1
            self.generator.change(epoch)

            logger.info(f"Start {epoch} of {self.cfg.epochs}")

            self.generator.requires_grad(True, train=True)
            self.generator.optimizer.zero_grad()

            # ── Early stopping & rollback ──
            self.record["tau"] = 0
            if self.record.early_stop("real_loss", self.cfg.patience):
                self.generator.change(
                    self.record.find("real_loss", self.record("min_loss", float("inf")), "epoch"),
                    save=True,
                    load=True,
                )
                self.smodel.train_by_datas(self.online_dataset)
                output_element = AntennaPattern(self.generator(AntennaResponse.target.concat()))
                self.record["mutation"] = self.record("min_loss")
            else:
                output_element = AntennaPattern(self.generator(AntennaResponse.target.concat()))
                self.record["mutation"] = 0

            # ── 去重 & 模擬 ──
            if (
                "patch_pattern_buf" not in self.record
                or self.record.index("patch_pattern_buf", ~output_element) is None
            ):
                output_result = output_element.simulate()
                real_loss = output_result.criterion()
                stack_output_result = output_result.stack()
                self.smodel.train_one_data(output_element.series, stack_output_result)
                self.smodel.save()

                self.record["real_loss"] = real_loss.item()
                if self.record("real_loss") < self.record.average("real_loss"):
                    self.online_dataset.add_and_save([~output_element, stack_output_result])
                jump = 0
            else:
                stack_output_result, real_loss = self.record.find(
                    "patch_pattern_buf", ~output_element, ("patch_result_buf", "real_loss")
                )
                self.record["real_loss"] = real_loss
                jump += 1

            # ── 更新最小 loss ──
            min_loss = self.record("min_loss", float("inf"))
            if self.record("real_loss") <= min_loss:
                min_loss = self.record("real_loss")
                self.record["de"] = 0
                config["best_epoch"] = epoch
            else:
                self.record.add("de", 1, default=0)
            self.record["min_loss"] = min_loss

            # ── 快取 ──
            self.record["patch_pattern_buf"] = ~output_element
            self.record["patch_result_buf"] = stack_output_result
            self.record["r_feed"] = self.r_feed(~output_element)

            # ── 透過代理模型更新生成器 ──
            response = self.smodel(output_element.series)
            loss = response.criterion()

            if self.cfg.total_variation_loss_weight > 0:
                loss = loss + output_element.total_variation_loss(self.cfg.total_variation_loss_weight)
            if self.cfg.island_suppression_loss_weight > 0:
                loss = loss + output_element.island_suppression_loss(self.cfg.island_suppression_loss_weight)
            if self.sc_loss and self.cfg.spectral_connectivity_loss_weight > 0:
                loss = loss + self.cfg.spectral_connectivity_loss_weight * self.sc_loss(
                    output_element.size_converter(output_shape="B, 1, H, W")
                )
            if self.gc_loss and self.cfg.gap_closing_loss_weight > 0:
                loss = loss + self.cfg.gap_closing_loss_weight * self.gc_loss(
                    output_element.size_converter(output_shape="B, 1, H, W")
                )

            loss.backward()
            self.generator.step(scheduler_param=real_loss)
            self.generator.model.eval()

            self.record["fake_loss"] = loss.item()
            self.generator.save()

            self.record["epoch"] = epoch
            self.record["time"] = round(time() - start, 1)
            self.record.save(f"{epoch} times")

            logger.info(
                f"End {epoch} of {self.cfg.epochs}, "
                f"Loss: {self.record('real_loss'):4f}, "
                f"Time: {self.record('time')} s, "
                f"jump: {jump}"
            )

        logger.success(f"訓練完成！最小 Loss: {self.record.custom('real_loss', min)}")
