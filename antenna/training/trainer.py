"""
統一訓練迴圈，取代原始 train_*.py 腳本。

使用方式::

    from antenna.training.trainer import Trainer
    trainer = Trainer(cfg)
    trainer.run()
"""

from time import time
from typing import Any, Callable

import torch
from loguru import logger
from omegaconf import DictConfig

from antenna.core.pattern import AntennaPattern
from antenna.core.response import AntennaResponse
from antenna.losses.patch_losses import custom_loss_g, custom_loss_minmax, custom_loss_r
from antenna.losses.regularization import FeedReachability, GapClosingLoss, SpectralConnectivityLoss
from antenna.models.base import Models
from antenna.models.generators.gumbel_sigmoid_gen import GumbelSigmoidGEN
from antenna.models.generators.sigmoid_gen import SigmoidGEN
from antenna.ris import custom_loss as ris_custom_loss
from antenna.schedulers.adaptive_cyclical import AdaptiveCyclicalScheduler
from antenna.smodels import OldSM
from antenna.utils.config import config
from antenna.utils.data import DataManager
from antenna.utils.record import Record
from antenna.utils.web import connect_default_drive

# 損失函數對應表（YAML `loss_fn` 欄位映射至實作）。
LOSS_FN_REGISTRY: dict[str, Callable[..., torch.Tensor]] = {
    "custom_loss_minmax": custom_loss_minmax,
    "custom_loss_r": custom_loss_r,
    "custom_loss_g": custom_loss_g,
    # RIS 專用 loss，供 ``conf/response/ris.yaml`` 使用。
    "custom_loss": ris_custom_loss,
}

# 模型對應表（YAML `model` 欄位映射至生成器類別）。
MODEL_REGISTRY: dict[str, type[torch.nn.Module]] = {
    "sigmoid_gen": SigmoidGEN,
    "gumbel_sigmoid_gen": GumbelSigmoidGEN,
}

# 模擬器對應表（YAML `simulator` 欄位映射至模擬器 factory）。
# 透過 lazy import 避免在測試環境沒有 HFSS COM 時載入失敗。


def _single_port_factory(cfg: DictConfig, result_path):
    from antenna.patch.patch_simulator.single_port import SinglePortSimulator

    return SinglePortSimulator(record_path=result_path)


def _dual_port_factory(cfg: DictConfig, result_path):
    from antenna.patch.patch_simulator.dual_port import DualPortSimulator

    return DualPortSimulator(record_path=result_path)


def _ris_factory(cfg: DictConfig, result_path):
    from antenna.ris.simulate_ris import RISSimulator

    coord = cfg.pattern.coordinate
    return RISSimulator(element_num=coord[1] - coord[0])


SIMULATOR_REGISTRY: dict[str, Callable[[DictConfig, Any], Any]] = {
    "single_port": _single_port_factory,
    "dual_port": _dual_port_factory,
    "ris": _ris_factory,
}


class Trainer:
    """統一訓練器，從 Hydra DictConfig 驅動。

    執行 ``__init__`` 即完成所有前置設定（環境、路徑、模擬器、模型與追蹤），
    呼叫 :meth:`run` 後才進入真正的 epoch loop。
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg: DictConfig = cfg
        # 依序建立：環境 → 路徑 → 追蹤（含 Record）→ 模擬器 → 天線 → 模型。
        # 追蹤放在模型之前，讓 resume 時 _setup_models() 可以讀取 self.record。
        self._setup_environment()
        self._setup_paths()
        self._setup_tracking()
        self._setup_simulator()
        self._setup_antenna()
        self._setup_models()

    # ── 初始化階段 ──────────────────────────────────

    def _setup_environment(self) -> None:
        """設定裝置、網路磁碟。"""
        config.device = self.cfg.environment.device
        connect_default_drive(self.cfg.environment.network_drive_letter)

    def _setup_paths(self) -> None:
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

    def _setup_simulator(self) -> None:
        """依據 ``cfg.simulator`` 建立並註冊模擬器。"""
        sim_type = self.cfg.simulator
        factory = SIMULATOR_REGISTRY.get(sim_type)
        if factory is None:
            raise ValueError(f"未知的模擬器: {sim_type}（可用: {sorted(SIMULATOR_REGISTRY)})")
        self.simulator = factory(self.cfg, self.result_path)
        AntennaPattern.register_simulator(self.simulator)

    def _setup_antenna(self) -> None:
        """設定 AntennaPattern 與 AntennaResponse。"""
        coord = tuple(self.cfg.pattern.coordinate)
        AntennaPattern.setDefaultCoordinate(coord)

        # 設定 Response
        resp_cfg = self.cfg.response
        AntennaResponse.registerLabels(*resp_cfg.labels, x=resp_cfg.x)

        # 註冊每個 label 的 target 與 loss hook
        self._targets: dict[str, torch.Tensor] = {}
        for label, label_cfg in resp_cfg.label_configs.items():
            target_cfg = label_cfg.target
            target_tensor = AntennaResponse.registerTargetResponse(
                target_cfg.side, target_cfg.center, target_cfg.width, label=label
            )
            self._targets[label] = target_tensor

            loss_fn = LOSS_FN_REGISTRY.get(label_cfg.loss_fn)
            if loss_fn is None:
                raise ValueError(f"未知的損失函數: {label_cfg.loss_fn}（可用: {sorted(LOSS_FN_REGISTRY)})")
            AntennaResponse.registerLossHook(loss_fn, label=label, target=target_tensor, **label_cfg.loss_params)

    def _setup_models(self) -> None:
        """建立生成器、優化器、排程器、代理模型。"""
        # 生成器
        model_cls = MODEL_REGISTRY.get(self.cfg.model)
        if model_cls is None:
            raise ValueError(f"未知的模型: {self.cfg.model}（可用: {sorted(MODEL_REGISTRY)})")
        self.model = model_cls()

        # 優化器
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.cfg.optimizer.lr,
            betas=tuple(self.cfg.optimizer.betas),
        )

        # 排程器
        self.scheduler = self._build_scheduler(self.cfg.scheduler)

        # Models 封裝
        self.generator: Models = Models(
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

        # 載入檢查點（需 self.record 已於 _setup_tracking 中建立）
        if self.continue_run and hasattr(self, "record") and ("epoch" in self.record):
            self.generator.change(self.record("epoch"), load=True)
            self.smodel.load()

    def _build_scheduler(self, sched_cfg: DictConfig):
        """依 ``sched_cfg._target_`` 字串建立對應的 scheduler。

        - ``"none"`` 或未知 target → 回傳 ``None``。
        - 缺少 ``_target_`` 欄位時視為未設定，回傳 ``None``。
        """
        target = sched_cfg.get("_target_", "none") if sched_cfg is not None else "none"
        if target == "none":
            return None
        if "AdaptiveCyclicalScheduler" in target:
            return AdaptiveCyclicalScheduler(
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
                tau_callback=self._make_tau_callback(),
            )
        if "ReduceLROnPlateau" in target:
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                factor=sched_cfg.get("factor", 0.5),
                patience=sched_cfg.get("patience", 10),
                min_lr=sched_cfg.get("min_lr", 1e-6),
            )
        logger.warning(f"未知的 scheduler target: {target!r}，不使用 scheduler。")
        return None

    def _make_tau_callback(self) -> Callable[[float], None] | None:
        """若 generator 有可學習 tau 屬性（例如 GumbelSigmoidGEN），回傳一個
        讓 scheduler 直接覆寫 tau 的 callback；否則回傳 None，讓 scheduler
        使用預設 callback（僅更新 AntennaPattern.tau）。

        注意：scheduler 驅動的 tau 會 in-place 覆蓋梯度累積的更新，讓 tau
        退火完全 schedule-driven；這是刻意設計。
        """
        if not hasattr(self.model, "tau"):
            return None

        model = self.model

        def _cb(tau_value: float) -> None:
            with torch.no_grad():
                model.tau.data.fill_(float(tau_value))

        return _cb

    def _setup_tracking(self) -> None:
        """初始化記錄、資料集、正則化。"""
        self.record: Record = Record("temp", rootdir=self.result_path, load=True)
        self.online_dataset = DataManager("online", rootdir=self.result_path)

        # 正則化損失
        self.sc_loss = SpectralConnectivityLoss() if self.cfg.spectral_connectivity_loss_weight > 0 else None
        self.gc_loss = GapClosingLoss() if self.cfg.gap_closing_loss_weight > 0 else None
        self.r_feed = FeedReachability.single_feed()

    # ── 訓練迴圈 ──────────────────────────────────

    def run(self) -> None:
        """主訓練迴圈。"""
        epoch = self.record("epoch", 0)
        jump = 0

        logger.info(f"開始訓練: {self.cfg.experiment_name}")
        logger.info(f"模型: {self.cfg.model}, 設備: {self.cfg.environment.device}, Epochs: {self.cfg.epochs}")

        # 開啟模擬器（HFSS COM 或 RIS）
        if hasattr(self.simulator, "open"):
            self.simulator.open()

        while epoch < self.cfg.epochs + 1:
            start = time()
            epoch += 1
            self.generator.change(epoch)

            # 開始新的模擬回合
            if hasattr(self.simulator, "start"):
                self.simulator.start(epoch)

            logger.info(f"Start {epoch} of {self.cfg.epochs}")

            self.generator.requires_grad(True, train=True)
            self.generator.optimizer.zero_grad()

            # ── Early stopping & rollback ──
            # 記錄 generator 當下的 tau（若有），供退火曲線檢視；沒有 tau 屬性則存 None。
            self.record["tau"] = float(self.model.tau.detach().cpu().item()) if hasattr(self.model, "tau") else None
            if self.record.early_stop("real_loss", self.cfg.patience):
                # Rollback：Models.load 會同步還原 scheduler state，會把 tau 退火進度也
                # 拉回去。為讓退火持續向前，這裡先備份 scheduler state，rollback 後再
                # 還原回來 —— 只有 model/optimizer 真的退回 best，scheduler 不動。
                sched_state = self.scheduler.state_dict() if self.scheduler else None
                self.generator.change(
                    self.record.find("real_loss", self.record("min_loss", float("inf")), "epoch"),
                    save=True,
                    load=True,
                )
                if self.scheduler and sched_state is not None:
                    self.scheduler.load_state_dict(sched_state)
                self.smodel.train_by_datas(self.online_dataset)
                target_in = AntennaResponse.target.concat().to(config.device)
                output_element = AntennaPattern(self.generator(target_in))
                self.record["mutation"] = self.record("min_loss")
            else:
                target_in = AntennaResponse.target.concat().to(config.device)
                output_element = AntennaPattern(self.generator(target_in))
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

            # 結束模擬回合
            if hasattr(self.simulator, "end"):
                self.simulator.end()
            if hasattr(self.simulator, "clean"):
                self.simulator.clean()

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
