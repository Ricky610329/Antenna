"""Hydra structured config dataclasses。"""

from dataclasses import dataclass, field
from typing import Any, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class EnvironmentConfig:
    device: str = "cpu"
    network_drive_letter: str = "T:"
    rootdir: str = ""
    dataset_path: str = ""


@dataclass
class PatternConfig:
    coordinate: list[int] = field(default_factory=lambda: [0, 25, 0, 25])


@dataclass
class ResponseTargetConfig:
    side: float = MISSING
    center: float = MISSING
    width: list[int] = MISSING


@dataclass
class ResponseLabelConfig:
    target: ResponseTargetConfig = MISSING
    loss_fn: str = MISSING
    loss_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResponseConfig:
    labels: list[str] = MISSING
    x: str = "n257"
    label_configs: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizerConfig:
    _target_: str = "torch.optim.Adam"
    lr: float = 0.005
    betas: list[float] = field(default_factory=lambda: [0.5, 0.999])


@dataclass
class SchedulerConfig:
    _target_: str = "none"


@dataclass
class SurrogateConfig:
    type: str = "old"
    pretrain_path: str | None = None
    training_mode: str = "one_data"
    hfss_min_loss: float = 0.1
    hfss_max_epoch: int = 20000
    hfss_lr: float = 0.001


@dataclass
class TrainConfig:
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    pattern: PatternConfig = field(default_factory=PatternConfig)
    response: ResponseConfig = MISSING
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    surrogate: SurrogateConfig = field(default_factory=SurrogateConfig)

    # 訓練參數
    epochs: int = 1000
    patience: int = 10
    mutation_rate: float = 0.001
    experiment_name: str = "[Patch-Single-{device}-{hash_id}]"

    # 模型設定
    model: str = "sigmoid_gen"
    simulator: str = "single_port"

    # 正則化權重
    total_variation_loss_weight: float = 0.0
    island_suppression_loss_weight: float = 0.0
    spectral_connectivity_loss_weight: float = 0.0
    gap_closing_loss_weight: float = 0.0

    # wandb
    wandb_project: str | None = None
    wandb_name: str | None = None


def register_configs():
    """向 Hydra ConfigStore 註冊 structured configs。"""
    cs = ConfigStore.instance()
    cs.store(name="train_config", node=TrainConfig)
