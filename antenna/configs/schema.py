"""Hydra structured config dataclasses。

所有欄位都必須對應到 `antenna/conf/` 下的 YAML 設定，
並對應到 `antenna.training.trainer.Trainer` 的 `self.cfg.xxx` 存取路徑。
"""

from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class EnvironmentConfig:
    """執行環境設定（裝置、網路磁碟、輸出根目錄）。"""

    device: str = "cpu"
    network_drive_letter: str = "T:"
    rootdir: str = ""


@dataclass
class PatternConfig:
    """AntennaPattern 預設座標範圍。"""

    coordinate: list[int] = field(default_factory=lambda: [0, 25, 0, 25])


@dataclass
class ResponseTargetConfig:
    """單一 label 的目標響應。"""

    side: float = MISSING
    center: float = MISSING
    width: list[int] = MISSING


@dataclass
class ResponseLabelConfig:
    """單一 label 的 target + loss 設定。"""

    target: ResponseTargetConfig = MISSING
    loss_fn: str = MISSING
    loss_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResponseConfig:
    """AntennaResponse 的 labels / x 軸 / 各 label 設定。"""

    labels: list[str] = MISSING
    x: str = "n257"
    label_configs: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizerConfig:
    """torch.optim.Adam 預設參數。"""

    _target_: str = "torch.optim.Adam"
    lr: float = 0.005
    betas: list[float] = field(default_factory=lambda: [0.5, 0.999])


@dataclass
class SchedulerConfig:
    """Scheduler 設定。`_target_ == "none"` 表示不使用 scheduler。"""

    _target_: str = "none"


@dataclass
class SurrogateConfig:
    """代理模型（OldSM）與 HFSS 訓練迴圈設定。

    目前僅使用 HFSS 相關欄位；trainer 透過 `antenna.utils.config` 的
    `HFSS.*` key 讀取 `hfss_lr` / `hfss_min_loss` / `hfss_max_epoch`。
    """

    hfss_min_loss: float = 0.1
    hfss_max_epoch: int = 20000
    hfss_lr: float = 0.001


@dataclass
class TrainConfig:
    """Hydra 訓練主設定，對應 `antenna/conf/config.yaml`。"""

    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    pattern: PatternConfig = field(default_factory=PatternConfig)
    response: ResponseConfig = MISSING
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    surrogate: SurrogateConfig = field(default_factory=SurrogateConfig)

    # 訓練參數
    epochs: int = 1000
    patience: int = 10
    experiment_name: str = "[Patch-Single-{device}-{hash_id}]"

    # 模型 / 模擬器類型（對應 trainer.py 的 MODEL_REGISTRY 與 simulator switch）
    model: str = "sigmoid_gen"
    simulator: str = "single_port"

    # 正則化權重（0 表示關閉）
    total_variation_loss_weight: float = 0.0
    island_suppression_loss_weight: float = 0.0
    spectral_connectivity_loss_weight: float = 0.0
    gap_closing_loss_weight: float = 0.0


def register_configs() -> None:
    """向 Hydra ConfigStore 註冊 structured configs。"""
    cs = ConfigStore.instance()
    cs.store(name="train_config", node=TrainConfig)
