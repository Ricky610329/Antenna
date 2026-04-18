"""Hydra structured config 測試。

驗證：
1. 預設值可合理地透過 `OmegaConf.structured` 實例化。
2. `conf/experiment/*.yaml`（train_single / train_dual / train_ris）能透過
   `hydra.compose` merge 進 schema 而不 raise。
3. CLI override（例如 `epochs=2000`）可正確套用。
4. schema 與 YAML 的欄位已對齊（沒有死欄位）。
"""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from antenna.configs.schema import (
    EnvironmentConfig,
    OptimizerConfig,
    PatternConfig,
    SchedulerConfig,
    SurrogateConfig,
    TrainConfig,
    register_configs,
)

# ── 預設值 ──────────────────────────────────────────


def test_environment_config_defaults():
    """EnvironmentConfig 預設值合理。"""
    env = OmegaConf.structured(EnvironmentConfig)
    assert env.device == "cpu"
    assert env.network_drive_letter == "T:"
    assert env.rootdir == ""


def test_pattern_config_defaults():
    """PatternConfig 預設座標為 25x25。"""
    pattern = OmegaConf.structured(PatternConfig)
    assert list(pattern.coordinate) == [0, 25, 0, 25]


def test_optimizer_config_defaults():
    """OptimizerConfig 預設使用 Adam。"""
    opt = OmegaConf.structured(OptimizerConfig)
    assert opt._target_ == "torch.optim.Adam"
    assert opt.lr == 0.005
    assert list(opt.betas) == [0.5, 0.999]


def test_scheduler_config_defaults():
    """SchedulerConfig 預設為 'none'（不使用 scheduler）。"""
    sched = OmegaConf.structured(SchedulerConfig)
    assert sched._target_ == "none"


def test_surrogate_config_defaults():
    """SurrogateConfig 預設為 OldSM 設定。"""
    sur = OmegaConf.structured(SurrogateConfig)
    assert sur.type == "old"
    assert sur.pretrain_path is None
    assert sur.training_mode == "one_data"
    assert sur.hfss_min_loss == pytest.approx(0.1)
    assert sur.hfss_max_epoch == 20000
    assert sur.hfss_lr == pytest.approx(0.001)


def test_train_config_defaults():
    """TrainConfig 的 top-level 欄位與訓練參數預設值。"""
    cfg = OmegaConf.structured(TrainConfig)
    assert cfg.epochs == 1000
    assert cfg.patience == 10
    assert cfg.model == "sigmoid_gen"
    assert cfg.simulator == "single_port"
    # 正則化預設全關
    assert cfg.total_variation_loss_weight == 0.0
    assert cfg.island_suppression_loss_weight == 0.0
    assert cfg.spectral_connectivity_loss_weight == 0.0
    assert cfg.gap_closing_loss_weight == 0.0


# ── Hydra compose 整合 ─────────────────────────────


CONF_DIR = str((Path(__file__).resolve().parent.parent / "antenna" / "conf").resolve())


@pytest.fixture(autouse=True)
def _register_schema():
    """在每個測試執行前註冊 structured configs。"""
    register_configs()


@pytest.mark.parametrize(
    "experiment",
    ["train_single", "train_dual", "train_ris"],
)
def test_experiment_compose(experiment):
    """驗 `conf/experiment/*.yaml` 能成功 merge 至 schema。"""
    with initialize_config_dir(version_base=None, config_dir=CONF_DIR):
        cfg = compose(config_name="config", overrides=[f"+experiment={experiment}"])
        assert cfg is not None
        assert cfg.epochs > 0
        assert cfg.response is not None
        assert len(cfg.response.labels) >= 1


def test_experiment_train_single_fields():
    """train_single 實驗使用 sigmoid_gen 與 single_port。"""
    with initialize_config_dir(version_base=None, config_dir=CONF_DIR):
        cfg = compose(config_name="config", overrides=["+experiment=train_single"])
        assert cfg.model == "sigmoid_gen"
        assert cfg.simulator == "single_port"
        assert cfg.epochs == 1000
        # train_single 使用 AdaptiveCyclicalScheduler
        assert "AdaptiveCyclical" in cfg.scheduler._target_
        # response 會切換為 single_port（S11, Gain）
        labels = list(cfg.response.labels)
        assert "S11" in labels
        assert "Gain" in labels


def test_experiment_train_dual_fields():
    """train_dual 實驗使用 gumbel_sigmoid_gen 與 dual_port。"""
    with initialize_config_dir(version_base=None, config_dir=CONF_DIR):
        cfg = compose(config_name="config", overrides=["+experiment=train_dual"])
        assert cfg.model == "gumbel_sigmoid_gen"
        assert cfg.simulator == "dual_port"
        assert cfg.epochs == 2000
        assert cfg.patience == 100
        labels = list(cfg.response.labels)
        assert "S11" in labels
        assert "S21" in labels
        assert "S22" in labels


def test_experiment_train_ris_fields():
    """train_ris 實驗覆蓋 pattern.coordinate 為 40x40。"""
    with initialize_config_dir(version_base=None, config_dir=CONF_DIR):
        cfg = compose(config_name="config", overrides=["+experiment=train_ris"])
        assert cfg.model == "gumbel_sigmoid_gen"
        assert cfg.simulator == "ris"
        assert list(cfg.pattern.coordinate) == [0, 40, 0, 40]
        # ris 使用 ReduceLROnPlateau
        assert "ReduceLROnPlateau" in cfg.scheduler._target_


# ── CLI override ───────────────────────────────────


def test_cli_override_epochs():
    """epochs=2000 override 可正確套用。"""
    with initialize_config_dir(version_base=None, config_dir=CONF_DIR):
        cfg = compose(
            config_name="config",
            overrides=["+experiment=train_single", "epochs=2000"],
        )
        assert cfg.epochs == 2000


def test_cli_override_environment_device():
    """environment.device override 可正確套用。"""
    with initialize_config_dir(version_base=None, config_dir=CONF_DIR):
        cfg = compose(
            config_name="config",
            overrides=[
                "+experiment=train_single",
                "environment.device=cuda:0",
            ],
        )
        assert cfg.environment.device == "cuda:0"


def test_cli_override_regularization_weights():
    """正則化權重可透過 override 調整。"""
    with initialize_config_dir(version_base=None, config_dir=CONF_DIR):
        cfg = compose(
            config_name="config",
            overrides=[
                "+experiment=train_single",
                "total_variation_loss_weight=0.5",
                "island_suppression_loss_weight=0.25",
            ],
        )
        assert cfg.total_variation_loss_weight == pytest.approx(0.5)
        assert cfg.island_suppression_loss_weight == pytest.approx(0.25)


# ── Schema 與 YAML 對齊 ────────────────────────────


def test_schema_no_dead_fields():
    """已知的死欄位（dataset_path / mutation_rate / wandb_*）不應存在於 schema。"""
    cfg = OmegaConf.structured(TrainConfig)
    assert "dataset_path" not in cfg.environment
    assert "mutation_rate" not in cfg
    assert "wandb_project" not in cfg
    assert "wandb_name" not in cfg


def test_config_yaml_no_dead_fields():
    """config.yaml 不應含已棄用的欄位。"""
    conf_path = Path(__file__).resolve().parent.parent / "antenna" / "conf"
    text = (conf_path / "config.yaml").read_text(encoding="utf-8")
    assert "mutation_rate" not in text
    env_text = (conf_path / "environment" / "default.yaml").read_text(encoding="utf-8")
    assert "dataset_path" not in env_text
