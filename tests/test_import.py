"""基本 import 測試，確認套件結構正常。"""


def test_core_imports():
    from antenna.core.pattern import AntennaPattern
    from antenna.core.response import AntennaResponse, MultiResponses, TargetResponse

    assert AntennaPattern is not None
    assert AntennaResponse is not None
    assert MultiResponses is not None
    assert TargetResponse is not None


def test_top_level_reexports():
    from antenna import AntennaPattern, AntennaResponse, MultiResponses, TargetResponse

    assert AntennaPattern is not None
    assert AntennaResponse is not None


def test_utils_imports():
    from antenna.utils.config import Config, MultiConfig, config
    from antenna.utils.figure import Figure
    from antenna.utils.path import Path
    from antenna.utils.record import Record

    assert isinstance(config, Config)
    assert Path is not None
    assert Record is not None
    assert Figure is not None
    assert MultiConfig is not None


def test_models_imports():
    from antenna.models import Models
    from antenna.models.base import Models as ModelsBase
    from antenna.models.components import BiScaleNorm

    assert Models is ModelsBase
    assert BiScaleNorm is not None


def test_generators_imports():
    from antenna.models.generators import (
        CVAE,
        SPGEN,
        GumbelSigmoidGEN,
        OldGEN,
        SigmoidGEN,
    )

    assert SigmoidGEN is not None
    assert GumbelSigmoidGEN is not None
    assert OldGEN is not None
    assert SPGEN is not None
    assert CVAE is not None


def test_losses_imports():
    from antenna.losses.interval import custom_loss_interval
    from antenna.losses.mirror import FlipMode, gumbel_sinkhorn_rectangular, mirror
    from antenna.losses.patch_losses import (
        custom_loss_g,
        custom_loss_minmax,
        custom_loss_r,
        interval_loss,
    )
    from antenna.losses.regularization import (
        FeedReachability,
        GapClosingLoss,
        SpectralConnectivityLoss,
    )

    assert custom_loss_interval is not None
    assert mirror is not None
    assert custom_loss_r is not None
    assert SpectralConnectivityLoss is not None


def test_schedulers_imports():
    from antenna.schedulers.adaptive_cyclical import AdaptiveCyclicalScheduler

    assert AdaptiveCyclicalScheduler is not None


def test_backward_compat_functions():
    """確認 antenna.functions re-export hub 仍然可用。"""
    from antenna.functions import (
        AdaptiveCyclicalScheduler,
        FeedReachability,
        GapClosingLoss,
        SpectralConnectivityLoss,
        custom_loss_interval,
        mirror,
    )

    assert AdaptiveCyclicalScheduler is not None
    assert mirror is not None


def test_configs_imports():
    from antenna.configs.schema import TrainConfig, register_configs

    assert TrainConfig is not None
    assert register_configs is not None


def test_simulators_reexport_hub():
    """antenna.simulators 是 re-export hub，必須涵蓋四個模擬器。"""
    from antenna.simulators import (
        DualPortSimulator,
        PatchSimulator,
        RISSimulator,
        SinglePortSimulator,
    )

    assert PatchSimulator is not None
    assert SinglePortSimulator is not None
    assert DualPortSimulator is not None
    assert RISSimulator is not None


def test_smodels_shim():
    """antenna.smodels 為向後相容 shim，需 re-export 代理模型相關類別。"""
    from antenna.smodels import HFSSNet, OldSM, SurrogateModel, UNetSM

    assert SurrogateModel is not None
    assert HFSSNet is not None
    assert OldSM is not None
    assert UNetSM is not None


def test_ranger_import():
    """antenna.ranger 仍提供 Ranger optimizer 供既有訓練腳本使用。"""
    from antenna.ranger import Ranger

    assert Ranger is not None


def test_main_module_entry():
    """確認 python -m antenna 的 entry point 可被匯入。"""
    from antenna.__main__ import main

    assert callable(main)
