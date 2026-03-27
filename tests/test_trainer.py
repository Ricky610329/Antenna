"""Trainer class 基本測試。"""


def test_trainer_import():
    from antenna.training.trainer import Trainer

    assert Trainer is not None


def test_loss_fn_registry():
    from antenna.training.trainer import LOSS_FN_REGISTRY

    assert "custom_loss_minmax" in LOSS_FN_REGISTRY
    assert "custom_loss_r" in LOSS_FN_REGISTRY
    assert "custom_loss_g" in LOSS_FN_REGISTRY


def test_model_registry():
    from antenna.training.trainer import MODEL_REGISTRY

    assert "sigmoid_gen" in MODEL_REGISTRY
    assert "gumbel_sigmoid_gen" in MODEL_REGISTRY
