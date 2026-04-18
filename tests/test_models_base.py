"""`antenna.models.base.Models` 的單元測試。

僅使用 `torch.nn.Linear` 與 CPU tensor，不依賴 HFSS / CUDA。
"""

import pytest
import torch
from torch import nn, optim

from antenna.models.base import Models


def _build_models(tmp_path, name: str = "unit_model"):
    """建立可供測試的 `Models` 實例。"""
    model = nn.Linear(4, 2)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    criterion = nn.MSELoss()

    return Models(
        name=name,
        rootdir=str(tmp_path),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device="cpu",
    )


class TestDeviceProperty:
    """`device` property 的快取與 setter 行為。"""

    def test_device_cached_on_init(self, tmp_path):
        models = _build_models(tmp_path)
        assert models.device is not None
        assert str(models.device) == "cpu"

    def test_device_property_returns_cached_value_without_parameter_iteration(self, tmp_path):
        models = _build_models(tmp_path)
        # 連續取得 device 不應因為每次 iterate parameters 而變更結果
        first = models.device
        second = models.device
        assert first == second
        assert first is models._device

    def test_device_setter_moves_model_and_updates_cache(self, tmp_path):
        models = _build_models(tmp_path)
        models.device = "cpu"
        # 設定後 model 實際 parameters 應與 cache 一致
        assert next(models.model.parameters()).device == models.device

    def test_float_tensor_cpu(self, tmp_path):
        models = _build_models(tmp_path)
        assert models.FloatTensor is torch.FloatTensor


class TestCheckpointRoundTrip:
    """save / load checkpoint 的往返測試。"""

    def test_save_produces_file(self, tmp_path):
        models = _build_models(tmp_path)
        saved = models.save()
        assert saved.exists()
        assert saved.suffix == ".pth"

    def test_save_as_custom_path(self, tmp_path):
        models = _build_models(tmp_path)
        custom = tmp_path / "custom.pth"
        saved = models.save_as(custom)
        assert saved.exists()
        assert str(saved).endswith("custom.pth")

    def test_load_restores_model_state(self, tmp_path):
        models = _build_models(tmp_path)

        # 修改 model 權重，儲存
        with torch.no_grad():
            for param in models.model.parameters():
                param.fill_(0.1234)
        models.save()
        saved_state = {k: v.clone() for k, v in models.model.state_dict().items()}

        # 重置權重，確認有差異
        with torch.no_grad():
            for param in models.model.parameters():
                param.fill_(0.0)
        assert not torch.allclose(next(models.model.parameters()), next(iter(saved_state.values())))

        # 讀回後，權重應與儲存前一致
        models.load()
        for key, saved_tensor in saved_state.items():
            assert torch.allclose(models.model.state_dict()[key], saved_tensor)

    def test_load_mismatch_raises(self, tmp_path):
        models = _build_models(tmp_path)
        models.save()

        # 更換 criterion 讓 __str__() 改變，模擬 checkpoint title mismatch
        models.criterion = nn.L1Loss()
        with pytest.raises(RuntimeError, match="correct model file"):
            models.load()

    def test_load_force_bypasses_title_check(self, tmp_path):
        models = _build_models(tmp_path)
        models.save()

        models.criterion = nn.L1Loss()
        # force=True 應直接套用 checkpoint
        models.load(force=True)

    def test_checkpoint_build_structure(self, tmp_path):
        models = _build_models(tmp_path)
        ckpt = models.checkpoint(load=False)
        assert "title" in ckpt
        assert "model_state_dict" in ckpt
        assert "optimizer_state_dict" in ckpt
        assert "scheduler_state_dict" in ckpt
        assert "record_state_dict" in ckpt
        assert "device" in ckpt


class TestNaming:
    """name 與 change() 行為。"""

    def test_explicit_name_sets_immediately(self, tmp_path):
        models = _build_models(tmp_path, name="no_placeholder")
        assert models.name == "no_placeholder"

    def test_placeholder_defers_name_until_change(self, tmp_path):
        model = nn.Linear(4, 2)
        models = Models(
            name="run_{label}",
            rootdir=str(tmp_path),
            model=model,
            optimizer=optim.SGD(model.parameters(), lr=0.01),
            scheduler=optim.lr_scheduler.StepLR(optim.SGD(model.parameters(), lr=0.01), step_size=1),
            criterion=nn.MSELoss(),
            device="cpu",
        )
        assert models.name is None
        models.change("epoch5")
        assert models.name == "run_epoch5"

    def test_model_file_requires_name(self, tmp_path):
        model = nn.Linear(4, 2)
        models = Models(
            name="run_{label}",
            rootdir=str(tmp_path),
            model=model,
            optimizer=optim.SGD(model.parameters(), lr=0.01),
            scheduler=optim.lr_scheduler.StepLR(optim.SGD(model.parameters(), lr=0.01), step_size=1),
            criterion=nn.MSELoss(),
            device="cpu",
        )
        with pytest.raises(AssertionError):
            _ = models.model_file


class TestOptimizerSchedulerStep:
    """整合 step（optimizer + scheduler）不 crash。"""

    def test_step_runs_optimizer_and_scheduler(self, tmp_path):
        models = _build_models(tmp_path)

        # 建立 graph：計算 loss，backward 後 step
        x = torch.randn(3, 4)
        y = torch.randn(3, 2)
        pred = models(x)
        loss = models.criterion(pred, y)
        loss.backward()

        before_lr = models.optimizer.param_groups[0]["lr"]
        models.step()
        after_lr = models.optimizer.param_groups[0]["lr"]
        # StepLR(step_size=1, gamma=0.9)：step 一次後 lr 應 * 0.9
        assert after_lr == pytest.approx(before_lr * 0.9)

    def test_step_without_scheduler(self, tmp_path):
        model = nn.Linear(4, 2)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        models = Models(
            name="no_scheduler",
            rootdir=str(tmp_path),
            model=model,
            optimizer=optimizer,
            scheduler=None,
            criterion=nn.MSELoss(),
            device="cpu",
        )

        x = torch.randn(3, 4)
        y = torch.randn(3, 2)
        loss = models.criterion(models(x), y)
        loss.backward()
        # 無 scheduler 也不該 crash
        models.step()


class TestRequiresGrad:
    def test_requires_grad_toggle(self, tmp_path):
        models = _build_models(tmp_path)
        assert models.requires_grad(False) is False
        assert all(not p.requires_grad for p in models.model.parameters())
        assert models.requires_grad(True) is True
        assert all(p.requires_grad for p in models.model.parameters())

    def test_requires_grad_switches_train_eval(self, tmp_path):
        models = _build_models(tmp_path)
        models.requires_grad(True, train=True)
        assert models.model.training is True
        models.requires_grad(True, train=False)
        assert models.model.training is False


class TestCallForward:
    def test_call_delegates_to_model(self, tmp_path):
        models = _build_models(tmp_path)
        x = torch.randn(1, 4)
        out = models(x)
        assert out.shape == (1, 2)


class TestStrRepresentation:
    def test_str_contains_component_names(self, tmp_path):
        models = _build_models(tmp_path)
        text = str(models)
        assert "Linear" in text
        assert "SGD" in text
        assert "StepLR" in text
        assert "MSELoss" in text
