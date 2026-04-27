"""`antenna.training.trainer.Trainer` 的單元測試。

本測試模組透過 mock 掉外部依賴（HFSS 模擬器、網路磁碟、代理模型），
以純 Python / CPU 的方式驗證 Trainer 的 Hydra 驅動流程。

主要涵蓋：
- registry 完整性（loss / model / simulator）
- Trainer 初始化（single_port / dual_port / ris）各階段
- scheduler 建立（none / AdaptiveCyclical / ReduceLROnPlateau / 未知 target / 缺欄位）
- 錯誤路徑（未知模型、未知 simulator、未知 loss_fn）
- checkpoint / resume 行為
- import / registry 鎖定測試
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf

from antenna.training import trainer as trainer_module
from antenna.training.trainer import (
    LOSS_FN_REGISTRY,
    MODEL_REGISTRY,
    SIMULATOR_REGISTRY,
    Trainer,
)

# ── 共用 helpers ────────────────────────────────────


def _make_cfg(
    *,
    model: str = "sigmoid_gen",
    simulator: str = "single_port",
    coordinate: list[int] | None = None,
    scheduler_target: str = "none",
    response_labels: list[str] | None = None,
    response_x: str = "n257",
    label_configs: dict[str, dict[str, Any]] | None = None,
    epochs: int = 2,
    patience: int = 10,
    total_variation: float = 0.0,
    island_suppression: float = 0.0,
    spectral_connectivity: float = 0.0,
    gap_closing: float = 0.0,
):
    """建立適用於 Trainer 測試的 DictConfig。預設為 single_port + sigmoid。"""
    coordinate = coordinate if coordinate is not None else [0, 4, 0, 4]
    response_labels = response_labels if response_labels is not None else ["S11"]
    label_configs = (
        label_configs
        if label_configs is not None
        else {
            "S11": {
                "target": {"side": 0, "center": -10, "width": [1, 0, 1, 0, 1]},
                "loss_fn": "custom_loss_minmax",
                "loss_params": {"method": "low"},
            }
        }
    )

    return OmegaConf.create(
        {
            "experiment_name": "unit-test",
            "model": model,
            "simulator": simulator,
            "epochs": epochs,
            "patience": patience,
            "total_variation_loss_weight": total_variation,
            "island_suppression_loss_weight": island_suppression,
            "spectral_connectivity_loss_weight": spectral_connectivity,
            "gap_closing_loss_weight": gap_closing,
            "environment": {"device": "cpu", "network_drive_letter": "T:", "rootdir": ""},
            "pattern": {"coordinate": coordinate},
            "optimizer": {"_target_": "torch.optim.Adam", "lr": 0.005, "betas": [0.5, 0.999]},
            "scheduler": {"_target_": scheduler_target},
            "surrogate": {
                "type": "old",
                "pretrain_path": None,
                "training_mode": "one_data",
                "hfss_min_loss": 0.1,
                "hfss_max_epoch": 100,
                "hfss_lr": 0.001,
            },
            "response": {
                "labels": response_labels,
                "x": response_x,
                "label_configs": label_configs,
            },
        }
    )


@pytest.fixture
def mocked_externals(tmp_path, monkeypatch):
    """將 Trainer 對外的依賴全部 mock 掉，回傳 patch 物件方便檢查呼叫情形。"""
    # 避免任何網路磁碟嘗試
    monkeypatch.setattr(trainer_module, "connect_default_drive", MagicMock(return_value=True))

    # get_result_path：回傳 tmp_path 下的資料夾 + continue_run=False
    fake_result_path = MagicMock()
    fake_result_path.joinpath = lambda sub: _FakePath(tmp_path / sub)

    def fake_get_result_path(name, *, rootdir=None, enable_exception_handler=False):
        return _FakePath(tmp_path / name.replace("/", "_")), False

    import antenna as _antenna

    monkeypatch.setattr(_antenna, "get_result_path", fake_get_result_path)

    # SinglePortSimulator / DualPortSimulator / RISSimulator factories：
    # 回傳輕量假物件，避免觸發 HFSS / PyWin32。
    fake_sim = MagicMock(spec=["open", "start", "end", "clean", "__call__"])
    fake_sim.return_value = {"S11": torch.zeros(17), "Gain": torch.zeros(17)}
    monkeypatch.setattr(trainer_module, "_single_port_factory", lambda cfg, path: fake_sim)
    monkeypatch.setattr(trainer_module, "_dual_port_factory", lambda cfg, path: fake_sim)
    monkeypatch.setattr(trainer_module, "_ris_factory", lambda cfg, path: fake_sim)
    # 重新覆蓋 registry（module-level 已建立，指回新 factory）
    monkeypatch.setitem(trainer_module.SIMULATOR_REGISTRY, "single_port", lambda cfg, path: fake_sim)
    monkeypatch.setitem(trainer_module.SIMULATOR_REGISTRY, "dual_port", lambda cfg, path: fake_sim)
    monkeypatch.setitem(trainer_module.SIMULATOR_REGISTRY, "ris", lambda cfg, path: fake_sim)

    # OldSM：換成 MagicMock 避免真的建立 HFSSNet + Ranger
    fake_smodel = MagicMock()
    fake_smodel.return_value = MagicMock()
    fake_smodel.load = MagicMock()
    monkeypatch.setattr(trainer_module, "OldSM", MagicMock(return_value=fake_smodel))

    return {"simulator": fake_sim, "smodel": fake_smodel, "tmp_path": tmp_path}


class _FakePath:
    """最小 Path 介面，支援 Trainer 所需的 joinpath / not_exist_create / str。"""

    def __init__(self, path):
        from pathlib import Path as _Path

        self._p = _Path(path)
        self._p.mkdir(parents=True, exist_ok=True)

    def joinpath(self, *sub):
        return _FakePath(self._p.joinpath(*sub))

    def not_exist_create(self):
        self._p.mkdir(parents=True, exist_ok=True)
        return self

    def exists(self):
        return self._p.exists()

    def __str__(self):
        return str(self._p)

    def __fspath__(self):
        return str(self._p)

    def __truediv__(self, other):
        return _FakePath(self._p / other)


# ── Import / Registry 鎖定 ──────────────────────────


def test_trainer_import():
    """Trainer 類別可直接 import。"""
    assert Trainer is not None


def test_trainer_exposed_from_package():
    """antenna.training 應 re-export Trainer。"""
    import antenna.training as pkg

    assert pkg.Trainer is Trainer


def test_loss_fn_registry_keys():
    """LOSS_FN_REGISTRY 需含 patch + ris 全部四種 loss。"""
    assert set(LOSS_FN_REGISTRY) == {
        "custom_loss_minmax",
        "custom_loss_r",
        "custom_loss_g",
        "custom_loss",
    }


def test_loss_fn_registry_values_are_callable():
    """所有 registry 內的 loss fn 都應 callable。"""
    for name, fn in LOSS_FN_REGISTRY.items():
        assert callable(fn), f"{name} 非 callable"


def test_model_registry_keys():
    """MODEL_REGISTRY 應含所有目前支援的生成器。"""
    assert set(MODEL_REGISTRY) == {"sigmoid_gen", "gumbel_sigmoid_gen", "wide_gumbel_sigmoid_gen"}


def test_model_registry_values_are_classes():
    """registry 中的值應為 `torch.nn.Module` 的子類別。"""
    for name, cls in MODEL_REGISTRY.items():
        assert isinstance(cls, type), f"{name} 不是類別"
        assert issubclass(cls, torch.nn.Module), f"{name} 非 nn.Module"


def test_simulator_registry_keys():
    """SIMULATOR_REGISTRY 應含三種 simulator。"""
    assert set(SIMULATOR_REGISTRY) == {"single_port", "dual_port", "ris"}


def test_simulator_registry_values_are_callable():
    """factory 必為 callable。"""
    for name, factory in SIMULATOR_REGISTRY.items():
        assert callable(factory), f"{name} 非 callable"


# ── Trainer 初始化：single_port / dual_port / ris ────


class TestTrainerInit:
    """驗證 Trainer __init__ 於各種 config 下都能建立物件。"""

    def test_init_single_port(self, mocked_externals):
        cfg = _make_cfg(model="sigmoid_gen", simulator="single_port")
        trainer = Trainer(cfg)
        assert trainer.cfg is cfg
        assert trainer.simulator is mocked_externals["simulator"]
        assert trainer.model is not None
        assert trainer.optimizer is not None
        # scheduler 預設為 "none"
        assert trainer.scheduler is None

    def test_init_dual_port(self, mocked_externals):
        cfg = _make_cfg(
            model="gumbel_sigmoid_gen",
            simulator="dual_port",
            response_labels=["S11", "S21"],
            label_configs={
                "S11": {
                    "target": {"side": -2.5, "center": -10, "width": [1, 0, 1, 0, 1]},
                    "loss_fn": "custom_loss_r",
                    "loss_params": {},
                },
                "S21": {
                    "target": {"side": -2.5, "center": -10, "width": [1, 0, 1, 0, 1]},
                    "loss_fn": "custom_loss_r",
                    "loss_params": {},
                },
            },
        )
        trainer = Trainer(cfg)
        assert trainer.simulator is mocked_externals["simulator"]
        assert len(trainer._targets) == 2
        assert set(trainer._targets) == {"S11", "S21"}

    def test_init_ris(self, mocked_externals):
        cfg = _make_cfg(
            model="gumbel_sigmoid_gen",
            simulator="ris",
            coordinate=[0, 4, 0, 4],
            response_labels=["response"],
            response_x="ris",
            label_configs={
                "response": {
                    "target": {"side": -20, "center": 0, "width": [140, 0, 40, 0, 181]},
                    "loss_fn": "custom_loss",
                    "loss_params": {},
                }
            },
        )
        trainer = Trainer(cfg)
        assert trainer.simulator is mocked_externals["simulator"]
        # RIS 的 x 軸總長應為 361
        assert len(trainer._targets["response"]) == 361

    def test_init_registers_simulator_on_antenna_pattern(self, mocked_externals):
        """_setup_simulator 必需呼叫 AntennaPattern.register_simulator。"""
        from antenna.core.pattern import AntennaPattern

        cfg = _make_cfg()
        Trainer(cfg)
        # 註冊後 class-level 的 _simulator 應是 fake_sim
        assert AntennaPattern._simulator is mocked_externals["simulator"]

    def test_setup_order_record_before_models(self, mocked_externals):
        """record 必須在 models 之前建立（供 resume 讀取）。"""
        cfg = _make_cfg()
        trainer = Trainer(cfg)
        # 初始化完畢後同時存在
        assert hasattr(trainer, "record")
        assert hasattr(trainer, "generator")
        assert hasattr(trainer, "smodel")


# ── 錯誤路徑 ────────────────────────────────────────


class TestTrainerInitErrors:
    """不合法的 config 應給出清晰錯誤訊息。"""

    def test_unknown_model(self, mocked_externals):
        cfg = _make_cfg(model="does_not_exist")
        with pytest.raises(ValueError, match="未知的模型"):
            Trainer(cfg)

    def test_unknown_simulator(self, mocked_externals):
        cfg = _make_cfg(simulator="mystery_sim")
        with pytest.raises(ValueError, match="未知的模擬器"):
            Trainer(cfg)

    def test_unknown_loss_fn(self, mocked_externals):
        cfg = _make_cfg(
            label_configs={
                "S11": {
                    "target": {"side": 0, "center": -10, "width": [1, 0, 1, 0, 1]},
                    "loss_fn": "totally_bogus_loss",
                    "loss_params": {},
                }
            }
        )
        with pytest.raises(ValueError, match="未知的損失函數"):
            Trainer(cfg)

    def test_error_message_lists_available_options(self, mocked_externals):
        """錯誤訊息應列出可用的選項，協助使用者 debug。"""
        cfg = _make_cfg(model="nope")
        with pytest.raises(ValueError) as exc:
            Trainer(cfg)
        # 應包含 sigmoid_gen 或 gumbel_sigmoid_gen
        assert "sigmoid_gen" in str(exc.value)


# ── scheduler 建立 ──────────────────────────────────


class TestBuildScheduler:
    """Trainer._build_scheduler 分派邏輯。"""

    def _make_trainer(self, cfg, mocked_externals):
        return Trainer(cfg)

    def test_scheduler_none(self, mocked_externals):
        cfg = _make_cfg(scheduler_target="none")
        trainer = self._make_trainer(cfg, mocked_externals)
        assert trainer.scheduler is None

    def test_scheduler_adaptive_cyclical(self, mocked_externals):
        cfg = _make_cfg(scheduler_target="antenna.schedulers.adaptive_cyclical.AdaptiveCyclicalScheduler")
        # 補充預設欄位（Trainer._build_scheduler 用 .get() 讀取，有 fallback）
        trainer = self._make_trainer(cfg, mocked_externals)
        from antenna.schedulers.adaptive_cyclical import AdaptiveCyclicalScheduler

        assert isinstance(trainer.scheduler, AdaptiveCyclicalScheduler)

    def test_scheduler_reduce_on_plateau(self, mocked_externals):
        cfg = _make_cfg(scheduler_target="torch.optim.lr_scheduler.ReduceLROnPlateau")
        trainer = self._make_trainer(cfg, mocked_externals)
        assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

    def test_scheduler_unknown_target_returns_none(self, mocked_externals):
        """未知的 target 回傳 None 且 log warning，而不是 crash。"""
        cfg = _make_cfg(scheduler_target="some.unknown.Scheduler")
        trainer = self._make_trainer(cfg, mocked_externals)
        assert trainer.scheduler is None

    def test_scheduler_missing_target_field(self, mocked_externals):
        """scheduler config 缺少 _target_ 欄位時應視為 none。"""
        cfg = _make_cfg()
        # 把 scheduler 設為完全空的 dict
        cfg.scheduler = OmegaConf.create({})
        trainer = self._make_trainer(cfg, mocked_externals)
        assert trainer.scheduler is None

    def test_scheduler_adaptive_custom_params(self, mocked_externals):
        cfg = _make_cfg(scheduler_target="antenna.schedulers.adaptive_cyclical.AdaptiveCyclicalScheduler")
        cfg.scheduler.T_0 = 50
        cfg.scheduler.lr_max = 0.002
        trainer = self._make_trainer(cfg, mocked_externals)
        assert trainer.scheduler.T_0 == 50
        # AdaptiveCyclical 會在建構時立刻套用暖身 lr（等於 lr_min），
        # 驗 optimizer 仍綁定同一個 scheduler
        assert trainer.scheduler.optimizer is trainer.optimizer


# ── registry override（確保可擴充） ────────────────


class TestRegistryExtensibility:
    """registry 是普通 dict，允許使用者擴充自訂 loss/model/simulator。"""

    def test_loss_fn_registry_can_be_extended(self):
        assert "my_custom" not in LOSS_FN_REGISTRY
        try:
            LOSS_FN_REGISTRY["my_custom"] = lambda *a, **k: torch.zeros(())
            assert "my_custom" in LOSS_FN_REGISTRY
        finally:
            LOSS_FN_REGISTRY.pop("my_custom", None)

    def test_simulator_registry_can_be_extended(self):
        assert "fake" not in SIMULATOR_REGISTRY
        try:
            SIMULATOR_REGISTRY["fake"] = lambda cfg, path: None
            assert "fake" in SIMULATOR_REGISTRY
        finally:
            SIMULATOR_REGISTRY.pop("fake", None)


# ── 型別註解 / docstring 存在性 ────────────────────


class TestPublicAPI:
    """確保公開 API 的 docstring / 型別存在。"""

    def test_trainer_has_docstring(self):
        assert Trainer.__doc__ and "訓練器" in Trainer.__doc__

    def test_run_has_annotation(self):
        import inspect

        sig = inspect.signature(Trainer.run)
        assert sig.return_annotation is None or sig.return_annotation is type(None)

    def test_init_has_cfg_annotation(self):
        import inspect

        from omegaconf import DictConfig

        sig = inspect.signature(Trainer.__init__)
        assert sig.parameters["cfg"].annotation is DictConfig


# ── Resume / Checkpoint 行為（較輕量的 smoke） ───────


class TestResumeBehavior:
    """驗 continue_run + record 已有 epoch 時的 resume 流程。"""

    def test_resume_when_record_has_epoch(self, mocked_externals, tmp_path, monkeypatch):
        """continue_run=True 且 record 有 epoch 時，Trainer 嘗試 load checkpoint。"""
        # 覆寫 get_result_path 以模擬 continue_run=True
        fake_path = _FakePath(tmp_path / "resume")

        def fake_get_result_path(name, *, rootdir=None, enable_exception_handler=False):
            return fake_path, True  # continue_run=True

        import antenna as _antenna

        monkeypatch.setattr(_antenna, "get_result_path", fake_get_result_path)

        # 預先塞 record.pickle 到 result_path，讓 Record 載入時有 epoch 鍵
        from antenna.utils.record import Record

        rec = Record("temp", rootdir=str(fake_path))
        rec["epoch"] = 3
        rec["real_loss"] = 0.5
        rec.save()

        cfg = _make_cfg()

        # generator.change + smodel.load 應會被呼叫
        # 但因為 Models.load 需要實體 .pth 檔案，實務上會失敗；用 try/except 包覆確認 attempt
        with (
            patch.object(trainer_module.Models, "change", return_value=None) as mock_change,
            patch.object(trainer_module.Models, "load", return_value=None),
        ):
            try:
                Trainer(cfg)
            except Exception:
                # Models.change 被 mock，load_torch 仍可能 raise；只關心 change 有被呼叫
                pass
            # continue_run=True + record.epoch=3 時，generator.change(3, load=True) 應被呼叫
            assert mock_change.called

    def test_fresh_start_does_not_load_checkpoint(self, mocked_externals):
        """continue_run=False 時不應觸發 checkpoint 載入。"""
        cfg = _make_cfg()
        with patch.object(trainer_module.Models, "load") as mock_load:
            Trainer(cfg)
            # fresh 新訓練不該呼叫 Models.load
            mock_load.assert_not_called()


# ── Factory function 本身的基本檢查 ─────────────────


class TestFactories:
    """單純驗 factory 存在且能被 registry 找到。"""

    def test_single_port_factory_callable(self):
        assert callable(trainer_module._single_port_factory)

    def test_dual_port_factory_callable(self):
        assert callable(trainer_module._dual_port_factory)

    def test_ris_factory_callable(self):
        assert callable(trainer_module._ris_factory)

    def test_ris_factory_uses_coordinate_width(self, monkeypatch):
        """ris factory 會把 coordinate[1]-coordinate[0] 傳給 RISSimulator。"""
        captured: dict[str, Any] = {}

        class _FakeRIS:
            def __init__(self, element_num):
                captured["element_num"] = element_num

        # 動態將 RISSimulator 換成 _FakeRIS
        import antenna.ris.simulate_ris as sim_mod

        monkeypatch.setattr(sim_mod, "RISSimulator", _FakeRIS)
        cfg = _make_cfg(coordinate=[0, 8, 0, 8])
        trainer_module._ris_factory(cfg, None)
        assert captured["element_num"] == 8
