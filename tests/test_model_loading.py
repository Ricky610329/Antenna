"""
模型建構與載入的 mock 測試（config 驅動）。

涵蓋三塊（皆無需真實權重檔 / HFSS）：
  - build_generator / build_surrogate：架構由 cfg.generator / cfg.surrogate 的 type/hidden 決定。
  - prepare_models：模型載入策略（續跑 / GEN 預載入 / SM 預訓練 / KuoHung 暖身）的分支。
  - build_simulator：port → 真實模擬器 class。
"""
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from antenna.training import (
    TrainConfig, prepare_models, build_simulator,
    build_generator, build_surrogate,
    GENERATOR_REGISTRY, SURROGATE_REGISTRY,
)
from antenna.utils import config
from antenna.utils.utils import Record


@pytest.fixture
def _hfss_lr():
    """build_surrogate 經 OldSM → Ranger 需要 config['HFSS.lr'] (平時由 run_training 設定)。"""
    config["HFSS.lr"] = 0.001
    yield


def _single_cfg():
    return TrainConfig(
        name="t", port="single",
        targets={
            "S11": {"side": 0, "center": -10, "width": [5, 0, 7, 0, 5], "method": "low"},
            "Gain": {"side": -19, "center": 4, "width": [5, 0, 7, 0, 5], "method": "high"},
        },
    )


def _dual_cfg():
    t = {"side": -1.25, "center": -12, "width": [4, 2, 5, 2, 4], "interval": [-1, 1]}
    return TrainConfig(
        name="d", port="dual",
        targets={"S11": t, "S22": t,
                 "S21": {"side": -20, "center": -3, "width": [3, 0, 11, 0, 3], "interval": [-1, 1]}},
    )


class _FakeDS:
    def __init__(self, n): self.n = n
    def __len__(self): return self.n


def _linears(seq):
    return [m for m in seq if isinstance(m, nn.Linear)]


# ── build_generator / build_surrogate：架構由 config 的 type/hidden 決定 ─────

def test_registries_have_defaults():
    assert "sigmoid" in GENERATOR_REGISTRY
    assert "mlp" in SURROGATE_REGISTRY


def test_build_generator_default_arch():
    """未指定 generator → 預設架構 (hidden=(1024,1024) → 2 隱藏層 + 1 輸出)。"""
    g = build_generator(_single_cfg())
    ls = _linears(g.fc_patch)
    assert len(ls) == 3
    assert ls[0].out_features == 1024 and ls[1].out_features == 1024


def test_build_generator_custom_hidden():
    """cfg.generator.hidden 改變寬度/層數。"""
    cfg = _single_cfg(); cfg.generator = {"type": "sigmoid", "hidden": [64, 32, 16]}
    g = build_generator(cfg)
    ls = _linears(g.fc_patch)
    assert len(ls) == 4                                  # 3 隱藏 + 1 輸出
    assert [l.out_features for l in ls[:3]] == [64, 32, 16]


def test_build_surrogate_default_arch(tmp_path, _hfss_lr):
    """未指定 surrogate.hidden → 預設 (2048,1024,512,128,64) → 5 隱藏 + 1 輸出。"""
    sm = build_surrogate(_single_cfg(), tmp_path)
    ls = _linears(sm.model.fc_patch)
    assert len(ls) == 6
    assert [l.out_features for l in ls[:5]] == [2048, 1024, 512, 128, 64]


def test_build_surrogate_custom_hidden(tmp_path, _hfss_lr):
    cfg = _single_cfg(); cfg.surrogate = {"type": "mlp", "hidden": [128, 64]}
    sm = build_surrogate(cfg, tmp_path)
    ls = _linears(sm.model.fc_patch)
    assert len(ls) == 3                                  # 2 隱藏 + 1 輸出
    assert [l.out_features for l in ls[:2]] == [128, 64]


# ── prepare_models：載入策略分支 ──────────────────────────────────────────

def test_prepare_resume(tmp_path):
    """(1) 續跑：TEMP 有 epoch → 載回 GEN/SM、回傳該 epoch；其餘分支全部略過。"""
    TEMP = Record("t", rootdir=str(tmp_path)); TEMP["epoch"] = 5
    gen, sm = MagicMock(), MagicMock()
    start = prepare_models(_single_cfg(), gen, sm, TEMP, continue_run=True,
                           gen_pretrained_path=str(tmp_path / "g.pth"),
                           sm_pretrained_path=str(tmp_path / "s.pth"),
                           offline_dataset=_FakeDS(3), warmup=MagicMock())
    assert start == 5
    gen.change.assert_called_once_with(5, load=True)
    sm.load.assert_called_once()
    gen.pre_load_model.assert_not_called()
    sm.pre_load_model.assert_not_called()
    sm.train_by_datas.assert_not_called()


def test_prepare_gen_pretrained(tmp_path):
    """(2) GEN 預載入：gen_pretrained_path 存在 → generator.pre_load_model。"""
    f = tmp_path / "gen.pth"; f.write_bytes(b"x")
    gen, sm = MagicMock(), MagicMock()
    prepare_models(_single_cfg(), gen, sm, Record("t", rootdir=str(tmp_path)),
                   gen_pretrained_path=str(f))
    gen.pre_load_model.assert_called_once_with(str(f))


def test_prepare_gen_pretrained_missing(tmp_path):
    """GEN 預載入路徑不存在 → 跳過、不報錯。"""
    gen, sm = MagicMock(), MagicMock()
    prepare_models(_single_cfg(), gen, sm, Record("t", rootdir=str(tmp_path)),
                   gen_pretrained_path=str(tmp_path / "nope.pth"))
    gen.pre_load_model.assert_not_called()


def test_prepare_sm_pretrained(tmp_path):
    """(3) SM 預訓練檔存在 → smodel.pre_load_model；不走離線預訓練。"""
    f = tmp_path / "sm.pth"; f.write_bytes(b"x")
    gen, sm = MagicMock(), MagicMock()
    start = prepare_models(_single_cfg(), gen, sm, Record("t", rootdir=str(tmp_path)),
                           sm_pretrained_path=str(f), offline_dataset=_FakeDS(3))
    assert start == 0
    sm.pre_load_model.assert_called_once_with(str(f))
    sm.train_by_datas.assert_not_called()
    sm.load.assert_not_called()


def test_prepare_offline(tmp_path):
    """(3) 無 SM 預訓練檔但有離線資料集 → smodel.train_by_datas。"""
    gen, sm = MagicMock(), MagicMock()
    prepare_models(_single_cfg(), gen, sm, Record("t", rootdir=str(tmp_path)),
                   sm_pretrained_path=None, offline_dataset=_FakeDS(3))
    sm.train_by_datas.assert_called_once()
    sm.pre_load_model.assert_not_called()


def test_prepare_sm_pretrained_missing_falls_to_offline(tmp_path):
    """SM 預訓練路徑不存在 → 跳過，改走離線預訓練。"""
    gen, sm = MagicMock(), MagicMock()
    prepare_models(_single_cfg(), gen, sm, Record("t", rootdir=str(tmp_path)),
                   sm_pretrained_path=str(tmp_path / "nope.pth"), offline_dataset=_FakeDS(2))
    sm.pre_load_model.assert_not_called()
    sm.train_by_datas.assert_called_once()


def test_prepare_warmup_called(tmp_path):
    """(4) 暖身：warmup(smodel) 被呼叫。"""
    gen, sm = MagicMock(), MagicMock(); warmup = MagicMock()
    prepare_models(_single_cfg(), gen, sm, Record("t", rootdir=str(tmp_path)), warmup=warmup)
    warmup.assert_called_once_with(sm)


def test_prepare_compose_gen_sm_warmup(tmp_path):
    """非續跑時：GEN 預載入 + SM 預訓練 + 暖身可同時發生。"""
    gf = tmp_path / "g.pth"; gf.write_bytes(b"x")
    sf = tmp_path / "s.pth"; sf.write_bytes(b"x")
    gen, sm = MagicMock(), MagicMock(); warmup = MagicMock()
    prepare_models(_single_cfg(), gen, sm, Record("t", rootdir=str(tmp_path)),
                   gen_pretrained_path=str(gf), sm_pretrained_path=str(sf), warmup=warmup)
    gen.pre_load_model.assert_called_once_with(str(gf))
    sm.pre_load_model.assert_called_once_with(str(sf))
    warmup.assert_called_once_with(sm)


def test_prepare_fresh(tmp_path):
    """皆無 → 不載入任何東西 (GEN/SM 從隨機權重起步)。"""
    gen, sm = MagicMock(), MagicMock()
    start = prepare_models(_single_cfg(), gen, sm, Record("t", rootdir=str(tmp_path)))
    assert start == 0
    sm.load.assert_not_called(); sm.pre_load_model.assert_not_called()
    sm.train_by_datas.assert_not_called()
    gen.change.assert_not_called(); gen.pre_load_model.assert_not_called()


# ── build_simulator：port → 真實模擬器 class ──────────────────────────────

def test_build_simulator_by_port(tmp_path):
    from antenna.patch import SinglePortSimulator, DualPortSimulator
    assert isinstance(build_simulator(_single_cfg(), tmp_path / "s"), SinglePortSimulator)
    assert isinstance(build_simulator(_dual_cfg(), tmp_path / "d"), DualPortSimulator)
