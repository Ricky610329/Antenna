"""
模型載入 (prepare_surrogate) 與模擬器解析 (build_simulator) 的 mock 測試。

prepare_surrogate 是「SM 載入策略」的模組化實作，由 config 的 surrogate 區段驅動。
這裡用 mock SM/GEN 驗證四種分支各自呼叫正確的方法 (無需真實權重檔/HFSS)。
"""
from unittest.mock import MagicMock

import pytest

from antenna.training import TrainConfig, prepare_surrogate, build_simulator
from antenna.utils.utils import Record


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


# ── prepare_surrogate 四種分支 ────────────────────────────────────────────

def test_prepare_resume(tmp_path):
    """(1) 續跑：TEMP 有 epoch → 載回 GEN/SM、回傳該 epoch；其餘分支不執行。"""
    TEMP = Record("t", rootdir=str(tmp_path)); TEMP["epoch"] = 5
    gen, sm = MagicMock(), MagicMock()
    start = prepare_surrogate(_single_cfg(), gen, sm, TEMP, continue_run=True,
                              pretrained_sm_path=str(tmp_path / "x.pth"), offline_dataset=_FakeDS(3))
    assert start == 5
    gen.change.assert_called_once_with(5, load=True)
    sm.load.assert_called_once()
    sm.pre_load_model.assert_not_called()
    sm.train_by_datas.assert_not_called()


def test_prepare_pretrained(tmp_path):
    """(2) 預訓練檔存在 → smodel.pre_load_model；不走離線預訓練。"""
    f = tmp_path / "sm.pth"; f.write_bytes(b"x")
    TEMP = Record("t", rootdir=str(tmp_path))
    gen, sm = MagicMock(), MagicMock()
    start = prepare_surrogate(_single_cfg(), gen, sm, TEMP,
                              pretrained_sm_path=str(f), offline_dataset=_FakeDS(3))
    assert start == 0
    sm.pre_load_model.assert_called_once_with(str(f))
    sm.train_by_datas.assert_not_called()
    sm.load.assert_not_called()


def test_prepare_offline(tmp_path):
    """(3) 無預訓練檔但有離線資料集 → smodel.train_by_datas。"""
    TEMP = Record("t", rootdir=str(tmp_path))
    gen, sm = MagicMock(), MagicMock()
    start = prepare_surrogate(_single_cfg(), gen, sm, TEMP,
                              pretrained_sm_path=None, offline_dataset=_FakeDS(3))
    assert start == 0
    sm.train_by_datas.assert_called_once()
    sm.pre_load_model.assert_not_called()


def test_prepare_fresh(tmp_path):
    """(4) 皆無 → 不載入任何東西 (SM 從隨機權重起步)。"""
    TEMP = Record("t", rootdir=str(tmp_path))
    gen, sm = MagicMock(), MagicMock()
    start = prepare_surrogate(_single_cfg(), gen, sm, TEMP)
    assert start == 0
    sm.load.assert_not_called()
    sm.pre_load_model.assert_not_called()
    sm.train_by_datas.assert_not_called()
    gen.change.assert_not_called()


def test_prepare_pretrained_missing_falls_to_offline(tmp_path):
    """預訓練路徑不存在 → 跳過 (2)，改走 (3) 離線預訓練。"""
    TEMP = Record("t", rootdir=str(tmp_path))
    gen, sm = MagicMock(), MagicMock()
    prepare_surrogate(_single_cfg(), gen, sm, TEMP,
                      pretrained_sm_path=str(tmp_path / "nope.pth"), offline_dataset=_FakeDS(2))
    sm.pre_load_model.assert_not_called()
    sm.train_by_datas.assert_called_once()


# ── build_simulator：port → 真實模擬器 class ──────────────────────────────

def test_build_simulator_by_port(tmp_path):
    from antenna.patch import SinglePortSimulator, DualPortSimulator
    s = build_simulator(_single_cfg(), tmp_path / "s")
    assert isinstance(s, SinglePortSimulator)
    d = build_simulator(_dual_cfg(), tmp_path / "d")
    assert isinstance(d, DualPortSimulator)
