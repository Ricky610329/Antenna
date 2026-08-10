"""
tests/test_dual_margin.py — dual-port 判準尺 (`worst_margin_dual` / `dual_energy_max`) 單元測試。

判準定案 (2026-08-10, proposal-dual-kickoff §2)：**wm_dual = min(m1..m4)，m5/m6 記帳不進 min**。
本檔對 `configs/dual_base.yaml` 的**實際 targets** (S11/S22 width 有斜邊) 手工構造已知響應，
驗六項數值 + mask 索引正確。只測純函式 (不碰 NAS / HFSS / 結果夾 I/O)。

#! 為什麼要專門測「mask 索引」：dual 的 S11/S22 width=[4,2,5,2,4] 有斜邊，而
#  np.linspace(side, center, 2) 只產端點 → 斜邊的兩點一個等於 side、一個等於 center。
#  沿用 single 的 width 切片算術 (w[0]+w[1] : +w[2] = 6:11) 會**漏掉 idx 5 與 idx 11**
#  這兩個真的等於 center 的頻點 = 靜默切錯帶。下面的 trap 值就是專門釘死這個坑。
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from antenna.losses import _dual_target_curve, dual_energy_max, worst_margin_dual
from antenna.training import PORT_SPECS, load_config

LABELS = PORT_SPECS["dual"]["labels"]          # ['S11', 'S21', 'S22'] — dual response 列序
CONFIG = Path(__file__).resolve().parents[1] / "configs" / "dual_base.yaml"


def _targets():
    """直接讀 configs/dual_base.yaml 的實際 targets (不是測試自造的簡化版)。"""
    return load_config(CONFIG).targets


def _response():
    """手工構造一筆 (3, 17) 響應，六項 margin 皆為已知值。

    S11: 帶內 idx5-11 最高點 = -13.0 (落在 idx11 斜邊點)   → m1 = -12 - (-13.0)  = +1.00
         帶外 idx0-4∪12-16 最低點 = -0.9 (idx12)          → m5 = -0.9 - (-1.25) = +0.35
    S21: 通帶 idx3-13 最低點 = -3.5 (idx3)                → m3 = -3.5 - (-3)    = -0.50
         阻帶 idx0-2∪14-16 最高點 = -24.0 (idx14)         → m4 = -20 - (-24.0)  = +4.00
    S22: 帶內 idx5-11 最高點 = -15.5 (idx11)              → m2 = -12 - (-15.5)  = +3.50
         帶外 idx0-4∪12-16 最低點 = -5.0 (idx12)          → m6 = -5.0 - (-1.25) = -3.75
    → wm = min(m1..m4) = -0.50 (m6 = -3.75 更低，但**刻意不進 min**)。
    """
    r = torch.zeros(3, 17)

    # ── S11 (列 0) ────────────────────────────────────────────────
    r[0, 0:4] = -0.5
    r[0, 4] = -0.7                 # 帶外 (斜邊左端點 == side)
    r[0, 5] = -14.0                # ★ trap：帶內左界 (切片算術 6:11 會漏掉)
    r[0, 6:11] = -20.0             # 帶內中央平台
    r[0, 11] = -13.0               # ★ trap：帶內右界 = 帶內最高點 → 決定 m1
    r[0, 12] = -0.9                # 帶外 (斜邊右端點 == side) = 帶外最低點 → 決定 m5
    r[0, 13:17] = -0.5

    # ── S21 (列 1)：width [3,0,11,0,3] 無斜邊 ──────────────────────
    r[1, 0:2] = -30.0
    r[1, 2] = -25.0                # 阻帶
    r[1, 3] = -3.5                 # ★ trap：通帶左界 = 通帶最低點 → 決定 m3
    r[1, 4:13] = -1.0              # 通帶
    r[1, 13] = -3.2                # ★ trap：通帶右界
    r[1, 14] = -24.0               # 阻帶最高點 → 決定 m4
    r[1, 15:17] = -35.0

    # ── S22 (列 2) ────────────────────────────────────────────────
    r[2, 0:4] = -0.5
    r[2, 4] = -0.6
    r[2, 5] = -16.0                # ★ trap：帶內左界
    r[2, 6:11] = -22.0
    r[2, 11] = -15.5               # ★ trap：帶內右界 = 帶內最高點 → 決定 m2
    r[2, 12] = -5.0                # 帶外最低點 → m6 = -3.75 (比 wm 還低，用來證明不進 min)
    r[2, 13:17] = -0.4
    return r


# ── mask 索引 ────────────────────────────────────────────────────────────────

def test_dual_target_curve_mask_indices():
    """dual_base.yaml 的 targets 展開後，四個頻點集合必須落在文件寫死的索引上。"""
    t = _targets()
    s11 = _dual_target_curve(t["S11"], 17, "S11")
    s21 = _dual_target_curve(t["S21"], 17, "S21")
    s22 = _dual_target_curve(t["S22"], 17, "S22")

    # S11/S22：帶內 (== min == center -12) = idx 5-11；帶外 (== max == side -1.25) = idx 0-4 ∪ 12-16
    for curve in (s11, s22):
        assert list(np.where(curve == curve.min())[0]) == list(range(5, 12))
        assert list(np.where(curve == curve.max())[0]) == [0, 1, 2, 3, 4, 12, 13, 14, 15, 16]
        assert curve.min() == pytest.approx(-12.0) and curve.max() == pytest.approx(-1.25)

    # S21：通帶 (== max == center -3) = idx 3-13；阻帶 (== min == side -20) = idx 0-2 ∪ 14-16
    assert list(np.where(s21 == s21.max())[0]) == list(range(3, 14))
    assert list(np.where(s21 == s21.min())[0]) == [0, 1, 2, 14, 15, 16]
    assert s21.max() == pytest.approx(-3.0) and s21.min() == pytest.approx(-20.0)


def test_dual_target_curve_width_mismatch_raises():
    """width 展開長度 ≠ 響應點數 → 明確 ValueError (而非靜默切錯帶)。"""
    t = _targets()
    with pytest.raises(ValueError, match=r"展開成|width"):
        _dual_target_curve(t["S11"], 25, "S11")


# ── 六項 margin 數值 ──────────────────────────────────────────────────────────

def test_worst_margin_dual_six_items():
    """六項 margin 對照手工算出的期望值 (門檻全部從 targets 的 side/center 讀)。"""
    wm, per = worst_margin_dual(_response(), LABELS, _targets())
    assert per["m1"] == pytest.approx(1.00)      # -12 - (-13.0)
    assert per["m2"] == pytest.approx(3.50)      # -12 - (-15.5)
    assert per["m3"] == pytest.approx(-0.50)     # -3.5 - (-3)
    assert per["m4"] == pytest.approx(4.00)      # -20 - (-24.0)
    assert per["m5"] == pytest.approx(0.35)      # -0.9 - (-1.25)
    assert per["m6"] == pytest.approx(-3.75)     # -5.0 - (-1.25)
    assert wm == pytest.approx(-0.50)


def test_worst_margin_dual_excludes_m5_m6_from_min():
    """m6 = -3.75 遠低於 wm = -0.50，wm 仍必須是 min(m1..m4) → 證明 m5/m6 只記帳不進 min。"""
    wm, per = worst_margin_dual(_response(), LABELS, _targets())
    assert per["m6"] < wm                        # 若 m5/m6 誤入 min，這行與下行必有一條紅
    assert wm == pytest.approx(min(per["m" + str(i)] for i in (1, 2, 3, 4)))


def test_worst_margin_dual_per_label_margins():
    """per 另含每 label 主 margin：S11=m1、S22=m2、S21=min(m3,m4)。"""
    _, per = worst_margin_dual(_response(), LABELS, _targets())
    assert per["S11"] == pytest.approx(per["m1"])
    assert per["S22"] == pytest.approx(per["m2"])
    assert per["S21"] == pytest.approx(min(per["m3"], per["m4"]))


def test_worst_margin_dual_uses_mask_not_width_slice():
    """帶內邊界 (idx 5 / 11) 必須算進帶內——這是 width 切片算術會漏掉的兩點。"""
    t = _targets()
    r = _response()
    r[0, 11] = -12.0               # 帶內右界剛好貼齊 spec → m1 = 0
    _, per = worst_margin_dual(r, LABELS, t)
    assert per["m1"] == pytest.approx(0.0)
    r[0, 5] = -11.0                # 帶內左界違規 1dB → m1 = -1 (切片算術會看不到，仍回 0)
    _, per2 = worst_margin_dual(r, LABELS, t)
    assert per2["m1"] == pytest.approx(-1.0)


def test_worst_margin_dual_accepts_flat_response():
    """response 攤平成 (51,) 也能 reshape 回 (3,17) 正確計算。"""
    wm_flat, _ = worst_margin_dual(_response().reshape(-1), LABELS, _targets())
    assert wm_flat == pytest.approx(-0.50)


def test_worst_margin_dual_rejects_single_labels():
    """single-port 的 labels 丟進來 → 明確 ValueError (別讓兩把尺互串)。"""
    with pytest.raises(ValueError, match=r"S11.*S21.*S22|worst_margin"):
        worst_margin_dual(torch.zeros(2, 17), ["S11", "Gain"], _targets())


def test_worst_margin_dual_rejects_flipped_target_direction():
    """S11 被寫成 center > side (方向反了) → fail-fast，不靜默反號。"""
    t = _targets()
    t["S11"] = {**t["S11"], "side": -12, "center": -1.25}
    with pytest.raises(ValueError, match=r"center < side|帶內壓低"):
        worst_margin_dual(_response(), LABELS, t)


# ── 能量自證 ─────────────────────────────────────────────────────────────────

def test_dual_energy_max():
    """max over 頻點 of (|S11|²+|S21|², |S22|²+|S21|²)；dB→線性，>1 = 壞檔。"""
    r = torch.zeros(3, 17)
    r[0], r[1], r[2] = -10.0, -20.0, -30.0        # 0.1 / 0.01 / 0.001 → max(0.11, 0.011) = 0.11
    assert dual_energy_max(r) == pytest.approx(0.11, abs=1e-6)
    r[0, 3] = 0.0                                  # 單一頻點 S11 全反射 → 1 + 0.01 = 1.01 (>1 = 壞檔)
    assert dual_energy_max(r) == pytest.approx(1.01, abs=1e-6)
