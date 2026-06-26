"""
tests/test_benchmark.py — benchmark_vs_random 的 worst-margin 計算單元測試。

worst-margin = in-band(中央平台)對 spec 的最差餘裕(正=達標、越高越好);定義與 custom_loss_minmax
的嚴格點一致。只測純函式 (不碰 NAS / HFSS / 結果夾 I/O)。
"""
import pytest
import torch

from antenna.training import TrainConfig
from antenna.losses import worst_margin       # 共用定義 (script 與 training.py 都用這份)

LABELS = ["S11", "Gain"]                       # 單埠 response 列序


def _cfg():
    return TrainConfig(
        name="t", port="single",
        targets={
            "S11": {"side": 0, "center": -10, "width": [5, 0, 7, 0, 5], "method": "low"},
            "Gain": {"side": -19, "center": 4, "width": [5, 0, 7, 0, 5], "method": "high"},
        },
    )


def test_worst_margin_satisfied():
    """帶內 S11=-15(≤-10)、Gain=6(≥4) → margin S11=5、Gain=2、worst=2(>0 達標)。"""
    cfg = _cfg()
    r = torch.zeros(2, 17)
    r[0, 5:12] = -15.0          # S11 in-band (中央平台 = 索引 5..11)
    r[1, 5:12] = 6.0            # Gain in-band
    w, m = worst_margin(r, LABELS, cfg.targets)
    assert m["S11"] == pytest.approx(5.0)
    assert m["Gain"] == pytest.approx(2.0)
    assert w == pytest.approx(2.0)


def test_worst_margin_violated_is_negative():
    """帶內 S11 有一點 -8(>-10 未達標)→ margin S11=-2、worst=-2(<0)。"""
    cfg = _cfg()
    r = torch.zeros(2, 17)
    r[0, 5:12] = -15.0
    r[1, 5:12] = 6.0
    r[0, 8] = -8.0              # 帶內一點違反 (最高點)
    w, m = worst_margin(r, LABELS, cfg.targets)
    assert m["S11"] == pytest.approx(-2.0)
    assert w == pytest.approx(-2.0)


def test_worst_margin_ignores_outband():
    """帶外(平台/斜邊外)再爛也不影響 worst-margin(只看 in-band 中央平台)。"""
    cfg = _cfg()
    r = torch.zeros(2, 17)
    r[0, 5:12] = -15.0
    r[1, 5:12] = 6.0
    r[0, 0] = 100.0            # 帶外 S11 超爛
    r[0, 16] = 100.0
    r[1, 0] = -100.0          # 帶外 Gain 超爛
    w, _ = worst_margin(r, LABELS, cfg.targets)
    assert w == pytest.approx(2.0)   # 不受帶外影響


def test_worst_margin_accepts_flat_response():
    """response 攤平成 (34,) 也能 reshape 回 (2,17) 正確計算。"""
    cfg = _cfg()
    r = torch.zeros(2, 17)
    r[0, 5:12] = -15.0
    r[1, 5:12] = 6.0
    w, _ = worst_margin(r.reshape(-1), LABELS, cfg.targets)
    assert w == pytest.approx(2.0)
