"""
tests/test_scripts.py — status.py / analyze.py 的純函式測試（不碰 NAS）。

只測可離線驗的純邏輯（機器名解析、欄位過濾、cosine 基底、中位差）;掃 NAS 的部分靠實跑驗。
"""
import numpy as np

from script.status import _machine, _num
from script.analyze import _cos_basis, _mad


def test_machine_parse():
    assert _machine("[Patch-single-216-2c121f] pixel_single_r3_explore") == "216"
    assert _machine("[Patch-single-37-e6a4f4] x") == "37"
    assert _machine("no-pattern-here") == "?"


def test_num_filters_empty_and_nan():
    rows = [{"a": "1.5"}, {"a": ""}, {"a": "nan"}, {"a": "2"}, {"b": "9"}]
    assert _num(rows, "a") == [1.5, 2.0]      # 空/nan/缺欄都略過


def test_cos_basis_shape_and_k0_constant():
    theta = np.linspace(-180, 180, 181)
    B = _cos_basis(theta, 8)
    assert B.shape == (8, 181)
    assert np.allclose(B[0], 1.0)             # k=0 mode = cos(0) = 常數 1


def test_mad_median_abs_delta():
    assert _mad([1.0, 3.0, 3.0, 6.0]) == 2.0  # 逐差 2,0,3 → 中位 2
