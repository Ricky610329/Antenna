"""
tests/test_scripts.py — status.py / analyze.py 的純函式測試（不碰 NAS）。

只測可離線驗的純邏輯（機器名解析、欄位過濾、cosine 基底、中位差）;掃 NAS 的部分靠實跑驗。
"""
import numpy as np

from script.status import _machine, _num, _liveness
from script.analyze import _cos_basis, _mad


def test_machine_parse():
    assert _machine("[Patch-single-216-2c121f] pixel_single_r3_explore") == "216"
    assert _machine("[Patch-single-37-e6a4f4] x") == "37"
    assert _machine("no-pattern-here") == "?"


def test_num_filters_empty_and_nan():
    rows = [{"a": "1.5"}, {"a": ""}, {"a": "nan"}, {"a": "2"}, {"b": "9"}]
    assert _num(rows, "a") == [1.5, 2.0]      # 空/nan/缺欄都略過


def _lv(**kw):
    base = dict(state="running", advanced=False, elapsed_enough=False, age_min=1.0, tpe_min=5.0)
    base.update(kw)
    return _liveness(**base)


def test_liveness_terminal_states_authoritative():
    """status.json 的 crashed/finished 是權威終態，不會被誤標成『卡住/在跑』。"""
    assert _lv(state="crashed", age_min=1) == ("當機", False)     # 剛當機也是當機（不看新鮮度）
    assert _lv(state="finished") == ("已完成", False)


def test_liveness_advance_is_proof_of_alive():
    """epoch 比上次掃描前進 = 鐵證在跑（即使心跳看似舊）。"""
    assert _lv(advanced=True, age_min=999) == ("在跑", True)


def test_liveness_running_fresh_vs_stuck():
    """宣稱 running：心跳新且未到判定點 → 在跑?；心跳久沒更新 / 隔>1.5ep 沒前進 → 疑卡住（抓硬砍/凍住）。"""
    assert _lv(state="running", age_min=3, tpe_min=5) == ("在跑?", True)
    assert _lv(state="running", age_min=999, tpe_min=5)[0] == "疑卡住"        # 心跳久沒更新
    assert _lv(state="running", elapsed_enough=True, age_min=1)[0] == "疑卡住"  # 該前進卻沒前進


def test_liveness_legacy_run_without_status_json():
    """無 status.json 的舊 run → 純時間啟發式：新→在跑?、舊→停止。"""
    assert _lv(state=None, age_min=1, tpe_min=5) == ("在跑?", True)
    assert _lv(state=None, age_min=999, tpe_min=5) == ("停止", False)


def test_cos_basis_shape_and_k0_constant():
    theta = np.linspace(-180, 180, 181)
    B = _cos_basis(theta, 8)
    assert B.shape == (8, 181)
    assert np.allclose(B[0], 1.0)             # k=0 mode = cos(0) = 常數 1


def test_mad_median_abs_delta():
    assert _mad([1.0, 3.0, 3.0, 6.0]) == 2.0  # 逐差 2,0,3 → 中位 2
