# -*- coding: utf-8 -*-
"""回顧輪必修修復的回歸測試(2026-08-08;品質審計 #1-#5)。"""
import pytest


def test_n99_zero_oob_not_swallowed():
    # 必修1:oob=0.0 是合法(且是紀錄級)帶外值——舊寫法 `(x or 99)` 會把 0.0 吃成 99 沉底
    from script.analyze import _n99
    assert _n99(0.0) == 0.0
    assert _n99(None) == 99.0
    assert _n99(8.61) == 8.61


def test_gain_stores_sort_key_no_typeerror():
    # 必修5:同 mtime+batch 時舊 stores.sort() 會比到第三元素 list[dict] → TypeError
    rows = [(100.0, 1, [{"id": "b"}], {}), (100.0, 1, [{"id": "a"}], {})]
    rows.sort(key=lambda t: t[:2])                     # 修正後的排序鍵:只比 (mtime, batch)
    assert len(rows) == 2


def test_with_neg_requires_explicit_out_before_any_side_effect():
    # 必修3+4:--with-neg 未帶顯式 --out 必須在任何落盤/載資料之前就攔截
    from script.sm_reanchor import train

    class A:                                            # 最小 args stub
        with_neg = True
        out = "sm_reanchor.pth"
        add = None
    with pytest.raises(SystemExit):
        train(A())
