"""R80 對稱續探：起點排序。"""
import numpy as np

from script.r80_sym_climb import parent_order


def test_violation_ties_among_passing_break_by_wm():
    mg = np.array([[0.04, 0.5], [-1.0, 0.2], [0.6, 0.7], [0.5, 0.55]])
    assert list(parent_order(mg, "violation", 3)) == [2, 3, 0]
    assert list(parent_order(mg, "wm", 4)) == [2, 3, 0, 1]
