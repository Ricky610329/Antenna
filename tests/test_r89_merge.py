import numpy as np

from script import r81_curve_sm as sm
from script import r89_merge as mg


def test_diff_groups_merges_down_to_max_groups():
    base = np.zeros((25, 25), bool)
    donor = base.copy()
    for r, c in [(5, 5), (5, 20), (20, 5), (20, 20), (12, 12), (12, 14)]:
        donor[r, c] = True
    groups = mg.diff_groups(base, donor, max_groups=3)
    assert len(groups) == 3
    assert np.array_equal(np.logical_or.reduce(groups), base != donor)   # 合併後仍完整覆蓋相異格


def test_transplants_skip_full_set_and_keep_big_structure():
    base = sm.enforce_dual_geometry(np.zeros((25, 25), bool))
    base[10:13, 3:9] = True                                              # 一塊大金屬
    donor = base.copy()
    donor[18, 18] = True                                                 # 兩個零散小島
    donor[6, 20] = True
    out = mg.transplants(base, donor, max_groups=5)
    assert len(out) == 2                                                 # 2 塊 → 只取單塊移植（不含空集與全集）
    assert all(mg.big_components(q) == mg.big_components(base) for q, _ in out)
    donor2 = base.copy()
    donor2[11, 3:9] = False                                              # 把大塊切成兩半 → 大塊數改變
    donor2[18, 18] = True
    assert all(mg.big_components(q) == mg.big_components(base) for q, _ in mg.transplants(base, donor2))
