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


def test_improving_moves_keeps_small_gains_from_good_parents():
    z = np.zeros(625, bool)
    a = z.copy(); a[100] = True
    b = z.copy(); b[[100, 101, 102, 103, 104]] = True
    pats = {"P": z, "A": a, "B": b, "C": a.copy(), "Q": z.copy()}
    scores = {"P": -0.3, "A": -0.2, "B": -0.1, "C": -0.4, "Q": -2.0}
    parents = {"A": "P", "B": "P", "C": "P", "D": "P"}
    out = mg.improving_moves(scores, pats, parents, max_cells=4)
    assert [m[1] for m in out] == ["A"]                                # B 改 5 格、C 變差、D 沒量
    assert mg.improving_moves(scores, pats, {"A": "Q"}, min_parent=-0.5) == []   # 親代太差不收


def test_stack_candidates_apply_moves_and_rank_by_gain_sum():
    base = sm.enforce_dual_geometry(np.zeros((25, 25), bool)).reshape(-1)
    moves = [(0.10, "A", np.array([30]), np.array([True])),
             (0.05, "B", np.array([200]), np.array([True])),
             (0.02, "C", np.array([30]), np.array([True]))]           # C 與 A 同格同值
    out = mg.stack_candidates(base, moves, sizes=(2,))
    assert out[0][1] == ["A", "B"] and abs(out[0][2] - 0.15) < 1e-9
    assert all(q.reshape(-1)[[30, 200]].any() for q, _, _ in out)


def test_edge_moves_skip_present_and_unsafe_side_effects():
    z = np.zeros(625, bool)
    stats = {(10, True): [(0.15, {"s21_pass": 0.15, "s21_low_stop": 0.0, "s21_high_stop": 1.0,
                                   "s11_match": 0.0, "s22_match": 0.0})],
             (11, True): [(0.20, {"s21_pass": 0.2, "s21_low_stop": -1.0, "s21_high_stop": 0.0,
                                   "s11_match": 0.0, "s22_match": 0.0})],          # 低阻會不過
             (12, False): [(0.30, {m: 0.0 for m in mg.MARGINS})],                 # 底已是空白
             (13, True): [(-0.05, {m: 0.0 for m in mg.MARGINS})]}                 # 不抬高
    bm = {"s21_pass": -0.05, "s21_low_stop": 0.6, "s21_high_stop": 3.0, "s11_match": 1.0, "s22_match": 3.0}
    out = mg.edge_moves(stats, z, bm)
    assert [(k, v) for k, v, _, _ in out] == [(10, True)]


def test_predicted_moves_rank_by_min_margin_and_pair_additively():
    z = np.zeros(625, bool)
    zero = {m: 0.0 for m in mg.MARGINS}
    stats = {(1, True): [(0.0, dict(zero, s21_high_stop=1.5))],                   # 修高阻
             (2, True): [(0.0, dict(zero, s21_high_stop=1.0, s21_pass=-0.5))],    # 修高阻但傷通帶
             (3, True): [(0.0, dict(zero, s21_high_stop=-0.2))]}
    bm = {"s21_pass": 0.12, "s21_low_stop": 0.8, "s21_high_stop": -1.2, "s11_match": 1.2, "s22_match": 4.9}
    out = mg.predicted_moves(stats, z, bm, top_pairs=3)
    best_cells, best_wm, _ = out[0]
    assert best_cells == [(1, True)] and abs(best_wm - 0.12) < 1e-9          # 1 單步：高阻 +0.3、通帶 0.12 → 0.12
    pair = next(x for x in out if len(x[0]) == 2 and {c for c, _ in x[0]} == {1, 2})
    assert abs(pair[1] - (-0.38)) < 1e-9                                      # 兩兩可加：通帶 0.12−0.5
