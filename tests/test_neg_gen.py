# -*- coding: utf-8 -*-
"""neg_gen(負片生成器)契約測試:決定性/承重塊/可製造性後處理/覆蓋選席。"""
import numpy as np
from scipy.ndimage import label

from script.neg_gen import ARMS, FEED, feed_block, farthest_point, gen_pool

S4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])


def test_deterministic():
    a = gen_pool(7, 14)
    b = gen_pool(7, 14)
    assert len(a) == len(b) == 14
    for (m1, t1), (m2, t2) in zip(a, b):
        assert (m1 == m2).all() and t1 == t2


def test_feed_block_and_manufacturability():
    fb = feed_block(5)
    for m, meta in gen_pool(11, 21):
        assert m[fb].all(), f"{meta} 承重塊破洞"
        assert m[FEED]
        lab, n = label(m, structure=S4)
        fid = lab[FEED]
        for i in range(1, n + 1):
            if i != fid:
                assert (lab == i).sum() >= 4, f"{meta} 粉塵件"
        vl, vn = label(~m, structure=S4)
        for i in range(1, vn + 1):
            assert (vl == i).sum() > 2, f"{meta} 針孔未縫"


def test_pool_unique_and_arms():
    pool = gen_pool(3, 28)
    keys = {m.tobytes() for m, _ in pool}
    assert len(keys) == len(pool)
    assert {t["arm"] for _, t in pool} == set(ARMS)


def test_farthest_point():
    pool = gen_pool(5, 20)
    idx = farthest_point(pool, 6, seed=1)
    assert len(idx) == len(set(idx)) == 6


def test_pool_seed_round_disjoint():
    # r51b2 撞 r50b2 事故回歸(2026-08-02):round 必須參與 OOD 池 seed 推導,
    # 跨輪同批號不得同池;舊有效空間(base+batch, batch≤54)也不得被新式撞上。
    from script.dedust import _pool_seed
    base = 20260808
    assert _pool_seed(base, 51, 2) != _pool_seed(base, 50, 2)
    news = {_pool_seed(base, r, b) for r in range(50, 60) for b in range(1, 55)}
    olds = {base + b for b in range(1, 55)}
    assert not news & olds


def test_diag_bridge_sites():
    # R54 菱形橋:偵測決定性/零接點=零菱形/L 型不加料/相鄰角碰撞縮橋(核准計畫 §6)
    import numpy as np
    from antenna.patch.patch_simulator.single_port import diag_bridge_sites
    z = np.zeros((25, 25), dtype=bool)
    assert diag_bridge_sites(z, 0.10, 0.2) == ([], 0)          # 空盤
    m = z.copy(); m[3, 3] = m[4, 4] = True                     # 真對角
    s, sk = diag_bridge_sites(m, 0.10, 0.2)
    assert len(s) == 1 and sk == 0 and abs(s[0][0] - 0.8) < 1e-9 and abs(s[0][1] - 0.8) < 1e-9
    m2 = m.copy(); m2[3, 4] = True                             # L 型(一正交位有金屬)=不加料
    assert diag_bridge_sites(m2, 0.10, 0.2)[0] == []
    m3 = z.copy(); m3[3, 3] = m3[4, 4] = m3[5, 3] = True       # 同格相鄰角雙橋(X 谷)
    s3, _ = diag_bridge_sites(m3, 0.14, 0.2)
    assert len(s3) == 2 and all(w < 0.14 for _, _, w in s3)    # 0.14 觸碰撞規則→縮
    s3b, _ = diag_bridge_sites(m3, 0.10, 0.2)
    assert all(abs(w - 0.10) < 1e-9 for _, _, w in s3b)        # 0.10 不縮
