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
