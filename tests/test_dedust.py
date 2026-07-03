# -*- coding: utf-8 -*-
"""tests/test_dedust.py — script/dedust.py 純函式（除塵/碎片統計/家族聚類/rad 窗餘裕）。零 NAS、零 HFSS。

除塵變體會直接燒 HFSS 預算（round-07），拔錯件＝浪費真模擬 → 這裡把變體生成的行為釘死。
"""
import numpy as np

from script.dedust import (FEED, close_holes, cluster_families, perturb_repair, piece_stats,
                           rad_window_margin, smooth_blob, strip_small)


def _pattern(*blocks):
    """25×25 全 0，依 (r0, r1, c0, c1) 填 1。"""
    p = np.zeros((25, 25), bool)
    for r0, r1, c0, c1 in blocks:
        p[r0:r1, c0:c1] = True
    return p


BIG = (0, 10, 0, 10)        # 100px 主件
FEED_BLK = (20, 25, 10, 15)  # 25px、蓋住 feed 像素 (24,12)


class TestStripSmall:
    def test_removes_dust_keeps_big(self):
        p = _pattern(BIG, FEED_BLK, (0, 1, 20, 21), (5, 6, 20, 22))   # +1px 粉塵 +2px 骨牌
        out, removed = strip_small(p, 2)                               # 拔 <2px → 只拔 1px
        assert removed == 1
        assert out.sum() == p.sum() - 1
        out, removed = strip_small(p, 4)                               # 拔 <4px → 1px+2px 都拔
        assert removed == 3
        assert out[0:10, 0:10].all() and out[24, 12]                   # 主件與 feed 組不動

    def test_protects_feed_component(self):
        p = _pattern(BIG, (24, 25, 12, 13))        # feed 組只有 1px——低於門檻仍必須保留
        out, removed = strip_small(p, 4)
        assert removed == 0 and out[24, 12]

    def test_noop_when_clean(self):
        p = _pattern(BIG, FEED_BLK)
        out, removed = strip_small(p, 4)
        assert removed == 0 and (out == p).all()


def test_piece_stats():
    p = _pattern(BIG, FEED_BLK, (0, 1, 20, 21), (5, 6, 20, 22), (12, 13, 20, 23))  # +1px +2px +3px
    s = piece_stats(p)
    assert s == dict(n_comp=5, n_1px=1, n_2_3px=2, main_px=100, metal_px=131)


def test_cluster_families_groups_neighbors():
    a = _pattern(BIG, FEED_BLK)
    a2 = a.copy()
    a2[0, 20:23] = True                            # a 翻 3px＝同家族
    b = _pattern((12, 22, 12, 22), FEED_BLK)       # 大跳＝另一家族
    labels = cluster_families([a, a2, b], max_dist=10)
    assert labels[0] == labels[1] != labels[2]


def test_close_holes_fills_enclosed_only():
    p = _pattern(BIG, FEED_BLK)
    p[3:5, 3:5] = False                            # 被主件包住的 2×2 洞
    out, filled = close_holes(p)
    assert filled == 4 and out[3:5, 3:5].all()
    out2, filled2 = close_holes(out)               # 沒洞 → no-op
    assert filled2 == 0 and (out2 == out).all()
    q = _pattern(BIG, FEED_BLK)
    q[0, 3] = False                                # 觸邊凹口不是洞、不能填
    assert close_holes(q)[1] == 0


class TestPerturbRepair:
    def test_clean_and_deterministic(self):
        p = _pattern(BIG, FEED_BLK)
        a = perturb_repair(p, 32, seed=7)
        b = perturb_repair(p, 32, seed=7)
        assert (a == b).all()                      # 同 seed 決定性（可續跑/可重現）
        assert a[FEED]                             # feed 永遠金屬
        assert piece_stats(a)["n_1px"] == 0        # 修復後無 1px 粉塵
        assert (perturb_repair(p, 32, seed=8) != a).any()   # 不同 seed 真的不同


def test_smooth_blob_clean_and_deterministic():
    a = smooth_blob(seed=3)
    assert (a == smooth_blob(seed=3)).all()
    assert a[FEED]
    s = piece_stats(a)
    assert s["n_1px"] == 0 and s["metal_px"] > 0


class TestRadWindowMargin:
    def test_flat_gain_passes(self):
        theta = np.arange(-90.0, 91.0)
        assert rad_window_margin(theta, np.full(181, 5.0), 45, 3) == 3.0   # 平坦 → 餘裕=floor

    def test_dip_inside_window_fails(self):
        theta = np.arange(-90.0, 91.0)
        gain = np.full(181, 5.0)
        gain[theta == 30] = 0.0                    # 窗內掉 5dB > floor 3
        assert rad_window_margin(theta, gain, 45, 3) == -2.0

    def test_dip_outside_window_ignored(self):
        theta = np.arange(-90.0, 91.0)
        gain = np.full(181, 5.0)
        gain[theta == 60] = -10.0                  # 窗外崩掉不影響 ±45 覆蓋
        assert rad_window_margin(theta, gain, 45, 3) == 3.0
