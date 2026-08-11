# -*- coding: utf-8 -*-
"""dual 自產(_dual_selfgen_cands)最小契約:決定性/饋墊必保/對稱性/去重。零 NAS。"""
import numpy as np

from script.dedust import _dual_selfgen_cands, _dual_pad_mask


def _gen(seed, anchors=None, hist=None, want=8):
    rng = np.random.default_rng(seed)
    return _dual_selfgen_cands(rng, anchors or [], hist if hist is not None else set(), want)


def test_deterministic_same_seed():
    a = _gen(42)
    b = _gen(42)
    assert len(a) == len(b) == 8
    for (p1, s1, _), (p2, s2, _) in zip(a, b):
        assert s1 == s2 and (p1 == p2).all()


def test_pads_forced_and_symmetric():
    pad = _dual_pad_mask()
    for p, src, _ in _gen(7):
        assert p[pad].all(), "雙饋墊必保"
        if src == "symr":
            assert (p == p[::-1, :]).all(), "對稱隨機必須上下對稱"


def test_dedup_via_hist():
    hist = set()
    a = _gen(1, hist=hist, want=6)
    keys = {p.tobytes() for p, _, _ in a}
    assert len(keys) == 6, "批內去重"
    b = _gen(1, hist=hist, want=6)          # 同 seed 但 hist 已含前批 → 全部重生成
    assert all(p.tobytes() not in keys for p, _, _ in b)


def test_anchor_mode_used_and_pads_kept():
    base = np.zeros((25, 25), bool)
    base[5:20, 5:20] = True                  # 225 px,過金屬量下限(150)
    base[_dual_pad_mask()] = True
    out = _gen(3, anchors=[("t", base)], want=10)
    srcs = {s for _, s, _ in out}
    assert "anchor_nbr" in srcs and "symr" in srcs
    for p, s, meta in out:
        if s == "anchor_nbr":
            assert meta["parent"] == "t" and 1 <= meta["d"] <= 4
            assert p[_dual_pad_mask()].all()
