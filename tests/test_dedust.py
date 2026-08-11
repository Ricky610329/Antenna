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


def test_spread_idx_stratified_deterministic():
    """R9 近標帶分層抽樣：均勻、含頭尾、決定性；帶內不足直接全取。"""
    from script.dedust import spread_idx
    idx = spread_idx(133, 12)
    assert len(idx) == 12 and idx[0] == 0 and idx[-1] == 132
    assert idx == sorted(idx) and len(set(idx)) == 12   # 嚴格遞增、無重複
    assert spread_idx(133, 12) == idx                    # 決定性
    assert spread_idx(5, 12) == [0, 1, 2, 3, 4]          # 不足 → 全取


def test_symmetrize_full_and_partial():
    """對稱先驗變體：half=12 全鏡射（外 12 欄左右一致）；half=10 中央 5 欄保留原樣。"""
    from script.dedust import symmetrize, FEED
    rng = np.random.default_rng(7)
    p = rng.random((25, 25)) < 0.5
    p[FEED] = True
    full = symmetrize(p, 12)
    assert all((full[:, 24 - j] == full[:, j]).all() for j in range(12))
    part = symmetrize(p, 10)
    assert all((part[:, 24 - j] == part[:, j]).all() for j in range(10))
    assert full[FEED] and part[FEED]                     # feed 永遠金屬


def test_occlude_block_surgical_and_feed_safe():
    """遮蔽掃描：只清目標區塊、不修復（孤件保留）；feed 像素在被遮區塊內也保留。"""
    from script.dedust import occlude_block, FEED
    p = np.ones((25, 25), bool)
    q, removed = occlude_block(p, 0, 0)
    assert removed == 25 and not q[0:5, 0:5].any() and q[5:].all()   # 只動 (0,0) 區塊
    q2, removed2 = occlude_block(p, 4, 2)                            # feed (24,12) 落在 b(4,2)
    assert q2[FEED] and removed2 == 24                               # feed 保留 → 只拔 24
    empty = np.zeros((25, 25), bool)
    assert occlude_block(empty, 1, 1)[1] <= 0                        # 空區塊=無資訊


def test_perturb_blocks_confined():
    """遮蔽圖知情編輯：翻轉只落在指定區塊集合內。滿板驗「移除」不出塊（空板加點會被除塵拔＝預期,不適合驗）。"""
    from script.dedust import perturb_blocks
    p = np.ones((25, 25), bool)
    q = perturb_blocks(p, 8, seed=1, blocks=((0, 1), (0, 2)))
    removed = p & ~q
    rows, cols = np.where(removed)
    assert removed.sum() > 0
    assert rows.max() < 5 and 5 <= cols.min() and cols.max() < 15   # 挖洞只在 (0,1)/(0,2) 兩塊內


def test_align_curve_regression_shifted_17_points():
    """回歸 (2026-07-06 w17 翻案根因)：Interpolating 掃頻回傳『恰 17 點但頻點偏格』時,
    舊邏輯 (點數≠17 才內插) 按索引錯位 → 一律按頻率值對位。"""
    from antenna.patch.patch_simulator.single_port import align_curve
    exp = np.linspace(24, 32, 17)
    # 情境1: 網格完全相符 → 原值直通
    vals = np.arange(17.0)
    assert (align_curve(exp, vals, exp) == vals).all()
    # 情境2: 恰 17 點但整體偏移 0.5GHz → 必須內插回網格,不可按索引直塞 (舊 bug)
    shifted = exp + 0.5
    out = align_curve(shifted, vals, exp)
    assert not (out == vals).all()                      # 舊行為=原樣直塞 → 抓到就是退化
    assert abs(out[8] - np.interp(exp[8], shifted, vals)) < 1e-12
    # 情境3: 點數≠17 → 內插 (原本就有的行為)
    dense = np.linspace(24, 32, 33)
    assert len(align_curve(dense, np.linspace(0, 1, 33), exp)) == 17


def test_edge_sets_feed_protected():
    """公差掃描的合法擾動位置：金屬邊緣+貼金屬介質;feed 永遠不可動。"""
    from script.dedust import edge_sets, FEED
    p = np.zeros((25, 25), bool)
    p[10:15, 10:15] = 1
    p[FEED] = True
    em, ed = edge_sets(p)
    assert em[10, 10] and not em[12, 12]        # 角=邊緣,中心=內部
    assert ed[9, 12] and not ed[0, 0]           # 貼金屬的介質才算
    assert not em[FEED]                          # feed 保護


def test_oob_metrics_directions():
    """帶外選擇性：S11 取遠帶外最壞(min,想要高)、Gain 取遠帶外最壞(max,想要低)、惡度=差。"""
    from script.dedust import oob_metrics
    r = np.zeros((2, 17))
    r[0, :] = -12.0; r[0, 0] = -3.0            # 帶外 S11 貼 0 較好 → min 取到 -12(帶外壞點在別處)?
    r[0, 16] = -12.0
    r[1, :] = 5.0; r[1, 1] = -8.0
    m = oob_metrics(r)
    assert m["oob_s11_min"] == -12.0            # 遠帶外 8 點中最負的 S11
    assert m["oob_gain_max"] == 5.0             # 遠帶外最高的 Gain
    assert m["oob_bad"] == 17.0
    assert m["oob_gain_max_lo"] == 5.0 and m["oob_gain_max_hi"] == 5.0   # 兩側分項
    assert m["rolloff_lo"] == 0.0               # 帶緣 5.0 − 低側最高 5.0
    assert 24.0 <= m["oob_gain_argmax"] <= 32.0


def test_add_block_new_component_or_none():
    """add_block：留 gap=獨立新件（組數+1）;貼太近會併件→回 None;出界→None。"""
    from script.dedust import add_block, piece_stats
    base = np.zeros((25, 25), bool)
    base[20:25, 8:17] = True                     # 下方主件（含 feed (24,12)）
    out = add_block(base, 2, 3, 3, 3)            # 遠處 3×3 → 新件
    assert out is not None and piece_stats(out)["n_comp"] == 2
    assert add_block(base, 19, 8, 1, 4) is None  # 緊貼主件上緣 (gap=1 不足) → 併件風險 → None
    assert add_block(base, 24, 23, 2, 3) is None # 出界 → None


def test_add_bridge_connects_components():
    """add_bridge：懸浮件被 L 形 1px 橋接到饋電主件 → 組件數減一;材料只增不減。"""
    from script.dedust import add_bridge, piece_stats
    base = np.zeros((25, 25), bool)
    base[20:25, 8:17] = True                     # 主件（含 feed）
    base[5:8, 10:14] = True                      # 懸浮件
    out = add_bridge(base, comp_rank=1, pair_rank=0)
    assert out is not None
    assert piece_stats(out)["n_comp"] == 1
    assert out.sum() > base.sum()                # 只加金屬
    assert add_bridge(base, comp_rank=2) is None # 沒有第二個懸浮件


def test_resize_component_grow_shrink():
    """resize_component：wings 成組縮放保拓撲;grow 不併件;shrink 出碎片回 None。"""
    from script.dedust import resize_component, piece_stats
    base = np.zeros((25, 25), bool)
    base[18:25, 6:19] = True                     # 主件(含 feed)
    base[2:6, 3:8] = True                        # 左翼 4×5
    base[2:6, 17:22] = True                      # 右翼
    g = resize_component(base, "wings", 1)
    assert g is not None and piece_stats(g)["n_comp"] == 3
    assert g.sum() > base.sum()
    sh = resize_component(base, "wings", -1)
    assert sh is not None and sh.sum() < base.sum() and piece_stats(sh)["n_comp"] == 3
    assert resize_component(base, "wings", -2) is None   # 4×5 翼縮 2 圈 → 消失
    gm = resize_component(base, "main", 1)
    assert gm is not None and piece_stats(gm)["n_comp"] == 3   # 沒併到翼


def test_sel_score_lexicographic():
    """sel_score（價值軸單標量,2026-07-12）:過線+buffer=純 oob;缺口罰 κ;無 rad 視為未過。"""
    from script.dedust import sel_score
    assert sel_score(0.20, 0.1, 9.0) == 9.0          # 合格 → 純 oob
    assert sel_score(0.50, 0.1, 9.0) == 9.0          # 帶內餘裕不加分（Ricky 定調）
    assert sel_score(0.05, 0.1, 9.0) == 10.0         # wm 缺口 0.10 × κ10
    assert sel_score(0.20, -0.1, 9.0) == 10.0        # rad 缺口 0.10 × κ10
    assert sel_score(0.20, None, 9.0) == 19.0        # 無方向圖=rad 視 -1.0 → 罰 10


def test_oob_metrics_contrast_sides():
    """contrast_lo/hi＝帶內 Gain min − 帶外各側 Gain max（相對選擇性追蹤欄,2026-07-12）。"""
    import numpy as np
    from script.dedust import oob_metrics
    s11 = np.full(17, -12.0)
    gain = np.full(17, 5.0)
    gain[:4] = 4.5                               # 低側帶外高（裙擺）
    gain[13:] = -2.0                             # 高側帶外低（健康滾降）
    gain[8] = 4.2                                # 帶內最低點
    m = oob_metrics(np.stack([s11, gain]))
    assert m["contrast_lo"] == round(4.2 - 4.5, 2) == -0.3
    assert m["contrast_hi"] == round(4.2 - (-2.0), 2) == 6.2
    assert m["oob_gain_max"] == 4.5


def test_jobs_add_concurrent_lock(tmp_path, monkeypatch):
    """jobs.json 並發壞檔回歸（2026-07-22/07-24 兩起實錘）：8 執行緒同時 jobs-add
    → 鎖檔序列化讀-改-寫,不丟單、不壞檔、鎖釋放乾淨。"""
    import json
    import threading
    from types import SimpleNamespace
    import script.dedust as dd
    monkeypatch.setattr(dd, "DATASET_PATH", tmp_path)
    for i in range(8):
        d = tmp_path / f"in{i}"
        d.mkdir()
        (d / "manifest.json").write_text("[]", encoding="utf-8")
    errs = []

    def add(i):
        try:
            dd.jobs_add(SimpleNamespace(input=f"in{i}", store=f"st{i}", prio=3))
        except BaseException as e:                       # SystemExit 也算失敗
            errs.append(e)

    ts = [threading.Thread(target=add, args=(i,)) for i in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errs
    jobs = json.loads((tmp_path / "jobs.json").read_text(encoding="utf-8"))
    assert sorted(j["store"] for j in jobs) == [f"st{i}" for i in range(8)]
    assert not (tmp_path / "jobs_state" / "jobs.lock").exists()


def test_group_mutate_semantics():
    """組級變異（R41 C 臂）回歸：FEED 不可清、非 diag 算子不動骨架組（>25px 凍結）、
    每次變異 diff>0 且 op 記帳完整。"""
    import numpy as np
    from scipy.ndimage import label
    from script.dedust import _group_mutate, FEED
    p0 = np.zeros((25, 25), bool)
    p0[12:22, 2:22] = True                      # 骨架 200px
    p0[2:5, 3:8] = True                         # 中件 15px
    p0[5, 15] = p0[5, 16] = True                # 小件 2px
    p0[8, 20] = True                            # 小件 1px
    p0[FEED] = True
    lab, _ = label(p0, structure=np.ones((3, 3), bool))
    big = {i for i in range(1, lab.max() + 1) if (lab == i).sum() > 25}
    big_mask = np.isin(lab, list(big))
    rng = np.random.default_rng(7)
    seen_ops = set()
    for _ in range(300):
        r = _group_mutate(p0, rng)
        if r is None:
            continue
        q, op, d = r
        assert q[FEED], "FEED 被清掉"
        assert d == int((q != p0).sum()) > 0
        assert op[0].startswith("grp_")
        seen_ops.add(op[0])
        if op[0] != "grp_diag":
            assert not ((q != p0) & big_mask).any(), f"{op[0]} 動到骨架"
    assert {"grp_grow", "grp_shrink", "grp_move", "grp_del"} <= seen_ops


def test_diag_clean_group_preserving():
    """對角清潔組保持約束（Ricky 2026-07-26「補實或移除都要符合組的規範」）：
    ①雙口徑組數守恆 ②橋接對角一律不動 ③FEED 不可清。"""
    import numpy as np
    from script.dedust import diag_clean, _grp_counts, FEED
    p = np.zeros((25, 25), bool)
    p[10:16, 4:12] = True                    # 主件
    p[16, 12] = True                         # 橋接對角:只靠 (15,11)-(16,12) 對角連
    p[5, 3] = p[6, 4] = True                 # 孤立橋接對角（兩端 4-conn 不連通）
    p[12, 12] = p[11, 13] = p[12, 14] = True # 冗餘側:與主件另有路徑? 不必然,交給演算法判
    p[FEED] = True
    base = _grp_counts(p)
    q, n, log = diag_clean(p)
    assert _grp_counts(q) == base, "組數未守恆"
    assert q[FEED], "FEED 被清"
    kinds = {k for _, _, k, _ in log}
    assert "bridge" in kinds, "測資應含橋接對角"
    for r, c, kind, act in log:
        if kind == "bridge":
            assert act == "skip", "橋接對角被動了"
    # 冗餘對角若有被清,前後 diff 僅限被清處
    assert int((q != p).sum()) == n


def test_rand_grammar_contract():
    """組文法採樣器契約（R43 前置）：五文法皆出合法 pattern（FEED/金屬界）、決定性、
    GD 主零對角 vs GDd 主帶對角（解耦對比）。"""
    import numpy as np
    from script.dedust import _rand_grammar, diag_bridge, FEED
    for gs in ("GA", "GA2", "GB", "GC", "GD", "GDd"):
        rng = np.random.default_rng(7)
        outs = []
        fails = 0
        while len(outs) < 20 and fails < 400:
            q = _rand_grammar(rng, gs)
            if q is None:
                fails += 1
                continue
            assert q[FEED] and 140 <= int(q.sum()) <= 560
            outs.append(q)
        assert len(outs) == 20, f"{gs} 產出不足"
        # 決定性:同 seed 重抽第一張一致
        rng2 = np.random.default_rng(7)
        q2 = None
        while q2 is None:
            q2 = _rand_grammar(rng2, gs)
        assert (q2 == outs[0]).all(), f"{gs} 非決定性"
        dbs = [diag_bridge(q) for q in outs]
        if gs == "GD":
            assert np.mean(np.array(dbs) == 0) >= 0.7, "GD 應主零對角"
        if gs == "GDd":
            assert np.mean(np.array(dbs) > 0) >= 0.7, "GDd 應主帶對角"


def test_tri_rad_zero_is_qualified():
    """audit 2026-07-29: rad_margin 0.00/−0.0 是合法合格值,falsy `or -9` 會吃掉貼線解。"""
    from script.dedust import _tri
    assert _tri({"wm": [0, 0, 0.2], "rad_margin": 0.0})
    assert _tri({"wm": [0, 0, 0.2], "rad_margin": -0.0})
    assert _tri({"wm": [0, 0, 0.2], "rad_margin": 0.5})
    assert not _tri({"wm": [0, 0, 0.2]})                       # 缺鍵=不合格
    assert not _tri({"wm": [0, 0, 0.2], "rad_margin": None})   # None=缺件
    assert not _tri({"wm": [0, 0, 0.2], "rad_margin": -0.01})
    assert not _tri({"wm": [0, 0, -0.1], "rad_margin": 1.0})


def test_group_mutate_contract_after_reweight():
    """analysis-07 權重化後契約不變:回傳 (q,op,diff)、feed 恆真、決定性(同 seed 同輸出)。"""
    import numpy as np
    from script.dedust import _group_mutate, FEED
    rng = np.random.default_rng(11)
    base = np.zeros((25, 25), bool)
    base[10:20, 8:18] = True      # 主件
    base[2:5, 2:5] = True         # 中件
    base[22, 22] = True           # 小件
    base[FEED] = True
    outs = []
    fails = 0
    while len(outs) < 15 and fails < 300:
        r = _group_mutate(base, rng)
        if r is None:
            fails += 1
            continue
        q, op, d = r
        assert q[FEED] and d >= 1 and isinstance(op, list) and op[0].startswith("grp_")
        outs.append((q, tuple(op)))
    assert len(outs) == 15, "組級變異產出不足"
    rng2 = np.random.default_rng(11)
    r2 = None
    while r2 is None:
        r2 = _group_mutate(base, rng2)
    assert (r2[0] == outs[0][0]).all(), "組級變異非決定性"


def test_cs_sort_key_notarize_first():
    """audit 2026-07-29: 公證店(dedust_rNNn*)前移——「certified 先見先贏」去重不變式。"""
    import importlib.util
    import os
    spec = importlib.util.spec_from_file_location(
        "_smr_keyonly", os.path.join(os.path.dirname(__file__), "..", "script", "sm_reanchor.py"))
    src = open(spec.origin, encoding="utf-8").read()
    ns = {}
    start = src.index("def _cs_sort_key")
    end = src.index("def _load_clean_stores")
    exec(src[start:end], ns)   # 只執行排序鍵函式,避免模組層 CLEAN_STORES 掃 NAS
    key = ns["_cs_sort_key"]
    names = ["dedust_r45b3a", "dedust_r23n1w", "dedust_auto37", "dedust_r42n2a", "dedust_c45g2_p01"]
    assert sorted(names, key=key)[:2] == ["dedust_r23n1w", "dedust_r42n2a"]
    assert key("dedust_r9n1") == 0 and key("dedust_r45b3a") == 1


# ---------------------------------------------------------------- dual 開線（2026-08-10，D2 #3-#6/#8/#9）
def _dual_pattern(seed=0):
    import numpy as np
    return np.random.default_rng(seed).random((25, 25)) < 0.5


class TestDualHelpers:
    """dual 可製造閘與上下鏡像（single 的 FEED/symmetrize 是**左右**軸，dual 必須另立一套）。"""

    def test_pads_forced_full(self):
        from script.dedust import dual_pads
        q = dual_pads(np.zeros((25, 25), bool))
        assert q[0:5, 10:15].all() and q[20:25, 10:15].all()     # 上下兩個 5×5 饋墊
        assert q.sum() == 50                                      # 只補饋墊，不多動一格
        assert q[0, 12] and q[24, 12]                             # 饋電點 [(0,12),(24,12)]

    def test_mirror_is_row_axis_and_keeps_pads(self):
        from script.dedust import dual_mirror
        q = dual_mirror(_dual_pattern(3))
        assert (q == q[::-1]).all()                               # 沿 row 中線（row 12 為軸）對稱
        assert q[0:5, 10:15].all() and q[20:25, 10:15].all()
        assert (dual_mirror(_dual_pattern(3)) == q).all()         # 決定性（純函式）

    def test_flip_avoids_pads_and_counts(self):
        from script.dedust import dual_flip, dual_pads
        p = dual_pads(_dual_pattern(5))
        rng = np.random.default_rng(11)
        q = dual_flip(p, 3, rng)
        assert int((q != p).sum()) == 3                           # 恰翻 d 格
        assert q[0:5, 10:15].all() and q[20:25, 10:15].all()      # 饋墊沒被翻掉
        assert not ((q != p)[0:5, 10:15]).any() and not ((q != p)[20:25, 10:15]).any()
        assert (dual_flip(p, 3, np.random.default_rng(11)) == q).all()   # 同 seed 決定性


def test_cli_help_renders(monkeypatch, capsys):
    """回歸（2026-08-10 修）：help 字串裡的**字面 %** 沒跳脫會讓 `dedust --help` 整個炸
    （argparse 對 help 做 %-格式化；select-r25 的「48%」是實犯）。順帶當全部子命令 help 的語法閘。"""
    import sys
    import pytest
    import script.dedust as dd
    monkeypatch.setattr(sys, "argv", ["dedust", "--help"])
    with pytest.raises(SystemExit) as e:
        dd.main()
    assert e.value.code == 0
    assert "select-dual" in capsys.readouterr().out


def test_port_sweep_defaults_match_simulators():
    """跨實作對帳：dedust 的 `--sweep` port 預設表必須等於兩個模擬器**自己**的預設。
    dual 的 harvest_dual 一萬筆真值全是 Fast 跑的——共用一個字面值（Interpolating）會靜默換分佈。"""
    import inspect
    from antenna.patch import DualPortSimulator, SinglePortRadSimulator
    from script.dedust import PORT_SWEEP
    for port, cls in (("single", SinglePortRadSimulator), ("dual", DualPortSimulator)):
        sig = inspect.signature(cls.__init__)
        if "sweep_type" not in sig.parameters:             # single_rad 由父類吃 **kwargs
            sig = inspect.signature(cls.__mro__[1].__init__)
        assert PORT_SWEEP[port] == sig.parameters["sweep_type"].default, f"{port} 掃頻預設不一致"


def _fake_dual_losses(monkeypatch, per=None, energy=0.91):
    """施工包 A（antenna.losses.worst_margin_dual / dual_energy_max）落地前也能驗 dedust 這側的配管。"""
    import antenna.losses as L
    per = per if per is not None else dict(m1=1.0, m2=-2.0, m3=0.5, m4=3.0, m5=-9.0, m6=-8.0)
    monkeypatch.setattr(L, "worst_margin_dual",
                        lambda resp, labels, targets: (min(per[k] for k in ("m1", "m2", "m3", "m4")), per),
                        raising=False)
    monkeypatch.setattr(L, "dual_energy_max", lambda resp: energy, raising=False)
    return per


def test_dual_metrics_fields(monkeypatch):
    """results.json 的 dual 欄位契約：m1..m6 全入檔（m5/m6=記帳欄）+ energy_max 3 位（1.001/0.999 要
    分得出）+ s11_s22_gap（鏡像假說 round-57 §1③ 的直接量，取**最壞頻點**）。"""
    from script.dedust import dual_metrics
    _fake_dual_losses(monkeypatch, energy=1.0004)
    resp = np.zeros((3, 17))
    resp[2, 4] = -2.5                              # S22 在 idx4 與 S11 差 2.5dB（其餘同值）
    out = dual_metrics(resp, ["S11", "S21", "S22"], {})
    assert set(out) == {"m1", "m2", "m3", "m4", "m5", "m6", "energy_max", "s11_s22_gap"}
    assert out["m2"] == -2.0 and out["m5"] == -9.0
    assert out["energy_max"] == 1.0                # 3 位（>1 的告警不會被 2 位四捨五入吃掉方向）
    assert out["s11_s22_gap"] == 2.5               # max_f |S11−S22|，不是平均（身分等式看最壞點）
    assert dual_metrics(np.zeros((3, 17)), ["S11", "S21", "S22"], {})["s11_s22_gap"] == 0.0
    assert "rad" not in out and "sel" not in out and "oob_bad" not in out


def test_dual_metrics_rejects_incomplete_per(monkeypatch):
    """契約守衛：per 少一項就炸（別讓缺欄靜默進 results.json）。"""
    import pytest
    from script.dedust import dual_metrics
    _fake_dual_losses(monkeypatch, per=dict(m1=1.0, m2=2.0, m3=3.0, m4=4.0))
    with pytest.raises(KeyError):
        dual_metrics(np.zeros((3, 17)), ["S11", "S21", "S22"], {})


def test_dual_label_margins_mapping():
    """entry["wm"] 的 per-label 欄：S11=m1 / S21=min(m3,m4) / S22=m2；**m5/m6 不進**（決策點③）。"""
    from script.dedust import _dual_label_margins
    per = dict(m1=1.0, m2=2.0, m3=0.5, m4=-3.0, m5=-99.0, m6=-99.0)
    assert _dual_label_margins(per, ["S11", "S21", "S22"]) == [1.0, -3.0, 2.0]
    per2 = dict(per, S11=7.0, S21=8.0, S22=9.0)          # per 若自帶 label 鍵則優先
    assert _dual_label_margins(per2, ["S11", "S21", "S22"]) == [7.0, 8.0, 9.0]


def _mk_input(root, name, ids_ports, pattern):
    """造一個迷你輸入夾：manifest（可帶 port）+ 每 id 一個 .pt。"""
    import json
    import torch
    d = root / name
    d.mkdir()
    man = []
    for pid, port in ids_ports:
        m = {"id": pid, "kind": "dual" if port == "dual" else "orig"}
        if port:
            m["port"] = port
        man.append(m)
        torch.save(torch.tensor(pattern.astype("float32")), str(d / f"{pid}.pt"))
    (d / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    return d


def test_check_dup_splits_by_port_domain(tmp_path, monkeypatch):
    """查重分域（D2 #6）：同一張 pattern 在 single 夾與 dual 夾各測一次＝合法對照，不算重複；
    同域撞到才 exit 1。"""
    import pytest
    import script.dedust as dd
    from types import SimpleNamespace
    monkeypatch.setattr(dd, "DATASET_PATH", tmp_path)
    p = _dual_pattern(7)
    _mk_input(tmp_path, "dedust_hist_input", [("h1", None)], p)          # 歷史 single 夾（無 port 欄）
    _mk_input(tmp_path, "dedust_d1_input", [("d1", "dual")], p)          # dual 夾，同一張 pattern
    dd.check_dup(SimpleNamespace(input="dedust_d1_input"))               # 跨域 → 不得判重複

    _mk_input(tmp_path, "dedust_d2_input", [("d2", "dual")], p)          # 另一個 dual 夾，同 pattern
    with pytest.raises(SystemExit):
        dd.check_dup(SimpleNamespace(input="dedust_d2_input"))           # 同域 → 必須攔

    _mk_input(tmp_path, "dedust_s2_input", [("s2", None)], p)            # single 對 single 也照攔
    with pytest.raises(SystemExit):
        dd.check_dup(SimpleNamespace(input="dedust_s2_input"))


def test_check_dup_exempts_slotw_geometry_variants(tmp_path, monkeypatch):
    """R60 縫寬變體＝**bits 不變**的幾何變體（照 meshconv/diagbridge 前例）→ 查重豁免。
    不豁免的話，同一張 parent 掃五種縫寬會被自己攔下來、整輪發不了車。"""
    import json
    import pytest
    from types import SimpleNamespace
    import script.dedust as dd
    monkeypatch.setattr(dd, "DATASET_PATH", tmp_path)
    p = _dual_pattern(9)

    def _kind(folder, kind):
        f = tmp_path / folder / "manifest.json"
        man = json.loads(f.read_text(encoding="utf-8"))
        for m in man:
            m["kind"] = kind
        f.write_text(json.dumps(man), encoding="utf-8")

    _mk_input(tmp_path, "dedust_r60sw50_input", [("d59_x~sw50", "dual")], p)
    _mk_input(tmp_path, "dedust_r60sw75_input", [("d59_x~sw75", "dual")], p)   # 同 bits、不同縫寬
    _kind("dedust_r60sw50_input", "slotw")
    _kind("dedust_r60sw75_input", "slotw")
    dd.check_dup(SimpleNamespace(input="dedust_r60sw75_input"))                # 豁免 → 不得 exit 1

    #! 反面對照：同樣的 bits 重複、kind 不在豁免集合 → 必須攔（證明上面不是「根本沒查到」）
    _mk_input(tmp_path, "dedust_r60plain_input", [("d59_x_plain", "dual")], p)
    with pytest.raises(SystemExit):
        dd.check_dup(SimpleNamespace(input="dedust_r60plain_input"))


def test_hfss_setup_whitelist_is_port_scoped():
    """幾何鍵分域（鐵則 7）：`slot_spec` 只有 dual 模擬器有、`diag_bridge_w`/`pixel_count`
    只有 single 有——給錯域要顯性擋掉，不能靜默吞（吞掉＝整夾用預設幾何白燒）。"""
    from script.dedust import _hfss_setup_keys
    assert "slot_spec" in _hfss_setup_keys("dual") and "slot_spec" not in _hfss_setup_keys("single")
    assert {"diag_bridge_w", "pixel_count"} <= _hfss_setup_keys("single")
    assert not {"diag_bridge_w", "pixel_count"} & _hfss_setup_keys("dual")
    assert {"timeout", "max_delta_s", "max_passes"} <= _hfss_setup_keys("dual") & _hfss_setup_keys("single")


def test_hfss_setup_keys_are_real_simulator_params():
    """跨實作對帳：白名單的鍵是 `SIM_CLS(**hfss_setup)` 直接 pass-through 的 → 每個鍵都必須
    真的是該 port 模擬器的建構參數，否則發車當下才 TypeError（`timeout` 例外：只進看門狗）。"""
    import inspect
    from antenna.patch import DualPortSimulator, SinglePortRadSimulator
    from script.dedust import _hfss_setup_keys
    for port, cls in (("dual", DualPortSimulator), ("single", SinglePortRadSimulator)):
        sig = inspect.signature(cls.__init__)
        if "sweep_type" not in sig.parameters:                 # single_rad 由父類吃 **kwargs
            sig = inspect.signature(cls.__mro__[1].__init__)
        for k in _hfss_setup_keys(port) - {"timeout"}:
            assert k in sig.parameters, f"{port}: {k} 不是 {cls.__name__} 的建構參數"


def test_jobs_add_carries_config(tmp_path, monkeypatch):
    """派工鏈帶 config（D2 #5）：給了才寫進 job dict；不給＝worker 沿用自己的 --config。"""
    import json
    from types import SimpleNamespace
    import script.dedust as dd
    monkeypatch.setattr(dd, "DATASET_PATH", tmp_path)
    for n in ("inA", "inB"):
        (tmp_path / n).mkdir()
        (tmp_path / n / "manifest.json").write_text("[]", encoding="utf-8")
    dd.jobs_add(SimpleNamespace(input="inA", store="stA", prio=3, config="configs/dual_r1_eval.yaml"))
    dd.jobs_add(SimpleNamespace(input="inB", store="stB", prio=3))       # 舊呼叫端（無 config 屬性）不得炸
    jobs = {j["store"]: j for j in json.loads((tmp_path / "jobs.json").read_text(encoding="utf-8"))}
    assert jobs["stA"]["config"] == "configs/dual_r1_eval.yaml"
    assert "config" not in jobs["stB"]


def test_report_dual_is_port_aware(tmp_path, monkeypatch, capsys):
    """report port-aware（D2 #9）：dual 表頭＝m1..m4/energy，無 rad、無三標；能量>1 要告警。"""
    import json
    from types import SimpleNamespace
    import script.dedust as dd
    monkeypatch.setattr(dd, "DATASET_PATH", tmp_path)
    _mk_input(tmp_path, "dedust_r99b1_input",
              [("d99b1_a_00", "dual"), ("d99b1_s_00", "dual"), ("d99b1_r_00", "dual"),
               ("d99b1_r_01", "dual")], _dual_pattern(1))
    man = json.loads((tmp_path / "dedust_r99b1_input" / "manifest.json").read_text(encoding="utf-8"))
    for m, arm in zip(man, ("a", "s", "r", "r")):
        m["arm"] = arm                                       # r_00=error 列、r_01=待跑列（驗欄數）
    (tmp_path / "dedust_r99b1_input" / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    (tmp_path / "dedust_r99b1").mkdir()
    (tmp_path / "dedust_r99b1" / "results.json").write_text(json.dumps({
        "d99b1_a_00": {"wm": [1.0, -0.5, 2.0, -0.5], "m1": 1.0, "m2": 2.0, "m3": -0.5, "m4": 3.0,
                       "m5": -9.0, "m6": -8.0, "energy_max": 0.91, "s11_s22_gap": 9.2,
                       "time_s": 120.0},
        "d99b1_s_00": {"wm": [1.0, 1.0, 2.0, 1.0], "m1": 1.0, "m2": 1.0, "m3": 2.0, "m4": 4.0,
                       "m5": -9.0, "m6": -9.0, "energy_max": 1.004, "s11_s22_gap": 0.02,
                       "time_s": 120.0},
        "d99b1_r_00": {"error": "COM 例外", "attempts": 1}}), encoding="utf-8")
    dd.report(SimpleNamespace(input="dedust_r99b1_input", store="dedust_r99b1"))
    out = capsys.readouterr().out
    #? 每一列的欄數必須等於表頭（error/待跑 列曾少一格 → markdown 表整個錯位）
    rows = [ln for ln in out.splitlines() if ln.startswith("|")]
    assert len({ln.count("|") for ln in rows}) == 1, "表格欄數不一致"
    assert "m1 S11帶內" in out and "energy" in out
    assert "rad" not in out and "三標" not in out            # dual 無遠場 → 這兩個概念不該出現
    assert "合格（worst=min(m1..m4) ≥ 0）：1 筆" in out
    assert "energy_max>1" in out                             # 能量自證告警
    assert "鏡像假說" in out and "成立" in out                # §1③ 判準直接落在表尾（0.02 < 1dB）


def test_select_dual_arm_table(tmp_path, monkeypatch):
    """select-dual 臂別契約（D2 #8）：a/n/r/s 席次、id 規範、port 欄、饋墊閘、鏡像對稱、決定性。
    ⚠ 這批 = 100 筆真 HFSS（~數小時機時），生成錯了才發現＝燒掉的預算回不來。"""
    import json
    import os
    import numpy as np
    import torch
    from types import SimpleNamespace
    import antenna.losses as L
    import antenna.utils.store as ST
    import script.dedust as dd

    N = 12
    #? 真實的 harvest pattern 本來就疊了 M_feed（雙饋墊全滿）——除了 idx 10，用來驗
    #  「可製造閘動到錨」時 pads_forced 會被標記（動過就不能拿來當存檔 y vs 重測的對帳）
    pats = [dd.dual_pads(np.random.default_rng(100 + i).random((25, 25)) < 0.5) for i in range(N)]
    pats[10] = pats[10].copy()
    pats[10][0, 10] = False
    resps = [torch.full((3, 17), float(i)) for i in range(N)]

    class FakeStore:
        def __init__(self, rootdir, verbose=True):
            pass

        def __len__(self):
            return N

        def __getitem__(self, i):
            return torch.tensor(pats[i].astype(np.float32)), resps[i]

    monkeypatch.setattr(ST, "SampleStore", FakeStore)
    monkeypatch.setattr(L, "worst_margin_dual",
                        lambda y, labels, targets: (float(torch.as_tensor(y)[0, 0]), {}), raising=False)
    monkeypatch.setattr(dd, "DATASET_PATH", tmp_path)
    (tmp_path / "harvest_dual").mkdir()
    cfg = os.path.join(dd.REPO, "configs", "dual_base.yaml")
    args = SimpleNamespace(round=99, batch=1, config=cfg, harvest="harvest_dual",
                           anchors=4, per_anchor=2, rand=3, seed=20260810)
    dd.select_dual(args)

    d = tmp_path / "dedust_r99b1_input"
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    assert len(man) == 4 + 8 + 3 + 3
    by = {}
    for m in man:
        by.setdefault(m["arm"], []).append(m)
        assert m["port"] == "dual" and m["kind"] == "dual"
        assert m["id"].startswith("d99b1_")
        q = np.asarray(torch.load(str(d / f"{m['id']}.pt"), weights_only=True)).reshape(25, 25) > 0.5
        assert q[0:5, 10:15].all() and q[20:25, 10:15].all(), "可製造閘：雙饋墊必全滿"
    assert [len(by[a]) for a in "anrs"] == [4, 8, 3, 3]
    assert [m["harvest_wm"] for m in by["a"]] == [11.0, 10.0, 9.0, 8.0]      # wm 降冪選錨
    assert [m["pads_forced"] for m in by["a"]] == [False, True, False, False]  # 只有被閘動過的那筆標記
    a_ids = {m["id"] for m in by["a"]}
    r_ids = {m["id"] for m in by["r"]}
    assert all(m["parent"] in a_ids and m["d"] in (1, 2, 3) for m in by["n"])
    assert all(m["parent"] in r_ids for m in by["s"])
    assert {m["d"] for m in by["n"]} == {1, 2, 3}                            # d 階梯有鋪開
    for m in by["s"]:                                                        # 鏡像臂真的上下對稱
        q = np.asarray(torch.load(str(d / f"{m['id']}.pt"), weights_only=True)).reshape(25, 25) > 0.5
        assert (q == q[::-1]).all()
    # 決定性：同參數重跑 → 位元級相同（可重現、可續跑）
    sig = {m["id"]: open(str(d / f"{m['id']}.pt"), "rb").read() for m in man}
    import shutil
    shutil.rmtree(str(d))
    dd.select_dual(args)
    assert {m["id"]: open(str(d / f"{m['id']}.pt"), "rb").read()
            for m in json.loads((d / "manifest.json").read_text(encoding="utf-8"))} == sig


def test_select_dual_end_to_end_real_ruler(tmp_path, monkeypatch):
    """零 mock 整合：真 SampleStore + 真 `worst_margin_dual` + 真 dual config 走一遍 select-dual。
    盯的是「錨真的是按同一把判準尺挑的」——排序若跟手算不一致，首批 20 個錨就選錯了。"""
    import json
    import os
    import numpy as np
    import torch
    from types import SimpleNamespace
    import script.dedust as dd
    from antenna.losses import worst_margin_dual
    from antenna.training import PORT_SPECS, load_config
    from antenna.utils.store import SampleStore

    monkeypatch.setattr(dd, "DATASET_PATH", tmp_path)
    src = SampleStore(tmp_path / "harvest_dual", verbose=False)
    rng = np.random.default_rng(2026)
    for i in range(6):
        x = torch.tensor((rng.random((25, 25)) < 0.5).astype(np.float32))
        y = torch.tensor(rng.uniform(-30, 0, (3, 17)).astype(np.float32))
        src.add(x, y)
    cfg = load_config(os.path.join(dd.REPO, "configs", "dual_base.yaml"))
    labels = PORT_SPECS["dual"]["labels"]
    ss = SampleStore(tmp_path / "harvest_dual", verbose=False)
    want = sorted(((float(worst_margin_dual(ss[i][1], labels, cfg.targets)[0]), i) for i in range(6)),
                  key=lambda t: (-t[0], t[1]))[:3]

    dd.select_dual(SimpleNamespace(round=98, batch=1,
                                   config=os.path.join(dd.REPO, "configs", "dual_base.yaml"),
                                   harvest="harvest_dual", anchors=3, per_anchor=1, rand=2,
                                   seed=20260810))
    man = json.loads((tmp_path / "dedust_r98b1_input" / "manifest.json").read_text(encoding="utf-8"))
    a = [m for m in man if m["arm"] == "a"]
    assert [m["harvest_idx"] for m in a] == [i for _, i in want]          # 錨＝真尺排序的前 N
    assert [m["harvest_wm"] for m in a] == [round(w, 3) for w, _ in want]
    assert len(man) == 3 + 3 + 2 + 2


def test_scheduler_steps_per_epoch(tmp_path):
    """audit 2026-07-29: scheduler 以 epoch 為單位(原 per-batch → 首 epoch 撞 1e-6 地板)。"""
    import torch
    from antenna.zoo import SURROGATES
    sm = SURROGATES["mlp"](str(tmp_path), 25 * 25, (2, 17))
    data = [(torch.rand(25, 25), torch.rand(2, 17)) for _ in range(8)]
    sm.train_by_datas(data, epochs=3, batch_size=4, verbose=False, early_stop=False)
    assert sm.scheduler.last_epoch == 3, f"scheduler 步數 {sm.scheduler.last_epoch}≠epochs(應 per-epoch)"
    assert sm.optimizer.param_groups[0]["lr"] > 1e-5, "lr 不應在 3 epoch 內撞地板"
