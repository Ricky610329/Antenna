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


def test_scheduler_steps_per_epoch(tmp_path):
    """audit 2026-07-29: scheduler 以 epoch 為單位(原 per-batch → 首 epoch 撞 1e-6 地板)。"""
    import torch
    from antenna.zoo import SURROGATES
    sm = SURROGATES["mlp"](str(tmp_path), 25 * 25, (2, 17))
    data = [(torch.rand(25, 25), torch.rand(2, 17)) for _ in range(8)]
    sm.train_by_datas(data, epochs=3, batch_size=4, verbose=False, early_stop=False)
    assert sm.scheduler.last_epoch == 3, f"scheduler 步數 {sm.scheduler.last_epoch}≠epochs(應 per-epoch)"
    assert sm.optimizer.param_groups[0]["lr"] > 1e-5, "lr 不應在 3 epoch 內撞地板"
