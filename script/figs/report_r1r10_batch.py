# -*- coding: utf-8 -*-
"""report_r1r10_batch.py — 成果報告圖 F4-F11（docs/report/assets/）：R6 期望邊界/分布、
R7 粉塵承重、analysis-01 結構歸因、R8 四臂、R9 家族/校正/s05。
資料:tmp/expected_best/ 與 tmp/pattern_anatomy/ 快取＋NAS dedust_* stores。"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from script.figs.report_r1r10_style import (  # noqa: E402
    REPO, INK, INK2, MUTED, GRID, SURF, RED, DBLUE, AQUA, ORANGE, GREEN, PURPLE,
    plt, style_ax, save, show_pattern)

from antenna.utils import config as _config, DATASET_PATH  # noqa: E402
_config.device = "cpu"
import torch  # noqa: E402

EB = os.path.join(REPO, "tmp", "expected_best")
PA = os.path.join(REPO, "tmp", "pattern_anatomy")


def _load_json(store, name="results.json"):
    return json.load(open(str(DATASET_PATH.joinpath(store, name)), encoding="utf-8"))


def _load_man(folder):
    return {m["id"]: m for m in _load_json(folder, "manifest.json")}


def _loadp(folder, pid):
    return np.asarray(torch.load(str(DATASET_PATH.joinpath(folder, f"{pid}.pt")),
                                 weights_only=True)).reshape(25, 25) > 0.5


# ==== F4 R6 期望邊界 ====
def _ebest_pool(margins, ns):
    """池經驗分布 best-of-N 期望（iid 有放回閉式解）。"""
    v = np.sort(margins)
    n = len(v)
    i = np.arange(1, n + 1)
    out = []
    for N in ns:
        w = (i / n) ** N - ((i - 1) / n) ** N
        out.append(float((v * w).sum()))
    return np.asarray(out)


def _mean_curve(npz, ks):
    """多 run best-so-far 曲線 → E[best@k]（k 超出長度取末值;僅平均涵蓋該 k 的 run）。"""
    d = np.load(npz, allow_pickle=True)
    curves = [np.asarray(d[k], float) for k in d.files]
    curves = [c[1] if c.ndim == 2 else c for c in curves]      # 容 (x,y) 或 y
    out = []
    for k in ks:
        vals = [c[min(k, len(c)) - 1] for c in curves if len(c) >= min(k, 30)]
        out.append(np.mean(vals))
    return np.asarray(out)


def fig4():
    h = np.load(os.path.join(EB, "harvest_margins.npy"))
    ks = np.unique(np.geomspace(1, 1000, 40).astype(int))
    ours = _mean_curve(os.path.join(EB, "ours_curves.npz"), ks)
    sen = _mean_curve(os.path.join(EB, "senior_curves.npz"), ks)
    pool = _ebest_pool(h, ks)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.0))
    ax = axes[0]
    ax.plot(ks, ours, color=DBLUE, lw=2.4, label="我們線上 GD（16 runs 平均）")
    ax.plot(ks, sen, color=GREEN, lw=2.4, label="學長線上（41 runs 平均）")
    ax.plot(ks, pool, color=MUTED, lw=2.0, ls="--", label="池抽樣 best-of-N（閉式解）")
    fit = -9.18 + 0.75 * np.log(ks)
    ax.plot(ks, fit, color=DBLUE, lw=1.0, ls=":", label="fit：−9.18 + 0.75·ln k")
    ax.axhline(0, color=RED, ls=":", lw=1.4)
    ax.text(1.25, 0.25, "spec 達標線", color=RED, fontsize=9)
    ax.set_xscale("log")
    style_ax(ax, "HFSS 模擬預算 k（log）", "E[best worst-margin @ k] (dB)",
             "期望爬升：對數慢爬,期望到不了 spec")
    ax.legend(fontsize=8.8, loc="lower right", framealpha=0.94).get_frame().set_edgecolor(GRID)

    ax = axes[1]
    v = np.sort(h)
    n = len(v)
    for T, c in ((0.0, RED), (-1.0, ORANGE), (-3.0, DBLUE)):
        FT = np.searchsorted(v, T) / n
        ax.plot(ks, 1 - FT ** ks, color=c, lw=2.2, label=f"池抽樣 P(達 {T:+.0f} dB)")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    style_ax(ax, "HFSS 模擬預算 k（log）", "P(best@k ≥ 門檻)",
             "達標機率：靠躍遷不靠爬,達 0 dB 要 ~450 筆池抽樣")
    ax.legend(fontsize=8.8, loc="upper left", framealpha=0.94).get_frame().set_edgecolor(GRID)
    fig.suptitle("R6 離線期望基準：三方法同一把尺（margin 全用現行 targets 重算）",
                 color=INK, fontsize=13)
    fig.tight_layout()
    save(fig, "f04_expected_best.png")


# ==== F5 分布≫策略 ====
def fig5():
    h = np.load(os.path.join(EB, "harvest_margins.npy"))
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    ax.hist(h, bins=90, color="#b9cbe3", edgecolor=SURF, lw=0.3, zorder=2)
    marks = [(-8.38, RED, "真 uniform random\nbest-of-10 = −8.38（R8-D）", 0.94, "right"),
             (np.median(h), MUTED, f"池中位 {np.median(h):.2f}", 0.60, "left"),
             (-2.89, DBLUE, "我們線上最佳\n−2.89（R4）", 0.94, "right"),
             (float(h.max()), GREEN, f"池最佳 {h.max():+.2f}\n（oracle）", 0.60, "right")]
    for x, c, lab, ty, ha in marks:
        ax.axvline(x, color=c, lw=1.8, ls="--", zorder=3)
        ax.text(x + (0.15 if ha == "left" else -0.15), ax.get_ylim()[1] * ty, lab,
                color=c, fontsize=8.8, va="top", ha=ha)
    style_ax(ax, "worst-margin (dB,現行 targets 重算)", "池樣本數",
             "分布 ≫ 策略：學長 24k 池 vs 均勻隨機差 ~5dB——「輸 random」輸的是分布,不是搜尋",
             tfs=12)
    ax.set_xlim(-26, 2.5)
    save(fig, "f05_pool_dist.png")


# ==== F6 R7 粉塵承重 ====
def fig6():
    res = _load_json("dedust_r7")
    W = lambda i: res[i]["wm"][2]
    fig = plt.figure(figsize=(11.6, 4.9))
    gs = fig.add_gridspec(1, 4, width_ratios=[2.1, 1, 1, 1], wspace=0.25)

    ax = fig.add_subplot(gs[0])
    ids = [f"p{k:02d}" for k in range(5)]
    dd = [W(i + "_d3") - W(i + "_orig") for i in ids]
    colors = [ORANGE if abs(v) > 0.5 else AQUA for v in dd]
    y = np.arange(5)[::-1]
    ax.barh(y, dd, height=0.6, color=colors, zorder=3)
    ax.axvline(0, color=INK2, lw=1)
    for yy, v, i in zip(y, dd, ids):
        ax.text(min(v, 0) - 0.4, yy, f"{i}  {v:+.2f}", color=INK2, fontsize=9, ha="right", va="center")
    ax.set_yticks([])
    ax.set_xlim(-19.5, 2.5)
    style_ax(ax, "Δ worst-margin：除塵後 − 原樣 (dB)", "",
             "除塵代價：4/5 崩 −5~−17,唯 p03 近零", tfs=11)

    p03o = _loadp("dedust_r7_input", "p03_orig")
    p03d = _loadp("dedust_r7_input", "p03_d3")
    p01o = _loadp("dedust_r7_input", "p01_orig")
    ax = fig.add_subplot(gs[1])
    show_pattern(ax, p01o, "p01 原樣（碎片雲）\n粉塵=共振一部分", tfs=9.2)
    ax = fig.add_subplot(gs[2])
    show_pattern(ax, p03o, "p03 原樣（整塊型）\nHFSS −2.66", tfs=9.2)
    ax = fig.add_subplot(gs[3])
    img = p03d.astype(int)
    img[p03o != p03d] = 2
    from matplotlib.colors import ListedColormap
    ax.imshow(img, cmap=ListedColormap([SURF, DBLUE, ORANGE]), vmin=0, vmax=2,
              origin="upper", interpolation="nearest")
    ax.scatter([12], [24], marker="^", s=46, color=AQUA, zorder=5, edgecolor=SURF, lw=0.8)
    ax.set_title("p03_d3 除塵 19px（橘）\n−2.68,rad +0.24 ✓", color=INK, fontsize=9.2)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)
    fig.suptitle("R7 粉塵承重：乾淨解「用搜的」不能「用修的」——整塊型 p03 是例外,成為可製造紀錄起點",
                 color=INK, fontsize=12.5)
    save(fig, "f06_dust_loadbearing.png")


# ==== F7 analysis-01 結構歸因 ====
def fig7():
    tr = np.load(os.path.join(PA, "trajs.npz"))
    d = np.load(os.path.join(PA, "pool.npz"))
    ok = ~np.isnan(d["wm"][:, 2])
    wm, feats = d["wm"][ok], d["feats"][ok]
    FEATURES = ("n_comp", "main_frac", "r_feed", "metal_frac", "sym_lr", "perim_ratio", "n_holes", "feed_touch")
    F = {k: feats[:, i] for i, k in enumerate(FEATURES)}

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.4))
    ax = axes[0]
    for dk, mk, c, lab in (("d_ours", "dm_ours", DBLUE, "我們（翻轉數）"),
                           ("d_sen", "dm_sen", GREEN, "學長（Hamming）")):
        dd, dm = np.asarray(tr[dk], float), np.asarray(tr[mk], float)
        bins = np.unique(np.geomspace(1, max(dd.max(), 2), 14).astype(int))
        xs, ys = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (dd >= lo) & (dd < hi)
            if m.sum() >= 8:
                xs.append(np.sqrt(lo * hi))
                ys.append(np.median(dm[m]))
        ax.plot(xs, ys, "o-", color=c, lw=2, ms=4.5, label=lab)
    ax.set_xscale("log")
    style_ax(ax, "pattern 距離（像素,log）", "中位 |Δ worst-margin| (dB)",
             "地形非抽獎：近距小擾動→小變化", tfs=10.5)
    ax.legend(fontsize=8.8, framealpha=0.94).get_frame().set_edgecolor(GRID)

    def paired(ax, hold, comp, feat, cg, lab_good, lab_bad, yl, ti):
        """配對對比:按 hold-margin 分十分位,各 bin 內取 comp-margin 前/後 25%,比 feat 中位。
        （對齊 analysis-01 的配對法——原始邊際有混淆,不能直接看。）"""
        qs = np.quantile(hold, np.linspace(0, 1, 11))
        xs, good, bad = [], [], []
        for lo, hi in zip(qs[:-1], qs[1:]):
            m = (hold >= lo) & (hold < hi)
            if m.sum() < 40:
                continue
            cv, fv = comp[m], feat[m]
            q1, q3 = np.quantile(cv, [0.75, 0.25])
            xs.append(np.median(hold[m]))
            good.append(np.median(fv[cv >= q1]))
            bad.append(np.median(fv[cv <= q3]))
        ax.plot(xs, bad, "o-", color=MUTED, lw=2, ms=4, label=lab_bad)
        ax.plot(xs, good, "o-", color=cg, lw=2.2, ms=4.5, label=lab_good)
        dmed = np.median(np.asarray(good) - np.asarray(bad))
        style_ax(ax, "", yl, ti.format(d=dmed), tfs=10.5)
        ax.legend(fontsize=8.6, framealpha=0.94).get_frame().set_edgecolor(GRID)
        return dmed

    ax = axes[1]
    paired(ax, wm[:, 1], wm[:, 0], F["n_comp"], DBLUE,
           "S11 佳（同水位前 25%）", "S11 差（後 25%）", "中位組件數 n_comp",
           "同 Gain 水位下:S11 好的組件數少（Δ中位 {d:+.1f}）")
    ax.set_xlabel("Gain margin 十分位（固定水位）", color=INK2, fontsize=10)

    ax = axes[2]
    paired(ax, wm[:, 0], wm[:, 1], F["n_holes"], ORANGE,
           "Gain 佳（同水位前 25%）", "Gain 差（後 25%）", "中位洞數 n_holes",
           "同 S11 水位下:Gain 好的洞少（Δ中位 {d:+.1f}）")
    ax.set_xlabel("S11 margin 十分位（固定水位）", color=INK2, fontsize=10)
    fig.suptitle("analysis-01 結構歸因（24k 池）：先驗要分工——S11 與 Gain 的敵人不同,共同敵人=細碎",
                 color=INK, fontsize=12.5)
    fig.tight_layout()
    save(fig, "f07_anatomy.png")


# ==== F8 R8 四臂 ====
def fig8():
    man = _load_man("dedust_r8_input")
    res = _load_json("dedust_r8")
    W = lambda i: res[i]["wm"][2]
    G = lambda i: res[i]["wm"][1]
    R = lambda i: res[i]["rad_margin"]

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 9.2))
    ax = axes[0][0]                                       # A 臂
    fams = [f"a{f:02d}" for f in range(15)]
    dd = sorted(W(f + "_d3") - W(f + "_orig") for f in fams)
    y = np.arange(15)
    ax.barh(y, dd, height=0.62, color=[AQUA if v > 0 else ORANGE for v in dd], zorder=3)
    ax.axvspan(-0.5, 0.5, color=GRID, alpha=0.5, zorder=1)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks([])
    style_ax(ax, "Δ worst-margin：除塵後 − 原樣 (dB)", "乾淨前緣 15 名（排序）",
             "A 臂敗：整塊型除塵 |Δ| 中位 1.17\n（R7 p03 的「近零代價」不是通則）", tfs=10.5)

    ax = axes[0][1]                                       # B 臂
    pairs = ["b00_holes", "b01_holes", "b02_holes", "b03_holes"]
    x = np.arange(4)
    for off, vals, c, lab in ((-0.27, [G(p) - G(man[p]["base_id"]) for p in pairs], DBLUE, "Δ Gain"),
                              (0.0, [R(p) - R(man[p]["base_id"]) for p in pairs], PURPLE, "Δ rad 餘裕"),
                              (0.27, [W(p) - W(man[p]["base_id"]) for p in pairs], MUTED, "Δ worst")):
        ax.bar(x + off, vals, width=0.24, color=c, label=lab, zorder=3)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p[:3]}\n(補{-man[p]['removed_px']}px)" for p in pairs], fontsize=8.6)
    ax.legend(fontsize=8.6, framealpha=0.94).get_frame().set_edgecolor(GRID)
    style_ax(ax, "", "Δ (dB)", "B 臂敗：補洞非因果——rad 四筆全負\n（「Gain←少洞」是相關不是因果）", tfs=10.5)

    ax = axes[1][0]                                       # C 臂
    groups = {"orig": (DBLUE, "o", "池內"), "d3": ("#2a78d6", "s", "池內編輯"),
              "probe": ("#5598e7", "^", "池內鄰域"), "holes": ("#86b6ef", "D", "補洞"),
              "blob": (ORANGE, "o", "池外 blob"), "rand": (RED, "s", "池外 random")}
    lim = (-26, 3)
    ax.plot(lim, lim, color=INK2, ls="--", lw=1.2)
    for kind, (c, mk, lab) in groups.items():
        ids = [i for i in res if man[i]["kind"] == kind and man[i].get("sm_wm")]
        ax.scatter([W(i) for i in ids], [man[i]["sm_wm"][2] for i in ids], s=26, color=c,
                   marker=mk, alpha=0.85, label=lab, edgecolor=SURF, lw=0.4, zorder=3)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.94).get_frame().set_edgecolor(GRID)
    style_ax(ax, "HFSS 真值 (dB)", "SM 預測 (dB)",
             "C 臂半亮：SM 池內誤差 1.5-2.4dB、池外 4-5.5\n（重錨前;「什麼時候能信 SM」有了地圖）", tfs=10.5)

    ax = axes[1][1]                                       # D 臂
    dw = sorted(W(i) for i in res if i.startswith("d"))
    ax.scatter(dw, np.zeros(len(dw)), s=52, color=RED, zorder=4, label="真 uniform random 10 筆")
    for v, c, lab, ha in ((-3.47, DBLUE, "池抽樣 E[best-of-10]\n−3.47", "left"),
                          (dw[-1], RED, f"uniform best-of-10\n{dw[-1]:.2f}", "right")):
        ax.axvline(v, color=c, ls="--", lw=1.5, zorder=2)
        ax.text(v + (0.15 if ha == "left" else -0.15), 0.92, lab, color=c, fontsize=8.6,
                va="top", ha=ha)
    ax.set_ylim(-0.5, 1.05)
    ax.set_yticks([])
    ax.set_xlim(-20, -2)
    style_ax(ax, "worst-margin (dB,越右越好)", "",
             "D 臂實錘：uniform 輸池抽樣 ~5dB\n（分布≫策略的直接證據）", tfs=10.5)
    fig.suptitle("R8 乾淨子空間測繪（97 筆,四臂對照）：兩敗一半亮一實錘——批次即知識",
                 color=INK, fontsize=13)
    fig.tight_layout()
    save(fig, "f08_r8_arms.png")


# ==== F9 R9 家族普查 ====
def fig9():
    d = np.load(os.path.join(PA, "pool.npz"))
    ok = ~np.isnan(d["wm"][:, 2])
    worst = d["wm"][ok][:, 2]
    pats = np.unpackbits(d["packed"][ok], axis=1)[:, :625].reshape(-1, 25, 25).astype(bool)
    order = np.argsort(worst)[::-1][:300]
    leaders, members = [], {}
    for i in order:
        for L in leaders:
            if np.count_nonzero(pats[i] != pats[L]) <= 100:
                members[L].append(i)
                break
        else:
            leaders.append(i)
            members[i] = [i]
    # 家族編號＝leader 出現順序（wm 降冪）,與 R9 探索錨點的 F0-F5 命名一致
    fig = plt.figure(figsize=(11.8, 5.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=0.18)
    ax = fig.add_subplot(gs[0])
    show_n = 12
    x = np.arange(show_n)
    ax.bar(x, [len(members[L]) for L in leaders[:show_n]], color=DBLUE, width=0.62, zorder=3)
    for xx, L in zip(x, leaders[:show_n]):
        ax.text(xx, len(members[L]) + 1.2, f"{worst[L]:+.2f}", ha="center",
                fontsize=8.2, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"F{k}" for k in range(show_n)], fontsize=9)
    style_ax(ax, f"家族（top-300 greedy 聚類,共 {len(leaders)} 族;編號=leader wm 降冪;柱上=leader wm）",
             "族成員數", "池頂端不是一種長相：29 個家族", tfs=11.5)

    sub = gs[1].subgridspec(2, 3, hspace=0.3, wspace=0.08)
    for k in range(6):
        ax = fig.add_subplot(sub[k // 3, k % 3])
        L = leaders[k]
        show_pattern(ax, pats[L], f"F{k}（{worst[L]:+.2f}）", tfs=8.6)
    fig.suptitle("R9 跨家族普查：探索錨點從「單一家族」擴到全池頂端（右=前六族代表,含粉塵）",
                 color=INK, fontsize=12.5)
    save(fig, "f09_families.png")


# ==== F10 R9 校正散點 ====
def fig10():
    man = _load_man("dedust_r9_input")
    res = _load_json("dedust_r9")
    r8m = _load_man("dedust_r8_input")
    r8 = _load_json("dedust_r8")
    ok = {i for i in res if "wm" in res[i]}
    W = lambda i: res[i]["wm"][2]
    fig, ax = plt.subplots(figsize=(7.8, 6.6))
    lim = (-3.6, 1.2)
    ax.plot(lim, lim, ls="--", color=INK2, lw=1.3)
    ax.axhline(0, color=RED, ls=":", lw=1.1)
    ax.axvline(0, color=RED, ls=":", lw=1.1)
    pv, cv = [], []
    for pre, c, mk, lab in (("t", DBLUE, "o", "T 帳面達標"), ("n", "#2a78d6", "s", "N 近標 [−1,0)"),
                            ("m", "#86b6ef", "^", "M 深帶 [−3,−1)")):
        ids = [i for i in ok if i.startswith(pre) and i[1].isdigit()]
        x = [man[i]["pool_wm"][2] for i in ids]
        y = [W(i) for i in ids]
        pv += x
        cv += y
        ax.scatter(x, y, s=40, color=c, marker=mk, label=lab, zorder=4, edgecolor=SURF, lw=0.5)
    a8 = [(r8m[f"a{k:02d}_orig"]["pool_wm"][2], r8[f"a{k:02d}_orig"]["wm"][2]) for k in range(15)]
    ax.scatter([a for a, _ in a8], [b for _, b in a8], s=34, color=MUTED, marker="x",
               label="R8 A 臂（另一家族,漂移偏下）", zorder=3)
    (b0, b1), *_ = np.linalg.lstsq(np.vstack([np.ones(len(pv)), pv]).T, cv, rcond=None)
    xs = np.linspace(*lim, 10)
    ax.plot(xs, b0 + b1 * xs, color=DBLUE, lw=1.6,
            label=f"fit：現行 = {b0:+.2f} + {b1:.2f}·池值（σ 0.77）")
    ax.set_xlim(lim)
    ax.set_ylim(-4.6, 1.2)
    style_ax(ax, "池記錄值 worst-margin (dB)", "現行 HFSS worst-margin (dB)",
             "歷史資料的校正：漂移家族依賴——頂帶 ±0.4 可信,可折價使用", tfs=12)
    ax.legend(fontsize=8.6, loc="lower right", framealpha=0.94).get_frame().set_edgecolor(GRID)
    save(fig, "f10_calibration.png")


# ==== F11 s05 對稱化首勝 ====
def fig11():
    pool = np.load(os.path.join(PA, "pool.npz"))
    okp = ~np.isnan(pool["wm"][:, 2])
    pats_pool = np.unpackbits(pool["packed"][okp], axis=1)[:, :625].reshape(-1, 25, 25).astype(bool)
    r9man = _load_man("dedust_r9_input")
    f2_idx = next(m["anchor_pool_idx"] for m in r9man.values()
                  if m.get("anchor") == "F2" and m.get("flip_k") == 0)
    f2 = pats_pool[f2_idx]
    s05 = _loadp("dedust_r9_input", "s05_1050")
    from antenna.utils.store import SampleStore
    store = SampleStore(DATASET_PATH.joinpath("dedust_r9"), verbose=False)
    resp = None
    for i in range(len(store)):
        x, y = store[i]
        if ((np.asarray(x).reshape(25, 25) > 0.5) == s05).all():
            resp = np.asarray(y).reshape(2, -1)
            break
    freq = 26.5 + (np.arange(resp.shape[1]) - 5) * 0.5

    fig = plt.figure(figsize=(11.8, 4.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1.5, 1.5], wspace=0.3)
    ax = fig.add_subplot(gs[0])
    show_pattern(ax, f2, "F2 錨點（含粉塵）\n池 −0.01,不可製造", tfs=9.4)
    ax = fig.add_subplot(gs[1])
    show_pattern(ax, s05, "s05 ＝ F2 的 10-5-10 對稱化\n零粉塵、全件 ≥4px ✓", tfs=9.4)
    for k, (idx, spec_y, lab, low) in enumerate(((0, -10, "S11 (dB)", True), (1, 4, "Gain (dB)", False))):
        ax = fig.add_subplot(gs[2 + k])
        ax.axvspan(26.5, 29.5, color=GRID, alpha=0.45)
        ax.axhline(spec_y, color=RED, ls=":", lw=1.3)
        ax.plot(freq, resp[idx], color=PURPLE, lw=2.3)
        band = resp[idx][5:12]
        m = (spec_y - band.max()) if low else (band.min() - spec_y)
        style_ax(ax, "頻率 (GHz)", lab,
                 f"{lab.split(' ')[0]} margin {m:+.2f}" + ("　✓" if m > 0 else ""), tfs=10.5)
    fig.suptitle("R9 對稱化首勝：s05 wm −0.29（可製造紀錄 −2.68 → −0.29）——「乾淨解用構造的」首次兌現",
                 color=INK, fontsize=12.5)
    save(fig, "f11_s05_story.png")


if __name__ == "__main__":
    fig4()
    fig5()
    fig6()
    fig7()
    fig8()
    fig9()
    fig10()
    fig11()
