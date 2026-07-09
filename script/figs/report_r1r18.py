# -*- coding: utf-8 -*-
"""report_r1r18.py — 總進度報告圖 E1-E6（docs/report/assets/,progress-r1-r18 用）。
E1 每 round 最佳 gallery（R7-R18,12 格）   E2 血統演進鏈（F2→…→a024,橘=相對上一代差異）
E3 可製造紀錄時間軸（R7→R17,+假象攔截標記） E4 帶外 Pareto（全歷史真值,地板 9.0 視覺化）
E5 分組/尺寸答案（等金屬三態+尺寸階梯）      E6 新王 a024 vs 前王 i02（pattern+三標曲線）
資料:NAS dedust stores（E4 快取 tmp/report_r1r18_oob.npz）。全部決定性可重跑。"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from script.figs.report_r1r10_style import (  # noqa: E402
    REPO, INK, INK2, MUTED, GRID, SURF, RED, DBLUE, AQUA, ORANGE, GREEN, GOLD,
    plt, style_ax, save)
from matplotlib.colors import ListedColormap  # noqa: E402

from antenna.utils import config as _config, DATASET_PATH  # noqa: E402
_config.device = "cpu"
import torch  # noqa: E402
from antenna.utils.store import SampleStore  # noqa: E402
from script.dedust import oob_metrics  # noqa: E402

FEED = (24, 12)
_CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])


def loadp(folder, pid):
    return np.asarray(torch.load(str(DATASET_PATH.joinpath(folder, f"{pid}.pt")),
                                 weights_only=True)).reshape(25, 25) > 0.5


def cell(ax, p, base=None, tcolor=INK):
    """一格 pattern:藍=金屬,橘=相對 base 的差異像素。"""
    img = p.astype(int)
    if base is not None:
        img[p != base] = 2
    ax.imshow(img, cmap=ListedColormap([SURF, DBLUE, ORANGE]), vmin=0, vmax=2,
              origin="upper", interpolation="nearest")
    ax.scatter([FEED[1]], [FEED[0]], marker="^", s=30, color=AQUA, zorder=5,
               edgecolor=SURF, lw=0.7)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)


# ==== E1 每 round 最佳 gallery ====
CELLS = [  # (round, folder, pid, 標題行2, 標題行3)
    ("R7",  "dedust_r7_input",   "p03_d3",           "p03_d3  −2.68 / rad +0.24", "除塵例外＝可製造紀錄起點"),
    ("R8",  "dedust_r8_input",   "c05_probe",        "c05_probe  −2.58 / +0.02",  "測繪輪（規則產出,紀錄未動）"),
    ("R9",  "dedust_r9_input",   "s05_1050",         "s05  −0.29（rad 未過）",     "10-5-10 對稱化首勝 +2.39"),
    ("R10", "dedust_ref2_input", "c21_sm",           "c21_sm  +0.20 / +0.12",     "八冠軍 certified,首批三標全過"),
    ("R11", "dedust_ref3_input", "c25_a15w10_2_22",  "c25  +0.22 / +0.34",        "組數階梯（a15＋翼對,5 塊）"),
    ("R12", "dedust_wide_input", "x00_c21k2",        "x00  +0.19 / +0.19",        "bakeoff 製造首選（缺陷存活 72%）"),
    ("R13", "dedust_blocks_input", "g39_a15_4b",     "g39_a15_4b  +0.14 / +0.29", "4 塊三標過（4-5 塊甜蜜點）"),
    ("R14", "dedust_ref2_input", "c21_sm",           "最佳＝對照組 c21（+0.20）",   "機理輪:翼＝引擎 6dB／尖銳最優"),
    ("R15", "dedust_r15inf_input", "i02_r15",        "i02  +0.29 / +0.21（公證）", "範式對比;換王 i02"),
    ("R16", "dedust_addmap_input", "a024_c25r9c11s3", "a024  +0.35 / +0.12",      "收益圖唯一正點＝現任紀錄"),
    ("R17", "dedust_r17_input",  "cc_x00_r5s2_r8s3", "cc_x00…  +0.21 / oob 9.82", "帶外主目標;a024 公證換王"),
    ("R18", "dedust_wide_input", "x20_a00k8",        "x20  +0.08 / oob 9.15",     "挖礦公證;帶外地板 9.0 定案"),
]


def fig_e1():
    fig, axes = plt.subplots(3, 4, figsize=(12.4, 10.4))
    for ax, (rd, fol, pid, l2, l3) in zip(axes.ravel(), CELLS):
        cell(ax, loadp(fol, pid))
        ax.set_title(f"{rd}｜{l2}\n{l3}", color=INK, fontsize=9.2, pad=5)
    fig.suptitle("每 round 最佳（可製造/三標優先;wm / rad,dB）——R7 除塵例外到 R18 帶外定案",
                 color=INK, fontsize=13.5)
    fig.tight_layout(rect=[0, 0, 1, 0.965], h_pad=2.2)
    save(fig, "e01_round_best_gallery.png")


# ==== E2 血統演進鏈 ====
def fig_e2():
    r9man = {m["id"]: m for m in json.load(open(str(DATASET_PATH.joinpath(
        "dedust_r9_input", "manifest.json")), encoding="utf-8"))}
    pool = np.load(os.path.join(REPO, "tmp", "pattern_anatomy", "pool.npz"))
    okp = ~np.isnan(pool["wm"][:, 2])
    pats_pool = np.unpackbits(pool["packed"][okp], axis=1)[:, :625].reshape(-1, 25, 25).astype(bool)
    f2 = pats_pool[next(m["anchor_pool_idx"] for m in r9man.values()
                        if m.get("anchor") == "F2" and m.get("flip_k") == 0)]
    chain = [
        ("① F2 錨點（學長池）", "池 −0.01·含粉塵不可製造", f2, None),
        ("② s05 ＝10-5-10 對稱化", "−0.29（R9,可製造紀錄）", loadp("dedust_r9_input", "s05_1050"), f2),
        ("③ w17 ＝翻8px＋再對稱化", "−0.06（R10,公證後）", loadp("dedust_ref1_input", "w17_k8"), None),
        ("④ a15 ＝三臂精修", "+0.03·rad +0.56（R10）", loadp("dedust_ref2_input", "a15_k4"), None),
        ("⑤ c25 ＝＋翼對（組數階梯）", "+0.22·rad +0.34（R11）", loadp("dedust_ref3_input", "c25_a15w10_2_22"), None),
        ("⑥ a024 ＝＋中央 3×3", "+0.35·rad +0.12（R17 公證）", loadp("dedust_addmap_input", "a024_c25r9c11s3"), None),
    ]
    for i in range(2, len(chain)):
        chain[i] = (chain[i][0], chain[i][1], chain[i][2], chain[i - 1][2])
    fig, axes = plt.subplots(1, 6, figsize=(15.2, 3.6))
    for ax, (t1, t2, p, base) in zip(axes, chain):
        cell(ax, p, base)
        ax.set_title(f"{t1}\n{t2}", color=INK, fontsize=8.8, pad=4)
    fig.suptitle("血統演進鏈：碎片雲 → 對稱化 → 精修 → 組件級加塊（橘＝相對上一代的差異像素;"
                 "i02 +0.29 為 c25 另一子代）", color=INK, fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, "e02_lineage_chain.png")


# ==== E3 紀錄時間軸 ====
MILE = [  # (累計批次 HFSS, 紀錄 wm, 標籤, 附註)
    (15,   -2.68, "R7｜p03_d3",  "除塵例外"),
    (274,  -0.29, "R9｜s05",     "對稱化 +2.39"),
    (451,   0.20, "R10｜c21_sm", "八冠軍·首過 spec"),
    (932,   0.22, "R11｜c25",    "組數階梯"),
    (1288,  0.29, "R15｜i02",    "組件空間+中央塊"),
    (1408,  0.35, "R17｜a024",   "收益圖唯一正點\n（R16 出土,R17 公證）"),
]


def fig_e3():
    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    xs = [0] + [m[0] for m in MILE] + [1442]
    ys = [-2.68] + [m[1] for m in MILE] + [0.35]
    ax.step(xs, ys, where="post", color=DBLUE, lw=2.6, zorder=3)
    ax.axhline(0, color=RED, ls=":", lw=1.5)
    ax.text(20, 0.05, "spec 達標線", color=RED, fontsize=9.5, va="bottom")
    off = [(30, -0.5), (-200, -0.62), (25, -0.85), (25, -0.9), (-95, -1.05), (-55, 0.45)]
    for (x, y, lab, note), (dx, dy) in zip(MILE, off):
        ax.scatter([x], [y], s=60, color=ORANGE if y >= 0 else DBLUE, zorder=4,
                   edgecolor=SURF, lw=1.1)
        ax.annotate(f"{lab}  {y:+.2f}\n{note}", (x, y), (x + dx, y + dy), fontsize=8.8,
                    color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.plot([0, 1442], [-2.89, -2.89], color=MUTED, ls="--", lw=1.3)
    ax.text(1420, -2.83, "線上學習線最佳（R4 E+D −2.89,含粉塵不可製造）",
            color=MUTED, fontsize=8.8, ha="right", va="bottom")
    for x, y, txt, dx, dy in ((451, 0.48, "w17 +0.48", 30, 0.35), (1442, 0.32, "b20 +0.32", -250, -0.78)):
        ax.scatter([x], [y], s=52, marker="x", color=RED, zorder=4, lw=2)
        ax.annotate(f"✗ {txt} 假象（公證攔截）", (x, y), (x + dx, y + dy),
                    fontsize=8.4, color=RED,
                    arrowprops=dict(arrowstyle="-", color=RED, lw=0.6, alpha=0.5))
    ax.set_xlim(-20, 1500)
    ax.set_ylim(-3.6, 1.25)
    style_ax(ax, "累計批次 HFSS 模擬筆數（R7 起,共 1442）", "可製造紀錄 worst-margin (dB)",
             "可製造紀錄推進 −2.68 → +0.35（R7→R17）與兩次假象攔截", tfs=12.5)
    save(fig, "e03_record_timeline.png")


# ==== E4 帶外 Pareto ====
def _scan_oob():
    cache = os.path.join(REPO, "tmp", "report_r1r18_oob.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return d["wm"], d["oob"], d["tri"]
    from scipy.ndimage import label as _label
    stores = [d for d in os.listdir(str(DATASET_PATH))
              if d.startswith("dedust_") and not d.endswith("_input") and not d.endswith("_src")]
    wms, oobs, tris, seen = [], [], [], set()
    for st in stores:
        inp = st + "_input"
        if not DATASET_PATH.joinpath(st, "results.json").exists() or not DATASET_PATH.joinpath(inp).is_dir():
            continue
        res = json.load(open(str(DATASET_PATH.joinpath(st, "results.json")), encoding="utf-8"))
        smap = {}
        s = SampleStore(DATASET_PATH.joinpath(st), verbose=False)
        for k in range(len(s)):
            x, y = s[k]
            smap[(np.asarray(x).reshape(-1) > 0.5).tobytes()] = np.asarray(y).reshape(2, -1)
        for i, r in res.items():
            if "wm" not in r:
                continue
            f = DATASET_PATH.joinpath(inp, i + ".pt")
            if not f.exists():
                continue
            p = np.asarray(torch.load(str(f), weights_only=True)).reshape(25, 25) > 0.5
            key = p.reshape(-1).tobytes()
            if key in seen:
                continue
            resp = smap.get(key)
            if resp is None or resp.shape[1] < 17:
                continue
            seen.add(key)
            lab, n = _label(p, structure=_CROSS)
            sizes = np.bincount(lab.ravel())[1:]
            manuf = n > 0 and bool((sizes >= 4).all())
            rad = r.get("rad_margin")
            wms.append(r["wm"][2])
            oobs.append(oob_metrics(resp)["oob_bad"])
            tris.append(bool(manuf and r["wm"][2] >= 0 and rad is not None and rad >= 0))
    wm, oob, tri = np.array(wms), np.array(oobs), np.array(tris)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, wm=wm, oob=oob, tri=tri)
    return wm, oob, tri


def fig_e4():
    wm, oob, tri = _scan_oob()
    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    ax.scatter(wm[~tri], oob[~tri], s=12, color=MUTED, alpha=0.35, lw=0, label="全歷史（非三標）")
    ax.scatter(wm[tri], oob[tri], s=26, color=DBLUE, alpha=0.85, lw=0, label="三標過（可製造+rad）")
    ax.axhline(9.04, color=RED, ls=":", lw=1.5)
    ax.text(-16, 9.25, "三標內帶外地板 9.04（c18_sm）", color=RED, fontsize=9, va="bottom")
    ax.axvline(0, color=GREEN, ls=":", lw=1.3)
    for x, y, lab, dx, dy in ((0.07, 9.04, "c18_sm 9.04", -3.2, -1.1), (0.08, 9.15, "x20 9.15", 0.5, -0.75),
                              (0.35, 10.72, "a024（margin 王）", 0.4, 0.9),
                              (-24.57, 6.84, "救援臂極值 6.84（wm −24.6=帶內全滅）", 1.0, 0.9),
                              (-4.11, 8.81, "vslot_c18 8.81（非三標）", -5.2, -1.7)):
        ax.scatter([x], [y], s=52, color=GOLD, zorder=4, edgecolor=INK, lw=0.7)
        ax.annotate(lab, (x, y), (x + dx, y + dy), fontsize=8.6, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_xlim(-26.5, 2.2)
    ax.set_ylim(5.5, 24)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.94).get_frame().set_edgecolor(GRID)
    style_ax(ax, "帶內 worst-margin (dB,右=好)", "帶外惡度 oob_bad (dB,低=好)",
             f"帶外 × 帶內全歷史地形（{len(wm)} 互異真值）——右下角（三標＋帶外<9）為空＝地板", tfs=12)
    save(fig, "e04_oob_pareto.png")


# ==== E5 分組/尺寸答案 ====
def fig_e5():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.4))
    grp = {"c25": (0.24, 0.17, -2.59), "x00": (-0.01, 0.03, -2.47)}
    xs = np.arange(3)
    for i, (aid, v) in enumerate(grp.items()):
        b = a1.bar(xs + i * 0.38, v, width=0.34, color=(DBLUE, ORANGE)[i], label=aid)
        for r, val in zip(b, v):
            a1.text(r.get_x() + r.get_width() / 2, val + (0.09 if val >= 0 else -0.32),
                    f"{val:+.2f}", ha="center", fontsize=8.8, color=INK2)
    a1.axhline(0, color=MUTED, lw=1)
    a1.set_xticks(xs + 0.19)
    a1.set_xticklabels(["1 塊（4×3）", "2 塊（2 個 2×3）", "3 塊（3 個 2×2）"], fontsize=9.5)
    a1.set_ylim(-3.3, 0.9)
    a1.legend(fontsize=9, framealpha=0.94).get_frame().set_edgecolor(GRID)
    style_ax(a1, "等金屬 12px 的分組方式", "wm (dB)", "分組答案：集中 ≫ 分散（R17-S）", tfs=11.5)
    lad = {"c25": (-0.02, 0.35, 0.24, -0.07), "x00": (0.04, 0.02, -0.01, -0.21)}
    xs = np.arange(4)
    for i, (aid, v) in enumerate(lad.items()):
        b = a2.bar(xs + i * 0.38, v, width=0.34, color=(DBLUE, ORANGE)[i], label=aid)
        for r, val in zip(b, v):
            a2.text(r.get_x() + r.get_width() / 2, val + (0.02 if val >= 0 else -0.06),
                    f"{val:+.2f}", ha="center", fontsize=8.8, color=INK2)
    a2.axhline(0, color=MUTED, lw=1)
    a2.set_xticks(xs + 0.19)
    a2.set_xticklabels(["2×2", "3×3", "4×3", "5×3"], fontsize=10)
    a2.set_ylim(-0.42, 0.52)
    a2.legend(fontsize=9, framealpha=0.94).get_frame().set_edgecolor(GRID)
    style_ax(a2, "中央塊尺寸（col11,下緣對齊）", "wm (dB)", "尺寸階梯：收益峰＝3×3（R16+R17）", tfs=11.5)
    fig.tight_layout()
    save(fig, "e05_grouping_answer.png")


# ==== E6 新王 a024 vs 前王 i02 ====
def _find_resp(store, pat):
    s = SampleStore(DATASET_PATH.joinpath(store), verbose=False)
    for i in range(len(s)):
        x, y = s[i]
        if ((np.asarray(x).reshape(25, 25) > 0.5) == pat).all():
            return np.asarray(y).reshape(2, -1)
    raise SystemExit(f"{store} 無響應")


def fig_e6():
    c25 = loadp("dedust_ref3_input", "c25_a15w10_2_22")
    a024 = loadp("dedust_addmap_input", "a024_c25r9c11s3")
    i02 = loadp("dedust_r15inf_input", "i02_r15")
    ra, ri = _find_resp("dedust_addmap", a024), _find_resp("dedust_r15inf", i02)
    rada = torch.load(str(DATASET_PATH.joinpath("dedust_addmap", "rad", "a024_c25r9c11s3.pt")), weights_only=True)
    radi = torch.load(str(DATASET_PATH.joinpath("dedust_r15inf", "rad", "i02_r15.pt")), weights_only=True)
    freq = 26.5 + (np.arange(ra.shape[1]) - 5) * 0.5
    fig = plt.figure(figsize=(13.2, 6.4))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1.35, 1.35], hspace=0.42, wspace=0.34)
    axp = fig.add_subplot(gs[:, 0])
    cell(axp, a024, c25)
    axp.set_title("新王 a024\nwm +0.35 · rad +0.12 · 公證3/3\n（橘＝vs c25 的中央 3×3）",
                  color=INK, fontsize=9.4)
    axc = fig.add_subplot(gs[:, 1])
    cell(axc, i02, c25)
    axc.set_title("前王 i02\nwm +0.29 · rad +0.21\n（c25＋中央 8px,兄弟解）", color=INK2, fontsize=9.4)
    for (gr, idx, spec, nm, low) in ((gs[0, 2], 0, -10, "S11", True), (gs[0, 3], 1, 4, "Gain", False)):
        ax = fig.add_subplot(gr)
        ax.axvspan(26.5, 29.5, color=GRID, alpha=0.45)
        ax.axhline(spec, color=RED, ls=":", lw=1.3)
        ax.plot(freq, ri[idx], color="#9fb4d4", lw=1.8, label="i02（前王）")
        ax.plot(freq, ra[idx], color=DBLUE, lw=2.3, label="a024（新王）")
        band = ra[idx][5:12]
        m = (spec - band.max()) if low else (band.min() - spec)
        style_ax(ax, "頻率 (GHz)", f"{nm} (dB)", f"{nm}（a024 margin {m:+.2f}）", tfs=10)
        if idx == 0:
            ax.legend(fontsize=8.6, loc="lower left", framealpha=0.94).get_frame().set_edgecolor(GRID)
    for (gr, cut) in ((gs[1, 2], "phi0"), (gs[1, 3], "phi90")):
        ax = fig.add_subplot(gr)
        ax.axvspan(-45, 45, color=GRID, alpha=0.45)
        ax.plot(np.asarray(radi["theta"]), np.asarray(radi[cut]), color="#9fb4d4", lw=1.8)
        ax.plot(np.asarray(rada["theta"]), np.asarray(rada[cut]), color=DBLUE, lw=2.3)
        ax.set_xlim(-90, 90); ax.set_xticks([-90, -45, 0, 45, 90])
        style_ax(ax, "θ (deg)", "Gain (dB)", f"Radiation {cut} — 灰帶＝±45° 窗", tfs=10.5)
    fig.suptitle("現任 margin 王 a024（R16 收益圖唯一正點,R17 公證 3/3）vs 前王 i02——兄弟解,同為 c25＋中央塊",
                 color=INK, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save(fig, "e06_newking_a024.png")


if __name__ == "__main__":
    which = sys.argv[1:] or ["e1", "e2", "e3", "e4", "e5", "e6"]
    for w in which:
        {"e1": fig_e1, "e2": fig_e2, "e3": fig_e3, "e4": fig_e4, "e5": fig_e5, "e6": fig_e6}[w]()
