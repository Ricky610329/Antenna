# -*- coding: utf-8 -*-
"""report_r1r10_champs.py — 成果報告圖 F12-F16（docs/report/assets/）：R10 血統/承重圖/
八冠軍 gallery/前三名曲線/+0.48 假象案例。資料:NAS dedust_* stores。"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from script.figs.report_r1r10_style import (  # noqa: E402
    REPO, INK, INK2, MUTED, GRID, SURF, RED, DBLUE, AQUA, ORANGE, GREEN,
    plt, style_ax, save)
from matplotlib.colors import ListedColormap  # noqa: E402

from antenna.utils import config as _config, DATASET_PATH  # noqa: E402
_config.device = "cpu"
import torch  # noqa: E402
from antenna.utils.store import SampleStore  # noqa: E402

FEED = (24, 12)
CHAMPS = ["c21_sm", "a00_k2", "b11_k2", "c10_sm", "c18_sm", "c17_sm", "a15_k4", "a11_k2"]


def _load_json(store, name="results.json"):
    return json.load(open(str(DATASET_PATH.joinpath(store, name)), encoding="utf-8"))


def _loadp(folder, pid):
    return np.asarray(torch.load(str(DATASET_PATH.joinpath(folder, f"{pid}.pt")),
                                 weights_only=True)).reshape(25, 25) > 0.5


def _find_resp(store_name, pat):
    store = SampleStore(DATASET_PATH.joinpath(store_name), verbose=False)
    for i in range(len(store)):
        x, y = store[i]
        if ((np.asarray(x).reshape(25, 25) > 0.5) == pat).all():
            return np.asarray(y).reshape(2, -1)
    raise SystemExit(f"{store_name} 找不到響應")


def _diff_show(ax, p, ref, title, tfs=9.4):
    """pattern；有 ref 時：綠＝加銅（p 有 ref 無）、紅＝去銅（p 無 ref 有）、藍＝共同金屬。"""
    img = p.astype(int)
    if ref is not None:
        img[p & ~ref] = 2          # 加銅
        img[(~p) & ref] = 3         # 去銅
        cmap = ListedColormap([SURF, DBLUE, GREEN, RED]); vmax = 3
    else:
        cmap = ListedColormap([SURF, DBLUE]); vmax = 1
    ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax,
              origin="upper", interpolation="nearest")
    ax.scatter([FEED[1]], [FEED[0]], marker="^", s=46, color=AQUA, zorder=5,
               edgecolor=SURF, lw=0.8)
    ax.set_title(title, color=INK, fontsize=tfs)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)


# ==== F12 血統鏈 ====
def fig12():
    pool = np.load(os.path.join(REPO, "tmp", "pattern_anatomy", "pool.npz"))
    okp = ~np.isnan(pool["wm"][:, 2])
    pats_pool = np.unpackbits(pool["packed"][okp], axis=1)[:, :625].reshape(-1, 25, 25).astype(bool)
    r9man = {m["id"]: m for m in _load_json("dedust_r9_input", "manifest.json")}
    f2 = pats_pool[next(m["anchor_pool_idx"] for m in r9man.values()
                        if m.get("anchor") == "F2" and m.get("flip_k") == 0)]
    s05 = _loadp("dedust_r9_input", "s05_1050")
    w17 = _loadp("dedust_ref1_input", "w17_k8")
    c21 = _loadp("dedust_ref2_input", "c21_sm")
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.9))
    _diff_show(axes[0], f2, None, "① F2 錨點（學長池,含粉塵）\n池 −0.01·不可製造")
    _diff_show(axes[1], s05, None, "② s05＝① 10-5-10 對稱化\n−0.29（R9,可製造紀錄）")
    _diff_show(axes[2], w17, s05, "③ w17＝② 翻8px＋再對稱化\n−0.06（公證 8/8 一致）")
    _diff_show(axes[3], c21, w17, "④ c21＝③ SM 導引翻 32px\n+0.20（三標全過,certified）")
    fig.suptitle("w17→冠軍血統：三步構造式編輯,從碎片雲到達標（綠＝加銅／紅＝去銅,相對上一代;每步算子+seed 可重現）",
                 color=INK, fontsize=12.2)
    fig.tight_layout()
    save(fig, "f12_lineage.png")


# ==== F13 承重圖 ====
def fig13():
    man = {m["id"]: m for m in _load_json("dedust_occl_input", "manifest.json")}
    occ = _load_json("dedust_occl")
    r9 = _load_json("dedust_r9")
    bases = {"s05_1050": ("s05（wm −0.29）", _loadp("dedust_r9_input", "s05_1050")),
             "g24_sm": ("g24（wm −1.85,rad 王族）", _loadp("dedust_r9_input", "g24_sm"))}
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.0))
    for ax, (src, (lab, pat)) in zip(axes, bases.items()):
        base = r9[src]["wm"][2]
        heat = np.full((5, 5), np.nan)
        for i, m in man.items():
            if m["source_id"] == src and i in occ and "wm" in occ[i]:
                br, bc = m["block"]
                heat[br, bc] = occ[i]["wm"][2] - base
        vmax = np.nanmax(np.abs(heat))
        img = np.kron(heat, np.ones((5, 5)))
        ax.imshow(pat.astype(int), cmap=ListedColormap([SURF, "#c9d6e8"]), vmin=0, vmax=1,
                  origin="upper", interpolation="nearest")
        h = ax.imshow(img, cmap="RdBu", vmin=-vmax, vmax=vmax, alpha=0.72,
                      origin="upper", interpolation="nearest")
        for k in range(6):
            ax.axhline(k * 5 - 0.5, color=SURF, lw=1.2)
            ax.axvline(k * 5 - 0.5, color=SURF, lw=1.2)
        ax.scatter([FEED[1]], [FEED[0]], marker="^", s=52, color=AQUA, zorder=5,
                   edgecolor=SURF, lw=0.8)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.set_title(lab, color=INK, fontsize=11)
        cb = fig.colorbar(h, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("Δ worst-margin (dB)", color=INK2, fontsize=9)
        cb.ax.tick_params(labelsize=8, colors=INK2)
    fig.suptitle("空間承重圖（5×5 遮蔽掃描,紅=遮了就崩=承重區、近白=低成本區）——\n知情編輯的地圖來源（R10 X 臂據此在低成本區編輯,冠軍 b11 兌現）",
                 color=INK, fontsize=12)
    fig.tight_layout()
    save(fig, "f13_occlusion.png")


# ==== F14 八冠軍 gallery ====
def fig14():
    man = {m["id"]: m for m in _load_json("dedust_ref2_input", "manifest.json")}
    rv = _load_json("dedust_ref2v")
    w17 = _loadp("dedust_ref1_input", "w17_k8")
    fig, axes = plt.subplots(2, 4, figsize=(12.6, 8.0))
    arm = {"A": "盲掃", "B": "知情編輯", "C": "SM 導引"}
    for ax, cid in zip(axes.flat, CHAMPS):
        p = _loadp("dedust_ref2_input", cid)
        v = rv[cid]
        _diff_show(ax, p, w17,
                   f"{cid}（{arm[man[cid]['family'][0]]}, k={man[cid].get('flip_k')}）\n"
                   f"wm +{v['wm'][2]:.2f} · rad +{v['rad_margin']:.2f}", tfs=9.6)
    fig.suptitle("八冠軍 gallery（綠＝加銅／紅＝去銅,相對前任 w17;三標全過,certified;按 wm 排序）\n"
                 "共同體質：3 組件、零粉塵、主件 240-250px——w17 高地是一整片高原",
                 color=INK, fontsize=12.5)
    fig.tight_layout(h_pad=3.0)
    save(fig, "f14_champions8.png")


# ==== F15 前三名三標曲線 ====
def fig15():
    TOP3 = [("c21_sm", DBLUE, "c21（旗艦,wm +0.20）"),
            ("b11_k2", ORANGE, "b11（知情編輯,rad 兩切面最佳）"),
            ("a15_k4", AQUA, "a15（rad 王 +0.56,wm +0.03）")]
    w17 = _loadp("dedust_ref1_input", "w17_k8")
    resp_w17 = _find_resp("dedust_verify_interp", w17)
    freq = 26.5 + (np.arange(resp_w17.shape[1]) - 5) * 0.5
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.4))
    for col, (idx, spec_y, lab) in enumerate(((0, -10, "S11 (dB) — spec ≤ −10"),
                                              (1, 4, "Gain (dB) — spec ≥ +4"))):
        ax = axes[0][col]
        ax.axvspan(26.5, 29.5, color=GRID, alpha=0.45)
        ax.axhline(spec_y, color=RED, ls=":", lw=1.3)
        ax.plot(freq, resp_w17[idx], color=MUTED, lw=1.6,
                label="w17（前任,−0.06）" if col == 0 else None)
        for cid, c, l in TOP3:
            r = _find_resp("dedust_ref2v", _loadp("dedust_ref2_input", cid))
            ax.plot(freq, r[idx], color=c, lw=2.1, label=l if col == 0 else None)
        style_ax(ax, "頻率 (GHz)", lab.split(" — ")[0], lab, tfs=11)
    axes[0][0].legend(fontsize=8.4, loc="lower left", framealpha=0.94).get_frame().set_edgecolor(GRID)
    radw = torch.load(str(DATASET_PATH.joinpath("dedust_verify_interp", "rad", "w17_k8.pt")),
                      weights_only=True)
    for col, cut in ((0, "phi0"), (1, "phi90")):
        ax = axes[1][col]
        ax.axvspan(-45, 45, color=GRID, alpha=0.45)
        ax.plot(np.asarray(radw["theta"]), np.asarray(radw[cut]), color=MUTED, lw=1.6)
        for cid, c, _l in TOP3:
            rad = torch.load(str(DATASET_PATH.joinpath("dedust_ref2v", "rad", f"{cid}.pt")),
                             weights_only=True)
            ax.plot(np.asarray(rad["theta"]), np.asarray(rad[cut]), color=c, lw=2.1)
        ax.set_xlim(-90, 90)
        ax.set_xticks([-90, -45, 0, 45, 90])
        style_ax(ax, "θ (deg)", "Gain (dB)", f"Radiation {cut} — 灰帶＝±45° 窗", tfs=11)
    fig.suptitle("前三名三標曲線（灰＝前任 w17）：三種出身、三種體質,全部三標過線",
                 color=INK, fontsize=13)
    fig.tight_layout()
    save(fig, "f15_top3_curves.png")


# ==== F16 +0.48 假象偵破 ====
def fig16():
    w17 = _loadp("dedust_ref1_input", "w17_k8")
    resp_bug = _find_resp("dedust_ref1", w17)          # 舊萃取碼（頻點偏格）
    resp_fix = _find_resp("dedust_w17rep", w17)        # 修復後（公證批）
    rep = _load_json("dedust_w17rep")
    freq = 26.5 + (np.arange(resp_fix.shape[1]) - 5) * 0.5

    fig = plt.figure(figsize=(11.6, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.22)
    ax = fig.add_subplot(gs[0])
    ax.axvspan(26.5, 29.5, color=GRID, alpha=0.45)
    ax.axhline(4, color=RED, ls=":", lw=1.3)
    ax.plot(freq, resp_bug[1], color=RED, lw=2.2, ls="--", label="舊萃取碼（頻點偏 0.5 GHz）")
    ax.plot(freq, resp_fix[1], color=DBLUE, lw=2.2, label="修復後（align_curve,公證值）")
    i = 9
    ax.annotate("", (freq[i] - 0.5, resp_fix[1][i]), (freq[i], resp_fix[1][i]),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.4))
    ax.text(freq[i] - 0.25, resp_fix[1][i] + 0.35, "整條曲線偏半格", color=INK2,
            fontsize=9, ha="center")
    style_ax(ax, "頻率 (GHz)", "Gain (dB)",
             "w17 的 Gain：同一筆模擬,兩種萃取——S11/rad 完全相同,只有 Gain 差", tfs=11)
    ax.legend(fontsize=9, loc="lower center", framealpha=0.94).get_frame().set_edgecolor(GRID)

    ax = fig.add_subplot(gs[1])
    reps = sorted(k for k in rep if "wm" in rep[k])
    vals = [rep[k]["wm"][2] for k in reps]
    ax.scatter([0], [0.48], s=110, color=RED, marker="X", zorder=4)
    ax.scatter(np.arange(1, len(vals) + 1), vals, s=64, color=DBLUE, zorder=4,
               edgecolor=SURF, lw=0.8)
    ax.axhline(0, color=RED, ls=":", lw=1.2)
    ax.annotate("+0.48 假象\n（舊碼單次量測）", (0, 0.48), (0.5, 0.30), fontsize=9,
                color=RED, arrowprops=dict(arrowstyle="-", color=RED, lw=0.9))
    ax.text(4.5, -0.16, f"公證 {len(vals)}/{len(vals)}＝−0.06\n（bit 級一致,含跨機）",
            color=DBLUE, fontsize=9.5, ha="center")
    ax.set_xticks([])
    ax.set_ylim(-0.35, 0.62)
    style_ax(ax, "量測次序", "worst-margin (dB)",
             "公證制度抓到它：8 次重測翻案", tfs=11)
    fig.suptitle("+0.48 假象偵破（R10）：萃取對位 bug 的個案——「紀錄級結論一律公證」鐵則的由來",
                 color=INK, fontsize=12.5)
    save(fig, "f16_phantom_case.png")


if __name__ == "__main__":
    fig12()
    fig13()
    fig14()
    fig15()
    fig16()
