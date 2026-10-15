"""R80 對稱天線兩張圖：性能爬升（R80 5,000 筆＋10/15 續探）與目前最佳設計卡片。

用法：python -m script.figs.r80_sym <輸出資料夾> [--id r80sc_b02_021_b1988fa4]
輸出：r80sym_climb.png、r80sym_best_card.png（各格正方形）。
WM＝26.5–29.5 GHz S11 ≤ −10 dB 與 Gain ≥ 4 dBi 兩項最差點取小（script/r80_sym_climb.py 同一把尺）。
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from script import r80_sym_climb as sc
from script.figs import r81_progress as rp
from script.figs.report_r1r10_style import polar_rad_ax
from script.profiled_batch import load_profile_config

DOT, LINE, MARK, NEW = "#9ec5f4", "#2a78d6", "#eb6834", "#e34948"
FREQS = np.arange(24, 32.01, .5)


def ordered(cfg):
    """全部唯一實測依 store 完成時間排序（results.json mtime；同 store 內依 id）。"""
    P, Y, ids, sts, _ = sc.load_rows(cfg)
    mt = {}
    for s in set(sts):
        root = sc.V2_ROOT if s.startswith("dedust_r80sc") else sc.R80_ROOT
        mt[s] = os.path.getmtime(root / s / "results.json")
    order = sorted(range(len(ids)), key=lambda i: (mt[sts[i]], sts[i], ids[i]))
    return P[order], Y[order], [ids[i] for i in order], [sts[i] for i in order]


def climb_figure(out, Y, sts):
    wm = sc.margins(Y).min(1)
    best = np.maximum.accumulate(wm)
    x = np.arange(1, len(wm) + 1)
    new = np.array([s.startswith("dedust_r80sc") for s in sts])
    fig, ax = plt.subplots(figsize=(7, 7), dpi=160)
    fig.patch.set_facecolor(rp.SURF)
    lo = -4
    ax.scatter(x[~new], np.clip(wm[~new], lo, None), s=5, color=DOT, lw=0, label="R80 每筆模擬")
    ax.scatter(x[new], np.clip(wm[new], lo, None), s=12, color=NEW, lw=0, label="10/15 續探每筆模擬")
    ax.step(x, best, where="post", color=LINE, lw=2.2, label="累計最佳")
    ax.axhline(0, color=rp.INK, lw=1.2, ls="--")
    ax.text(30, 0.05, "達標 WM = 0", fontsize=9, color=rp.INK)
    i0 = int(np.flatnonzero(new)[0])
    ax.axvline(x[i0], color=MARK, lw=0.8, ls=":")
    ax.text(x[i0] - 25, lo + 0.15, "續探開始：從 R80 前段左右成對翻格", rotation=90, ha="right", va="bottom",
            fontsize=8.5, color=MARK, bbox=dict(facecolor=rp.SURF, edgecolor="none", pad=1.5))
    ax.annotate(f"{best[-1]:+.3f}（已跨機公證）", xy=(x[-1], best[-1]), xytext=(-8, 12), textcoords="offset points",
                ha="right", fontsize=10, color=LINE)
    ax.set_xlim(0, x[-1] + 30)
    ax.set_ylim(lo, 0.5)
    ax.set_xlabel("累計 HFSS 模擬筆數", color=rp.MUTED)
    ax.set_ylabel("WM (dB)", color=rp.MUTED)
    ax.set_title("對稱天線性能爬升（S11 ≤ −10 dB、Gain ≥ 4 dBi，26.5–29.5 GHz）", loc="left", fontsize=11.5,
                 color=rp.INK)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    rp.style(ax)
    ax.set_box_aspect(1)
    n_pass = int((wm > 0).sum())
    fig.text(0.02, 0.01, f"共 {len(wm)} 筆唯一單次 HFSS，依完成時間排序；WM > 0 共 {n_pass} 筆；"
             f"低於 {lo} dB 的點壓在底線。", fontsize=8, color=rp.MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, facecolor=rp.SURF)
    plt.close(fig)


def curve(ax, y, label, thr, rel, unit):
    band = (FREQS >= 26.5) & (FREQS <= 29.5)
    y0, y1 = (thr, 5) if rel == "le" else (-25, thr)
    ax.fill_between([26.5, 29.5], y0, y1, color=rp.BAD, alpha=0.08, lw=0)
    ax.plot([26.5, 29.5], [thr, thr], color=rp.INK, lw=1.4, ls="--")
    ax.plot(FREQS, y, color=rp.CURVE, lw=2.0)
    m = (thr - y[band]).min() if rel == "le" else (y[band] - thr).min()
    ax.set_xlim(24, 32)
    ax.set_ylim(-25, 5) if label == "S11" else ax.set_ylim(-10, 10)
    ax.set_xticks([24, 26, 28, 30, 32])
    ax.set_xlabel("頻率 (GHz)", color=rp.MUTED, fontsize=9.5)
    ax.set_ylabel(f"{label} ({unit})", color=rp.MUTED, fontsize=9.5)
    ax.set_title(f"{label}：帶內餘裕 {m:+.2f} {unit}", loc="left", fontsize=10.5, color=rp.INK)
    rp.style(ax)
    ax.set_box_aspect(1)


def card_figure(out, rid, cfg):
    store = next(s for s in sorted(sc.V2_ROOT.glob("dedust_r80scb??s?"))
                 if rid in json.loads((s / "results.json").read_text(encoding="utf-8")))
    e = json.loads((store / "results.json").read_text(encoding="utf-8"))[rid]
    p, y = torch.load(store / e["sample_file"], weights_only=True, map_location="cpu")
    p, y = np.asarray(p).reshape(25, 25) > 0.5, np.asarray(y, dtype=float)
    rad = torch.load(store / e["rad_file"], weights_only=False)
    wm = sc.margins(y[None]).min()
    fig = plt.figure(figsize=(10, 10.4), dpi=160)
    fig.patch.set_facecolor(rp.SURF)
    curve(fig.add_subplot(2, 2, 1), y[0], "S11", -10, "le", "dB")
    curve(fig.add_subplot(2, 2, 2), y[1], "Gain", 4, "ge", "dBi")
    px = fig.add_subplot(2, 2, 3)
    px.imshow(p, cmap="Greys", origin="upper", interpolation="nearest")
    from antenna.patch.patch_simulator.single_port import diag_bridge_sites
    sites, _ = diag_bridge_sites(p, cfg.measurement["geometry"]["diag_bridge_w"], .2)
    if sites:
        cx, cy = np.array([(s[0], s[1]) for s in sites]).T
        px.scatter(cy / .2 - .5, cx / .2 - .5, marker="D", s=14, color=MARK, lw=0)
    px.axvline(12, color=MARK, lw=0.6, ls=":", alpha=.6)
    px.set_xticks([]); px.set_yticks([])
    px.set_title("pattern（25×25，下緣中央為饋點；虛線＝鏡射軸）", loc="left", fontsize=10.5, color=rp.INK)
    px.set_xlabel(f"橘色菱形＝對角接點的 0.1 mm 橋（{len(sites)} 處）", fontsize=8.5, color=rp.MUTED)
    pa = fig.add_subplot(2, 2, 4, projection="polar")
    th = np.asarray(rad["theta"], dtype=float)
    keep = np.abs(th) <= 90
    polar_rad_ax(pa, th[keep], [(np.asarray(rad["phi0"], float)[keep], rp.CURVE, "phi=0°", 1.8),
                                (np.asarray(rad["phi90"], float)[keep], MARK, "phi=90°", 1.8)], show_window=False)
    pa.set_title("28 GHz 方向圖（GainTotal，dBi）", loc="left", fontsize=10.5, color=rp.INK, pad=14)
    pa.legend(loc="lower left", bbox_to_anchor=(-0.1, -0.12), frameon=False, fontsize=8.5)
    fig.suptitle(f"目前最佳對稱天線 {rid}：WM {wm:+.3f} dB（218 原次、216 重測一致）",
                 x=0.03, ha="left", fontsize=12.5, color=rp.INK)
    fig.text(0.03, 0.01, "虛線＝spec（26.5–29.5 GHz S11 ≤ −10 dB、正向 RealizedGain ≥ 4 dBi），淡紅＝禁區。"
             "金屬左右精確鏡射。", fontsize=8.5, color=rp.MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(out, facecolor=rp.SURF)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--id", default="r80sc_b05_014_eb0a73b6")
    args = ap.parse_args(argv)
    plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
    cfg = load_profile_config(sc.CONFIG)
    args.outdir.mkdir(parents=True, exist_ok=True)
    P, Y, ids, sts = ordered(cfg)
    climb_figure(args.outdir / "r80sym_climb.png", Y, sts)
    card_figure(args.outdir / "r80sym_best_card.png", args.id, cfg)
    print("saved", args.outdir, len(ids))


if __name__ == "__main__":
    main()
