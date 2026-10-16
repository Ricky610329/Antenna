"""R81 學長 v6 規格（11–40 GHz，阻帶 ≤ −20 dB）兩張圖：性能爬升、目前最佳設計卡片。

用法：python -m script.figs.r81_v6 <輸出資料夾> [--id r81v2_b71_003_f8fd2069]
輸出：r81v6_climb.png、r81v6_best_card.png（各格正方形）。
資料：v2_v001 資料根中 11–40 GHz 量測的 store（補量 r81v6m02 與 b62 起各批），依批號排序；重測（~rep）不計入。
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from script import r81_curve_sm as sm
from script.figs import r81_progress as rp

DOT, LINE, MARK = "#9ec5f4", "#2a78d6", "#eb6834"
V6, V6A = sm.REPO / "configs" / "dual_r81_v6.yaml", sm.REPO / "configs" / "dual_r81_v6a.yaml"
EVENTS = [(62, "斷開＋大步變異 30 格：壓掉 13.5 GHz 峰"), (67, "饋墊附近細步"), (71, "12 格＋前 10 名起點")]


def rows_v6(cfg, grid):
    """11–40 GHz store 的全部 ok 實測，依（批號, store, id）排序；同圖只留第一筆。"""
    stores = [s for s in sm.V2_DATASET.glob("dedust_r81v6*s?")
              if re.search(r"(m02|b\d\d)s\d$", s.name) and (s / "results.json").exists()]
    key = lambda s: 0 if "m02" in s.name else int(re.search(r"b(\d\d)s\d$", s.name).group(1))
    rows, seen = [], set()
    for s in sorted(stores, key=lambda s: (key(s), s.name)):
        for r in sm.load_store(s, grid):
            if "~rep" in r["id"] or r["sha"] in seen:
                continue
            sc = sm.true_wm(r, cfg, grid)
            if sc:
                seen.add(r["sha"])
                r.update(batch=key(s), wm=sc["worst_margin"], margins=sc["margins"])
                rows.append(r)
    return rows


def climb_figure(out, rows):
    wm = np.array([r["wm"] for r in rows])
    best = np.maximum.accumulate(wm)
    x = np.arange(1, len(wm) + 1)
    lo = -20
    fig, ax = plt.subplots(figsize=(7, 7), dpi=160)
    fig.patch.set_facecolor(rp.SURF)
    ax.scatter(x, np.clip(wm, lo, None), s=8, color=DOT, lw=0, label="每筆模擬")
    ax.step(x, best, where="post", color=LINE, lw=2.2, label="累計最佳")
    ax.axhline(0, color=rp.INK, lw=1.2, ls="--")
    ax.text(4, 0.3, "達標 WM = 0", fontsize=9, color=rp.INK)
    batches = [r["batch"] for r in rows]
    for b, text in EVENTS:
        if b in batches:
            i = batches.index(b)
            ax.axvline(x[i], color=MARK, lw=0.8, ls=":")
            ax.text(x[i] - 3, lo + 0.4, text, rotation=90, ha="right", va="bottom", fontsize=8.5, color=MARK,
                    bbox=dict(facecolor=rp.SURF, edgecolor="none", pad=1.5))
    ax.annotate(f"{best[-1]:+.2f}（單次）", xy=(x[-1], best[-1]), xytext=(-6, 10), textcoords="offset points",
                ha="right", fontsize=10, color=LINE)
    ax.set_xlim(0, x[-1] + 4)
    ax.set_ylim(lo, 2)
    ax.set_xlabel("累計 HFSS 模擬筆數（11–40 GHz）", color=rp.MUTED)
    ax.set_ylabel("WM (dB)", color=rp.MUTED)
    ax.set_title("濾波器 v6 性能爬升（阻帶 11–19、36–40 ≤ −20 dB）", loc="left", fontsize=11.5, color=rp.INK)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    rp.style(ax)
    ax.set_box_aspect(1)
    fig.text(0.02, 0.01, f"共 {len(wm)} 筆單次 HFSS；前 24 筆為 18–40 好設計以 11–40 重量；低於 {lo} dB 的點壓在底線。",
             fontsize=8, color=rp.MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, facecolor=rp.SURF)
    plt.close(fig)
    print("saved", out, len(wm), round(best[-1], 3))


def curve(ax, grid, y, label, v6, v6a, margins):
    for spec, ls in ((v6a, ":"), (v6, "--")):
        for b in spec.score_spec["bands"]:
            if b["label"] != label:
                continue
            lo, hi, thr = b["start_ghz"], b["stop_ghz"], b["threshold"]
            ax.plot([lo, hi], [thr, thr], color=rp.INK, lw=1.4, ls=ls)
            if spec is v6:
                y0, y1 = (thr, 2) if b["relation"] == "le" else (-45, thr)
                ax.fill_between([lo, hi], y0, y1, color=rp.BAD, alpha=0.08, lw=0)
    ax.plot(grid, y, color=rp.CURVE, lw=2.0)
    ax.set_xlim(11, 40)
    ax.set_ylim(-45, 2)
    ax.set_xticks([11, 15, 19, 22, 26, 30, 36, 40])
    ax.set_xlabel("頻率 (GHz)", color=rp.MUTED, fontsize=9.5)
    ax.set_ylabel(f"{label} (dB)", color=rp.MUTED, fontsize=9.5)
    keys = [k for k in margins if k.startswith(label.lower())]
    ax.set_title(f"{label}：" + "、".join(f"{k.split('_', 1)[1]} {margins[k]:+.2f}" for k in keys) + " dB",
                 loc="left", fontsize=10.5, color=rp.INK)
    rp.style(ax)
    ax.set_box_aspect(1)


def card_figure(out, r, grid, v6, v6a, n):
    wma = sm.true_wm(r, v6a, grid)["worst_margin"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 10.6), dpi=160)
    fig.patch.set_facecolor(rp.SURF)
    for ax, (k, lab) in zip((axes[0, 0], axes[0, 1], axes[1, 0]), ((1, "S21"), (0, "S11"), (2, "S22"))):
        curve(ax, grid, r["y"][k], lab, v6, v6a, r["margins"])
    px = axes[1, 1]
    px.imshow(r["pattern"].reshape(25, 25), cmap="Greys", origin="upper", interpolation="nearest")
    px.set_xticks([]); px.set_yticks([])
    px.set_title("pattern（25×25，上下緣中央為兩埠饋墊，兩埠金屬不相連）", loc="left", fontsize=10, color=rp.INK)
    px.set_xlabel(r["id"], fontsize=8.5, color=rp.MUTED)
    fig.suptitle(f"濾波器 v6 目前最佳：WM {r['wm']:+.2f} dB（−20 版）、{wma:+.2f} dB（−15 版），單次未公證",
                 x=0.03, ha="left", fontsize=12.5, color=rp.INK)
    fig.text(0.03, 0.01, "虛線＝v6（阻帶 ≤ −20）、點線＝v6a（≤ −15）、淡紅＝v6 禁區；通帶 26–30 GHz ≥ −3，"
             f"S11/S22 26.5–29.5 GHz ≤ −10。{n} 筆 11–40 GHz 模擬中的最佳。", fontsize=8.5, color=rp.MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96), h_pad=3)
    fig.savefig(out, facecolor=rp.SURF)
    plt.close(fig)
    print("saved", out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--id", default=None, help="預設＝v6 WM 最高者")
    args = ap.parse_args(argv)
    plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
    v6, v6a = sm.load_cfg(V6), sm.load_cfg(V6A)
    grid = sm.frequency_grid(v6.measurement)
    rows = rows_v6(v6, grid)
    args.outdir.mkdir(parents=True, exist_ok=True)
    climb_figure(args.outdir / "r81v6_climb.png", rows)
    best = next(r for r in rows if r["id"] == args.id) if args.id else max(rows, key=lambda r: r["wm"])
    card_figure(args.outdir / "r81v6_best_card.png", best, grid, v6, v6a, len(rows))


if __name__ == "__main__":
    main()
