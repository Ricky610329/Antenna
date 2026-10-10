"""Figures for the Chinese extended abstract (docs/paper/build_extended_abstract.py).

Every figure is redrawn from frozen JSON next to this file; nothing here runs HFSS.
Each builder writes docs/paper/figures/extended-abstract-<name>.png and returns the
facts it asserted, which the PDF builder records in its QA report.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import ndimage

BASE = Path(__file__).resolve().parent
FIGDIR = BASE / "figures"
FREE = BASE / "single-port-evidence.json"
SYM = BASE / "symmetric-best-evidence.json"
FILTER = BASE / "final-candidate-response.json"
HISTORY = BASE / "paper-evidence-data.json"
FROZEN = BASE / "extended-abstract-figure-data.json"
WIDTH_IN = 7.0

BLUE, AMBER, TEAL = "#234c68", "#bd6532", "#3b8a7e"
INK, MUTED, GRID, PAPER = "#18212b", "#9eabb5", "#e3e8ec", "#f7f8f9"

plt.rcParams.update({"font.family": ["Microsoft JhengHei", "sans-serif"], "font.size": 7,
                     "axes.unicode_minus": False, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": .6,
                     "xtick.major.size": 2, "ytick.major.size": 2,
                     "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
                     "axes.titlesize": 7, "axes.titlepad": 3, "axes.labelpad": 1.5,
                     "legend.fontsize": 6, "legend.frameon": False})


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def components(bits: np.ndarray) -> int:
    """Metal pieces counted with diagonal contacts joined (the bridges)."""
    return int(ndimage.label(bits, structure=np.ones((3, 3)))[1])


def sym_bits(sym: dict) -> np.ndarray:
    return np.array([[int(c) for c in row] for row in sym["pattern_rows"]])


def save(fig, name: str) -> Path:
    """Save at 300 dpi; any missing-glyph warning (e.g. U+2212 in JhengHei) is an error."""
    path = FIGDIR / f"extended-abstract-{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig.savefig(path, dpi=300, metadata={"Software": None})
    plt.close(fig)
    missing = [str(w.message) for w in caught if "missing from" in str(w.message)]
    assert not missing, missing
    return path


def pattern_axis(ax, bits, color, title, mirror=False, ports=("feed",)):
    ax.imshow(bits, cmap=ListedColormap([PAPER, color]), interpolation="nearest", vmin=0, vmax=1)
    if mirror:
        ax.axvline(12, color=INK, lw=.6, ls=(0, (2, 2)))
    for port in ports:   # rows are drawn top-down, so row 24 (the feed edge) is at the bottom
        y, marker, text = {"feed": (25.4, "^", None), "P1": (25.4, "^", "P1"), "P2": (-1.4, "v", "P2")}[port]
        ax.scatter([12], [y], marker=marker, s=14, color=INK, zorder=3, clip_on=False)
        if text:
            ax.text(14, y, text, fontsize=6, va="center", clip_on=False)
    ax.set(xticks=[], yticks=[], xlim=(-.5, 24.5), ylim=(24.5, -.5))
    for side in ax.spines.values():
        side.set(visible=True, linewidth=.5, color=MUTED)
    ax.set_title(title, pad=9 if "P2" in ports else 3)


def freq_axis(ax, title, band=(26.5, 29.5)):
    if band:
        ax.axvspan(*band, color=GRID, lw=0)
    ax.set_xlim(24, 32)
    ax.set_xticks(range(24, 33, 2))
    ax.set_xlabel("頻率（GHz）")
    ax.set_title(title)
    ax.grid(color=GRID, lw=.4)


def limit(ax, x0, x1, y):
    ax.plot([x0, x1], [y, y], color=INK, lw=.7, ls=(0, (3, 2)))


def polar_axis(ax, theta, cuts, title):
    """Cuts normalised to boresight; shaded +/-45 deg window and its -3 dB floor."""
    t = np.radians(theta)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    w = np.radians(np.linspace(-45, 45, 91))
    ax.fill_between(w, -30, 3, color=GRID, lw=0)
    ax.plot(w, np.full_like(w, -3), color=INK, lw=.7, ls=(0, (3, 2)))
    margins = {}
    for label, cut, color in cuts:
        norm = np.asarray(cut) - np.asarray(cut)[np.asarray(theta) == 0][0]
        margins[label] = float(norm[np.abs(theta) <= 45].min() + 3)
        ax.plot(t, np.clip(norm, -30, 3), color=color, lw=1, label=label)
    ax.set_ylim(-30, 3)
    ax.set_yticks([-20, -10, -3])
    ax.set_yticklabels([])
    ax.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(["0°", "45°", "90°", "", "180°", "", "-90°", "-45°"], fontsize=5.5)
    ax.tick_params(pad=-1)
    ax.grid(color=GRID, lw=.4)
    ax.spines["polar"].set(linewidth=.5, color=MUTED)
    ax.set_title(title, pad=8)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.13), ncol=2, handlelength=1.2, columnspacing=.8)
    return margins


def band_margins(f, s11, gain):
    band = (f >= 26.5) & (f <= 29.5)
    return float(-10 - s11[band].max()), float(gain[band].min() - 4)


def workflow() -> dict:
    """Fig 1: the batch surrogate-learning loop."""
    fig = plt.figure(figsize=(WIDTH_IN, 1.66))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 100), ylim=(0, 23.6), xticks=[], yticks=[])
    ax.axis("off")
    boxes = [  # x0, x1, title, body, edge
        (.5, 17.5, "研究者＋AI agent", "設定目標與判準\n提出探索算子", BLUE),
        (21.5, 39.5, "候選池", "像素翻轉、嫁接重組\n隨機與對稱取樣", TEAL),
        (43.5, 61.5, "SM ensemble 篩選", "三種配額：預測較好\n／模型分歧／隨機", BLUE),
        (65.5, 80.5, "HFSS 批次模擬", "三台工作站\n並行求解", TEAL),
        (84.5, 99.5, "資料庫", "有效結果全部入庫\n（含不合格）", BLUE),
    ]
    y0, y1 = 9.2, 18.4
    for x0, x1, head, body, edge in boxes:
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=1.2",
                                    fc="#f3f6f8" if edge == BLUE else "#f1f6f5", ec=edge, lw=.9))
        ax.text((x0 + x1) / 2, y1 - 2.3, head, ha="center", va="center", fontsize=7.4, weight="bold", color=INK)
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 - 1.6, body, ha="center", va="center", fontsize=6.3,
                color="#33414d", linespacing=1.35)
    arrow = dict(arrowstyle="-|>", mutation_scale=7, lw=.9, color=INK, shrinkA=0, shrinkB=0)
    for (_, a1, *_), (b0, *_) in zip(boxes, boxes[1:]):
        ax.add_patch(FancyArrowPatch((a1 + .3, (y0 + y1) / 2), (b0 - .3, (y0 + y1) / 2), **arrow))
    # Loops: retrain the SM from the data (below) and analyse each batch (above).
    loop = dict(color=BLUE, lw=.9)
    ax.plot([92, 92, 52.5], [y0, 6.6, 6.6], **loop)
    ax.add_patch(FancyArrowPatch((52.5, 6.6), (52.5, y0 - .2), **{**arrow, "color": BLUE}))
    ax.text(72, 6.9, "定期重訓 SM", ha="center", va="bottom", fontsize=6.3, color=BLUE)
    ax.plot([92, 92, 9], [y1, 21.3, 21.3], **loop)
    ax.add_patch(FancyArrowPatch((9, 21.3), (9, y1 + .2), **{**arrow, "color": BLUE}))
    ax.text(50.5, 21.6, "每批結果分析 → 調整策略比例或提出新算子", ha="center", va="bottom", fontsize=6.3, color=BLUE)
    ax.add_patch(FancyBboxPatch((21.5, .6), 78, 3.6, boxstyle="round,pad=0,rounding_size=.8",
                                fc="#eef1f4", ec="none"))
    ax.text(60.5, 2.4, "emforge 平台：派工、去重、收件、評分（三台工作站共用）", ha="center", va="center",
            fontsize=6.3, color="#33414d")
    return {"path": save(fig, "workflow"), "facts": {"boxes": [b[2] for b in boxes]}}


def antenna() -> dict:
    """Fig 2: free-form delivery antenna (bits, S11, gain, 28 GHz cuts)."""
    free = load(FREE)
    bits = np.asarray(free["candidate"]["bits"], dtype=int)
    assert bits.shape == (25, 25) and bits[24, 12] == 1
    f = np.asarray(free["frequency_ghz"])
    s11 = np.asarray(free["response_db"]["S11"])
    gain = np.asarray(free["response_db"]["RealizedGainTotal"])
    m_s11, m_gain = band_margins(f, s11, gain)
    assert (round(m_s11, 2), round(m_gain, 2)) == (1.13, .77)
    rad = free["radiation"]
    fig = plt.figure(figsize=(WIDTH_IN, 1.9))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1.25, 1.25, 1.18], wspace=.42,
                          left=.012, right=.985, bottom=.24, top=.8)
    pattern_axis(fig.add_subplot(gs[0, 0]), bits, BLUE, f"(a) 金屬圖形\n{components(bits)} 個連通塊")
    ax = fig.add_subplot(gs[0, 1])
    freq_axis(ax, "(b) S11（dB）")
    limit(ax, 26.5, 29.5, -10)
    ax.plot(f, s11, color=BLUE, lw=1.1, marker="o", ms=1.6)
    ax.text(.04, .05, f"帶內餘裕 +{m_s11:.2f}", transform=ax.transAxes, fontsize=6, color=INK)
    ax = fig.add_subplot(gs[0, 2])
    freq_axis(ax, "(c) 正向實現增益（dBi）")
    limit(ax, 26.5, 29.5, 4)
    ax.plot(f, gain, color=BLUE, lw=1.1, marker="o", ms=1.6)
    ax.text(.04, .05, f"帶內餘裕 +{m_gain:.2f}（即 WM）", transform=ax.transAxes, fontsize=6, color=INK)
    ax = fig.add_subplot(gs[0, 3], projection="polar")
    margins = polar_axis(ax, np.asarray(rad["theta_deg"]),
                         [("φ = 0°", rad["phi0_db"], BLUE), ("φ = 90°", rad["phi90_db"], AMBER)],
                         "(d) 28 GHz radiation pattern")
    assert abs(margins["φ = 0°"] - rad["margin_db"]["phi0"]) < 1e-6
    assert abs(margins["φ = 90°"] - rad["margin_db"]["phi90"]) < 1e-6
    return {"path": save(fig, "antenna"),
            "facts": {"s11_margin": round(m_s11, 4), "gain_margin": round(m_gain, 4),
                      "rad_margin": {k: round(v, 4) for k, v in margins.items()}, "components_8conn": components(bits)}}


def oob() -> dict:
    """Fig 3: low-side out-of-band record (no bridges) against the bridged delivery design."""
    rec = load(FROZEN)["snapshot"]["oob_record"]
    free = load(FREE)
    bits = np.asarray(rec["bits"], dtype=int)
    assert bits[24, 12] == 1
    f = np.asarray(free["frequency_ghz"])
    s11, gain = np.asarray(rec["S11_db"]), np.asarray(rec["RealizedGainTotal_dbi"])
    lo = float(gain[f <= 25.5].max())
    m_s11, m_gain = band_margins(f, s11, gain)
    assert abs(lo - rec["snapshot"]["lo"]) < .01 and abs(min(m_s11, m_gain) - rec["snapshot"]["wm"]) < .01
    free_gain = np.asarray(free["response_db"]["RealizedGainTotal"])
    fig = plt.figure(figsize=(WIDTH_IN, 1.9))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1.25, 1.25, 1.18], wspace=.42,
                          left=.012, right=.985, bottom=.24, top=.8)
    pattern_axis(fig.add_subplot(gs[0, 0]), bits, TEAL, f"(a) 金屬圖形（無橋）\n{components(bits)} 個連通塊")
    ax = fig.add_subplot(gs[0, 1])
    freq_axis(ax, "(b) S11（dB）")
    limit(ax, 26.5, 29.5, -10)
    ax.plot(f, s11, color=TEAL, lw=1.1, marker="o", ms=1.6)
    ax.text(.04, .05, f"帶內餘裕 +{m_s11:.2f}", transform=ax.transAxes, fontsize=6, color=INK)
    ax = fig.add_subplot(gs[0, 2])
    freq_axis(ax, "(c) 正向實現增益（dBi）")
    ax.axvspan(24, 25.5, color="#f6e9e0", lw=0)
    limit(ax, 26.5, 29.5, 4)
    ax.plot(f, free_gain, color=MUTED, lw=.9, ls=(0, (2.5, 1.5)), label="圖 2 交付代表")
    ax.plot(f, gain, color=TEAL, lw=1.1, marker="o", ms=1.6, label=f"本設計 WM +{min(m_s11, m_gain):.2f}")
    peak = f[f <= 25.5][np.argmax(gain[f <= 25.5])]
    ax.annotate(f"低頻最大\n{lo:.2f} dBi", (peak, lo), xytext=(24.25, -.4), textcoords="data", fontsize=6,
                va="top", linespacing=1.1)
    ax.legend(loc="lower right", handlelength=1.6)
    ax = fig.add_subplot(gs[0, 3], projection="polar")
    margins = polar_axis(ax, np.asarray(rec["theta_deg"]),
                         [("φ = 0°", rec["phi0_db"], BLUE), ("φ = 90°", rec["phi90_db"], AMBER)],
                         "(d) 28 GHz radiation pattern")
    assert abs(min(margins.values()) - rec["snapshot"]["rad"]) < .01
    return {"path": save(fig, "oob"),
            "facts": {"lo_dbi": round(lo, 3), "wm_db": round(min(m_s11, m_gain), 3),
                      "rad_margin": round(min(margins.values()), 3), "components_8conn": components(bits),
                      "bridged_s0_wm_db": load(FROZEN)["snapshot"]["r54_bridge_s0"]["rows"][rec["id"]]["s0_bridged_wm"]}}


def filter_() -> dict:
    """Fig 4: dual-port filter best design and its best-so-far history."""
    resp = load(FILTER)
    hist = load(HISTORY)
    bits = np.asarray(hist["candidate"]["bits"], dtype=int)
    assert bits.shape == (25, 25)
    f = np.asarray(resp["frequency_ghz"])
    s11, s21, s22 = (np.asarray(resp["response_db"][k]) for k in ("S11", "S21", "S22"))
    band = (f >= 26.5) & (f <= 29.5)
    passband = (f >= 25.5) & (f <= 30.5)
    stop = (f <= 25) | (f >= 31)
    margins = [-10 - s11[band].max(), -10 - s22[band].max(), s21[passband].min() + 3, -15 - s21[stop].max()]
    assert [round(float(m), 2) for m in margins] == resp["derived_check"]["spec_v2_margins_rounded_0_01_db"]
    unmet = (passband & (s21 < -3)) | (stop & (s21 > -15))
    assert unmet.sum() == 6
    values = [h["value"] for h in hist["history"]]
    assert len(values) == 17 and values[0] == -5.9 and values[-1] == -2.39

    fig = plt.figure(figsize=(WIDTH_IN, 1.78))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1.3, 1.3, 1.3], wspace=.42,
                          left=.012, right=.985, bottom=.2, top=.8)
    pattern_axis(fig.add_subplot(gs[0, 0]), bits, BLUE, "(a) 金屬圖形", ports=("P1", "P2"))
    ax = fig.add_subplot(gs[0, 1])
    freq_axis(ax, "(b) S11、S22（dB）")
    limit(ax, 26.5, 29.5, -10)
    ax.plot(f, s11, color=BLUE, lw=1.1, marker="o", ms=1.6, label=f"S11 +{margins[0]:.2f}")
    ax.plot(f, s22, color=TEAL, lw=1.1, marker="s", ms=1.4, label=f"S22 +{margins[1]:.2f}")
    ax.legend(loc="lower right", handlelength=1.4)
    ax = fig.add_subplot(gs[0, 2])
    freq_axis(ax, "(c) S21（dB）", band=(25.5, 30.5))
    for x0, x1 in ((24, 25), (31, 32)):
        ax.axvspan(x0, x1, color="#f6e9e0", lw=0)
        limit(ax, x0, x1, -15)
    limit(ax, 25.5, 30.5, -3)
    ax.plot(f, s21, color=BLUE, lw=1.1, marker="o", ms=1.6)
    ax.scatter(f[unmet], s21[unmet], s=16, facecolor="none", edgecolor=AMBER, lw=.9, zorder=3)
    ax.text(28, -9, f"通帶 {margins[2]:.2f}\n阻帶 {margins[3]:.2f}", ha="center", va="center", fontsize=6)
    ax = fig.add_subplot(gs[0, 3])
    n = np.arange(1, len(values) + 1)
    ax.step(n, values, where="post", color=BLUE, lw=1.1)
    ax.plot(n, values, "o", color=BLUE, ms=1.8)
    ax.axhline(0, color=INK, lw=.7, ls=(0, (3, 2)))
    ax.annotate(f"{values[0]:.2f}", (1, values[0]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=6)
    ax.annotate(f"{values[-1]:.2f}", (n[-1], values[-1]), xytext=(0, -7), textcoords="offset points",
                ha="right", va="top", fontsize=6)
    ax.set(xlim=(0, 18), ylim=(-6.5, .5), xticks=[1, 5, 9, 13, 17])
    ax.set_xlabel("紀錄次序（08/13–08/24）")
    ax.set_title("(d) 最佳 WM 演進（dB）")
    ax.grid(color=GRID, lw=.4)
    return {"path": save(fig, "filter"),
            "facts": {"margins": [round(float(m), 4) for m in margins], "unmet_points": int(unmet.sum()),
                      "history": [values[0], values[-1], len(values)]}}


def symmetric() -> dict:
    """Fig 5: the R80 mirror-symmetric best against the free-form delivery design."""
    free, sym = load(FREE), load(SYM)
    rad = load(FROZEN)["symmetric_best_radiation"]
    bits = sym_bits(sym)
    assert bits[24, 12] == 1 and (bits[:, :12] == bits[:, 13:][:, ::-1]).all()
    f = np.asarray(sym["frequency_ghz"])
    assert np.array_equal(f, np.asarray(free["frequency_ghz"]))
    curves = {"sym": (np.asarray(sym["S11_db"]), np.asarray(sym["RealizedGainTotal_dbi"])),
              "free": (np.asarray(free["response_db"]["S11"]), np.asarray(free["response_db"]["RealizedGainTotal"]))}
    wm = {k: min(band_margins(f, *v)) for k, v in curves.items()}
    assert abs(wm["sym"] - sym["margins_db"]["wm"]) < 1e-9 and round(wm["free"], 2) == .77
    fig = plt.figure(figsize=(WIDTH_IN, 1.9))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1.25, 1.25, 1.18], wspace=.42,
                          left=.012, right=.985, bottom=.24, top=.8)
    pattern_axis(fig.add_subplot(gs[0, 0]), bits, AMBER, f"(a) 左右鏡射\n{components(bits)} 個連通塊", mirror=True)
    for k, (idx, title, lim) in enumerate(((0, "(b) S11（dB）", -10), (1, "(c) 正向實現增益（dBi）", 4))):
        ax = fig.add_subplot(gs[0, 1 + k])
        freq_axis(ax, title)
        limit(ax, 26.5, 29.5, lim)
        ax.plot(f, curves["free"][idx], color=MUTED, lw=.9, ls=(0, (2.5, 1.5)), label=f"自由 WM +{wm['free']:.2f}")
        ax.plot(f, curves["sym"][idx], color=AMBER, lw=1.1, marker="o", ms=1.6, label=f"鏡射 WM +{wm['sym']:.3f}")
        if k == 1:
            ax.legend(loc="lower left", handlelength=1.6)
    ax = fig.add_subplot(gs[0, 3], projection="polar")
    margins = polar_axis(ax, np.asarray(rad["theta_deg"]),
                         [("φ = 0°", rad["phi0_db"], BLUE), ("φ = 90°", rad["phi90_db"], AMBER)],
                         "(d) 28 GHz radiation pattern")
    assert abs(min(margins.values()) - sym["margins_db"]["radiation_window_28ghz"]) < 1e-3
    return {"path": save(fig, "symmetry"),
            "facts": {"wm_db": {k: round(float(v), 4) for k, v in wm.items()},
                      "components_8conn": components(bits), "rad_margin": {k: round(v, 4) for k, v in margins.items()},
                      "metal_fraction": round(float(bits.mean()), 4)}}


FIGURES = {"workflow": workflow, "antenna": antenna, "oob": oob, "filter": filter_,
           "symmetry": symmetric}
