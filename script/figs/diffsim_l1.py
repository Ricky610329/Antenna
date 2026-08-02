# -*- coding: utf-8 -*-
"""script/figs/diffsim_l1.py — analysis-08 的 L1 診斷四圖。

    python script/figs/diffsim_l1.py [--split dev] [--n 150]

產出 `docs/log/assets/analysis-08/`：
    l1_scatter.png    diffsim wm vs HFSS wm（分層著色）＋各層 ρ
    l1_curves.png     S11/Gain 曲線實例疊圖（模型 vs HFSS）
    l1_perfreq.png    每頻點 ρ（分層）——看模型在頻帶哪一段有訊號
    l1_channel.png    通道歸因：ρ(mS11) / ρ(mGain) / ρ(wm) 分層長條
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]   # 中文字型（同 figs/ 慣例）
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from script.diffsim import data as D            # noqa: E402
from script.diffsim.run import pick, run_l1     # noqa: E402
from script.diffsim.eval import margins, rank_rho   # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "docs", "log", "assets", "analysis-08")
COL = {"clean": "#2f6fed", "neg": "#e0663c", "senior": "#7a8794", "frozen": "#1f9d55"}
FREQ = np.linspace(24, 32, 17)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--er", type=float, default=3.0)
    ap.add_argument("--q", type=float, default=15.0)
    ap.add_argument("--gap", type=float, default=2.0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    idx = D.load()
    split, _ = D.assign_split(idx)
    sel = pick(idx, split, a.split, a.n)
    Y, st = idx["y"][sel], idx["stratum"][sel]
    P, _ = run_l1(idx, sel, er=a.er, q=a.q, gap=a.gap, diag=a.gap, rad_eff=True, batch=24)
    wmt, mst, mgt = margins(Y)
    wmp, msp, mgp = margins(P)
    strata = [s for s in ("clean", "neg", "senior", "frozen") if (st == s).any()]

    # ① 散點
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    for s in strata:
        m = st == s
        ax.scatter(wmp[m], wmt[m], s=14, alpha=.65, c=COL[s],
                   label=f"{s} (n={m.sum()}, ρ={rank_rho(wmp[m], wmt[m])[0]:+.2f})")
    ax.set_xlabel("diffsim L1 worst_margin (dB, 裸)")
    ax.set_ylabel("HFSS worst_margin (dB)")
    ax.set_title(f"L1 排序力（{a.split}）｜pooled ρ = {rank_rho(wmp, wmt)[0]:+.3f}")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "l1_scatter.png"), dpi=140)
    plt.close(fig)

    # ② 曲線實例（各層取 wm 最好的一筆）
    fig, axes = plt.subplots(2, len(strata), figsize=(3.4 * len(strata), 5.6), squeeze=False)
    for c, s in enumerate(strata):
        m = np.where(st == s)[0]
        i = m[np.argmax(wmt[m])]
        for r, (lab, lo, hi) in enumerate((("S11 (dB)", 0, 17), ("Gain (dB)", 17, 34))):
            ax = axes[r][c]
            ax.plot(FREQ, Y[i, lo:hi], "k-", lw=1.6, label="HFSS")
            ax.plot(FREQ, P[i, lo:hi], "--", c=COL[s], lw=1.6, label="diffsim L1")
            ax.axvspan(26.5, 29.5, color="0.85", zorder=0)
            ax.set_ylabel(lab if c == 0 else "")
            ax.set_title(f"{s} (HFSS wm {wmt[i]:+.2f})" if r == 0 else "")
            ax.grid(alpha=.25)
            if r == 1:
                ax.set_xlabel("f (GHz)")
            if r == 0 and c == 0:
                ax.legend(fontsize=8)
    fig.suptitle("曲線實例：模型抓輪廓、不復現（灰帶＝spec 中央平台 26.5–29.5GHz）", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "l1_curves.png"), dpi=140)
    plt.close(fig)

    # ③ 每頻點 ρ
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
    for k, (lab, off) in enumerate((("S11", 0), ("Gain", 17))):
        for s in strata:
            m = st == s
            axes[k].plot(FREQ, [rank_rho(P[m, off + j], Y[m, off + j])[0] for j in range(17)],
                         "o-", ms=3, c=COL[s], label=s)
        axes[k].axhline(0, c="k", lw=.8)
        axes[k].axvspan(26.5, 29.5, color="0.9", zorder=0)
        axes[k].set_title(f"每頻點 ρ — {lab}")
        axes[k].set_xlabel("f (GHz)")
        axes[k].grid(alpha=.25)
    axes[0].set_ylabel("Spearman ρ")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "l1_perfreq.png"), dpi=140)
    plt.close(fig)

    # ④ 通道歸因
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    keys = ["pooled"] + strata
    w, xs = 0.26, np.arange(len(keys))
    for k, (lab, pp, tt) in enumerate((("ρ(mS11)", msp, mst), ("ρ(mGain)", mgp, mgt),
                                       ("ρ(wm)=min", wmp, wmt))):
        v = [rank_rho(pp, tt)[0]] + [rank_rho(pp[st == s], tt[st == s])[0] for s in strata]
        ax.bar(xs + (k - 1) * w, v, w, label=lab)
    ax.axhline(0.40, c="crimson", ls="--", lw=1, label="gate 0.40")
    ax.axhline(0, c="k", lw=.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(keys)
    ax.set_ylabel("Spearman ρ")
    ax.set_title("通道歸因：wm = min(兩路) → 被弱的那路拖")
    ax.legend(fontsize=8)
    ax.grid(alpha=.25, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "l1_channel.png"), dpi=140)
    plt.close(fig)
    print("圖已落地：", OUT)


if __name__ == "__main__":
    main()
