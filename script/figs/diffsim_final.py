# -*- coding: utf-8 -*-
"""script/figs/diffsim_final.py — analysis-08 的最終對比圖（分層 ρ 全景）。

    python script/figs/diffsim_final.py

數字**寫死在本檔**（來源＝analysis-08 §6.2 的 val 報數），因為那些是「只跑一次」的
gate 讀數，不該每次畫圖就重跑一次 val。要更新請連同 analysis-08 一起改。
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "docs", "log", "assets", "analysis-08")

# val 120 筆，同一把尺（analysis-08 §6.2）
ROWS = [
    ("平凡基線（金屬比例）", [0.163, 0.614, -0.071, 0.162, np.nan], "#b9c0cc"),
    ("L1 腔模型（gate1 過）", [0.508, 0.756, 0.052, 0.352, 0.075], "#2f6fed"),
    ("L2 MoM（gate2 未過）", [0.306, 0.373, 0.441, -0.463, 0.259], "#e0663c"),
    ("L1 + 殘差頭（3 seed）", [0.663, 0.805, 0.368, 0.610, 0.200], "#1f9d55"),
    ("純資料對照組（同架構）", [0.664, 0.835, 0.197, 0.688, 0.296], "#8b5cf6"),
]
COLS = ["pooled", "clean\n(作戰區)", "neg\n(負片域)", "senior\n(粉塵域)", "frozen\n(OOD 尺)"]


def main():
    os.makedirs(OUT, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.2, 4.6))
    x = np.arange(len(COLS))
    w = 0.16
    for k, (name, v, c) in enumerate(ROWS):
        ax.bar(x + (k - 2) * w, v, w, label=name, color=c)
    ax.axhline(0, c="k", lw=.9)
    ax.axhline(0.40, c="crimson", ls="--", lw=1)
    ax.axhline(0.60, c="darkred", ls=":", lw=1)
    ax.text(len(COLS) - 0.42, 0.41, "gate1 0.40", color="crimson", fontsize=8)
    ax.text(len(COLS) - 0.42, 0.61, "gate2 0.60", color="darkred", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(COLS)
    ax.set_ylabel("Spearman ρ（diffsim wm vs HFSS wm）")
    ax.set_title("diffsim 全鏈對照｜val 120 筆・每層 30 筆（analysis-08 §6.2）\n"
                 "★ L1 與 L2 在互補的域上有效；★ 物理錨 vs 純資料對照 pooled 打平", fontsize=11)
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    ax.grid(alpha=.25, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "final_compare.png"), dpi=140, bbox_inches="tight")
    print("圖已落地：", os.path.join(OUT, "final_compare.png"))


if __name__ == "__main__":
    main()
