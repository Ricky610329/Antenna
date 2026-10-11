# -*- coding: utf-8 -*-
"""meeting_emforge_onboard.py — 「同學怎麼把自己的結構接上 emforge」示意圖（組會 2026-10 第 34 頁）。
介面名稱唯讀核對自 emforge repo：emforge/model.py（Simulator 協定 open/simulate/kill/close、Profile、Spec
score=min(measure[axis]+offset)＋gates）、emforge/specs.py（register_measure）、emforge/profiles.py
（<root>/registry.py 呼叫 register_profile／register_spec／register_measure）、docs/platform-quickstart.md
（fake_f1 假儀器流程、多機 worker）、emforge/runtime/__init__.py（tick：collect → notarize → schedule）。
純示意，無資料。"""
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from script.figs.meeting_figkit import (FULL, INK, INK2, MUTED, RED, DBLUE, AQUA, ORANGE, SURF,
                                        plt, save)

fig = plt.figure(figsize=FULL)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 120)
ax.set_ylim(0, 55)
ax.axis("off")


def box(x, y, w, h, fc, ec, lw=1.6, ls="-", r=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=2))


def arrow(p, q, c=INK2, ls="-", lw=1.8, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=18, color=c, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=3))


# ── 左：要填的四樣 ──
ax.text(1.5, 49.3, "你要寫的 4 樣東西", fontsize=15, color=INK, weight="bold", va="center")
ax.text(1.5, 46.6, "放在自己的 registry.py 登錄", fontsize=12.5, color=MUTED, va="center")
LEFT = [
    ("模擬器 adapter", "open · simulate · kill · close\n一筆設計進、一筆響應出"),
    ("評分 measure", "把響應換成指標\n例：每頻點 S11、Gain 餘裕"),
    ("規格 Spec", "分數＝各指標＋偏移取最小\n可加硬門檻"),
    ("模擬條件 Profile", "網格大小、必為金屬的饋墊\n逾時、幾何版本"),
]
ys = [35.2, 25.2, 15.2, 5.2]
for k, ((t, s), y) in enumerate(zip(LEFT, ys)):
    box(1.5, y, 34, 8.6, "#eef3fa", DBLUE)
    ax.text(3.3, y + 6.2, t, fontsize=14, color=DBLUE, weight="bold", va="center")
    ax.text(3.3, y + 2.6, s, fontsize=12.5, color=INK, va="center", linespacing=1.25)
    arrow((35.6, y + 4.3), (47.3, 33.0 - k * 5.0), c=DBLUE, lw=1.5)

# ── 中：Runtime ──
box(47.5, 11.0, 30, 31.5, "#f3f2ee", INK, lw=2.0)
ax.text(62.5, 39.4, "emforge Runtime", fontsize=16, color=INK, weight="bold", ha="center", va="center")
ax.text(62.5, 36.4, "平台負責，你不用寫", fontsize=12.5, color=MUTED, ha="center", va="center")
STEPS = ["驗證候選格式", "去重：同設計不重量", "排程派工、斷點續跑", "收件、存原始響應", "用你的規格評分",
         "破紀錄自動重測公證"]
for i, s in enumerate(STEPS):
    ax.text(51.0, 32.6 - i * 3.55, "▸ " + s, fontsize=13, color=INK, va="center")
# 上方：候選來源
box(52.5, 46.0, 20, 4.8, SURF, MUTED, ls="--")
ax.text(62.5, 48.4, "候選設計：演算法或人挑", fontsize=12.5, color=INK2, ha="center", va="center")
arrow((62.5, 45.9), (62.5, 42.7), c=MUTED)

# ── 右：worker／HFSS ──
ax.text(104.0, 49.3, "模擬機（可多台）", fontsize=15, color=INK, weight="bold", ha="center", va="center")
for i, y in enumerate([35.5, 24.0, 12.5]):
    box(91.5, y, 25, 8.4, "#eaf5f0", AQUA)
    ax.text(104.0, y + 5.6, f"worker {i + 1}", fontsize=14, color=AQUA, weight="bold", ha="center",
            va="center")
    ax.text(104.0, y + 2.4, "HFSS ＋ 你的 adapter", fontsize=12.5, color=INK, ha="center", va="center")
    arrow((77.8, y + 5.4), (91.3, y + 5.4), c=INK2, lw=1.6)
    arrow((91.3, y + 2.8), (77.8, y + 2.8), c=AQUA, lw=1.6)
ax.text(84.5, 41.6, "派工", fontsize=12.5, color=INK2, ha="center", va="bottom")
ax.text(84.5, 14.6, "回傳響應", fontsize=12.5, color=AQUA, ha="center", va="top")

# ── 下：沒有 HFSS 的試跑路徑 ──
box(47.5, 1.0, 69, 7.6, "#fdf1ea", ORANGE, ls="--", lw=1.6)
ax.text(49.5, 4.8, "沒有 HFSS 時：先用假儀器 fake_f1 跑完整條流程，步驟與真機相同",
        fontsize=13, color=ORANGE, weight="bold", va="center")

fig.text(0.99, 0.995, "限制：設計需為二值像素遮罩，每格有金屬或沒有", ha="right", va="top", fontsize=12.5,
         color=RED, bbox=dict(boxstyle="round,pad=0.35", fc=SURF, ec=RED, lw=1.2))
save(fig, "emforge_onboard.png")
