"""R81：舊 spec 好設計在新寬頻 spec 下的 S21（給學長說明用）。用法：python -m script.figs.r81_spec_gap <out.png>"""
import json, sys
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
B = "//140.123.106.219/temp/碩二_鄒穎麒's/antenna/experiments/r81_wide_filter_20261007/survey_v001/dataset/dedust_r81s21provisional01/"
res = json.load(open(B + "results.json", encoding="utf-8"))
v = res["r81s21p_000_02e2e5a1"]  # H79 family, old-spec good design
x = torch.load(B + v["sample_file"], weights_only=False)
s21 = np.asarray(x[1] if isinstance(x, (tuple, list)) else x["response"])[1]
f = np.linspace(16, 40, 49)

INK, MUTED, BLUE, BAD = "#0b0b0b", "#52514e", "#2a78d6", "#e34948"
fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")

# spec: shade the "not allowed" region in each band
ax.fill_between([16, 20], -20, 2, color=BAD, alpha=0.10, lw=0)
ax.fill_between([36, 40], -20, 2, color=BAD, alpha=0.10, lw=0)
ax.fill_between([26, 30], -3, -45, color=BAD, alpha=0.10, lw=0)
ax.fill_between([20, 26], -45, 2, color="#f0efec", lw=0)
ax.fill_between([30, 36], -45, 2, color="#f0efec", lw=0)
for (a, b, y) in [(16, 20, -20), (36, 40, -20), (26, 30, -3)]:
    ax.plot([a, b], [y, y], color=INK, lw=1.5, ls="--")

ax.plot(f, s21, color=BLUE, lw=2.2)

kw = dict(fontsize=10.5, color=INK)
ax.text(18, -24, "要求：≤ −20 dB", ha="center", **kw)
ax.text(38, -24, "要求：≤ −20 dB", ha="center", **kw)
ax.text(28, -1.2, "要求：≥ −3 dB", ha="center", **kw)
ax.text(23, -43, "緩衝區", ha="center", fontsize=10, color=MUTED)
ax.text(33, -43, "緩衝區", ha="center", fontsize=10, color=MUTED)

ax.annotate("24 GHz 壓到 −30 dB\n（舊 spec 達標的地方）", xy=(23.5, s21[15]), xytext=(17.0, -35),
            fontsize=10.5, color=INK, arrowprops=dict(arrowstyle="->", color=MUTED))
ax.annotate("往低頻又回升\n16 GHz 只有 −4 dB", xy=(16.2, s21[0]), xytext=(20.6, -3.5),
            fontsize=10.5, color=BAD, arrowprops=dict(arrowstyle="->", color=BAD))
ax.annotate("往高頻也回升", xy=(38.5, s21[45]), xytext=(36.6, -14.5),
            fontsize=10.5, color=BAD, arrowprops=dict(arrowstyle="->", color=BAD))

ax.set_xlim(16, 40); ax.set_ylim(-45, 2)
ax.set_xticks([16, 20, 24, 26, 28, 30, 32, 36, 40])
ax.set_xlabel("頻率 (GHz)", color=MUTED); ax.set_ylabel("S21 (dB)", color=MUTED)
ax.grid(axis="y", color="#e4e3df", lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color("#bdbcb6")
ax.tick_params(colors=MUTED)
ax.set_title("舊 spec 的好設計（H79 家族）在新 spec 下：通帶過、兩側阻帶回升",
             loc="left", fontsize=12.5, color=INK, pad=12)
fig.text(0.01, 0.01, "R81 寬頻實測 16–40 GHz，HFSS Discrete，2026-10-12；同家族另 9 筆形狀相同。紅色區＝不允許的範圍。",
         fontsize=8.5, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(sys.argv[1], facecolor=fig.get_facecolor())
print("saved", sys.argv[1])
