"""R81 濾波器最佳設計演進 GIF。累計最佳每刷新一次就是一格。
左 2×2：S21、S11、S22（規格線）、pattern；右：爬坡（線性 y 軸，隨動畫推進）。

--history v6a（預設）：11–40 GHz 量測起（m02、b62 起、R89 整合批）全以 v6a 尺評分；
  爬坡格同 r81v6a_climb.png 左格（−15～+2 dB、事件標註）。
--history all：b01 起主線各段（v2 → v3 → v4 → v4a → v4 → v4x → v6 → v6a）以當時規格評分、
  換規格即重新起算、底色＝規格段（v5／v5a 旁支、公證、掃描、填空池不列）。

用法：python -m script.figs.r81_best_gif <輸出資料夾> [--ms 200] [--history all]
輸出：r81v6a_best_evolution.gif
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from script import r81_curve_sm as sm
from script.figs import r81_progress as rp
from script.figs.r81_v6 import DOT, LINE, MG_AFTER, V6, V6A, climb_panel, rows_v6

PANELS = (("S21", 1), ("S11", 0), ("S22", 2))
LO = -15
STAGES = {                        # 名稱前綴 → (標籤, config)；r81v6b 依批號再分 v6／v6a
    "r81v2b": ("v2 阻帶 −15", "dual_r81_s21_v2"),
    "r81v3b": ("v3 阻帶 −20", "dual_r81_s21_v3"),
    "r81v4b": ("v4 加 S11/S22", "dual_r81_v4"),
    "r81v4ab": ("v4a −15 版", "dual_r81_v4a"),
    "r81v4xb": ("v4x 量到 18 GHz", "dual_r81_v4x"),
    "v6": ("v6 量到 11 GHz", "dual_r81_v6"),
    "v6a": ("v6a −15 版", "dual_r81_v6a"),
}
SHADE = ("#eef3fb", "#fbf3ea")


def stage_of(name):
    """store 名（去掉 dedust_ 與分片）→ (排序位置, 階段鍵)；不列入的回 None。"""
    m = re.fullmatch(r"(r81v2b|r81v3b|r81v4ab|r81v4xb|r81v4b)(\d+)", name)
    if m:
        return int(m.group(2)), m.group(1)
    m = re.fullmatch(r"r81v6b(\d+)", name)
    if m:
        b = int(m.group(1))
        return b, "v6" if b < 96 else "v6a"
    if name == "r81v6m02":
        return 61.5, "v6"
    m = re.fullmatch(r"r81v6mg(\d+)", name)
    if m:
        return MG_AFTER.get(int(m.group(1)), 999) + 0.5, "v6a"
    return None


def load_history(grid):
    stores = {}
    for d in sm.V2_DATASET.glob("dedust_r81*"):
        m = re.fullmatch(r"dedust_(r81\w+?)s\d+", d.name)
        if m and (d / "results.json").exists() and stage_of(m.group(1)):
            stores.setdefault(m.group(1), []).append(d)
    cfgs = {k: sm.load_cfg(sm.REPO / "configs" / f"{c}.yaml") for k, (_, c) in STAGES.items()}
    rows = []
    for name in sorted(stores, key=lambda n: (stage_of(n)[0], n)):
        pos, st = stage_of(name)
        for d in sorted(stores[name]):
            for r in sm.load_store(d, grid):
                sc = sm.true_wm(r, cfgs[st], grid)
                if sc:
                    rows.append({"r": r, "sc": sc, "stage": st})
    return rows, cfgs


def curve_panel(ax, grid, row, li, label, cfg, margins):
    for b in cfg.score_spec["bands"]:
        if b["label"] != label:
            continue
        lo, hi, thr = b["start_ghz"], b["stop_ghz"], b["threshold"]
        ax.axvspan(lo, hi, color="#f6dede" if b["relation"] == "le" else "#dff0e3", alpha=0.6, lw=0)
        ax.plot([lo, hi], [thr, thr], color=rp.INK, lw=1.3, ls="--")
    y = np.where(row["mask"][li], row["y"][li], np.nan)
    ax.plot(grid, y, color=LINE, lw=2.0)
    ax.set_xlim(11, 40)
    ax.set_ylim(-45, 2)
    ax.set_xticks([11, 15, 19, 22, 26, 30, 36, 40])
    ax.set_yticks([-40, -30, -20, -10, 0])
    ax.set_ylabel(f"{label} (dB)", color=rp.MUTED, fontsize=9)
    names = {"s21_pass": "通帶", "s21_low_stop": "低阻", "s21_high_stop": "高阻"}
    if label == "S21":
        t = "S21：" + "、".join(f"{names.get(k, k)} {v:+.2f}" for k, v in margins.items() if k.startswith("s21"))
    else:
        v = margins.get(label.lower() + "_match")
        t = f"{label}：匹配 {v:+.2f}" if v is not None else f"{label}：（此規格不考核）"
    ax.set_title(t, loc="left", fontsize=9.5, color=rp.INK)
    rp.style(ax)
    ax.set_box_aspect(1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--ms", type=int, default=200, help="每格毫秒")
    ap.add_argument("--hold", type=int, default=12, help="最後一格多停幾格")
    ap.add_argument("--history", choices=("v6a", "all"), default="v6a")
    args = ap.parse_args(argv)
    plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
    grid = np.asarray(sm.frequency_grid(sm.load_cfg(V6).measurement))
    if args.history == "all":
        rows, cfgs = load_history(grid)
        batches = None
    else:
        cfgs = {"v6a": sm.load_cfg(V6A)}
        rows = [{"r": r, "sc": sm.true_wm(r, cfgs["v6a"], grid), "stage": "v6a"} for r in rows_v6(sm.load_cfg(V6), grid)]
        batches = [x["r"]["batch"] for x in rows]
    w = np.array([x["sc"]["worst_margin"] for x in rows])
    st = [x["stage"] for x in rows]
    xs = np.arange(1, len(rows) + 1)
    cum = np.empty(len(rows))
    frames, segs, best = [], [], -np.inf
    for i in range(len(rows)):
        if i == 0 or st[i] != st[i - 1]:
            best = -np.inf
            segs.append([i, i, st[i]])
        segs[-1][1] = i
        if w[i] > best:
            best = w[i]
            frames.append(i)
        cum[i] = best
    print(f"{len(rows)} 筆、{len(segs)} 段、{len(frames)} 格；最後 {cum[-1]:+.3f}（{st[-1]}）")
    seq = frames + [frames[-1]] * args.hold
    fig = plt.figure(figsize=(17, 9.2), dpi=80)
    fig.patch.set_facecolor(rp.SURF)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 2])
    curve_axes = {"S21": fig.add_subplot(gs[0, 0]), "S11": fig.add_subplot(gs[0, 1]), "S22": fig.add_subplot(gs[1, 0])}
    ax_p = fig.add_subplot(gs[1, 1])
    ax_c = fig.add_subplot(gs[:, 2])

    def draw_stages(n, i):
        for j, (a, b, s) in enumerate(segs):
            ax_c.axvspan(xs[a] - 0.5, xs[b] + 0.5, color=SHADE[j % 2], lw=0)
            if xs[b] - xs[a] > 45:
                ax_c.text((xs[a] + xs[b]) / 2, 0.75, STAGES[s][0], ha="center", va="bottom", fontsize=8.5,
                          color=rp.MUTED, rotation=90)
        ax_c.scatter(xs[:n], np.clip(w[:n], LO, None), s=6, color=DOT, lw=0)
        for a, b, s in segs:
            if a < n:
                e = min(b, i) + 1
                ax_c.step(xs[a:e], cum[a:e], where="post", color=LINE, lw=2.2)
        ax_c.scatter([n], [cum[i]], s=70, color="#e0322b", zorder=5)
        ax_c.axhline(0, color=rp.INK, lw=1.2, ls="--")
        ax_c.set_ylim(LO, 3.5)
        ax_c.set_yticks([-15, -10, -5, 0])
        ax_c.set_xlim(0, xs[-1] + 10)
        ax_c.set_xlabel("累計 HFSS 模擬筆數（主線）", color=rp.MUTED)
        ax_c.set_ylabel("WM（各段當時規格，dB）", color=rp.MUTED)
        ax_c.set_title(f"爬坡：{STAGES[st[i]][0]} 段累計最佳 {cum[i]:+.2f} dB（換規格即重新起算；低於 −15 壓底）",
                       loc="left", fontsize=10.5, color=rp.INK)
        rp.style(ax_c)
        ax_c.set_box_aspect(1)

    def draw(k):
        i = seq[k]
        row, sc, stg = rows[i]["r"], rows[i]["sc"], st[i]
        for label, li in PANELS:
            ax = curve_axes[label]
            ax.clear()
            curve_panel(ax, grid, row, li, label, cfgs[stg], sc["margins"])
        curve_axes["S22"].set_xlabel("頻率 (GHz)", color=rp.MUTED, fontsize=9)
        ax_p.clear()
        ax_p.imshow(np.asarray(row["pattern"]).reshape(25, 25) > 0.5, cmap="Greys", origin="upper",
                    interpolation="nearest", vmin=0, vmax=1)
        ax_p.set_xticks([]); ax_p.set_yticks([])
        ax_p.set_title("pattern（25×25，上下緣中央為兩埠饋墊）", loc="left", fontsize=9.5, color=rp.INK)
        ax_p.set_box_aspect(1)
        ax_c.clear()
        n = i + 1
        if batches is not None:
            climb_panel(ax_c, xs[:n], w[:n], cum[:n], batches[:n], LO, 2, xmax=xs[-1])
            ax_c.scatter([n], [cum[i]], s=70, color="#e0322b", zorder=5)
            ax_c.set_yticks(range(-14, 3, 2))
            ax_c.legend(loc="upper right", frameon=False, fontsize=9)
            ax_c.set_title("濾波器 −15 版（v6a）性能爬升（阻帶 11–19、36–40 ≤ −15 dB）", loc="left", fontsize=11,
                           color=rp.INK)
        else:
            draw_stages(n, i)
        ww = sc["worst_margin"]
        head = "R81 濾波器 −15 版最佳演進" if batches is not None else f"R81 濾波器全程最佳演進　{STAGES[stg][0]}"
        fig.suptitle(f"{head}　第 {n} 筆模擬　WM {ww:+.3f} dB　（{min(k + 1, len(frames))}／{len(frames)}）\n{row['id']}",
                     x=0.03, ha="left", fontsize=12, color="#1a7f37" if ww >= 0 else rp.INK)
        fig.subplots_adjust(left=0.05, right=0.98, top=0.89, bottom=0.06, wspace=0.25, hspace=0.25)

    anim = FuncAnimation(fig, draw, frames=len(seq), interval=args.ms)
    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / "r81v6a_best_evolution.gif"
    anim.save(out, writer=PillowWriter(fps=1000 / args.ms))
    plt.close(fig)
    print("saved", out, out.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
