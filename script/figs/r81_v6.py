"""R81 學長 v6 規格（11–40 GHz，阻帶 ≤ −20 dB）兩張圖：性能爬升、目前最佳設計卡片。

用法：python -m script.figs.r81_v6 <輸出資料夾> [--id r81v2_b71_003_f8fd2069]
輸出：r81v6_climb.png、r81v6a_climb.png（−15 版尺回算全部）、r81v6_best_card.png、r81v6a_best_card.png（−15 版最佳，各格 margin 用 −15 版；各格正方形）。
資料：v2_v001 資料根中 11–40 GHz 量測的 store（補量 r81v6m02、b62 起各批、R89 整合批 mgNN），依批號排序；重測（~rep）不計入。
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
EVENTS = [(62, "斷開＋大步變異 30 格：壓掉 13.5 GHz 峰"), (67, "饋墊附近細步"), (71, "12 格＋前 10 名起點"),
          (74, "交界推移"), (76, "任意切縫重切"), (78, "取消選樣放寬"), (85, "變化量模型試用"), (89, "利用臂改用變化量模型"),
          (92, "爬山全交變化量模型"), (96, "目標改 −15 版＋探索臂改用變化量模型"), (101, "兩模型平均打分"),
          (107, "爬山改上下文模型＋利用臂改平均"), (112, "利用／探索臂小步限制"),
          (1001, "整合移植（R89）"), (1002, "疊加實測有效小步"), (1006, "通帶 30 GHz 專攻"), (1013, "擾動後修復＋預估模式"),
          (1032, "通帶雙緣成對翻轉（R97）"), (1034, "成對翻轉移植（R98）")]
MG_AFTER = {1: 112, 2: 113, 3: 113, 4: 114, 5: 114, 6: 114, 7: 114, 8: 114, 9: 114, 10: 114, 11: 114, 12: 114,
            13: 114, 14: 115, 15: 115, 16: 115, 17: 115, 18: 115, 19: 115}   # 整合批 mgNN 實際量測時段接在哪一批之後
NOTARIZED = {                     # v6a 跨機公證通過（round-89／90）；圖卡優先挑這些，標題註明
    "r81v6mg05_009_59eb9c0e": "216＝218", "r81v6mg06_002_fc2c5910": "216＝218", "r81v6mg07_003_329c2f14": "216＝218",
    "r81v6mg08_018_41808495": "216＝218", "r81v6mg13_019_32a4c2ad": "216＝37", "r81v6mg13_017_e23e29e6": "216＝37",
    "r81v6mg14_015_00661532": "216＝37", "r81v6mg15_005_c943952c": "216＝37",
    "r81v6mg18_011_9c63dfbd": "37＝216", "r81v6mg20_011_04c84008": "216＝218",
    "r81v6mg23_013_70f5551f": "218＝216",
    "r81v6mg23_018_d4347f75": "216＝37", "r81v6mg25_011_b1bf6978": "218＝216",
    "r81v6mg26_006_091f57e2": "216＝218", "r81v6mg28_012_5734eceb": "216＝218＝37",
    "r81v6mg30_005_375c13b7": "216＝37", "r81v6mg31_021_dddcbc7c": "218＝216",
    "r81v6mg32_011_733aebe5": "37＝216", "r81v6mg32_006_677c9758": "216＝37＝218",
    "r81v6mg34_000_b4bb9425": "37＝216＝218", "r81v6mg33_023_92bc2c9b": "218＝37",
    "r81v6mg34_001_cbf6d2ad": "216＝218", "r81v6mg36_003_72976679": "216＝37",
    "r81v6mg36_001_2f110a48": "218＝216＝37", "r81v6mg37_000_92ff9a8b": "37＝216＝218",
    "r81v6mg39_010_e496b1c7": "37＝216＝218", "r81v6mg41_007_3681998f": "216＝37"}


def rows_v6(cfg, grid):
    """11–40 GHz store 的全部 ok 實測，依（批號, store, id）排序；同圖只留第一筆。"""
    stores = [s for s in sm.V2_DATASET.glob("dedust_r81v6*s?")
              if re.search(r"(m02|b\d+|mg\d+)s\d$", s.name) and (s / "results.json").exists()]

    def key(s):
        """排序位置與批號：整合批 mgNN 排在 MG_AFTER 那一批之後，批號記成 1000＋NN（事件標註用）。"""
        if "m02" in s.name:
            return 0, 0
        m = re.search(r"mg(\d+)s\d$", s.name)
        if m:
            return MG_AFTER.get(int(m.group(1)), 999) + 0.5, 1000 + int(m.group(1))
        b = int(re.search(r"b(\d+)s\d$", s.name).group(1))
        return b, b

    rows, seen = [], set()
    for s in sorted(stores, key=lambda s: (key(s)[0], s.name)):
        for r in sm.load_store(s, grid):
            if "~rep" in r["id"] or r["sha"] in seen:
                continue
            sc = sm.true_wm(r, cfg, grid)
            if sc:
                seen.add(r["sha"])
                r.update(batch=key(s)[1], wm=sc["worst_margin"], margins=sc["margins"])
                rows.append(r)
    return rows


def climb_panel(ax, x, wm, best, batches, lo, hi, label_events=True, events_from=0, text_right=False, xmax=None):
    """xmax：固定右界（動畫逐格畫時用，事件間距與標字位移也以整段為準）。"""
    span = (xmax if xmax is not None else x[-1]) - x[0]
    ax.scatter(x, np.clip(wm, lo, None), s=8, color=DOT, lw=0, label="每筆模擬")
    ax.step(x, best, where="post", color=LINE, lw=2.2, label=f"累計最佳（目前 {best[-1]:+.2f}，單次）")
    ax.axhline(0, color=rp.INK, lw=1.2, ls="--")
    ax.text(x[0] + 3, (hi - lo) * 0.015, "達標 WM = 0", fontsize=9, color=rp.INK, va="bottom")
    last = -np.inf
    for b, text in EVENTS:
        if b in batches and b >= events_from:
            k = batches.index(b)
            ax.axvline(x[k], color=MARK, lw=0.8, ls=":")
            if label_events and x[k] - last >= span * 0.025:   # 太近的事件只畫線不寫字，免得疊字
                last = x[k]
                off = span * 0.006 * (1 if text_right else -1)
                ax.text(x[k] + off, lo + (hi - lo) * 0.02, text, rotation=90, ha="left" if text_right else "right",
                        va="bottom", fontsize=8.5, color=MARK, bbox=dict(facecolor=rp.SURF, edgecolor="none", pad=1.5))
    ax.set_xlim(x[0] - 1, (xmax if xmax is not None else x[-1]) + 4)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("累計 HFSS 模擬筆數（11–40 GHz）", color=rp.MUTED)
    ax.set_ylabel("WM (dB)", color=rp.MUTED)
    rp.style(ax)
    ax.set_box_aspect(1)


def climb_figure(out, rows, key="wm", lo=-20, title="濾波器 v6 性能爬升（阻帶 11–19、36–40 ≤ −20 dB）", note="",
                 zoom=None):
    """zoom＝(起始批號, 下限 dB)：右側加一格放大最近一段（各格正方形）。"""
    wm = np.array([r[key] for r in rows])
    best = np.maximum.accumulate(wm)
    x = np.arange(1, len(wm) + 1)
    batches = [r["batch"] for r in rows]
    fig, axes = plt.subplots(1, 2 if zoom else 1, figsize=(13.5, 7) if zoom else (7, 7), dpi=160, squeeze=False)
    fig.patch.set_facecolor(rp.SURF)
    ax = axes[0, 0]
    climb_panel(ax, x, wm, best, batches, lo, 2)
    ax.set_title(title, loc="left", fontsize=11.5, color=rp.INK)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    if zoom:
        b0, zlo = zoom
        k = next(i for i, b in enumerate(batches) if b >= b0)
        az = axes[0, 1]
        climb_panel(az, x[k:], wm[k:], best[k:], batches[k:], zlo, max(0.15, best[-1] + 0.06), events_from=b0,
                    text_right=True)
        az.set_title(f"放大：b{b0} 起（{len(wm) - k} 筆），{best[k]:+.2f} → {best[-1]:+.2f}", loc="left",
                     fontsize=11.5, color=rp.INK)
        ax.axvspan(x[k], x[-1], color=LINE, alpha=0.06, lw=0)
    fig.text(0.02, 0.01, f"共 {len(wm)} 筆單次 HFSS；前 24 筆為 18–40 好設計以 11–40 重量；低於下限的點壓在底線。{note}",
             fontsize=8, color=rp.MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
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


def card_figure(out, r, grid, v6, v6a, n, spec="v6"):
    """spec＝v6：標題與各格 margin 用 −20 版；spec＝v6a：改用 −15 版（給學長看 −15 達標進度）。"""
    sa = sm.true_wm(r, v6a, grid)
    wma = sa["worst_margin"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 10.6), dpi=160)
    fig.patch.set_facecolor(rp.SURF)
    mg = sa["margins"] if spec == "v6a" else r["margins"]
    for ax, (k, lab) in zip((axes[0, 0], axes[0, 1], axes[1, 0]), ((1, "S21"), (0, "S11"), (2, "S22"))):
        curve(ax, grid, r["y"][k], lab, v6, v6a, mg)
    px = axes[1, 1]
    px.imshow(r["pattern"].reshape(25, 25), cmap="Greys", origin="upper", interpolation="nearest")
    px.set_xticks([]); px.set_yticks([])
    px.set_title("pattern（25×25，上下緣中央為兩埠饋墊，兩埠金屬不相連）", loc="left", fontsize=10, color=rp.INK)
    px.set_xlabel(r["id"], fontsize=8.5, color=rp.MUTED)
    if spec == "v6a":
        tag = f"已公證（{NOTARIZED[r['id']]} 兩機一致）" if r["id"] in NOTARIZED else "單次未公證"
        title = f"濾波器 −15 版（v6a）目前最佳：WM {wma:+.3f} dB，{tag}；同一設計在 −20 版為 {r['wm']:+.2f} dB"
    else:
        title = f"濾波器 v6 目前最佳：WM {r['wm']:+.2f} dB（−20 版）、{wma:+.2f} dB（−15 版），單次未公證"
    fig.suptitle(title, x=0.03, ha="left", fontsize=12.5, color=rp.INK)
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
    for r in rows:
        r["wm_a"] = sm.true_wm(r, v6a, grid)["worst_margin"]
    climb_figure(args.outdir / "r81v6a_climb.png", rows, key="wm_a", lo=-15,
                 title="濾波器 −15 版（v6a）性能爬升（阻帶 11–19、36–40 ≤ −15 dB）",
                 note="\nb95 以前選樣以 −20 版為目標、b96 起改以 −15 版為目標；此圖全部以 −15 版尺回算。",
                 zoom=(96, -1.5))
    best = next(r for r in rows if r["id"] == args.id) if args.id else max(rows, key=lambda r: r["wm"])
    card_figure(args.outdir / "r81v6_best_card.png", best, grid, v6, v6a, len(rows))
    pool = [r for r in rows if r["id"] in NOTARIZED] or rows        # 有公證設計就從公證的挑，免得挑到單次離群值
    best_a = max(pool, key=lambda r: r["wm_a"])
    card_figure(args.outdir / "r81v6a_best_card.png", best_a, grid, v6, v6a, len(rows), spec="v6a")


if __name__ == "__main__":
    main()
