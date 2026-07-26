"""帕累托前緣圖（2026-07-26 Ricky「畫一個帕累托前沿」）。

雙面板:①主戰場 wm×oob_bad（rad 達標者著色,2D 前緣線,紀錄點標註）
②左側戰場 wm×lo（合格圈+左側合格解五筆星標,usable_lo 紀錄線）。
資料=script.analyze._load_truths（全 store 去重真值,零 HFSS）。

用法: python -m script.figs.pareto_front [--out docs/log/assets/round-43/pareto.png]
"""
import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from script.analyze import _load_truths  # noqa: E402
from script.dedust import oob_metrics  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("docs", "log", "assets", "round-43", "pareto.png"))
    args = ap.parse_args()

    rows = []
    for p, wm, rad, resp in _load_truths():
        if resp is None or wm is None:
            continue
        m = oob_metrics(np.asarray(resp))
        rows.append((float(wm), (float(rad) if rad is not None else None),
                     m["oob_bad"], m.get("oob_gain_max_lo")))
    wm = np.array([r[0] for r in rows])
    rad = np.array([(r[1] if r[1] is not None else np.nan) for r in rows])
    oob = np.array([r[2] for r in rows], float)
    lo = np.array([(r[3] if r[3] is not None else np.nan) for r in rows], float)
    print(f"真值 {len(rows)} 筆")
    radok = ~np.isnan(rad) & (rad >= 0)

    def front2d(xs, ys):
        """(max x, min y) 2D 非支配前緣（回傳排序後的點索引）。"""
        idx = np.argsort(-xs)
        out, best_y = [], np.inf
        for i in idx:
            if ys[i] < best_y:
                out.append(i)
                best_y = ys[i]
        return out[::-1]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))

    a = ax[0]
    sel = wm >= -2
    a.scatter(wm[sel & ~radok], oob[sel & ~radok], s=5, c="#c5c5c5", alpha=0.35, lw=0, label="rad 未達標")
    a.scatter(wm[sel & radok], oob[sel & radok], s=8, c="#1f77b4", alpha=0.55, lw=0, label="rad ≥0")
    fi = front2d(np.where(radok, wm, -99), np.where(radok, oob, 99))
    a.plot(wm[fi], oob[fi], "r-o", ms=4, lw=1.4, label="前緣（rad 達標）")
    a.axvline(0.15, color="gray", ls=":", lw=1)
    a.annotate("margin 王 +0.73", (0.73, 11.63), xytext=(0.30, 16),
               arrowprops=dict(arrowstyle="->", color="black"), fontsize=9)
    a.annotate("usable_oob 王 7.78", (0.15, 7.78), xytext=(-1.5, 5.5),
               arrowprops=dict(arrowstyle="->", color="black"), fontsize=9)
    a.set_xlabel("wm（帶內裕度,右=好）")
    a.set_ylabel("oob_bad（帶外,低=好）")
    a.set_title("① 主戰場:wm × 帶外（作戰區 wm≥−2）")
    a.legend(fontsize=9)
    a.grid(alpha=0.3)

    a = ax[1]
    sel = wm >= -2
    a.scatter(wm[sel & ~radok], lo[sel & ~radok], s=5, c="#c5c5c5", alpha=0.35, lw=0)
    a.scatter(wm[sel & radok], lo[sel & radok], s=8, c="#1f77b4", alpha=0.55, lw=0)
    q = radok & (wm >= 0.15)
    lq = q & (lo <= -2)
    a.scatter(wm[q & ~lq], lo[q & ~lq], s=22, facecolors="none", edgecolors="#ff7f0e",
              lw=1.0, label=f"合格 wm≥0.15∧rad≥0（{int(q.sum())}）")
    a.scatter(wm[lq], lo[lq], s=90, marker="*", c="red", label=f"左側合格 lo≤−2（{int(lq.sum())}）")
    a.axhline(-2, color="gray", ls=":", lw=1)
    a.axhline(-3.46, color="red", ls="--", lw=1)
    a.text(-1.95, -3.42, "usable_lo 紀錄 −3.46", fontsize=9, color="red")
    fi2 = front2d(np.where(q, wm, -99), np.where(q & ~np.isnan(lo), lo, 99))
    a.plot(wm[fi2], lo[fi2], "r-", lw=1.2, alpha=0.7, label="合格圈左側前緣")
    a.set_xlabel("wm（帶內裕度）")
    a.set_ylabel("oob_gain_max_lo（左側帶外,低=好）")
    a.set_title("② 左側戰場:wm × lo（星=左側合格解含公證複測）")
    a.legend(fontsize=9, loc="upper left")
    a.grid(alpha=0.3)

    fig.suptitle(f"帕累托前緣（全史真值 n={len(rows)};2026-07-26）", y=0.99)
    fig.tight_layout()
    outp = os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    fig.savefig(outp, dpi=130, bbox_inches="tight")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
