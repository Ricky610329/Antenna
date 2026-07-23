"""response 空間 PCA 地圖（S11+Gain 曲線,rad 不進;2026-07-23 Ricky「用 SM 的資料畫 response PCA」）。

背景=左右側拆帳制/response 多樣性訓 SM 提案（decisions/scratch 2026-07-23）:
pattern 多樣性≠response 覆蓋——本圖直接檢驗「882 合格解在 response 空間是同一聚落」。
資料=sm_reanchor._load_clean 同鍋（乾淨真值全集,去重）;每點=一筆 (S11||Gain) 攤平向量;
PCA=numpy SVD;三面板:①wm 著色（合格圈註記）②左側 lo 著色 ③右側 hi 著色。

用法: python -m script.figs.response_map [--out docs/log/assets/round-36/response_pca.png] [--max-n 0]
"""
import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from script.sm_reanchor import _load_clean, LABELS, _cfg  # noqa: E402（同鍋口徑,座標/spec 已裝）
from script.dedust import oob_metrics  # noqa: E402
from antenna.losses import worst_margin  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("docs", "log", "assets", "round-36", "response_pca.png"))
    ap.add_argument("--max-n", type=int, default=0, dest="max_n", help="抽樣上限（0=全量）")
    args = ap.parse_args()

    tr, ho = _load_clean()
    items = tr + ho
    if args.max_n and len(items) > args.max_n:
        idx = np.random.default_rng(0).choice(len(items), size=args.max_n, replace=False)
        items = [items[i] for i in idx]
    print(f"樣本 {len(items)} 筆（乾淨真值同鍋）")

    Y, wm, lo, hi = [], [], [], []
    for _, y in items:
        yy = y.reshape(len(LABELS), -1)
        Y.append(yy.numpy().ravel())
        w, _ = worst_margin(yy, LABELS, _cfg.targets)
        m = oob_metrics(yy.numpy())
        wm.append(float(w))
        lo.append(m.get("oob_gain_max_lo", np.nan))
        hi.append(m.get("oob_gain_max_hi", np.nan))
    Y = np.asarray(Y, dtype=np.float64)
    wm, lo, hi = map(np.asarray, (wm, lo, hi))

    Yc = Y - Y.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(Yc, full_matrices=False)
    pc = Yc @ vt[:2].T
    evr = (s ** 2 / (s ** 2).sum())[:2]
    q = wm >= 0.15  # 合格門檻的 wm 半邊（rad 不在 response 內,圖註明）

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))
    panels = [("wm（帶內裕度）", wm, "RdYlGn", (-8, 0.5)),
              ("左側 oob_gain_max_lo（低=好;全史牆 +3.7）", lo, "RdYlGn_r", (-6, 5)),
              ("右側 oob_gain_max_hi（低=好）", hi, "RdYlGn_r", (-7, 4))]
    for ax, (title, c, cmap, clim) in zip(axes, panels):
        sc = ax.scatter(pc[:, 0], pc[:, 1], c=np.clip(c, *clim), cmap=cmap, s=4, alpha=0.45, lw=0)
        ax.scatter(pc[q, 0], pc[q, 1], facecolors="none", edgecolors="black", s=26, lw=0.7,
                   label=f"wm≥0.15（{int(q.sum())} 筆）")
        ax.set_title(title)
        ax.set_xlabel(f"PC1（{evr[0] * 100:.0f}%）")
        ax.set_ylabel(f"PC2（{evr[1] * 100:.0f}%）")
        ax.legend(loc="best", fontsize=9)
        fig.colorbar(sc, ax=ax, shrink=0.85)
    fig.suptitle(f"response 空間 PCA（S11+Gain 攤平,rad 不進;n={len(items)};黑圈=wm 過線族）", y=1.00)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.join(REPO, args.out)), exist_ok=True)
    outp = os.path.join(REPO, args.out)
    fig.savefig(outp, dpi=130, bbox_inches="tight")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
