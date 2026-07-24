"""資料分布地圖：pattern 空間 × response 空間 PCA 對照（2026-07-25 Ricky「想看資料分布」）。

同鍋（sm_reanchor._load_clean 乾淨真值全集,去重）一次算兩個空間：
上排=pattern PCA（625 維二值攤平）;下排=response PCA（S11||Gain 攤平,rad 不進）。
左欄著色=wm（合格圈=黑圈）;右欄著色=王朝表型（dyn_struct,類別色）——
直接對答「pattern 多樣性 vs response 覆蓋是不是同一件事」＋王朝在兩空間各佔哪裡。
色圖用 viridis/類別色,不用紅綠（Ricky 渲染偏好）。

用法: python -m script.figs.data_map [--out docs/log/assets/round-39/data_map.png] [--max-n 0]
"""
import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from script.sm_reanchor import _load_clean, LABELS, _cfg  # noqa: E402
from script.dedust import dyn_struct  # noqa: E402
from antenna.losses import worst_margin  # noqa: E402


def _pca2(mat):
    c = mat - mat.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(c, full_matrices=False)
    return c @ vt[:2].T, (s ** 2 / (s ** 2).sum())[:2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("docs", "log", "assets", "round-39", "data_map.png"))
    ap.add_argument("--max-n", type=int, default=0, dest="max_n", help="抽樣上限（0=全量）")
    args = ap.parse_args()

    tr, ho = _load_clean()
    items = tr + ho
    if args.max_n and len(items) > args.max_n:
        idx = np.random.default_rng(0).choice(len(items), size=args.max_n, replace=False)
        items = [items[i] for i in idx]
    print(f"樣本 {len(items)} 筆（train {len(tr)} / held-out {len(ho)},去重後）")

    P, Y, wm, dyn = [], [], [], []
    for x, y in items:
        px = np.asarray(x).ravel()
        P.append((px > 0.5).astype(np.float64))
        yy = y.reshape(len(LABELS), -1)
        Y.append(yy.numpy().ravel())
        w, _ = worst_margin(yy, LABELS, _cfg.targets)
        wm.append(float(w))
        dyn.append(dyn_struct(px.reshape(25, 25)))
    P, Y = np.asarray(P), np.asarray(Y, dtype=np.float64)
    wm, dyn = np.asarray(wm), np.asarray(dyn)
    q = wm >= 0.15
    print(f"wm≥0.15: {int(q.sum())} 筆 | 王朝表型: {int(dyn.sum())} ({dyn.mean() * 100:.0f}%)")

    pcP, evP = _pca2(P)
    pcY, evY = _pca2(Y)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 2, figsize=(15, 12.5))
    for row, (pc, evr, space) in enumerate([(pcP, evP, "pattern（625 維二值）"),
                                            (pcY, evY, "response（S11||Gain）")]):
        ax = axes[row][0]
        sc = ax.scatter(pc[:, 0], pc[:, 1], c=np.clip(wm, -8, 0.5), cmap="viridis",
                        s=4, alpha=0.4, lw=0)
        ax.scatter(pc[q, 0], pc[q, 1], facecolors="none", edgecolors="red", s=24, lw=0.6,
                   label=f"wm≥0.15（{int(q.sum())} 筆）")
        ax.set_title(f"{space} PCA — 著色 wm")
        ax.legend(loc="best", fontsize=9)
        fig.colorbar(sc, ax=ax, shrink=0.8)

        ax = axes[row][1]
        ax.scatter(pc[~dyn, 0], pc[~dyn, 1], c="#888888", s=4, alpha=0.3, lw=0,
                   label=f"非王朝（{int((~dyn).sum())}）")
        ax.scatter(pc[dyn, 0], pc[dyn, 1], c="#1f77b4", s=5, alpha=0.5, lw=0,
                   label=f"王朝表型（{int(dyn.sum())}）")
        ax.set_title(f"{space} PCA — 王朝表型")
        ax.legend(loc="best", fontsize=9)
        for ax in axes[row]:
            ax.set_xlabel(f"PC1（{evr[0] * 100:.0f}%）")
            ax.set_ylabel(f"PC2（{evr[1] * 100:.0f}%）")
    fig.suptitle(f"資料分布地圖：pattern vs response 空間（同鍋 n={len(items)};紅圈=wm 過線）", y=0.995)
    fig.tight_layout()
    outp = os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    fig.savefig(outp, dpi=130, bbox_inches="tight")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
