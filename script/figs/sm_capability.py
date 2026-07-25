"""SM 能力量化面板（2026-07-25 Ricky「畫圖說明 SM 能力:held-out/收斂/前瞻/輸出多樣性」）。

四面板：①held-out 誤差軌跡（docs/kpi.csv:中位/P90/far 域,版本軸）②影子對決批前瞻
（two vs mlp 盲測 ρ,數字源=round-39/40 檔 analyze 三模段,硬編於下）③輸出 response
多樣性趨勢（近六批:批內 response 兩兩距中位+對鍋 NN 距中位,PCA 2D 口徑,排除自身洩漏）
④response PCA 覆蓋疊圖（鍋=灰,R40 三批著色）。

用法: python -m script.figs.sm_capability [--out docs/log/assets/round-41/sm_capability.png]
"""
import argparse
import csv
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from scipy.spatial import cKDTree  # noqa: E402
from antenna.utils import DATASET_PATH  # noqa: E402
from antenna.utils.store import SampleStore  # noqa: E402
from script.sm_reanchor import _load_clean  # noqa: E402

# 影子對決盲測（analyze batch 三模段,帳=round-38/39/40 檔;two 自 R38b1 起有數）
DUEL = [  # (批, mlp_err, mlp_rho, two_err, two_rho)
    ("r38b1", 2.31, 0.334, 2.00, 0.79), ("r38b2", 2.60, 0.269, 2.14, 0.76),
    ("r38b3", 2.44, 0.309, 2.18, 0.72), ("r39b2", 2.01, 0.077, 1.52, 0.786),
    ("r39b3", 1.46, 0.182, 1.32, 0.758), ("r40b1", 2.25, 0.187, 1.52, 0.820),
    ("r40b2", 2.09, 0.207, 1.67, 0.798), ("r40b3", 2.16, 0.145, 1.51, 0.856),
]
BATCHES = ["r39b1", "r39b2", "r39b3", "r40b1", "r40b2", "r40b3"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("docs", "log", "assets", "round-41", "sm_capability.png"))
    args = ap.parse_args()

    rows = list(csv.DictReader(open(os.path.join(REPO, "docs", "kpi.csv"), encoding="utf-8")))
    ver = [int("".join(c for c in r["sm"] if c.isdigit())) for r in rows]
    med = [float(r["wm_err_med"]) for r in rows]
    p90 = [float(r["wm_err_p90"]) for r in rows]
    far = [float(r["err_far"]) for r in rows]

    tr, ho = _load_clean()
    items = tr + ho
    Y = np.stack([np.asarray(y).ravel() for _, y in items]).astype(np.float64)
    keys = [np.asarray(x).tobytes() for x, _ in items]
    mu = Y.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(Y - mu, full_matrices=False)
    pc_all = (Y - mu) @ vt[:2].T
    key2i = {k: i for i, k in enumerate(keys)}

    div_in, div_nn, ok_b = [], [], []
    bpcs = {}
    for b in BATCHES:
        pcs, exc = [], set()
        for suf in "ab":
            sp = DATASET_PATH.joinpath(f"dedust_{b}{suf}")
            if not sp.is_dir():
                continue
            st = SampleStore(sp, verbose=False)
            for i in range(len(st)):
                x, y = st[i]
                pcs.append((np.asarray(y).ravel().astype(np.float64) - mu[0]) @ vt[:2].T)
                j = key2i.get(np.asarray(x).tobytes())
                if j is not None:
                    exc.add(j)
        if not pcs:
            continue
        pcs = np.stack(pcs)
        bpcs[b] = pcs
        mask = np.ones(len(pc_all), bool)
        mask[list(exc)] = False
        tree = cKDTree(pc_all[mask])
        d = tree.query(pcs)[0]
        pw = np.sqrt(((pcs[:, None] - pcs[None]) ** 2).sum(-1))
        div_in.append(float(np.median(pw[np.triu_indices(len(pcs), 1)])))
        div_nn.append(float(np.median(d)))
        ok_b.append(b)
        print(f"{b}: 批內兩兩距中位 {div_in[-1]:.1f} | 對鍋 NN 距中位 {div_nn[-1]:.2f}（n={len(pcs)}）")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(2, 2, figsize=(15, 11))

    a = ax[0][0]
    a.plot(ver, med, "o-", label="held-out |wm err| 中位")
    a.plot(ver, p90, "s--", alpha=0.6, label="P90")
    a.plot(ver, far, "^-", alpha=0.8, label="far 域（d_dyn 遠端）")
    a.set_xlabel("SM 版本（sm_reanchorNN）")
    a.set_ylabel("|pred − real| (dB)")
    a.set_title("① MLP 主錨 held-out 誤差軌跡")
    a.legend(fontsize=9)
    a.grid(alpha=0.3)

    a = ax[0][1]
    xs = range(len(DUEL))
    a.plot(xs, [d[2] for d in DUEL], "o--", color="#888", label="mlp 前瞻ρ")
    a.plot(xs, [d[4] for d in DUEL], "o-", color="#1f77b4", label="two(cnn2) 前瞻ρ")
    a2 = a.twinx()
    a2.plot(xs, [d[1] for d in DUEL], "s:", color="#bbb", alpha=0.7, label="mlp |err| 中位")
    a2.plot(xs, [d[3] for d in DUEL], "s-", color="#ff7f0e", alpha=0.7, label="two |err| 中位")
    a2.set_ylabel("盲測 |err| 中位 (dB)")
    a.set_xticks(list(xs))
    a.set_xticklabels([d[0] for d in DUEL], rotation=45, fontsize=8)
    a.set_ylabel("批前瞻 Spearman ρ")
    a.set_title("② 盲測對決:two vs mlp（收檔前預測 × 實測）")
    a.legend(loc="center left", fontsize=9)
    a2.legend(loc="lower right", fontsize=8)
    a.grid(alpha=0.3)

    a = ax[1][0]
    xs = range(len(ok_b))
    a.plot(xs, div_in, "o-", label="批內 response 兩兩距中位（發散度）")
    a.plot(xs, div_nn, "s-", label="對鍋 NN 距中位（新穎度,排除自身）")
    a.set_xticks(list(xs))
    a.set_xticklabels(ok_b, rotation=45, fontsize=8)
    a.set_ylabel("response PCA 距離")
    a.set_title("③ 輸出 response 多樣性（近六批）")
    a.legend(fontsize=9)
    a.grid(alpha=0.3)

    a = ax[1][1]
    a.scatter(pc_all[:, 0], pc_all[:, 1], c="#cccccc", s=3, alpha=0.3, lw=0, label=f"鍋（{len(pc_all)}）")
    for b, col in [("r40b1", "#1f77b4"), ("r40b2", "#ff7f0e"), ("r40b3", "#2ca02c")]:
        if b in bpcs:
            a.scatter(bpcs[b][:, 0], bpcs[b][:, 1], c=col, s=14, alpha=0.85, lw=0, label=b)
    a.set_xlabel("PC1")
    a.set_ylabel("PC2")
    a.set_title("④ R40 三批實測 response 落點 vs 鍋")
    a.legend(fontsize=9)
    fig.suptitle("SM 能力面板（2026-07-25;①=docs/kpi.csv ②=round 檔 analyze 三模段 ③④=實測 response）", y=0.995)
    fig.tight_layout()
    outp = os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    fig.savefig(outp, dpi=130, bbox_inches="tight")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
