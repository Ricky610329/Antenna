# -*- coding: utf-8 -*-
"""report_sprint48.py — R28-R29 48 小時衝刺總覽（2026-07-14~15;Ricky「作圖到報告內」）。
四 panel：①margin 王演進（24hr 三連跳）②多樣性換血儀表③adversarial training 閉環
④低側 Gain 誠實面板（Ricky:「很在意左側 Gain 沒壓低」——vs 學長碎片族 gap）。
用法: python -m script.figs.report_sprint48 [--out docs/report/assets/sprint48.png]"""
import argparse
import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
from antenna.utils import DATASET_PATH

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "docs", "report", "assets", "sprint48.png"))
    args = ap.parse_args()
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9))

    # ① margin 王演進
    ax = axes[0, 0]
    kings = [("c21", "07-07", 0.20), ("c25", "07-08", 0.22), ("i02", "07-09", 0.29),
             ("a024", "07-09", 0.35), ("r2_016", "07-11", 0.39), ("k23b1_021", "07-13", 0.41),
             ("m23b4_030", "07-13", 0.49), ("s28b3_005", "07-15", 0.50), ("o29b2_011", "07-15", 0.56)]
    xs = range(len(kings))
    ys = [k[2] for k in kings]
    ax.plot(xs, ys, "o-", color="#1f5fa8", lw=2, ms=6)
    ax.plot([7, 8], [ys[7], ys[8]], "o-", color="crimson", lw=2.5, ms=7, zorder=5)
    for i, (nm, dt, v) in enumerate(kings):
        ax.annotate(f"{nm}\n{v:+.2f}", (i, v), textcoords="offset points",
                    xytext=(0, 9 if i % 2 == 0 else -25), fontsize=7, ha="center")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([k[1] for k in kings], fontsize=7)
    ax.set_ylabel("worst-margin (dB)")
    ax.set_title("① margin 王演進——24hr 三連跳 0.49→0.50→0.56（紅=冷支連兩王,皆公證 3/3）", fontsize=10)
    ax.set_ylim(0.15, 0.62)
    ax.grid(alpha=0.3)

    # ② 多樣性換血
    ax = axes[0, 1]
    batches = ["r28b1\n(稅前)", "r28b2\n(加壓稅)", "r28b3", "r29b1\n(G臂)", "r29b2"]
    near = [24, 7, 7, 1, 3]
    blood = [33, 25, 25, 19, 13]
    fresh = [9, 12, 12, 27, 24]
    x = np.arange(len(batches))
    ax.plot(x, near, "o-", label="近王樣本 %（低=好）", color="crimson")
    ax.plot(x, blood, "s-", label="王系血統根 %", color="#c77f2e")
    ax.plot(x, fresh, "^-", label="無親新血 %（高=好）", color="#2e8b57")
    ax.set_xticks(x)
    ax.set_xticklabels(batches, fontsize=8)
    ax.set_ylabel("%")
    ax.set_title("② 多樣性換血（相似度稅+G 臂+新血注入）", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ③ adversarial 閉環
    ax = axes[1, 0]
    bands = ["free", "surg", "champ", "oobp"]
    b1 = [10.66, 12.20, 5.63, 12.62]
    b2 = [10.66, 9.31, 4.00, 10.47]
    x = np.arange(len(bands))
    ax.bar(x - 0.18, b1, 0.36, label="b1（攻 v33）", color="#9db8d9")
    ax.bar(x + 0.18, b2, 0.36, label="b2（攻 v34=吃過 b1 樣本）", color="#1f5fa8")
    for i in range(len(bands)):
        d = b2[i] - b1[i]
        ax.annotate(f"{d:+.1f}" if abs(d) > 0.01 else "±0", (i + 0.18, b2[i] + 0.2),
                    ha="center", fontsize=8, color="crimson" if d < 0 else "gray")
    ax.set_xticks(x)
    ax.set_xticklabels(bands)
    ax.set_ylabel("|pred - real| wm 中位 (dB)")
    ax.set_title("③ adversarial training 閉環——champ 帶收斂+遠帶學會誠實（吹牛樣本 24→0）", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # ④ 低側 Gain 誠實面板（Ricky 心病）：我方三標群 vs 學長碎片族 lo 峰分布
    ax = axes[1, 1]
    lo_ours, lo_king = [], None
    for d in os.listdir(str(DATASET_PATH)):
        rp = DATASET_PATH.joinpath(d, "results.json")
        if not d.startswith("dedust_") or d.endswith("_input") or not rp.exists():
            continue
        for k, v in json.load(open(str(rp), encoding="utf-8")).items():
            if "wm" not in v or "oob_gain_max_lo" not in v:
                continue
            if v["wm"][2] >= 0 and (v.get("rad_margin") or -9) >= 0:
                lo_ours.append(v["oob_gain_max_lo"])
            if k == "o29b2_011_o26b2_007_o2":
                lo_king = v["oob_gain_max_lo"]
    #? 學長原版五錨=舊 store（r7/r9 時代 results 無 oob 欄）——從 store 響應配對重算
    from antenna.utils.store import SampleStore
    from script.dedust import oob_metrics
    import torch
    seniors = {}
    for st, fol, ids in (("dedust_r9", "dedust_r9_input", ("t03_top", "t07_top", "t09_top", "n09_near")),
                         ("dedust_r7", "dedust_r7_input", ("p00_orig",))):
        smap = {}
        sto = SampleStore(DATASET_PATH.joinpath(st), verbose=False)
        for k in range(len(sto)):
            x, y = sto[k]
            smap[(np.asarray(x).reshape(-1) > 0.5).tobytes()] = np.asarray(y).reshape(2, -1)
        for i in ids:
            f = DATASET_PATH.joinpath(fol, i + ".pt")
            if f.exists():
                p = np.asarray(torch.load(str(f), weights_only=True)).reshape(-1) > 0.5
                resp = smap.get(p.tobytes())
                if resp is not None:
                    seniors[i[:3]] = oob_metrics(resp)["oob_gain_max_lo"]
    ax.hist(lo_ours, bins=30, color="#9db8d9", alpha=0.85, label=f"我方三標群（n={len(lo_ours)}）")
    if lo_king is not None:
        ax.axvline(lo_king, color="crimson", lw=2, label=f"新王 o29b2_011（{lo_king:+.1f}）")
    for j, (nm, v) in enumerate(sorted(seniors.items(), key=lambda t: t[1] or 0)):
        if v is not None:
            ax.axvline(v, color="#2e8b57", lw=1.2, ls="--")
            ax.annotate(nm, (v, ax.get_ylim()[1] * (0.93 - 0.07 * (j % 3))), fontsize=8,
                        color="#2e8b57", ha="center")
    ax.annotate("gap 6~8 dB\n（未解之地）", (0.5, ax.get_ylim()[1] * 0.5), fontsize=10,
                color="gray", ha="center", style="italic")
    ax.set_xlabel("低側 Gain 峰 (dBi;低=帶外壓得好)")
    ax.set_title("④ 低側 Gain 誠實面板——我方 vs 學長碎片族（綠虛線）:gap 未解,攻堅=oobp 碎片錨+lo-active", fontsize=9.5)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("R28–R29 48 小時衝刺總覽（2026-07-14~15;單次值已標注,紀錄皆公證 3/3）", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
