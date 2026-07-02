# -*- coding: utf-8 -*-
"""
script/round_report.py — 把一個 round 的多臂結果歸檔成圖 + markdown 數字。

薄編排層,盡量 reuse script/benchmark_vs_random.py(不重寫 worst-margin/曲線邏輯)。產三樣:
  (a) 每臂「最佳 pattern + S11/Gain vs spec」圖 → <out-dir>/<label>_best.png
  (b) worst-margin vs HFSS-call 疊圖(多臂 + random best-of-N) → <out-dir>/benchmark.png
  (c) 可貼進 docs/log/round-NN 的 §4 markdown 數字表 → 印 stdout(無副作用、人來定稿)

只讀 NAS、純離線、開發機可跑(沿用 config.device='cpu')。單埠(worst_margin)專用。用法:
    python -m script.round_report --round 01 \
        --runs single_guided_harvest single_guided_refit_harvest --labels dlf refit \
        --random-store harvest_single_random --at 250
"""
import argparse
import csv
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from antenna.utils import config, logger
config.device = "cpu"

from antenna.training import load_config, PORT_SPECS, setup_responses
from antenna.losses import worst_margin
from script.benchmark_vs_random import _resolve_run, run_curve, random_curve


def _best_row(run_dir, cfg):
    """掃 metrics.csv,回傳 worst_margin 最高(最接近達標)那筆 (epoch, wm, pattern, response)。"""
    labels = PORT_SPECS[cfg.port]["labels"]
    best = None
    with open(run_dir / "metrics.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            h = (row.get("pattern_hash") or "").strip()
            if not h or h == "nan":
                continue
            pt = run_dir / "patterns" / f"{h}.pt"
            if not pt.exists():
                continue
            patt, resp, _loss = torch.load(str(pt), weights_only=True)
            wm = worst_margin(resp, labels, cfg.targets)[0]
            if best is None or wm > best[1]:
                best = (int(row["epoch"]), wm, patt, resp)
    if best is None:
        raise SystemExit(f"{run_dir} 沒有可評估的 epoch(無 pattern_hash)")
    return best


def best_pattern_figure(run_dir, cfg, label, out_path):
    """畫某臂「最佳 pattern + 各 label response vs spec」→ out_path。回傳 (epoch, wm)。"""
    labels = PORT_SPECS[cfg.port]["labels"]
    ep, wm, patt, resp = _best_row(run_dir, cfg)
    spec = setup_responses(cfg)                      # 取 target 曲線 + GHz 軸(裝全域 spec,標準腳本可接受)
    x = np.asarray(spec.x()).reshape(-1)
    t = torch.as_tensor(resp).float().reshape(len(labels), -1).cpu().numpy()
    patt = torch.as_tensor(patt).cpu().numpy()
    w = cfg.targets[labels[0]]["width"]; lo, hi = w[0] + w[1], w[0] + w[1] + w[2]

    fig, axes = plt.subplots(1, 1 + len(labels), figsize=(5 * (1 + len(labels)), 4))
    axes[0].imshow(patt, cmap="gray_r", vmin=0, vmax=1)
    axes[0].set_title(f"{label} best ep{ep}\nworst_margin={wm:.2f} dB")
    axes[0].set_xticks([]); axes[0].set_yticks([])
    for j, lab in enumerate(labels):
        ax = axes[1 + j]
        tgt = np.asarray(spec[lab].response.detach().cpu()).reshape(-1)
        c = float(cfg.targets[lab]["center"]); low = cfg.targets[lab]["method"] == "low"
        ax.plot(x, t[j], "b-", lw=2, label=f"HFSS {lab}")
        ax.plot(x, tgt, "r--", lw=1.2, label="target")
        ax.axvspan(x[lo], x[hi - 1], color="orange", alpha=0.15, label="in-band")
        ax.axhline(c, color="g", ls=":", lw=1.2, label=f"spec {'<=' if low else '>='}{c:g}")
        band = t[j][lo:hi]; worst = float(band.max()) if low else float(band.min())
        ax.set_title(f"{lab}  in-band worst={worst:.2f}")
        ax.set_xlabel("freq (GHz)"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close(fig)
    return ep, wm


def benchmark_figure(run_dirs, labels, cfg0, random_store, at, out_path):
    """多臂 worst-margin best-so-far vs HFSS-call + random best-of-N 疊圖 → out_path。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    for rd, lab in zip(run_dirs, labels):
        ep, best = run_curve(rd)
        ax.plot(ep, best, lw=2, label=lab)
    if random_store:
        try:
            rx, rbest = random_curve(random_store, cfg0, at or 1000)
            ax.plot(rx, rbest, "k--", lw=2, label=f"random best-of-N ({random_store})")
        except Exception as e:
            logger.warning(f"random_store '{random_store}' 讀取失敗 → 跳過 random 線(只比各臂曲線):{e}")
    ax.axhline(0, color="r", ls=":", lw=1, label="spec met (margin=0)")
    ax.set_xlabel("HFSS calls"); ax.set_ylabel("best worst-margin so far (dB) [higher=better]")
    ax.set_title("worst-margin vs HFSS-call"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def _best_at(run_dir, at):
    """run 在 epoch ≤ at 的 best worst-margin so far(公平對標點);at=None → 全程最佳。"""
    ep, best = run_curve(run_dir)
    upto = [b for e, b in zip(ep, best) if at is None or e <= at]
    return upto[-1] if upto else float("nan")


def main():
    ap = argparse.ArgumentParser(description="round 結果歸檔:最佳 pattern 圖 + benchmark 疊圖 + markdown 數字")
    ap.add_argument("--round", required=True, help="round 編號(決定預設 out-dir = docs/log/assets/round-NN)")
    ap.add_argument("--runs", nargs="+", required=True, help="結果夾名(結尾相符)或路徑,多個=多臂")
    ap.add_argument("--labels", nargs="*", default=None, help="各臂顯示名(對齊 round 檔 A/B/C);省略用結果夾名")
    ap.add_argument("--random-store", default=None, help="random-sim 資料集名 → 疊 random best-of-N")
    ap.add_argument("--at", type=int, default=None, help="HFSS-call 預算(公平對標點;省略=全程)")
    ap.add_argument("--out-dir", default=None, help="圖輸出夾(預設 docs/log/assets/round-NN)")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join("docs", "log", "assets", f"round-{args.round}")
    os.makedirs(out_dir, exist_ok=True)
    run_dirs = [_resolve_run(r) for r in args.runs]
    labels = args.labels if args.labels else [rd.name for rd in run_dirs]
    if len(labels) != len(run_dirs):
        raise SystemExit(f"--labels 數量({len(labels)}) 與 --runs({len(run_dirs)}) 不符")
    cfg0 = load_config(str(run_dirs[0] / "config.yaml"))

    rows = []
    for rd, lab in zip(run_dirs, labels):
        cfg = load_config(str(rd / "config.yaml"))
        ep, wm = best_pattern_figure(rd, cfg, lab, os.path.join(out_dir, f"{lab}_best.png"))
        rows.append((lab, wm, ep))
    benchmark_figure(run_dirs, labels, cfg0, args.random_store, args.at, os.path.join(out_dir, "benchmark.png"))

    rand_at = None
    if args.random_store:
        try:
            rand_at = random_curve(args.random_store, cfg0, args.at or 1000)[1][-1]
        except Exception:
            rand_at = None

    # (c) markdown 數字表 → stdout(複製貼進 docs/log/round-NN 的 §4)
    rel = os.path.relpath(out_dir, os.path.join("docs", "log"))
    atlabel = f"@{args.at}" if args.at else "(全程)"
    print("\n" + "=" * 60 + "\n貼進 round 檔 §4:\n")
    head = "| 臂 | 最佳 worst_margin | 達到 epoch |"
    sep = "|---|---|---|"
    if rand_at is not None:
        head += f" vs random{atlabel} |"; sep += "---|"
    print(head); print(sep)
    for lab, wm, ep in sorted(rows, key=lambda r: -r[1]):
        line = f"| {lab} | {wm:.2f} dB | {ep} |"
        if rand_at is not None:
            line += f" {(_best_at(_resolve_run_by_label(run_dirs, labels, lab), args.at) - rand_at):+.2f} dB |"
        print(line)
    if rand_at is not None:
        print(f"\nrandom best-of-N {atlabel} = {rand_at:.2f} dB")
    print(f"圖: {rel}/<label>_best.png · {rel}/benchmark.png")


def _resolve_run_by_label(run_dirs, labels, lab):
    return run_dirs[labels.index(lab)]


if __name__ == "__main__":
    main()
