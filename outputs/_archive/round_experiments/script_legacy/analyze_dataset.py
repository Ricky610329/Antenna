"""
Round 67 — Dataset 分析工具

讀 dataset_v1/entries.jsonl 算統計：
- 哪些 config 達 flat-top（main_below_3dB == 0）
- Pareto trade-off 分佈
- 哪些變數對 worst_supp / ripple 影響最大
- 邊際 case (壞表現的 config 該補資料)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_dataset(path: Path) -> list[dict]:
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def analyze(entries: list[dict]) -> None:
    print(f"=== Dataset analysis: {len(entries)} entries ===\n")

    # Flatten Pareto into rows: one row per (config, ripple_weight)
    rows = []
    for e in entries:
        for p in e["pareto"]:
            row = {**e["config"], "ripple_weight": p["ripple_weight"], **p["metrics"]}
            rows.append(row)
    print(f"Total Pareto rows: {len(rows)}\n")

    # Per ripple_weight stats
    rws = sorted({r["ripple_weight"] for r in rows})
    print("== Per ripple_weight stats ==")
    print(f"{'rw':>5} | {'count':>5} | {'worst_mean':>10} | {'worst_max':>10} | "
          f"{'ripple_mean':>11} | {'flat-top%':>10}")
    print("-" * 75)
    for rw in rws:
        rs = [r for r in rows if r["ripple_weight"] == rw]
        worst = np.array([r["worst_supp"] for r in rs])
        ripple = np.array([r["main_ripple"] for r in rs])
        flat = sum(1 for r in rs if r["main_below_3dB"] == 0)
        print(f"{rw:5.1f} | {len(rs):5d} | {worst.mean():+10.2f} | {worst.max():+10.2f} | "
              f"{ripple.mean():11.2f} | {flat}/{len(rs)} ({100*flat/len(rs):4.0f}%)")

    # Best per (n, target_width_deg) cell at rw=2
    print("\n== Best worst_supp at rw=2 (flat-top mode) per (n × target_width) ==")
    rws_2 = [r for r in rows if r["ripple_weight"] == 2.0]
    if rws_2:
        ns = sorted({r["n"] for r in rws_2})
        widths = sorted({r["target_width_deg"] for r in rws_2})
        print(f"{'n \\ width':>10} | " + " | ".join(f"{w:>6.0f}°" for w in widths))
        print("-" * (12 + 11 * len(widths)))
        for n in ns:
            cells = []
            for w in widths:
                rs = [r for r in rws_2 if r["n"] == n and r["target_width_deg"] == w]
                if rs:
                    best = max(r["worst_supp"] for r in rs)
                    cells.append(f"{best:+6.2f}")
                else:
                    cells.append("  --  ")
            print(f"{n:>10} | " + " | ".join(cells))

    # Per target_theta_c (off-axis vs broadside)
    print("\n== Worst supp by target_theta_c (rw=2) ==")
    if rws_2:
        thetas = sorted({r["target_theta_c"] for r in rws_2})
        for tc in thetas:
            rs = [r for r in rws_2 if r["target_theta_c"] == tc]
            ws = [r["worst_supp"] for r in rs]
            flat = sum(1 for r in rs if r["main_below_3dB"] == 0)
            print(f"  θc={tc:+5.0f}°: mean worst={np.mean(ws):+.2f}, max={np.max(ws):+.2f}, "
                  f"flat-top {flat}/{len(rs)}")

    # Pareto trade-off scatter (text-based)
    print("\n== Pareto Trade-off ==")
    print(f"{'config':>50} | {'rw':>4} | {'worst':>7} | {'ripple':>7} | {'flat-top?':>9}")
    print("-" * 95)
    for e in entries[-10:]:  # 最後 10 個 entry 詳列
        for p in e["pareto"]:
            cfg = e["config"]
            label = f"{cfg['freq_ghz']}GHz n={cfg['n']} θc={cfg['target_theta_c']:+.0f} w={cfg['target_width_deg']:.0f}"
            ft = "yes" if p["metrics"]["main_below_3dB"] == 0 else "no"
            print(f"{label:>50} | {p['ripple_weight']:>4.1f} | "
                  f"{p['metrics']['worst_supp']:+7.2f} | "
                  f"{p['metrics']['main_ripple']:7.2f} | {ft:>9}")

    # Worst configs (low worst_supp at rw=2)
    if rws_2:
        sorted_rs = sorted(rws_2, key=lambda r: r["worst_supp"])
        print("\n== 最差 5 個 (rw=2, 需要補資料的 edge cases) ==")
        for r in sorted_rs[:5]:
            print(f"  freq={r['freq_ghz']}GHz n={r['n']} θc={r['target_theta_c']:+.0f} "
                  f"w={r['target_width_deg']:.0f}° → worst={r['worst_supp']:+.2f}, "
                  f"ripple={r['main_ripple']:.2f}, flat-top={'yes' if r['main_below_3dB']==0 else 'no'}")
        print("\n== 最佳 5 個 (rw=2) ==")
        for r in sorted_rs[-5:][::-1]:
            print(f"  freq={r['freq_ghz']}GHz n={r['n']} θc={r['target_theta_c']:+.0f} "
                  f"w={r['target_width_deg']:.0f}° → worst={r['worst_supp']:+.2f}, "
                  f"ripple={r['main_ripple']:.2f}, flat-top={'yes' if r['main_below_3dB']==0 else 'no'}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="outputs/dataset_v1/entries.jsonl")
    args = p.parse_args()
    entries = load_dataset(Path(args.dataset))
    analyze(entries)


if __name__ == "__main__":
    main()
