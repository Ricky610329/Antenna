"""
Round 70 — Surrogate scaling comparison: dataset_v1 (72) vs dataset_v2 (~216)

兩個 dataset 都訓練同 CNN 架構，比較:
- Test MSE vs dataset size
- worst_supp MAE
- 是否還有 systematic bias

對 patch 移植關鍵: 估算「需要多少 entries 才能達到 < 1 dB error」的 scaling law
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def run_surrogate(dataset: str, out_dir: str, epochs: int = 300) -> dict:
    """Train surrogate on given dataset, parse results."""
    cmd = [
        sys.executable, "script/train_surrogate.py",
        "--dataset", dataset,
        "--out_dir", out_dir,
        "--epochs", str(epochs),
        "--arch", "cnn",
        "--channels", "32",
        "--depth", "4",
    ]
    print(f"Running: {' '.join(cmd)}")
    env = {"PYTHONIOENCODING": "utf-8"}
    import os
    full_env = os.environ.copy()
    full_env.update(env)
    result = subprocess.run(cmd, capture_output=True, text=True, env=full_env)
    print(result.stdout[-2000:])
    if result.returncode != 0:
        print("STDERR:", result.stderr[-1000:])
        return {}

    # Parse final test_mse and MAE
    lines = result.stdout.strip().split("\n")
    final_test_mse = None
    mae_worst = None
    max_mae_worst = None
    n_entries = None
    for line in lines:
        if "Dataset:" in line and "entries" in line:
            try:
                n_entries = int(line.split()[1])
            except Exception:
                pass
        if line.strip().startswith("epoch"):
            try:
                parts = line.split()
                test_mse = float(parts[-1].replace("test_mse=", ""))
                final_test_mse = test_mse
            except Exception:
                pass
        if "Mean abs error in worst_supp" in line:
            mae_worst = float(line.split(":")[-1].strip().split()[0])
        if "Max abs error in worst_supp" in line:
            max_mae_worst = float(line.split(":")[-1].strip().split()[0])

    return {
        "n_entries": n_entries,
        "final_test_mse": final_test_mse,
        "mae_worst_supp": mae_worst,
        "max_mae_worst_supp": max_mae_worst,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", type=str, nargs="+",
                   default=["outputs/dataset_v1", "outputs/dataset_v2"])
    p.add_argument("--epochs", type=int, default=300)
    args = p.parse_args()

    results = []
    for ds in args.datasets:
        if not Path(ds, "entries.jsonl").exists():
            print(f"⚠ {ds} not ready, skipping")
            continue
        out = f"outputs/r70_compare_{Path(ds).name}"
        Path(out).mkdir(parents=True, exist_ok=True)
        r = run_surrogate(ds, out, args.epochs)
        r["dataset"] = Path(ds).name
        results.append(r)

    print("\n=== Surrogate Scaling Comparison ===")
    print(f"{'dataset':>15} | {'entries':>7} | {'test_mse':>9} | {'MAE worst':>10} | {'max MAE':>9}")
    print("-" * 65)
    for r in results:
        print(f"{r['dataset']:>15} | {r['n_entries']:>7} | "
              f"{r['final_test_mse']:>9.2f} | {r['mae_worst_supp']:>10.2f} | "
              f"{r['max_mae_worst_supp']:>9.2f}")

    # Save
    with open("outputs/r70_compare_summary.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
