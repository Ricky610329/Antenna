"""
Round 82 — Diverse dataset (R81 ranking failure 假設驗證)

對 dataset_v2 每 config, 額外 sample 3 個 random binary patterns + 跑 sim 算 response.
用 v2 (optimized) + v3_random (uniform binary) 合併訓 surrogate.

如果 ranking correlation 從 R81 0.031 顯著提升 → diversity hypothesis 證實.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from antenna.ris import RISSimulator
from antenna.utils.config import config


def supp_metrics(resp_np, main_lo, main_hi):
    main = resp_np[main_lo:main_hi]
    side = np.delete(resp_np, np.arange(main_lo, main_hi))
    return {
        "headline_supp": float(main.max() - side.max()),
        "worst_supp": float(main.min() - side.max()),
        "main_max": float(main.max()),
        "main_min": float(main.min()),
        "main_ripple": float(main.max() - main.min()),
        "side_max": float(side.max()),
        "side_mean": float(side.mean()),
        "side_std": float(side.std()),
        "main_below_3dB": int((main < -3.0).sum()),
        "main_total": len(main),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src_dataset", type=str, default="outputs/dataset_v2")
    p.add_argument("--out_dir", type=str, default="outputs/dataset_v3")
    p.add_argument("--n_random_per_config", type=int, default=3)
    p.add_argument("--device", type=str, default="cuda:0")
    args = p.parse_args()

    config.device = args.device
    src_root = Path(args.src_dataset)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "patterns").mkdir(exist_ok=True)
    (out_root / "responses").mkdir(exist_ok=True)

    # Load v2 entries
    src_entries = []
    with open(src_root / "entries.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                src_entries.append(json.loads(line))

    print(f"Source: {len(src_entries)} configs from dataset_v2")
    print(f"Adding {args.n_random_per_config} random patterns per config...")

    # Generate v3 entries:
    # - Copy all v2 Pareto entries (optimized)
    # - Add N random patterns per config
    rng = np.random.RandomState(42)
    v3_rows = []  # flat list of {config, ripple_weight, source_type, pattern_file, response_file, metrics}

    for entry_idx, e in enumerate(src_entries):
        cfg = e["config"]
        main_lo, main_hi = e["main_idx_range"]
        sim = RISSimulator(element_num=cfg["n"], freq_hz=cfg["freq"], inc_theta_deg=cfg["inc"])

        # Copy optimized entries from v2
        for p_entry in e["pareto"]:
            v3_rows.append({
                "config": cfg,
                "main_idx_range": [main_lo, main_hi],
                "ripple_weight": p_entry["ripple_weight"],
                "source": "optimized_v2",
                "pattern_path_src": str(src_root / p_entry["pattern_file"]),
                "response_path_src": str(src_root / p_entry["response_file"]),
                "metrics": p_entry["metrics"],
                "entry_idx": entry_idx,
            })

        # Add N random patterns per config
        for j in range(args.n_random_per_config):
            seed = entry_idx * 100 + j
            rng_local = np.random.RandomState(seed)
            pat = (rng_local.rand(cfg["n"], cfg["n"]) > 0.5).astype(np.float32)
            pat_t = torch.from_numpy(pat).to(args.device)
            with torch.no_grad():
                resp = sim(pat_t)["response"].cpu().numpy()
            metrics = supp_metrics(resp, main_lo, main_hi)
            # Save pattern + response
            sub_id = f"random_e{entry_idx:04d}_s{j}"
            np.save(out_root / "patterns" / f"{sub_id}.npy", pat)
            np.save(out_root / "responses" / f"{sub_id}.npy", resp)
            # Use rw=2 as default (the ripple weight is irrelevant for random patterns since not optimized for it)
            for rw_label in [0.0, 2.0]:  # mark for both rw labels for training balance
                v3_rows.append({
                    "config": cfg,
                    "main_idx_range": [main_lo, main_hi],
                    "ripple_weight": rw_label,
                    "source": "random",
                    "pattern_file": f"patterns/{sub_id}.npy",
                    "response_file": f"responses/{sub_id}.npy",
                    "metrics": metrics,
                    "entry_idx": entry_idx,
                })
        if (entry_idx + 1) % 10 == 0:
            print(f"  {entry_idx+1}/{len(src_entries)}")

    # Write entries.jsonl in flat format (one row per training sample)
    out_entries_path = out_root / "entries.jsonl"
    with open(out_entries_path, "w", encoding="utf-8") as f:
        # Group by entry_idx for compatibility with existing schema
        grouped = {}
        for r in v3_rows:
            k = r["entry_idx"]
            grouped.setdefault(k, {
                "entry_id": k,
                "config": r["config"],
                "main_idx_range": r["main_idx_range"],
                "pareto": [],
            })
            # Determine pattern/response files
            if r["source"] == "optimized_v2":
                # Reference v2 paths
                pat_relative = Path(r["pattern_path_src"]).name
                resp_relative = Path(r["response_path_src"]).name
                # Copy or symlink? Just copy for simplicity
                src_pat = Path(r["pattern_path_src"])
                src_resp = Path(r["response_path_src"])
                dst_pat = out_root / "patterns" / src_pat.name
                dst_resp = out_root / "responses" / src_resp.name
                if not dst_pat.exists():
                    np.save(dst_pat, np.load(src_pat))
                if not dst_resp.exists():
                    np.save(dst_resp, np.load(src_resp))
                pattern_file = f"patterns/{src_pat.name}"
                response_file = f"responses/{src_resp.name}"
            else:
                pattern_file = r["pattern_file"]
                response_file = r["response_file"]

            grouped[k]["pareto"].append({
                "ripple_weight": r["ripple_weight"],
                "metrics": r["metrics"],
                "source": r["source"],
                "pattern_file": pattern_file,
                "response_file": response_file,
            })

        for k in sorted(grouped.keys()):
            f.write(json.dumps(grouped[k], ensure_ascii=False) + "\n")

    total_pareto = sum(len(g["pareto"]) for g in grouped.values())
    print(f"\nDone. dataset_v3: {len(grouped)} entries, {total_pareto} Pareto rows total")
    print(f"  optimized rows: {sum(1 for r in v3_rows if r['source']=='optimized_v2')}")
    print(f"  random rows:    {sum(1 for r in v3_rows if r['source']=='random')}")
    print(f"Output: {out_entries_path}")


if __name__ == "__main__":
    main()
