"""
Round 66+ — Binary RIS Pattern Dataset Generator

每 entry 變化 (target_shape × target_θc × target_width × freq × n × inc × ripple_w)。
保存:
- entries.jsonl: 一行一筆 metadata (config + metrics)
- patterns/<id>.npy: binary pattern (n×n)
- responses/<id>.npy: dB response (361,)

對每個 (config, ripple_w) 跑 N seeds × free-phase GD with worst-case loss
取 best worst_supp seed，full Pareto frontier across ripple weights。

對 patch antenna 移植：把 binary RIS pattern 換成 patch geometry，把 RISSimulator
換成 surrogate model，loss/workflow/dataset schema 一致。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from antenna.ris import RISSimulator
from antenna.utils.config import config


def soft_max(x: torch.Tensor, beta: float) -> torch.Tensor:
    return (1.0 / beta) * torch.logsumexp(beta * x, dim=0)


def soft_min(x: torch.Tensor, beta: float) -> torch.Tensor:
    return -(1.0 / beta) * torch.logsumexp(-beta * x, dim=0)


def worst_case_loss(
    resp: torch.Tensor,
    main_mask: torch.Tensor,
    beta: float,
    ripple_weight: float,
) -> torch.Tensor:
    main_min = soft_min(resp[main_mask], beta)
    side_max = soft_max(resp[~main_mask], beta)
    loss = -(main_min - side_max)
    if ripple_weight > 0:
        main_max = soft_max(resp[main_mask], beta)
        ripple = main_max - main_min
        loss = loss + ripple_weight * ripple
    return loss


def evaluate(resp_np: np.ndarray, main_lo: int, main_hi: int) -> dict:
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


def optimize_one(
    sim: RISSimulator,
    n: int,
    main_lo: int,
    main_hi: int,
    seed: int,
    steps: int,
    lr: float,
    beta: float,
    ripple_weight: float,
    device: str,
) -> tuple[np.ndarray, dict]:
    torch.manual_seed(seed)
    params = nn.Parameter(torch.rand(n**2, device=device) * 2.0)
    opt = torch.optim.Adam([params], lr=lr)
    best_loss = float("inf")
    best_params = params.detach().clone()
    for step in range(steps):
        opt.zero_grad()
        resp = sim(params.reshape(n, n))["response"]
        mask = torch.zeros_like(resp, dtype=torch.bool)
        mask[main_lo:main_hi] = True
        loss = worst_case_loss(resp, mask, beta=beta, ripple_weight=ripple_weight)
        loss.backward()
        opt.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_params = params.detach().clone()

    with torch.no_grad():
        phase = (best_params * torch.pi) % (2 * torch.pi)
        bin_pat = ((phase > torch.pi / 2) & (phase < 3 * torch.pi / 2)).float().reshape(n, n)
        bin_resp = sim(bin_pat)["response"].cpu().numpy()
    return bin_pat.cpu().numpy(), bin_resp


def build_target(theta_center_deg: float, width_deg: float) -> tuple[int, int]:
    """θ ∈ [-90°, 90°] 對應 idx [0, 360]，每 0.5° 一個 sample。"""
    sample_per_deg = 2  # 1 / 0.5
    center_idx = int(round((theta_center_deg + 90) * sample_per_deg))
    half_w_idx = int(round(width_deg * sample_per_deg / 2))
    main_lo = max(0, center_idx - half_w_idx)
    main_hi = min(361, center_idx + half_w_idx)
    return main_lo, main_hi


def gen_entry(
    entry_id: int,
    config_d: dict,
    seeds: list[int],
    ripple_weights: list[float],
    steps: int,
    lr: float,
    beta: float,
    out_dir: Path,
    device: str,
) -> dict:
    """跑一個 config，含 Pareto frontier across ripple_weights。"""
    sim = RISSimulator(
        element_num=config_d["n"],
        freq_hz=config_d["freq"],
        inc_theta_deg=config_d["inc"],
    )
    main_lo, main_hi = build_target(config_d["target_theta_c"], config_d["target_width_deg"])

    pareto = []
    for rw in ripple_weights:
        # multi-restart for this ripple weight
        best_metrics = None
        best_pattern = None
        best_response = None
        seed_results = []
        for seed in seeds:
            bin_pat, bin_resp = optimize_one(
                sim, config_d["n"], main_lo, main_hi, seed,
                steps, lr, beta, rw, device,
            )
            metrics = evaluate(bin_resp, main_lo, main_hi)
            seed_results.append({"seed": seed, "worst_supp": metrics["worst_supp"]})
            if best_metrics is None or metrics["worst_supp"] > best_metrics["worst_supp"]:
                best_metrics = metrics
                best_pattern = bin_pat
                best_response = bin_resp
        # save best for this rw
        sub_id = f"entry{entry_id:04d}_rw{rw}"
        np.save(out_dir / "patterns" / f"{sub_id}.npy", best_pattern)
        np.save(out_dir / "responses" / f"{sub_id}.npy", best_response)
        pareto.append({
            "ripple_weight": rw,
            "metrics": best_metrics,
            "best_seed": max(seed_results, key=lambda x: x["worst_supp"])["seed"],
            "all_seeds": seed_results,
            "pattern_file": f"patterns/{sub_id}.npy",
            "response_file": f"responses/{sub_id}.npy",
        })

    return {
        "entry_id": entry_id,
        "config": config_d,
        "main_idx_range": [main_lo, main_hi],
        "pareto": pareto,
    }


def gen_configs(args: argparse.Namespace) -> list[dict]:
    configs = []
    for freq_ghz in args.freqs:
        for n in args.ns:
            for theta_c in args.theta_centers:
                for tw in args.target_widths:
                    for inc in args.inc_thetas:
                        configs.append({
                            "freq": freq_ghz * 1e9,
                            "freq_ghz": freq_ghz,
                            "n": n,
                            "target_theta_c": theta_c,
                            "target_width_deg": tw,
                            "inc": inc,
                            "target_shape": "flat_plateau",
                        })
    return configs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default="outputs/dataset_v1")
    p.add_argument("--freqs", type=float, nargs="+", default=[28.0, 38.0])
    p.add_argument("--ns", type=int, nargs="+", default=[21, 31, 41])
    p.add_argument("--theta_centers", type=float, nargs="+", default=[-30.0, 0.0, 30.0])
    p.add_argument("--target_widths", type=float, nargs="+", default=[10.0, 20.0, 30.0])
    p.add_argument("--inc_thetas", type=float, nargs="+", default=[51.0])
    p.add_argument("--ripple_weights", type=float, nargs="+", default=[0.0, 1.0, 2.0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--beta", type=float, default=20.0)
    p.add_argument("--max_entries", type=int, default=None,
                   help="限制最多跑幾個 config (sample mode)")
    p.add_argument("--device", type=str, default="cuda:0")
    args = p.parse_args()

    config.device = args.device

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "patterns").mkdir(exist_ok=True)
    (out_dir / "responses").mkdir(exist_ok=True)

    configs = gen_configs(args)
    if args.max_entries:
        configs = configs[: args.max_entries]
    print(f"Generating {len(configs)} configs × {len(args.ripple_weights)} rw "
          f"× {len(args.seeds)} seeds = {len(configs) * len(args.ripple_weights) * len(args.seeds)} runs")

    entries_path = out_dir / "entries.jsonl"
    start = time.time()
    with open(entries_path, "w", encoding="utf-8") as f:
        for i, cfg in enumerate(configs):
            t0 = time.time()
            entry = gen_entry(
                i, cfg, args.seeds, args.ripple_weights,
                args.steps, args.lr, args.beta, out_dir, args.device,
            )
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
            elapsed = time.time() - t0
            best_pareto = max(entry["pareto"], key=lambda x: x["metrics"]["worst_supp"])
            best_flat = next(
                (p for p in entry["pareto"]
                 if p["metrics"]["main_below_3dB"] == 0),
                None,
            )
            flat_str = (
                f"flat-top@rw={best_flat['ripple_weight']}: worst={best_flat['metrics']['worst_supp']:+.2f}"
                if best_flat else "no flat-top achieved"
            )
            print(
                f"[{i + 1:4d}/{len(configs)}] freq={cfg['freq_ghz']}GHz "
                f"n={cfg['n']} θc={cfg['target_theta_c']:+.0f} w={cfg['target_width_deg']:.0f}° | "
                f"best worst={best_pareto['metrics']['worst_supp']:+.2f} (rw={best_pareto['ripple_weight']}) "
                f"| {flat_str} "
                f"| {elapsed:.1f}s"
            )

    total = time.time() - start
    print(f"\nDone. {len(configs)} entries in {total/60:.1f} min "
          f"({total/len(configs):.1f}s per entry).")
    print(f"Output: {entries_path}")


if __name__ == "__main__":
    main()
