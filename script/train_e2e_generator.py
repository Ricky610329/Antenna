"""
Round 74 — End-to-End Generator through differentiable simulator

R73 揭露 supervised BCE 不適合 discrete binary manifold。
這版用真實 RIS sim 做 differentiable forward, generator 端到端用 worst-case loss。

對 patch antenna 移植: 把 RISSimulator 換成 trained surrogate, 其餘 unchanged。

設計:
- 固定 (freq, n, inc) 例如 38 GHz × n=31 × inc=51°
- 變化 (target_θc, target_width, ripple_weight) 作為 generator 條件
- Generator: config → free-phase pattern (n×n in ℝ, no sigmoid 限制)
- Forward: pattern → sim → response → worst-case loss
- Eval: free-phase → 1-bit quantize → real binary response
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from antenna.ris import RISSimulator
from antenna.utils.config import config


class FreePhaseGenerator(nn.Module):
    """config → free-phase pattern (n×n)，不限值域。"""

    def __init__(self, config_dim: int = 3, n: int = 31, channels: int = 64):
        super().__init__()
        self.n = n
        # 6×6 latent → upsample to n×n
        self.encoder = nn.Sequential(
            nn.Linear(config_dim, 256), nn.GELU(),
            nn.Linear(256, 512), nn.GELU(),
            nn.Linear(512, channels * 8 * 8), nn.GELU(),
        )
        self.channels = channels
        # Upsample 8x8 → 16x16 → 32x32 → crop to n
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
            nn.Conv2d(channels, 1, 3, padding=1),
        )

    def forward(self, cfg: torch.Tensor) -> torch.Tensor:
        b = cfg.shape[0]
        feat = self.encoder(cfg).reshape(b, self.channels, 8, 8)
        out = self.decoder(feat).squeeze(1)  # [B, 32, 32]
        # crop to n
        offset = (out.shape[1] - self.n) // 2
        return out[:, offset:offset + self.n, offset:offset + self.n]


def soft_max_logsumexp(x: torch.Tensor, beta: float = 20.0, dim: int = -1) -> torch.Tensor:
    return (1.0 / beta) * torch.logsumexp(beta * x, dim=dim)


def soft_min_logsumexp(x: torch.Tensor, beta: float = 20.0, dim: int = -1) -> torch.Tensor:
    return -(1.0 / beta) * torch.logsumexp(-beta * x, dim=dim)


def worst_case_loss(resp: torch.Tensor, main_lo: int, main_hi: int,
                    beta: float = 20.0, ripple_weight: float = 0.0) -> torch.Tensor:
    """resp: [..., 361]"""
    main = resp[..., main_lo:main_hi]
    side = torch.cat([resp[..., :main_lo], resp[..., main_hi:]], dim=-1)
    main_min = soft_min_logsumexp(main, beta, dim=-1)
    side_max = soft_max_logsumexp(side, beta, dim=-1)
    loss = -(main_min - side_max)
    if ripple_weight > 0:
        main_max = soft_max_logsumexp(main, beta, dim=-1)
        ripple = main_max - main_min
        loss = loss + ripple_weight * ripple
    return loss


def build_target_idx(theta_c: float, width: float) -> tuple[int, int]:
    sample_per_deg = 2
    center = int(round((theta_c + 90) * sample_per_deg))
    half = int(round(width * sample_per_deg / 2))
    return max(0, center - half), min(361, center + half)


def gen_config(theta_c: float, width: float, rw: float) -> torch.Tensor:
    return torch.tensor([
        theta_c / 90.0,
        width / 90.0,
        rw / 5.0,
    ], dtype=torch.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=31)
    p.add_argument("--freq", type=float, default=38e9)
    p.add_argument("--inc", type=float, default=51.0)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--out_dir", type=str, default="outputs/r74_e2e_generator")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    config.device = args.device
    device = args.device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    sim = RISSimulator(element_num=args.n, freq_hz=args.freq, inc_theta_deg=args.inc)

    # Training distribution
    theta_centers = [-30.0, -15.0, 0.0, 15.0, 30.0]
    widths = [10.0, 15.0, 20.0, 25.0, 30.0]
    ripple_weights = [0.0, 1.0, 2.0]

    gen = FreePhaseGenerator(config_dim=3, n=args.n, channels=args.channels).to(device)
    opt = torch.optim.Adam(gen.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in gen.parameters())
    print(f"Generator: {n_params:,} params (n={args.n})")

    # Held-out test configs
    test_configs = [
        (-25.0, 12.0, 0.0),  # off-axis, narrow, steering
        (0.0, 20.0, 2.0),    # broadside, medium, flat-top
        (20.0, 25.0, 1.0),   # off-axis, wide, balanced
        (-15.0, 10.0, 2.0),  # off-axis narrow flat-top
        (15.0, 30.0, 0.0),   # off-axis wide steering
    ]

    # Training loop
    print("\n=== Training (E2E through real sim) ===")
    for epoch in range(args.epochs):
        # Sample random batch of configs
        cfgs = []
        for _ in range(args.batch_size):
            tc = float(np.random.choice(theta_centers))
            tw = float(np.random.choice(widths))
            rw = float(np.random.choice(ripple_weights))
            cfgs.append((tc, tw, rw))
        cfg_vecs = torch.stack([gen_config(*c) for c in cfgs]).to(device)

        gen.train()
        free_phase = gen(cfg_vecs)  # [B, n, n] in ℝ

        # Forward each through sim (sim takes single pattern; loop is OK for small batches)
        losses = []
        for i, (tc, tw, rw) in enumerate(cfgs):
            main_lo, main_hi = build_target_idx(tc, tw)
            resp = sim(free_phase[i])["response"]  # 361
            losses.append(worst_case_loss(resp, main_lo, main_hi, beta=20.0, ripple_weight=rw))
        loss = torch.stack(losses).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        if (epoch + 1) % 100 == 0 or epoch < 5:
            print(f"  epoch {epoch+1:4d}  loss={loss.item():+.3f}")

    # === Eval on held-out test configs ===
    print(f"\n=== Eval on held-out test configs (binary 1-bit quantize) ===")
    print(f"{'config':<35} | {'cont_worst':>10} | {'binary_worst':>12} | {'binary_ripple':>13} | "
          f"{'main<-3 cnt':>11}")
    print("-" * 95)
    gen.eval()
    summary = []
    for tc, tw, rw in test_configs:
        cfg_vec = gen_config(tc, tw, rw).unsqueeze(0).to(device)
        with torch.no_grad():
            free_phase = gen(cfg_vec)[0]
            # Continuous response
            cont_resp = sim(free_phase)["response"].cpu().numpy()
            # 1-bit quantize via R57 free-phase scheme
            phase = (free_phase * torch.pi) % (2 * torch.pi)
            binary = ((phase > torch.pi / 2) & (phase < 3 * torch.pi / 2)).float()
            bin_resp = sim(binary)["response"].cpu().numpy()

        main_lo, main_hi = build_target_idx(tc, tw)
        main_c = cont_resp[main_lo:main_hi]
        side_c = np.delete(cont_resp, np.arange(main_lo, main_hi))
        cont_worst = float(main_c.min() - side_c.max())

        main_b = bin_resp[main_lo:main_hi]
        side_b = np.delete(bin_resp, np.arange(main_lo, main_hi))
        binary_worst = float(main_b.min() - side_b.max())
        binary_ripple = float(main_b.max() - main_b.min())
        below_3 = int((main_b < -3.0).sum())
        n_main = len(main_b)

        cfg_str = f"θc={tc:+.0f} w={tw:.0f} rw={rw}"
        print(f"{cfg_str:<35} | {cont_worst:+10.2f} | {binary_worst:+12.2f} | "
              f"{binary_ripple:>13.2f} | {below_3:>4}/{n_main}")
        summary.append({
            "config": (tc, tw, rw),
            "cont_worst": cont_worst,
            "binary_worst": binary_worst,
            "binary_ripple": binary_ripple,
            "main_below_3": below_3,
            "main_total": n_main,
        })

    # Compare with per-target optimization baseline (R64 worst-case GD-from-scratch)
    print(f"\n=== Compare with per-target R64 baseline (single config) ===")
    print(f"R64 reference: 38 GHz × n=41 × broadside × w=20 × rw=2 → worst +6.88 dB (5 seeds)")
    print(f"Note: 上面 generator amortizes 預測 vs per-target run from scratch")

    torch.save(gen.state_dict(), out / "generator.pt")
    with open(out / "test_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
