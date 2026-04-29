"""探 RIS 邊長 element_num 對 suppression 上限的影響。

對每個 element_num（10/15/20/25），跑 3 組代表性 target，看 suppression 變化。
RIS 越大 element 越多，理論上波束更銳利（aperture↑ → 主峰窄、sidelobe 低）。
本 script 量化這個關係，給使用者選擇陣列大小的依據。

用法：
    python script/sweep_element_num.py --steps 1500 --device cuda:0
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from torch import nn

from antenna.core.pattern import AntennaPattern
from antenna.core.response import AntennaResponse
from antenna.ris import RISSimulator, custom_loss_tolerance
from antenna.utils.config import config


def design_one(
    sim: RISSimulator, target: torch.Tensor,
    *, element_num: int, main_lo: int, main_hi: int,
    steps: int, lr: float, n_restarts: int, seed: int,
) -> dict:
    pattern_size = element_num * element_num
    best_supp = -np.inf
    best_hard_resp = None
    best_pattern = None
    for restart in range(n_restarts):
        torch.manual_seed(seed + restart)
        logits = nn.Parameter(torch.randn(pattern_size, device=config.device))
        optimizer = torch.optim.Adam([logits], lr=lr)
        best_loss_local = float("inf")
        best_logits_local = logits.detach().clone()
        for _ in range(steps):
            optimizer.zero_grad()
            soft = torch.sigmoid(logits)
            pat = soft.reshape(element_num, element_num)
            resp = sim(pat)["response"]
            loss = custom_loss_tolerance(
                resp, target, sidelobe_threshold=-25.0, main_target=0.0, main_weight=5.0,
            )
            loss.backward(); optimizer.step()
            if loss.item() < best_loss_local:
                best_loss_local = loss.item()
                best_logits_local = logits.detach().clone()

        with torch.no_grad():
            hard = (torch.sigmoid(best_logits_local) > 0.5).float().reshape(element_num, element_num)
            hard_resp = sim(hard)["response"].cpu().numpy()
            main_idx = np.arange(main_lo, min(main_hi, len(hard_resp)))
            side_idx = np.array([i for i in range(len(hard_resp)) if i not in set(main_idx.tolist())])
            mp = float(hard_resp[main_idx].max())
            sm = float(hard_resp[side_idx].max())
            supp = mp - sm
            if supp > best_supp:
                best_supp = supp
                best_hard_resp = hard_resp
                best_pattern = hard.cpu().numpy()
    return {
        "suppression": best_supp,
        "hard_resp": best_hard_resp,
        "pattern": best_pattern,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--element_nums", type=int, nargs="+", default=[10, 15, 20, 25])
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--n_restarts", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="outputs/sweep_element_num")
    args = p.parse_args()

    if args.device:
        config.device = args.device
    elif torch.cuda.is_available():
        config.device = "cuda:0"
    logger.info(f"裝置：{config.device}")

    AntennaResponse.registerLabels("response", x="ris")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    THETA_DEG = np.arange(-90, 90.1, 0.5)

    # 三組代表 target（plateau 寬 40，分別在 left/center/right）
    target_configs = [
        ("left",   91, 40),    # θ_center ≈ -34°
        ("center", 154, 40),   # θ_center ≈ +2°
        ("right",  217, 40),   # θ_center ≈ +38°
    ]

    results = {}
    for n in args.element_nums:
        AntennaPattern.setDefaultCoordinate((0, n, 0, n))
        sim = RISSimulator(element_num=n)
        results[n] = {}
        for name, start, w in target_configs:
            target_np = np.full(361, -20.0, dtype=np.float32)
            target_np[start:start + w] = 0.0
            target = torch.tensor(target_np, device=config.device)
            info = design_one(
                sim, target, element_num=n,
                main_lo=start, main_hi=start + w,
                steps=args.steps, lr=args.lr,
                n_restarts=args.n_restarts, seed=args.seed,
            )
            results[n][name] = info
            logger.info(
                f"  n={n}, target={name}: suppression={info['suppression']:+.2f} dB"
            )

    # Plot suppression vs element_num for each target
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, _, _ in target_configs:
        xs = list(results.keys())
        ys = [results[n][name]["suppression"] for n in xs]
        ax.plot(xs, ys, marker="o", label=f"target {name}")
    ax.set_xlabel("element_num (RIS 邊長)")
    ax.set_ylabel("suppression (dB)")
    ax.set_title("Suppression upper limit vs RIS array size")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "vs_element_num.png", dpi=110)
    plt.close(fig)

    # Markdown
    lines = ["# Element_num sweep — suppression vs RIS 邊長", "",
             f"Setting: 28 GHz, inc θ=-40°/φ=90°, {args.steps} steps × "
             f"{args.n_restarts} restarts (取最佳)", "",
             "| element_num | total cells | target_left dB | target_center dB | target_right dB |",
             "|-------------|-------------|----------------|------------------|-----------------|"]
    for n in args.element_nums:
        cells = n * n
        l = results[n]["left"]["suppression"]
        c = results[n]["center"]["suppression"]
        r = results[n]["right"]["suppression"]
        lines.append(f"| {n} | {cells} | {l:+.2f} | {c:+.2f} | {r:+.2f} |")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    logger.success(f"完成 → {out_dir}/")


if __name__ == "__main__":
    main()
