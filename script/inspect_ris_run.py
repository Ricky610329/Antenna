"""檢視 RIS 訓練 run 的 loss 曲線、pattern 演化、target vs predicted 響應。

用法：
    python script/inspect_ris_run.py result/<run_name>
"""

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from antenna.core.pattern import AntennaPattern
from antenna.core.response import AntennaResponse
from antenna.models.generators.gumbel_sigmoid_gen import GumbelSigmoidGEN
from antenna.ris import RISSimulator
from antenna.utils.config import config


def load_record(run_dir: Path) -> dict:
    with open(run_dir / "temp.record", "rb") as f:
        r = pickle.load(f)
    return r["_data"]


def plot_loss(data: dict, out: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    epochs = data["epoch"]
    ax[0].plot(epochs, data["real_loss"], label="real_loss", marker="o")
    ax[0].plot(epochs, data["fake_loss"], label="fake_loss (surrogate)", marker="x")
    ax[0].plot(epochs, data["min_loss"], label="min_loss", linestyle="--")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("loss")
    ax[0].set_title("Loss curves")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot(epochs, data["r_feed"], label="r_feed", marker="s")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("value")
    ax[1].set_title("r_feed (FeedReachability)")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  loss curves → {out}")


_SETUP_DONE = False


def _setup_once(pattern_coord: tuple) -> None:
    """只 register 一次（全域狀態）。"""
    global _SETUP_DONE
    AntennaPattern.setDefaultCoordinate(pattern_coord)
    if _SETUP_DONE:
        return
    AntennaResponse.registerLabels("response", x="ris")
    AntennaResponse.registerTargetResponse(-20.0, 0.0, (140, 0, 40, 0, 181), "response")
    _SETUP_DONE = True


def load_generator(ckpt_path: Path) -> GumbelSigmoidGEN:
    gen = GumbelSigmoidGEN()
    ckpt = torch.load(ckpt_path, map_location=config.device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    gen.load_state_dict(state)
    gen.eval()
    return gen


def plot_pattern_evolution(run_dir: Path, coord: tuple, out: Path) -> None:
    ckpt_dir = run_dir / "checkpoint"
    ckpts = sorted(ckpt_dir.glob("generator_*.pth"),
                   key=lambda p: int(p.stem.split("_")[1]))
    if not ckpts:
        print("  no generator checkpoints found")
        return

    _setup_once(coord)

    sample_epochs = [1, len(ckpts) // 2, len(ckpts)]
    sample_epochs = sorted(set(e for e in sample_epochs if 1 <= e <= len(ckpts)))

    fig, axes = plt.subplots(1, len(sample_epochs), figsize=(4 * len(sample_epochs), 4))
    if len(sample_epochs) == 1:
        axes = [axes]

    target = AntennaResponse.target.concat().to(config.device)
    for ax, ep in zip(axes, sample_epochs):
        gen = load_generator(ckpt_dir / f"generator_{ep}.pth")
        with torch.no_grad():
            logits = gen(target)
        n = coord[1] - coord[0]
        img = torch.sigmoid(logits.detach()).cpu().numpy().reshape(n, n)
        ax.imshow(img, cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(f"epoch {ep} (sigmoid(logits))")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  pattern evolution → {out}")


def plot_response_vs_target(run_dir: Path, coord: tuple, out: Path) -> None:
    ckpt_dir = run_dir / "checkpoint"
    ckpts = sorted(ckpt_dir.glob("generator_*.pth"),
                   key=lambda p: int(p.stem.split("_")[1]))
    if not ckpts:
        return

    last_ep = len(ckpts)
    _setup_once(coord)

    sim = RISSimulator(element_num=coord[1] - coord[0])
    target = AntennaResponse.target.concat().to(config.device)

    gen = load_generator(ckpt_dir / f"generator_{last_ep}.pth")
    with torch.no_grad():
        logits = gen(target)
        pattern = (torch.sigmoid(logits) > 0.5).float().reshape(coord[1] - coord[0], -1)
        pattern_2d = pattern.detach()
        response = sim(pattern_2d)

    if isinstance(response, dict):
        response = next(iter(response.values()))
    response_np = response.detach().cpu().numpy() if torch.is_tensor(response) else np.asarray(response)

    target_np = target.detach().cpu().numpy()
    x = np.arange(len(target_np))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, target_np, label="target", linewidth=2)
    ax.plot(x, response_np.flatten()[:len(target_np)], label=f"epoch {last_ep}", linewidth=1.5, alpha=0.8)
    ax.set_xlabel("sample index")
    ax.set_ylabel("dB")
    ax.set_title("Target vs Final Response (last epoch)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  response vs target → {out}")


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    config.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    run_dir = Path(sys.argv[1]).resolve()
    print(f"inspecting {run_dir}")

    data = load_record(run_dir)
    n = len(data["epoch"])
    print(f"  epochs recorded: {n}")
    print(f"  min_loss reached: {min(data['real_loss']):.4f} at epoch {data['epoch'][np.argmin(data['real_loss'])]}")

    pic = run_dir / "pic"
    pic.mkdir(exist_ok=True)

    plot_loss(data, pic / "loss_curves.png")

    # 從 experiment 名稱推斷 coordinate — 或從 config 讀
    coord_guess = (0, 15, 0, 15)  # smoke default
    if "40x40" in run_dir.name or "20ep" in run_dir.name:
        coord_guess = (0, 40, 0, 40)
    plot_pattern_evolution(run_dir, coord_guess, pic / "pattern_evolution.png")
    plot_response_vs_target(run_dir, coord_guess, pic / "response_vs_target.png")


if __name__ == "__main__":
    main()
