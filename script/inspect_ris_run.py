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
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    epochs = data["epoch"]
    ax[0].plot(epochs, data["real_loss"], label="real_loss", marker="o", markersize=3)
    ax[0].plot(epochs, data["fake_loss"], label="fake_loss (surrogate)", marker="x", markersize=3)
    ax[0].plot(epochs, data["min_loss"], label="min_loss", linestyle="--")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("loss")
    ax[0].set_title("Loss curves")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    tau_vals = [t for t in data.get("tau", []) if t is not None]
    if tau_vals:
        ax[1].plot(epochs[: len(tau_vals)], tau_vals, label="tau", marker=".", markersize=3, color="tab:orange")
        ax[1].set_xlabel("epoch")
        ax[1].set_ylabel("tau")
        ax[1].set_title("Gumbel-Sigmoid tau (退火)")
        ax[1].legend()
        ax[1].grid(alpha=0.3)
    else:
        ax[1].text(0.5, 0.5, "no tau logged\n(older run)", ha="center", va="center", transform=ax[1].transAxes)
        ax[1].set_title("tau (n/a)")

    ax[2].plot(epochs, data["r_feed"], label="r_feed", marker="s", markersize=3)
    ax[2].set_xlabel("epoch")
    ax[2].set_ylabel("value")
    ax[2].set_title("r_feed (FeedReachability)")
    ax[2].legend()
    ax[2].grid(alpha=0.3)

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
    ckpts = sorted(ckpt_dir.glob("generator_*.pth"), key=lambda p: int(p.stem.split("_")[1]))
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


def _forward_pattern(gen, target, n: int) -> np.ndarray:
    """硬二值化 pattern 並丟進 RIS sim，回傳響應 numpy 陣列。"""
    with torch.no_grad():
        logits = gen(target)
        pattern = (torch.sigmoid(logits) > 0.5).float().reshape(n, -1)
    sim = RISSimulator(element_num=n)
    response = sim(pattern)
    if isinstance(response, dict):
        response = next(iter(response.values()))
    return response.detach().cpu().numpy() if torch.is_tensor(response) else np.asarray(response)


def plot_response_vs_target(run_dir: Path, coord: tuple, out: Path) -> None:
    ckpt_dir = run_dir / "checkpoint"
    ckpts = sorted(ckpt_dir.glob("generator_*.pth"), key=lambda p: int(p.stem.split("_")[1]))
    if not ckpts:
        return

    last_ep = len(ckpts)
    _setup_once(coord)
    target = AntennaResponse.target.concat().to(config.device)

    # 找 best epoch (real_loss 最小)
    import pickle

    with open(run_dir / "temp.record", "rb") as f:
        rec_data = pickle.load(f)["_data"]
    real_losses = rec_data.get("real_loss", [])
    epochs = rec_data.get("epoch", list(range(1, len(real_losses) + 1)))
    best_ep = int(epochs[int(np.argmin(real_losses))]) if real_losses else last_ep

    n = coord[1] - coord[0]
    target_np = target.detach().cpu().numpy()
    x = np.arange(len(target_np))

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, target_np, label="target", linewidth=2)

    # 畫 best 與 last（若 best == last 只畫一條）
    for ep, style in [
        (best_ep, {"linewidth": 2, "alpha": 0.9}),
        (last_ep, {"linewidth": 1.5, "alpha": 0.6, "linestyle": "--"}),
    ]:
        if not (ckpt_dir / f"generator_{ep}.pth").exists():
            continue
        gen = load_generator(ckpt_dir / f"generator_{ep}.pth")
        resp = _forward_pattern(gen, target, n)
        label = f"epoch {ep} (best)" if ep == best_ep else f"epoch {ep} (last)"
        if best_ep == last_ep:
            label = f"epoch {ep} (best=last)"
            ax.plot(x, resp.flatten()[: len(target_np)], label=label, **style)
            break
        ax.plot(x, resp.flatten()[: len(target_np)], label=label, **style)

    ax.set_xlabel("sample index")
    ax.set_ylabel("dB")
    ax.set_title(f"Target vs Best/Last Response (best@ep{best_ep}, last@ep{last_ep})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  response vs target → {out}")


def detect_coord(run_dir: Path) -> tuple:
    """從第一個 generator checkpoint 的 state_dict 推斷 pattern 邊長。"""
    ckpt_dir = run_dir / "checkpoint"
    first = next(ckpt_dir.glob("generator_*.pth"), None)
    if first is None:
        return (0, 15, 0, 15)
    ckpt = torch.load(first, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    # 最後一個 Linear 的 out_features 就是 pattern 攤平後的大小
    last_fc_bias = [v for k, v in state.items() if k.endswith(".bias") and "fc_patch" in k][-1]
    pattern_size = last_fc_bias.shape[0]
    n = int(pattern_size**0.5)
    if n * n != pattern_size:
        raise ValueError(f"非方形 pattern_size={pattern_size}，無法推斷 coord")
    print(f"  detected coordinate: (0, {n}, 0, {n}) from pattern_size={pattern_size}")
    return (0, n, 0, n)


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

    coord = detect_coord(run_dir)
    plot_pattern_evolution(run_dir, coord, pic / "pattern_evolution.png")
    plot_response_vs_target(run_dir, coord, pic / "response_vs_target.png")


if __name__ == "__main__":
    main()
