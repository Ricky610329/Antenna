"""Cross-run 比較工具：把多個 result/RIS-* 的 loss / tau / response 疊在一張圖。

用法：
    python script/compare_ris_runs.py result/RIS-v4-* result/RIS-v7-* result/RIS-v9-*

會輸出：
    result/_compare/<timestamp>/loss_compare.png
    result/_compare/<timestamp>/tau_compare.png
    result/_compare/<timestamp>/response_compare.png
"""

import pickle
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from antenna.core.pattern import AntennaPattern
from antenna.core.response import AntennaResponse
from antenna.models.generators.gumbel_sigmoid_gen import GumbelSigmoidGEN
from antenna.ris import RISSimulator
from antenna.utils.config import config


def _load_record(run_dir: Path) -> dict:
    with open(run_dir / "temp.record", "rb") as f:
        return pickle.load(f)["_data"]


def _load_target_args(run_dir: Path) -> tuple:
    """從 result/<run>/config.yaml 讀 target；若無 fallback 到預設 V4-V9 target。"""
    cfg_path = run_dir / "config.yaml"
    if cfg_path.exists():
        try:
            from omegaconf import OmegaConf

            cfg = OmegaConf.load(cfg_path)
            tgt = cfg.response.label_configs.response.target
            return (float(tgt.side), float(tgt.center), tuple(tgt.width))
        except Exception:
            pass
    return (-20.0, 0.0, (140, 0, 40, 0, 181))


def _detect_coord(run_dir: Path) -> tuple[int, int, int, int]:
    first = next((run_dir / "checkpoint").glob("generator_*.pth"), None)
    if first is None:
        raise FileNotFoundError(f"no checkpoint in {run_dir}")
    ckpt = torch.load(first, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    last_fc_bias = [v for k, v in state.items() if k.endswith(".bias") and "fc_patch" in k][-1]
    pattern_size = last_fc_bias.shape[0]
    n = int(pattern_size**0.5)
    return (0, n, 0, n)


def _hard_response(run_dir: Path, coord: tuple[int, int, int, int], target: torch.Tensor) -> np.ndarray:
    """載入 best epoch 的 generator，hard-binarize → simulate，回傳 dB 響應。"""
    data = _load_record(run_dir)
    real_losses = data["real_loss"]
    epochs = data["epoch"]
    best_ep = int(epochs[int(np.argmin(real_losses))])

    gen = GumbelSigmoidGEN()
    ckpt = torch.load(
        run_dir / "checkpoint" / f"generator_{best_ep}.pth", map_location=config.device, weights_only=False
    )
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    gen.load_state_dict(state)
    gen.eval()

    n = coord[1] - coord[0]
    with torch.no_grad():
        logits = gen(target)
        pattern = (torch.sigmoid(logits) > 0.5).float().reshape(n, -1)
    sim = RISSimulator(element_num=n)
    resp = sim(pattern)
    if isinstance(resp, dict):
        resp = next(iter(resp.values()))
    return resp.detach().cpu().numpy().flatten(), best_ep


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    config.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    run_dirs = [Path(p).resolve() for p in sys.argv[1:]]
    print(f"comparing {len(run_dirs)} runs")
    for d in run_dirs:
        print(f"  - {d.name}")

    out_dir = Path("result/_compare") / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Loss curve overlay
    fig, ax = plt.subplots(figsize=(11, 5))
    for d in run_dirs:
        try:
            data = _load_record(d)
        except FileNotFoundError:
            print(f"  skip {d.name}: no record")
            continue
        ep = data["epoch"]
        rl = data["real_loss"]
        ax.plot(ep, rl, label=f"{d.name} (min={min(rl):.3f})", alpha=0.7, linewidth=1.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("real_loss")
    ax.set_title(f"Loss comparison ({len(run_dirs)} runs)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "loss_compare.png", dpi=120)
    plt.close(fig)
    print(f"  → {out_dir / 'loss_compare.png'}")

    # 2. Tau overlay
    fig, ax = plt.subplots(figsize=(11, 5))
    for d in run_dirs:
        data = _load_record(d)
        tau = [t for t in data.get("tau", []) if t is not None]
        if not tau:
            continue
        ep = data["epoch"][: len(tau)]
        ax.plot(ep, tau, label=d.name, alpha=0.7, linewidth=1.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("tau")
    ax.set_title("Gumbel-Sigmoid tau comparison")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "tau_compare.png", dpi=120)
    plt.close(fig)
    print(f"  → {out_dir / 'tau_compare.png'}")

    # 3. Response overlay (best epoch of each run, hard-binarized)
    # 每個 run 各自從其 result/<run>/config.yaml 讀回訓練時用的 target；舊 run
    # 沒存 cfg.yaml 的話 fallback 到預設 (140, 0, 40, 0, 181)。同 plot 上會多一條
    # target 線：若所有 run 都同一 target 就一條，否則畫多條（每個 run 各一）。
    fig, ax = plt.subplots(figsize=(13, 5))
    drawn_targets: list[tuple] = []  # 已畫過的 target args 去重
    for d in run_dirs:
        try:
            coord = _detect_coord(d)
        except FileNotFoundError:
            continue

        # 重設 AntennaResponse 全域狀態以套用此 run 的 target
        AntennaResponse.target.responses.clear()
        AntennaResponse.target.metadata.clear()
        AntennaPattern.setDefaultCoordinate(coord)
        AntennaResponse.registerLabels("response", x="ris")
        target_args = _load_target_args(d)
        AntennaResponse.registerTargetResponse(*target_args, "response")
        target = AntennaResponse.target.concat().to(config.device)

        # 畫此 run 的 target（同 args 不重畫）
        if target_args not in drawn_targets:
            drawn_targets.append(target_args)
            t_np = target.detach().cpu().numpy()
            ax.plot(
                np.arange(len(t_np)),
                t_np,
                label=f"target {target_args[2]}",
                linewidth=2,
                linestyle=":",
                alpha=0.8,
            )

        try:
            resp, best_ep = _hard_response(d, coord, target)
            ax.plot(
                np.arange(len(resp))[: len(target)],
                resp[: len(target)],
                label=f"{d.name}@ep{best_ep}",
                alpha=0.75,
                linewidth=1.3,
            )
        except Exception as e:
            print(f"  skip response for {d.name}: {e}")
    ax.set_xlabel("sample index")
    ax.set_ylabel("dB")
    n_tgt = len(drawn_targets)
    title_suffix = f" ({n_tgt} different target{'s' if n_tgt > 1 else ''})"
    ax.set_title(f"Best-epoch hard-binarized response{title_suffix}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "response_compare.png", dpi=120)
    plt.close(fig)
    print(f"  → {out_dir / 'response_compare.png'}")

    # 4. Markdown 總表（min_loss / tau range / rollback / coord）
    rows = [
        "| run | coord | epochs | min_loss@ep | tau range | rollbacks |",
        "|-----|-------|--------|-------------|-----------|-----------|",
    ]
    for d in run_dirs:
        try:
            data = _load_record(d)
            coord = _detect_coord(d)
        except FileNotFoundError:
            continue
        rl = data["real_loss"]
        ep = data["epoch"]
        best_ep = ep[int(np.argmin(rl))]
        tau = [t for t in data.get("tau", []) if t is not None]
        tau_str = f"{min(tau):.2f}–{max(tau):.2f}" if tau else "n/a"
        rb = sum(1 for x in data.get("mutation", []) if x)
        n = coord[1] - coord[0]
        rows.append(f"| {d.name} | {n}×{n} | {len(rl)} | {min(rl):.3f}@ep{best_ep} | {tau_str} | {rb} |")
    summary = "\n".join(rows)
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(f"  → {out_dir / 'summary.md'}")
    print("\n" + summary)


if __name__ == "__main__":
    main()
