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
from antenna.models.generators.biased_gumbel_sigmoid_gen import BiasedGumbelSigmoidGEN
from antenna.models.generators.gumbel_sigmoid_gen import GumbelSigmoidGEN
from antenna.models.generators.sigmoid_gen import SigmoidGEN
from antenna.models.generators.wide_gumbel_sigmoid_gen import WideGumbelSigmoidGEN
from antenna.ris import RISSimulator
from antenna.utils.config import config

# 已知的 RIS generator 類別（順序 = 嘗試載入順序）
_GEN_REGISTRY = (
    BiasedGumbelSigmoidGEN,
    WideGumbelSigmoidGEN,
    GumbelSigmoidGEN,
    SigmoidGEN,
)


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


def _load_target_from_cfg(run_dir: Path) -> tuple[float, float, tuple] | None:
    """嘗試從 result/<run>/config.yaml 讀出 target (side, center, width)。

    若該 run 是用早期 trainer 跑的（沒寫 config.yaml）則回傳 None，
    呼叫端會 fallback 到預設 target。
    """
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        return None
    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(cfg_path)
        tgt = cfg.response.label_configs.response.target
        return float(tgt.side), float(tgt.center), tuple(tgt.width)
    except Exception:
        return None


def _setup_once(pattern_coord: tuple, run_dir: Path | None = None) -> None:
    """只 register 一次（全域狀態）。target 優先從 run dir 的 config.yaml 讀；
    讀不到 fallback 到預設 V4–V9 target (40-sample plateau)。"""
    global _SETUP_DONE
    AntennaPattern.setDefaultCoordinate(pattern_coord)
    if _SETUP_DONE:
        return
    AntennaResponse.registerLabels("response", x="ris")
    target_args = (-20.0, 0.0, (140, 0, 40, 0, 181))
    if run_dir is not None:
        loaded = _load_target_from_cfg(run_dir)
        if loaded is not None:
            target_args = loaded
            print(f"  loaded target from config.yaml: side={loaded[0]} center={loaded[1]} width={loaded[2]}")
    AntennaResponse.registerTargetResponse(*target_args, "response")
    _SETUP_DONE = True


def load_generator(ckpt_path: Path):
    """載入 RIS generator checkpoint。逐一嘗試 registry 內的類別，第一個 state_dict 對得起來的勝出。"""
    ckpt = torch.load(ckpt_path, map_location=config.device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    last_err: Exception | None = None
    for cls in _GEN_REGISTRY:
        gen = cls()
        try:
            gen.load_state_dict(state)
            gen.eval()
            return gen
        except RuntimeError as e:
            last_err = e
            continue
    raise RuntimeError(f"無法從 {ckpt_path.name} 載入任何已知 generator，最後錯誤：{last_err}")


def plot_pattern_evolution(run_dir: Path, coord: tuple, out: Path) -> None:
    ckpt_dir = run_dir / "checkpoint"
    ckpts = sorted(ckpt_dir.glob("generator_*.pth"), key=lambda p: int(p.stem.split("_")[1]))
    if not ckpts:
        print("  no generator checkpoints found")
        return

    _setup_once(coord, run_dir)

    sample_epochs = [1, len(ckpts) // 2, len(ckpts)]
    sample_epochs = sorted(set(e for e in sample_epochs if 1 <= e <= len(ckpts)))

    fig, axes = plt.subplots(1, len(sample_epochs), figsize=(4 * len(sample_epochs), 4))
    if len(sample_epochs) == 1:
        axes = [axes]

    target = AntennaResponse.target.concat().to(config.device)
    for ax, ep in zip(axes, sample_epochs):
        gen = load_generator(ckpt_dir / f"generator_{ep}.pth")
        with torch.no_grad():
            # gen(target) 已是 [0, 1] 軟輸出（GumbelSigmoid 已套用），不再 sigmoid
            soft = gen(target).detach()
        n = coord[1] - coord[0]
        # 一律 hard binarization（{0, 1} 對應實機 RIS 相位 {0, π}）
        hard = (soft > 0.5).float().cpu().numpy().reshape(n, n)
        on = int(hard.sum())
        ax.imshow(hard, cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(f"epoch {ep}\n{on}/{n * n} on ({100 * on / (n * n):.0f}%)")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  pattern evolution → {out}")


def _forward_pattern(gen, target, n: int) -> np.ndarray:
    """硬二值化 pattern 並丟進 RIS sim，回傳響應 numpy 陣列。

    注意：``gen(target)`` 對 GumbelSigmoidGEN 系列回傳 **已套用 Gumbel-Sigmoid** 後
    的 [0, 1] 軟輸出，不能再 ``sigmoid()`` 一次，否則永遠 > 0.5（hard 全 1）。
    """
    with torch.no_grad():
        soft = gen(target)
        pattern = (soft > 0.5).float().reshape(n, -1)
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
    _setup_once(coord, run_dir)
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


def dump_samples(run_dir: Path, coord: tuple, data: dict, out_dir: Path, n_samples: int = 10) -> None:
    """對 best epoch generator 餵 N 組不同 target，每組存一張 3-panel 圖：
    (input target response, output binary pattern, actual response)。

    用於檢驗 generator 是否真是 conditional：若 N 個不同 target 卻得到
    N 個相同 pattern，就確認 collapse 假說（§8.4d）。
    """
    ckpt_dir = run_dir / "checkpoint"
    real_losses = data.get("real_loss", [])
    epochs = data.get("epoch", [])
    if not real_losses:
        return
    best_ep = int(epochs[int(np.argmin(real_losses))])
    ckpt_path = ckpt_dir / f"generator_{best_ep}.pth"
    if not ckpt_path.exists():
        return

    AntennaPattern.setDefaultCoordinate(coord)
    if not AntennaResponse.target.metadata:
        AntennaResponse.registerLabels("response", x="ris")
        # placeholder target，真正的 target 我們手動產
        AntennaResponse.registerTargetResponse(-20.0, 0.0, (140, 0, 40, 0, 181), "response")

    n = coord[1] - coord[0]
    sim = RISSimulator(element_num=n)
    gen = load_generator(ckpt_path)

    samples_dir = out_dir / "samples"
    samples_dir.mkdir(exist_ok=True)

    # 10 個變化：交替變動 plateau 位置與 plateau 寬度
    sample_configs = []
    for i in range(n_samples):
        # plateau 位置：在 90 至 240 之間移動
        center_pos = 110 + i * 14
        # plateau 寬度：20 ~ 60
        plateau_w = 20 + (i % 5) * 10
        left_w = max(0, center_pos - plateau_w // 2)
        right_w = max(0, 361 - left_w - plateau_w)
        sample_configs.append((-20.0, 0.0, (left_w, 0, plateau_w, 0, right_w)))

    rng = torch.Generator(device="cpu")
    rng.manual_seed(42)

    rows = []
    for idx, (side, center, width) in enumerate(sample_configs, start=1):
        # 重置 target tensor，但保留 labels metadata（不然 registerTargetResponse 會喊「沒有 labels」）
        AntennaResponse.registerLabels("response", x="ris")
        AntennaResponse.registerTargetResponse(side, center, width, "response")
        target = AntennaResponse.target.concat().to(config.device)

        with torch.no_grad():
            # gen(target) 對 GumbelSigmoidGEN 系列已是 [0, 1] 軟輸出（不能再 sigmoid）
            soft = gen(target)
            hard = (soft > 0.5).float().reshape(n, n)
            response = sim(hard)
        if isinstance(response, dict):
            response = next(iter(response.values()))
        resp_np = response.detach().cpu().numpy().flatten()

        target_np = target.detach().cpu().numpy()
        hard_np = hard.cpu().numpy()
        on_count = int(hard_np.sum())

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        x = np.arange(len(target_np))
        axes[0].plot(x, target_np, color="tab:blue", linewidth=2)
        axes[0].set_title(f"Input target response\nplateau@{width[0]}–{width[0] + width[2]} ({width[2]} samples)")
        axes[0].set_xlabel("sample idx")
        axes[0].set_ylabel("dB")
        axes[0].grid(alpha=0.3)
        axes[0].set_ylim(-30, 5)

        axes[1].imshow(hard_np, cmap="gray_r", vmin=0, vmax=1)
        axes[1].set_title(f"Output pattern (hard-binarized)\n{on_count}/{n * n} on ({100 * on_count / (n * n):.0f}%)")
        axes[1].axis("off")

        # ── main beam vs sidelobe 物理指標 ──
        main_lo, main_hi = width[0], width[0] + width[2]
        main_idx = np.arange(main_lo, min(main_hi, len(resp_np)))
        side_idx = np.array([i for i in range(len(resp_np)) if i not in set(main_idx.tolist())])
        main_peak = float(resp_np[main_idx].max()) if len(main_idx) else float("nan")
        side_max = float(resp_np[side_idx].max()) if len(side_idx) else float("nan")
        suppression = main_peak - side_max  # 正值表示 main beam 高於最大 sidelobe

        axes[2].plot(x, target_np, color="tab:blue", linewidth=1.5, label="target", alpha=0.7)
        axes[2].plot(x[: len(resp_np)], resp_np, color="tab:orange", linewidth=1.5, label="actual")
        # 標出 main beam 區間
        axes[2].axvspan(main_lo, main_hi, color="green", alpha=0.08, label="main beam region")
        axes[2].set_title(
            f"Actual response — main_peak={main_peak:.2f} dB, "
            f"side_max={side_max:.2f} dB, suppression={suppression:.2f} dB"
        )
        axes[2].set_xlabel("sample idx")
        axes[2].set_ylabel("dB")
        axes[2].legend(fontsize=9)
        axes[2].grid(alpha=0.3)

        fig.suptitle(f"Sample {idx}/{n_samples} — best epoch {best_ep}")
        fig.tight_layout()
        out_path = samples_dir / f"sample_{idx:02d}.png"
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        rows.append((idx, width[2], width[0], on_count, main_peak, side_max, suppression))

    print(f"  dumped {n_samples} samples → {samples_dir}/")
    # 摘要表 — 報出物理指標而非單純 max
    summary_lines = [
        "| # | plateau_w | plateau_start | pattern on% | main_peak dB | side_max dB | suppression dB |",
        "|---|-----------|---------------|-------------|--------------|-------------|----------------|",
    ]
    for idx, w, start, on, mp, sm, sup in rows:
        summary_lines.append(
            f"| {idx} | {w} | {start} | {100 * on / (n * n):.0f}% | "
            f"{mp:.2f} | {sm:.2f} | {sup:+.2f} |"
        )
    # 整體統計
    sup_vals = [r[6] for r in rows]
    summary_lines.append("")
    summary_lines.append(f"**suppression 統計**：mean={np.mean(sup_vals):+.2f} dB, "
                         f"min={np.min(sup_vals):+.2f}, max={np.max(sup_vals):+.2f} dB")
    summary_lines.append("> 正值 = main beam 高於最大 sidelobe（好）；負值 = 主峰反而被 sidelobe 蓋過（壞）。")
    (samples_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"  summary → {samples_dir / 'summary.md'}")


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
    plot_best_hard_pattern(run_dir, coord, data, pic / "best_pattern_hard.png")
    dump_samples(run_dir, coord, data, pic, n_samples=10)


def plot_best_hard_pattern(run_dir: Path, coord: tuple, data: dict, out: Path) -> None:
    """畫出 best epoch 的硬二值化 pattern（實際部署時會看到的銅箔分布）。"""
    ckpt_dir = run_dir / "checkpoint"
    real_losses = data.get("real_loss", [])
    epochs = data.get("epoch", [])
    if not real_losses:
        return
    best_ep = int(epochs[int(np.argmin(real_losses))])
    ckpt_path = ckpt_dir / f"generator_{best_ep}.pth"
    if not ckpt_path.exists():
        print(f"  no ckpt at best epoch {best_ep}")
        return

    _setup_once(coord, run_dir)
    target = AntennaResponse.target.concat().to(config.device)
    gen = load_generator(ckpt_path)

    n = coord[1] - coord[0]
    with torch.no_grad():
        # gen(target) 已是 [0, 1] 軟輸出，不再 sigmoid
        soft = gen(target).cpu().numpy().reshape(n, n)
        hard = (soft > 0.5).astype(np.float32)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(soft, cmap="gray_r", vmin=0, vmax=1)
    axes[0].set_title(f"epoch {best_ep} soft sigmoid(logits)")
    axes[0].axis("off")
    axes[1].imshow(hard, cmap="gray_r", vmin=0, vmax=1)
    on = int(hard.sum())
    axes[1].set_title(f"hard-binarized (>0.5)\n{on}/{n * n} on ({100 * on / (n * n):.0f}%)")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  best hard pattern → {out}")


if __name__ == "__main__":
    main()
