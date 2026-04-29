"""對 phase 1（連續相位）run 做後處理量化 + binary 評估。

**動機**：實驗顯示 BinarySTE direct GD 表現遠差於「continuous GD → hard」
（−3.34 vs +3.05 dB suppression）。**位元遷移的 phase 2 微調可能多此一舉**——
直接拿 phase 1 連續權重的硬二值化輸出可能就是最佳。

本 script:
1. 載入指定 phase 1 run 的 best epoch generator
2. 對 10 個 sample target 各跑一次 forward
3. 直接 hard-binarize（不經過 phase 2 fine-tune），算 suppression 指標
4. 輸出 markdown 摘要 + 樣本三聯圖

用法：
    python script/post_quantize_eval.py result/RIS-phase1-v4
"""

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
from antenna.ris import RISSimulator
from antenna.utils.config import config

_GENS = (BiasedGumbelSigmoidGEN, GumbelSigmoidGEN, SigmoidGEN)


def load_gen(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location=config.device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    last_err = None
    for cls in _GENS:
        gen = cls()
        try:
            gen.load_state_dict(state)
            gen.eval()
            return gen
        except RuntimeError as e:
            last_err = e
    raise RuntimeError(f"無法載入 {ckpt_path}：{last_err}")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    run_dir = Path(sys.argv[1]).resolve()
    config.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 取 best epoch
    import pickle
    rec = pickle.load(open(run_dir / "temp.record", "rb"))["_data"]
    rl = rec["real_loss"]; ep = rec["epoch"]
    best_ep = int(ep[int(np.argmin(rl))])
    ckpt = run_dir / "checkpoint" / f"generator_{best_ep}.pth"
    print(f"best epoch: {best_ep}, min_loss: {min(rl):.2f}")

    # 推測 element_num（從第一個 ckpt 形狀）
    state_first = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = state_first.get("model_state_dict", state_first)
    fc_bias_keys = [k for k in sd if k.endswith(".bias") and "fc_patch" in k]
    pattern_size = sd[fc_bias_keys[-1]].shape[0]
    n = int(pattern_size**0.5)
    assert n * n == pattern_size, f"非方形 pattern_size={pattern_size}"
    print(f"detected element_num={n}")

    AntennaPattern.setDefaultCoordinate((0, n, 0, n))
    AntennaResponse.registerLabels("response", x="ris")
    # 用標準 target placeholder，下面會替換
    AntennaResponse.target(side=-20.0, center=0.0, width=(140, 0, 40, 0, 181), label="response", add=True)

    sim = RISSimulator(element_num=n)
    gen = load_gen(ckpt).to(config.device)

    # 10 個變化 target（與 inspect_ris_run.dump_samples 一致）
    sample_configs = []
    for i in range(10):
        center_pos = 110 + i * 14
        plateau_w = 20 + (i % 5) * 10
        left_w = max(0, center_pos - plateau_w // 2)
        right_w = max(0, 361 - left_w - plateau_w)
        sample_configs.append((-20.0, 0.0, (left_w, 0, plateau_w, 0, right_w)))

    out_dir = run_dir / "pic" / "post_quantize"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    THETA_DEG = np.arange(-90, 90.1, 0.5)
    for idx, (side, center, width) in enumerate(sample_configs, start=1):
        AntennaResponse.registerLabels("response", x="ris")
        AntennaResponse.registerTargetResponse(side, center, width, "response")
        target = AntennaResponse.target.concat().to(config.device)

        with torch.no_grad():
            soft = gen(target)
            # 直接硬二值化——不經 BinarySTE
            hard = (soft > 0.5).float().reshape(n, n)
            resp = sim(hard)["response"].cpu().numpy()

        on_count = int(hard.sum())
        on_rate = on_count / (n * n)
        main_lo, main_hi = width[0], width[0] + width[2]
        main_idx = np.arange(main_lo, min(main_hi, len(resp)))
        side_idx = np.array([i for i in range(len(resp)) if i not in set(main_idx.tolist())])
        main_peak = float(resp[main_idx].max()) if len(main_idx) else float("nan")
        side_max = float(resp[side_idx].max()) if len(side_idx) else float("nan")
        suppression = main_peak - side_max

        rows.append((idx, width[2], width[0], on_rate, main_peak, side_max, suppression))

        # 三聯圖
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(np.arange(len(target)), target.cpu().numpy(), color="tab:blue", linewidth=2)
        axes[0].set_title(f"target plateau@{width[0]}-{width[0]+width[2]}")
        axes[0].set_xlabel("idx"); axes[0].set_ylabel("dB"); axes[0].grid(alpha=0.3)
        axes[0].set_ylim(-30, 5)

        axes[1].imshow(hard.cpu().numpy(), cmap="gray_r", vmin=0, vmax=1)
        axes[1].set_title(f"post-quantized pattern\n{on_count}/{n*n} on ({on_rate:.0%})")
        axes[1].axis("off")

        axes[2].plot(np.arange(len(target)), target.cpu().numpy(), label="target", linewidth=1.5, alpha=0.7)
        axes[2].plot(np.arange(len(resp)), resp, label="actual", linewidth=1.5)
        axes[2].axvspan(main_lo, main_hi, color="green", alpha=0.08)
        axes[2].set_title(
            f"main_peak={main_peak:+.2f}, side_max={side_max:+.2f}, suppression={suppression:+.2f} dB"
        )
        axes[2].set_xlabel("idx"); axes[2].set_ylabel("dB"); axes[2].legend(); axes[2].grid(alpha=0.3)

        fig.suptitle(f"Sample {idx}/10 — post-quantized phase 1 (best epoch {best_ep})")
        fig.tight_layout()
        fig.savefig(out_dir / f"sample_{idx:02d}.png", dpi=110)
        plt.close(fig)

    # summary
    lines = [
        "# Post-quantization 評估（phase 1 連續權重 → hard binary）",
        f"Run: `{run_dir.name}` (best epoch {best_ep})",
        "",
        "| # | plateau_w | plateau_start | on% | main_peak dB | side_max dB | suppression dB |",
        "|---|-----------|---------------|-----|--------------|-------------|----------------|",
    ]
    for idx, w, st, on, mp, sm, sup in rows:
        lines.append(
            f"| {idx} | {w} | {st} | {on:.0%} | {mp:.2f} | {sm:.2f} | {sup:+.2f} |"
        )
    sup_vals = [r[6] for r in rows]
    lines += [
        "",
        f"**suppression mean**: {np.mean(sup_vals):+.2f} dB, "
        f"min={np.min(sup_vals):+.2f}, max={np.max(sup_vals):+.2f}",
        "",
        "對照：",
        "- v1 (binary STE): mean=−4.08 dB",
        "- v2 (反 collapse 三 combo): mean=−1.84 dB",
        "- v3 (位元遷移): mean=−2.21 dB",
        "- direct GD (continuous→hard upper bound): +3.05 dB",
    ]
    summary = "\n".join(lines)
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
