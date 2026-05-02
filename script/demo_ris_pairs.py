"""產生 10 個不同 RIS pattern 與其遠場響應的配對展示圖。

每個 pattern 都是 25x25 的相位 mask（值 0~1，代表 0~pi 相位），
透過 RISSimulator 解析式陣列因子計算遠場 dB 響應。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from antenna.ris import RISSimulator

ELEMENT_NUM = 25
THETA_DEG = np.arange(-90, 90.1, 0.5)  # 與 simulate_ris._calAF 的取樣一致


def make_patterns(n_side: int, seed: int = 0) -> dict[str, torch.Tensor]:
    """構造 10 個有代表性的相位 pattern。"""
    rng = np.random.default_rng(seed)
    patterns: dict[str, torch.Tensor] = {}

    # 1. 全 0（所有元件相位 0）
    patterns["all_zeros"] = torch.zeros(n_side, n_side)

    # 2. 全 1（所有元件相位 pi）
    patterns["all_ones"] = torch.ones(n_side, n_side)

    # 3. 棋盤格
    cb = np.indices((n_side, n_side)).sum(axis=0) % 2
    patterns["checkerboard"] = torch.tensor(cb, dtype=torch.float32)

    # 4. 縱向條紋
    stripes_v = np.tile([0, 1], (n_side, n_side // 2 + 1))[:, :n_side]
    patterns["vertical_stripes"] = torch.tensor(stripes_v, dtype=torch.float32)

    # 5. 橫向條紋
    stripes_h = np.tile([[0], [1]], (n_side // 2 + 1, n_side))[:n_side]
    patterns["horizontal_stripes"] = torch.tensor(stripes_h, dtype=torch.float32)

    # 6. 隨機二值（50%）
    patterns["random_binary"] = torch.tensor(
        rng.integers(0, 2, (n_side, n_side)), dtype=torch.float32
    )

    # 7. 中心圓盤為 1，其餘為 0
    yy, xx = np.mgrid[:n_side, :n_side]
    cx = cy = (n_side - 1) / 2
    disk = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (n_side / 4) ** 2
    patterns["center_disk"] = torch.tensor(disk, dtype=torch.float32)

    # 8. 連續相位坡度（線性）
    ramp = np.linspace(0, 1, n_side)[None, :].repeat(n_side, axis=0)
    patterns["linear_ramp"] = torch.tensor(ramp, dtype=torch.float32)

    # 9. 同心圓波（連續）
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    rings = (np.sin(r * np.pi / 3) + 1) / 2
    patterns["concentric_rings"] = torch.tensor(rings, dtype=torch.float32)

    # 10. 高斯雜訊（連續，0~1 clamp）
    noise = rng.normal(0.5, 0.25, (n_side, n_side)).clip(0, 1)
    patterns["gaussian_noise"] = torch.tensor(noise, dtype=torch.float32)

    return patterns


def main():
    out_dir = Path(__file__).resolve().parent / "ris_pairs"
    out_dir.mkdir(exist_ok=True)

    sim = RISSimulator(element_num=ELEMENT_NUM)
    patterns = make_patterns(ELEMENT_NUM)

    # 個別三聯圖（pattern / 響應曲線）
    fig, axes = plt.subplots(10, 2, figsize=(11, 30))
    summary_rows = []
    for row, (name, pat) in enumerate(patterns.items()):
        with torch.no_grad():
            resp = sim(pat)["response"].cpu().numpy()  # shape (361,)

        # pattern heatmap
        ax_p = axes[row, 0]
        im = ax_p.imshow(pat.numpy(), cmap="gray", vmin=0, vmax=1)
        ax_p.set_title(f"#{row + 1} {name}  (mean phase / π = {pat.mean():.2f})")
        ax_p.set_xticks([])
        ax_p.set_yticks([])
        plt.colorbar(im, ax=ax_p, fraction=0.046, pad=0.04)

        # response curve
        ax_r = axes[row, 1]
        ax_r.plot(THETA_DEG, resp, linewidth=1.0)
        ax_r.set_xlabel("theta (deg)")
        ax_r.set_ylabel("normalized |AF| (dB)")
        ax_r.set_ylim(-50, 2)
        ax_r.grid(True, alpha=0.3)
        peak_idx = int(np.argmax(resp))
        peak_theta = THETA_DEG[peak_idx]
        ax_r.axvline(peak_theta, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
        ax_r.set_title(f"peak @ θ={peak_theta:+.1f}°  (max=0 dB by norm.)")

        summary_rows.append((name, peak_theta, float(resp.min())))

    fig.suptitle(
        f"RIS far-field response — element_num={ELEMENT_NUM}, freq=28 GHz, inc θ=-40°/φ=90°",
        fontsize=12,
        y=1.0,
    )
    plt.tight_layout()
    fig_path = out_dir / "ris_10_pairs.png"
    plt.savefig(fig_path, dpi=110, bbox_inches="tight")
    plt.close()

    # 簡短摘要 (peak 角度 / sidelobe 最低值)
    print(f"saved: {fig_path}")
    print(f"{'pattern':<20s} {'peak θ (deg)':>14s} {'min dB':>10s}")
    for name, peak, mn in summary_rows:
        print(f"{name:<20s} {peak:>14.1f} {mn:>10.2f}")


if __name__ == "__main__":
    main()
