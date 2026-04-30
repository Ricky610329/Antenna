"""
Round 71 — Dataset 視覺化 gallery

把 dataset_v1 36 entries 每個畫:
- 兩張 binary pattern (rw=0 vs rw=2)
- 兩條 response curve
- 標 metrics (worst, ripple, flat-top compliance)

證據導向 — 看看 binary RIS 設計空間長什麼樣。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


THETA_DEG = np.arange(-90, 90.1, 0.5)


def render_one_entry(entry: dict, root: Path, ax_pat0, ax_pat2, ax_resp):
    """一個 entry 畫到 3 個 axes 上 (rw=0 pattern, rw=2 pattern, response 比較)。"""
    cfg = entry["config"]
    main_lo, main_hi = entry["main_idx_range"]

    rw0 = next((p for p in entry["pareto"] if p["ripple_weight"] == 0.0), None)
    rw2 = next((p for p in entry["pareto"] if p["ripple_weight"] == 2.0), None)

    if rw0:
        pat0 = np.load(root / rw0["pattern_file"])
        resp0 = np.load(root / rw0["response_file"])
        ax_pat0.imshow(pat0, cmap="binary", vmin=0, vmax=1, aspect="equal")
        m = rw0["metrics"]
        ax_pat0.set_title(
            f"rw=0  worst={m['worst_supp']:+.1f}\nripple={m['main_ripple']:.1f}",
            fontsize=8,
        )
    ax_pat0.set_xticks([])
    ax_pat0.set_yticks([])

    if rw2:
        pat2 = np.load(root / rw2["pattern_file"])
        resp2 = np.load(root / rw2["response_file"])
        ax_pat2.imshow(pat2, cmap="binary", vmin=0, vmax=1, aspect="equal")
        m = rw2["metrics"]
        flat = "FLAT" if m["main_below_3dB"] == 0 else "NoFlat"
        ax_pat2.set_title(
            f"rw=2 [{flat}] worst={m['worst_supp']:+.1f}\nripple={m['main_ripple']:.1f}",
            fontsize=8,
        )
    ax_pat2.set_xticks([])
    ax_pat2.set_yticks([])

    # Response comparison
    if rw0:
        ax_resp.plot(THETA_DEG, resp0, "r-", linewidth=0.7, alpha=0.7, label="rw=0")
    if rw2:
        ax_resp.plot(THETA_DEG, resp2, "b-", linewidth=0.7, alpha=0.9, label="rw=2")
    main_lo_deg = -90 + main_lo * 0.5
    main_hi_deg = -90 + main_hi * 0.5
    ax_resp.axvspan(main_lo_deg, main_hi_deg, color="green", alpha=0.15)
    ax_resp.axhline(-3, color="black", linewidth=0.4, linestyle="--", alpha=0.5)
    ax_resp.axhline(0, color="black", linewidth=0.4, alpha=0.5)
    ax_resp.set_ylim(-40, 5)
    ax_resp.set_xlim(-90, 90)
    ax_resp.set_xticks([-60, 0, 60])
    ax_resp.set_yticks([-30, 0])
    ax_resp.tick_params(axis="both", labelsize=6)
    ax_resp.set_title(
        f"{cfg['freq_ghz']}GHz n={cfg['n']} θc={cfg['target_theta_c']:+.0f}° "
        f"w={cfg['target_width_deg']:.0f}°",
        fontsize=8,
    )


def main() -> None:
    root = Path("outputs/dataset_v1")
    entries = []
    with open(root / "entries.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    n = len(entries)
    cols = 6  # 6 entries per row
    rows = (n + cols - 1) // cols
    print(f"Rendering {n} entries in {rows} rows × {cols} cols")

    fig, axes = plt.subplots(rows, cols * 3, figsize=(cols * 3 * 1.4, rows * 1.6))
    # axes shape: (rows, cols*3)

    for i, e in enumerate(entries):
        r = i // cols
        c = i % cols
        ax_pat0 = axes[r, c * 3]
        ax_pat2 = axes[r, c * 3 + 1]
        ax_resp = axes[r, c * 3 + 2]
        render_one_entry(e, root, ax_pat0, ax_pat2, ax_resp)

    # Hide unused
    for i in range(len(entries), rows * cols):
        r = i // cols
        c = i % cols
        for j in range(3):
            axes[r, c * 3 + j].axis("off")

    fig.suptitle(
        "Dataset v1 (36 entries) — Binary RIS Pattern + Response\n"
        "Each entry: [rw=0 pattern] [rw=2 pattern] [response curves: red=rw=0, blue=rw=2]",
        fontsize=11,
    )
    fig.tight_layout()
    out = "outputs/dataset_v1_gallery.png"
    fig.savefig(out, dpi=100, bbox_inches="tight")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
