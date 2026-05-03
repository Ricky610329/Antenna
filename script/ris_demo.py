"""ris_demo — `ris_core` 的最小 runnable 範例。

跑兩個 spec：
  1. 解析模式 (analytical sim 同時做梯度與 eval)
  2. Surrogate-loop 模式 (warm-start surrogate 做梯度，sim 做 eval)

並印出每 seed 的 worst / flat-top / 是否啟用 early-stop。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# 讓 `python script/ris_demo.py` 不裝套件也能 import sibling
sys.path.insert(0, str(Path(__file__).parent))

from ris_core import (
    build_warmstart_surrogate,
    optimize_ris_1bit,
    optimize_ris_1bit_multifreq,
)

DEVICE = "cuda:0"
SPEC = dict(n=51, inc_deg=51.0, freq_hz=38e9, width_deg=10.0, n_restarts=2, gd_steps=300)


def _print_result(label: str, r, elapsed: float) -> None:
    print(f"\n[{label}] tier={r.recipe['tier']} (rw={r.recipe['rw']}, lam={r.recipe['lambda_mean']})")
    for s in r.seed_results:
        print(
            f"  seed {s['seed']}: worst={s['worst']:+.2f} dB  ripple={s['ripple']:.2f}  "
            f"flat_top={s['flat_top']}  early_stop={s['used_early_stop']}"
        )
    print(
        f"  best worst={r.best['worst']:+.2f} dB | flat={r.n_flat_top}/{r.n_restarts} | "
        f"ES used in {r.n_early_stop_used}/{r.n_restarts} | took {elapsed:.1f}s"
    )


if __name__ == "__main__":
    print("=" * 80)
    print("ris_demo — 1-bit RIS pipeline smoke test")
    print(f"  spec: {SPEC}")
    print("=" * 80)

    t0 = time.time()
    r_ana = optimize_ris_1bit(**SPEC, device=DEVICE)
    _print_result("analytical", r_ana, time.time() - t0)

    sur = build_warmstart_surrogate(SPEC["n"], SPEC["freq_hz"], SPEC["inc_deg"], device=DEVICE)
    t0 = time.time()
    r_sur = optimize_ris_1bit(**SPEC, forward_fn=sur, device=DEVICE)
    _print_result("surrogate-loop", r_sur, time.time() - t0)

    # multi-freq smoke (R154)
    t0 = time.time()
    mf = optimize_ris_1bit_multifreq(
        n=SPEC["n"], inc_deg=SPEC["inc_deg"], freqs_hz=[36e9, 38e9, 40e9],
        width_deg=SPEC["width_deg"], n_restarts=1, gd_steps=300, device=DEVICE,
    )
    print(f"\n[multi-freq] best min-worst across 3 freqs = {mf['best_min_worst']:+.2f} dB"
          f" (took {time.time()-t0:.1f}s)")
    for f, m in zip(mf["freqs"], mf["per_seed"][0]["per_freq"]):
        print(f"  {f/1e9:>4g} GHz: worst={m['worst']:+.2f}  flat_top={m['flat_top']}")
