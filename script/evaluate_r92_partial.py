"""Quick eval of saved R92 partial results."""

import numpy as np
import torch
import sys
sys.path.insert(0, "script")
from antenna.ris import RISSimulator
from antenna.utils.config import config
from methodology_demo import evaluate_metrics

config.device = "cuda:0"

main_lo, main_hi = 162, 192

for n in [41, 51]:
    pat = np.load(f"outputs/r92_aperture_scaling/n{n}_best_pattern.npy")
    resp = np.load(f"outputs/r92_aperture_scaling/n{n}_best_response.npy")
    m = evaluate_metrics(resp, main_lo, main_hi)
    print(f"n={n} ({n*0.5}λ aperture, {n*n} elements):")
    print(f"  worst supp:    {m['worst_supp']:+.2f} dB")
    print(f"  ripple:        {m['main_ripple']:.2f} dB")
    print(f"  flat-top compliant: {m['flat_top_compliant']}")
    print(f"  side max:      {m['side_max']:+.2f} dB")
    print(f"  on-rate:       {pat.mean()*100:.1f}%")
    print()
