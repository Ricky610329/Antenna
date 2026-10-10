"""Freeze the small data behind the extended-abstract figures (read-only, run once).

Usage: OMP_NUM_THREADS=1 python docs/paper/freeze_extended_abstract_data.py

The sources are gitignored (pattern-browser snapshot of 2026-08-10, R80 best-render
NPZ), so the derived numbers are written, with source SHA-256, to
docs/paper/extended-abstract-figure-data.json. Nothing here runs HFSS or trains.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SNAP = ROOT / "application/pattern_browser/data"
LOCAL = ROOT / "docs/log/assets/r80_local_variation_result_20261009.json"
SYM = ROOT / "docs/paper/symmetric-best-evidence.json"
R54 = ROOT / "docs/log/assets/round-54/r54_close_analysis.md"
OUT = ROOT / "docs/paper/extended-abstract-figure-data.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def r54_bridge_s0(ids: list[str]) -> dict:
    """S0-mesh WM with 0.10 mm diagonal bridges, from the R54 probe table (section 7).

    The snapshot's db100_wm for these probe parents is an S1-mesh value, so it is not
    used for the no-bridge vs bridged comparison."""
    lines = R54.read_text(encoding="utf-8").splitlines()
    head = next(i for i, l in enumerate(lines) if l.startswith("| 親 | S0 原wm | S0 Δ50 | S0 Δ75 | S0 Δ100 |"))
    out = {}
    for line in lines[head + 2:]:
        if not line.startswith("| `"):
            break
        cells = [c.strip() for c in line.strip("|").split("|")]
        pid = cells[0].strip("`")
        if pid in ids:
            wm, delta = float(cells[1]), float(cells[4])
            out[pid] = {"s0_wm": wm, "s0_delta_0p10": delta, "s0_bridged_wm": round(wm + delta, 2)}
    assert sorted(out) == sorted(ids), out
    return {"source": rel(R54), "source_sha256": sha(R54), "table": "section 7, column S0 Δ100", "rows": out}


def snapshot() -> dict:
    packed = np.load(SNAP / "patterns.npz")["packed"]
    bits = np.unpackbits(packed, axis=1)[:, :625].reshape(-1, 25, 25)
    meta = json.loads((SNAP / "meta.json").read_text(encoding="utf-8"))
    assert len(meta) == len(bits)
    col = lambda k: np.array([np.nan if m.get(k) is None else m[k] for m in meta], float)
    wm, lo, n8 = col("wm"), col("lo"), col("n8")
    asym = (bits[:, :, :12] != bits[:, :, 13:][:, :, ::-1]).mean(axis=(1, 2))

    def unique(mask):   # first occurrence of each distinct pattern
        seen, keep = set(), []
        for i in np.flatnonzero(mask):
            key = packed[i].tobytes()
            if key not in seen:
                seen.add(key)
                keep.append(i)
        return np.array(keep)

    passing = unique(np.isfinite(wm) & (wm >= 0))
    strong = unique(np.isfinite(wm) & (wm >= .3))
    n8p = n8[passing].astype(int)
    scatter = passing[np.isfinite(lo[passing])]
    low = scatter[lo[scatter] <= 0]
    by_id = {}
    for m in meta:
        by_id.setdefault(m["id"], m)
    probes = [m for m in meta if (m.get("store") or "").startswith("dedust_r55sym")]
    r55 = []
    for m in sorted(probes, key=lambda m: m["id"]):
        style, parent = m["id"][5], m["id"][7:]    # y55s_<L|R|O>_<parent>
        r55.append({"id": m["id"], "style": style, "parent": parent,
                    "parent_wm": by_id[parent]["wm"], "wm": m["wm"]})
    drops = [r["wm"] - r["parent_wm"] for r in r55]
    # Low-side out-of-band record (docs/records.json usable_lo), measured without bridges.
    k = next(i for i, m in enumerate(meta) if m["id"] == "c41grp2p02_02")
    resp, rad = np.load(SNAP / "resp.npz"), np.load(SNAP / "rad.npz")
    assert resp["ids"][k] == rad["ids"][k] == meta[k]["id"] and rad["has_rad"][k]
    s11, gain = (resp["resp"][k, c].astype(float) for c in (0, 1))
    assert abs(gain[:4].max() - meta[k]["lo"]) < .01   # 24-25.5 GHz, float16 storage
    oob = {"id": meta[k]["id"], "store": meta[k]["store"], "bridge": "none (S0)",
           "bits": bits[k].tolist(), "S11_db": [round(v, 3) for v in s11],
           "RealizedGainTotal_dbi": [round(v, 3) for v in gain],
           "theta_deg": rad["theta"].astype(float).tolist(),
           "phi0_db": [round(float(v), 3) for v in rad["phi0"][k]],
           "phi90_db": [round(float(v), 3) for v in rad["phi90"][k]],
           "snapshot": {f: meta[k][f] for f in ("wm", "rad", "lo", "n8", "ndiag")},
           "note": "float16 snapshot curves; record value -3.46 dBi notarized 3/3 (docs/records.json). "
                   "Snapshot main rows are no-bridge (build_index.py excludes ~ variants); bridged S0 value in r54_bridge_s0."}
    return {
        "source": {"meta": rel(SNAP / "meta.json"), "meta_sha256": sha(SNAP / "meta.json"),
                   "resp_sha256": sha(SNAP / "resp.npz"), "rad_sha256": sha(SNAP / "rad.npz"),
                   "patterns": rel(SNAP / "patterns.npz"), "patterns_sha256": sha(SNAP / "patterns.npz"),
                   "snapshot_date": "2026-08-10", "rows": len(meta),
                   "note": "Pattern-browser snapshot; mixed bridge conditions and selection purposes, descriptive only."},
        "asymmetry_definition": "A = mean XOR of the left 12 columns vs the mirrored right 12 columns (0 = exact left-right mirror).",
        "passing_unique": {
            "definition": "distinct patterns with wm >= 0 (first occurrence)",
            "count": int(len(passing)),
            "n8_histogram": {str(k): int(v) for k, v in zip(*np.unique(n8p, return_counts=True))},
            "n8_median": float(np.median(n8p)), "n8_p90": float(np.percentile(n8p, 90)),
            "n8_at_least_13": int(np.sum(n8p >= 13)),
            "asym_lo_points": [[round(float(asym[i]), 4), round(float(lo[i]), 2)] for i in scatter],
            "lo_le_0_count": int(len(low)), "lo_le_0_min_asym": round(float(asym[low].min()), 4),
        },
        "wm_ge_0p3_unique": {
            "count": int(len(strong)),
            "exact": int(np.sum(asym[strong] == 0)),
            "near": int(np.sum((asym[strong] > 0) & (asym[strong] <= .1))),
            "asym": int(np.sum(asym[strong] > .1)),
        },
        "r55_posthoc_symmetrization": {
            "rows": r55, "count": len(r55), "parents": len({r["parent"] for r in r55}),
            "median_drop_db": round(float(np.median(drops)), 2),
            "note": "13 parents x L/R/O mirror styles (duplicates removed in R55); docs/log/round-55-finetune.md",
        },
        "oob_record": oob,
        "r54_bridge_s0": r54_bridge_s0(["c48nq1p05_16", "c41grp2p02_02"]),
    }


def local32() -> dict:
    data = json.loads(LOCAL.read_text(encoding="utf-8"))
    rows = [{"rank": r["selection_rank"], "sm_lcb_db": round(r["prospective_lcb_score"], 3),
             "sm_mean_db": round(r["prospective_wmmean"], 3),
             "hfss_wm_db": round(r["actual"]["factory_margin_db"], 4)} for r in data["rows"]]
    assert len(rows) == 32
    return {"source": rel(LOCAL), "source_sha256": sha(LOCAL),
            "anchor_wm_db": round(data["trial_identity"]["anchor_raw_hfss_wm_db"], 4),
            "spearman_lcb_vs_wm": round(data["outcomes"]["selected_cohort_spearman"]["prospective_lcb_vs_actual_wm"]["rho"], 3),
            "best_rank": data["outcomes"]["max_wm_selection_rank"], "rows": rows,
            "note": "SM-selected 32 neighbours (4 physical pixels) of the -0.25 dB anchor; profile-holdout model data-v031."}


def symmetric_radiation() -> dict:
    sym = json.loads(SYM.read_text(encoding="utf-8"))
    npz = ROOT / sym["sources"]["npz"]["path"]
    assert sha(npz) == sym["sources"]["npz"]["sha256"]
    z = np.load(npz)
    theta = z["radiation_theta_deg"]
    phi0, phi90 = z["phi0_db"][0], z["phi90_db"][0]
    window = np.abs(theta) <= 45
    margin = {k: float((c[window] - c[theta == 0][0]).min() + 3) for k, c in (("phi0", phi0), ("phi90", phi90))}
    assert abs(min(margin.values()) - sym["margins_db"]["radiation_window_28ghz"]) < 1e-3, margin
    return {"source": sym["sources"]["npz"], "quantity": "dB(GainTotal) cuts at 28 GHz, 2-degree grid",
            "theta_deg": theta.tolist(),
            "phi0_db": [round(float(v), 4) for v in phi0], "phi90_db": [round(float(v), 4) for v in phi90],
            "window_margin_db": {k: round(v, 4) for k, v in margin.items()}}


def main() -> None:
    out = {"schema_version": 1, "scope": "Frozen inputs for docs/paper/extended_abstract_figures.py; no new EM simulation.",
           "snapshot": snapshot(), "r80_local32": local32(), "symmetric_best_radiation": symmetric_radiation()}
    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    s = out["snapshot"]
    print(OUT.relative_to(ROOT).as_posix(), OUT.stat().st_size, "bytes")
    print({k: v for k, v in s["passing_unique"].items() if k != "asym_lo_points"})
    print(s["wm_ge_0p3_unique"], s["r55_posthoc_symmetrization"]["count"], s["r55_posthoc_symmetrization"]["median_drop_db"])
    print(out["symmetric_best_radiation"]["window_margin_db"], out["r80_local32"]["spearman_lcb_vs_wm"])


if __name__ == "__main__":
    main()
