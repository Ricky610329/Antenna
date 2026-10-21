"""R92 非 a0 家族普查（2026-10-22）：v6 在 a0 家族撞結構牆（低阻槓桿一推，通帶／高阻各掉 4–7 dB）。
從學長 24–32 GHz 歷史設計中挑「兩埠斷開、通帶好、與 a0 最佳差很多」的設計，以 11–40 GHz 重量，
看是否有家族天生帶低頻抑制（v6 低阻 11–19 ≤ −20）。挑法：通帶 margin 前段池內做 maxmin Hamming 分散。

  python -m script.r92_explore_seeds --n 24 --out <dir>
  python -m script.r81_curve_sm enqueue --input <dir> --name r81v6x01 --prio -3 --shards 3 --config configs/dual_r81_v6.yaml
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from antenna.measurement import measurement_id, score_spec_id
from script import r81_curve_sm as sm
from script.profiled_batch import load_profile_config, pattern_sha256, validate_input

V6 = sm.REPO / "configs" / "dual_r81_v6.yaml"


def maxmin_pick(P, n, start=0):
    """bool (N,625) 中以 Hamming 距離做 maxmin 挑 n 個（從 start 開始）。"""
    picks = [start]
    d = (P != P[start]).sum(1)
    while len(picks) < min(n, len(P)):
        i = int(np.argmax(d))
        picks.append(i)
        d = np.minimum(d, (P != P[i]).sum(1))
    return picks


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--pool", type=int, default=600, help="通帶 margin 前幾名進候選池")
    ap.add_argument("--min-dist", type=int, default=150, help="與 a0 v6a 最佳的最小 Hamming 距離")
    ap.add_argument("--ref", default="r81v6mg15_005_c943952c", help="a0 參考設計 id")
    ap.add_argument("--name", default="r81v6x01")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    cfg = sm.load_cfg(V6)
    grid = np.asarray(sm.frequency_grid(cfg.measurement))
    hist = sm.load_history(sm.HIST_DIR, grid)
    wide = sm.load_wide(sm.WIDE_ROOT, cfg, grid)
    ref = next(np.asarray(r["pattern"]).reshape(-1) > 0.5 for r in wide if r["id"] == args.ref)
    sl = {s[0]: s for s in sm.band_slices(cfg.score_spec, grid)}
    _, li, lo, hi, *_ = sl["s21_pass"]
    P = np.asarray(hist["P"]).reshape(-1, 625) > 0.5
    Y, M = np.asarray(hist["Y"]), np.asarray(hist["M"])
    ok = M[:, li, lo:hi].all(1)
    pm = np.where(ok, np.nan_to_num(Y[:, li, lo:hi], nan=-99).min(1) + 3.0, -np.inf)
    P = np.array([sm.enforce_dual_geometry(p).reshape(-1) for p in P])
    disc = np.array([not sm.pads_connected(p) for p in P])
    far = (P != ref).sum(1) >= args.min_dist
    cand = np.flatnonzero(disc & far & np.isfinite(pm))
    cand = cand[np.argsort(-pm[cand])][:args.pool]
    print(f"歷史 {len(P)}；兩埠斷開 {int(disc.sum())}；與 a0 差 ≥ {args.min_dist} 格 {int((disc & far).sum())}；"
          f"池 {len(cand)}（通帶 margin {pm[cand].min():+.2f}～{pm[cand].max():+.2f}）")
    known = {r["sha"] for r in wide} | sm.input_hashes(sm.WIDE_ROOT)
    picks = []
    for j in maxmin_pick(P[cand], args.n * 2):
        i = int(cand[j])
        h = pattern_sha256(P[i].reshape(25, 25))
        if h not in known:
            known.add(h)
            picks.append((i, h))
        if len(picks) >= args.n:
            break
    pcfg = load_profile_config(V6)
    mid, sid = measurement_id(pcfg.measurement), score_spec_id(pcfg.score_spec)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    man = []
    for k, (i, h) in enumerate(picks):
        rid = f"{args.name}_{k:03d}_{h[:8]}"
        torch.save(torch.tensor(P[i].reshape(25, 25).astype(np.float32)), out / f"{rid}.pt")
        man.append({"id": rid, "batch": 1, "port": "dual", "kind": "explore_seed", "measurement_id": mid,
                    "score_spec_id": sid, "pattern_file": f"{rid}.pt", "pattern_sha256": h,
                    "lineage_id": str(hist["lineage"][i]) if "lineage" in hist else "hist",
                    "parent_id": f"hist:{hist['sha'][i][:16]}", "selection_arm": "explore_seed", "op": "hist_seed",
                    "hist_pass_margin": float(pm[i]), "dist_a0": int((P[i] != ref).sum())})
    (out / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "measurement.json").write_text(json.dumps(pcfg.measurement, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "score_spec.json").write_text(json.dumps(pcfg.score_spec, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "config.yaml").write_text(V6.read_text(encoding="utf-8"), encoding="utf-8")
    validate_input(out, pcfg)
    print(f"→ {out}：{len(man)} 筆；與 a0 差 {min(m['dist_a0'] for m in man)}–{max(m['dist_a0'] for m in man)} 格；"
          f"歷史通帶 margin {min(m['hist_pass_margin'] for m in man):+.2f}～{max(m['hist_pass_margin'] for m in man):+.2f}")


if __name__ == "__main__":
    sys.exit(main())
