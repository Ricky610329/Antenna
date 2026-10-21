"""R89 整合批（2026-10-21，Ricky 提議「整合前段設計、不動連通特性下調拓撲」）：前段設計通帶有兩種卡法——
多數卡上緣 30 GHz、少數卡下緣 26 GHz 而 30 GHz 較好。以卡上緣的前幾名為底、30 GHz 較好的前段設計為捐贈者，
把兩者相異的格依位置分成 ≤ max_groups 個小區塊，列舉「移植哪幾塊」的全部組合；過濾掉兩埠接通或改變
大塊金屬結構（≥ big 格的 4 連通塊數）的候選，再用變化量模型與上下文模型的平均分排序挑 n 筆。

  python -m script.r89_merge --n 24 --out <dir>
  python -m script.r81_curve_sm enqueue --input <dir> --name r81v6mg01 --prio -3 --shards 3 --config configs/dual_r81_v6a.yaml
"""
import argparse
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from antenna.measurement import measurement_id, score_spec_id
from script import r81_curve_sm as sm
from script.profiled_batch import load_profile_config, pattern_sha256, validate_input

V6A = sm.REPO / "configs" / "dual_r81_v6a.yaml"
FOUR = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]


def big_components(p, big=4):
    """≥ big 格的金屬 4 連通塊數（零散小島不計）。"""
    lab, n = ndimage.label(p, structure=FOUR)
    return int((np.bincount(lab.ravel())[1:] >= big).sum())


def diff_groups(base, donor, max_groups=5):
    """base 與 donor 相異的格，依 8 連通分塊；塊數超過 max_groups 時反覆合併質心最近的兩塊。"""
    lab, n = ndimage.label(base != donor, structure=np.ones((3, 3)))
    groups = [lab == i for i in range(1, n + 1)]
    while len(groups) > max_groups:
        cen = [np.argwhere(g).mean(0) for g in groups]
        i, j = min(itertools.combinations(range(len(groups)), 2), key=lambda t: np.linalg.norm(cen[t[0]] - cen[t[1]]))
        groups[i] = groups[i] | groups[j]
        del groups[j]
    return groups


def transplants(base, donor, max_groups=5, big=4):
    """列舉移植組合（不含空集合與全集＝捐贈者本身）；回傳 [(pattern, 用到的塊號)]，已過濾接通與大塊結構改變。"""
    groups = diff_groups(base, donor, max_groups)
    nb = big_components(base, big)
    out = []
    for k in range(1, len(groups)):
        for sub in itertools.combinations(range(len(groups)), k):
            q = base.copy()
            for i in sub:
                q[groups[i]] = donor[groups[i]]
            q = sm.enforce_dual_geometry(q)
            if sm.pads_connected(q) or big_components(q, big) != nb:
                continue
            out.append((q, list(sub)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--bases", type=int, default=3, help="底：通帶最差點在 30 GHz 的前幾名")
    ap.add_argument("--donors", type=int, default=5, help="捐贈者：WM > --donor-wm 中 30 GHz S21 最好的幾筆")
    ap.add_argument("--donor-wm", type=float, default=-0.5)
    ap.add_argument("--max-groups", type=int, default=5)
    ap.add_argument("--big", type=int, default=4)
    ap.add_argument("--per-pair", type=int, default=3)
    ap.add_argument("--name", default="r81v6mg01")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    cfg = sm.load_cfg(V6A)
    grid = np.asarray(sm.frequency_grid(cfg.measurement))
    wide = sm.load_wide(sm.WIDE_ROOT, cfg, grid)
    i30 = int(np.argmin(abs(grid - 30)))
    pb = (grid >= 26) & (grid <= 30)
    cand = []
    for r in wide:
        if not re.search(r"r81v6b\d+s\d$", r["store"]):
            continue
        sc = sm.true_wm(r, cfg, grid)
        if sc:
            cand.append((sc["worst_margin"], r))
    cand.sort(key=lambda t: -t[0])
    bases = [r for w, r in cand if grid[pb][np.argmin(r["y"][1][pb])] == 30.0][:args.bases]
    donors = sorted([r for w, r in cand if w > args.donor_wm and r not in bases],
                    key=lambda r: -r["y"][1][i30])[:args.donors]
    pats, meta = [], []
    known = {r["sha"] for r in wide} | sm.input_hashes(sm.WIDE_ROOT)
    for b in bases:
        bp = np.asarray(b["pattern"]).reshape(25, 25) > 0.5
        for d in donors:
            dp = np.asarray(d["pattern"]).reshape(25, 25) > 0.5
            for q, sub in transplants(bp, dp, args.max_groups, args.big):
                h = pattern_sha256(q)
                if h in known:
                    continue
                known.add(h)
                pats.append(q.reshape(-1))
                meta.append({"base": b, "donor": d, "groups": sub, "sha": h, "n_changed": int((q != bp).sum())})
    print(f"底 {[b['id'][6:14] for b in bases]}；捐贈者 {[d['id'][6:14] for d in donors]}；候選 {len(pats)}")
    from script import r82_ctx_sm as cx
    from script import r82_delta_sm as dm
    P = np.asarray(pats, dtype=np.float32)
    slices = sm.band_slices(cfg.score_spec, grid)
    ddata = dm.load_all(cfg, grid, sm.load_history(sm.HIST_DIR, grid), wide)
    cnn = dm.score_candidates(P, slices, ddata)["wm_mean"]
    ctx = cx.score_candidates(P, slices, ddata)["wm_mean"]
    ens = np.nan_to_num((cnn + ctx) / 2, nan=-np.inf)
    picks, per = [], {}
    for i in np.argsort(-ens, kind="stable"):                      # 每組（底, 捐贈者）≤ per_pair 筆
        key = (meta[i]["base"]["id"], meta[i]["donor"]["id"])
        if per.get(key, 0) >= args.per_pair:
            continue
        per[key] = per.get(key, 0) + 1
        picks.append(int(i))
        if len(picks) >= args.n:
            break
    pcfg = load_profile_config(V6A)
    mid, sid = measurement_id(pcfg.measurement), score_spec_id(pcfg.score_spec)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    man = []
    for k, i in enumerate(picks):
        m = meta[i]
        rid = f"{args.name}_{k:03d}_{m['sha'][:8]}"
        torch.save(torch.tensor(P[i].reshape(25, 25)), out / f"{rid}.pt")
        man.append({"id": rid, "batch": 1, "port": "dual", "kind": "merge", "measurement_id": mid, "score_spec_id": sid,
                    "pattern_file": f"{rid}.pt", "pattern_sha256": m["sha"], "lineage_id": m["base"]["lineage"],
                    "parent_id": f"{m['base']['store']}:{m['base']['id']}",
                    "donor_id": f"{m['donor']['store']}:{m['donor']['id']}", "selection_arm": "merge", "op": "transplant",
                    "groups": m["groups"], "n_changed": m["n_changed"],
                    "delta_wm_mean": float(cnn[i]), "ctx_wm_mean": float(ctx[i])})
    (out / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "measurement.json").write_text(json.dumps(pcfg.measurement, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "score_spec.json").write_text(json.dumps(pcfg.score_spec, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "config.yaml").write_text(V6A.read_text(encoding="utf-8"), encoding="utf-8")
    validate_input(out, pcfg)
    print(f"→ {out}：{len(man)} 筆；每筆改 {min(x['n_changed'] for x in man)}–{max(x['n_changed'] for x in man)} 格；"
          f"平均分預測 {ens[picks].min():+.2f}..{ens[picks].max():+.2f}；組合數 {len(per)}")


if __name__ == "__main__":
    sys.exit(main())
