"""R89 整合批（2026-10-21，Ricky 提議「整合前段設計、不動連通特性下調拓撲」）：前段設計通帶有兩種卡法——
多數卡上緣 30 GHz、少數卡下緣 26 GHz 而 30 GHz 較好。以卡上緣的前幾名為底、30 GHz 較好的前段設計為捐贈者，
把兩者相異的格依位置分成 ≤ max_groups 個小區塊，列舉「移植哪幾塊」的全部組合；過濾掉兩埠接通或改變
大塊金屬結構（≥ big 格的 4 連通塊數）的候選，再用變化量模型與上下文模型的平均分排序挑 n 筆。

第二輪（mg02 起）：--base-mode top 取 WM 前幾名為底（不論卡哪一緣），捐贈者依底卡住的那一緣挑；
--rank small 以移植格數少者優先（mg01 模型排序 ρ 僅 0.08，前段模型無鑑別力）；--stack N 另以 --stack-base 為底，
把「實測比親代好、改 ≤ 4 格」的小步移植兩兩／三三疊加（依個別增益和排序）。

  python -m script.r89_merge --n 24 --out <dir>                                            # mg01
  python -m script.r89_merge --n 24 --base-mode top --rank small --stack 12 --stack-base r81v2_b111_010_0dcdaf1d --name r81v6mg02 --out <dir>
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
STORE_RE = r"r81v6(b\d+|mg\d+)s\d$"
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


def improving_moves(scores, pats, parents, max_cells=4, min_parent=-0.5):
    """親代 WM > min_parent、實測比親代好、改 ≤ max_cells 格的子代 → [(增益, 子代 id, 格索引, 子代在這些格的值)]，依增益由大到小。
    scores／pats：id → WM／bool 625；parents：子代 id → 親代 id。"""
    out = []
    for cid, pid in parents.items():
        if cid not in scores or pid not in scores or scores[pid] <= min_parent:
            continue
        cells = np.flatnonzero(pats[cid] != pats[pid])
        gain = scores[cid] - scores[pid]
        if 1 <= len(cells) <= max_cells and gain > 0:
            out.append((gain, cid, cells, pats[cid][cells]))
    return sorted(out, key=lambda t: -t[0])


def stack_candidates(base, moves, sizes=(2, 3), big=4):
    """把小步移植（格→值）兩兩／三三套到 base 上；略過對 base 無作用的組合、兩埠接通與大塊結構改變者。
    回傳 [(pattern 25×25, 用到的子代 id, 增益和)]，依增益和由大到小。"""
    nb = big_components(base.reshape(25, 25), big)
    out = []
    for k in sizes:
        for sub in itertools.combinations(range(len(moves)), k):
            q = base.reshape(-1).copy()
            for i in sub:
                q[moves[i][2]] = moves[i][3]
            q = sm.enforce_dual_geometry(q)
            if (q.reshape(-1) == base.reshape(-1)).all() or sm.pads_connected(q) or big_components(q, big) != nb:
                continue
            out.append((q, [moves[i][1] for i in sub], float(sum(moves[i][0] for i in sub))))
    return sorted(out, key=lambda t: -t[2])


MARGINS = ("s21_pass", "s21_low_stop", "s21_high_stop", "s11_match", "s22_match")


def single_flip_stats(rows, parents, i_f, min_parent=-1.0):
    """單格翻轉的實測配對（親代 WM > min_parent）→ {(格索引, 新值): [(ΔS21@f, {margin: Δ})...]}。
    rows：id → (bool 625, true_wm 結果, S21 曲線)；parents：子代 id → 親代 id。"""
    out = {}
    for cid, pid in parents.items():
        if cid not in rows or pid not in rows:
            continue
        c, cs, cy = rows[cid]
        p, ps, py = rows[pid]
        if ps["worst_margin"] <= min_parent:
            continue
        fl = np.flatnonzero(c != p)
        if len(fl) != 1:
            continue
        k = int(fl[0])
        out.setdefault((k, bool(c[k])), []).append(
            (float(cy[i_f] - py[i_f]), {m: cs["margins"][m] - ps["margins"][m] for m in MARGINS}))
    return out


def edge_moves(stats, base_pat, base_margins, keep=0.1):
    """依平均 ΔS21@f 由大到小，挑 base 尚未是該值、且（base margin＋平均 Δ）其餘四項都 ≥ keep 的單格翻轉。
    回傳 [(格, 新值, 平均 ΔS21@f, 樣本數)]。"""
    out = []
    for (k, val), v in stats.items():
        if bool(base_pat[k]) == val:
            continue
        d = np.mean([x[0] for x in v])
        if d <= 0:
            continue
        pred = {m: base_margins[m] + np.mean([x[1][m] for x in v]) for m in MARGINS if m != "s21_pass"}
        if min(pred.values()) >= keep:
            out.append((k, val, float(d), len(v)))
    return sorted(out, key=lambda t: -t[2])


def predicted_moves(stats, base_pat, base_margins, top_pairs=8):
    """單格翻轉（及前 top_pairs 個的兩兩組合）套到 base 的預估：五項 margin＝base＋實測平均 Δ（兩兩視為可加），
    預估 WM＝其中最小。回傳 [(格清單 [(格, 值)], 預估 WM, 樣本數下限)]，依預估 WM 由大到小。"""
    singles = []
    for (k, val), v in stats.items():
        if bool(base_pat[k]) == val:
            continue
        d = {m: float(np.mean([x[1][m] for x in v])) for m in MARGINS}
        singles.append(([(k, val)], d, len(v)))
    pred = lambda d: min(base_margins[m] + d[m] for m in MARGINS)
    singles.sort(key=lambda t: -pred(t[1]))
    out = [(c, pred(d), n) for c, d, n in singles]
    for (c1, d1, n1), (c2, d2, n2) in itertools.combinations(singles[:top_pairs], 2):
        if c1[0][0] == c2[0][0]:
            continue
        out.append((c1 + c2, pred({m: d1[m] + d2[m] for m in MARGINS}), min(n1, n2)))
    return sorted(out, key=lambda t: -t[1])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--bases", type=int, default=3, help="底的筆數")
    ap.add_argument("--base-mode", choices=("upper30", "top"), default="upper30",
                    help="upper30＝通帶最差點在 30 GHz 的前幾名（mg01）；top＝WM 前幾名，捐贈者依底卡住的那一緣挑")
    ap.add_argument("--donors", type=int, default=5, help="每個底的捐贈者數：WM > --donor-wm 中該緣 S21 最好的幾筆")
    ap.add_argument("--donor-wm", type=float, default=-0.5)
    ap.add_argument("--max-groups", type=int, default=5)
    ap.add_argument("--big", type=int, default=4)
    ap.add_argument("--per-pair", type=int, default=3)
    ap.add_argument("--rank", choices=("model", "small"), default="model",
                    help="model＝變化量＋上下文模型平均分（mg01）；small＝移植格數少者優先")
    ap.add_argument("--stack", type=int, default=0, help="疊加候選的名額（從 n 中撥出）")
    ap.add_argument("--stack-base", default=None, help="疊加的底設計 id（逗號分隔可多個，名額平分）")
    ap.add_argument("--stack-same-parent", action="store_true",
                    help="只疊「親代就是這個底」的已實測有效小步（同底驗證過的單步才組合）")
    ap.add_argument("--stack-k", type=int, default=8, help="取增益前幾名的小步移植來疊加")
    ap.add_argument("--stack-sizes", default="2,3", help="疊加幾個小步（逗號分隔；1＝單步套到新底）")
    ap.add_argument("--stack-min-parent", type=float, default=-0.25, help="小步移植的親代 WM 下限（只收前段的改動）")
    ap.add_argument("--edge", type=float, default=None,
                    help="通帶邊緣專攻（GHz，如 30）：只產「能抬高該頻點 S21 的單格翻轉」——依前段單格翻轉實測平均增益，"
                         "套到 WM 前 --edge-bases 名；另加每個底最乾淨前 --edge-pair 格的兩兩組合")
    ap.add_argument("--edge-bases", type=int, default=2)
    ap.add_argument("--predict", default=None,
                    help="預估模式（逗號分隔的底 id）：依前段單格翻轉實測平均 Δ 預估五項 margin，依預估 WM 挑單格與兩兩"
                         "（用於擾動後修復：如通帶過了但高阻破的底）")
    ap.add_argument("--edge-pair", type=int, default=3)
    ap.add_argument("--config", type=Path, default=V6A, help="評分／量測設定（預設 v6a；切回 v6 用 configs/dual_r81_v6.yaml）")
    ap.add_argument("--seed", type=int, default=89)
    ap.add_argument("--name", default="r81v6mg01")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    cfg = sm.load_cfg(args.config)
    grid = np.asarray(sm.frequency_grid(cfg.measurement))
    wide = sm.load_wide(sm.WIDE_ROOT, cfg, grid)
    edge = {26.0: int(np.argmin(abs(grid - 26))), 30.0: int(np.argmin(abs(grid - 30)))}
    pb = (grid >= 26) & (grid <= 30)
    cand = []
    for r in wide:
        if not re.search(STORE_RE if args.base_mode == "top" else r"r81v6b\d+s\d$", r["store"]):
            continue
        sc = sm.true_wm(r, cfg, grid)
        if sc:
            cand.append((sc["worst_margin"], r))
    cand.sort(key=lambda t: -t[0])
    worst_edge = lambda r: 30.0 if r["y"][1][edge[30.0]] <= r["y"][1][edge[26.0]] else 26.0
    if args.base_mode == "upper30":
        bases = [r for w, r in cand if grid[pb][np.argmin(r["y"][1][pb])] == 30.0][:args.bases]
    else:
        bases = [r for w, r in cand][:args.bases]
    if args.edge is not None or args.predict:
        if args.edge is None:
            args.edge = 30.0
        return edge_main(args, cfg, grid, wide, cand)
    pool = [r for w, r in cand if w > args.donor_wm]
    pats, meta = [], []
    known = {r["sha"] for r in wide} | sm.input_hashes(sm.WIDE_ROOT)
    pairs = []
    for b in bases:
        e = 30.0 if args.base_mode == "upper30" else worst_edge(b)
        ds = sorted([r for r in pool if r is not b and r not in bases], key=lambda r: -r["y"][1][edge[e]])[:args.donors]
        pairs += [(b, d) for d in ds]
    for b, d in pairs:
        bp = np.asarray(b["pattern"]).reshape(25, 25) > 0.5
        dp = np.asarray(d["pattern"]).reshape(25, 25) > 0.5
        for q, sub in transplants(bp, dp, args.max_groups, args.big):
            h = pattern_sha256(q)
            if h in known:
                continue
            known.add(h)
            pats.append(q.reshape(-1))
            meta.append({"base": b, "donor": d, "groups": sub, "sha": h, "n_changed": int((q != bp).sum())})
    print(f"底 {[b['id'][6:14] for b in bases]}；組合 {len(pairs)}；移植候選 {len(pats)}")
    stack = []
    if args.stack:
        man = {}
        for f in sm.V2_DATASET.glob("dedust_r81v6*_input/manifest.json"):
            for m in json.loads(f.read_text(encoding="utf-8")):
                man[m["id"]] = m.get("parent_id", "").split(":")[-1]
        scores = {r["id"]: w for w, r in cand}
        ptn = {r["id"]: np.asarray(r["pattern"]).reshape(-1) > 0.5 for w, r in cand}
        sizes = tuple(int(x) for x in args.stack_sizes.split(","))
        bases_s = args.stack_base.split(",")
        for bi, bid in enumerate(bases_s):                         # 多個底時名額平分
            quota = args.stack // len(bases_s) + (1 if bi < args.stack % len(bases_s) else 0)
            sb = next(r for w, r in cand if r["id"] == bid)
            moves = [m for m in improving_moves(scores, ptn, man, min_parent=args.stack_min_parent)
                     if (ptn[sb["id"]][m[2]] != m[3]).any()                       # 底已做到的改動不算
                     and (not args.stack_same_parent or man.get(m[1]) == sb["id"])][:args.stack_k]
            print(f"底 {bid[6:18]} 疊加用的小步：", [(m[1][6:16], round(m[0], 3), len(m[2])) for m in moves])
            got = 0
            for q, src, gain in stack_candidates(ptn[sb["id"]], moves, sizes=sizes, big=args.big):
                h = pattern_sha256(q)
                if h in known:
                    continue
                known.add(h)
                stack.append({"pattern": q.reshape(-1), "sha": h, "base": sb, "src": src, "gain": gain,
                              "n_changed": int((q.reshape(-1) != ptn[sb["id"]]).sum())})
                got += 1
                if got >= quota:
                    break
    P = np.asarray(pats, dtype=np.float32).reshape(-1, 625)
    n_tr = args.n - len(stack)
    if args.rank == "model":
        from script import r82_ctx_sm as cx
        from script import r82_delta_sm as dm
        slices = sm.band_slices(cfg.score_spec, grid)
        ddata = dm.load_all(cfg, grid, sm.load_history(sm.HIST_DIR, grid), wide)
        cnn = dm.score_candidates(P, slices, ddata)["wm_mean"]
        ctx = cx.score_candidates(P, slices, ddata)["wm_mean"]
        order = np.argsort(-np.nan_to_num((cnn + ctx) / 2, nan=-np.inf), kind="stable")
    else:
        cnn = ctx = np.full(len(P), np.nan)
        order = np.lexsort((rng.random(len(P)), [m["n_changed"] for m in meta]))
    picks, per = [], {}
    for i in order if n_tr > 0 else []:                            # 每組（底, 捐贈者）≤ per_pair 筆
        key = (meta[i]["base"]["id"], meta[i]["donor"]["id"])
        if per.get(key, 0) >= args.per_pair:
            continue
        per[key] = per.get(key, 0) + 1
        picks.append(int(i))
        if len(picks) >= n_tr:
            break
    pcfg = load_profile_config(args.config)
    mid, sid = measurement_id(pcfg.measurement), score_spec_id(pcfg.score_spec)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    man = []

    def row(k, pat, h, base, **extra):
        rid = f"{args.name}_{k:03d}_{h[:8]}"
        torch.save(torch.tensor(np.asarray(pat, dtype=np.float32).reshape(25, 25)), out / f"{rid}.pt")
        man.append({"id": rid, "batch": 1, "port": "dual", "kind": "merge", "measurement_id": mid, "score_spec_id": sid,
                    "pattern_file": f"{rid}.pt", "pattern_sha256": h, "lineage_id": base["lineage"],
                    "parent_id": f"{base['store']}:{base['id']}", **extra})

    for st in stack:
        row(len(man), st["pattern"], st["sha"], st["base"], selection_arm="stack", op="stack", moves=st["src"],
            gain_sum=st["gain"], n_changed=st["n_changed"])
    for i in picks:
        m = meta[i]
        row(len(man), P[i], m["sha"], m["base"], donor_id=f"{m['donor']['store']}:{m['donor']['id']}",
            selection_arm="merge", op="transplant", groups=m["groups"], n_changed=m["n_changed"],
            delta_wm_mean=float(cnn[i]) if np.isfinite(cnn[i]) else None,
            ctx_wm_mean=float(ctx[i]) if np.isfinite(ctx[i]) else None)
    (out / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "measurement.json").write_text(json.dumps(pcfg.measurement, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "score_spec.json").write_text(json.dumps(pcfg.score_spec, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "config.yaml").write_text(Path(args.config).read_text(encoding="utf-8"), encoding="utf-8")
    validate_input(out, pcfg)
    print(f"→ {out}：{len(man)} 筆（疊加 {len(stack)}、移植 {len(picks)}）；每筆改 {min(x['n_changed'] for x in man)}–"
          f"{max(x['n_changed'] for x in man)} 格；移植組合數 {len(per)}")


def edge_main(args, cfg, grid, wide, cand):
    """--edge 模式：單格翻轉套到前幾名，候選 ≤ n，依底輪流取。"""
    i_f = int(np.argmin(abs(grid - args.edge)))
    rows, parents = {}, {}
    for w, r in cand:
        rows[r["id"]] = (np.asarray(r["pattern"]).reshape(-1) > 0.5, sm.true_wm(r, cfg, grid), r["y"][1])
    for f in sm.V2_DATASET.glob("dedust_r81v6*_input/manifest.json"):
        for m in json.loads(f.read_text(encoding="utf-8")):
            parents[m["id"]] = m.get("parent_id", "").split(":")[-1]
    stats = single_flip_stats(rows, parents, i_f)
    known = {r["sha"] for r in wide} | sm.input_hashes(sm.WIDE_ROOT)
    per_base = []
    if args.predict:
        ids = args.predict.split(",")
        bases = [(w, r) for w, r in cand if r["id"] in ids]
    else:
        bases = cand[:args.edge_bases]
    for w, b in bases:
        bp, bs, _ = rows[b["id"]]
        nb = big_components(bp.reshape(25, 25), args.big)
        items = []
        if args.predict:
            pm = predicted_moves(stats, bp, bs["margins"])
            print(f"底 {b['id'][6:18]}（{w:+.3f}）預估前 8：",
                  [([(k // 25, k % 25, "加" if v else "挖") for k, v in c], round(pw, 3), n) for c, pw, n in pm[:8]])
            combos = [[(k, v, pw, n) for k, v in c] for c, pw, n in pm]
        else:
            mv = edge_moves(stats, bp, bs["margins"])
            print(f"底 {b['id'][6:18]}（{w:+.3f}）可用翻轉 {len(mv)}：",
                  [((k // 25, k % 25), "加銅" if v else "挖空", round(d, 3), n) for k, v, d, n in mv[:8]])
            combos = [[m] for m in mv] + [list(c) for c in itertools.combinations(mv[:args.edge_pair], 2)]
        for combo in combos:
            q = bp.copy()
            for k, v, _, _ in combo:
                q[k] = v
            q = sm.enforce_dual_geometry(q)
            h = pattern_sha256(q)
            if h in known or sm.pads_connected(q) or big_components(q, args.big) != nb:
                continue
            known.add(h)
            items.append((q, h, b, combo))
        if not args.predict:                                       # 上緣專攻：兩兩優先；預估模式：照預估 WM 順序
            items = [x for x in items if len(x[3]) == 2] + [x for x in items if len(x[3]) == 1]
        per_base.append(items)
    picks = []
    while len(picks) < args.n and any(per_base):                 # 各底輪流取
        for lst in per_base:
            if lst and len(picks) < args.n:
                picks.append(lst.pop(0))
    pcfg = load_profile_config(args.config)
    mid, sid = measurement_id(pcfg.measurement), score_spec_id(pcfg.score_spec)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    man = []
    for k, (q, h, b, combo) in enumerate(picks):
        rid = f"{args.name}_{k:03d}_{h[:8]}"
        torch.save(torch.tensor(q.reshape(25, 25).astype(np.float32)), out / f"{rid}.pt")
        man.append({"id": rid, "batch": 1, "port": "dual", "kind": "merge", "measurement_id": mid, "score_spec_id": sid,
                    "pattern_file": f"{rid}.pt", "pattern_sha256": h, "lineage_id": b["lineage"],
                    "parent_id": f"{b['store']}:{b['id']}", "selection_arm": "predict" if args.predict else "edge",
                    "op": "predict" if args.predict else f"edge{args.edge:g}",
                    "cells": [[int(c // 25), int(c % 25), bool(v)] for c, v, _, _ in combo],
                    "pred_gain": float(combo[0][2] if args.predict else sum(d for _, _, d, _ in combo)),
                    "n_changed": len(combo)})
    (out / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "measurement.json").write_text(json.dumps(pcfg.measurement, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "score_spec.json").write_text(json.dumps(pcfg.score_spec, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "config.yaml").write_text(Path(args.config).read_text(encoding="utf-8"), encoding="utf-8")
    validate_input(out, pcfg)
    print(f"→ {out}：{len(man)} 筆（兩兩 {sum(len(x[3]) == 2 for x in picks)}、單格 {sum(len(x[3]) == 1 for x in picks)}）")


if __name__ == "__main__":
    sys.exit(main())
