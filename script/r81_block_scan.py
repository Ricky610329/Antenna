"""R85 區塊掃描（2026-10-20，emforge session 交來的「拉扯區外破局」評估）：以一個底設計為準，把拉扯區以外的
板面切成 4×4 小塊，每筆只反轉一塊（反轉後兩埠接通就改成把該塊清空），量出各區對五項 margin 與零點的影響。

拉扯區＝兩主體之間的縫（第 8–13 列、9–23 欄）＋上埠墊右側（2–5 列、15–16 欄）；埠墊不動。

  python -m script.r81_block_scan --base r81v2_b92_016_1921041c --n 24 --out <dir>
  python -m script.r81_curve_sm enqueue --input <dir> --name r81v6sc01 --prio -3 --shards 3 --config configs/dual_r81_v6.yaml
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
TUG = np.zeros((25, 25), bool)
TUG[8:14, 9:24] = True
TUG[2:6, 15:17] = True
LEVERS = [(2, 7), (3, 9), (6, 9), (4, 15), (17, 10), (22, 16)]   # 交接檔觀察到的外區低阻槓桿格，所在小塊優先掃


def blocks(size=4):
    """拉扯區與埠墊以外的 size×size 小塊（每塊回傳可動格的 bool 遮罩），依可動格數由多到少。"""
    free = ~TUG & ~sm.PADS
    out = []
    for r in range(0, 25, size):
        for c in range(0, 25, size):
            m = np.zeros((25, 25), bool)
            m[r:r + size, c:c + size] = True
            m &= free
            if m.sum() >= size * size // 2:
                out.append(((r, c), m))
    lever = lambda m: any(m[r, c] for r, c in LEVERS)
    return sorted(out, key=lambda t: (not lever(t[1]), -t[1].sum()))


def perturb(base, mask):
    """反轉 mask 內的格；兩埠接通就改成清空該區（清空不會接通）。回傳 (pattern, 動作)。"""
    q = base.copy()
    q[mask] = ~q[mask]
    q = sm.enforce_dual_geometry(q)
    if not sm.pads_connected(q):
        return q, "flip"
    q = base.copy()
    q[mask] = False
    return sm.enforce_dual_geometry(q), "clear"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="底設計 id（v6 寬頻真值）")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--size", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    cfg = sm.load_cfg(V6)
    grid = sm.frequency_grid(cfg.measurement)
    wide = sm.load_wide(sm.WIDE_ROOT, cfg, grid)
    base = next(r for r in wide if r["id"] == args.base)
    bp = np.asarray(base["pattern"]).reshape(25, 25) > 0.5
    pcfg = load_profile_config(V6)
    mid, sid = measurement_id(pcfg.measurement), score_spec_id(pcfg.score_spec)
    known = {r["sha"] for r in wide} | sm.input_hashes(sm.WIDE_ROOT)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    man = []
    for (r0, c0), m in blocks(args.size):
        if len(man) >= args.n:
            break
        q, act = perturb(bp, m)
        h = pattern_sha256(q)
        if h in known or sm.pads_connected(q):
            continue
        known.add(h)
        rid = f"r81v6sc01_{len(man):03d}_{h[:8]}"
        torch.save(torch.tensor(q.astype(np.float32)), out / f"{rid}.pt")
        man.append({"id": rid, "batch": 1, "port": "dual", "kind": "block_scan", "measurement_id": mid,
                    "score_spec_id": sid, "pattern_file": f"{rid}.pt", "pattern_sha256": h, "lineage_id": base["lineage"],
                    "parent_id": f"{base['store']}:{base['id']}", "selection_arm": "block_scan", "op": act,
                    "block": [r0, c0, args.size], "n_changed": int((q != bp).sum())})
    (out / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "measurement.json").write_text(json.dumps(pcfg.measurement, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "score_spec.json").write_text(json.dumps(pcfg.score_spec, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "config.yaml").write_text(V6.read_text(encoding="utf-8"), encoding="utf-8")
    validate_input(out, pcfg)
    print(f"→ {out}：{len(man)} 筆；動作 {sum(x['op'] == 'flip' for x in man)} 反轉／{sum(x['op'] == 'clear' for x in man)} 清空；"
          f"每筆改 {min(x['n_changed'] for x in man)}–{max(x['n_changed'] for x in man)} 格")


if __name__ == "__main__":
    sys.exit(main())
