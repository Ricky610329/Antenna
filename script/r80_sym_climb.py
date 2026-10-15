"""R80 對稱天線續探（2026-10-15）：從 R80 實測前段做左右成對小步變異，派給單一機台。

量測與 R80 完全相同（configs/single_r80_sym_climb.yaml，measurement_id 998f5008），資料可直接併用。
性能尺離線計算：26.5–29.5 GHz S11 ≤ −10 dB、正向 RealizedGain ≥ 4 dBi，WM＝兩項最差點取小。

  python -m script.r80_sym_climb build --batch 1 --n 24 --out <dir>
  python -m script.r80_sym_climb enqueue --input <dir> --name r80sc01 --machine 218
  python -m script.r80_sym_climb score            # R80 全部＋續探 store 的前段
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from antenna.measurement import measurement_id, score_spec_id
from script.profiled_batch import load_profile_config, pattern_sha256, validate_input, validate_pattern
from script import r81_curve_sm as r81

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "single_r80_sym_climb.yaml"
EXP = Path(r"\\140.123.106.219\temp\碩二_鄒穎麒's\antenna\experiments")
R80_ROOT = EXP / "r80_symmetry_20261007" / "dataset"
V2_ROOT = r81.V2_DATASET                     # 續探 store 跟 R81 同一個佇列與資料根
CACHE = REPO / "tmp" / "r80_sym_cache.npz"
BAND = slice(5, 12)                          # 24–32 GHz／0.5 格點中的 26.5–29.5
S11_MAX, GAIN_MIN = -10.0, 4.0


def margins(y):
    """y (N,2,17) → (N,2)：S11 與 Gain 在帶內最差點的餘裕。"""
    y = np.asarray(y)[..., BAND]
    return np.stack([(S11_MAX - y[:, 0]).min(1), (y[:, 1] - GAIN_MIN).min(1)], axis=1)


def _stores(root, mid):
    for d in sorted(Path(root).iterdir()):
        try:
            m = json.loads((d / "measurement.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if measurement_id(m) == mid and (d / "results.json").exists():
            yield d


def load_rows(cfg, refresh=False):
    """R80 與續探 store 的全部 ok 實測 → (P bool N×625, Y N×2×17, ids, stores)。同圖只留第一筆。
    R80 已凍結，快取在 tmp/；續探 store 每次重讀。"""
    mid = measurement_id(cfg.measurement)
    cached = {}
    if CACHE.exists() and not refresh:
        z = np.load(CACHE, allow_pickle=False)
        cached = {"P": z["P"], "Y": z["Y"], "ids": list(z["ids"]), "stores": list(z["stores"])}
    else:
        P, Y, ids, sts = [], [], [], []
        for d in _stores(R80_ROOT, mid):
            res = json.loads((d / "results.json").read_text(encoding="utf-8"))
            for rid, e in sorted(res.items()):
                if isinstance(e, dict) and e.get("status") == "ok":
                    p, y = torch.load(d / e["sample_file"], weights_only=True, map_location="cpu")
                    P.append(np.asarray(p).reshape(-1) > 0.5)
                    Y.append(np.asarray(y, dtype=float))
                    ids.append(rid)
                    sts.append(d.name)
        cached = {"P": np.asarray(P), "Y": np.asarray(Y), "ids": ids, "stores": sts}
        CACHE.parent.mkdir(exist_ok=True)
        np.savez_compressed(CACHE, P=cached["P"], Y=cached["Y"], ids=np.array(ids), stores=np.array(sts))
    P, Y, ids, sts = list(cached["P"]), list(cached["Y"]), cached["ids"], cached["stores"]
    for d in _stores(V2_ROOT, mid):
        res = json.loads((d / "results.json").read_text(encoding="utf-8"))
        for rid, e in sorted(res.items()):
            if isinstance(e, dict) and e.get("status") == "ok":
                p, y = torch.load(d / e["sample_file"], weights_only=True, map_location="cpu")
                P.append(np.asarray(p).reshape(-1) > 0.5)
                Y.append(np.asarray(y, dtype=float))
                ids.append(rid)
                sts.append(d.name)
    seen, keep = set(), []
    for i, p in enumerate(P):
        h = pattern_sha256(p.reshape(25, 25))
        if h not in seen:
            seen.add(h)
            keep.append(i)
    P, Y = np.asarray(P)[keep], np.asarray(Y)[keep]
    return P, Y, [ids[i] for i in keep], [sts[i] for i in keep], seen


def mirror_flip(p, rng, k):
    """翻 k 組左右成對像素（中線欄只翻一格），保留饋點 (24,12)。"""
    q = p.reshape(25, 25).copy()
    cells = rng.choice(25 * 13, size=k, replace=False)
    for c in cells:
        r, col = divmod(int(c), 13)
        q[r, col] = ~q[r, col]
        q[r, 24 - col] = q[r, col]
    q[24, 12] = True
    return q


def build(args):
    cfg = load_profile_config(args.config)
    P, Y, ids, sts, known = load_rows(cfg, args.refresh)
    mg = margins(Y)
    wm = mg.min(1)
    key = wm if args.parent_score == "wm" else np.minimum(mg, 0).sum(1)
    order = np.argsort(-key, kind="stable")[:args.top]
    print(f"已知 {len(P)} 筆唯一；起點前 {args.top}：" +
          "、".join(f"{ids[i][-14:]} {wm[i]:+.2f}" for i in order))
    rng = np.random.default_rng(80_000 + args.batch)
    share = [args.n - args.n // 2] + [0] * (len(order) - 1)       # 第一名拿一半，其餘平分
    rest = args.n - share[0]
    for k in range(1, len(order)):
        share[k] = rest // (len(order) - 1) + (1 if k - 1 < rest % (len(order) - 1) else 0)
    picks, seen = [], set(known)
    for i, m in zip(order, share):
        got, tries = 0, 0
        while got < m and tries < 2000:
            tries += 1
            q = mirror_flip(P[i], rng, int(rng.integers(1, args.max_pairs + 1)))
            h = pattern_sha256(q)
            if h in seen:
                continue
            try:
                validate_pattern(q, cfg)                          # 位元與實際橋接幾何都要左右對稱
            except ValueError:
                continue
            seen.add(h)
            picks.append((q, h, ids[i]))
            got += 1
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"拒絕覆寫非空輸出夾：{out}")
    out.mkdir(parents=True, exist_ok=True)
    mid, sid = measurement_id(cfg.measurement), score_spec_id(cfg.score_spec)
    rows = []
    for seq, (q, h, parent) in enumerate(picks):
        rid = f"r80sc_b{args.batch:02d}_{seq:03d}_{h[:8]}"
        torch.save(torch.tensor(q.astype(np.float32)), out / f"{rid}.pt")
        rows.append({"id": rid, "batch": args.batch, "port": "single", "kind": "r80sym_climb",
                     "measurement_id": mid, "score_spec_id": sid, "pattern_file": f"{rid}.pt",
                     "pattern_sha256": h, "lineage_id": parent, "parent_id": parent, "selection_arm": "climb"})
    (out / "manifest.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "measurement.json").write_text(json.dumps(cfg.measurement, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "score_spec.json").write_text(json.dumps(cfg.score_spec, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "config.yaml").write_text(Path(args.config).read_text(encoding="utf-8"), encoding="utf-8")
    validate_input(out, cfg)
    print(f"→ {out}：{len(rows)} 筆（validate_input 通過）")


def score(args):
    cfg = load_profile_config(args.config)
    P, Y, ids, sts, _ = load_rows(cfg, args.refresh)
    mg = margins(Y)
    wm = mg.min(1)
    for i in np.argsort(-wm, kind="stable")[:args.top]:
        print(f"{wm[i]:+.3f}  S11 {mg[i, 0]:+.2f}  Gain {mg[i, 1]:+.2f}  {sts[i]}:{ids[i]}")
    new = [i for i, s in enumerate(sts) if s.startswith("dedust_r80sc")]
    if new:
        print(f"續探 {len(new)} 筆，最佳 {wm[new].max():+.3f}；R80 原 {len(P) - len(new)} 筆最佳 "
              f"{np.delete(wm, new).max():+.3f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--batch", type=int, required=True)
    b.add_argument("--n", type=int, default=24)
    b.add_argument("--top", type=int, default=6)
    b.add_argument("--max-pairs", type=int, default=3)
    b.add_argument("--parent-score", choices=("wm", "violation"), default="violation")
    b.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--top", type=int, default=10)
    q = sub.add_parser("enqueue")
    q.add_argument("--input", required=True)
    q.add_argument("--name", required=True)
    q.add_argument("--machine", required=True, help="IP 末段，只給這台認領")
    q.add_argument("--prio", type=int, default=-1, help="比濾波器 prio 0 優先，釘選機台先做對稱批")
    q.add_argument("--shards", type=int, default=3)
    for x in (b, s, q):
        x.add_argument("--config", type=Path, default=CONFIG)
    for x in (b, s):
        x.add_argument("--refresh", action="store_true", help="重建 R80 快取")
    args = ap.parse_args(argv)
    if args.cmd == "enqueue":
        r81.enqueue(argparse.Namespace(input=args.input, name=args.name, prio=args.prio, shards=args.shards,
                                       dataset_root=V2_ROOT, config=args.config, machine=args.machine))
    else:
        {"build": build, "score": score}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
