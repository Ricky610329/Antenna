"""
script/benchmark_vs_random.py — worst-margin(dB) vs HFSS-call 的客觀 benchmark。

把一個 run 的「搜尋效率」畫成 **best worst-margin so far vs epoch(≈HFSS-call)** 曲線,多個 run 疊圖,
並可對比 **random best-of-N**。用途:**一天驗一版(~250 epoch)** 時不看「有沒有達標」(學長要 ~1000、
又隨機),改看「曲線誰升得快/高」——這才是一天能下的客觀判斷。

worst-margin(dB)定義(與 custom_loss_minmax 的嚴格點一致 = 論文 in-band spec):
  對每個 label 取「中央平台(center plateau)」= width[0]+width[1] : +width[2] 的頻點 (n257 即 ~26.5-29.5GHz),
    method=low  (S11) : margin = center − max(pred_band)   (正 = 帶內都低於 center → 達標)
    method=high (Gain): margin = min(pred_band) − center   (正 = 帶內都高於 center → 達標)
  worst-margin = min over labels。**正 = 滿足 S11 & Gain;值即餘裕(或違反量,負)。** 越高越好。

用法(可在開發機跑,純離線讀 metrics.csv + patterns/;結果夾在 NAS 或本地皆可):
    python -m script.benchmark_vs_random --runs pixel_single_guided_harvest pixel_single_sc_rad_boundary_harvest
    python -m script.benchmark_vs_random --runs A B --random-store harvest_single_random --at 250
    python -m script.benchmark_vs_random --runs /abs/path/to/result_dir          # 也吃絕對路徑
"""
import argparse
import os
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from antenna.utils import config, ROOTDIR, DATASET_PATH, logger
config.device = "cpu"

from antenna.training import load_config, PORT_SPECS
from antenna.losses import worst_margin   # 共用定義 (與 training.py 每 epoch 落 csv 的 worst_margin 同一份)


def _margin(response, cfg):
    """便利包裝：用 config 的 port labels + targets 算 worst_margin (回 worst 純量)。"""
    return worst_margin(response, PORT_SPECS[cfg.port]["labels"], cfg.targets)[0]


def _resolve_run(name_or_path):
    """run 名 → 結果夾路徑：先試絕對/相對路徑；否則在 ROOTDIR/result 下找結尾相符的最新一個。"""
    p = Path(name_or_path)
    if p.is_dir():
        return p
    rd = ROOTDIR.joinpath("result")
    cands = [d for d in os.listdir(str(rd)) if name_or_path in d] if rd.is_dir() else []
    if not cands:
        raise SystemExit(f"找不到 run：{name_or_path} (不是路徑、也不在 {rd} 下)")
    # 取最近活動的那個 (mtime 最大)
    best = max(cands, key=lambda d: max((os.path.getmtime(os.path.join(r, f))
              for r, _, fs in os.walk(str(rd.joinpath(d))) for f in fs), default=0))
    return rd.joinpath(best)


def run_curve(run_dir):
    """讀一個 run → (epochs, best_worst_margin_so_far)。跳過 skip/無 hash 的 epoch。"""
    import pandas as pd
    run_dir = Path(run_dir)
    cfg = load_config(str(run_dir / "config.yaml"))
    df = pd.read_csv(str(run_dir / "metrics.csv"))
    epochs, wm = [], []
    for _, row in df.iterrows():
        h = str(row.get("pattern_hash", "") or "")
        if not h or h == "nan":
            continue                          # skip 的 epoch (無真實響應)
        f = run_dir / "patterns" / f"{h}.pt"
        if not f.exists():
            continue
        _patt, resp, _loss = torch.load(str(f), weights_only=True)
        m = _margin(resp, cfg)
        epochs.append(int(row["epoch"])); wm.append(m)
    if not wm:
        raise SystemExit(f"{run_dir} 沒有可評估的 epoch")
    # best-so-far (worst-margin 越高越好 → 累計最大)
    best = []
    cur = -1e9
    for v in wm:
        cur = max(cur, v); best.append(cur)
    return epochs, best


def random_curve(store_name, cfg, n_max):
    """random best-of-N：把 random-sim 資料集當「隨機抽樣」,回傳 best worst-margin so far (前 n_max 筆)。"""
    from antenna.utils.store import SampleStore
    store = SampleStore(DATASET_PATH.joinpath(store_name), verbose=False)
    best, cur = [], -1e9
    for i in range(min(len(store), n_max)):
        _x, y = store[i]
        m = _margin(y, cfg)
        cur = max(cur, m); best.append(cur)
    return list(range(1, len(best) + 1)), best


def main():
    ap = argparse.ArgumentParser(description="worst-margin(dB) vs HFSS-call benchmark (離線)")
    ap.add_argument("--runs", nargs="+", required=True, help="結果夾名(結尾相符)或路徑,可多個疊圖")
    ap.add_argument("--random-store", default=None, help="random-sim 資料集名 (DATASET_PATH 下) → 畫 random best-of-N")
    ap.add_argument("--at", type=int, default=None, help="在此 epoch 預算下印各 run 的 best worst-margin (公平對標點)")
    ap.add_argument("--out", default="tmp/report/benchmark.png", help="疊圖輸出 PNG")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(8, 5))
    summary = []
    cfg0 = None
    for name in args.runs:
        run_dir = _resolve_run(name)
        ep, best = run_curve(run_dir)
        if cfg0 is None:
            cfg0 = load_config(str(run_dir / "config.yaml"))
        ax.plot(ep, best, lw=2, label=run_dir.name[:40])
        at = args.at or ep[-1]
        upto = [b for e, b in zip(ep, best) if e <= at]
        summary.append((run_dir.name, ep[-1], upto[-1] if upto else float("nan"), at))

    if args.random_store and cfg0 is not None:
        n_max = args.at or max(s[1] for s in summary)
        rx, rbest = random_curve(args.random_store, cfg0, n_max)
        ax.plot(rx, rbest, "k--", lw=2, label=f"random best-of-N ({args.random_store})")

    ax.axhline(0, color="r", ls=":", lw=1, label="spec 達標線 (margin=0)")
    ax.set_xlabel("epoch (≈ HFSS-call)"); ax.set_ylabel("best worst-margin so far (dB)  [越高越好, >0=達標]")
    ax.set_title("worst-margin vs HFSS-call"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.tight_layout(); fig.savefig(args.out, dpi=120); plt.close(fig)

    print("\n" + "=" * 70)
    print(f"{'run':<48}{'epochs':>8}{'best-margin':>14}")
    for nm, last_ep, m, at in sorted(summary, key=lambda s: -s[2]):
        print(f"{nm[:46]:<48}{last_ep:>8}{m:>13.2f}  (@≤{at})")
    print("=" * 70)
    print(f"判讀：margin 越高越好;>0 = 帶內滿足 S11 & Gain。同 epoch 預算下比誰高/升得快;")
    print(f"      贏不過 random best-of-N(黑虛線)= 學習式搜尋沒有發揮。圖 → {args.out}")


if __name__ == "__main__":
    main()
