"""R81 指定設計的單張卡片（S21／S11／S22 各一格＋pattern，2×2 正方形），附重測結果。

用法：python -m script.figs.r81_design_card <輸出資料夾> <設計 id> [<設計 id> ...]
       [--primary configs/dual_r81_v4a.yaml] [--final configs/dual_r81_v4.yaml]
重測（id~rep1）若存在，標題一併列出原件與重測 WM。
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from script import r81_curve_sm as sm
from script.figs import r81_progress as rp


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", type=Path)
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--primary", type=Path, default=sm.REPO / "configs" / "dual_r81_v4a.yaml")
    ap.add_argument("--final", type=Path, default=sm.REPO / "configs" / "dual_r81_v4.yaml")
    args = ap.parse_args(argv)
    plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
    primary, final = sm.load_cfg(args.primary), sm.load_cfg(args.final)
    grid = sm.frequency_grid(primary.measurement)
    root = sm.WIDE_ROOT / "v2_v001" / "dataset"
    rows = {}
    for store in sorted(root.glob("dedust_r81v*")):
        if (store / "results.json").exists() and (store / "measurement.json").exists():
            for r in sm.load_store(store, grid):
                a, b = sm.true_wm(r, primary, grid), sm.true_wm(r, final, grid)
                if a and b:
                    r.update(wm=a["worst_margin"], wm_final=b["worst_margin"], margins=a["margins"])
                    rows[r["id"]] = r
    args.outdir.mkdir(parents=True, exist_ok=True)
    for i in args.ids:
        r = rows[i]
        rep = rows.get(i + "~rep1")
        out = args.outdir / f"card_{i}.png"
        note = f"（跨機重測 {rep['wm']:+.2f} dB，公證通過）" if rep and rep["wm"] > 0 else "（單次，未公證）"
        fig = rp.best_figure(out, r, grid, primary, final, len(rows),
                             scope_note="原件單次 HFSS；重測為另一台機器同一圖形重跑。")
        fig.suptitle(f"{i}：階段 spec WM {r['wm']:+.2f} dB {note}／最終 spec WM {r['wm_final']:+.2f} dB",
                     x=0.03, ha="left", fontsize=12, color=rp.INK)
        fig.savefig(out, facecolor=rp.SURF)
        plt.close(fig)
        print("saved", out)


if __name__ == "__main__":
    main()
