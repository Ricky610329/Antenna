"""
script/collect_radiation.py — 收集 / 累積方向圖資料 (預設存 harvest_single_rad)。

方向圖只是「幾何的函數」→ 任何 pattern 丟 SinglePortRadSimulator 跑一發 HFSS 就有方向圖，
**不管該 pattern 當初訓練時有沒有用 radiation loss**。本支把指定來源的 pattern 跑出方向圖、
累積到 DATASET_PATH/<out> 的 SampleStore (一筆一檔、hash 去重 → 可重複跑、只補沒收過的)。

設計目的：往後持續累積 rad 資料 (docs Stage 3 的 harvest_single_rad)，供 rad head 預訓練 /
比較不同設計的波束長相。**獨立新名、永不寫回 before-rad 舊集** (configs/README 規則)。

存的格式：每筆 (pattern, rad)，rad = (3, n_theta) 張量 = [theta(度), phi0 gain(dB), phi90 gain(dB)]。

用法 (正式機，需 Ansys HFSS；開發機不可跑)：
    # 從 result 夾收集「各自最佳 K 張」(報告 / 比較不同實驗的波束；不管有沒有 rad loss)
    python -m script.collect_radiation --runs pixel_single_sc_rad,pixel_single_sc_mirror,pixel_single_sc_rad_boundary --top-k 1
    # 從 harvest_single 收集「criterion 最好的 N 張」建 rad 預訓練集 (Stage 3)
    python -m script.collect_radiation --dataset harvest_single --best 500
    # 自備 pattern .pt 檔
    python -m script.collect_radiation --pattern good1.pt good2.pt
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")                    # 無頭繪圖 (存 PNG，不開視窗)
import matplotlib.pyplot as plt
plt.rcParams["axes.unicode_minus"] = False

from antenna.utils import config, ROOTDIR, DATASET_PATH, logger
config.device = "cpu"

from antenna import AntennaPattern
from antenna.patch import SinglePortRadSimulator
from antenna.training import load_config, setup_responses
from antenna.utils.store import SampleStore, fingerprint
from antenna.response import MultiResponses


def _bin(p) -> torch.Tensor:
    """任意輸入 → (25,25) 純 0/1 (模擬器要求二值；fingerprint/存檔都用這個正規形)。"""
    return (torch.as_tensor(p).float().reshape(25, 25) > 0.5).float()


def _plot_radiation(stack, tag, figdir, idx):
    """存一張 radiation 圖 (**極座標、全英文**)。stack=(3,n_theta)=[theta, phi0, phi90]。

    主波束朝上 (theta=0 在頂)、半徑=gain(dB)、**dB 標籤對在正上方中軸**。
    dB 軸**自動取到 5 的倍數、包住整個 pattern**(峰值不超出畫面，仍每環 5 dB；下界限深 30 dB)。
    標 +/-45° 覆蓋窗 + 邊界線 + G0-3dB 圈 (學長放寬到 +/-45)。
    """
    th, p0, p90 = stack[0].numpy(), stack[1].numpy(), stack[2].numpy()
    o = th.argsort()                                  # HFSS 匯出序可能未排序 → 先排好
    th, p0, p90 = th[o], p0[o], p90[o]
    bi = int(np.abs(th).argmin())
    g0 = float(max(p0[bi], p90[bi]))                  # boresight 增益
    gmax = float(max(p0.max(), p90.max())); gmin = float(min(p0.min(), p90.min()))
    rmax = int(np.ceil((gmax + 0.5) / 5.0) * 5)       # 上界取到 5 倍數、含峰值 (不超出畫面)
    rmin = int(max(np.floor(gmin / 5.0) * 5, rmax - 30))   # 下界取 5 倍數、限深 30 dB (深零點不拉爆)
    fig = plt.figure(figsize=(6.6, 6.9))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)   # 0 度朝上、+角度在右
    ax.set_rlim(0, rmax - rmin); ax.set_rlabel_position(0)        # dB 標籤對在上方中軸
    # +/-45° 覆蓋窗 + 兩條邊界線 + 標字
    ax.fill_between(np.deg2rad(np.linspace(-45, 45, 60)), 0, rmax - rmin, color="gold", alpha=0.12)
    for a in (-45, 45):
        ax.plot([np.deg2rad(a)] * 2, [0, rmax - rmin], color="darkorange", ls="--", lw=1.5)
    ax.text(np.deg2rad(45), (rmax - rmin) * 1.04, "+45", color="darkorange", fontsize=9, ha="left")
    ax.text(np.deg2rad(-45), (rmax - rmin) * 1.04, "-45", color="darkorange", fontsize=9, ha="right")
    # G0-3dB 圈 + 兩切面 (低端 clip 到 rmin，深零點不溢出中心)
    ax.plot(np.deg2rad(np.linspace(-180, 180, 361)), np.full(361, (g0 - 3) - rmin), "r--", lw=1.2, label="G0-3dB")
    ax.plot(np.deg2rad(th), np.clip(p0, rmin, None) - rmin, "b-", lw=2, label="phi=0 (E)")
    ax.plot(np.deg2rad(th), np.clip(p90, rmin, None) - rmin, "g-", lw=2, label="phi=90 (H)")
    rt = list(range(rmin, rmax + 1, 5))               # 每環 5 dB
    ax.set_rticks([t - rmin for t in rt]); ax.set_yticklabels([str(t) for t in rt], fontsize=7.5)
    ax.set_thetagrids(range(0, 360, 30),
                      ["0", "30", "60", "90", "120", "150", "180", "-150", "-120", "-90", "-60", "-30"],
                      fontsize=8)
    ax.set_title(f"Radiation: {tag}  (G0={g0:.1f} dB)", fontsize=10, pad=16)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=7.5)
    fig.tight_layout()
    safe = "".join(c if c.isalnum() else "_" for c in tag)[:50]
    path = os.path.join(figdir, f"rad_{idx:02d}_{safe}.png")
    fig.savefig(path, dpi=120); plt.close(fig)
    return path


def _patterns_from_runs(suffixes, top_k):
    """從 result 夾收集各 run 的『最佳 top_k 張』pattern (依真實 sim_loss)。"""
    import pandas as pd
    rd = ROOTDIR.joinpath("result")
    out = []
    for suf in suffixes:
        dirs = [d for d in os.listdir(str(rd)) if d.endswith(suf)]
        if not dirs:
            logger.warning(f"找不到 result 夾結尾為 {suf!r}")
            continue
        d = sorted(dirs, key=lambda x: max((os.path.getmtime(os.path.join(r, f))
              for r, _, fs in os.walk(str(rd.joinpath(x))) for f in fs), default=0))[-1]   # 最近活動那個
        p = rd.joinpath(d)
        df = pd.read_csv(str(p.joinpath("metrics.csv")))
        for _, row in df.nsmallest(top_k, "sim_loss").iterrows():
            patt, _resp, loss = torch.load(str(p.joinpath("patterns", f"{row['pattern_hash']}.pt")),
                                           weights_only=True)
            out.append((patt, f"{suf}@ep{int(row['epoch'])}(loss={float(loss):.2f})"))
    return out


def _patterns_from_dataset(name, best):
    """從 harvest SampleStore 收集『criterion 最好的 best 張』pattern (spec 須先安裝)。"""
    store = SampleStore(DATASET_PATH.joinpath(name), verbose=False)
    scored = []
    for i in range(len(store)):
        x, y = store[i]
        try:
            scored.append((float(MultiResponses(y.float()).criterion()), x))
        except Exception:
            continue
    scored.sort(key=lambda t: t[0])
    return [(x, f"{name}#{r}(crit={c:.2f})") for r, (c, x) in enumerate(scored[:best])]


def main():
    ap = argparse.ArgumentParser(description="收集 / 累積方向圖資料 (需正式機 HFSS)")
    ap.add_argument("--runs", default=None, help="逗號分隔的 result 夾結尾 (各取 --top-k 最佳)")
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--dataset", default=None, help="harvest SampleStore 名 (取 --best 最佳)")
    ap.add_argument("--best", type=int, default=500)
    ap.add_argument("--pattern", nargs="*", default=None, help="自備 pattern .pt 檔")
    ap.add_argument("--out", default="harvest_single_rad", help="累積到 DATASET_PATH 下的 rad 集名")
    ap.add_argument("--config", default="configs/single_base.yaml", help="single config (建 spec / 排序用)")
    ap.add_argument("--record", default="_radiation_collect", help="HFSS 模擬暫存輸出根目錄")
    ap.add_argument("--figdir", default="tmp/report/radiation", help="每張 pattern 的 radiation 圖輸出夾 (英文標籤)")
    args = ap.parse_args()

    AntennaPattern.setDefaultCoordinate((0, 25, 0, 25))
    cfg = load_config(args.config)
    if cfg.port != "single":
        raise SystemExit(f"方向圖收集目前只支援 port=single，但 config 是 {cfg.port!r}")
    setup_responses(cfg)                              # 安裝 spec → criterion 可用、response 機制就緒

    # 1) 蒐集要收的 pattern (可混多來源)
    items = []
    if args.runs:
        items += _patterns_from_runs([s.strip() for s in args.runs.split(",")], args.top_k)
    if args.dataset:
        items += _patterns_from_dataset(args.dataset, args.best)
    if args.pattern:
        items += [(torch.load(f, weights_only=True), Path(f).name) for f in args.pattern]
    if not items:
        raise SystemExit("沒指定來源：用 --runs / --dataset / --pattern 至少一個")

    # 2) 比對 store：已收過 → 從 store 讀 rad 直接出圖 (免 HFSS)；沒收過 → 跑 HFSS。**每張都會出圖**。
    out_store = SampleStore(DATASET_PATH.joinpath(args.out), verbose=True)
    stored = {fingerprint(out_store[i][0]): out_store[i][1] for i in range(len(out_store))}
    os.makedirs(args.figdir, exist_ok=True)
    have = [(p, tag) for p, tag in items if fingerprint(_bin(p)) in stored]
    todo = [(p, tag) for p, tag in items if fingerprint(_bin(p)) not in stored]
    print(f"來源共 {len(items)} 張：已收過 {len(have)} 張(直接出圖)、需跑 HFSS {len(todo)} 張")

    n = 0
    # 已收過的：從 store 讀 rad、直接畫 (不碰 HFSS)
    for p, tag in have:
        fig = _plot_radiation(stored[fingerprint(_bin(p))], tag, args.figdir, n)
        print(f"    [已收] {tag} → {fig}"); n += 1

    # 沒收過的：開 HFSS 跑、累積、畫
    added = 0
    if todo:
        sim = SinglePortRadSimulator(record_path=str(Path(args.record).resolve()))
        sim.open()
        try:
            for j, (p, tag) in enumerate(todo):
                patt = _bin(p)
                print(f"    [HFSS {j + 1}/{len(todo)}] {tag}  金屬={int(patt.sum())} … 求解中", flush=True)
                sim.start(n); sim(patt); sim.end()
                rad = sim.last_radiation
                if not (isinstance(rad, dict) and rad.get("theta") is not None):
                    err = rad.get("error") if isinstance(rad, dict) else rad
                    logger.warning(f"  方向圖萃取失敗，跳過：{err}"); continue
                stack = torch.stack([rad["theta"].float(), rad["phi0"].float(), rad["phi90"].float()])
                print(f"    radiation 圖 → {_plot_radiation(stack, tag, args.figdir, n)}")
                if out_store.add(patt, stack):
                    added += 1
                n += 1
        finally:
            sim.quit()
    print(f"\n✅ 完成：出圖 {n} 張 → {args.figdir}/ ；新增 {added} 筆到 {DATASET_PATH.joinpath(args.out)} (共 {len(out_store)} 筆)")


if __name__ == "__main__":
    main()
