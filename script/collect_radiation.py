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

import torch

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

    # 2) 已收過的 pattern 跳過 (依 pattern fingerprint，省 HFSS)
    out_store = SampleStore(DATASET_PATH.joinpath(args.out), verbose=True)
    seen = {fingerprint(out_store[i][0]) for i in range(len(out_store))}
    todo = [(p, tag) for p, tag in items if fingerprint(_bin(p)) not in seen]
    print(f"來源共 {len(items)} 張、已收過 {len(items) - len(todo)} 張 → 這次跑 {len(todo)} 張 HFSS")
    if not todo:
        print("沒有新 pattern 要收。"); return

    # 3) 跑 HFSS 方向圖、累積 (HFSS 開一次 → 迴圈 → 關)
    sim = SinglePortRadSimulator(record_path=str(Path(args.record).resolve()))
    sim.open()
    added = 0
    try:
        for num, (p, tag) in enumerate(todo):
            patt = _bin(p)
            print(f"[{num + 1}/{len(todo)}] {tag}  金屬={int(patt.sum())} … HFSS 求解中", flush=True)
            sim.start(num)
            sim(patt)
            sim.end()
            rad = sim.last_radiation
            if not (isinstance(rad, dict) and rad.get("theta") is not None):
                err = rad.get("error") if isinstance(rad, dict) else rad
                logger.warning(f"  方向圖萃取失敗，跳過：{err}")
                continue
            stack = torch.stack([rad["theta"].float(), rad["phi0"].float(), rad["phi90"].float()])  # (3, n_theta)
            if out_store.add(patt, stack):
                added += 1
    finally:
        sim.quit()
    print(f"\n✅ 完成：新增 {added} 筆 → {DATASET_PATH.joinpath(args.out)}  (共 {len(out_store)} 筆)")


if __name__ == "__main__":
    main()
