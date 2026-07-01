# -*- coding: utf-8 -*-
"""
script/status.py — 掃 NAS result/ 各 run 的即時狀態（取代手動猜，減少 ONGOING 狀態 churn）。

每 run：機器 / epoch / 每epoch耗時 / alive|卡住 / 最佳 worst_margin(+後20均) / skip。**純讀 NAS、不改任何東西。**

用法（開發機可跑，離線讀 NAS）：
    python -m script.status                       # 全部 run
    python -m script.status --match single_r3     # 名稱含 single_r3 的
    python -m script.status --match single_r3 --md  # markdown 表（可貼進 configs/ONGOING.md）
"""
import argparse
import csv
import os
import time as _time

from antenna.utils import config, ROOTDIR
config.device = "cpu"


def _num(rows, key):
    out = []
    for r in rows:
        v = r.get(key, "")
        if v not in ("", "nan", None):
            try:
                out.append(float(v))
            except ValueError:
                pass
    return out


def _machine(name):
    """資料夾名 '[Patch-single-<ip>-<hash>] …' → 取 <ip>（機器識別）。"""
    try:
        return name.split("Patch-single-")[1].split("-")[0]
    except Exception:
        return "?"


def scan(match=None):
    """回傳每個 run 的狀態 dict list（依機器排序）。"""
    rd = ROOTDIR.joinpath("result")
    runs = []
    for d in os.listdir(str(rd)):
        if match and match not in d:
            continue
        csvp = rd.joinpath(d, "metrics.csv")
        if not csvp.exists():
            continue
        with open(str(csvp), newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        age_min = (_time.time() - os.path.getmtime(str(csvp))) / 60.0
        t = _num(rows, "time")
        tpe_min = (sum(t[-20:]) / len(t[-20:]) / 60.0) if t else 0.0     # 後20 epoch 平均耗時(分)
        wm = _num(rows, "worst_margin")
        skip = sum(1 for r in rows if r.get("skipped") == "1.0")
        # 卡住判定：距上次寫入 > max(20分, 3×每epoch耗時) → 多半 HFSS 卡/掛
        stalled = tpe_min > 0 and age_min > max(20.0, 3.0 * tpe_min)
        runs.append(dict(
            short=d.split("]")[-1].strip().replace("pixel_", ""), machine=_machine(d),
            epoch=len(rows), tpe_min=tpe_min, age_min=age_min, stalled=stalled, skip=skip,
            best_wm=(max(wm) if wm else None),
            recent_wm=(sum(wm[-20:]) / len(wm[-20:]) if wm else None),
        ))
    return sorted(runs, key=lambda r: (r["machine"], r["short"]))


def _fmt(v, spec="{:.2f}"):
    return spec.format(v) if v is not None else "—"


def main():
    ap = argparse.ArgumentParser(description="掃 NAS result/ 各 run 即時狀態（純讀、不改）")
    ap.add_argument("--match", default=None, help="只列名稱含此字串的 run")
    ap.add_argument("--md", action="store_true", help="輸出 markdown 表（可貼 ONGOING）")
    args = ap.parse_args()

    runs = scan(args.match)
    if not runs:
        print("（找不到符合的 run）")
        return
    if args.md:
        print("| run | 機器 | epoch | 分/ep | 狀態 | 最佳 wm | 後20均 | skip |")
        print("|---|---|---|---|---|---|---|---|")
        for r in runs:
            state = "⚠卡住" if r["stalled"] else "🔵跑動"
            print(f"| {r['short'][:32]} | {r['machine']} | {r['epoch']} | {_fmt(r['tpe_min'],'{:.0f}')} "
                  f"| {state}({r['age_min']:.0f}分前) | {_fmt(r['best_wm'])} | {_fmt(r['recent_wm'])} | {r['skip']} |")
    else:
        print(f"現在 {_time.strftime('%m/%d %H:%M')}  ({len(runs)} runs)")
        for r in runs:
            state = "⚠ 卡住/停" if r["stalled"] else "跑動中"
            print(f"  [{r['machine']}] {r['short'][:38]:<38} ep={r['epoch']:>4} "
                  f"{_fmt(r['tpe_min'],'{:.0f}'):>3}分/ep  {state}({r['age_min']:.0f}分前)  "
                  f"wm最佳={_fmt(r['best_wm'])} 後20={_fmt(r['recent_wm'])} skip={r['skip']}")


if __name__ == "__main__":
    main()
