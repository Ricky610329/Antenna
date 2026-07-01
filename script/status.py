# -*- coding: utf-8 -*-
"""
script/status.py — 掃 NAS result/ 各 run 的即時狀態（取代手動猜，減少 ONGOING 狀態 churn）。

每 run：機器 / epoch / 每epoch耗時 / 狀態 / 最佳 worst_margin(+後20均) / skip。**純讀 NAS、不改任何東西。**

狀態怎麼判（由準到糙）：
  1) status.json（train.py 每 epoch 寫的心跳，含 state=running|finished|crashed）＝**權威終態**：
     當機/已完成一眼看出，不再把「正常結束」誤標成「卡住」。
  2) epoch 是否比「上次掃描」前進 → 前進＝**確認在跑**（鐵證）；宣稱 running 但隔了 >1.5 個 epoch
     還沒動、或心跳久沒更新 → **疑卡住**（抓硬砍/凍住：status.json 停在 running 但其實死了）。
     「上次掃描」快照存在本機 tmp/status_snapshot.json（**非 NAS**，git 忽略）；第一次掃無快照 → 退回心跳新鮮度。
  3) 無 status.json 的舊 run → 純時間啟發式（寫入夠新＝在跑? / 否則停止）。
  ⚠ 一次掃描分不出「剛停不到 1 個 epoch」與「mid-epoch 還在解」；連掃兩次看 epoch 有沒有前進最準。

用法（開發機可跑，離線讀 NAS）：
    python -m script.status                       # 全部 run
    python -m script.status --match single_r3     # 名稱含 single_r3 的
    python -m script.status --match single_r3 --md  # markdown 表（可貼進 configs/ONGOING.md）
"""
import argparse
import csv
import json
import os
import sys
import time as _time

from antenna.utils import config, ROOTDIR
config.device = "cpu"

#? 上次掃描快照：本機 tmp/（非 NAS、git 忽略）；比對 epoch 是否前進 → 「確認在跑 / 疑卡住」。
SNAPSHOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "tmp", "status_snapshot.json")


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


def _read_status(run_dir):
    """讀 <run>/status.json（train.py 每 epoch 寫的心跳）→ (state, hb_age_min)。
    state ∈ running|finished|crashed（權威終態）；無檔/壞檔 → (None, None)。"""
    p = run_dir.joinpath("status.json")
    if not p.exists():
        return None, None
    try:
        s = json.load(open(str(p), encoding="utf-8"))
    except Exception:
        return None, None
    hb_age_min = (_time.time() - os.path.getmtime(str(p))) / 60.0
    return s.get("state"), hb_age_min


def _liveness(*, state, advanced, elapsed_enough, age_min, tpe_min):
    """判 run 現況 → (label, is_live)。純函式（好測）。優先 status.json 終態（crashed/finished 權威）；
    running 再看「epoch 是否比上次掃描前進」(鐵證) + 心跳新鮮度；無 status.json 退回純時間啟發式。

    :param state:          status.json 的 state（running|finished|crashed|None）。
    :param advanced:       epoch 比上次快照前進（鐵證在跑）。
    :param elapsed_enough: 距上次快照已過 >1.5 個 epoch 卻仍沒前進（該動沒動 → 卡）。
    :param age_min:        心跳（或 metrics）距今分鐘，用來判新鮮度。
    :param tpe_min:        每 epoch 耗時（分），定「新鮮」門檻。
    """
    if state == "crashed":
        return "當機", False
    if state == "finished":
        return "已完成", False
    if advanced:
        return "在跑", True                      # 鐵證：epoch 前進
    fresh = age_min <= (max(20.0, 3.0 * tpe_min) if tpe_min > 0 else 20.0)
    if state == "running":
        if elapsed_enough or not fresh:          # 隔了>1.5ep 沒前進 / 心跳久沒更新 → 硬砍或凍住
            return "疑卡住", False
        return "在跑?", True                      # 宣稱 running、心跳新，但還沒確認前進
    return ("在跑?", True) if fresh else ("停止", False)   # 無 status.json 的舊 run


def _load_snapshot():
    """上次掃描的 {run 資料夾名: {epoch, ts}}（本機 tmp/、非 NAS）。壞/無 → {}。"""
    try:
        return json.load(open(SNAPSHOT, encoding="utf-8"))
    except Exception:
        return {}


def _save_snapshot(runs):
    """存本次快照（epoch + 現在時間）供下次比對 epoch 是否前進。寫本機 tmp/，NAS 一律不動。"""
    try:
        os.makedirs(os.path.dirname(SNAPSHOT), exist_ok=True)
        now = _time.time()
        snap = {r["dir"]: {"epoch": r["epoch"], "ts": now} for r in runs}
        json.dump(snap, open(SNAPSHOT, "w", encoding="utf-8"))
    except Exception:
        pass   # 快照純加分：寫不進去（唯讀碟等）也不該擋住看狀態


def scan(match=None, prev=None):
    """回傳每個 run 的狀態 dict list（依機器排序）。prev = 上次快照（{run: {epoch, ts}}）。"""
    prev = prev or {}
    rd = ROOTDIR.joinpath("result")
    now = _time.time()
    runs = []
    for d in os.listdir(str(rd)):
        if match and match not in d:
            continue
        run_dir = rd.joinpath(d)
        csvp = run_dir.joinpath("metrics.csv")
        if not csvp.exists():
            continue
        with open(str(csvp), newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        epoch = len(rows)
        age_min = (now - os.path.getmtime(str(csvp))) / 60.0
        t = _num(rows, "time")
        tpe_min = (sum(t[-20:]) / len(t[-20:]) / 60.0) if t else 0.0     # 後20 epoch 平均耗時(分)
        wm = _num(rows, "worst_margin")
        skip = sum(1 for r in rows if r.get("skipped") == "1.0")
        # 狀態：status.json 終態 + 「epoch 比上次掃描前進?」（鐵證在跑 / 該動沒動=卡）。
        state, hb_age_min = _read_status(run_dir)
        p = prev.get(d)
        advanced = bool(p and epoch > p.get("epoch", -1))
        elapsed_enough = bool(p and not advanced and tpe_min > 0
                              and (now - p.get("ts", now)) >= 1.5 * tpe_min * 60)
        age_for_fresh = hb_age_min if hb_age_min is not None else age_min
        label, is_live = _liveness(state=state, advanced=advanced, elapsed_enough=elapsed_enough,
                                   age_min=age_for_fresh, tpe_min=tpe_min)
        runs.append(dict(
            dir=d, short=d.split("]")[-1].strip().replace("pixel_", ""), machine=_machine(d),
            epoch=epoch, tpe_min=tpe_min, age_min=age_min, hb_age_min=hb_age_min, state=state,
            label=label, is_live=is_live, skip=skip,
            best_wm=(max(wm) if wm else None),
            recent_wm=(sum(wm[-20:]) / len(wm[-20:]) if wm else None),
        ))
    return sorted(runs, key=lambda r: (r["machine"], r["short"]))


def _fmt(v, spec="{:.2f}"):
    return spec.format(v) if v is not None else "—"


def main():
    #? Windows 主控台常是 cp950(Big5)，遇到非 Big5 字元會 UnicodeEncodeError → 強制 utf-8 輸出、免設 env。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="掃 NAS result/ 各 run 即時狀態（純讀 NAS、不改）")
    ap.add_argument("--match", default=None, help="只列名稱含此字串的 run")
    ap.add_argument("--md", action="store_true", help="輸出 markdown 表（可貼 ONGOING）")
    ap.add_argument("--live", action="store_true", help="只列判定為『在跑/在跑?/疑卡住』的 run（濾掉已結束/當機的舊 run）")
    args = ap.parse_args()

    prev = _load_snapshot()
    runs = scan(args.match, prev)
    _save_snapshot(runs)              # 存本次 epoch 快照供下次比對（本機 tmp/、不動 NAS）
    if args.live:
        runs = [r for r in runs if r["is_live"] or r["label"] == "疑卡住"]
    if not runs:
        print("（找不到符合的 run）")
        return
    if args.md:
        print("| run | 機器 | epoch | 分/ep | 狀態 | 最佳 wm | 後20均 | skip |")
        print("|---|---|---|---|---|---|---|---|")
        for r in runs:
            print(f"| {r['short'][:32]} | {r['machine']} | {r['epoch']} | {_fmt(r['tpe_min'],'{:.0f}')} "
                  f"| {r['label']}({r['age_min']:.0f}分前) | {_fmt(r['best_wm'])} | {_fmt(r['recent_wm'])} | {r['skip']} |")
    else:
        print(f"現在 {_time.strftime('%m/%d %H:%M')}  ({len(runs)} runs；連掃兩次看 epoch 有沒有前進最準)")
        for r in runs:
            print(f"  [{r['machine']}] {r['short'][:38]:<38} ep={r['epoch']:>4} "
                  f"{_fmt(r['tpe_min'],'{:.0f}'):>3}分/ep  {r['label']:<6}({r['age_min']:.0f}分前)  "
                  f"wm最佳={_fmt(r['best_wm'])} 後20={_fmt(r['recent_wm'])} skip={r['skip']}")


if __name__ == "__main__":
    main()
