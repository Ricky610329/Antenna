# -*- coding: utf-8 -*-
"""script/online_loop.py — 線上學習迴圈(R73 首航;Ricky 2026-08-15「一次候選幾個 pattern 再送 HFSS」)。

每迭代:生成候選 → SM v5 ensemble 預測 → **受約束獲取**(預測三軸底線 ≥ 門檻內
argmax 預測 m4' + 可行域高分歧)→ 小店 jobs-add(prio 3,K 筆)→ 輪詢收檔 → 回鍋
→ 每 RETRAIN_EVERY 迭代全重訓 SM(mfg 鍋自動吸收新店)→ 下一迭代。

紀律:單例鎖(tmp/online_loop.lock)/`tmp/online73.STOP` 隨時煞車/每迭代 manifest 帶
pred_*(前瞻 ρ 帳)/店=dedust_<prefix>NN(kind=diagbridge,標準橋,雙章由 worker 蓋)。
判準與收檔規則在 round 檔(docs/log/round-73-*),本檔只執行不判準。

用法(開發機,背景跑):
    SM_DUAL_VER=v5 python -m script.online_loop --iters 15 --k 9 --prefix r73o
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import script.sm_dual as SD                                  # noqa: E402
from script.dedust import _dir, dual_pads, dual_flip, DATASET_PATH  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STOP_FILE = os.path.join(REPO, "tmp", "online73.STOP")
LOCK = os.path.join(REPO, "tmp", "online_loop.lock")
LOG = os.path.join(REPO, "tmp", "online73_log.jsonl")
OFF = np.array([2, 2, 0, 5, 0, 0], np.float32)               # 規格 v2 平移(=SPEC_V2_OFF+帶外零)

D = str(DATASET_PATH)


def find_pattern(vid):
    """從任何 dual 25×25 輸入夾撈 pattern(bits)。"""
    import glob
    for fol in glob.glob(os.path.join(D, "dedust_*_input")):
        f = os.path.join(fol, vid + ".pt")
        if os.path.exists(f):
            return np.asarray(torch.load(f, weights_only=True)).reshape(25, 25) > 0.5
    raise SystemExit(f"找不到 {vid}")


def load_hist():
    import glob
    hist = set()
    for fol in glob.glob(os.path.join(D, "dedust_*_input")):
        try:
            man = json.load(open(os.path.join(fol, "manifest.json"), encoding="utf-8"))
        except Exception:
            continue
        if not (man and isinstance(man, list) and man[0].get("port") == "dual"
                and man[0].get("pixel_count", 25) == 25):
            continue
        for m in man:
            f = os.path.join(fol, m["id"] + ".pt")
            if os.path.exists(f):
                hist.add((np.asarray(torch.load(f, weights_only=True)).reshape(-1) > 0.5).tobytes())
    return hist


def gen_candidates(frontier, donor, king, hist, rng, want=400):
    """縫合候選:捐贈者塊移植(2×2~6×6)+ frontier 點翻 d1-3 + 欄段交換。"""
    out = []
    tries = 0
    while len(out) < want and tries < want * 30:
        tries += 1
        r = rng.random()
        if r < 0.5:                                          # 塊移植:donor → frontier
            h, w = int(rng.integers(2, 7)), int(rng.integers(2, 7))
            r0, c0 = int(rng.integers(0, 25 - h)), int(rng.integers(0, 25 - w))
            q = frontier.copy()
            q[r0:r0 + h, c0:c0 + w] = donor[r0:r0 + h, c0:c0 + w]
        elif r < 0.8:                                        # 點翻
            base = frontier if rng.random() < 0.7 else king
            q = dual_flip(base, int(rng.integers(1, 4)), rng)
            q = np.asarray(q)
        else:                                                # 欄段
            c1, c2 = sorted(rng.choice(np.arange(2, 24), size=2, replace=False))
            q = frontier.copy()
            q[:, c1:c2] = donor[:, c1:c2]
        q = dual_pads(q)
        b = q.reshape(-1).tobytes()
        if not (150 <= int(q.sum()) <= 550) or b in hist:
            continue
        hist.add(b)
        out.append(q.reshape(-1))
    return out


def acquire(X, models, trio_floor, k, rng):
    """受約束獲取:pred trio=min(m1',m2',m3) ≥ trio_floor 中 argmax pred m4'(k-2)
    +可行域高分歧(2)。回 (索引清單, 預測表)。可行域空 → 放寬 0.5 再試。"""
    _, G = SD.predict(models, np.stack(X).astype(np.float32))
    Gm = (G + OFF).mean(0)                                   # (n,6) 平移後平均
    std4 = SD.wm_r2_from_margins(G).std(0)
    trio = Gm[:, :3].min(1)
    floor = trio_floor
    feas = np.flatnonzero(trio >= floor)
    while len(feas) < max(k, 20) and floor > -10:        # 鬆到可行域 ≥ max(k,20)(SM 頂端壓縮:
        floor -= 0.5                                     # 預測 trio 系統性偏低,絕對門檻靠不住,
        feas = np.flatnonzero(trio >= floor)             # 排位才可信——見 round-72 §b2 判讀)
    order = feas[np.argsort(Gm[feas, 3])[::-1]]
    picks = list(order[:max(1, k - 2)])
    for i in feas[np.argsort(std4[feas])[::-1]]:
        if len(picks) >= k:
            break
        if i not in picks:
            picks.append(int(i))
    return [int(i) for i in picks[:k]], Gm, floor


def dispatch_and_wait(iter_no, prefix, X, picks, Gm, timeout_min=90):
    store = f"dedust_{prefix}{iter_no:02d}"
    ind = _dir(store + "_input")
    ind.mkdir(parents=True, exist_ok=True)
    man = []
    for j, i in enumerate(picks):
        vid = f"d{prefix}{iter_no:02d}_{j:02d}"
        torch.save(torch.tensor(np.asarray(X[i]).reshape(25, 25), dtype=torch.float32),
                   str(ind.joinpath(vid + ".pt")))
        man.append(dict(id=vid, kind="diagbridge", port="dual", arm="ol",
                        family="DUAL_OL73", diag_bridge_w=0.075, sm_ver=SD.SM_VER,
                        pred_m1p=round(float(Gm[i, 0]), 2), pred_m2p=round(float(Gm[i, 1]), 2),
                        pred_m3=round(float(Gm[i, 2]), 2), pred_m4p=round(float(Gm[i, 3]), 2),
                        metal_px=int(np.asarray(X[i]).sum())))
    json.dump(man, open(str(ind.joinpath("manifest.json")), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump({"diag_bridge_w": 0.075}, open(str(ind.joinpath("hfss_setup.json")), "w", encoding="utf-8"))
    subprocess.run([sys.executable, "-X", "utf8", "-m", "script.dedust", "jobs-add",
                    "--input", store + "_input", "--store", store, "--prio", "3",
                    "--config", "configs/dual_r1_eval.yaml"], check=True)
    t0 = time.time()
    rj = os.path.join(D, store, "results.json")
    while time.time() - t0 < timeout_min * 60:
        if os.path.exists(STOP_FILE):
            return store, None
        if os.path.exists(rj):
            try:
                res = json.load(open(rj, encoding="utf-8"))
                done = sum(1 for v in res.values() if isinstance(v, dict) and ("m1" in v or "error" in v))
                if done >= len(picks):
                    return store, res
            except Exception:
                pass
        time.sleep(60)
    return store, json.load(open(rj, encoding="utf-8")) if os.path.exists(rj) else {}


def wm2(v):
    return min(v["m1"] + 2, v["m2"] + 2, v["m3"], v["m4"] + 5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--k", type=int, default=9)
    ap.add_argument("--prefix", default="r73o")
    ap.add_argument("--frontier", default="d72b3_W_25")       # 三軸體
    ap.add_argument("--donor", default="d72b2_c_84")          # 阻帶捐贈
    ap.add_argument("--king", default="d72b3_K_d_29")
    ap.add_argument("--retrain-every", type=int, default=3)
    ap.add_argument("--seed", type=int, default=73001)
    args = ap.parse_args()
    if os.path.exists(LOCK):
        raise SystemExit(f"已有 online_loop 在跑({LOCK});煞車用 STOP 檔非殺進程。")
    open(LOCK, "w").write(str(os.getpid()))
    try:
        rng = np.random.default_rng(args.seed)
        frontier = find_pattern(args.frontier)
        donor = find_pattern(args.donor)
        king = find_pattern(args.king)
        frontier_id, frontier_score = args.frontier, None
        hist = load_hist()
        for it in range(1, args.iters + 1):
            if os.path.exists(STOP_FILE):
                print(f"[loop] STOP 檔存在,迭代 {it} 前停止")
                break
            models = SD.load_models()
            X = gen_candidates(frontier, donor, king, hist, rng)
            if len(X) < args.k:
                print(f"[loop] 候選不足({len(X)}),停止")
                break
            # trio 底線:frontier 的實測 trio − 0.7(首迭代用預測近似)
            _, Gf = SD.predict(models, frontier.reshape(1, -1).astype(np.float32))
            trio_now = float((Gf.mean(0)[0, :3] + OFF[:3]).min())
            picks, Gm, floor = acquire(X, models, trio_now - 0.7, args.k, rng)
            print(f"[iter {it}] 候選 {len(X)} → 選 {len(picks)}(trio 底線 {floor:+.2f});發車…")
            store, res = dispatch_and_wait(it, args.prefix, X, picks, Gm)
            if res is None:
                print("[loop] STOP 中斷"); break
            ent = [(vid, v) for vid, v in res.items() if isinstance(v, dict) and "m1" in v]
            best = max(ent, key=lambda kv: wm2(kv[1]), default=None)
            log = dict(iter=it, store=store, n=len(ent))
            if best:
                vid, v = best
                log |= dict(best=vid, wm=round(wm2(v), 2), m3=v["m3"], m4p=round(v["m4"] + 5, 2))
                # frontier 更新:實測「trio≥−2 內 m4' 最大」優先,否則 wm 最大
                def trio_of(v):
                    return min(v["m1"] + 2, v["m2"] + 2, v["m3"])
                grow = [(vid2, v2) for vid2, v2 in ent if trio_of(v2) >= -2]
                pick_f = max(grow, key=lambda kv: kv[1]["m4"], default=None) or best and (vid, v)
                if pick_f:
                    fid, fv = pick_f if isinstance(pick_f, tuple) else best
                    new_score = (trio_of(fv), fv["m4"])
                    if frontier_score is None or new_score > frontier_score:
                        frontier_score = new_score
                        frontier_id = fid
                        frontier = find_pattern(fid)
                        log["frontier"] = fid
                meets = [vid2 for vid2, v2 in ent if v2["m3"] > -2 and v2["m4"] + 5 > -2]
                if meets:
                    log["meet"] = meets
                    print(f"[iter {it}] ★會師:{meets}")
            with open(LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(log, ensure_ascii=False) + "\n")
            print(f"[iter {it}] {log}")
            if it % args.retrain_every == 0:
                print(f"[iter {it}] SM 重訓(先清快取→重掃吸收本迴圈新店)…")
                if os.path.exists(SD.CACHE_PATH):
                    os.remove(SD.CACHE_PATH)              #! 必在 train 前:否則重訓用舊鍋
                env = dict(os.environ, SM_DUAL_VER=SD.SM_VER)
                subprocess.run([sys.executable, "-X", "utf8", "-m", "script.sm_dual", "train"],
                               env=env, check=False, capture_output=True)
        print(f"[loop] 結束;frontier={frontier_id}")
    finally:
        try:
            os.remove(LOCK)
        except OSError:
            pass


if __name__ == "__main__":
    SD.throttle()
    main()
