# -*- coding: utf-8 -*-
"""script/grind_loop.py — dual 慢磨維持的**無人值守迴圈**(Ricky 2026-08-21:「不用激活就放著跑」)。

把「補池 chunk → 收檔 → 判讀 → 破王則公證 → 週期重訓」這條我一直手動重跑的鏈寫成純腳本
(**迴圈裡沒有 LLM**),可 detached 長跑。治理合約(decisions「迴圈=儀器,round=判決書」)
在此嚴格生效:

* **迴圈不自己加冕**——破紀錄候選會自動公證(select-repeat ×2, prio 2),但結果只寫進
  `tmp/pending_records.jsonl`(待審),**絕不改 `docs/records_dual.json`**;換王仍由人/round 定。
* **異常就停**:單 chunk error 率 > 20%、機器全部靜默 > STALL_MIN、或 STOP 檔存在 → 收工。
* **開發機禮儀**:單例鎖 + 降優先權;HFSS 全在正式機(本迴圈只做生成/判讀/等待)。

用法(detached 長跑,見 §啟動):
    SM_DUAL_VER=v5 python -m script.grind_loop --chunks 200 --retrain-every 3
狀態:`tmp/grind_loop_status.json`(每步更新,人可讀)/ 日誌:`tmp/grind_loop.jsonl`
煞車:建立 `tmp/grind_loop.STOP`
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from antenna.utils import DATASET_PATH                      # noqa: E402

D = str(DATASET_PATH)
STOP = os.path.join(REPO, "tmp", "grind_loop.STOP")
LOCK = os.path.join(REPO, "tmp", "grind_loop.lock")
LOG = os.path.join(REPO, "tmp", "grind_loop.jsonl")
STATUS = os.path.join(REPO, "tmp", "grind_loop_status.json")
PENDING = os.path.join(REPO, "tmp", "pending_records.jsonl")
RECORDS = os.path.join(REPO, "docs", "records_dual.json")
STALL_MIN = 90            # 所有 store 這麼久沒新結果 = 機器出事
ERR_RATE_STOP = 0.20      # 單 chunk error 率上限


def wm2(v):
    return min(v["m1"] + 2, v["m2"] + 2, v["m3"], v["m4"] + 5)


def _run(args, timeout=3600):
    """跑子命令,回 (rc, stdout)。"""
    env = dict(os.environ)
    env.setdefault("SM_DUAL_VER", "v5")
    p = subprocess.run([sys.executable, "-X", "utf8", "-m"] + args, cwd=REPO, env=env,
                       capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def current_record():
    with open(RECORDS, encoding="utf-8") as f:
        return float(json.load(f)["wm_mfg"]["value"])


def latest_chunk_no():
    import glob as g
    ns = []
    for p in g.glob(os.path.join(D, "dedust_smp*")):
        b = os.path.basename(p)
        if b.endswith("_input"):
            b = b[:-6]
        try:
            ns.append(int(b[len("dedust_smp"):-1]))
        except ValueError:
            pass
    return max(ns) if ns else 0


def read_store(stores):
    """回 [(arm, vid, wm, entry)],以及 (完成數, error 數)。"""
    rows, done, err = [], 0, 0
    for st in stores:
        rj = os.path.join(D, st, "results.json")
        if not os.path.exists(rj):
            continue
        try:
            res = json.load(open(rj, encoding="utf-8"))
        except Exception:
            continue
        try:
            man = {m["id"]: m for m in json.load(
                open(os.path.join(D, st + "_input", "manifest.json"), encoding="utf-8"))}
        except Exception:
            man = {}
        for vid, v in res.items():
            done += 1
            if isinstance(v, dict) and "m1" in v:
                rows.append((man.get(vid, {}).get("arm", "?"), vid, wm2(v), v))
            else:
                err += 1
    return rows, done, err


def wait_stores(stores, poll=120, stall_min=STALL_MIN):
    """輪詢到全部終態;回 (ok, 訊息)。stale 偵測=久無新結果。"""
    t_last, n_last = time.time(), -1
    while True:
        if os.path.exists(STOP):
            return False, "STOP 檔"
        rows, done, err = read_store(stores)
        state_done = 0
        for st in stores:
            js = os.path.join(D, "jobs_state", st + ".json")
            if os.path.exists(js):
                try:
                    if json.load(open(js, encoding="utf-8")).get("status") in ("done", "DONE"):
                        state_done += 1
                except Exception:
                    pass
        if done != n_last:
            n_last, t_last = done, time.time()
        if state_done >= len(stores) or done >= 90:
            return True, f"完成 {done}"
        if (time.time() - t_last) / 60 > stall_min:
            return False, f"停滯 {stall_min} 分無新結果(done={done})"
        time.sleep(poll)


def notarize(vid, src_input, tag):
    """公證:重測 ×2(prio 2)→ 等 → 回 (值列表, 一致?)。不動 records。"""
    ni = f"dedust_{tag}_input"
    rc, out = _run(["script.dedust", "select-repeat", "--source-input", src_input,
                    "--id", vid, "--n", "2", "--input", ni])
    if rc != 0:
        return [], False, out[-200:]
    try:
        json.dump({"diag_bridge_w": 0.075},
                  open(os.path.join(D, ni, "hfss_setup.json"), "w", encoding="utf-8"))
        mp = os.path.join(D, ni, "manifest.json")
        man = json.load(open(mp, encoding="utf-8"))
        for m in man:
            m["kind"] = "diagbridge"
        json.dump(man, open(mp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    except Exception as e:
        return [], False, f"公證夾設定失敗 {e}"
    _run(["script.dedust", "jobs-add", "--input", ni, "--store", f"dedust_{tag}",
          "--prio", "2", "--config", "configs/dual_r1_eval.yaml"])
    ok, msg = wait_stores([f"dedust_{tag}"], poll=60, stall_min=45)
    rows, _, _ = read_store([f"dedust_{tag}"])
    vals = sorted(round(r[2], 3) for r in rows)
    return vals, (len(vals) >= 2 and max(vals) - min(vals) <= 0.03), msg


def write_status(**kw):
    kw["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(STATUS, "w", encoding="utf-8") as f:
        json.dump(kw, f, ensure_ascii=False, indent=1)


def log(rec):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps(rec, ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=200, help="最多跑幾個 chunk")
    ap.add_argument("--retrain-every", type=int, default=3)
    ap.add_argument("--n", type=int, default=90)
    ap.add_argument("--cand", type=int, default=6000)
    args = ap.parse_args()
    if os.path.exists(LOCK):
        raise SystemExit(f"已有 grind_loop 在跑({LOCK});煞車請建 {STOP}")
    open(LOCK, "w", encoding="utf-8").write(str(os.getpid()))
    try:
        import psutil
        psutil.Process(os.getpid()).nice(getattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS", 10))
    except Exception:
        pass
    seed = int(time.time()) % 100000
    try:
        for it in range(1, args.chunks + 1):
            if os.path.exists(STOP):
                log({"event": "stop", "reason": "STOP 檔", "iter": it}); break
            rec = current_record()
            k = latest_chunk_no() + 1
            stores = [f"dedust_smp{k:03d}{s}" for s in "abc"]
            write_status(state="生成中", chunk=k, iter=it, record=rec)
            rc, out = _run(["script.dedust", "select-smpool", "--n", str(args.n),
                            "--cand", str(args.cand), "--seed", str(seed + it), "--dispatch"])
            if rc != 0:
                log({"event": "error", "stage": "select-smpool", "out": out[-300:]}); break
            write_status(state="等待收檔", chunk=k, iter=it, record=rec, stores=stores)
            ok, msg = wait_stores(stores)
            rows, done, err = read_store(stores)
            if not ok:
                log({"event": "stop", "reason": msg, "chunk": k}); break
            if done and err / max(done, 1) > ERR_RATE_STOP:
                log({"event": "stop", "reason": f"error 率 {err}/{done}", "chunk": k}); break
            if not rows:
                log({"event": "stop", "reason": "無有效結果", "chunk": k}); break
            arm, vid, best, v = max(rows, key=lambda r: r[2])
            entry = {"event": "chunk", "chunk": k, "n": len(rows), "err": err,
                     "best": round(best, 3), "id": vid, "arm": arm,
                     "sum": round(v["m3"] + v["m4"] + 5, 2),
                     "imb": round(abs(v["m3"] - (v["m4"] + 5)), 2), "record": rec}
            if best > rec:                                   # 破紀錄 → 公證(不加冕)
                src = None
                for s in "abc":
                    if os.path.exists(os.path.join(D, f"dedust_smp{k:03d}{s}_input", vid + ".pt")):
                        src = f"dedust_smp{k:03d}{s}_input"; break
                if src:
                    write_status(state="公證中", chunk=k, iter=it, record=rec, cand=vid, cand_wm=best)
                    vals, consistent, nmsg = notarize(vid, src, f"gl{k:03d}n")
                    entry |= {"notarize": vals, "consistent": consistent, "note": nmsg}
                    if consistent and vals and max(vals) > rec:
                        with open(PENDING, "a", encoding="utf-8") as f:
                            f.write(json.dumps({"ts": time.strftime("%F %T"), "id": vid,
                                                "wm": max(vals), "prev_record": rec,
                                                "repeats": vals, "chunk": k,
                                                "status": "待審(迴圈不加冕)"}, ensure_ascii=False) + "\n")
                        entry["pending_record"] = True
            log(entry)
            if it % args.retrain_every == 0:
                write_status(state="SM 重訓", chunk=k, iter=it, record=rec)
                cache = os.path.join(REPO, "tmp", "sm_dual_pool_v5.npz")
                if os.path.exists(cache):
                    os.remove(cache)
                _run(["script.sm_dual", "train"], timeout=5400)
                rc2, out2 = _run(["script.sm_dual", "eval"], timeout=1800)
                gate = [ln for ln in out2.splitlines() if "品質閘" in ln]
                log({"event": "retrain", "chunk": k, "gate": gate[-1] if gate else "?"})
        write_status(state="結束")
    finally:
        try:
            os.remove(LOCK)
        except OSError:
            pass


if __name__ == "__main__":
    main()
