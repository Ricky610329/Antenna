"""Pattern Browser API 冒煙測試:以 --fixture 啟 server 子行程 → 打每個 API → 驗 schema。

用法(任一目錄): python -X utf8 application/pattern_browser/selftest.py
退出碼: 0=全過, 1=有失敗。tests/ 正式測試不動,本檔只服務瀏覽器工具自己。
v2:加 /api/resp、/api/radc、/api/targets、lite/新 sort;最後用臨時 data/(只有
patterns.npz+meta.json、無曲線檔)再開一個 server,驗缺檔容錯(has_* 全 False、曲線 API 回 null)。
"""
import base64
import http.client
import json
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
SERVER = PKG_DIR / "server.py"
META_KEYS = {"id", "folder", "store", "wm", "rad", "lo", "sel", "total", "n4", "n8",
             "largest8_frac", "ndiag", "has_db100", "has_sl100", "db100_wm", "sl100_wm", "kind",
             "has_resp", "has_rad"}  # 後兩者=v2
NFREQ, NTHETA = 17, 181


def num_or_null(seq):
    return all(v is None or isinstance(v, (int, float)) for v in seq)

FAILS = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {name}" + (f" — {detail}" if (detail and not cond) else ""))
    if not cond:
        FAILS.append(name)


def get(base, path):
    try:
        with urllib.request.urlopen(base + path, timeout=10) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def get_json(base, path, want_status=200):
    status, body = get(base, path)
    if status != want_status:
        return status, None
    return status, json.loads(body.decode("utf-8"))


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_ready(base, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(base + "/api/stats", timeout=2) as r:
                if r.status == 200:
                    return True
        except OSError:
            time.sleep(0.25)
    return False


def missing_curves_test():
    """開發期資料層契約:data/ 只有 patterns.npz+meta.json(無 v2 曲線檔)→ server 照常起,
    has_resp/has_rad 全 False、曲線 API 回 null、stats 覆蓋數=0。"""
    import shutil

    import numpy as np
    tmp = Path(tempfile.mkdtemp(prefix="pb_selftest_"))
    rng = np.random.default_rng(3)
    mats = (rng.random((3, 625)) > 0.5).astype(np.uint8)
    np.savez(tmp / "patterns.npz",
             packed=np.stack([np.packbits(m) for m in mats]),
             ids=np.array(["t0", "t1", "t2"]))
    metas = [{"id": f"t{i}", "folder": "tmp", "store": None, "wm": 0.2, "rad": 1.0, "lo": -5.0,
              "sel": -10.0, "total": int(mats[i].sum()), "n4": 1, "n8": 1, "largest8_frac": 1.0,
              "ndiag": 0, "has_db100": False, "has_sl100": False, "db100_wm": None,
              "sl100_wm": None, "kind": None} for i in range(3)]
    (tmp / "meta.json").write_text(json.dumps(metas), encoding="utf-8")
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(
        [sys.executable, "-X", "utf8", str(SERVER), "--data", str(tmp), "--port", str(port)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
    try:
        if not wait_ready(base):
            check("缺曲線檔 server 30s 內就緒", False)
            return
        st, data = get_json(base, "/api/list?limit=10")
        check("缺曲線檔:list 照常+has_* 全 False", st == 200 and data["total"] == 3 and
              all(r["has_resp"] is False and r["has_rad"] is False for r in data["rows"]))
        st, r = get_json(base, "/api/resp?ids=t0,t1")
        check("缺曲線檔:/api/resp 回 null", st == 200 and r["t0"] is None and r["t1"] is None)
        st, r = get_json(base, "/api/radc?ids=t0")
        check("缺曲線檔:/api/radc 回 null", st == 200 and r["t0"] is None)
        st, stats = get_json(base, "/api/stats")
        check("缺曲線檔:stats resp/rad_count=0", st == 200 and
              stats["resp_count"] == 0 and stats["rad_count"] == 0)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(
        [sys.executable, "-X", "utf8", str(SERVER), "--fixture", "--port", str(port)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
    try:
        if not wait_ready(base):
            print("[FAIL] server 沒有在 30s 內就緒")
            return 1

        # ---- 靜態檔 ----
        st, body = get(base, "/")
        check("GET / 回 index.html", st == 200 and b"app.js" in body)
        st, _ = get(base, "/static/app.js")
        check("GET /static/app.js", st == 200)
        st, _ = get(base, "/static/style.css")
        check("GET /static/style.css", st == 200)
        st, body = get(base, "/static/help.js")
        check("GET /static/help.js(說明頁內容)", st == 200 and b"HELP_HTML" in body)
        helptxt = body.decode("utf-8") if st == 200 else ""
        check("help 章節齊(快速開始/資料/範例/名詞表/FAQ)",
              all(k in helptxt for k in ("h-quickstart", "h-data", "h-views", "h-persp",
                                         "h-examples", "h-glossary", "h-faq")))
        check("help 五個 step-by-step 範例",
              all(f"h-ex-{c}" in helptxt for c in "abcde"))
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)  # raw path,繞過 urllib 正規化
        conn.request("GET", "/static/../server.py")
        resp = conn.getresponse()
        resp.read()
        check("路徑跳脫被擋(/static/../server.py)", resp.status in (403, 404), f"status={resp.status}")
        conn.close()

        # ---- /api/stats ----
        st, stats = get_json(base, "/api/stats")
        check("/api/stats 200", st == 200)
        check("stats schema", stats is not None and
              all(k in stats for k in ("total", "qualified", "ndiag_hist", "n8_hist",
                                       "db100_count", "sl100_count")))
        check("stats.total == 200(fixture)", stats and stats["total"] == 200)
        check("stats 直方圖形狀", stats and
              all(isinstance(p, list) and len(p) == 2 for p in stats["ndiag_hist"] + stats["n8_hist"]))

        # ---- /api/list ----
        st, data = get_json(base, "/api/list?limit=7")
        check("/api/list 200", st == 200)
        check("list 分頁", data and data["total"] == 200 and len(data["rows"]) == 7)
        row = data["rows"][0]
        check("list row meta 欄齊", META_KEYS <= set(row), f"缺 {META_KEYS - set(row)}")
        check("list row bits_b64=79 bytes", len(base64.b64decode(row["bits_b64"])) == 79)

        st, data = get_json(base, "/api/list?f_n8_min=2&f_n8_max=3&limit=200")
        check("篩選 f_n8_min/max", st == 200 and data["rows"] and
              all(2 <= r["n8"] <= 3 for r in data["rows"]))
        st, data = get_json(base, "/api/list?f_qual=1&limit=200")
        check("篩選 f_qual=1(wm≥0)", st == 200 and
              all(r["wm"] is not None and r["wm"] >= 0 for r in data["rows"]))
        st, data = get_json(base, "/api/list?sort=wm&dir=asc&limit=200")
        wms = [r["wm"] for r in data["rows"]]
        vals = [w for w in wms if w is not None]
        check("排序 wm asc + None 殿後", vals == sorted(vals) and
              wms[:len(vals)] == vals, "非遞增或 None 沒排最後")
        st, data = get_json(base, "/api/list?q=fx000&limit=50")
        check("id 子串搜尋 q=fx000", st == 200 and data["total"] == 10 and
              all("fx000" in r["id"] for r in data["rows"]))
        st, _ = get_json(base, "/api/list?sort=nope", want_status=400)
        check("list 壞 sort → 400", st == 400)

        # ---- /api/pattern ----
        st, p = get_json(base, "/api/pattern/fx0000")
        check("/api/pattern 200", st == 200)
        check("pattern bits=625 個 0/1", p and len(p["bits"]) == 625 and
              set(p["bits"]) <= {0, 1})
        check("pattern bits 總和==total", p and sum(p["bits"]) == p["total"])
        check("pattern diag_sites 形狀", p and isinstance(p["diag_sites"], list) and
              all(len(s) == 3 for s in p["diag_sites"]))
        check("pattern ndiag==len(diag_sites)", p and p["ndiag"] == len(p["diag_sites"]))
        check("diag_sites 座標在 0..5mm", p and
              all(0 <= s[0] <= 5 and 0 <= s[1] <= 5 for s in p["diag_sites"]))

        st, pv = get_json(base, "/api/pattern/fx0000~db100")
        check("變體 id 可查(fx0000~db100)", st == 200)
        check("變體 bits 同親本+有標記", pv and pv["bits"] == p["bits"] and
              pv.get("variant") == "db100" and pv.get("variant_of") == "fx0000" and
              pv["id"] == "fx0000~db100")
        st, _ = get_json(base, "/api/pattern/no_such_id", want_status=404)
        check("未知 id → 404", st == 404)

        # ---- /api/hamming ----
        st, rows = get_json(base, "/api/hamming?id=fx0000&maxd=625&limit=10")
        ok = st == 200 and isinstance(rows, list) and len(rows) == 10
        check("/api/hamming 200+陣列", ok)
        rows = rows if ok else []
        ds = [r["d"] for r in rows]
        check("hamming 距離遞增排序", bool(rows) and ds == sorted(ds))
        check("hamming 排除自身", bool(rows) and all(r["id"] != "fx0000" for r in rows))
        near = rows[0] if rows else {}
        check("hamming 最近鄰=fx0001, d=3(fixture 植入)",
              near.get("id") == "fx0001" and near.get("d") == 3,
              f"got {near.get('id')} d={near.get('d')}")
        check("hamming row 含 meta+bits_b64", bool(rows) and "wm" in rows[0] and "bits_b64" in rows[0])
        st, rows2 = get_json(base, "/api/hamming?id=fx0000&maxd=3&limit=50")
        check("hamming maxd 過濾", st == 200 and isinstance(rows2, list) and
              all(r["d"] <= 3 for r in rows2))
        st, _ = get_json(base, "/api/hamming", want_status=400)
        check("hamming 缺 id → 400", st == 400)

        # ---- /api/compare ----
        st, items = get_json(base, "/api/compare?ids=fx0000,fx0001")
        check("/api/compare 200+2 筆", st == 200 and isinstance(items, list) and len(items) == 2)
        xor = sum(1 for a, b in zip(items[0]["bits"], items[1]["bits"]) if a != b)
        check("compare XOR 數=3(對應植入差異)", xor == 3, f"xor={xor}")
        st, _ = get_json(base, "/api/compare?ids=fx0000", want_status=400)
        check("compare 1 筆 → 400", st == 400)
        st, _ = get_json(base, "/api/compare?ids=a,b,c,d,e", want_status=400)
        check("compare 5 筆 → 400", st == 400)

        # ---- v2 /api/targets ----
        st, tg = get_json(base, "/api/targets")
        check("/api/targets 200", st == 200)
        check("targets schema", tg is not None and all(
            k in tg for k in ("band", "s11_max", "gain_min", "wm_buffer", "rad_window", "rad_floor")))
        check("targets band=[26.5,29.5]", tg and tg["band"] == [26.5, 29.5])
        check("targets freqs 17 點 24–32", tg and len(tg.get("freqs", [])) == NFREQ and
              tg["freqs"][0] == 24 and tg["freqs"][-1] == 32)

        # ---- v2 /api/resp ----
        st, r = get_json(base, "/api/resp?ids=fx0000,fx0000~db100,no_such_id")
        check("/api/resp 200", st == 200)
        c = (r or {}).get("fx0000")
        check("resp fx0000 s11/gain 各 17 點", c is not None and
              len(c["s11"]) == NFREQ and len(c["gain"]) == NFREQ and
              num_or_null(c["s11"]) and num_or_null(c["gain"]))
        cv = (r or {}).get("fx0000~db100")
        check("resp 變體曲線(fx0000~db100)", cv is not None and len(cv["s11"]) == NFREQ)
        check("resp 未知 id → null", r is not None and r.get("no_such_id") is None)
        st, _ = get_json(base, "/api/resp", want_status=400)
        check("resp 缺 ids → 400", st == 400)

        # ---- v2 /api/radc ----
        st, r = get_json(base, "/api/radc?ids=fx0000,fx0000~sl100,no_such_id")
        check("/api/radc 200", st == 200)
        c = (r or {}).get("fx0000")
        check("radc fx0000 theta/phi0/phi90 各 181 點", c is not None and
              len(c["theta"]) == NTHETA and len(c["phi0"]) == NTHETA and len(c["phi90"]) == NTHETA)
        check("radc theta -90..90", c is not None and c["theta"][0] == -90 and c["theta"][-1] == 90)
        cv = (r or {}).get("fx0000~sl100")
        check("radc 變體切面(fx0000~sl100)", cv is not None and len(cv["phi0"]) == NTHETA)
        check("radc 未知 id → null", r is not None and r.get("no_such_id") is None)

        # ---- v2 list:has_resp/has_rad、lite、新 sort ----
        st, data = get_json(base, "/api/list?limit=1")
        row = data["rows"][0] if (data and data["rows"]) else {}
        check("list row 含 has_resp/has_rad", "has_resp" in row and "has_rad" in row)
        st, data = get_json(base, "/api/list?lite=1&limit=5000")
        check("list lite=1(整批,無 bits_b64)", st == 200 and len(data["rows"]) == 200 and
              all("bits_b64" not in r for r in data["rows"]))
        no_resp = next((r["id"] for r in data["rows"] if not r["has_resp"]), None)
        if no_resp:
            st, r = get_json(base, f"/api/resp?ids={no_resp}")
            check(f"無曲線 id({no_resp}) → null", st == 200 and r.get(no_resp) is None)
        else:
            check("fixture 應有無曲線樣本", False, "has_resp 全 True?")
        no_rad = next((r["id"] for r in data["rows"] if not r["has_rad"]), None)
        if no_rad:
            st, r = get_json(base, f"/api/radc?ids={no_rad}")
            check(f"無 rad id({no_rad}) → null", st == 200 and r.get(no_rad) is None)
        st, data = get_json(base, "/api/list?sort=sel&dir=asc&limit=200")
        sels = [r["sel"] for r in data["rows"] if r["sel"] is not None]
        check("sort=sel(※擴充)", st == 200 and sels == sorted(sels))
        st, data = get_json(base, "/api/list?sort=db100_wm&dir=desc&limit=200")
        dbs = [r["db100_wm"] for r in data["rows"] if r["db100_wm"] is not None]
        check("sort=db100_wm(可製造欄)", st == 200 and dbs == sorted(dbs, reverse=True))

        # ---- v2 stats 曲線覆蓋 ----
        st, stats = get_json(base, "/api/stats")
        check("stats resp_count/rad_count", stats is not None and
              isinstance(stats.get("resp_count"), int) and stats["resp_count"] > 0 and
              isinstance(stats.get("rad_count"), int) and stats["rad_count"] > 0)

        # ---- 效能粗檢(fixture 200 筆只是冒煙;36k 見 server 註記) ----
        t0 = time.time()
        get_json(base, "/api/hamming?id=fx0000&maxd=625&limit=50")
        dt = (time.time() - t0) * 1000
        check(f"hamming 冒煙耗時 {dt:.0f}ms < 300ms", dt < 300)

    except Exception:
        import traceback
        traceback.print_exc()
        FAILS.append("未預期例外(見上方 traceback)")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        out = proc.stdout.read() if proc.stdout else ""
        if FAILS and out:
            print("---- server 輸出(除錯用) ----")
            print(out[-3000:])

    try:
        missing_curves_test()
    except Exception:
        import traceback
        traceback.print_exc()
        FAILS.append("缺曲線檔測試未預期例外(見上方 traceback)")

    print(f"\n{'全部通過' if not FAILS else '失敗 ' + str(len(FAILS)) + ' 項:' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
