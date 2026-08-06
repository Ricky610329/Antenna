"""Pattern Browser API 冒煙測試:以 --fixture 啟 server 子行程 → 打每個 API → 驗 schema。

用法(任一目錄): python -X utf8 application/pattern_browser/selftest.py
退出碼: 0=全過, 1=有失敗。tests/ 正式測試不動,本檔只服務瀏覽器工具自己。
"""
import base64
import http.client
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
SERVER = PKG_DIR / "server.py"
META_KEYS = {"id", "folder", "store", "wm", "rad", "lo", "sel", "total", "n4", "n8",
             "largest8_frac", "ndiag", "has_db100", "has_sl100", "db100_wm", "sl100_wm", "kind"}

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

    print(f"\n{'全部通過' if not FAILS else '失敗 ' + str(len(FAILS)) + ' 項:' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
