"""Pattern Browser — server(stdlib http.server + numpy,零新依賴)。

規格契約 = 同目錄 SPEC.md。啟動(repo 根、ant env):
    python -m application.pattern_browser.server --port 8321
    python -m application.pattern_browser.server --fixture   # 無 data/ 時的開發模式(200 筆假資料)

資料 = data/patterns.npz(packed uint8 [N,79] + ids 同序) + data/meta.json(list[N])——由 build_index.py 產生。
v2 曲線資料(可缺,缺=has_resp/has_rad 全 False、曲線 API 回 null):
  data/resp.npz(resp float16 [N,2,17]+has_resp)、data/rad.npz(theta[181]+phi0/phi90 [N,181]+has_rad)、
  data/variant_resp.json({變體id:{s11,gain,phi0|null,phi90|null}})——皆與 patterns.npz 的 ids 同序。
API 全 GET/JSON;對契約的擴充(標 ※,皆為加欄不改形):
  /api/list      ※每列多回 bits_b64(79 bytes packed base64,前端畫縮圖免逐筆抓 pattern);
                 ※lite=1 → 不含 bits_b64 且 limit 上限放寬到 50000(散點/帕累托整批取);
                 ※sort 多收 sel、db100_wm(製造視角排行榜用)
  /api/pattern   ※多回 diag_sites=[[cx,cy,w],...](HFSS mm 座標,45° 菱形站點)與 bits_b64
  /api/hamming   ※每列多回 bits_b64
  /api/stats     ※多回 wm_hist(0.5dB 分箱)、wm_nonnull、resp_count、rad_count
  /api/resp      v2:ids=a,b,c → {id:{s11:[17],gain:[17]}|null};※未知/無曲線 id 回 null 不 404(批次語意)
  /api/radc      v2:ids=a,b,c → {id:{theta:[181],phi0:[181],phi90:[181]}|null};同上 null 語意
  /api/targets   v2:規格常數單一來源;※多回 freqs(17 點 24–32GHz,前端頻率軸)

渲染方向鐵則:第一索引 i 朝下、第二索引 j 朝右=饋線邊在圖下緣(pixel (i,j)→rect(x=j*s,y=i*s))。
diag_sites 的 cx 沿 HFSS X 軸,與圖面上下相反;前端轉圖座標 x=cy/0.2*s, y=(5-cx)/0.2*s。
"""
import argparse
import base64
import json
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import numpy as np

PKG_DIR = Path(__file__).resolve().parent
STATIC_DIR = PKG_DIR / "static"
DEFAULT_DATA_DIR = PKG_DIR / "data"
GRID = 25
PIXEL_MM = 0.2
DIAG_W_MM = 0.10
POP = np.array([bin(v).count("1") for v in range(256)], dtype=np.uint16)  # byte popcount 查表

META_KEYS = ("id", "folder", "store", "wm", "rad", "lo", "sel", "total", "n4", "n8",
             "largest8_frac", "ndiag", "has_db100", "has_sl100", "db100_wm", "sl100_wm", "kind")
SORT_COLS = ("wm", "lo", "rad", "ndiag", "n8", "total", "sel", "db100_wm")  # 後兩者=※擴充
NFREQ, NTHETA = 17, 181
FREQS = [round(24.0 + 0.5 * k, 1) for k in range(NFREQ)]        # 24–32GHz 17 點(契約)
THETA_DEFAULT = np.linspace(-90.0, 90.0, NTHETA)
TARGETS = {"band": [26.5, 29.5], "s11_max": -10, "gain_min": 4, "wm_buffer": 0.15,
           "rad_window": 45, "rad_floor": 3, "freqs": FREQS}     # freqs=※擴充
CTYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
          ".css": "text/css; charset=utf-8", ".json": "application/json; charset=utf-8",
          ".svg": "image/svg+xml", ".png": "image/png", ".ico": "image/x-icon"}


class ApiError(Exception):
    def __init__(self, status, msg):
        super().__init__(msg)
        self.status = status


def diag_sites_hfss(mat_display):
    """display 方向(i 朝下)的 25x25 bool → 45° 菱形站點 [[cx,cy,w],...](HFSS mm 座標)。

    條件=對角同開∧兩正交位皆空(同 antenna/patch/patch_simulator/single_port.py 的
    diag_bridge_sites;不 import 是避免把 torch/loguru 拖進零依賴 server)。
    w=0.10/p=0.2 時碰撞縮橋規則永不觸發(相鄰站淨距 0.2-0.2/√2≈0.0586≥0.05),瀏覽用直接略過。
    HFSS X 沿第一索引軸且與圖面上下相反 → 先 flipud 到 HFSS 方向再照原式 cx=(a+1)p;
    前端轉回圖座標 x=cy/0.2*s, y=(5-cx)/0.2*s 會剛好落在對角接點角上。"""
    m = np.asarray(mat_display, dtype=bool)[::-1, :]
    p = PIXEL_MM
    sites = []
    for a in range(GRID - 1):
        for b in range(GRID):
            if b < GRID - 1 and m[a, b] and m[a + 1, b + 1] and not m[a + 1, b] and not m[a, b + 1]:
                sites.append([round((a + 1) * p, 6), round((b + 1) * p, 6), DIAG_W_MM])
            if b > 0 and m[a, b] and m[a + 1, b - 1] and not m[a + 1, b] and not m[a, b - 1]:
                sites.append([round((a + 1) * p, 6), round(b * p, 6), DIAG_W_MM])
    return sites


def _curve_list(arr):
    """曲線陣列 → JSON 安全 list(NaN/inf→null,小數 3 位;收 list 或 ndarray)。"""
    out = []
    for v in np.asarray(arr, dtype=float):
        out.append(round(float(v), 3) if np.isfinite(v) else None)
    return out


class Store:
    """全部載進記憶體:packed [N,79] uint8 + meta list[dict];數值欄轉 numpy 供向量化篩選/排序。

    curves(v2,可 None=無曲線資料):{resp[N,2,17], has_resp[N], theta[181], phi0/phi90[N,181],
    has_rad[N], variant_resp{變體id:{s11,gain,phi0|null,phi90|null}}};has_resp/has_rad 回寫進 meta。"""

    def __init__(self, packed, ids, metas, curves=None):
        self.packed = np.ascontiguousarray(packed, dtype=np.uint8)
        self.ids = [str(x) for x in ids]
        self.metas = metas
        self.n = len(self.ids)
        self.id2idx = {pid: i for i, pid in enumerate(self.ids)}
        self.ids_l = np.array([s.lower() for s in self.ids])
        self.col = {c: np.array([m[c] if m.get(c) is not None else np.nan for m in metas],
                                dtype=float) for c in SORT_COLS}
        self.has_db100 = np.array([bool(m.get("has_db100")) for m in metas])
        self.has_sl100 = np.array([bool(m.get("has_sl100")) for m in metas])
        cv = curves or {}
        self.resp = cv.get("resp")                      # float [N,2,17] 或 None
        self.phi0 = cv.get("phi0")                      # float [N,181] 或 None
        self.phi90 = cv.get("phi90")
        self.theta = cv.get("theta")
        self.theta_list = _curve_list(self.theta if self.theta is not None else THETA_DEFAULT)
        self.variant_resp = cv.get("variant_resp") or {}
        hr = cv.get("has_resp")
        self.has_resp = np.asarray(hr, dtype=bool) if hr is not None else np.zeros(self.n, bool)
        hd = cv.get("has_rad")
        self.has_rad = np.asarray(hd, dtype=bool) if hd is not None else np.zeros(self.n, bool)
        for i, m in enumerate(self.metas):              # server 為真相源,回寫 meta(契約:list 每列要有)
            m["has_resp"] = bool(self.has_resp[i])
            m["has_rad"] = bool(self.has_rad[i])
        self.stats = self._build_stats()

    def _build_stats(self):
        wm = self.col["wm"]
        wm_ok = wm[~np.isnan(wm)]

        def hist(colname):
            v = self.col[colname]
            v = v[~np.isnan(v)].astype(int)
            u, c = np.unique(v, return_counts=True)
            return [[int(a), int(b)] for a, b in zip(u, c)]

        stats = {
            "total": self.n,
            "qualified": int(((wm >= 0.15) & (self.col["rad"] >= 0)).sum()),  # 合格=wm≥0.15(buffer)∧rad≥0
            "wm_nonnull": int(wm_ok.size),      # ※擴充
            "ndiag_hist": hist("ndiag"),
            "n8_hist": hist("n8"),
            "db100_count": int(self.has_db100.sum()),
            "sl100_count": int(self.has_sl100.sum()),
            "resp_count": int(self.has_resp.sum()),   # ※擴充:曲線覆蓋
            "rad_count": int(self.has_rad.sum()),     # ※擴充
        }
        if wm_ok.size:  # ※擴充:wm 0.5dB 分箱(總攬長條圖)
            lo = float(np.floor(wm_ok.min() * 2) / 2)
            hi = float(np.ceil(wm_ok.max() * 2) / 2)
            edges = np.arange(lo, hi + 0.5 + 1e-9, 0.5)
            cnt, _ = np.histogram(wm_ok, bins=edges)
            stats["wm_hist"] = {"edges": [round(float(e), 3) for e in edges],
                                "counts": [int(x) for x in cnt]}
        else:
            stats["wm_hist"] = {"edges": [], "counts": []}
        return stats

    def bits(self, idx):
        return np.unpackbits(self.packed[idx])[: GRID * GRID]

    def b64(self, idx):
        return base64.b64encode(self.packed[idx].tobytes()).decode("ascii")

    def resolve(self, pid):
        """id → (idx, variant|None);變體 id(含 ~)解析到親本。"""
        parent, _, variant = pid.partition("~")
        idx = self.id2idx.get(parent)
        if idx is None:
            raise ApiError(404, f"未知 id: {pid}")
        return idx, (variant or None)

    # ---- v2 曲線(未知/無曲線 一律回 None,批次語意) ----

    def resp_of(self, pid):
        if "~" in pid:
            v = self.variant_resp.get(pid)
            if not v or v.get("s11") is None or v.get("gain") is None:
                return None
            return {"s11": _curve_list(v["s11"]), "gain": _curve_list(v["gain"])}
        idx = self.id2idx.get(pid)
        if idx is None or self.resp is None or not self.has_resp[idx]:
            return None
        return {"s11": _curve_list(self.resp[idx, 0]), "gain": _curve_list(self.resp[idx, 1])}

    def radc_of(self, pid):
        if "~" in pid:
            v = self.variant_resp.get(pid)
            if not v or (v.get("phi0") is None and v.get("phi90") is None):
                return None
            return {"theta": self.theta_list,
                    "phi0": _curve_list(v["phi0"]) if v.get("phi0") is not None else None,
                    "phi90": _curve_list(v["phi90"]) if v.get("phi90") is not None else None}
        idx = self.id2idx.get(pid)
        if idx is None or self.phi0 is None or not self.has_rad[idx]:
            return None
        return {"theta": self.theta_list,
                "phi0": _curve_list(self.phi0[idx]), "phi90": _curve_list(self.phi90[idx])}


# ---------------------------------------------------------------- API 實作

def _qnum(qs, name):
    v = qs.get(name, [None])[0]
    if v in (None, ""):
        return None
    try:
        return float(v)
    except ValueError:
        raise ApiError(400, f"參數 {name} 需為數字: {v}")


def _qint(qs, name, default, lo, hi):
    v = qs.get(name, [None])[0]
    if v in (None, ""):
        return default
    try:
        return max(lo, min(hi, int(v)))
    except ValueError:
        raise ApiError(400, f"參數 {name} 需為整數: {v}")


def api_list(st, qs):
    mask = np.ones(st.n, dtype=bool)
    if qs.get("f_qual", [""])[0] == "1":
        # 合格=專案正式口徑 wm≥0.15(buffer)∧rad≥0(NaN 比較恆 False → 無值自動排除)
        mask &= (st.col["wm"] >= 0.15) & (st.col["rad"] >= 0)
    for name, colname, ge in (("f_diag_min", "ndiag", True), ("f_diag_max", "ndiag", False),
                              ("f_n8_min", "n8", True), ("f_n8_max", "n8", False),
                              ("f_wm_min", "wm", True), ("f_lo_max", "lo", False)):
        v = _qnum(qs, name)
        if v is not None:
            mask &= (st.col[colname] >= v) if ge else (st.col[colname] <= v)
    if qs.get("f_has_db100", [""])[0] == "1":
        mask &= st.has_db100
    if qs.get("f_has_sl100", [""])[0] == "1":
        mask &= st.has_sl100
    q = qs.get("q", [""])[0].strip().lower()
    if q:
        mask &= np.char.find(st.ids_l, q) >= 0
    idx = np.flatnonzero(mask)

    sort = qs.get("sort", ["wm"])[0]
    if sort not in SORT_COLS:
        raise ApiError(400, f"sort 需為 {'/'.join(SORT_COLS)}: {sort}")
    asc = qs.get("dir", ["desc"])[0] == "asc"
    key = st.col[sort][idx]
    key = np.where(np.isnan(key), np.inf, key if asc else -key)  # 無值一律排最後
    idx = idx[np.argsort(key, kind="stable")]

    lite = qs.get("lite", [""])[0] == "1"  # ※擴充:散點/帕累托整批取,不帶 bits_b64
    offset = _qint(qs, "offset", 0, 0, 10 ** 9)
    limit = _qint(qs, "limit", 50, 1, 50000 if lite else 1000)
    rows = []
    for i in idx[offset:offset + limit]:
        row = dict(st.metas[i])
        if not lite:
            row["bits_b64"] = st.b64(i)  # ※契約擴充:前端縮圖用
        rows.append(row)
    return {"total": int(idx.size), "offset": offset, "limit": limit, "rows": rows}


def api_pattern(st, pid):
    if not pid:
        raise ApiError(400, "缺 id")
    idx, variant = st.resolve(pid)
    out = dict(st.metas[idx])
    if variant:  # 變體 id:bits 同親本,meta 標變體
        out["id"] = pid
        out["variant"] = variant
        out["variant_of"] = st.ids[idx]
    bits = st.bits(idx)
    out["bits"] = [int(b) for b in bits]
    out["diag_sites"] = diag_sites_hfss(bits.reshape(GRID, GRID))  # ※契約擴充
    out["bits_b64"] = st.b64(idx)  # ※契約擴充
    return out


def api_hamming(st, qs):
    pid = qs.get("id", [""])[0]
    if not pid:
        raise ApiError(400, "缺參數 id")
    idx, _ = st.resolve(pid)
    maxd = _qint(qs, "maxd", GRID * GRID, 0, GRID * GRID)
    limit = _qint(qs, "limit", 50, 1, 2000)
    # packed XOR + popcount 查表,36k 列 << 100ms;padding 7 bits 恆 0 不影響距離
    d = POP[np.bitwise_xor(st.packed, st.packed[idx])].sum(axis=1, dtype=np.int32)
    d[idx] = np.iinfo(np.int32).max  # 排除自身
    sel = np.flatnonzero(d <= maxd)
    sel = sel[np.argsort(d[sel], kind="stable")][:limit]
    rows = []
    for i in sel:
        row = dict(st.metas[i])
        row["d"] = int(d[i])
        row["bits_b64"] = st.b64(i)  # ※契約擴充
        rows.append(row)
    return rows  # 契約:距離排序的陣列


def api_compare(st, qs):
    raw = qs.get("ids", [""])[0]
    ids = [s for s in (t.strip() for t in raw.split(",")) if s]
    if not 2 <= len(ids) <= 4:
        raise ApiError(400, f"ids 需 2~4 個(收到 {len(ids)})")
    return [api_pattern(st, pid) for pid in ids]  # 契約:各 bits+meta 的陣列


def _ids_param(qs, cap=32):
    raw = qs.get("ids", [""])[0]
    ids = [s for s in (t.strip() for t in raw.split(",")) if s]
    if not ids:
        raise ApiError(400, "缺參數 ids")
    if len(ids) > cap:
        raise ApiError(400, f"ids 最多 {cap} 個(收到 {len(ids)})")
    return ids


def api_resp(st, qs):
    """v2:{id:{s11:[17],gain:[17]}|null};未知/無曲線 → null(批次語意,不 404)。"""
    return {pid: st.resp_of(pid) for pid in _ids_param(qs)}


def api_radc(st, qs):
    """v2:{id:{theta:[181],phi0:[181],phi90:[181]}|null};變體切面可個別 null。"""
    return {pid: st.radc_of(pid) for pid in _ids_param(qs)}


# ---------------------------------------------------------------- 資料載入

def load_data(data_dir):
    npz_p = data_dir / "patterns.npz"
    meta_p = data_dir / "meta.json"
    missing = [p for p in (npz_p, meta_p) if not p.is_file()]
    if missing:
        lines = "\n".join(f"  缺 {p}" for p in missing)
        sys.exit(f"[pattern_browser] 找不到資料檔:\n{lines}\n"
                 f"請先跑 build_index 產生 data/(python -m application.pattern_browser.build_index),\n"
                 f"或加 --fixture 用 200 筆假資料啟動開發模式。")
    with np.load(npz_p, allow_pickle=False) as z:
        if "packed" not in z or "ids" not in z:
            sys.exit(f"[pattern_browser] {npz_p} 缺 packed/ids 欄位,請重跑 build_index。")
        packed = np.asarray(z["packed"], dtype=np.uint8)
        npz_ids = [str(x) for x in z["ids"]]
    if packed.ndim != 2 or packed.shape[1] != 79:
        sys.exit(f"[pattern_browser] packed 形狀 {packed.shape} 不符契約 [N,79],請重跑 build_index。")
    try:
        metas = json.loads(meta_p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        sys.exit(f"[pattern_browser] {meta_p} 不是合法 JSON({e}),請重跑 build_index。")

    row_of = {pid: i for i, pid in enumerate(npz_ids)}
    keep, rows, dropped = [], [], 0
    for m in metas:  # 以 meta 為主表,按 id 對齊 npz(容忍兩檔順序/數量不一致)
        i = row_of.get(str(m.get("id")))
        if i is None:
            dropped += 1
            continue
        keep.append(m)
        rows.append(i)
    if dropped:
        print(f"[pattern_browser] 警告:meta.json 有 {dropped} 筆 id 不在 patterns.npz,已略過")
    if not keep:
        sys.exit("[pattern_browser] data/ 內容對不上(meta 與 npz 的 id 無交集),請重跑 build_index。")
    rows = np.array(rows)
    return packed[rows], [m["id"] for m in keep], keep, rows, len(npz_ids)


def load_curves(data_dir, rows, n_src):
    """v2 曲線檔容錯載入:resp.npz / rad.npz / variant_resp.json 任一缺/壞 → 當作沒有(不擋啟動)。

    rows=meta 對齊後的 npz 列索引(曲線檔與 patterns.npz 同序 → 用同一索引重排);n_src=npz 原始筆數。"""
    curves = {}

    def _npz(name, fields):
        p = data_dir / name
        if not p.is_file():
            return None
        try:
            with np.load(p, allow_pickle=False) as z:
                return {k: np.asarray(z[k]) for k in fields}
        except Exception as e:
            print(f"[pattern_browser] 警告:讀 {p} 失敗({e}),曲線當作缺")
            return None

    r = _npz("resp.npz", ("resp", "has_resp"))
    if r is not None:
        if r["resp"].shape == (n_src, 2, NFREQ) and r["has_resp"].shape == (n_src,):
            curves["resp"] = r["resp"].astype(np.float32)[rows]
            curves["has_resp"] = r["has_resp"].astype(bool)[rows]
        else:
            print(f"[pattern_browser] 警告:resp.npz 形狀 {r['resp'].shape} 對不上 N={n_src},曲線當作缺")
    d = _npz("rad.npz", ("theta", "phi0", "phi90", "has_rad"))
    if d is not None:
        ok = (d["theta"].shape == (NTHETA,) and d["phi0"].shape == (n_src, NTHETA)
              and d["phi90"].shape == (n_src, NTHETA) and d["has_rad"].shape == (n_src,))
        if ok:
            curves["theta"] = d["theta"].astype(np.float32)
            curves["phi0"] = d["phi0"].astype(np.float32)[rows]
            curves["phi90"] = d["phi90"].astype(np.float32)[rows]
            curves["has_rad"] = d["has_rad"].astype(bool)[rows]
        else:
            print(f"[pattern_browser] 警告:rad.npz 形狀對不上 N={n_src},曲線當作缺")
    vp = data_dir / "variant_resp.json"
    if vp.is_file():
        try:
            v = json.loads(vp.read_text(encoding="utf-8"))
            if isinstance(v, dict):
                curves["variant_resp"] = v
            else:
                print("[pattern_browser] 警告:variant_resp.json 不是 dict,當作缺")
        except (json.JSONDecodeError, OSError) as e:
            print(f"[pattern_browser] 警告:讀 {vp} 失敗({e}),當作缺")
    return curves


# ---------------------------------------------------------------- dev fixture

def _component_sizes(mat, diag):
    """flood fill 連通元件大小(fixture 專用;正式 meta 由 build_index 算)。"""
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diag:
        nbrs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    seen = np.zeros(mat.shape, dtype=bool)
    sizes = []
    for i0 in range(mat.shape[0]):
        for j0 in range(mat.shape[1]):
            if not mat[i0, j0] or seen[i0, j0]:
                continue
            stack = [(i0, j0)]
            seen[i0, j0] = True
            c = 0
            while stack:
                i, j = stack.pop()
                c += 1
                for di, dj in nbrs:
                    a, b = i + di, j + dj
                    if 0 <= a < mat.shape[0] and 0 <= b < mat.shape[1] and mat[a, b] and not seen[a, b]:
                        seen[a, b] = True
                        stack.append((a, b))
            sizes.append(c)
    return sizes


def make_fixture(n=200, seed=7):
    """決定性假資料:塊狀隨機 pattern + 全 meta 欄位;fx0001=fx0000 翻固定 3 bit(近鄰驗證用)。"""
    rng = np.random.default_rng(seed)
    stores = ["r55fixa", "r55fixb", "r56fixc", None]
    kinds = ["select", "hill", "graft", None]
    packed_rows, ids, metas = [], [], []
    mat0 = None
    for k in range(n):
        if k == 1 and mat0 is not None:
            mat = mat0.copy()
            for i, j in ((2, 3), (10, 17), (20, 5)):  # 固定翻 3 bit → hamming d=3
                mat[i, j] = not mat[i, j]
        else:
            base = rng.random((7, 7))
            img = np.kron(base, np.ones((4, 4)))[:GRID, :GRID] + 0.18 * rng.random((GRID, GRID))
            mat = img > np.quantile(img, float(rng.uniform(0.40, 0.55)))
        mat[GRID - 1, 10:15] = True  # 饋線邊示意(渲染鐵則:i 朝下=圖下緣)
        if k == 0:
            mat0 = mat.copy()
        pid = f"fx{k:04d}"
        total = int(mat.sum())
        s8 = _component_sizes(mat, diag=True)
        s4 = _component_sizes(mat, diag=False)
        wm = None if rng.random() < 0.15 else round(float(rng.normal(-1.0, 0.9)), 3)
        has_db = bool(rng.random() < 0.30)
        has_sl = bool(rng.random() < 0.20)

        def _abl(has):
            if not has:
                return None
            return round((wm if wm is not None else -1.5) - abs(float(rng.normal(0.3, 0.25))), 3)

        metas.append({
            "id": pid,
            "folder": f"fixture/{pid}",
            "store": stores[int(rng.integers(0, len(stores)))],
            "wm": wm,
            "rad": None if wm is None else round(float(rng.normal(3.0, 1.0)), 3),
            "lo": None if rng.random() < 0.10 else round(float(rng.normal(9.0, 2.0)), 3),
            "sel": round(float(rng.normal(-15.0, 4.0)), 3),
            "total": total,
            "n4": len(s4),
            "n8": len(s8),
            "largest8_frac": round(max(s8) / total, 4) if total else 0.0,
            "ndiag": len(diag_sites_hfss(mat)),
            "has_db100": has_db,
            "has_sl100": has_sl,
            "db100_wm": _abl(has_db),
            "sl100_wm": _abl(has_sl),
            "kind": kinds[int(rng.integers(0, len(kinds)))],
        })
        ids.append(pid)
        packed_rows.append(np.packbits(mat.astype(np.uint8).ravel()))
    return np.stack(packed_rows), ids, metas


def make_fixture_curves(metas, seed=11):
    """fixture 的 v2 假曲線(獨立 rng,不動 make_fixture 的亂數流=保住既有 hamming 測試)。

    fx0000 強制有 resp+rad+兩種變體曲線(selftest 錨點);其餘 has_resp≈85%、has_rad≈70%、
    變體曲線≈90%(有消融旗標才生)。曲線形狀=諧振凹陷 S11+緩斜 Gain+拋物線主波束 rad。"""
    rng = np.random.default_rng(seed)
    n = len(metas)
    freqs = np.array(FREQS)
    resp = np.full((n, 2, NFREQ), np.nan, dtype=np.float32)
    has_resp = np.zeros(n, dtype=bool)
    phi0 = np.full((n, NTHETA), np.nan, dtype=np.float32)
    phi90 = np.full((n, NTHETA), np.nan, dtype=np.float32)
    has_rad = np.zeros(n, dtype=bool)
    variant_resp = {}

    def s11_curve():
        f0, w = rng.uniform(25.5, 30.5), rng.uniform(0.6, 1.8)
        return (-1.5 - rng.uniform(5, 22) * np.exp(-((freqs - f0) / w) ** 2)
                + rng.normal(0, 0.15, NFREQ))

    def gain_curve():
        return (rng.uniform(2.5, 6.5) + rng.uniform(-0.1, 0.1) * (freqs - 28)
                + 0.25 * np.sin(freqs * rng.uniform(0.8, 1.4)))

    def rad_cut(g0):
        tilt = rng.normal(0, 6)
        return (g0 - rng.uniform(8, 18) * ((THETA_DEFAULT - tilt) / 90.0) ** 2
                + 0.8 * np.sin(THETA_DEFAULT / rng.uniform(9, 14)))

    for i, m in enumerate(metas):
        force = i == 0
        if rng.random() < 0.85 or force:
            has_resp[i] = True
            resp[i, 0], resp[i, 1] = s11_curve(), gain_curve()
        if rng.random() < 0.70 or force:
            has_rad[i] = True
            g0 = rng.uniform(3.0, 7.0)
            phi0[i], phi90[i] = rad_cut(g0), rad_cut(g0 - rng.uniform(0.0, 1.5))
        for tag, flag in (("db100", m["has_db100"]), ("sl100", m["has_sl100"])):
            if not (force or (flag and rng.random() < 0.90)):
                continue
            base_s11 = resp[i, 0] if has_resp[i] else s11_curve()
            base_g = resp[i, 1] if has_resp[i] else gain_curve()
            cut_ok = has_rad[i] and (force or rng.random() < 0.8)
            shift = rng.uniform(0.2, 1.0)
            variant_resp[f"{m['id']}~{tag}"] = {
                "s11": [round(float(v), 3) for v in base_s11 * rng.uniform(0.6, 0.9)],
                "gain": [round(float(v), 3) for v in base_g - rng.uniform(0.2, 0.8)],
                "phi0": [round(float(v), 3) for v in phi0[i] - shift] if cut_ok else None,
                "phi90": [round(float(v), 3) for v in phi90[i] - shift] if cut_ok else None,
            }
    return {"resp": resp, "has_resp": has_resp, "theta": THETA_DEFAULT.astype(np.float32),
            "phi0": phi0, "phi90": phi90, "has_rad": has_rad, "variant_resp": variant_resp}


# ---------------------------------------------------------------- HTTP handler

class Handler(BaseHTTPRequestHandler):
    store = None  # main() 注入 Store
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("· %s %s\n" % (self.command, self.path.split("?")[0]))

    def do_GET(self):
        u = urlsplit(self.path)
        path = unquote(u.path)
        try:
            if path.startswith("/api/"):
                self._api(path, parse_qs(u.query))
            else:
                self._static(path)
        except ApiError as e:
            self._json({"error": str(e)}, e.status)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception:
            traceback.print_exc()
            self._json({"error": "internal server error"}, 500)

    def _api(self, path, qs):
        st = self.store
        if path == "/api/list":
            self._json(api_list(st, qs))
        elif path == "/api/stats":
            self._json(st.stats)
        elif path == "/api/hamming":
            self._json(api_hamming(st, qs))
        elif path == "/api/compare":
            self._json(api_compare(st, qs))
        elif path == "/api/resp":
            self._json(api_resp(st, qs))
        elif path == "/api/radc":
            self._json(api_radc(st, qs))
        elif path == "/api/targets":
            self._json(TARGETS)
        elif path.startswith("/api/pattern/"):
            self._json(api_pattern(st, path[len("/api/pattern/"):]))
        else:
            raise ApiError(404, f"未知 API: {path}")

    def _static(self, path):
        if path in ("/", "/index.html"):
            rel = "index.html"
        elif path == "/favicon.ico":
            self._body(b"", "image/x-icon")
            return
        elif path.startswith("/static/"):
            rel = path[len("/static/"):]
        else:
            raise ApiError(404, f"未知路徑: {path}")
        base = STATIC_DIR.resolve()
        f = (base / rel).resolve()
        try:
            f.relative_to(base)
        except ValueError:
            raise ApiError(403, "禁止存取")
        if not f.is_file():
            raise ApiError(404, f"檔案不存在: {rel}")
        self._body(f.read_bytes(), CTYPES.get(f.suffix.lower(), "application/octet-stream"))

    def _json(self, obj, status=200):
        self._body(json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8", status)

    def _body(self, data, ctype, status=200):
        try:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass


def main(argv=None):
    ap = argparse.ArgumentParser(description="Pattern Browser server(規格契約=SPEC.md)")
    ap.add_argument("--port", type=int, default=8321)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR,
                    help="data/ 目錄(patterns.npz + meta.json)")
    ap.add_argument("--fixture", action="store_true",
                    help="不讀 data/,用 200 筆決定性假資料啟動(開發模式)")
    args = ap.parse_args(argv)
    if args.fixture:
        packed, ids, metas = make_fixture()
        curves = make_fixture_curves(metas)
        src = "fixture(200 筆假資料,開發模式)"
    else:
        packed, ids, metas, rows, n_src = load_data(args.data)
        curves = load_curves(args.data, rows, n_src)
        src = str(args.data)
    Handler.store = Store(packed, ids, metas, curves)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    st = Handler.store
    print(f"[pattern_browser] {st.n} 筆 pattern 已載入({src});"
          f"曲線覆蓋 resp {int(st.has_resp.sum())}/rad {int(st.has_rad.sum())}/"
          f"變體 {len(st.variant_resp)}")
    print(f"[pattern_browser] http://{args.host}:{args.port}/  (Ctrl+C 停止)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[pattern_browser] 已停止")


if __name__ == "__main__":
    main()
