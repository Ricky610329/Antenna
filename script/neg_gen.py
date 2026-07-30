# -*- coding: utf-8 -*-
"""負片(slot/aperture 體系)生成器——R50 型態體系軸的生成端(Ricky 2026-07-31 拍板規格)。

家族(全部決定性 seed):
  eng       v1 工程語彙:底板(整版/矩形/切角)×五洞型(閉縫/開口縫/洞/梳齒/環縫浮島),壁厚≥2
  grf_neg   GRF level-cut 負片(相關高斯場+低閾值=有機挖洞板,f 0.60-0.82 × σ 1.0-2.0)
  grf_inv   GRF 反向(高閾值=有機斑塊正片,f 0.28-0.52)
  grf_lab   GRF 迷宮帶(中閾值,percolation 臨界,f 0.53-0.58)
  bool_cut  Boolean 切片挖空(隨機橢球/旋轉箱被平面切,解析橢圓/旋轉矩形,尺寸譜天然)
  bool_keep Boolean 切片保留(反向版)
  sierp     Sierpinski carpet 工程探針(L1;L2=修正版 3×3 次洞——標準 1px L2 撞針孔規則)

硬約束(探索期,Ricky 2026-07-31):
  FEED_PAD  feed 下方 pad×pad 全金屬(承重區;預設 5,R50 之後「持續放寬」=調小此參數)
  後處理鏈  除塵(金屬件 <4px 移除)+ 針孔縫合(空件 ≤2px 填回)——可製造性生成端寫死

CLI:
  python -m script.neg_gen sheet --out tmp/neg_sheet.png [--seed S]   # 樣張(可重跑資產源)
  python -m script.neg_gen pool  --n 200 --seed S --out tmp/pool.npz  # 生成池(供 select-r50 選席)
接線:R50 開輪時 dedust.py select-r50 引用 gen_pool()+farthest_point();判準照 decisions
「型態體系軸」條(≥10 輪/三標免疫/KPI=SM 域冷啟動曲線)。
"""
import argparse
import sys

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter, label

N = 25
FEED = (24, 12)
FEED_PAD_DEFAULT = 5                     # 承重塊邊長(貼底置中);放寬=調小
S4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
CHEB = np.ones((3, 3), bool)
WALL = 2                                 # eng 臂:洞-洞/洞-板緣最小壁厚
YY, XX = np.mgrid[0:N, 0:N].astype(float)

ARMS = ("eng", "grf_neg", "grf_inv", "grf_lab", "bool_cut", "bool_keep", "sierp")


def feed_block(pad=FEED_PAD_DEFAULT):
    z = np.zeros((N, N), bool)
    c0 = FEED[1] - pad // 2
    z[N - pad:N, c0:c0 + pad] = True
    return z


def postprocess(metal, pad=FEED_PAD_DEFAULT):
    """承重塊回填 → 針孔縫合(≤2px) → 除塵(<4px,含承重塊的件豁免)。"""
    m = metal | feed_block(pad)
    vl, vn = label(~m, structure=S4)
    for i in range(1, vn + 1):
        if (vl == i).sum() <= 2:
            m[vl == i] = True
    lab_, n = label(m, structure=S4)
    fid = lab_[FEED]
    for i in range(1, n + 1):
        if i != fid and (lab_ == i).sum() < 4:
            m[lab_ == i] = False
    return m


# ---------- GRF level-cut ----------
def _grf(rng, f_metal, sigma, sym=False):
    fld = gaussian_filter(rng.normal(size=(N, N)), sigma)
    if sym:
        fld = (fld + fld[:, ::-1]) / 2
    return fld >= np.quantile(fld, 1 - f_metal)


def gen_grf(rng, band):
    lo, hi = band
    f = rng.uniform(lo, hi)
    sigma = rng.uniform(1.0, 2.0)
    sym = rng.random() < 0.3             # 鏡射:非對稱 = 3:7
    return _grf(rng, f, sigma, sym), {"f": round(f, 2), "sigma": round(sigma, 2), "sym": sym}


# ---------- Boolean 切片 ----------
def _ellipse(cx, cy, a, b, th):
    dx, dy = XX - cx, YY - cy
    u = dx * np.cos(th) + dy * np.sin(th)
    v = -dx * np.sin(th) + dy * np.cos(th)
    return (u / max(a, .5)) ** 2 + (v / max(b, .5)) ** 2 <= 1.0


def _rect(cx, cy, w, h, th):
    dx, dy = XX - cx, YY - cy
    u = dx * np.cos(th) + dy * np.sin(th)
    v = -dx * np.sin(th) + dy * np.cos(th)
    return (np.abs(u) <= w / 2) & (np.abs(v) <= h / 2)


def _grains(rng, n_max):
    """3D 形體隨機切片流:切深 u → sqrt(1-u²) 尺寸譜(含小碎洞)。"""
    for _ in range(n_max):
        cx, cy = rng.uniform(-2, N + 2), rng.uniform(-2, N + 2)
        th = rng.uniform(0, np.pi)
        s = np.sqrt(1 - rng.uniform(0, 1) ** 2)
        if rng.random() < 0.7:
            yield _ellipse(cx, cy, rng.uniform(2, 7) * s, rng.uniform(1.5, 5) * s, th)
        else:
            yield _rect(cx, cy, rng.uniform(3, 10) * s, rng.uniform(2, 6) * s, th)


def gen_bool(rng, keep):
    f = rng.uniform(0.35, 0.50) if keep else rng.uniform(0.55, 0.75)
    m = np.zeros((N, N), bool) if keep else np.ones((N, N), bool)
    for g in _grains(rng, 60):
        if (m.mean() >= f) if keep else (m.mean() <= f):
            break
        m = (m | g) if keep else (m & ~g)
    return m, {"f": round(f, 2), "mode": "keep" if keep else "cut"}


# ---------- Sierpinski 探針 ----------
def gen_sierp(rng):
    m = np.ones((N, N), bool)
    m[10:15, 10:15] = False
    lvl2 = rng.random() < 0.5
    if lvl2:                              # 修正版 L2(標準 1px 洞會被針孔規則吃掉)
        for bi in (0, 2, 4):
            for bj in (0, 2, 4):
                if (bi, bj) == (2, 2):
                    continue
                m[bi * 5 + 1:bi * 5 + 4, bj * 5 + 1:bj * 5 + 4] = False
    return m, {"lvl2": lvl2}


# ---------- eng(v1 工程語彙) ----------
class _Carver:
    def __init__(self, sheet, guard, sym):
        self.sheet = sheet.copy()
        self.metal = sheet.copy()
        self.voids = np.zeros((N, N), bool)
        self.islands = np.zeros((N, N), bool)
        self.guard = guard
        self.sym = sym

    def _ok(self, v, closed, isl=None):
        if not self.metal[v].all() or (v & self.guard).any() or (v & self.islands).any():
            return False
        if (v & binary_dilation(self.voids, CHEB, iterations=WALL)).any():
            return False
        if closed and (v & binary_dilation(~self.sheet, CHEB, iterations=WALL)).any():
            return False
        m2 = self.metal & ~v
        if isl is not None:
            m2 |= isl
        lab_, _ = label(m2, structure=S4)
        fid = lab_[FEED]
        if fid == 0:
            return False
        allowed = (lab_ == fid) | self.islands | (isl if isl is not None else False)
        return not (m2 & ~allowed).any()

    def carve(self, v, closed, isl=None):
        if self.sym:
            v = v | v[:, ::-1]
            isl = None if isl is None else (isl | isl[:, ::-1])
        if not self._ok(v, closed, isl):
            return False
        self.metal &= ~v
        self.voids |= v
        if isl is not None:
            self.metal |= isl
            self.islands |= isl
        return True


def gen_eng(rng, guard):
    kind = ("full", "rect", "cut")[int(rng.integers(0, 3))]
    m = np.zeros((N, N), bool)
    if kind == "full":
        m[:] = True
    else:
        h = int(rng.integers(19, 25)); w = int(rng.integers(17, 24))
        c0 = int(np.clip(12 - w // 2 + rng.integers(-2, 3), 0, N - w))
        m[N - h:, c0:c0 + w] = True
        if kind == "cut":
            k = int(rng.integers(4, 8))
            for i in range(k):
                m[N - h + i, c0:c0 + (k - i)] = False
                m[N - h + i, c0 + w - (k - i):c0 + w] = False
    cv = _Carver(m, guard, rng.random() < 0.3)
    n_target = int(rng.integers(4, 12))
    made = 0
    for _ in range(n_target * 8):
        if made >= n_target:
            break
        op = rng.random()
        v = np.zeros((N, N), bool)
        if op < 0.45:                                    # 閉縫
            L, W = int(rng.integers(3, 14)), int(rng.integers(1, 3))
            h2, w2 = (W, L) if rng.random() < 0.5 else (L, W)
            r, c = int(rng.integers(0, N - h2)), int(rng.integers(0, N - w2))
            v[r:r + h2, c:c + w2] = True
            made += cv.carve(v, True)
        elif op < 0.7:                                   # 開口縫(上/左/右緣)
            rows, cols = np.where(cv.sheet)
            top, left, right = rows.min(), cols.min(), cols.max()
            d, W = int(rng.integers(3, 8)), int(rng.integers(1, 3))
            e = int(rng.integers(0, 3))
            if e == 0:
                c = int(rng.integers(left, max(left + 1, right - W + 1))); v[top:top + d, c:c + W] = True
            elif e == 1:
                r = int(rng.integers(top, N - 5)); v[r:r + W, left:left + d] = True
            else:
                r = int(rng.integers(top, N - 5)); v[r:r + W, right - d + 1:right + 1] = True
            made += cv.carve(v, False)
        elif op < 0.9:                                   # 洞
            h2, w2 = int(rng.integers(2, 6)), int(rng.integers(2, 6))
            r, c = int(rng.integers(0, N - h2)), int(rng.integers(0, N - w2))
            v[r:r + h2, c:c + w2] = True
            made += cv.carve(v, True)
        else:                                            # 環縫浮島
            s = int(rng.integers(3, 6)); foot = s + 2
            r, c = int(rng.integers(0, N - foot)), int(rng.integers(0, N - foot))
            v[r:r + foot, c:c + foot] = True
            isl = np.zeros((N, N), bool); isl[r + 1:r + 1 + s, c + 1:c + 1 + s] = True
            made += cv.carve(v & ~isl, True, isl)
    return cv.metal, {"base": kind, "voids": made}


def gen_one(arm, rng, pad=FEED_PAD_DEFAULT):
    if arm == "eng":
        raw, meta = gen_eng(rng, feed_block(pad))
    elif arm == "grf_neg":
        raw, meta = gen_grf(rng, (0.60, 0.82))
    elif arm == "grf_inv":
        raw, meta = gen_grf(rng, (0.28, 0.52))
    elif arm == "grf_lab":
        raw, meta = gen_grf(rng, (0.53, 0.58))
    elif arm == "bool_cut":
        raw, meta = gen_bool(rng, keep=False)
    elif arm == "bool_keep":
        raw, meta = gen_bool(rng, keep=True)
    elif arm == "sierp":
        raw, meta = gen_sierp(rng)
    else:
        raise ValueError(f"unknown arm {arm}")
    m = postprocess(raw, pad)
    meta.update(arm=arm, f_final=round(float(m.mean()), 3))
    return m, meta


def gen_pool(seed, n, arms=ARMS, pad=FEED_PAD_DEFAULT):
    """決定性生成池:輪抽 arms、批內 pattern 去重。回 [(pattern bool(25,25), meta), …]。"""
    out, seen = [], set()
    k = 0
    while len(out) < n and k < n * 20:
        arm = arms[k % len(arms)]
        m, meta = gen_one(arm, np.random.default_rng(seed * 1_000_003 + k), pad)
        k += 1
        key = m.tobytes()
        if key in seen:
            continue
        seen.add(key)
        out.append((m, meta))
    return out


def farthest_point(pool, k, seed=0):
    """覆蓋選席:對池內 pattern 做 farthest-point(Hamming),回索引列表。"""
    X = np.stack([p.reshape(-1) for p, _ in pool]).astype(np.int16)
    rng = np.random.default_rng(seed)
    picked = [int(rng.integers(0, len(pool)))]
    d = np.abs(X - X[picked[0]]).sum(axis=1)
    while len(picked) < min(k, len(pool)):
        i = int(d.argmax())
        picked.append(i)
        d = np.minimum(d, np.abs(X - X[i]).sum(axis=1))
    return picked


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sh = sub.add_parser("sheet"); sh.add_argument("--out", required=True); sh.add_argument("--seed", type=int, default=20260731)
    pl = sub.add_parser("pool"); pl.add_argument("--n", type=int, default=200); pl.add_argument("--seed", type=int, required=True)
    pl.add_argument("--out", required=True); pl.add_argument("--pad", type=int, default=FEED_PAD_DEFAULT)
    a = ap.parse_args()
    if a.cmd == "pool":
        pool = gen_pool(a.seed, a.n, pad=a.pad)
        np.savez_compressed(a.out, patterns=np.stack([p for p, _ in pool]),
                            meta=np.array([str(m) for _, m in pool]))
        print(f"pool {len(pool)} 筆 → {a.out}(seed {a.seed}, pad {a.pad})")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
    pool = gen_pool(a.seed, 16)
    fig, axes = plt.subplots(4, 4, figsize=(13, 13.5))
    for i, (m, meta) in enumerate(pool):
        ax = axes[i // 4][i % 4]
        img = np.ones((N, N, 3)); img[m] = (0.15, 0.25, 0.5)
        ax.imshow(img, interpolation="nearest")
        ax.scatter([FEED[1]], [FEED[0]], c="red", s=40, marker="s")
        ax.set_title(f"{meta['arm']} f={meta['f_final']}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"neg_gen 樣張(seed {a.seed};feed 承重塊 {FEED_PAD_DEFAULT}×{FEED_PAD_DEFAULT})")
    fig.tight_layout(); fig.savefig(a.out, dpi=120)
    print("saved", a.out)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
