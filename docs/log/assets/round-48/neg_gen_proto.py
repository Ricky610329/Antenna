# -*- coding: utf-8 -*-
"""負片生成器雛形——16 樣張 dry-run(零上機,純看質感)。
文法:底板(整版/矩形/切角)×五洞型(閉縫/開口縫/洞/梳/環縫)×約束(壁厚2/4連通含feed/feed保護半徑)。
"""
import sys
import numpy as np
from scipy.ndimage import label, binary_dilation

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
N = 25
FEED = (24, 12)
CHEB = np.ones((3, 3), bool)          # Chebyshev 膨脹核
WALL = 2                              # 洞-洞 / 洞-板緣 最小壁厚

def base_sheet(kind, rng):
    m = np.zeros((N, N), bool)
    if kind == "full":
        m[:] = True
    elif kind == "rect":
        h = int(rng.integers(19, 25)); w = int(rng.integers(17, 24))
        c0 = int(np.clip(12 - w // 2 + rng.integers(-2, 3), 0, N - w))
        m[N - h:, c0:c0 + w] = True
    elif kind == "cut":
        h = int(rng.integers(21, 25)); w = int(rng.integers(19, 24))
        c0 = int(np.clip(12 - w // 2, 0, N - w))
        m[N - h:, c0:c0 + w] = True
        k = int(rng.integers(4, 8))
        for i in range(k):            # 上緣兩角斜切
            m[N - h + i, c0:c0 + (k - i)] = False
            m[N - h + i, c0 + w - (k - i):c0 + w] = False
    m[FEED] = True
    return m

def feed_zone(rp):
    z = np.zeros((N, N), bool); z[FEED] = True
    if rp > 0:
        z = binary_dilation(z, CHEB, iterations=rp)
    return z

class Carver:
    def __init__(self, sheet, rp, sym):
        self.sheet = sheet.copy()      # 底板(不變,量壁厚用)
        self.metal = sheet.copy()      # 現行金屬
        self.voids = np.zeros((N, N), bool)
        self.islands = np.zeros((N, N), bool)   # 環縫浮島(連通豁免區)
        self.guard = feed_zone(rp)
        self.sym = sym
        self.counts = {}

    def _ok(self, v, closed, new_island=None):
        if not self.metal[v].all():
            return False                                    # 必須挖在現有金屬上
        if (v & self.guard).any() or (v & self.islands).any():
            return False
        near_void = binary_dilation(self.voids, CHEB, iterations=WALL)
        if (v & near_void).any():
            return False                                    # 洞-洞壁厚
        if closed:
            near_out = binary_dilation(~self.sheet, CHEB, iterations=WALL)
            if (v & near_out).any():
                return False                                # 閉洞離板緣壁厚
        m2 = self.metal & ~v
        if new_island is not None:
            m2 = m2 | new_island
        lab, _n = label(m2, structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
        fid = lab[FEED]
        if fid == 0:
            return False
        allowed = (lab == fid) | self.islands | (new_island if new_island is not None else False)
        if (m2 & ~allowed).any():
            return False                                    # 不得產生計畫外碎片
        return True

    def _apply(self, v, kind, new_island=None):
        self.metal &= ~v
        self.voids |= v
        if new_island is not None:
            self.metal |= new_island
            self.islands |= new_island
        self.counts[kind] = self.counts.get(kind, 0) + 1

    def _mirror(self, v):
        return v[:, ::-1]

    def try_void(self, v, kind, closed, new_island=None):
        if self.sym:
            vm = v | self._mirror(v)
            im = None if new_island is None else (new_island | self._mirror(new_island))
            if self._ok(vm, closed, im):
                self._apply(vm, kind, im); return True
            return False
        if self._ok(v, closed, new_island):
            self._apply(v, kind, new_island); return True
        return False

    # --- 五洞型 ---
    def slot(self, rng):
        L = int(rng.integers(3, 14)); W = int(rng.integers(1, 3))
        horiz = bool(rng.integers(0, 2))
        h, w = (W, L) if horiz else (L, W)
        r = int(rng.integers(0, N - h)); c = int(rng.integers(0, N - w))
        v = np.zeros((N, N), bool); v[r:r + h, c:c + w] = True
        return self.try_void(v, "縫", True)

    def hole(self, rng):
        h = int(rng.integers(2, 6)); w = int(rng.integers(2, 6))
        r = int(rng.integers(0, N - h)); c = int(rng.integers(0, N - w))
        v = np.zeros((N, N), bool); v[r:r + h, c:c + w] = True
        return self.try_void(v, "洞", True)

    def notch(self, rng):
        rows, cols = np.where(self.sheet)
        top, left, right = rows.min(), cols.min(), cols.max()
        edge = ["top", "left", "right"][int(rng.integers(0, 3))]
        d = int(rng.integers(3, 8)); W = int(rng.integers(1, 3))
        v = np.zeros((N, N), bool)
        if edge == "top":
            c = int(rng.integers(left, right - W + 1)); v[top:top + d, c:c + W] = True
        elif edge == "left":
            r = int(rng.integers(top, N - 4 - W)); v[r:r + W, left:left + d] = True
        else:
            r = int(rng.integers(top, N - 4 - W)); v[r:r + W, right - d + 1:right + 1] = True
        return self.try_void(v, "開口縫", False)

    def comb(self, rng):
        rows, cols = np.where(self.sheet)
        top, left, right = rows.min(), cols.min(), cols.max()
        d = int(rng.integers(3, 6)); W = int(rng.integers(1, 3)); period = W + WALL + int(rng.integers(0, 2))
        c = left + int(rng.integers(0, period)); made = 0
        while c + W <= right + 1:
            v = np.zeros((N, N), bool); v[top:top + d, c:c + W] = True
            if self.try_void(v, "梳齒", False):
                made += 1
            c += period
        return made > 0

    def ring(self, rng):
        s = int(rng.integers(3, 6))                       # 浮島邊長
        foot = s + 2                                       # 環寬 1
        r = int(rng.integers(0, N - foot)); c = int(rng.integers(0, N - foot))
        v = np.zeros((N, N), bool); v[r:r + foot, c:c + foot] = True
        isl = np.zeros((N, N), bool); isl[r + 1:r + 1 + s, c + 1:c + 1 + s] = True
        v &= ~isl
        return self.try_void(v, "環縫", True, new_island=isl)

OPS = {"縫": "slot", "洞": "hole", "開口縫": "notch", "梳": "comb", "環縫": "ring"}

def gen(seed, base, plan, rp=1, sym=False, tries=40):
    rng = np.random.default_rng(seed)
    cv = Carver(base_sheet(base, rng), rp, sym)
    for kind, n in plan:
        for _ in range(n):
            for _t in range(tries):
                if getattr(cv, OPS[kind])(rng):
                    break
    return cv

SPECS = [  # (base, plan, rp, sym, 標籤)
    ("full", [("縫", 3)], 1, False, "整版+閉縫×3"),
    ("full", [("縫", 6)], 1, False, "整版+閉縫×6"),
    ("full", [("縫", 9)], 0, False, "整版+閉縫×9 rp0"),
    ("rect", [("縫", 5)], 2, False, "矩形板+閉縫×5 rp2"),
    ("rect", [("開口縫", 3)], 1, False, "矩形板+開口縫×3"),
    ("rect", [("開口縫", 5)], 1, False, "矩形板+開口縫×5"),
    ("rect", [("梳", 1)], 1, False, "矩形板+上緣梳"),
    ("cut",  [("開口縫", 2), ("縫", 3)], 1, False, "切角板+開縫2+閉縫3"),
    ("full", [("洞", 5)], 1, False, "整版+洞×5"),
    ("rect", [("洞", 7)], 1, False, "矩形板+洞×7"),
    ("full", [("環縫", 2), ("縫", 2)], 1, False, "整版+環縫2(浮島)+縫2"),
    ("rect", [("環縫", 1), ("洞", 3)], 1, False, "矩形板+環縫1+洞3"),
    ("full", [("縫", 3), ("洞", 3), ("開口縫", 2)], 1, False, "混語法"),
    ("full", [("縫", 4), ("洞", 2)], 1, True, "整版混語法·鏡射"),
    ("rect", [("梳", 1), ("縫", 3)], 1, True, "矩形梳+縫·鏡射"),
    ("full", [("縫", 8), ("洞", 5), ("開口縫", 3)], 1, False, "重挖(低密度端)"),
]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(4, 4, figsize=(14, 14.5))
for k, (base, plan, rp, sym, tag) in enumerate(SPECS):
    cv = gen(20260731 * 100 + k, base, plan, rp, sym)
    ax = axes[k // 4][k % 4]
    img = np.zeros((N, N, 3))
    img[cv.metal] = (0.15, 0.25, 0.5)          # 金屬=深藍
    img[cv.islands] = (0.85, 0.45, 0.1)        # 浮島=橘(也是金屬)
    img[~cv.metal & cv.sheet] = (1, 1, 1)      # 洞=白
    img[~cv.sheet & ~cv.metal] = (0.92, 0.92, 0.92)  # 板外=淺灰
    ax.imshow(img, interpolation="nearest")
    ax.scatter([FEED[1]], [FEED[0]], c="red", s=45, marker="s", zorder=5)
    dens = cv.metal.sum() / (N * N)
    cnt = "+".join(f"{v}{k2}" for k2, v in cv.counts.items()) or "無"
    ax.set_title(f"#{k+1} {tag}\n{cnt} | 密度 {dens:.2f}", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    print(f"#{k+1:2d} {tag:22s} 挖成 {cnt:24s} 密度(全格) {dens:.2f}")
fig.suptitle("負片生成器雛形樣張 ×16(dry-run;深藍=金屬板/白=洞/橘=環縫浮島/紅=feed;壁厚≥2·4連通含feed)", fontsize=13)
fig.tight_layout()
out = r"C:\Users\Ricky\AppData\Local\Temp\claude\C--Users-Ricky-Documents-GitHub-Antenna\514acb31-4aa0-43ec-a3b3-98cdf8e2a623\scratchpad\neg_gen_samples.png"
fig.savefig(out, dpi=120)
print("saved", out)
