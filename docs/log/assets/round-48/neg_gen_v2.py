# -*- coding: utf-8 -*-
"""負片生成器 v2 樣張——隨機幾何家族(GRF level-cut / Boolean 切片 / 工程探針)。
GRF:相關高斯場+閾值,掃閾值=負片↔迷宮↔反向(有機正片)連續體。
Boolean:隨機橢球/旋轉箱被平面切(解析:橢圓/旋轉矩形),挖空或保留。
約束後處理:feed 金屬+pad、除塵(<4px)、針孔縫合(≤2px 洞)。
"""
import sys
import numpy as np
from scipy.ndimage import label, gaussian_filter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
N = 25
FEED = (24, 12)
S4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
YY, XX = np.mgrid[0:N, 0:N].astype(float)

def postprocess(metal):
    """feed 金屬+pad、除塵 <4px、針孔縫合 ≤2px。回 (pattern, n_metal_comp, n_void_comp)。"""
    m = metal.copy()
    m[FEED] = True
    lab, n = label(m, structure=S4)
    if (lab == lab[FEED]).sum() < 4:                      # feed 孤立 → 3×3 pad 貼底
        m[22:25, 11:14] = True
    # 針孔縫合:洞(空)4-連通件 ≤2px 填回金屬
    vl, vn = label(~m, structure=S4)
    for i in range(1, vn + 1):
        if (vl == i).sum() <= 2:
            m[vl == i] = True
    # 除塵:金屬件 <4px 移除(feed 件豁免)
    lab, n = label(m, structure=S4)
    fid = lab[FEED]
    for i in range(1, n + 1):
        if i != fid and (lab == i).sum() < 4:
            m[lab == i] = False
    lab, nm = label(m, structure=S4)
    _, nv = label(~m, structure=S4)
    return m, nm, nv

# ---------- GRF level-cut ----------
def grf(seed, f_metal, sigma, sym=False):
    rng = np.random.default_rng(seed)
    fld = gaussian_filter(rng.normal(size=(N, N)), sigma)
    if sym:
        fld = (fld + fld[:, ::-1]) / 2
    q = np.quantile(fld, 1 - f_metal)
    return fld >= q

# ---------- Boolean 切片 ----------
def _ellipse_mask(cx, cy, a, b, th):
    dx, dy = XX - cx, YY - cy
    u = dx * np.cos(th) + dy * np.sin(th)
    v = -dx * np.sin(th) + dy * np.cos(th)
    return (u / max(a, .5)) ** 2 + (v / max(b, .5)) ** 2 <= 1.0

def _rect_mask(cx, cy, w, h, th):
    dx, dy = XX - cx, YY - cy
    u = dx * np.cos(th) + dy * np.sin(th)
    v = -dx * np.sin(th) + dy * np.cos(th)
    return (np.abs(u) <= w / 2) & (np.abs(v) <= h / 2)

def bool_grains(seed, n_max, mix=("ell",)):
    """隨機形體切片流:橢球切片=橢圓(軸長×sqrt(1-u^2) 天然尺寸譜),旋轉箱切片=旋轉矩形。"""
    rng = np.random.default_rng(seed)
    for _ in range(n_max):
        kind = mix[int(rng.integers(0, len(mix)))]
        cx, cy = rng.uniform(-2, N + 2), rng.uniform(-2, N + 2)
        th = rng.uniform(0, np.pi)
        scale = np.sqrt(1 - rng.uniform(0, 1) ** 2)        # 平面切深 → 尺寸譜(含小碎洞)
        if kind == "ell":
            A, B = rng.uniform(2, 7), rng.uniform(1.5, 5)
            yield _ellipse_mask(cx, cy, A * scale, B * scale, th)
        else:
            W, H = rng.uniform(3, 10), rng.uniform(2, 6)
            yield _rect_mask(cx, cy, W * scale, H * scale, th)

def bool_carve(seed, f_target, mix=("ell",)):
    m = np.ones((N, N), bool)
    for g in bool_grains(seed, 60, mix):
        if m.mean() <= f_target:
            break
        m &= ~g
    return m

def bool_keep(seed, f_target, mix=("ell",)):
    m = np.zeros((N, N), bool)
    for g in bool_grains(seed, 60, mix):
        if m.mean() >= f_target:
            break
        m |= g
    return m

# ---------- 工程探針 ----------
def sierpinski(level2=False):
    m = np.ones((N, N), bool)
    m[10:15, 10:15] = False                                # L1:挖中央 5×5
    if level2:
        # 標準 L2(各塊中心 1px)會被針孔縫合規則吃掉=撞可製造性底線
        # → 修正版 carpet:周圍八塊挖 3×3 次洞(塊距 5、洞 3 → 壁厚 2 恰好合規)
        for bi in range(5):
            for bj in range(5):
                if (bi, bj) == (2, 2) or bi == 1 or bj == 1 or bi == 3 or bj == 3:
                    continue
                m[bi * 5 + 1:bi * 5 + 4, bj * 5 + 1:bj * 5 + 4] = False
    return m

def hybrid(seed):
    m = grf(seed, 0.74, 1.5)
    m[4:6, 5:17] = False                                   # 工程縫 ×2 疊在有機板上
    m[12:19, 18:20] = False
    return m

SPECS = [
    ("GRF負",  lambda s: grf(s, 0.80, 1.0),  "f.80 σ1.0"),
    ("GRF負",  lambda s: grf(s, 0.72, 1.5),  "f.72 σ1.5"),
    ("GRF負",  lambda s: grf(s, 0.65, 2.0),  "f.65 σ2.0"),
    ("GRF負",  lambda s: grf(s, 0.60, 1.2),  "f.60 σ1.2"),
    ("GRF反向", lambda s: grf(s, 0.30, 1.0),  "f.30 σ1.0"),
    ("GRF反向", lambda s: grf(s, 0.40, 1.5),  "f.40 σ1.5"),
    ("GRF反向", lambda s: grf(s, 0.50, 2.0),  "f.50 σ2.0"),
    ("GRF迷宮", lambda s: grf(s, 0.55, 1.2),  "f.55 σ1.2"),
    ("Bool切負", lambda s: bool_carve(s, 0.72, ("ell",)),        "橢圓 f.72"),
    ("Bool切負", lambda s: bool_carve(s, 0.60, ("ell",)),        "橢圓 f.60"),
    ("Bool切負", lambda s: bool_carve(s, 0.66, ("ell", "rect")), "橢+旋矩 f.66"),
    ("Bool切保留", lambda s: bool_keep(s, 0.42, ("ell",)),        "橢圓 f.42(反向)"),
    ("Sierpinski", lambda s: sierpinski(False), "L1(25=5²)"),
    ("Sierpinski", lambda s: sierpinski(True),  "L1+L2"),
    ("混血",  hybrid,                          "GRF負+工程縫"),
    ("GRF負·鏡射", lambda s: grf(s, 0.68, 1.5, sym=True), "f.68 σ1.5 sym"),
]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(4, 4, figsize=(14, 14.5))
for k, (fam, fn, tag) in enumerate(SPECS):
    raw = fn(20260731 * 1000 + k)
    m, nm, nv = postprocess(raw)
    ax = axes[k // 4][k % 4]
    img = np.ones((N, N, 3))
    img[m] = (0.15, 0.25, 0.5)
    ax.imshow(img, interpolation="nearest")
    ax.scatter([FEED[1]], [FEED[0]], c="red", s=45, marker="s", zorder=5)
    ax.set_title(f"#{k+1} {fam} {tag}\n金屬 {m.mean():.2f} | 件{nm} 洞{nv}", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    print(f"#{k+1:2d} {fam:8s} {tag:16s} 金屬 {m.mean():.2f} 金屬件 {nm} 洞 {nv}")
fig.suptitle("負片生成器 v2 樣張 ×16——隨機幾何家族(GRF level-cut/Boolean 切片/工程探針;深藍=金屬/白=空/紅=feed;後處理=feed pad+除塵+針孔縫合)", fontsize=12)
fig.tight_layout()
out = r"C:\Users\Ricky\AppData\Local\Temp\claude\C--Users-Ricky-Documents-GitHub-Antenna\514acb31-4aa0-43ec-a3b3-98cdf8e2a623\scratchpad\neg_gen_samples_v2.png"
fig.savefig(out, dpi=120)
print("saved", out)
