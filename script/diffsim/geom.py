# -*- coding: utf-8 -*-
"""script/diffsim/geom.py — 幾何與材料常數（唯一真相：`single_port.py` + `sab/single_port.sab`）。

`docs/diffsim.md` §1 的「待確認」在 2026-08-02 階段 0 用 SAB 二進位解析**清零**（見 `sab_probe()`）：

| 物件 | 範圍 (mm) | 備註 |
|---|---|---|
| Sub (RO4003) | x[−7.5, 27.5] y[−7.5, 12.5] z[0, 0.508] | 板 35×20mm |
| GND (copper) | 同上 footprint，z[−0.035, 0] | 有限地平面 |
| feed_line | x[5.0, 27.5] y[1.95, 3.05] z[0.508, 0.543] | **寬 1.1mm、長 22.5mm** |
| Rectangle1 (port) | x=27.5 面，y[1.95, 3.05] z[0, 0.508] | Lumped 50Ω |
| 貼片畫布 | x[0, 5] y[0, 5]，25×25 格 @0.2mm | 像素 (row=x, col=y) |

**關鍵推論（本鏈少算一整段的理由）**：饋線 W/h = 1.1/0.508 = 2.17 → Z₀ ≈ 51Ω ≈ 埠阻抗。
均勻匹配線只轉相位、**不改 |S11|**；而資料集只用 dB(S11)（幅值）→ 22.5mm 饋線長度對主 KPI 無影響，
L1/L2 都不必建模饋線本體。饋線唯一進場的地方＝它以 1.1mm 寬接在貼片 x=5mm 邊上（見 `feed_weights`）。
"""
import numpy as np

N = 25                      # 每邊像素數
DX = 0.2e-3                 # 像素邊長 (m)
H = 0.508e-3                # 基板厚 (m)
CU_T = 0.035e-3             # 銅厚 (m)
EPS_R = 3.55                # Rogers RO4003
TAN_D = 0.0027
SIGMA_CU = 5.8e7            # 銅導電率 (S/m)

FEED_ROW = N - 1            # 饋線接在 x 最大側（x = 4.8–5.0mm）；FEED 像素 (24,12) 對得上
FEED_Y = (1.95e-3, 3.05e-3)  # 饋線寬度覆蓋的 y 區間（.sab 實測）
Z0 = 50.0                   # 埠參考阻抗

BOARD_X = (-7.5e-3, 27.5e-3)
BOARD_Y = (-7.5e-3, 12.5e-3)

C0 = 299792458.0
MU0 = 4e-7 * np.pi
EPS0 = 1.0 / (MU0 * C0 * C0)
ETA0 = MU0 * C0
FREQS = np.linspace(24e9, 32e9, 17)      # 與資料集 17 點同格


def feed_weights() -> np.ndarray:
    """饋線接觸權重 (25,)：貼片 x=5mm 邊上各 y 格與饋線 1.1mm 寬的重疊長度（歸一化）。

    邊緣饋電不是點饋——1.1mm 寬跨 5.5 格。點取樣會高估高階模耦合，故用寬度平均。
    """
    y0, y1 = FEED_Y
    lo = np.arange(N) * DX
    hi = lo + DX
    ov = np.clip(np.minimum(hi, y1) - np.maximum(lo, y0), 0.0, None)
    return ov / ov.sum()


def sab_probe(path=None) -> dict:
    """解析 ACIS SAB（tag 0x13 = position，3×float64）→ 各 z 層的 xy 外框。階段 0 的量測工具。"""
    import struct
    import os
    if path is None:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "antenna", "patch", "patch_simulator", "sab", "single_port.sab")
    b = open(path, "rb").read()
    pts, i = [], 0
    while i < len(b) - 24:
        if b[i] == 0x13:
            v = struct.unpack_from("<3d", b, i + 1)
            if all(abs(x) < 1e4 and (x == 0 or abs(x) > 1e-12) for x in v):
                pts.append(v)
                i += 25
                continue
        i += 1
    p = np.round(np.array(pts), 4)
    out = {}
    for z in sorted(set(p[:, 2].tolist())):
        m = p[:, 2] == z
        out[z] = dict(n=int(m.sum()), x=(p[m, 0].min(), p[m, 0].max()), y=(p[m, 1].min(), p[m, 1].max()))
    return out


if __name__ == "__main__":
    for z, d in sab_probe().items():
        print(f"z={z:+.4f}mm  n={d['n']:3d}  x{d['x']}  y{d['y']}")
    fw = feed_weights()
    print("feed weights (col:w):", {i: round(float(w), 3) for i, w in enumerate(fw) if w > 0})
