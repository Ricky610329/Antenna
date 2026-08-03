# -*- coding: utf-8 -*-
"""script/diffsim/l3.py — L3：精確分層介質 Green's function（Sommerfeld 積分 → 查表核）。

`docs/log/analysis-09-diffsim-l3.md` 的階段 A。**L3 不是推翻 L2，是換掉 L2 的一個函式**
（`DCIMKernel.table`）——MoM 骨架已被獨立稽核逐條通過，不重寫。

## 為什麼要換

現行 2 鏡像 DCIM 的函數族 `Σ a·e^{−jknR}/(4πR)` 衰減不慢於 1/r，**結構上表達不出**
表面波的柱面波 `e^{−jβρ}/√ρ`（Şimşek 2006：球面波湊不出柱面波）。實測差距（28GHz）：

| ρ | \\|G_A\\| 比 | 相位差 | \\|G_V\\| 比 | 相位差 |
|---|---|---|---|---|
| 0.2mm | 1.12 | −0.0° | 1.02 | 3.1° |
| 6.93mm | 1.18 | −0.3° | **7.55** | **76.8°** |

**錯的幾乎全在 G_V**——而 formulation C 下 `G̃_A = Ṽ_i^h/(jωμ₀)` 只含 TE 線、帶內無極點
（TE₁ 截止 92.4GHz）⇒ **TM₀ 表面波完全由 G_V 攜帶，G_A 的權重是 0**。

## 為什麼「離線算、不可微」不影響可微性（Ricky 的硬約束）

核只是 stackup 的函數，**與設計變數 ρ 完全無關**。梯度鏈是
`ρ → wgt → load → (Z + diag(load)) → solve → S11`，**Z（含核）在這條鏈上是常數**
（已用 `test_l2_gradient_check_vs_finite_difference` 對中央差分實測確認）。
⇒ 用 scipy 離線算、存成 buffer、執行期純查表，對 ∂L/∂ρ **零損失**。

而且格心偏移 `d² = gi²+gj²` 是**整數** → 相異距離很少（建到 80×25 也只有 **1409 個**）
→ **精確查表，連內插都不需要**。表就以 `d²` 為索引（`grid_d2`），所以**與格網無關**：
加饋線、改貼片尺寸都共用同一張表，只要 `d²` 在覆蓋內。全表 749 KiB、離線 ~1 分鐘。

## 公式（Michalski–Mosig formulation C，e^{+jωt} 慣例）

源與觀察點都在 z=h 介面上、PEC 地在 z=0：

    k_z = −j·√(k_ρ² − k²)                    ← principal sqrt（proper sheet）
    Z^h = ωμ₀/k_z ,  Z^e = k_z/(ωε)
    Ṽ_i = Z₀·(j Z₁ tan(k_z1 h)) / (Z₀ + j Z₁ tan(k_z1 h))     ← 短路端接的並聯
    G̃_A = Ṽ_i^h/(jωμ₀) ,   G̃_V = (jωε₀/k_ρ²)(Ṽ_i^e − Ṽ_i^h)
    G(ρ) = (1/2π)∫₀^∞ G̃(k_ρ) J₀(k_ρρ) k_ρ dk_ρ

⚠ **這兩行係數是推導 + 數值驗證的，不是文獻逐字抄**（εr=1 時精確重現「源−鏡像」閉式到 1.5e-4，
對係數/符號/分支錯誤致命敏感）。獨立對帳（empymod）另派 agent 進行中。

## 兩個實作上的坑（都踩過，寫下來免得重蹈）

1. **路徑方向與時間慣例綁死**：e^{+jωt} 下極點在下半平面 ⇒ 路徑往**上**偏折，
   且分支必須用 `k_z = −j√(...)` 的 principal sqrt（不是「算完再判 Im>0 翻號」，
   那會在變形段跳到 improper sheet）。做錯的症狀是**虛部整個變號**。
   （SMUTHI 往下偏折，因為它用 e^{−iωt}——抄它的參數可以，抄方向會錯號。）
2. **兩個核的靜態扣除項不一樣**：`G̃_V` 的大 k_ρ 漸近是 `1/((εr+1)k_ρ)`，不是 `1/(2k_ρ)`
   ——介面上的電荷看到的是 (εr+1)/2 的平均介電常數。共用同一個扣除項會讓 G_V 的尾巴不收斂。
"""
import os

import numpy as np
import torch
from scipy.special import jv

from .geom import DX, H, EPS_R, TAN_D, C0, MU0, EPS0, FREQS

ERC = EPS_R * (1 - 1j * TAN_D)                 # 複數介電常數（含損耗角）
B_REG = 0.4 * DX                               # 自項正則化半徑（沿用 L2；階段 B 才做真積分）
_GL = {}


def _kz(k, krho):
    """principal sqrt，proper sheet。**不可以**改成「算完判 Im 再翻號」（見模組 docstring 坑 ①）。"""
    return -1j * np.sqrt(np.asarray(krho, dtype=complex) ** 2 - np.asarray(k, dtype=complex) ** 2)


def gtilde(krho, f, er):
    """譜域核 (G̃_A, G̃_V)。"""
    w = 2 * np.pi * f
    k0 = w / C0
    kz0, kz1 = _kz(k0, krho), _kz(k0 * np.sqrt(er), krho)
    z0h, z1h = w * MU0 / kz0, w * MU0 / kz1
    z0e, z1e = kz0 / (w * EPS0), kz1 / (w * EPS0 * er)
    t = np.tan(kz1 * H)
    vh = z0h * (1j * z1h * t) / (z0h + 1j * z1h * t)
    ve = z0e * (1j * z1e * t) / (z0e + 1j * z1e * t)
    return vh / (1j * w * MU0), (1j * w * EPS0 / krho ** 2) * (ve - vh)


def _static_spec(krho, er):
    """兩個核的大 k_ρ 漸近（扣掉才收斂）。⚠ G_V 帶 (εr+1) 因子，與 G_A 不同（坑 ②）。"""
    th = np.tanh(krho * H)
    return (1 - np.exp(-2 * krho * H)) / (2 * krho), th / (krho * (er + th))


def _static_space(rho, er, nimg: int = 80):
    """上面兩項的**精確**空間域對應（扣除不改答案、只改收斂速度）。

    G_V 那條就是微帶的古典靜態鏡像級數，K=(εr−1)/(εr+1)=0.56 → 30 項就到 1e-8。
    """
    a = 1 / (4 * np.pi * rho) - 1 / (4 * np.pi * np.sqrt(rho ** 2 + 4 * H ** 2))
    k = (er - 1) / (er + 1)
    v = 1 / rho
    for n in range(1, nimg + 1):
        v = v + ((-k) ** n - (-k) ** (n - 1)) / np.sqrt(rho ** 2 + (2 * n * H) ** 2)
    return a, v / (2 * np.pi * (er + 1))


def sommerfeld(rho, f, er=ERC, *, nmax=4.0, delta=0.10, ktail=200.0, n=400, er_static=None):
    """(G_A, G_V) at 距離 `rho`（m）、頻率 `f`（Hz）。路徑往**上**偏折（見坑 ①）。

    偏折深度上限 δ < 2/(k₀·ρ_max)（SMUTHI 規則，因為 J₀ 在複平面指數成長）。
    本專案 ρ_max = 6.93mm = 0.65λ₀ ⇒ 上限 ~0.4，而 TM₀ 極點離分支點只有 0.018–0.034·k₀
    ⇒ **路徑離奇異點遠一個數量級以上，不需要 residue 抽取或極點追蹤。**
    """
    rho = np.atleast_1d(np.asarray(rho, dtype=float))
    k0 = 2 * np.pi * f / C0
    ers = float(np.real(er)) if er_static is None else er_static
    if n not in _GL:
        _GL[n] = np.polynomial.legendre.leggauss(n)
    x, w = _GL[n]

    def integ(krho):
        k = krho[:, None]
        ga, gv = gtilde(k, f, er)
        qa, qv = _static_spec(k, ers)
        b = jv(0, k * rho[None, :]) * k / (2 * np.pi)
        return np.stack([(ga - qa) * b, (gv - qv) * b])

    pts = [0.0 + 0j, 1j * delta * k0, (nmax + 1j * delta) * k0, nmax * k0, ktail * k0]
    tot = np.zeros((2, rho.size), dtype=complex)
    for a, b in zip(pts[:-1], pts[1:]):
        t = 0.5 * (b - a) * x + 0.5 * (b + a)
        tot += 0.5 * (b - a) * np.sum(w[None, :, None] * integ(t), axis=1)
    sa, sv = _static_space(rho, ers)
    return tot[0] + sa, tot[1] + sv


def tm0_neff(f, er=EPS_R):
    """TM₀ 表面波的有效折射率（解 k_z1·tan(k_z1 h) = εr·α₀）。診斷/驗證用。

    #! εr ≤ 1 時**沒有束縛表面波**（n_eff 的區間 [1, √εr] 是空的）——直接回 1.0，
    #  不要讓 brentq 去解一個上下界顛倒的區間（原本會拋 ValueError，掃 εr 的驗證實驗踩過）。
    """
    from scipy.optimize import brentq
    if er <= 1.0 + 1e-12:
        return 1.0
    k0 = 2 * np.pi * f / C0

    def res(nf):
        kz1 = k0 * np.sqrt(max(er - nf ** 2, 1e-18))
        a0 = k0 * np.sqrt(max(nf ** 2 - 1.0, 1e-18))
        return kz1 * np.tan(kz1 * H) - er * a0

    return brentq(res, 1.0 + 1e-12, np.sqrt(er) - 1e-9, xtol=1e-14)


# ---------------------------------------------------------------- 查表核
#! 表覆蓋到 (nr, nc) = (80, 25)：25×25 貼片 + 最長 55 列（11mm）饋線。
#  ★ 2026-08-03 改成**以整數平方距離 d² = (r/dx)² 為索引**（原本綁死 l2 的 (NR, NC)）。
#  格點距離一律是 `dx·√整數`，所以 d² 是**精確的整數鍵**，查表零容差問題；
#  而且換格網（加饋線、改貼片尺寸）只要 d² 在覆蓋範圍內就能**共用同一張表**。
TAB_NR, TAB_NC = 80, 25


def grid_d2(nr: int, nc: int) -> np.ndarray:
    """(nr, nc) 格網會用到的所有整數平方距離，去重排序。`l2` 的 `rtab` 是它的 `dx·√d²`。"""
    gi, gj = np.meshgrid(np.arange(nr), np.arange(nc), indexing="ij")
    return np.unique((gi ** 2 + gj ** 2).ravel())


def build_table(freqs=FREQS, b_reg: float = B_REG, path: str = None, verbose: bool = True,
                nr: int = TAB_NR, nc: int = TAB_NC):
    """離線算全表 → npz。索引 = 整數平方距離 d²（見上）。"""
    from . import data as D
    path = path or os.path.join(D.CACHE_DIR, "l3_table.npz")
    d2 = grid_d2(nr, nc)
    #? 自項/近格：統一用 r_eff = √(r²+b_reg²) 正則化（沿用 L2 的 b_reg，階段 A 不做真積分）。
    #  對 r=0 給有限值；對 r=6.93mm 只差 0.007%，遠端不受影響。
    reff = np.sqrt((DX ** 2) * d2 + b_reg ** 2)
    if verbose:
        print(f"格網 {nr}×{nc} → {len(d2)} 個相異距離；{len(freqs)} 個頻率")
    out = np.empty((2, len(freqs), len(d2)), dtype=np.complex128)
    for i, f in enumerate(freqs):
        out[0, i], out[1, i] = sommerfeld(reff, float(f))
        if verbose:
            print(f"  {f / 1e9:5.1f} GHz  ok", flush=True)
    np.savez_compressed(path, table=out, d2=d2, freqs=np.asarray(freqs), b_reg=b_reg)
    if verbose:
        print(f"落地 {path}（{out.nbytes / 1024:.0f} KiB）")
    return path


class L3Kernel(torch.nn.Module):
    """精確 Sommerfeld 表的查表核。介面與 `DCIMKernel.table` 相同，可直接替換。

    表是 `register_buffer`（不是 Parameter）——它是**物理常數**，不參與擬合。
    要保留「節點可置換」時，在它上面疊零初始化的殘差層（`LearnedKernel(base=L3Kernel())`），
    記帳就很乾淨：物理節＝精確表，學習節＝距離相依的複數殘差。
    """

    def __init__(self, path: str = None, dtype=torch.float64):
        super().__init__()
        from . import data as D
        path = path or os.path.join(D.CACHE_DIR, "l3_table.npz")
        if not os.path.exists(path):
            raise SystemExit(f"找不到 {path}——先跑 python -m script.diffsim.l3 build")
        z = np.load(path)
        if "d2" not in z:
            raise SystemExit(f"{path} 是舊格式（綁死 l2 格網）——重建：python -m script.diffsim.l3 build")
        cd = torch.complex128 if dtype == torch.float64 else torch.complex64
        self.register_buffer("tab", torch.as_tensor(z["table"], dtype=cd))
        self.register_buffer("d2", torch.as_tensor(z["d2"], dtype=torch.int64))
        self.register_buffer("k0s", torch.as_tensor(2 * np.pi * z["freqs"] / C0, dtype=dtype))
        self.b_reg = float(z["b_reg"])
        self._icache = {}

    def _dist_index(self, r: torch.Tensor) -> torch.Tensor:
        """距離表 → 表內索引。鍵是**整數** d² = (r/dx)²，所以是精確比對、不是容差比對。"""
        key = (int(r.shape[0]), float(r[-1]))
        idx = self._icache.get(key)
        if idx is None:
            d2 = torch.round((r.double() / DX) ** 2).to(torch.int64)
            idx = torch.searchsorted(self.d2, d2).clamp_(max=self.d2.numel() - 1)
            if not torch.equal(self.d2[idx], d2):
                bad = int(d2[self.d2[idx] != d2].max())
                raise ValueError(f"距離 d²={bad} 超出 L3 表覆蓋（建到 d²={int(self.d2[-1])}）"
                                 f"——用更大的格網要重建：python -m script.diffsim.l3 build")
            self._icache[key] = idx
        return idx

    def table(self, r: torch.Tensor, k0: torch.Tensor) -> torch.Tensor:
        """(2, nf, ntab)。`r` 任意順序、任意長度，只要每個距離都在表的 d² 覆蓋內。"""
        fi = (k0.reshape(-1, 1) - self.k0s.reshape(1, -1)).abs().argmin(-1)
        #! 頻率必須落在建表的格上——查最近鄰若偏太多代表叫方在用沒建過的頻率。
        if float((self.k0s[fi] - k0.reshape(-1)).abs().max() / self.k0s.mean()) > 1e-3:
            raise ValueError("要求的頻率不在 L3 表上（表只建了資料集的 17 點）")
        return self.tab[:, fi, :][..., self._dist_index(r)]


def main():
    import argparse
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="L3 精確分層 Green's function")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--b-reg", type=float, default=B_REG, dest="b_reg")
    sub.add_parser("check")
    a = ap.parse_args()
    if a.cmd == "build":
        build_table(b_reg=a.b_reg)
    else:
        rho = np.array([0.2, 1.0, 3.0, 6.93]) * 1e-3
        for f in (24e9, 28e9, 32e9):
            print(f"{f / 1e9:.0f}GHz  TM0 n_eff = {tm0_neff(f):.4f}")
        ga, gv = sommerfeld(rho, 28e9, 1.0)
        k0 = 2 * np.pi * 28e9 / C0
        r1 = np.sqrt(rho ** 2 + (2 * H) ** 2)
        ref = np.exp(-1j * k0 * rho) / (4 * np.pi * rho) - np.exp(-1j * k0 * r1) / (4 * np.pi * r1)
        print(f"er=1 對閉式：G_A {np.abs(ga - ref).max() / np.abs(ref).max():.2e}"
              f"  G_V {np.abs(gv - ref).max() / np.abs(ref).max():.2e}")


if __name__ == "__main__":
    main()
