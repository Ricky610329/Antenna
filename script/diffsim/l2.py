# -*- coding: utf-8 -*-
"""script/diffsim/l2.py — L2 空間域 MoM（rooftop 基底 + 可擬合的分層 Green's function）。

`docs/diffsim.md` §3 L2 的甜蜜點方案 (b)：**求解器本身可微，所以「擬核」＝對解算器反傳**，
不用另外寫擬合器。相對 L1 的關鍵差別——**不假設「特徵尺寸 ≫ 基板厚」**：
L1 的硬磁牆理想化在 0.2mm 像素 / 0.508mm 基板上本來就破掉（analysis-08 §3.3 診斷），
MoM 直接解真實表面電流，這個假設整個不存在。

## 未知數與矩陣

均勻格 rooftop：x 向屋頂在 (i,j)-(i+1,j) 面上、y 向在 (i,j)-(i,j+1) 面上。
延伸格多兩列饋線樁（永遠金屬，只在饋線寬度 col 9–15），delta-gap 電壓源就打在
x = 5.0mm 的接面上——**參考面放這裡是合法的**，因為饋線是 51Ω 匹配線（`geom.py`）。

混合位（MPIE，Michalski–Mosig formulation C，G_A^xy = 0）：

    Z_mn = jωμ·dx²·G_A(Δ)·δ_同向  +  (dx²/jωε)·Σ_{p,q} s_p s_q G_V(c_p − c_q)

兩項都只查「格心偏移」的表 → 平移不變 → 一張 (2·NR−1)×(2·NC−1) 的小表就夠。

## 核的參數化：離散複數鏡像（DCIM）

分層介質的 Green's function 標準做法，不是隨便湊的函數族：

    G(r) = Σ_i a_i · exp(−j k₀ n_i √(r² + b_i²)) / (4π √(r² + b_i²))

物理初值（接地介質板的古典近似）＝「源 − 地鏡像」兩項：a=(+1, −1)、b=(b_reg, 2h)、
n=(√εr, √εr)。b_reg 同時扮演自項的正則化（rooftop 自耦合不會發散）。
之後 (a_i, b_i, n_i) 全部當複數參數，用可微鏈在 `fit` 分割上端到端擬。
"""
import numpy as np
import torch

from .geom import N, DX, H, EPS_R, Z0, C0, MU0, EPS0, FREQS, FEED_ROW, feed_weights

NSTUB = 1                     # 饋線樁列數（讓 x=5.0mm 接面上的屋頂有立足點）
NR = N + NSTUB                # 延伸格列數（x 方向）
NC = N                        # 行數（y 方向）


def stub_mask() -> np.ndarray:
    """饋線樁：延伸列在饋線寬度（col 9–15，`geom.feed_weights` 的支撐）上永遠是金屬。"""
    m = np.zeros((NSTUB, NC), dtype=np.float64)
    m[:, feed_weights() > 0] = 1.0
    return m


class DCIMKernel(torch.nn.Module):
    """G_A / G_V 各一組複數鏡像。參數是 log 空間的 b（保正）與自由複數 a、n。"""

    def __init__(self, n_img: int = 3, dtype=torch.float64):
        super().__init__()
        self.n_img = n_img
        er = float(np.sqrt(EPS_R))
        a = torch.zeros(2, n_img, 2, dtype=dtype)      # [kernel, image, (re, im)]
        b = torch.zeros(2, n_img, dtype=dtype)
        n = torch.zeros(2, n_img, 2, dtype=dtype)
        for k in range(2):
            a[k, 0, 0] = 1.0                            # 直接項
            a[k, 1, 0] = -1.0 if n_img > 1 else 0.0     # 地鏡像（反號）
            b[k, 0] = np.log(0.4 * DX)                  # 自項正則化半徑
            if n_img > 1:
                b[k, 1] = np.log(2 * H)
            for i in range(2, n_img):
                a[k, i, 0] = 0.0
                b[k, i] = np.log(4 * H * (i - 1))
            n[k, :, 0] = er
        self.a = torch.nn.Parameter(a)
        self.b = torch.nn.Parameter(b)
        self.n = torch.nn.Parameter(n)

    def table(self, r: torch.Tensor, k0: torch.Tensor) -> torch.Tensor:
        """r (…,) 距離表、k0 (nf,) → (2, nf, …) 的 G_A / G_V 值。"""
        a = torch.complex(self.a[..., 0], self.a[..., 1])          # (2, n_img)
        n = torch.complex(self.n[..., 0], self.n[..., 1])
        b = self.b.exp()
        rr = torch.sqrt(r[None, None, ...] ** 2 + (b ** 2)[:, :, None])          # (2,ni,…)
        ph = torch.exp(-1j * k0[None, None, :, None] * n[:, :, None, None]
                       * rr[:, :, None, :])                                      # (2,ni,nf,…)
        return (a[:, :, None, None] * ph / (4 * np.pi * rr[:, :, None, :])).sum(1)


class MoML2:
    """L2 MoM 求解器。核可擬（`kernel` 的參數），對 Z 與電流全程可微。"""

    def __init__(self, kernel: DCIMKernel = None, device="cpu", dtype=torch.float64,
                 n_theta: int = 13, n_phi: int = 24):
        self.device = torch.device(device)
        self.dtype = dtype
        self.cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        self.kernel = (kernel or DCIMKernel(dtype=dtype)).to(self.device)
        self._build_topology()
        self._setup_farfield(n_theta, n_phi)

    # ----------------------------------------------------------------- 拓樸
    def _build_topology(self):
        cell = np.arange(NR * NC).reshape(NR, NC)
        # x 屋頂：面在 (i,j)-(i+1,j)；y 屋頂：面在 (i,j)-(i,j+1)
        xa, xb = cell[:-1, :].ravel(), cell[1:, :].ravel()
        ya, yb = cell[:, :-1].ravel(), cell[:, 1:].ravel()
        self.nx, self.ny = len(xa), len(ya)
        self.nb = self.nx + self.ny
        self.cell_a = torch.as_tensor(np.concatenate([xa, ya]), dtype=torch.long)
        self.cell_b = torch.as_tensor(np.concatenate([xb, yb]), dtype=torch.long)
        self.is_x = torch.as_tensor(np.concatenate([np.ones(self.nx), np.zeros(self.ny)]),
                                    dtype=torch.bool)
        ci, cj = np.divmod(np.arange(NR * NC), NC)
        self.ci = torch.as_tensor(ci, dtype=torch.long)
        self.cj = torch.as_tensor(cj, dtype=torch.long)
        # 格心偏移表索引（平移不變 → 只查 (di+NR-1, dj+NC-1)）
        di = ci[:, None] - ci[None, :] + (NR - 1)
        dj = cj[:, None] - cj[None, :] + (NC - 1)
        self.off = torch.as_tensor(di * (2 * NC - 1) + dj, dtype=torch.long).to(self.device)
        gi, gj = np.meshgrid(np.arange(-(NR - 1), NR), np.arange(-(NC - 1), NC), indexing="ij")
        self.rtab = torch.as_tensor(DX * np.sqrt(gi ** 2 + gj ** 2).ravel(),
                                    dtype=self.dtype, device=self.device)
        # 饋電：x=5.0mm 接面上的 x 屋頂（i=N-1 與 i=N 之間）
        drive = np.zeros(self.nb, dtype=bool)
        drive[: self.nx] = (cell[:-1, :].ravel() // NC == FEED_ROW) & \
                           (np.tile(np.arange(NC), NR - 1) >= 0)
        row_of_x = np.repeat(np.arange(NR - 1), NC)
        col_of_x = np.tile(np.arange(NC), NR - 1)
        fw = feed_weights() > 0
        drive[: self.nx] = (row_of_x == FEED_ROW) & fw[col_of_x]
        self.drive = torch.as_tensor(drive, device=self.device)
        self.cell_a, self.cell_b = self.cell_a.to(self.device), self.cell_b.to(self.device)
        self.is_x = self.is_x.to(self.device)

    def edge_density(self, rho_ext: torch.Tensor) -> torch.Tensor:
        """(B, NR, NC) → (B, nb)：屋頂存在權重 ρ_a·ρ_b（連續鬆弛天然可微）。"""
        r = rho_ext.reshape(rho_ext.shape[0], -1)
        return r[:, self.cell_a] * r[:, self.cell_b]

    def extend(self, rho: torch.Tensor) -> torch.Tensor:
        """(B,25,25) 貼片 → (B,NR,NC) 含饋線樁。"""
        st = torch.as_tensor(stub_mask(), dtype=rho.dtype, device=rho.device)
        return torch.cat([rho, st[None].expand(rho.shape[0], -1, -1)], 1)

    # ----------------------------------------------------------------- 矩陣
    def impedance_at(self, fk: float) -> torch.Tensor:
        """單一頻率 → (nb, nb) 的 Z。**與樣本無關**（平移不變核）→ 整批共用，這是 L2 便宜的原因。

        逐頻率算不是為了優雅，是記憶體：(17, 1249, 1249) complex128 就 424MB，
        再乘 batch 直接爆。
        """
        k0 = torch.as_tensor([2 * np.pi * fk / C0], dtype=self.dtype, device=self.device)
        g = self.kernel.table(self.rtab, k0)[:, 0]              # (2, ntab)
        ga, gv = g[0][self.off], g[1][self.off]                 # (ncell, ncell)
        w = 2 * np.pi * fk
        same = (self.is_x[:, None] == self.is_x[None, :])
        a, b = self.cell_a, self.cell_b
        #! 兩項的 dx 冪次不同，寫錯就整條物理翻掉（實測踩過：A 項誤用 dx² 讓電感項比電容項
        #  大 2.5e5 倍 → 完全不是準靜態，Zin 直接 4e7Ω）。∫f_m dS = dx²（屋頂：三角×脈衝），
        #  故 A 項 ∫∫f_m f_n G_A ≈ dx⁴·G_A；而 ∫∇·f_m over cell = ±dx，故 V 項是 dx²。
        #  兩項都是 Ω·m²；比值 zv/za = 1/(k²dx²·G_A/G_VV) ≈ 99 → 格尺度上電容主導 ✓ 準靜態。
        za = (1j * w * MU0 * DX ** 4) * ga[a][:, a] * same      # A 項：同向才有（G_A^xy = 0）
        gvv = gv[a][:, a] - gv[a][:, b] - gv[b][:, a] + gv[b][:, b]   # 電荷偶極（面兩側 ±1）
        return za + (DX * DX / (1j * w * EPS0)) * gvv

    # ----------------------------------------------------------------- 求解
    def solve(self, rho: torch.Tensor, freqs=None, r_open: float = 1e6) -> dict:
        """(B,25,25) → {'S11','Gain'}（dB, 17 點）。

        void 屋頂用「電阻加載」關掉（R = r_open·(1−ρ)/max(ρ,ε)）——SIMP for conductors 的
        標準做法，二值時等價於只在金屬邊上解，但保持固定維度、可批次、對 ρ 可微。
        """
        if freqs is None:
            freqs = FREQS
        f = torch.as_tensor(np.asarray(freqs), dtype=self.dtype, device=self.device)
        rho_e = self.extend(rho.to(self.dtype))
        wgt = self.edge_density(rho_e)                          # (B, nb)
        load = (r_open * (1.0 - wgt) / wgt.clamp_min(1e-6)).to(self.cdtype)   # (B, nb)
        v = torch.zeros(self.nb, dtype=self.cdtype, device=self.device)
        v[self.drive] = DX                                      # delta-gap，V = 1
        cur = torch.stack([torch.linalg.solve(
            self.impedance_at(float(fk))[None] + torch.diag_embed(load),
            v.expand(rho.shape[0], self.nb)) for fk in f], 1)   # (B, nf, nb)
        itot = (cur[:, :, self.drive].sum(-1) * DX)
        zin = 1.0 / itot
        gam = (zin - Z0) / (zin + Z0)
        s11 = 20 * torch.log10(gam.abs().clamp(1e-6, 1.0))
        u0, prad = self.farfield(cur, f)
        d0 = (4 * np.pi * u0 / prad.clamp_min(1e-300)).clamp_min(1e-4)
        mism = (1 - gam.abs() ** 2).clamp_min(1e-6)
        return dict(S11=s11, Gain=10 * torch.log10(d0 * mism), Zin=zin, D0=d0, Prad=prad, J=cur)

    # ----------------------------------------------------------------- 遠場
    def _setup_farfield(self, n_theta, n_phi):
        th = torch.linspace(0.0, np.pi / 2, n_theta, dtype=self.dtype, device=self.device)
        ph = torch.linspace(0.0, 2 * np.pi, n_phi + 1, dtype=self.dtype, device=self.device)[:-1]
        T, P = torch.meshgrid(th, ph, indexing="ij")
        self.dth, self.dph = float(th[1] - th[0]), float(2 * np.pi / n_phi)
        u, v = torch.sin(T) * torch.cos(P), torch.sin(T) * torch.sin(P)
        # 屋頂面心座標
        ax = (self.ci[self.cell_a] + self.ci[self.cell_b] + 1) * 0.5 * DX
        ay = (self.cj[self.cell_a] + self.cj[self.cell_b] + 1) * 0.5 * DX
        self._geo = (u[:, :, None] * ax.to(self.device) + v[:, :, None] * ay.to(self.device))
        self.sinT, self.cosT = torch.sin(T), torch.cos(T)
        self.cosP, self.sinP = torch.cos(P), torch.sin(P)
        self._ffc = {}

    def farfield(self, cur, f):
        """電流片遠場：水平電流 + 地平面鏡像 → 因子 2j·sin(k₀h·cosθ)（薄基板≈2j k₀h cosθ）。"""
        key = (float(f[0]), float(f[-1]), len(f))
        if key not in self._ffc:
            k0 = (2 * np.pi * f / C0).to(self.dtype)
            self._ffc[key] = (torch.exp(1j * (k0[:, None, None, None] * self._geo).to(self.cdtype)),
                              (2j * torch.sin(k0[:, None, None] * H * self.cosT[None])).to(self.cdtype))
        ph, img = self._ffc[key]
        jx = cur * self.is_x
        jy = cur * (~self.is_x)
        sx = torch.einsum("bfp,ftup->bftu", jx, ph) * img[None]
        sy = torch.einsum("bfp,ftup->bftu", jy, ph) * img[None]
        # 遠場 ∝ r̂ ×(r̂ × J) 的橫向分量
        et = self.cosT * (self.cosP * sx + self.sinP * sy)
        ep = -self.sinP * sx + self.cosP * sy
        pw = et.abs() ** 2 + ep.abs() ** 2
        u0 = pw[:, :, 0, :].mean(-1)
        prad = (pw * self.sinT).sum((-1, -2)) * self.dth * self.dph
        return u0, prad

    def predict(self, patterns, batch: int = 4) -> np.ndarray:
        p = np.asarray(patterns, dtype=np.float64).reshape(-1, N, N)
        out = np.empty((len(p), 34))
        for i in range(0, len(p), batch):
            with torch.no_grad():
                r = self.solve(torch.as_tensor(p[i:i + batch], dtype=self.dtype, device=self.device))
            out[i:i + batch, :17] = r["S11"].cpu().numpy()
            out[i:i + batch, 17:] = r["Gain"].cpu().numpy()
        return out
