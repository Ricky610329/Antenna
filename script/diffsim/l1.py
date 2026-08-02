# -*- coding: utf-8 -*-
"""script/diffsim/l1.py — L1 可微廣義腔模型（`docs/diffsim.md` §3 L1）。

**物理**：薄基板（h ≪ λ）貼片＝上下電牆、側面磁牆（PMC）的腔。場只有 E_z(x,y)，
滿足 2D Helmholtz + Neumann 邊界。任意像素形狀 → 金屬區上的特徵問題：

    A φ = λ M φ          A = 密度加權 5 點 Laplacian（面權 w = ρ_a ρ_b，天然連續鬆弛）
                         M = diag(ρ + ε)         k_n² = λ_n / dx²
    Z_in(ω) = jωμ₀h Σ_n ⟨ψ_n⟩_feed² / (k_n² − k̃²)        k̃² = k²(1 − j/Q)
    S11 = (Z_in − 50)/(Z_in + 50)

**遠場**：腔模型的輻射源＝周邊磁流 M = −2h E_z (n̂ × ẑ)（×2 為地平面鏡像）。
E_z 是全模態疊加（含多模干涉），故 D₀ 與 Realized Gain 都有形狀相依性：

    RealizedGain(dB) = 10log₁₀ D₀ + 10log₁₀(1 − |Γ|²) + 10log₁₀ e_r

**空洞格處理**：void 格加 α(1−ρ) 對角罰 + 質量 ε → 其特徵值被推到 α/ε（遠離物理帶），
二值時與金屬區**精確解耦**（w=0），故等價於「只在金屬區解」但保持固定 625 維、可批次、可微。
非連通島嶼不接饋點 → ⟨ψ⟩_feed = 0 自然去耦（指導書 §3 已預期）。

**已知不做的**（L1 定位就是「抓輪廓」）：輻射 Q 的精度、金屬島間空氣耦合、饋線本體效應、
負片域的 slot 模式反轉。饋線長度不必建模——理由見 `geom.py` docstring（50Ω 匹配線不改 |S11|）。
"""
import numpy as np
import torch

from .geom import (N, DX, H, EPS_R, TAN_D, SIGMA_CU, Z0, C0, MU0, EPS0, ETA0,
                   FREQS, FEED_ROW, feed_weights)

_IDX = np.arange(N * N).reshape(N, N)


def _pairs():
    """5 點鄰接的 (a, b) 平面索引對：x 方向（列間）與 y 方向（行間）。"""
    ax = np.stack([_IDX[:-1, :].ravel(), _IDX[1:, :].ravel()])
    ay = np.stack([_IDX[:, :-1].ravel(), _IDX[:, 1:].ravel()])
    return torch.as_tensor(ax, dtype=torch.long), torch.as_tensor(ay, dtype=torch.long)


class CavityL1:
    """L1 腔模型。所有可擬參數都是**幾個純量**（見 `params`），fit 只在 fit 分割上做。

    :param n_modes: 模態展開取的最低階數（含 λ=0 的靜電容模）。
    :param er_eff:  有效 εr——邊緣場（fringing）使等效腔大於實體，實務上以此一顆旋鈕吸收。
    :param q:       總 Q（L1a 用單一經驗常數；`self_q=True` 改用自洽輻射 Q）。
    """

    def __init__(self, n_modes: int = 30, er_eff: float = EPS_R, q: float = 20.0,
                 alpha: float = 1.0, eps_mass: float = 1e-2, self_q: bool = False,
                 n_theta: int = 19, n_phi: int = 36, device: str = "cpu", dtype=torch.float64):
        self.n_modes = n_modes
        self.er_eff = er_eff
        self.q = q
        self.alpha = alpha
        self.eps_mass = eps_mass
        self.self_q = self_q
        self.device = torch.device(device)
        self.dtype = dtype
        self.cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        ax, ay = _pairs()
        self.ax, self.ay = ax.to(self.device), ay.to(self.device)
        fw = torch.as_tensor(feed_weights(), dtype=dtype, device=self.device)
        self.fw_full = torch.zeros(N * N, dtype=dtype, device=self.device)
        self.fw_full[_IDX[FEED_ROW, :]] = fw
        self._setup_farfield(n_theta, n_phi)

    # ----------------------------------------------------------------- 遠場預備
    def _setup_farfield(self, n_theta, n_phi):
        th = torch.linspace(0.0, np.pi / 2, n_theta, dtype=self.dtype, device=self.device)
        ph = torch.linspace(0.0, 2 * np.pi, n_phi + 1, dtype=self.dtype, device=self.device)[:-1]
        T, P = torch.meshgrid(th, ph, indexing="ij")
        self.th, self.T, self.P = th, T, P
        self.dth = float(th[1] - th[0])
        self.dph = float(2 * np.pi / n_phi)
        u = torch.sin(T) * torch.cos(P)                       # (nt, np)
        v = torch.sin(T) * torch.sin(P)
        cx = (torch.arange(N, dtype=self.dtype, device=self.device) + 0.5) * DX
        X = cx[:, None].expand(N, N).reshape(-1)              # row = x
        Y = cx[None, :].expand(N, N).reshape(-1)              # col = y
        k0 = torch.as_tensor(2 * np.pi * FREQS / C0, dtype=self.dtype, device=self.device)
        # 相位矩陣 (nf, nt, np, 625)：源都在 z≈0，只有 xy 相位
        phase = k0[:, None, None, None] * (u[None, :, :, None] * X + v[None, :, :, None] * Y)
        self.ph_exp = torch.exp(1j * phase.to(self.cdtype))
        self.sinT = torch.sin(T)
        self.cosT = torch.cos(T)
        self.cosP, self.sinP = torch.cos(P), torch.sin(P)

    # ----------------------------------------------------------------- 特徵問題
    def modes(self, rho: torch.Tensor):
        """rho (B,25,25) ∈ [0,1] → (lam (B,m), psi (B,625,m))；psi 為 M-正交歸一。"""
        B = rho.shape[0]
        r = rho.reshape(B, -1).to(self.dtype)
        wx = (r[:, self.ax[0]] * r[:, self.ax[1]])            # (B, 600)
        wy = (r[:, self.ay[0]] * r[:, self.ay[1]])
        A = torch.zeros(B, N * N, N * N, dtype=self.dtype, device=self.device)
        flat = A.view(B, -1)
        for pair, w in ((self.ax, wx), (self.ay, wy)):
            a, b = pair[0], pair[1]
            flat.scatter_add_(1, (a * (N * N) + b).expand(B, -1), -w)
            flat.scatter_add_(1, (b * (N * N) + a).expand(B, -1), -w)
        deg = torch.zeros(B, N * N, dtype=self.dtype, device=self.device)
        for pair, w in ((self.ax, wx), (self.ay, wy)):
            deg.scatter_add_(1, pair[0].expand(B, -1), w)
            deg.scatter_add_(1, pair[1].expand(B, -1), w)
        diag = deg + self.alpha * (1.0 - r)                   # void 罰項：把空洞模推離物理帶
        eye = torch.arange(N * N, device=self.device)
        flat.scatter_add_(1, (eye * (N * N) + eye).expand(B, -1), diag)
        m = r + self.eps_mass * (1.0 - r)     # 金屬格質量剛好 1（不偏移 k_n），void 格 ε
        s = m.rsqrt()
        S = A * s[:, :, None] * s[:, None, :]                 # 對稱化廣義問題
        S = 0.5 * (S + S.transpose(1, 2))
        lam, phi = torch.linalg.eigh(S)
        lam = lam[:, :self.n_modes]
        psi = (phi[:, :, :self.n_modes] * s[:, :, None])      # ψ = M^{-1/2} φ，滿足 ψᵀMψ = I
        return lam.clamp_min(0.0), psi

    # ----------------------------------------------------------------- 響應
    def forward(self, rho: torch.Tensor, freqs=None) -> dict:
        """rho (B,25,25) → {'S11': (B,17) dB, 'Gain': (B,17) dB, ...}（皆可微）。"""
        if freqs is None:
            freqs = FREQS
        f = torch.as_tensor(np.asarray(freqs), dtype=self.dtype, device=self.device)
        B = rho.shape[0]
        r = rho.reshape(B, -1).to(self.dtype)
        lam, psi = self.modes(rho)
        kn2 = lam / (DX * DX)                                          # (B,m) 1/m²
        psi_p = psi / DX                                               # 物理歸一 ∫|ψ|²dA = 1
        pf = (psi_p * self.fw_full[None, :, None]).sum(1)              # ⟨ψ_n⟩_feed (B,m)

        w = 2 * np.pi * f
        k2 = (w / C0) ** 2 * self.er_eff                               # (nf,)
        k2t = k2.to(self.cdtype) * (1 - 1j / self.q)
        den = kn2[:, None, :].to(self.cdtype) - k2t[None, :, None]     # (B,nf,m)
        num = (pf ** 2)[:, None, :].to(self.cdtype)
        zin = 1j * (w * MU0 * H).to(self.cdtype)[None, :] * (num / den).sum(-1)
        gam = (zin - Z0) / (zin + Z0)
        s11_db = 20 * torch.log10(gam.abs().clamp_min(1e-6))

        # 場 → 周邊磁流 → 遠場
        an = 1j * (w * MU0).to(self.cdtype)[None, :, None] * pf[:, None, :].to(self.cdtype) / (-den)
        ez = torch.einsum("bfm,bpm->bfp", an, psi_p.to(self.cdtype))   # (B,nf,625) E_z
        gpx, gmx, gpy, gmy = self._edge_weights(r)
        mx = -2 * H * DX * ez * (gpy - gmy)[:, None, :].to(self.cdtype)
        my = -2 * H * DX * ez * (gmx - gpx)[:, None, :].to(self.cdtype)
        sx = torch.einsum("bfp,ftup->bftu", mx, self.ph_exp)           # (B,nf,nt,np)
        sy = torch.einsum("bfp,ftup->bftu", my, self.ph_exp)
        cross2 = (self.cosT ** 2) * (sx.abs() ** 2 + sy.abs() ** 2) \
            + (self.sinT * (self.cosP * sy - self.sinP * sx).abs()) ** 2
        u0 = cross2[:, :, 0, :].mean(-1)                               # θ=0（各 φ 同值，取平均穩定）
        prad = (cross2 * self.sinT).sum((-1, -2)) * self.dth * self.dph
        d0 = 4 * np.pi * u0 / prad.clamp_min(1e-300)
        mism = (1 - gam.abs() ** 2).clamp_min(1e-6)
        gain_db = 10 * torch.log10(d0.clamp_min(1e-12)) + 10 * torch.log10(mism)
        return dict(S11=s11_db, Gain=gain_db, Zin=zin, D0=d0, Prad=prad, lam=lam)

    def _edge_weights(self, r):
        """各格四面的「金屬→空氣」權重 g = ρ_i(1−ρ_j)（外界 ρ=0）。回 (+x, −x, +y, −y)。"""
        R = r.reshape(-1, N, N)
        z = torch.zeros_like(R[:, :1, :])
        zc = torch.zeros_like(R[:, :, :1])
        npx = torch.cat([R[:, 1:, :], z], 1)
        nmx = torch.cat([z, R[:, :-1, :]], 1)
        npy = torch.cat([R[:, :, 1:], zc], 2)
        nmy = torch.cat([zc, R[:, :, :-1]], 2)
        f = lambda nb: (R * (1 - nb)).reshape(-1, N * N)      # noqa: E731
        return f(npx), f(nmx), f(npy), f(nmy)

    # ----------------------------------------------------------------- 便利介面
    def predict(self, patterns, batch: int = 16, progress: bool = False) -> np.ndarray:
        """(n,625) 或 (n,25,25) → (n,34) 的 [S11(17), Gain(17)] dB，格式與資料集 y 一致。"""
        p = np.asarray(patterns, dtype=np.float64).reshape(-1, N, N)
        out = np.empty((len(p), 34), dtype=np.float64)
        for i in range(0, len(p), batch):
            chunk = torch.as_tensor(p[i:i + batch], dtype=self.dtype, device=self.device)
            with torch.no_grad():
                res = self.forward(chunk)
            out[i:i + batch, :17] = res["S11"].cpu().numpy()
            out[i:i + batch, 17:] = res["Gain"].cpu().numpy()
            if progress and (i // batch) % 20 == 0:
                print(f"  {i}/{len(p)}", flush=True)
        return out


def q_material() -> float:
    """材料損耗上限（介質 + 導體），輻射 Q 之外的部分。診斷用。"""
    qd = 1.0 / TAN_D
    f0 = 28e9
    delta_s = np.sqrt(2.0 / (2 * np.pi * f0 * MU0 * SIGMA_CU))
    qc = H / delta_s
    return 1.0 / (1.0 / qd + 1.0 / qc)
