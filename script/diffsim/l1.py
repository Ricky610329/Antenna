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


def _gap_pairs():
    """電容耦合對：① 隔一格的同行/同列（中間 void ＝ 0.2mm 縫）② 對角相鄰。

    這不是可有可無的修飾——像素 0.2mm 但基板厚 0.508mm，0.2mm 縫的串聯電容
    與單格的對地電容同量級（C_gap/C_cell ~ O(1)），把縫當理想開路會讓結構
    比真實破碎得多。回 (直線隔格對, 中間格, 對角對)。
    """
    g = np.stack([np.concatenate([_IDX[:-2, :].ravel(), _IDX[:, :-2].ravel()]),
                  np.concatenate([_IDX[2:, :].ravel(), _IDX[:, 2:].ravel()])])
    mid = np.concatenate([_IDX[1:-1, :].ravel(), _IDX[:, 1:-1].ravel()])
    d = np.stack([np.concatenate([_IDX[:-1, :-1].ravel(), _IDX[:-1, 1:].ravel()]),
                  np.concatenate([_IDX[1:, 1:].ravel(), _IDX[1:, :-1].ravel()])])
    return (torch.as_tensor(g, dtype=torch.long), torch.as_tensor(mid, dtype=torch.long),
            torch.as_tensor(d, dtype=torch.long))


class CavityL1:
    """L1 腔模型。所有可擬參數都是**幾個純量**（見 `params`），fit 只在 fit 分割上做。

    :param n_modes: 模態展開取的最低階數（含 λ=0 的靜電容模）。
    :param er_eff:  有效 εr——邊緣場（fringing）使等效腔大於實體，實務上以此一顆旋鈕吸收。
    :param q:       總 Q（L1a 用單一經驗常數；`self_q=True` 改用自洽輻射 Q）。
    """

    def __init__(self, n_modes: int = None, er_eff: float = EPS_R, q: float = 20.0,
                 gap: float = 0.0, diag: float = 0.0, q_modes: int = 16, rad_eff: bool = False,
                 alpha: float = 1.0, eps_mass: float = 1e-2, self_q: bool = False,
                 n_theta: int = 13, n_phi: int = 24, device: str = "cpu", dtype=torch.float64):
        #? n_modes=None＝**全模態**（eigh 本來就把 625 個都算出來了，截斷只會製造假象：
        #  加一顆不接饋點的離島會把主貼片的模擠出「最低 N 個」→ S11 平白變動。
        #  全取＝離散問題的精確解，也少一個超參數。二值時 void 模 ψ_feed 恆 0、自動不貢獻。
        self.n_modes = n_modes
        self.er_eff = er_eff
        self.q = q
        self.gap = gap
        self.diag = diag
        self.q_modes = q_modes
        self.rad_eff = rad_eff
        self.alpha = alpha
        self.eps_mass = eps_mass
        self.self_q = self_q
        self.device = torch.device(device)
        self.dtype = dtype
        self.cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        ax, ay = _pairs()
        self.ax, self.ay = ax.to(self.device), ay.to(self.device)
        gp, gm, dg = _gap_pairs()
        self.gp, self.gmid, self.dg = gp.to(self.device), gm.to(self.device), dg.to(self.device)
        fw = torch.as_tensor(feed_weights(), dtype=dtype, device=self.device)
        self.fw_full = torch.zeros(N * N, dtype=dtype, device=self.device)
        self.fw_full[_IDX[FEED_ROW, :]] = fw
        self._setup_farfield(n_theta, n_phi)

    # ----------------------------------------------------------------- 遠場預備
    def _setup_farfield(self, n_theta, n_phi):
        th = torch.linspace(0.0, np.pi / 2, n_theta, dtype=self.dtype, device=self.device)
        ph = torch.linspace(0.0, 2 * np.pi, n_phi + 1, dtype=self.dtype, device=self.device)[:-1]
        T, P = torch.meshgrid(th, ph, indexing="ij")
        self.dth = float(th[1] - th[0])
        self.dph = float(2 * np.pi / n_phi)
        self.u = torch.sin(T) * torch.cos(P)                  # (nt, np)
        self.v = torch.sin(T) * torch.sin(P)
        cx = (torch.arange(N, dtype=self.dtype, device=self.device) + 0.5) * DX
        X = cx[:, None].expand(N, N).reshape(-1)              # row = x
        Y = cx[None, :].expand(N, N).reshape(-1)              # col = y
        self._geo = (self.u[:, :, None] * X + self.v[:, :, None] * Y)   # (nt,np,625) 幾何相位
        self._ph_cache = {}
        self._shift_cache = {}
        self.sinT, self.cosT = torch.sin(T), torch.cos(T)
        self.cosP, self.sinP = torch.cos(P), torch.sin(P)

    def _ph_exp(self, f):
        """(nf, nt, np, 625) 的 e^{jk₀(uX+vY)}（格心相位）＋ 半格位移因子 (cu, cv)。

        半格位移不可省：把四個面的磁流都塞回格心，1 格寬細條的 +y/−y 面會**恰好抵銷**
        → P_rad = 0 的假象（實測踩到）。面在格心 ±dx/2，相位差 e^{±jk₀·(u 或 v)·dx/2}，
        可從格求和外提，所以只多兩個係數、不多一個維度。
        """
        key = (float(f[0]), float(f[-1]), len(f))
        if key not in self._ph_cache:
            k0 = (2 * np.pi * f / C0).to(self.dtype)
            self._ph_cache[key] = torch.exp(1j * (k0[:, None, None, None] * self._geo).to(self.cdtype))
            cu = torch.exp(1j * (k0[:, None, None] * self.u[None] * DX / 2).to(self.cdtype))
            cv = torch.exp(1j * (k0[:, None, None] * self.v[None] * DX / 2).to(self.cdtype))
            self._shift_cache[key] = (cu, cv)
        return self._ph_cache[key], self._shift_cache[key]

    # ----------------------------------------------------------------- 特徵問題
    @staticmethod
    def _laplacian(w, pair, out, deg):
        """把一組邊權 w 累加成 Laplacian（off-diag −w、diag +w），就地寫進 out/deg。"""
        nb, n2 = out.shape[0], N * N
        flat = out.view(nb, -1)
        a, b = pair[0], pair[1]
        flat.scatter_add_(1, (a * n2 + b).expand(nb, -1), -w)
        flat.scatter_add_(1, (b * n2 + a).expand(nb, -1), -w)
        deg.scatter_add_(1, a.expand(nb, -1), w)
        deg.scatter_add_(1, b.expand(nb, -1), w)

    def modes(self, rho: torch.Tensor):
        """rho (B,25,25) ∈ [0,1] → (lam (B,m), psi (B,625,m))；psi 為 **B-正交歸一**。

        平面電路觀點：節點電壓 V=−hE_z，金屬↔金屬面＝串聯電感 jωL（L=μ₀h/方塊），
        每格對地電容 C=ε dx²/h，縫＝串聯電容 C_g。整理後（同乘 jωL）：

            (A − λ·B) V = jωL·I ,  λ = k²dx² ,  A = Lap(金屬鍵) + α·diag(1−ρ)
                                    B = diag(ρ+ε(1−ρ)) + γ_gap·Lap(隔格鍵) + γ_diag·Lap(對角鍵)

        關鍵：縫的 −ω²LC_g 與 k² 項**同頻率相依**（都 ∝ ω²）→ γ = C_g/C_cell 是純常數，
        縫耦合可以整個吸進質量矩陣，**仍然是一個廣義特徵問題**（不必逐頻率解線性系統）。
        """
        nb = rho.shape[0]
        n2 = N * N
        r = rho.reshape(nb, -1).to(self.dtype)
        A = torch.zeros(nb, n2, n2, dtype=self.dtype, device=self.device)
        degA = torch.zeros(nb, n2, dtype=self.dtype, device=self.device)
        for pair in (self.ax, self.ay):
            self._laplacian(r[:, pair[0]] * r[:, pair[1]], pair, A, degA)
        eye = torch.arange(n2, device=self.device)
        A.view(nb, -1).scatter_add_(1, (eye * n2 + eye).expand(nb, -1),
                                    degA + self.alpha * (1.0 - r))

        mass = r + self.eps_mass * (1.0 - r)   # 金屬格質量剛好 1（不偏移 k_n），void 格 ε
        if self.gap or self.diag:
            Bm = torch.zeros(nb, n2, n2, dtype=self.dtype, device=self.device)
            degB = torch.zeros(nb, n2, dtype=self.dtype, device=self.device)
            if self.gap:                       # 隔一格且中間是 void ＝ 0.2mm 實體縫
                wg = self.gap * r[:, self.gp[0]] * r[:, self.gp[1]] * (1 - r[:, self.gmid])
                self._laplacian(wg, self.gp, Bm, degB)
            if self.diag:                      # 對角相接：只有角碰角，走電容不走電流
                wd = self.diag * r[:, self.dg[0]] * r[:, self.dg[1]]
                self._laplacian(wd, self.dg, Bm, degB)
            Bm.view(nb, -1).scatter_add_(1, (eye * n2 + eye).expand(nb, -1), degB + mass)
            Bm = 0.5 * (Bm + Bm.transpose(1, 2))
            L = torch.linalg.cholesky(Bm)
            S = torch.linalg.solve_triangular(L, A, upper=False)
            S = torch.linalg.solve_triangular(L, S.transpose(1, 2), upper=False).transpose(1, 2)
            S = 0.5 * (S + S.transpose(1, 2))
            lam, phi = torch.linalg.eigh(S)
            psi = torch.linalg.solve_triangular(L.transpose(1, 2), phi, upper=True)
        else:
            s = mass.rsqrt()
            S = A * s[:, :, None] * s[:, None, :]
            S = 0.5 * (S + S.transpose(1, 2))
            lam, phi = torch.linalg.eigh(S)
            psi = phi * s[:, :, None]
        if self.n_modes is not None:
            lam, psi = lam[:, :self.n_modes], psi[:, :, :self.n_modes]
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
        #? 饋線只接得到「真的是金屬」的格：權重先乘 ρ 再重新歸一（總注入電流 I 固定）。
        #  漏掉這一步，void 格的高階假模（質量 ε → ψ 放大 10×）會從饋點灌進阻抗。
        fw = self.fw_full[None, :] * r
        fw = fw / fw.sum(1, keepdim=True).clamp_min(1e-12)
        pf = (psi_p * fw[:, :, None]).sum(1)                           # ⟨ψ_n⟩_feed (B,m)

        w = 2 * np.pi * f
        k2 = (w / C0) ** 2 * self.er_eff                               # (nf,)
        gpx, gmx, gpy, gmy = self._edge_weights(r)
        ph, (cu, cv) = self._ph_exp(f)
        edges = (gpx, gmx, gpy, gmy)

        q_rad = pick_ix = None
        if self.self_q or self.rad_eff:
            qn, q_rad, pick_ix = self._mode_q(psi_p, kn2, edges)
        if self.self_q:
            den = (kn2[:, None, :] - k2[None, :, None]).to(self.cdtype) \
                + 1j * (kn2 / qn)[:, None, :].to(self.cdtype)
        else:
            k2t = k2.to(self.cdtype) * (1 - 1j / self.q)
            den = kn2[:, None, :].to(self.cdtype) - k2t[None, :, None]  # (B,nf,m)
        num = (pf ** 2)[:, None, :].to(self.cdtype)
        zin = 1j * (w * MU0 * H).to(self.cdtype)[None, :] * (num / den).sum(-1)
        gam = (zin - Z0) / (zin + Z0)
        s11_db = 20 * torch.log10(gam.abs().clamp_min(1e-6))

        # 場（全模態疊加，含多模干涉）→ 周邊磁流 → 遠場
        an = 1j * (w * MU0).to(self.cdtype)[None, :, None] * pf[:, None, :].to(self.cdtype) / (-den)
        ez = torch.einsum("bfm,bpm->bfp", an, psi_p.to(self.cdtype))   # (B,nf,625) E_z
        u0, prad = self._farfield(ez, edges, ph, cu, cv)
        d0 = (4 * np.pi * u0 / prad.clamp_min(1e-300)).clamp_min(1e-4)  # 地板 −40dBi＝HFSS 懲罰值口徑
        mism = (1 - gam.abs() ** 2).clamp_min(1e-6)

        if self.rad_eff:
            #? 輻射效率 e_r = Q_mat/(Q_rad + Q_mat)：抓「會共振但不輻射」（能量困在結構裡的
            #  粉塵型）vs「真的會輻射」的差別——D₀ 只管方向性，抓不到這件事。
            #  只用 Q_rad（形狀的函數），**不走 P_rad/½Re(Zin) 的功率平衡**：Q 固定時 Re(Zin)
            #  反映的是「對饋點的耦合強度」不是真實損耗，那條路實測會把 Gain 弄壞（見 §3.1）。
            ix = pick_ix[:, None, :].expand(-1, den.shape[1], -1)
            wgt = (num.expand_as(den).gather(2, ix) / den.gather(2, ix)).abs()   # 各模當下的貢獻
            qr = (wgt * q_rad[:, None, :]).sum(-1) / wgt.sum(-1).clamp_min(1e-30)
            qm = q_material()
            e_r = (qm / (qr + qm)).clamp(1e-4, 1.0)
        elif self.self_q:
            cpw = ((2 * np.pi * f / C0) ** 2 / (32 * np.pi ** 2 * ETA0))[None, :]
            e_r = ((cpw * prad) / (0.5 * zin.real.clamp_min(1e-30))).clamp(1e-4, 1.0)
        else:
            e_r = torch.ones_like(d0)
        gain_db = 10 * torch.log10(d0 * e_r * mism)
        return dict(S11=s11_db, Gain=gain_db, Zin=zin, D0=d0, Prad=prad, eff=e_r, lam=lam)

    def _farfield(self, ez, edges, ph, cu, cv):
        """(B,K,625) 的 E_z → (u0 (B,K), prad_raw (B,K))。周邊磁流 M = −2h E_z (n̂ × ẑ)。

        四個面各帶自己的半格相位（不可省，見 `_ph_exp`）；×2 ＝ 地平面鏡像。
        回的是**相對**量（少一個 k₀²/(32π²η₀) 前因子），D₀ 用不到，自洽 Q 才乘回去。
        """
        gpx, gmx, gpy, gmy = edges
        amp = (-2 * H * DX) * ez
        A = {k: torch.einsum("bfp,ftup->bftu", amp * g[:, None, :].to(self.cdtype), ph)
             for k, g in (("px", gpx), ("mx", gmx), ("py", gpy), ("my", gmy))}
        sx = cv[None] * A["py"] - cv.conj()[None] * A["my"]            # n̂=±ŷ → M ∥ x̂
        sy = cu.conj()[None] * A["mx"] - cu[None] * A["px"]            # n̂=±x̂ → M ∥ ŷ
        cross2 = (self.cosT ** 2) * (sx.abs() ** 2 + sy.abs() ** 2) \
            + (self.sinT * (self.cosP * sy - self.sinP * sx).abs()) ** 2
        u0 = cross2[:, :, 0, :].mean(-1)                               # θ=0（各 φ 同值，取平均穩定）
        prad = (cross2 * self.sinT).sum((-1, -2)) * self.dth * self.dph
        return u0, prad

    def _mode_q(self, psi_p, kn2, edges):
        """逐模自洽 Q：Q_rad,n = ω_n·W_n / P_rad,n（W_n = εh/2，模態歸一 ∫|ψ|²dA = 1）。

        固定經驗 Q 的問題：諧振深度（S11 谷深）與輻射效率其實是**形狀的函數**——同樣共振，
        整片型會輻射、粉塵型把能量困在結構裡。只算頻帶附近的 M 個模（其餘遠離共振、Q 不影響），
        相位一律用頻帶中心，P_rad 再按 k₀ⁿ² 解析縮放。
        """
        nb, m = kn2.shape
        M = min(self.q_modes, m)
        kc2 = float(np.mean((2 * np.pi * FREQS / C0) ** 2 * self.er_eff))
        pick = torch.topk(-(kn2 - kc2).abs(), M, dim=1).indices               # (B,M)
        psi_sel = torch.gather(psi_p, 2, pick[:, None, :].expand(-1, N * N, -1))
        fc = torch.as_tensor([float(np.sqrt(kc2)) * C0 / np.sqrt(self.er_eff) / (2 * np.pi)],
                             dtype=self.dtype, device=self.device)
        ph, (cu, cv) = self._ph_exp(fc)
        _, prad = self._farfield(psi_sel.transpose(1, 2).to(self.cdtype), edges, ph, cu, cv)
        kn = kn2.gather(1, pick).clamp_min(1e-6).sqrt()                       # (B,M) in dielectric
        k0n = kn / np.sqrt(self.er_eff)
        p_rad = (k0n ** 2 / (32 * np.pi ** 2 * ETA0)) * prad                  # ∝k₀² 解析縮放
        wn = k0n * C0                                                         # ω_n
        q_rad = (wn * (self.er_eff * EPS0 * H / 2) / p_rad.clamp_min(1e-300)).clamp(0.5, 1e5)
        q_sel = 1.0 / (1.0 / q_rad + 1.0 / q_material())
        return torch.full_like(kn2, float(self.q)).scatter(1, pick, q_sel), q_rad, pick

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
