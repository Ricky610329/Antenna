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

from .geom import N, DX, H, EPS_R, Z0, C0, MU0, EPS0, FREQS, FEED_ROW, feed_weights  # noqa: F401

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
            a[k, 0, 0] = 1.0                            # 直接項（介質內近場）
            n[k, 0, 0] = er
            if n_img > 1:
                a[k, 1, 0] = -1.0                       # 地鏡像（反號）
                b[k, 1] = np.log(2 * H)
                n[k, 1, 0] = er
            b[k, 0] = np.log(0.4 * DX)                  # 自項正則化半徑
            #? 第三支刻意用 **n = 1（自由空間）**：輻射是以 k₀ 在空氣中傳的，
            #  兩支都用 n=√εr 的話整個頻譜都壓在介質內 → 輻射電阻系統性偏低 ~10×、
            #  S11 谷深不夠（實測 dev 中位 −2.7dB vs 真值 −12.4dB）。
            #  初值取 0（保住解析校準的共振頻率不被動到）——這一項對 loss 是線性的，
            #  a=0 處梯度仍非零，擬核照樣長得出來。
            for i in range(2, n_img):
                a[k, i, 0] = 0.0
                b[k, i] = np.log(2 * H * i)
                n[k, i, 0] = 1.0 if i == 2 else er
        self.a = torch.nn.Parameter(a)
        self.b = torch.nn.Parameter(b)
        self.n = torch.nn.Parameter(n)

    def table(self, r: torch.Tensor, k0: torch.Tensor) -> torch.Tensor:
        """r (…,) 距離表、k0 (nf,) → (2, nf, …) 的 G_A / G_V 值。"""
        a = torch.complex(self.a[..., 0], self.a[..., 1])          # (2, n_img)
        n = torch.complex(self.n[..., 0], self.n[..., 1])
        #? b 夾在 [0.05dx, 40h]：b 太小 → 自項 1/(4πb) 發散、Z 奇異 → 擬核當場 NaN。
        b = self.b.clamp(np.log(0.05 * DX), np.log(40 * H)).exp()
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
    def _solve_masked(self, Z, wgt, v, thr=0.5):
        """只在金屬屋頂上解（把它們排到前面、取左上子矩陣）。

        二值 pattern 下這與電阻加載等價，但快 ~6×（解算成本 ∝ n³，1249 → ~700）。
        代價：對 ρ 不可微 → 擬核（ρ 固定）與排序用這條；設計優化用 `r_open` 那條。
        """
        nb_batch = wgt.shape[0]
        keep = wgt > thr
        nmax = int(keep.sum(1).max().item())
        order = torch.argsort((~keep).to(self.dtype), dim=1, stable=True)[:, :nmax]   # 金屬在前
        Zs = Z[order[:, :, None], order[:, None, :]]                                  # (B,nmax,nmax)
        alive = torch.gather(keep, 1, order)                                          # (B,nmax)
        eye = torch.eye(nmax, dtype=self.cdtype, device=Z.device)[None]
        Zs = torch.where(alive[:, :, None] & alive[:, None, :], Zs, eye.expand_as(Zs))
        vs = torch.gather(v.expand(nb_batch, -1), 1, order) * alive
        cur = torch.zeros(nb_batch, self.nb, dtype=self.cdtype, device=Z.device)
        return cur.scatter(1, order, torch.linalg.solve(Zs, vs) * alive)

    def solve(self, rho: torch.Tensor, freqs=None, r_open: float = None) -> dict:
        """(B,25,25) → {'S11','Gain'}（dB, 17 點）。

        `r_open` 給值時走「電阻加載」（R = r_open·(1−ρ)/max(ρ,ε)）＝SIMP for conductors，
        對 ρ 可微、給設計優化用；預設 None 走遮罩解，快 ~6×、給擬核與排序用。
        """
        if freqs is None:
            freqs = FREQS
        f = torch.as_tensor(np.asarray(freqs), dtype=self.dtype, device=self.device)
        rho_e = self.extend(rho.to(self.dtype))
        wgt = self.edge_density(rho_e)                          # (B, nb)
        v = torch.zeros(self.nb, dtype=self.cdtype, device=self.device)
        v[self.drive] = DX                                      # delta-gap，V = 1
        cur = []
        for fk in f:
            Z = self.impedance_at(float(fk))
            if r_open is None:
                cur.append(self._solve_masked(Z, wgt, v))
            else:
                load = (r_open * (1.0 - wgt) / wgt.clamp_min(1e-6)).to(self.cdtype)
                cur.append(torch.linalg.solve(Z[None] + torch.diag_embed(load),
                                              v.expand(wgt.shape[0], self.nb)))
        cur = torch.stack(cur, 1)                               # (B, nf, nb)
        itot = (cur[:, :, self.drive].sum(-1) * DX)
        zin = 1.0 / itot
        gam = (zin - Z0) / (zin + Z0)
        #? dB 地板 −50/−40：HFSS 實務範圍就這麼大，不設地板時 |Γ|→0 會產生 −120dB
        #  離群值，擬核時直接把梯度炸成 NaN（實測踩到）。
        s11 = 20 * torch.log10(gam.abs().clamp(3.2e-3, 1.0))
        u0, prad = self.farfield(cur, f)
        d0 = (4 * np.pi * u0 / prad.clamp_min(1e-300)).clamp_min(1e-4)
        mism = (1 - gam.abs() ** 2).clamp_min(1e-6)
        gain = (10 * torch.log10(d0 * mism)).clamp_min(-40.0)
        return dict(S11=s11, Gain=gain, Zin=zin, D0=d0, Prad=prad, J=cur)

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

    def resonances(self, rects, fmin=8e9, fmax=44e9, nf=73) -> np.ndarray:
        """一組矩形貼片 → 各自 Re(Zin) 峰值頻率（Hz）。校準用的觀測量。"""
        f = np.linspace(fmin, fmax, nf)
        p = torch.zeros(len(rects), N, N, dtype=self.dtype, device=self.device)
        for b, (i0, i1, j0, j1) in enumerate(rects):
            p[b, i0:i1, j0:j1] = 1.0
        with torch.no_grad():
            z = self.solve(p, freqs=f)["Zin"].real.cpu().numpy()
        return f[z.argmax(1)]

    def predict(self, patterns, batch: int = 4) -> np.ndarray:
        p = np.asarray(patterns, dtype=np.float64).reshape(-1, N, N)
        out = np.empty((len(p), 34))
        for i in range(0, len(p), batch):
            with torch.no_grad():
                r = self.solve(torch.as_tensor(p[i:i + batch], dtype=self.dtype, device=self.device))
            out[i:i + batch, :17] = r["S11"].cpu().numpy()
            out[i:i + batch, 17:] = r["Gain"].cpu().numpy()
        return out


# ---------------------------------------------------------------- 解析校準（不用 HFSS 資料）
def patch_fr(L: float, W: float) -> float:
    """矩形微帶貼片的閉式共振頻率（Hammerstad εeff + 邊緣延伸 ΔL）。單位 m → Hz。

    校準第一步刻意**不用 HFSS 資料**：先讓求解器的相速度對上已知的微帶物理，
    才輪到用資料擬細節。這樣「擬合」不會變成把物理錯誤藏進參數裡。
    """
    wh = W / H
    ee = (EPS_R + 1) / 2 + (EPS_R - 1) / 2 / np.sqrt(1 + 12 / wh)
    dl = H * 0.412 * (ee + 0.3) * (wh + 0.264) / ((ee - 0.258) * (wh + 0.8))
    return C0 / (2 * (L + 2 * dl) * np.sqrt(ee))


CAL_RECTS = [(25 - n, 25, (N - w) // 2, (N - w) // 2 + w)
             for n, w in ((25, 25), (25, 17), (17, 17), (17, 13), (14, 14), (14, 10), (11, 11))]


def calibrate_analytic(scales=None, n_img: int = 3, verbose: bool = True):
    """掃 G_A 振幅的整體縮放，讓模型共振頻率對上閉式解。回 (最佳 scale, 相對誤差)。

    只有一個純量——初值的 (源 − 地鏡像) 結構本來就對，缺的是 A 項與 V 項的相對權重
    （＝相速度）。`docs/diffsim.md` §3 說的「擬核」從這裡起步，不是從亂數起步。
    """
    tgt = np.array([patch_fr(n * DX, w * DX) for n, w in
                    ((25, 25), (25, 17), (17, 17), (17, 13), (14, 14), (14, 10), (11, 11))])
    best = None
    for s in (scales if scales is not None else np.geomspace(1.0, 8.0, 17)):
        m = MoML2(kernel=DCIMKernel(n_img=n_img))
        with torch.no_grad():
            m.kernel.a.data[0, :, 0] *= s
        got = m.resonances(CAL_RECTS)
        err = float(np.sqrt(np.mean((np.log(got / tgt)) ** 2)))
        if verbose:
            print(f"  a_A×{s:5.2f}: RMS log 誤差 {err:.4f}  f_model={np.round(got / 1e9, 1)}",
                  flush=True)
        if best is None or err < best[1]:
            best = (float(s), err)
    if verbose:
        print(f"  目標 f_analytic = {np.round(tgt / 1e9, 1)} GHz")
        print(f"**最佳 a_A 縮放 = {best[0]:.3f}（RMS log 誤差 {best[1]:.4f} ~ {best[1] * 100:.1f}%）**")
    return best


class LearnedKernel(torch.nn.Module):
    """**核 K 黑盒化**（`docs/diffsim.md` §3 方案 c／§4 可置換節點表第一項）。

    設計是**殘差式**，不是整個丟掉物理：

        G(r, k) = G_DCIM(r, k) · (1 + net(feat)) + net_add(feat) / (4π√(r²+b²))

    - 保留 DCIM 骨架 → 1/r 奇異性、e^{−jkr} 相位、源−鏡像相消結構全部免學。
    - `net` 最後一層權重初始化為 **0** → 初始輸出恆等於 DCIM，**擬合從現況起步**、
      梯度天然流動，不會一開始就把已經對的共振頻率打壞。
    - 特徵只用 (k₀r, log r, k₀h) 三個**無因次量** → 天然對 r/頻率平滑，
      而且與樣本無關 → 每個 batch 只算一次表，成本可忽略。

    記帳（指導書 §4 要求「哪一節是學的要記帳」）：物理節＝DCIM 骨架與相消結構；
    學習節＝距離相依的複數修正。這一節換掉之後，「共振頻率對閉式解 1.6%」這條
    保證就**不再自動成立**，要重新量。
    """

    def __init__(self, base: DCIMKernel = None, width: int = 64, dtype=torch.float64):
        super().__init__()
        self.base = base or DCIMKernel(dtype=dtype)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, width), torch.nn.SiLU(),
            torch.nn.Linear(width, width), torch.nn.SiLU(),
            torch.nn.Linear(width, 8),          # 2 核 × (乘性 re,im) + 2 核 × (加性 re,im)
        )
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)
        self.to(dtype)

    def table(self, r: torch.Tensor, k0: torch.Tensor) -> torch.Tensor:
        g = self.base.table(r, k0)                                  # (2, nf, ntab)
        kr = k0[:, None] * r[None, :]                               # (nf, ntab)
        feat = torch.stack([kr,
                            torch.log(r[None, :].expand_as(kr) / DX + 1e-3),
                            (k0[:, None] * H).expand_as(kr)], -1)
        o = self.net(feat)                                          # (nf, ntab, 8)
        mul = 1.0 + torch.complex(o[..., 0:2], o[..., 2:4]).permute(2, 0, 1)
        add = torch.complex(o[..., 4:6], o[..., 6:8]).permute(2, 0, 1)
        reg = (r ** 2 + (0.4 * DX) ** 2).sqrt()[None, None, :]
        return g * mul + add / (4 * np.pi * reg)
