# -*- coding: utf-8 -*-
"""script/diffsim/l2.py — L2 空間域 MoM（rooftop 基底 + 可擬合的分層 Green's function）。

`docs/diffsim.md` §3 L2 的甜蜜點方案 (b)：**求解器本身可微，所以「擬核」＝對解算器反傳**，
不用另外寫擬合器。相對 L1 的關鍵差別——**不假設「特徵尺寸 ≫ 基板厚」**：
L1 的硬磁牆理想化在 0.2mm 像素 / 0.508mm 基板上本來就破掉（analysis-08 §3.3 診斷），
MoM 直接解真實表面電流，這個假設整個不存在。

## 未知數與矩陣

均勻格 rooftop：x 向屋頂在 (i,j)-(i+1,j) 面上、y 向在 (i,j)-(i,j+1) 面上。

## 埠（★ 2026-08-03 改，analysis-10 §37）

舊做法把 delta-gap 直接壓在貼片 x = 5.0mm 的邊上，理由是「饋線是 51Ω 匹配線、
只轉相位不改 |S11|」。**那句話對，但推論錯**：饋線同時定義了**埠的場結構**，
壓在 0.2mm 一格上的集總源不是 TEM 入射波，對同一個模態振幅注入 **2.2 倍**電流。
現在 `feed_len > 0` 把真饋線建進格網、源移到遠端，`S11` 由**駐波法**萃取
（`gamma_from_line`）——就是 HFSS wave port 的做法，不需要 `Z_c` 也不需要 de-embed。
設定組合登記在 `SOLVERS`，別散在呼叫端（本輪為此踩過四次）。

混合位（MPIE，Michalski–Mosig formulation C，G_A^xy = 0）：

    Z_mn = jωμ·dx⁴·G_A(Δ)·δ_同向  +  (dx²/jωε)·Σ_{p,q} s_p s_q G_V(c_p − c_q)

兩項都只查「格心偏移」的表 → 平移不變 → 一張 (2·NR−1)×(2·NC−1) 的小表就夠。

## 核的參數化：離散複數鏡像（DCIM）

分層介質的 Green's function 標準做法，不是隨便湊的函數族：

    G(r) = Σ_i a_i · exp(−j k₀ n_i √(r² + b_i²)) / (4π √(r² + b_i²))

物理初值（接地介質板的古典近似）＝「源 − 地鏡像」兩項：a=(+1, −1)、b=(b_reg, 2h)、
**n=(1, 1)＝空氣波數**（★ 2026-08-03 修正：舊版用 √εr，等於在描述「浸在無限介質裡的地板」，
造成 ~15× 偽損耗、能量守恆 η 只有 0.066，詳見 `DCIMKernel.__init__`）。
介質的效果改由 **v_scale 除在 G_V** 上表達（相速相同、不放大偽輻射）。
b_reg ⚠ **是自由參數不是「正則化」**——實測 Im(Zin) 隨它變 4.6 倍。
之後 (a_i, b_i, n_i) 全部當複數參數，用可微鏈在 `fit` 分割上端到端擬。
"""
import contextlib

import numpy as np
import torch

from .geom import (N, DX, H, EPS_R, Z0, C0, MU0, EPS0, ETA0, FREQS, FEED_ROW,  # noqa: F401
                   feed_weights, line_cols, microstrip_eeff)

@contextlib.contextmanager
def _mkl_single_thread():
    """把 `torch.linalg.solve` 圈在單執行緒裡跑。

    #! 2026-08-03 實測：**本機 MKL 的 batched complex128 LU 在 `num_threads > 1` 時會壞**——
    #  底層印 `oneMKL ERROR: Parameter 6 was incorrect on entry to ZLASWP`，
    #  然後 torch 拋 `Pivots given to lu_solve must all be >= 1`。同一批資料
    #  `num_threads=1` 完全正常、`=6` 必炸（`scratchpad/` 有最小重現）。
    #  **這不是奇異矩陣**（我一度誤判成環流零空間/孤島，`solve_ex`+`pinv` 也攔不住，
    #  因為例外是在 `lu_solve` 的輸入檢查階段就拋的，根本沒回傳 info）。
    #  獨立確認：delta-gap 稽核 agent 在完全不同的實驗裡撞到同一件事。
    #  只圈 solve 不動全域，其餘運算照樣多執行緒；`set_num_threads` 本身極輕量。
    """
    n = torch.get_num_threads()
    if n > 1:
        torch.set_num_threads(1)
    try:
        yield
    finally:
        if n > 1:
            torch.set_num_threads(n)


NSTUB = 1                     # 集總埠模式的延伸列數（讓 x=5.0mm 接面上的屋頂有立足點）
NR = N + NSTUB                # 延伸格列數（x 方向）——**集總埠模式**的預設值
NC = N                        # 行數（y 方向）
LINE_COLS = line_cols()       # 饋線本體佔的格欄（幾何唯一真相在 `geom.line_cols`）


class DCIMKernel(torch.nn.Module):
    """G_A / G_V 各一組複數鏡像。參數是 log 空間的 b（保正）與自由複數 a、n。"""

    def __init__(self, n_img: int = 3, v_scale: float = 1.0, dtype=torch.float64):
        """
        :param v_scale: 相速校準——**除在 G_V 上**（不是乘在 G_A 上）。

        ★ 2026-08-03 修正（fan-out 稽核，兩支 agent 獨立量到）。舊版把 n=√εr 放進
        **兩支鏡像的指數**，那描述的不是「接地薄板 + 上方空氣」，而是
        **「浸在無限大 εr=3.55 介質裡的 PEC 地板」**——會往介質裡狂輻射。
        相消殘項 ∝ n·[1 − sinc(n·k₀·2h)]，n=1.884 對 n=1 是 6.4×（≈n³）；
        再乘上舊校準為了對共振頻率而把 a_A 放大的 2.9× → **偽損耗 ≈ 19×**。
        實測能量守恆 η = P_rad/P_in（無耗 MoM 應 ≈1）：舊版 **0.066**、n=1 版 **0.9999**
        ——93% 的輸入功率被 Z 矩陣吃掉、從沒進遠場。

        修法是同一個相速縮放**換一邊放**：v = 1/√(L'C')，把 s 除在 C'（G_V）上與
        乘在 L'（G_A）上給**完全相同的相速**，但不放大 A 項的偽輻射。
        """
        super().__init__()
        self.n_img = n_img
        a = torch.zeros(2, n_img, 2, dtype=dtype)      # [kernel, image, (re, im)]
        b = torch.zeros(2, n_img, dtype=dtype)
        n = torch.zeros(2, n_img, 2, dtype=dtype)
        amp = (1.0, 1.0 / max(v_scale, 1e-6))          # (G_A, G_V)：相速校準除在 G_V
        for k in range(2):
            a[k, 0, 0] = amp[k]                         # 直接項
            if n_img > 1:
                a[k, 1, 0] = -amp[k]                    # 地鏡像（反號）→ 相消殘項＝輻射
                b[k, 1] = np.log(2 * H)
            #? 相消一律在 **k₀（空氣）** 評估——這是能量守恆成立的前提（見上）。
            n[k, :, 0] = 1.0
            b[k, 0] = np.log(0.4 * DX)                  # ⚠ 是自由參數，不是「正則化」
            #? 額外支預設惰性（a=0）。⚠ 舊註解宣稱「第三支用 n=1 能救輻射電阻」，
            #  **兩支 agent 各自實測否證**（a₃ 加大 → S11 反而更淺、η 更低）：
            #  加一支未相消的單極比正確的相消殘項大 ~20×，增加的是**輸入功率**不是遠場功率
            #  ——等於往漏水的桶子再鑽一個洞。
            for i in range(2, n_img):
                b[k, i] = np.log(2 * H * i)
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
                 n_theta: int = 13, n_phi: int = 24, half_port: bool = False,
                 layered_ff: bool = False, ff_er: float = None, feed_len: int = 0,
                 zc_ref: float = None, ff_line: bool = True, diag: bool = False):
        """
        :param feed_len: **饋線列數**（0 = 集總埠，舊行為）。>0 就把真實微帶饋線建進格網、
            delta-gap 移到饋線遠端、`S11` 改用**駐波法**萃取（見 `gamma_from_line`）。
        :param zc_ref: 饋線模式下把 Γ 從「線自己的 Z_c」重歸一化到 50Ω 用的 Z_c（Ω）。
            ★ **正常路徑留 None**——模型饋線實測 `Z_c ≈ 50 Ω`（analysis-10 §40：線中 EMF 法
            50.65、靜電電容法 49.07，兩種線寬互相驗證 3–4%），≈ 真實 1.1mm 線的 51.0 Ω
            ≈ 埠 50 Ω ⇒ **不需要重歸一化**（殘差 1.2%）。這個參數只留給消融實驗。
            ⚠ 若真要用，**別拿 `solve()['Zc']` 的值**——那條含 1.3–1.4× 灌水（見 `solve`）。

        ★ 為什麼要建饋線（analysis-10 §37，2026-08-03）：把 delta-gap 直接壓在貼片邊
        0.2mm 的一格上，對**同一個模態振幅注入 2.2 倍電流** ⇒ `Re(Zin)` 低 ~2 倍。
        無因次證據：`I_feed/I_max` 打邊 ÷ 接線 = **埠因子 1.30–1.60**
        （⚠ §41.1 修正——原寫 1.71 是對兩邊用了不同的頻率準則；方向保留、「硬下界」撤回）。
        ⭐ **更硬的一條是共振頻率**（外部錨、與 Balanis 無關）：對閉式解 `l3` −7~−8%、
        `l3fl` **−1.5~−3%**，而且 fit 分割上的谷位置 ρ 也跟著漲（clean +0.234 → +0.345）。
        而 HFSS 那邊埠在 22.5mm 饋線的遠端（`single_port.py:374-407`）—— 貼片被真實
        TEM 行進波餵，`DoDeembed` 只搬參考面（相位），不會把它變回集總源。
        ⚠ 舊的「1/√N 離散化天花板」結論（§19）是統計假象，已撤回：真網格收斂
        （固定貼片改 dx、未知數 ×8.7）比值只 0.369 → 0.387。
        """
        self.feed_len = int(feed_len)
        self.diag = bool(diag)
        self.zc_ref = zc_ref
        #! `ff_line=False`：**饋線電流不進遠場**。理由不是調參，是我們的線長度是假的——
        #  模型 9mm、真實 22.5mm ⇒ 它在遠場的干涉結構**本來就是虛構的**；
        #  而且它的電流**不隨貼片面積變**，正好稀釋掉真值裡最強的那個機制。
        #  實測（analysis-10 §43，fit 分割 n=400、判準先宣告）：
        #    `ρ(高頻懸崖, 金屬面積)` **−0.056 → +0.218**（真值 +0.313）＝機制朝真值走；
        #    top-60 命中率 13.3% → 21.7%，**Δ+8.28%、P(>0)=99.2%、P(<0)=0.0%**。
        #  ⇒ **機制與用途一起動**，與本輪五次「指標漲用途沒漲」不同。
        self.ff_line = ff_line
        self.nc = NC
        self.nr = N + (self.feed_len if self.feed_len > 0 else NSTUB)
        self.half_port = half_port          # 半屋頂埠，見 `impedance_at`；用 L3 核時建議開
        if self.feed_len > 0 and half_port:
            #! 半屋頂埠是「沒有饋線」的補償（假設電荷流進未建模的線）。線建進來之後
            #  驅動屋頂兩側都是真金屬，再扣掉一側的電荷偶極就是**憑空少算電荷**。
            raise ValueError("feed_len > 0 與 half_port 互斥——半屋頂埠是沒有饋線時的補償")
        self.layered_ff = layered_ff        # 分層遠場因子，見 `_ff_factors`；用 L3 核時**必開**
        #! 遠場的 εr **必須與核的 εr 一致**，否則能量守恆會壞（實測不匹配時 η 可 >1）。
        #  預設 `geom.EPS_R`＝真實板材；只有掃 εr 的驗證實驗需要覆寫。
        self.ff_er = EPS_R if ff_er is None else ff_er
        self.device = torch.device(device)
        self.dtype = dtype
        self.cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        self.kernel = (kernel or DCIMKernel(dtype=dtype)).to(self.device)
        self._build_topology()
        self._setup_farfield(n_theta, n_phi)

    # ----------------------------------------------------------------- 拓樸
    def _build_topology(self):
        nr, nc = self.nr, self.nc
        cell = np.arange(nr * nc).reshape(nr, nc)
        # x 屋頂：面在 (i,j)-(i+1,j)；y 屋頂：面在 (i,j)-(i,j+1)
        xa, xb = cell[:-1, :].ravel(), cell[1:, :].ravel()
        ya, yb = cell[:, :-1].ravel(), cell[:, 1:].ravel()
        ca = np.concatenate([xa, ya])
        cb = np.concatenate([xb, yb])
        #? **電流矩向量** M = 電荷 × 分離向量（單位 dx²）。軸向屋頂是 (1,0)/(0,1)；
        #  對角是 (1,±1)。A 項因此從「同向才有」推廣成 `M_m · M_n`（見 `impedance_at`）。
        mx = np.concatenate([np.ones(len(xa)), np.zeros(len(ya))])
        my = np.concatenate([np.zeros(len(xa)), np.ones(len(ya))])
        cc1 = np.full(len(ca), -1, dtype=np.int64)     # 對角的兩個「角落格」（見 `edge_density`）
        cc2 = cc1.copy()
        if self.diag:
            #! 對角基底（analysis-10 §45）：HFSS 的像素盒是 `pixel + 0.01mm`
            #  ⇒ **對角相鄰的像素在角落重疊 0.01×0.01mm、Unite 後是同一塊導體**，
            #  而 rooftop 基底只有邊相鄰 ⇒ 全史 57.1% 的樣本幾何與 HFSS 不符。
            #  主對角 (i,j)-(i+1,j+1) 與反對角 (i,j+1)-(i+1,j) **都連通**（兩者重疊都是 0.01×0.01）。
            d1a, d1b = cell[:-1, :-1].ravel(), cell[1:, 1:].ravel()     # 主對角 M=(1,+1)
            d2a, d2b = cell[:-1, 1:].ravel(), cell[1:, :-1].ravel()     # 反對角 M=(1,−1)
            #? 角落格＝該對角的兩個「繞路」鄰居；兩者都是 void 時對角才是唯一通路（見 `edge_density`）。
            #! ★ 2026-08-04 修：**主對角與反對角的繞路鄰居不一樣**，原本兩者共用同一組
            #  ⇒ 反對角拿到的「角落格」**就是它自己的兩個端點**
            #  ⇒ `wgt = ρa·ρb·(1−ρb)·(1−ρa) ≡ 0` ⇒ **757 條反對角基底全史零活化**。
            #  而真實的角落橋裡**反對角佔 47.5%** ⇒ 等於只做了一半。
            #  （對抗式複核 agent 抓到，並用「只有反對角橋」的子集當**天然安慰劑**證實：
            #   主對角組 Δρ +0.2707 P=100%，反對角組 Δρ +0.0003 P=34.7%。）
            #  主對角 (i,j)-(i+1,j+1) 的繞路是 (i+1,j) 與 (i,j+1)；
            #  反對角 (i,j+1)-(i+1,j) 的繞路是 (i,j)＝a−1 與 (i+1,j+1)＝b+1。
            c1a, c1b = cell[1:, :-1].ravel(), cell[:-1, 1:].ravel()
            ca = np.concatenate([ca, d1a, d2a])
            cb = np.concatenate([cb, d1b, d2b])
            nd = len(d1a)
            mx = np.concatenate([mx, np.ones(2 * nd)])
            my = np.concatenate([my, np.ones(nd), -np.ones(nd)])
            cc1 = np.concatenate([cc1, c1a, d2a - 1])
            cc2 = np.concatenate([cc2, c1b, d2b + 1])
        isx = mx.astype(bool) & (my == 0)             # 純 x 向（線探針與舊行為用）
        #? 延伸區只有饋線那幾欄可能是金屬 → 其餘的屋頂**永遠**被遮罩掉，直接不要生成。
        #  （集總埠模式 live 全 True ⇒ 與舊版逐位相同，golden 不動。）
        live = np.ones((nr, nc), bool)
        if self.feed_len > 0:
            live[N:, :] = False
            live[N:, LINE_COLS] = True
        lv = live.ravel()
        keep = lv[ca] & lv[cb]
        row_a, col_a = np.divmod(ca, nc)
        if self.feed_len > 0:
            #? delta-gap 打在**饋線遠端**最外側的 x 屋頂（列 nr-2 | nr-1）。
            #  末端那一格金屬 = 0.2mm 開路殘段，它的假影全留在源附近，
            #  而 Γ 是在遠離源的窗裡從駐波萃取的 ⇒ 影響為零。
            drv = isx & (row_a == nr - 2) & np.isin(col_a, LINE_COLS)
        else:
            drv = isx & (row_a == FEED_ROW) & (feed_weights() > 0)[col_a]
        ca, cb, isx, drv = ca[keep], cb[keep], isx[keep], drv[keep]
        mx, my, cc1, cc2 = mx[keep], my[keep], cc1[keep], cc2[keep]
        self.nx = int(isx.sum())
        self.ny = int((~isx).sum())
        self.nb = len(ca)
        self.cell_a = torch.as_tensor(ca, dtype=torch.long, device=self.device)
        self.cell_b = torch.as_tensor(cb, dtype=torch.long, device=self.device)
        self.is_x = torch.as_tensor(isx, dtype=torch.bool, device=self.device)
        self.mx = torch.as_tensor(mx, dtype=self.dtype, device=self.device)
        self.my = torch.as_tensor(my, dtype=self.dtype, device=self.device)
        self.is_diag = torch.as_tensor(cc1 >= 0, device=self.device)
        self.corner1 = torch.as_tensor(np.maximum(cc1, 0), dtype=torch.long, device=self.device)
        self.corner2 = torch.as_tensor(np.maximum(cc2, 0), dtype=torch.long, device=self.device)
        ci, cj = np.divmod(np.arange(nr * nc), nc)
        self.ci = torch.as_tensor(ci, dtype=torch.long)
        self.cj = torch.as_tensor(cj, dtype=torch.long)
        # 格心偏移表索引（平移不變 → 只查 (di+nr-1, dj+nc-1)）
        di = ci[:, None] - ci[None, :] + (nr - 1)
        dj = cj[:, None] - cj[None, :] + (nc - 1)
        self.off = torch.as_tensor(di * (2 * nc - 1) + dj, dtype=torch.long).to(self.device)
        gi, gj = np.meshgrid(np.arange(-(nr - 1), nr), np.arange(-(nc - 1), nc), indexing="ij")
        self.rtab = torch.as_tensor(DX * np.sqrt(gi ** 2 + gj ** 2).ravel(),
                                    dtype=self.dtype, device=self.device)
        self.drive = torch.as_tensor(drv, device=self.device)
        #? 貼片基底＝兩端點都在貼片列的；接面屋頂（列 N−1|N）算貼片側，它帶的是貼片電流。
        self.is_patch = torch.as_tensor(row_a[keep] < N, device=self.device)
        self.sb = (~self.drive).to(self.dtype)      # 電荷偶極的 b 側權重（半屋頂埠時驅動屋頂為 0）
        if self.feed_len > 0:
            self._build_line_probe(ca, isx, row_a[keep])

    def _build_line_probe(self, ca, isx, row_a):
        """駐波量測的取樣矩陣 (nb, n_face)：第 m 欄挑出「饋線第 m 個橫截面」的所有 x 屋頂。

        面 m 位於列 (N−1+m) 與 (N+m) 之間，m = 0 就是**貼片/饋線接面**（參考面）。
        該截面的總 x 電流 `I_m = dx·Σ_j J_x` —— 這就是傳輸線上的電流波。
        """
        nface = self.nr - N          # m = 0 … nr-1-N（最後一個是驅動面）
        sel = np.zeros((self.nb, nface), dtype=np.float64)
        rel = row_a - (N - 1)
        #? 跨越該截面的**全部** x 向電流＝x 屋頂 + 對角（對角的 mx 也是 1）。
        #  舊版只挑 x 屋頂；開 `diag` 後不含對角會低估截面電流 ⇒ Γ 偏掉。
        mxv = self.mx.cpu().numpy()
        ok = (mxv != 0) & (rel >= 0) & (rel < nface)
        sel[np.nonzero(ok)[0], rel[ok]] = mxv[ok]
        del isx
        self.line_sel = torch.as_tensor(sel, dtype=self.dtype, device=self.device)
        self.n_face = nface
        #? 線的相位常數初估（Hammerstad）：用來挑駐波遞迴的步距 p，讓 p·β·dx ≈ π/2
        #  ——差分在 π/2 附近條件數最好（β·dx 只有 0.19 rad，一步遞迴會被放大 ~5×）。
        self.line_eeff = microstrip_eeff(len(LINE_COLS) * DX)
        del ca

    def edge_density(self, rho_ext: torch.Tensor) -> torch.Tensor:
        """(B, nr, nc) → (B, nb)：屋頂存在權重 ρ_a·ρ_b（連續鬆弛天然可微）。"""
        r = rho_ext.reshape(rho_ext.shape[0], -1)
        w = r[:, self.cell_a] * r[:, self.cell_b]
        if self.diag:
            #! ★ 對角**只在它是唯一通路時**啟用：兩個「角落格」都是 void 才算。
            #  若角落有金屬，電流走那條 200µm 寬的路，10µm 的角落橋是並聯的高阻抗支路
            #  ⇒ 可忽略。這讓對角基底在實心區**完全不活化**，成本幾乎為零，
            #  而在真正需要它的破碎區才進來。（基底集合仍是**樣本無關**的 ⇒ Z 矩陣照樣共用。）
            gate = (1.0 - r[:, self.corner1]) * (1.0 - r[:, self.corner2])
            w = torch.where(self.is_diag[None, :], w * gate, w)
        if self.feed_len > 0:
            return w                       # 饋線是真金屬 ⇒ 接面屋頂照一般規則，不用特例
        #? 集總埠模式：驅動屋頂跨 x=5.0mm 接面，`cell_b` 在延伸列（永遠 void）。
        #  外側是**未建模的饋線本體**、永遠是金屬 → 存在權重只看貼片側那一格，
        #  否則驅動屋頂 wgt ≡ 0 → 全開路。
        return torch.where(self.drive[None, :], r[:, self.cell_a], w)

    def ext_mask(self) -> np.ndarray:
        """延伸列的金屬遮罩 (nr−N, nc)。集總埠模式全 0；饋線模式在 `LINE_COLS` 放金屬。

        #! 集總埠模式**不放金屬**（2026-08-03 修正，原本在 col 9–15 放 0.2mm 死端 stub）。
        #  那是 22.5mm 連續饋線的**壞近似**——憑空多一條寄生邊緣，而真實饋線那一側是
        #  延續下去的傳輸線、不是開路端。實測（稽核 agent，dev n=240，配對 bootstrap）：
        #  Δρ(m_s11) = **+0.051**，95% CI [+0.014, +0.089]，P(>0)=**99.6%**，且快 5%。
        #  ⇒ 這條教訓的正解就是 `feed_len > 0`：與其近似饋線，不如把它建出來。
        """
        m = np.zeros((self.nr - N, self.nc), dtype=np.float64)
        if self.feed_len > 0:
            m[:, LINE_COLS] = 1.0
        return m

    def extend(self, rho: torch.Tensor) -> torch.Tensor:
        """(B,25,25) 貼片 → (B,nr,nc) 含延伸列。"""
        st = torch.as_tensor(self.ext_mask(), dtype=rho.dtype, device=rho.device)
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
        #? A 項的方向因子＝**電流矩向量的內積** `M_m·M_n`（單位 dx²）。
        #  軸向對軸向退化回舊的 `same`（同向 1、異向 0）；對角自然帶 √2 的長度與 ±45° 的方向。
        same = (self.mx[:, None] * self.mx[None, :] + self.my[:, None] * self.my[None, :])
        a, b = self.cell_a, self.cell_b
        #! 兩項的 dx 冪次不同，寫錯就整條物理翻掉（實測踩過：A 項誤用 dx² 讓電感項比電容項
        #  大 2.5e5 倍 → 完全不是準靜態，Zin 直接 4e7Ω）。∫f_m dS = dx²（屋頂：三角×脈衝），
        #  故 A 項 ∫∫f_m f_n G_A ≈ dx⁴·G_A；而 ∫∇·f_m over cell = ±dx，故 V 項是 dx²。
        #  兩項都是 Ω·m²；比值 zv/za = 1/(k²dx²·G_A/G_VV) ≈ 99 → 格尺度上電容主導 ✓ 準靜態。
        #! ★ 2026-08-04 修：A 項原本對**每條基底**都拿 `cell_a` 當求積點。軸向屋頂的形心
        #  離 `cell_a` 是 (0.5,0) 或 (0,0.5)·dx，**對角是 (0.5,0.5)·dx** ⇒ 不一致。
        #  在 `l3fl` 裡那個不一致**剛好完全抵銷**（x–y 的 `same`=0、同型內偏移相等），
        #  是 `M_m·M_n` 讓「對角×y 屋頂」不再是 0 才把它曝露出來。
        #  症狀：幾何對 `j→24−j` 精確對稱，但 `l3fl` 鏡像後 max|ΔS11| = **0.0000 dB**、
        #  `l3fld` 是 **18.55 dB**（對抗式複核 agent 用這條**不依賴真值的硬不變式**定位）。
        #  修法：四點平均 `¼[G(a,a)+G(a,b)+G(b,a)+G(b,b)]` —— 它以形心為中心、
        #  且對 a↔b 對調**協變** ⇒ 鏡像對稱自動成立。V 項本來就是四點，形式一致。
        gaa = 0.25 * (ga[a][:, a] + ga[a][:, b] + ga[b][:, a] + ga[b][:, b])
        za = (1j * w * MU0 * DX ** 4) * gaa * same              # A 項：方向因子 M_m·M_n
        if self.half_port:
            #? 半屋頂埠：驅動屋頂跨 x=5.0mm 接面，外側是**連續饋線**——電荷流走、不在那裡累積，
            #  所以它的電荷偶極只保留貼片側（s_b = 0）；其餘屋頂照常（s_b = 1）。
            #  平面 MoM 的標準埠模型。實測（L3 核，`clean_OOS` 817 筆，2026-08-03）：
            #    谷位置 ρ **+0.052 → +0.188**（×3.6）、同格比例 5%→15%、|Δ格| 中位 4.0→3.0
            #    Im(Zin) 中位 **−50.9 → −8.9 Ω**、Re(Zin) p90 28.2 → 96.5 Ω
            #    （min-S11 = −12.4 dB 反推的純電阻解是 30.7 或 81.6 Ω ⇒ 量級變合理）
            #  ⚠⚠ **只在 L3 核上開**（`half_port` 預設 False 就是為此）。實測：
            #    DCIM 核 + half_port : min Re **−0.289**、η 中位 **1.369**、η>1 佔 **100%** ← 非物理
            #    L3   核 + half_port : min Re **+2.064**、η>1 佔 **0%**                     ← 正常
            #  稽核 agent 當初警告它會破 passivity/能量守恆，**在 DCIM 上是對的**，它只是沒測 L3。
            #  依賴關係由 `test_l2_half_port_kernel_dependency` 釘住。
            sb = self.sb.to(g.dtype)
            gvv = (gv[a][:, a] - sb[None, :] * gv[a][:, b]
                   - sb[:, None] * gv[b][:, a] + (sb[:, None] * sb[None, :]) * gv[b][:, b])
        else:
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
        if nmax == 0:                       # 整批都沒有金屬 → 電流恆為 0
            return cur
        with _mkl_single_thread():
            sol = torch.linalg.solve(Zs, vs)
        return cur.scatter(1, order, sol * alive)

    # ------------------------------------------------------------- 駐波埠（wave port）
    def gamma_from_line(self, cur: torch.Tensor, f: torch.Tensor,
                        skip_src: int = 5, skip_ref: int = 3, lossless: bool = True):
        """沿饋線的駐波 → 反射係數 Γ。這是 **HFSS wave port 的做法**，不是 de-embed。

        兩行進波模型（`u = m·dx`，`m=0` 在貼片/饋線接面＝參考面，u 往源為正）：

            I(u) = a·w^m + b·w^(−m),    w = e^{+jβ·dx}

        入射波由源（大 m）往貼片走，它在 +u 方向的電流是 `−V_i/Z_c`；反射波是 `+V_r/Z_c`
        ⇒ **Γ = V_r/V_i = −b/a**。

        為什麼**不**用 de-embed（analysis-10 §37.7/§37.8）：de-embed 需要模型自己的 `Z_c`
        （實測比 Hammerstad 低 13–26%），而且 `|Γ|≈0.8` 時 `(1+Γ)/(1−Γ)` 極敏感
        —— agent 量到 ±20% 的線長散佈（3/4/6/9/12mm → 297/359/385/529/451 Ω）。
        駐波法**不需要 `Z_c`、不需要參考面距離**，且 `|Γ|` 與參考面無關（只有相位會轉），
        而資料集只用 `dB(S11)`。

        兩段都是線性最小平方、全程可微：
        1. **相位常數**：兩行進波滿足 `I_{m+p} + I_{m−p} = 2cos(pβΔ)·I_m`
           ⇒ 對 `cos(pβΔ)` 的最小平方解。步距 `p` 取 `p·β·dx ≈ π/2`
           （`β·dx` 只有 0.19 rad，一步遞迴的誤差會被 `c/√(1−c²)` 放大 ~5×）。
        2. **振幅**：在 `(w^m, w^−m)` 基底上解 2×2 正規方程。

        :param lossless: 強制 `|w| = 1`（把線當無耗）。
            ⚠⚠ **這是本萃取器最大的一個未定案敏感度**（analysis-10 §41.2 A，對抗式複核找到，
            而 §38.2 的五條自證**沒有一條測到它**）。把 `|w|` 解出來：線確實幾乎無耗
            （0.02–0.43 dB / 9mm），但不是零 ⇒ **強制會讓 `|Γ|` 偏 0.2–4.6%、`Re(Zin)` 偏 ~19%**。
            但**放開也不對**：抽出來的衰減**隨負載變 24 倍** ⇒ 它吃到的不只是線衰減，
            放開等於把一個負載相依的假影灌進 Γ。
            ⇒ **兩個選項都不明顯正確**；預設維持 `True`（§38 的所有數字都是這個），
            要換必須在 `fit` 分割上比、判準先寫死。
        :return: (Γ, β, a, b)。`a`/`b` 是入射/反射的**電流波振幅（安培）**。
        """
        eps = torch.finfo(self.dtype).tiny
        im = torch.einsum("bfp,pm->bfm", cur, self.line_sel.to(cur.dtype)) * DX
        m0, m1 = skip_ref, self.n_face - skip_src        # 擬合窗 [m0, m1)：兩端都讓開
        beta0 = float((2 * np.pi * f.mean() / C0) * np.sqrt(self.line_eeff))
        p = max(1, min(int(round((np.pi / 2) / (beta0 * DX))), (m1 - m0 - 1) // 2))
        if m1 - m0 < 2 * p + 2:
            raise ValueError(f"饋線太短：擬合窗 {m1 - m0} 個面 < 遞迴步距 2p+2 = {2 * p + 2}"
                             f"（feed_len={self.feed_len}，至少要 ~{2 * p + 2 + skip_src + skip_ref + N - N}）")
        ic = im[..., m0 + p:m1 - p]
        num = (ic.conj() * (im[..., m0 + 2 * p:m1] + im[..., m0:m1 - 2 * p])).sum(-1)
        den = 2 * (ic.abs() ** 2).sum(-1)
        c = num / (den + eps)
        d = torch.sqrt(c * c - 1)
        wp = torch.where((c + d).imag < 0, c - d, c + d)        # 取 Im>0 那支＝往源傳的方向
        ang = (torch.angle(wp) / p).clamp(1e-9, np.pi - 1e-9)
        #! 步距 `p` 是用 **Hammerstad 估的 β** 挑的。若真實 β 與估計差太多，
        #  `p·β·dx` 會越過 π ⇒ `arccos` 繞圈 ⇒ **β 與 Γ 靜默錯掉**（實測估計差 2.5× 時
        #  Γ 誤差 0.46）。目前實測比值 1.046–1.068，安全——但「目前安全」不是不變式。
        #  （bug 獵捕 agent 2026-08-03 標的 S2。）
        ratio = float((ang.max() / (beta0 * DX)).item())
        if not 0.5 < ratio < 1.9:
            raise ValueError(f"駐波萃取的 β 與 Hammerstad 估計差 {ratio:.2f}×——"
                             f"步距 p={p} 已失效，Γ 不可信（改小 p 或檢查饋線幾何）")
        mv = torch.arange(m0, m1, dtype=self.dtype, device=cur.device)
        ph = ang[..., None] * mv
        one = torch.ones_like(ph)
        if lossless:
            wpos, wneg = torch.polar(one, ph), torch.polar(one, -ph)
        else:
            #? 放開 |w|：把每格衰減也解出來。⚠ 見 `lossless` 的說明——它不只吃到線衰減。
            lg = (torch.log(wp.abs().clamp_min(1e-12)) / p)[..., None] * mv
            wpos, wneg = torch.polar(lg.exp(), ph), torch.polar((-lg).exp(), -ph)
        y = im[..., m0:m1]
        g11 = (wpos.conj() * wpos).sum(-1)
        g12 = (wpos.conj() * wneg).sum(-1)
        g22 = (wneg.conj() * wneg).sum(-1)
        r1 = (wpos.conj() * y).sum(-1)
        r2 = (wneg.conj() * y).sum(-1)
        det = g11 * g22 - g12 * g12.conj()
        a = (g22 * r1 - g12 * r2) / (det + eps)
        b = (g11 * r2 - g12.conj() * r1) / (det + eps)
        gam = -b / (a + eps)
        #? |Γ|>1 是非物理（被動負載）。會發生在整批無金屬等退化情形；夾住而不是丟 NaN。
        gam = torch.where(gam.abs() > 1.0, gam / gam.abs().clamp_min(eps), gam)
        return gam, ang / DX, a, b

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
                with _mkl_single_thread():           # 同 `_solve_masked`：MKL 多執行緒複數 LU 會壞
                    cur.append(torch.linalg.solve(Z[None] + torch.diag_embed(load),
                                                  v.expand(wgt.shape[0], self.nb)))
        cur = torch.stack(cur, 1)                               # (B, nf, nb)
        itot = (cur[:, :, self.drive].sum(-1) * DX)
        #! 饋線接觸格全 void → itot 恆為 0 → 1/0 → 整筆 NaN。這是**確定性的除以零**，
        #  不是「矩陣接近奇異」（實測 cond 最大只有 3e3）。val 120 筆踩到 1 筆。
        #  跟 L1 一樣回「開路＝全反射」，別讓 NaN 傳出去。
        #  （饋線模式下不會發生：線本身永遠是金屬，源永遠有電流。）
        open_ckt = itot.abs() < 1e-30
        itot = torch.where(open_ckt, torch.full_like(itot, 1e-30), itot)
        zin_drv = 1.0 / itot                     # 源點阻抗（V=1 的 delta-gap）
        extra = {}
        if self.feed_len > 0:
            gam, beta, wa, wb = self.gamma_from_line(cur, f)
            #? Γ 是對**線自己的 Z_c** 定義的。`zc_ref` 給值就重歸一化到 Z₀=50Ω
            #  （＝ HFSS `RenormImp:="50ohm"` 做的事）。不給就當線是 50Ω。
            if self.zc_ref is not None:
                zl = self.zc_ref * (1 + gam) / (1 - gam + 1e-300)
                gam = (zl - Z0) / (zl + Z0)
            zin = Z0 * (1 + gam) / (1 - gam + 1e-300)   # 50Ω 參考的等效輸入阻抗
            #! ⚠⚠ `Zc` **僅供診斷，已知含 1.3–1.4× 灌水，不要拿它做任何換算**
            #  （analysis-10 §40，2026-08-03 定案）。它由功率恆等式
            #  `Z_c = 2·P_in/(|a|²−|b|²)` 得到，前提是「源送進去的實功率 = 線上的淨波功率」
            #  ——**那個前提不成立**：超額功率正比於前進波功率 `|a|²`（`2P_in = c₁|a|²+c₂|b|²`
            #  擬合 R²≥0.996，`|c₁|/|c₂|=1.126`），出貨組態下 `P_leak/P_in ≈ 0.40`。
            #  後果是它**不是線的常數**：換貼片給 59.6–161.5 Ω，固定頻率掃負載給 66→106。
            #  ★ **模型饋線的真值是 `Z_c ≈ 50 Ω`**（線中 EMF 法 50.65、靜電電容法 49.07，
            #  兩種線寬互相驗證 3–4%；≈ 真實 1.1mm 線的 51.0 Ω ≈ 埠 50 Ω）
            #  ⇒ **`zc_ref` 不需要設**，Γ 直接就是 50Ω 參考的（殘差 1.2%）。
            extra = dict(Zdrive=zin_drv, beta=beta, _wamp=(wa, wb))
        else:
            zin = zin_drv
            gam = (zin - Z0) / (zin + Z0)
        #? dB 地板 −50/−40：HFSS 實務範圍就這麼大，不設地板時 |Γ|→0 會產生 −120dB
        #  離群值，擬核時直接把梯度炸成 NaN（實測踩到）。
        s11 = 20 * torch.log10(gam.abs().clamp(3.2e-3, 1.0))
        u0, prad = self.farfield(cur, f)
        #! **η 與 Gain 用的遠場必須分開**（2026-08-03）：
        #  η = P_rad/P_in 是**物理不變式**（唯一能抓到「核在偷吃功率」的低成本診斷）
        #  ⇒ 它必須用**全電流**，否則排除饋線會讓它憑空掉下去、不變式就失效了。
        #  Gain 用哪一組電流是**建模選擇**（`ff_line`），兩者不該混：
        #  混用會讓 `D₀·η` ≠ `4π·U₀/P_in`，數字沒有物理意義。
        prad_all = prad if (self.ff_line or self.feed_len == 0) else self.farfield(cur, f, True)[1]
        d0 = (4 * np.pi * u0 / prad.clamp_min(1e-300)).clamp_min(1e-4)
        #? **能量守恆 η = P_rad / P_in** —— 無耗 MoM 裡兩者必須相等，η 應 ≈ 1。
        #  這是唯一低成本、能抓到「核在偷偷吃功率」的診斷：舊核（n=√εr）實測 η = 0.066，
        #  也就是 93% 的輸入功率被 Z 矩陣吃掉、從沒進遠場——而 S11 曲線看起來完全正常。
        #  以後換核一律先看它，別再靠 ρ 好不好倒推物理對不對。
        #! 前因子：L2 的源是**電流**（E_far ∝ ωμ₀/4πr = k₀η₀/4πr）→ U = k₀²η₀/(32π²)·|S|²，
        #  η₀ 在**分子**。L1 的源是磁流（E_far ∝ k₀/4πr）→ η₀ 在分母。
        #  我一度把 L1 的前因子複製過來，η 差了 η₀² = 1.4e5 倍（算出 6e-6 而非 0.85）。
        #  S = Σ I_m·dx²·e^{jk·r} ⇒ |S|² 帶 dx⁴（`farfield` 回的 prad 沒有面積權重）。
        k0 = 2 * np.pi * f / C0
        pre = (k0 ** 2 * ETA0 / (32 * np.pi ** 2))[None, :] * (DX ** 4)
        p_rad = pre * prad_all                   # 不變式用全電流
        p_rad_ff = pre * prad                    # Gain 用選定的那組
        #! η 的 P_in 一定要用**源點**阻抗（V=1 的 delta-gap ⇒ P_in = ½Re(Zdrv)/|Zdrv|²）。
        #  饋線模式的 `zin` 是「參考面上、50Ω 歸一的等效阻抗」，拿它算功率會錯。
        p_in = 0.5 * zin_drv.real / zin_drv.abs() ** 2
        eta = p_rad / p_in.clamp_min(1e-300)
        if self.feed_len > 0:
            wa, wb = extra.pop("_wamp")
            extra["Zc"] = 2.0 * p_in / (wa.abs() ** 2 - wb.abs() ** 2).clamp_min(1e-300)
        mism = (1 - gam.abs() ** 2).clamp_min(1e-6)
        #! 2026-08-03 修：**漏乘輻射效率 `eta`**。`RealizedGain ≡ D₀·e_r·(1−|Γ|²)` 是定義，
        #  而 `e_r = P_rad/P_accepted` 就是這裡的 `eta`——它早就算好、放在回傳 dict 裡、
        #  **從來沒被乘進去**。同 repo 的 `l1.py:273` 寫的是 `d0 * e_r * mism`，一直是對的。
        #  零自由參數的公式缺陷，不是可調空間。
        #  ⚠ 這會改變數字，所以 `tests/data/diffsim_snapshot.npz` 同 commit 更新。
        #  實測（fan-out agent，正片 clean_OOS）：`l3` 高頻誤差 +6.07 → +0.04 dB；
        #  「誰在當 min」的判斷準確率 0.468 → 0.627（隨機 0.43）、Gain 主宰率 26% → 80%
        #  （真值 65%）。⚠ 但選批命中率只有 `l3` 10.0 → 16.7%（P(Δ>0)=0.93，**未達 0.95**）、
        #  `l3fl` **完全不動**（26.7%）——因為它把決定權**正確地**交給 Gain 通道，
        #  而那個通道自己只有 ρ≈+0.08 ⇒ **判對了誰卡、卻卡不準**。修它是為了公式正確，
        #  不是為了指標；用途上的空間在 `D₀`（見 analysis-10 §39）。
        #? `D₀·η_ff = 4π·U₀/P_in` —— 用同一組電流的 η，Gain 才等於「該組電流的實現增益」。
        eta_ff = eta if prad_all is prad else (p_rad_ff / p_in.clamp_min(1e-300))
        gain = (10 * torch.log10(d0 * eta_ff.clamp(1e-6, 1.0) * mism)).clamp_min(-40.0)
        s11 = torch.where(open_ckt, torch.zeros_like(s11), s11)      # 開路＝全反射 0dB
        gain = torch.where(open_ckt, torch.full_like(gain, -40.0), gain)
        #! `Prad`/`D0`/`Gain` 用的是 `ff_line` 選定的電流；`eta`/`Prad_all` 用**全電流**
        #  （不變式必須看全部，見上）。⇒ `ff_line=False` 時 `pre*Prad/p_in ≠ eta`，
        #  **這是刻意的，不是 bug**——但兩者不可混用（bug 獵捕 agent 標的 S3）。
        return dict(S11=s11, Gain=gain, Zin=zin, D0=d0, Prad=prad, Prad_all=prad_all,
                    eta=eta, J=cur, **extra)

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

    def farfield(self, cur, f, all_current: bool = False):
        """電流片遠場：水平電流 + 地平面鏡像 → 因子 2j·sin(k₀h·cosθ)（薄基板≈2j k₀h cosθ）。

        `all_current=True` 強制含饋線（給能量守恆不變式用；見 `solve`）。
        """
        key = (float(f[0]), float(f[-1]), len(f), self.layered_ff, self.ff_er)
        if key not in self._ffc:
            k0 = (2 * np.pi * f / C0).to(self.dtype)
            ph = torch.exp(1j * (k0[:, None, None, None] * self._geo).to(self.cdtype))
            self._ffc[key] = (ph,) + self._ff_factors(k0)
        ph, f_tm, f_te = self._ffc[key]
        if not all_current and not self.ff_line and self.feed_len > 0:
            cur = cur * self.is_patch          # 饋線電流不進遠場（見 `__init__`）
        jx = cur * self.mx          # 電流矩向量（軸向 1/0，對角 1/±1）
        jy = cur * self.my
        sx = torch.einsum("bfp,ftup->bftu", jx, ph)
        sy = torch.einsum("bfp,ftup->bftu", jy, ph)
        #! TM（θ 分量）與 TE（φ 分量）的介質因子**不同** —— 舊版對兩者乘同一個
        #  `2j·sin(k₀h·cosθ)`（＝自由空間＋PEC 鏡像，**不含 εr**），與 Z 的分層核不一致。
        et = f_tm * self.cosT * (self.cosP * sx + self.sinP * sy)
        ep = f_te * (-self.sinP * sx + self.cosP * sy)
        pw = et.abs() ** 2 + ep.abs() ** 2
        u0 = pw[:, :, 0, :].mean(-1)
        prad = (pw * self.sinT).sum((-1, -2)) * self.dth * self.dph
        return u0, prad

    def _ff_factors(self, k0):
        """遠場的 TM / TE 介質因子，(nf, n_theta, n_phi)。

        #! 2026-08-03 修正。舊版對 TM 與 TE **乘同一個** `2j·sin(k₀h·cosθ)`
        #  ——那是「PEC 地平面上方 h 的**自由空間**電流」的鏡像因子，**完全不含 εr**，
        #  而 `impedance_at` 的核（L3）是含介質的分層 Green's function ⇒ **兩者物理不一致**。
        #  症狀：能量守恆 η = P_rad/P_in 在 εr=1 時完美（1.0000，因為那時公式正好對），
        #  εr>1 時掉到 0.44，且假損耗 ∝ (εr−1)¹；而 `P_rad` 對所有 εr **完全不變**。
        #
        #  正確的因子＝譜域的 `Ṽ^e`／`Ṽ^h` 在 `k_ρ = k₀sinθ` 的取樣（與 Z 同核 ⇒ 自洽）。
        #  歸一化由「εr=1 必須退化回 `2j·sin(k₀h·cosθ)`」定死。
        #  ⚠ **TM 與 TE 不對稱**：`Ṽ^e` 的 `Z₀^e = k_z0/(ωε₀)` 已含一個 cosθ，`Ṽ^h` 沒有。
        #
        #  驗證（解析電流、17×17 貼片、f=20.8GHz，對 Mosig/Jackson–Alexopoulos 的
        #  `P_sw/P_sp = (3π/4)·k₀h·(1−1/εr)³/c₁`，`c₁ = 1−1/εr+0.4/εr²`）：
        #    εr    1.00    1.20    2.00    3.00    3.55
        #    修正  1.0000  0.9951  0.9052  0.8157  0.7830
        #    理論  1.0000  0.9946  0.9020  0.8214  0.7950   ← 最大差 1.2%
        #  （舊版同一組是 1.0000 / 0.8964 / 0.6267 / 0.4872 / 0.4441。）
        """
        if not self.layered_ff:
            img = (2j * torch.sin(k0[:, None, None] * H * self.cosT[None])).to(self.cdtype)
            return img, img
        w = (k0 * C0).to(self.dtype)
        krho = (k0[:, None, None] * self.sinT[None]).to(self.cdtype)
        kz0 = torch.sqrt((k0[:, None, None].to(self.cdtype)) ** 2 - krho ** 2)
        kz1 = torch.sqrt((k0[:, None, None].to(self.cdtype)) ** 2 * self.ff_er - krho ** 2)
        kz0 = torch.where(kz0.imag > 0, -kz0, kz0)          # principal sqrt（Im ≤ 0）
        kz1 = torch.where(kz1.imag > 0, -kz1, kz1)
        ww = w[:, None, None].to(self.cdtype)
        t = torch.tan(kz1 * H)
        #! `z0h = ωμ₀/kz0` 在 θ=π/2（kz0=0）是**複數除零 → NaN**（不是 inf）。
        #  把 kz0 乘進 TE 的分子分母就避開：F_TE = 2(j·z1h·t)·kz0/(ωμ₀ + j·z1h·t·kz0)，
        #  在掠射角自然給 0（該處確實沒有 TE 輻射）。TM 那支的 z0e ∝ kz0，天生不用除。
        #! 化簡掉 kz0：`F_TM = 2·Ṽ^e·ωε₀/kz0` 裡的 `Ṽ^e ∝ z0e = kz0/(ωε₀)`，
        #  約分後**完全不用除以 kz0** —— 否則 θ=π/2（kz0=0）會 0/0 給 NaN。同理 TE。
        #  εr=1 的退化：`F_TE = 2j·tan(x)/(1+j·tan(x)) = 2j·sin(x)·e^{−jx}`，
        #  **模長與舊版的 `2j·sin(x)` 相同** ⇒ 功率意義下精確退化（相位差不影響 |E|²）。
        #! 2026-08-03 二修（bug 獵捕 agent）：上面那兩條註解**只處理了 kz0=0，沒處理 kz1=0**。
        #  `ff_er = 1` 時 θ=π/2 有 kz0 = kz1 = 0 ⇒ `z1h = ωμ/kz1 = inf`、`t = tan(0) = 0`
        #  ⇒ `num_h = j·inf·0·0` = **NaN**；TM 那支則是 `0/0` ⇒ `Prad`/`eta`/`Gain` 全 NaN。
        #  ⚠ 諷刺的是本函式的**歸一化就是用「εr=1 要退化回 2j·sin(k₀h·cosθ)」定死的**，
        #  而那個組態正好是唯一會炸的。同一類 εr≤1 邊界坑在 `l3.tm0_neff` 才剛特判過（第二次）。
        #  修法：把 `tan(kz1·H)/kz1` 當一個整體（極限是 H），除法全部消掉。
        tiny = torch.finfo(self.dtype).eps
        safe1 = torch.where(kz1.abs() < tiny, torch.ones_like(kz1), kz1)
        tanc = torch.where(kz1.abs() < tiny, torch.full_like(kz1, H), t / safe1)   # → H
        num_e = 1j * (kz1 ** 2) * tanc / self.ff_er          # ＝ j·z1e·t·(ωε₀)
        den_e = kz0 + num_e                                  # ＝ (z0e + j·z1e·t)·(ωε₀)
        num_h = 1j * ww * MU0 * kz0 * tanc                   # ＝ j·z1h·t·kz0
        #? 兩者在 kz0 = kz1 = 0 都是**可去的** 0/0，極限是 0（掠射角沒有輻射）⇒ 直接給 0。
        z = torch.zeros_like(num_e)
        return (torch.where(den_e.abs() < tiny, z, 2 * num_e / torch.where(
                    den_e.abs() < tiny, torch.ones_like(den_e), den_e)),
                2 * num_h / (ww * MU0 + num_h))

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


#! 校準集**不能含全高（n=25）的矩形**：那種貼片會與饋線樁（第 26 列）連成一體，
#  有效長度不是 25 格，Hammerstad 閉式解不適用（實測 log 誤差 0.74 vs 其餘 0.01–0.03）。
CAL_SHAPES = ((21, 21), (21, 15), (17, 17), (17, 13), (14, 14), (14, 10), (11, 11))
CAL_RECTS = [(N - n, N, (N - w) // 2, (N - w) // 2 + w) for n, w in CAL_SHAPES]


def calibrate_analytic(scales=None, n_img: int = 3, verbose: bool = True):
    """掃 **v_scale（除在 G_V）**，讓模型共振頻率對上閉式解。回 (最佳 v_scale, 相對誤差)。

    只有一個純量——初值的 (源 − 地鏡像) 結構本來就對，缺的是 A 項與 V 項的相對權重
    （＝相速度 v = 1/√(L'C')）。`docs/diffsim.md` §3 說的「擬核」從這裡起步，不是從亂數起步。

    ⚠ 命名很要緊：舊版把這個縮放**乘在 G_A** 上（存檔 key `a_scale`），相速一樣但
    **偽輻射跟著放大 ~n³** —— 那正是 §10 修掉的病灶。讀到 `a_scale` 的人很容易照字面
    乘回 a_A，本輪就真的發生過（`build_l2` 沒跟著改，出貨核 η=0.372 而非 0.872）。
    """
    tgt = np.array([patch_fr(n * DX, w * DX) for n, w in CAL_SHAPES])
    best = None
    for s in (scales if scales is not None else np.geomspace(1.0, 8.0, 17)):
        #? 校準的是**相速**（A/V 的相對權重）。除在 G_V 上與乘在 G_A 上相速相同，
        #  但不放大 A 項的偽輻射——這正是 2026-08-03 修掉的病灶。
        m = MoML2(kernel=DCIMKernel(n_img=n_img, v_scale=float(s)))
        got = m.resonances(CAL_RECTS)
        err = float(np.sqrt(np.mean((np.log(got / tgt)) ** 2)))
        if verbose:
            print(f"  v_scale×{s:5.2f}: RMS log 誤差 {err:.4f}  f_model={np.round(got / 1e9, 1)}",
                  flush=True)
        if best is None or err < best[1]:
            best = (float(s), err)
    if verbose:
        print(f"  目標 f_analytic = {np.round(tgt / 1e9, 1)} GHz")
        print(f"**最佳 v_scale = {best[0]:.3f}（RMS log 誤差 {best[1]:.4f} ~ {best[1] * 100:.1f}%）**")
    return best


#! ---------------------------------------------------------------- 求解器登記表
#  ★ 一個名字 = 一組**完整**的物理設定（核 + 埠 + 遠場）。這張表存在的理由是本輪
#  踩過**四次**「宣稱修好但只落地一半」：`v_scale` 改了 `build_l2` 沒跟著改（出貨核
#  η=0.372 而非 0.872）、擬合核根本沒載入、遠場旗標只有實驗腳本開、埠模型同理。
#  共同結構都是「**設定散在呼叫端**，於是研究腳本與出貨路徑悄悄分岔」。
#  ⇒ 設定只准住在這裡；`run.py`、測試、實驗腳本一律用名字取。
SOLVERS = {
    #? 舊行為（特徵化快照守的那條、擬核唯一能走的那條——L3 表是物理常數不可擬）
    "dcim": dict(kernel="dcim"),
    #? 精確分層核 + 分層遠場 + 半屋頂埠（analysis-10 §31–§32 的組態）
    #! ⚠⚠ **不要拿 `l3` 當排序器用**（analysis-10 §44）。它在公式修對 `eta` 之後
    #  正片 top-60 命中率從 18.3% 掉到 **3.3%**——因為它的 `eta` 有 −4.01 dB 的
    #  **假頻率斜率**（集總埠的假影；理論只該 −0.32，`l3fl` 是 −0.54）。
    #  把一個正確的公式乘上一個錯的量，結果比兩個都錯還糟。留著只當**消融/對照**。
    "l3": dict(kernel="l3", layered_ff=True, half_port=True),
    #? ★ 真饋線 + 駐波 wave port（analysis-10 §37）。`half_port` 不再需要——
    #  它本來就是「沒有饋線」的補償。
    #  ⚠ §38/§42 報的數字是 `ff_line=True` 量的（那時還沒驗出來）；出貨值改 False 見 §43。
    "l3fl": dict(kernel="l3", layered_ff=True, feed_len=45, ff_line=False),
    #? ★★ **出貨組態**（2026-08-04）。再加**對角連通**（analysis-10 §45/§47）：
    #  HFSS 的像素盒 `pixel+0.01mm` ⇒ 對角相鄰在角落重疊 0.01×0.01mm、Unite 後導通，
    #  而 rooftop 只有邊相鄰 ⇒ 全史 57.1% 的樣本幾何與 HFSS 不符。
    #  對角**只在它是唯一通路時**啟用（兩個角落格都 void）⇒ 實心區逐位不變、成本近乎零。
    #  實測（fit 分割 clean n=680，判準發車前寫死）：
    #    ρ(wm) **+0.4571 → +0.7070**（Δ+0.2504，**P(>0)=100.0%**）
    #    top-60 命中率 **26.7% → 40.0%**（Δ+13.88%，P=99.6%）
    #    幾何不一致子集 ρ **+0.0515 → +0.5177** ⇒ §45 的因果歸屬證實。
    "l3fld": dict(kernel="l3", layered_ff=True, feed_len=45, ff_line=False, diag=True),
}


def build(name: str = "l3", kernel=None, **over):
    """依名字建 L2 求解器。`over` 覆寫個別欄位（消融實驗用，正常路徑別用）。"""
    if name not in SOLVERS:
        raise ValueError(f"未知 solver {name!r}；有 {sorted(SOLVERS)}")
    cfg = dict(SOLVERS[name])
    cfg.update(over)
    kind = cfg.pop("kernel")
    if kernel is None:
        if kind == "l3":
            from .l3 import L3Kernel
            kernel = L3Kernel()
        else:
            kernel = DCIMKernel()
    return MoML2(kernel=kernel, **cfg)


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
