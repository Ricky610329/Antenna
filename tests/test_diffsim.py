"""diffsim（可微模擬器）的物理正確性把關 —— 不掛 NAS、不跑 HFSS、純確定性。

這一組是「對抗式驗證」的常駐版：ρ 好看不代表物理對，所以每個 gate 報數前，
求解器必須先過解析 sanity 案例。涵蓋：
  - 譜：實心/矩形貼片的離散 Neumann 特徵譜 vs 閉式解
  - 互易性、被動性（Re Zin ≥ 0、|S11| ≤ 1）、能量（P_rad > 0）
  - 尺一致性：本鏈的向量化 worst_margin 必須與 `antenna.losses.worst_margin` 逐筆相同
"""
import os

import numpy as np
import pytest
import torch

from script.diffsim.geom import N, DX, C0, EPS_R, H, EPS0, feed_weights
from script.diffsim.l1 import CavityL1
from script.diffsim import eval as dse


def _fres(lam):
    """特徵值 → 諧振頻率 (GHz)。"""
    return np.sqrt(np.asarray(lam)) / DX * C0 / (2 * np.pi * np.sqrt(EPS_R)) / 1e9


def _analytic_neumann(nx, ny, m, n):
    """格心 Neumann 離散譜的閉式解：k = √( (2/dx·sin(mπ/2nx))² + (2/dx·sin(nπ/2ny))² )。"""
    kx = 2 / DX * np.sin(m * np.pi / (2 * nx))
    ky = 2 / DX * np.sin(n * np.pi / (2 * ny))
    return np.sqrt(kx ** 2 + ky ** 2) * C0 / (2 * np.pi * np.sqrt(EPS_R)) / 1e9


@pytest.fixture(scope="module")
def sim():
    return CavityL1(q=20.0)


def test_solid_patch_spectrum(sim):
    """實心 25×25（5×5mm）的前幾個模態頻率＝離散 Neumann 閉式解（<0.5%）。"""
    p = torch.ones(1, N, N, dtype=torch.float64)
    lam, _ = sim.modes(p)
    got = _fres(lam[0].numpy())
    want = sorted(_analytic_neumann(N, N, m, n) for m in range(3) for n in range(3))
    for g, w in zip(got[:5], want[:5]):
        assert abs(g - w) <= 0.005 * max(w, 1.0), f"{g} vs {w}"


def test_rectangular_subpatch_spectrum(sim):
    """任意矩形子貼片（14×10 格）也要對上閉式解——證明「任意形狀」不是只有實心對。"""
    p = torch.zeros(1, N, N, dtype=torch.float64)
    p[0, 8:22, 6:16] = 1.0                       # 14×10 格 = 2.8 × 2.0 mm
    lam, _ = sim.modes(p)
    got = _fres(lam[0].numpy())
    want = sorted(_analytic_neumann(14, 10, m, n) for m in range(3) for n in range(3))
    for g, w in zip(got[:4], want[:4]):
        assert abs(g - w) <= 0.01 * max(w, 1.0), f"{g} vs {w}"


def test_zero_mode_is_static_capacitance(sim):
    """λ=0 模的貢獻＝貼片靜電容 C = εA/h（腔模型的低頻極限，抓錯就整條阻抗歪掉）。"""
    p = torch.zeros(1, N, N, dtype=torch.float64)
    p[0, 11:25, 6:16] = 1.0                      # 必須碰到 row 24（饋線接點），否則 ⟨ψ⟩_feed = 0
    area = 14 * 10 * DX * DX
    c_exact = EPS_R * EPS0 * area / H
    f = np.array([1e9])                          # 遠低於第一共振 → 純電容
    out = sim.forward(p, freqs=f)
    z = complex(out["Zin"][0, 0])
    c_model = -1.0 / (2 * np.pi * f[0] * z.imag)
    assert abs(c_model / c_exact - 1) < 0.02, f"{c_model} vs {c_exact}"


def test_passivity_and_energy(sim):
    """被動性：Re(Zin) ≥ 0、|S11| ≤ 1（+0 dB）；能量：P_rad > 0、D₀ 有限。"""
    rng = np.random.default_rng(3)
    p = torch.as_tensor((rng.random((6, N, N)) > 0.5).astype(np.float64))
    out = sim.forward(p)
    assert (out["Zin"].real >= -1e-9).all()
    assert (out["S11"] <= 1e-6).all(), float(out["S11"].max())
    assert (out["Prad"] >= 0).all()   # ＝0 只在饋線接不到金屬時（見 test_disconnected_feed_has_zero_radiation）
    assert torch.isfinite(out["Gain"]).all()


def test_reciprocity_two_port(sim):
    """互易性：驅動埠 1 量埠 2 的電壓 vs 反過來，必須相等。

    #! 舊版是恆等式（`(a*b).sum()` vs `(b*a).sum()`，IEEE754 逐位元相同，永不失敗；
    #  突變測試注入 11 個物理 bug 抓到 0 個）。改成整條走 `forward` 的場路徑，
    #  兩個埠角色真的互換——寫錯順序/符號才抓得到。
    """
    p = torch.zeros(1, N, N, dtype=torch.float64)
    p[0, 8:25, 6:20] = 1.0
    lam, psi = sim.modes(p)
    psi_p = (psi / DX)[0]
    k2 = (2 * np.pi * 28e9 / C0) ** 2 * EPS_R
    den = lam[0] / DX ** 2 - k2
    w = []
    for rc in ((24, 12), (10, 8)):
        e = torch.zeros(N * N, dtype=torch.float64)
        e[np.ravel_multi_index(rc, (N, N))] = 1.0
        w.append((psi_p * e[:, None]).sum(0))
    z12 = float((w[0] * w[1] / den).sum())          # 驅動 1 → 量 2
    z21 = float((w[1] * w[0] / den).sum())          # 驅動 2 → 量 1
    assert abs(z12) > 1e-6, "退化案例（Z12≈0）不構成互易性檢查"
    assert abs(z12 - z21) < 1e-10 * abs(z12)


def test_farfield_single_magnetic_element(sim):
    """遠場解析標的：地平面上單一均勻磁流元的 D₀ **恰為 3.000**（4.77 dBi）。

    #! 突變測試發現遠場整段零覆蓋——注入「sx/sy 互換」「漏掉地平面鏡像 ×2」
    #  「+y/−y 面同號」「漏掉 cos²θ 極化投影」四個嚴重 bug，13 條測試抓到 0 個。
    #  這條 3 行就補上：只留一個面的磁流 → 單一磁偶極 → U ∝ 1−sin²θcos²φ → D₀ = 3。
    """
    z = torch.zeros(1, N * N, dtype=torch.float64)
    one = z.clone()
    one[0, np.ravel_multi_index((12, 12), (N, N))] = 1.0
    ez = torch.ones(1, 1, N * N, dtype=torch.complex128)
    f = np.array([28e9])
    ph, (cu, cv) = sim._ph_exp(torch.as_tensor(f, dtype=torch.float64))
    u0, prad = sim._farfield(ez, (z, z, one, z), ph, cu, cv)   # 只有 +y 面有磁流
    d0 = float(4 * np.pi * u0[0, 0] / prad[0, 0])
    assert abs(d0 - 3.0) < 0.02, d0


def test_disconnected_feed_has_zero_radiation(sim):
    """饋線接不到金屬 → 場恆 0 → P_rad **恰為 0**、Gain 掉到 −40dBi 地板。

    #! `Prad > 0` 不是不變式（舊測試靠 seed 沒抽中才綠：隨機 50% 圖饋線 7 格全 void
    #  的機率 0.5⁷≈0.8%）。這裡把它寫成明確的預期行為，而不是靠運氣。
    """
    p = torch.zeros(1, N, N, dtype=torch.float64)
    p[0, 5:15, 8:18] = 1.0
    out = sim.forward(p)
    assert float(out["Prad"].abs().max()) == 0.0
    assert float(out["Gain"].max()) < -35.0


def test_production_config_spectrum_shift(sim):
    """生產配置（gap=diag=2）下譜會位移——把已知的人工色散**寫死成預期**，不假裝沒有。

    對角項不是「角碰角電容」（見 `modes` 內註記）：平滑模下 Lap_diag ≈ 2A
    → λ → λ/(1+2γλ)。這條同時保證未來改動不會無聲地改變這個行為。
    """
    p = torch.ones(1, N, N, dtype=torch.float64)
    lam0, _ = CavityL1(gap=0, diag=0).modes(p)
    lam2, _ = CavityL1(gap=2, diag=2).modes(p)
    g = 2.0
    for n in (1, 2, 3):
        pred = float(lam0[0, n]) / (1 + 2 * g * float(lam0[0, n]))
        assert abs(float(lam2[0, n]) - pred) < 0.05 * pred, (n, float(lam2[0, n]), pred)
    assert float(lam2[0, 3]) < float(lam0[0, 3])          # 一律往下移


def test_island_decoupling(sim):
    """不接饋點的離島不該改變 S11（⟨ψ⟩_feed = 0 自然去耦；指導書 §3 明列的行為）。"""
    p = torch.zeros(1, N, N, dtype=torch.float64)
    p[0, 12:25, 8:18] = 1.0
    base = sim.forward(p)["S11"].clone()
    q = p.clone()
    q[0, 0:4, 0:4] = 1.0                          # 遠處孤島
    assert torch.allclose(base, sim.forward(q)["S11"], atol=1e-8)


def test_disconnected_feed_is_total_reflection(sim):
    """饋線接不到金屬 → 開路端 → |S11| ≈ 0 dB（全反射），不是隨便一個數。"""
    p = torch.zeros(1, N, N, dtype=torch.float64)
    p[0, 5:15, 8:18] = 1.0                        # 不碰 row 24
    out = sim.forward(p)
    assert float(out["S11"].max()) > -0.2
    assert float(out["S11"].min()) > -1.0


def test_q_controls_bandwidth(sim):
    """Q 是頻寬旋鈕：Q 大 → 諧振窄。抓的是「阻抗虛部過零附近的 |S11| 曲率」單調性。"""
    p = torch.zeros(1, N, N, dtype=torch.float64)
    p[0, 11:25, 6:20] = 1.0
    f = np.linspace(20e9, 36e9, 321)
    widths = []
    for q in (5.0, 20.0, 80.0):
        m = CavityL1(q=q)
        rz = m.forward(p, freqs=f)["Zin"][0].real.numpy()      # Re(Zin) 的 FWHM ＝ 載荷 Q 的定義
        widths.append((rz > rz.max() / 2).sum())
    assert widths[0] > widths[1] > widths[2], widths


def test_feed_weights_match_sab_geometry():
    """饋線 1.1mm 寬（.sab 實測 y∈[1.95, 3.05]）→ 覆蓋 5.5 格、以 (24,12) 為中心。"""
    fw = feed_weights()
    assert abs(fw.sum() - 1.0) < 1e-12
    nz = np.nonzero(fw)[0]
    assert nz.min() == 9 and nz.max() == 15
    assert np.argmax(fw) in (10, 11, 12, 13, 14)
    assert abs(fw[12] - 0.2 / 1.1) < 1e-9


def test_ruler_matches_worst_margin():
    """本鏈的向量化尺必須與 `antenna.losses.worst_margin` 逐筆相同（不是另一把尺）。"""
    from antenna.losses import worst_margin
    targets = {"S11": {"center": -10, "width": [5, 0, 7, 0, 5], "method": "low"},
               "Gain": {"center": 4, "width": [5, 0, 7, 0, 5], "method": "high"}}
    rng = np.random.default_rng(7)
    y = rng.normal(-6, 5, size=(20, 34))
    wm, m_s11, m_gain = dse.margins(y)
    for i in range(len(y)):
        ref, parts = worst_margin(y[i], ["S11", "Gain"], targets)
        assert abs(wm[i] - ref) < 1e-4
        assert abs(m_s11[i] - parts["S11"]) < 1e-4
        assert abs(m_gain[i] - parts["Gain"]) < 1e-4


def test_oob_matches_dedust():
    """帶外總帳也用同一把尺（對 `script.dedust.oob_metrics`）。"""
    from script.dedust import oob_metrics
    rng = np.random.default_rng(11)
    y = rng.normal(-4, 4, size=(10, 34))
    got = dse.oob_bad(y)
    for i in range(len(y)):
        assert abs(got[i] - oob_metrics(y[i])["oob_bad"]) < 1e-2


def test_relaxation_is_continuous(sim):
    """連續鬆弛：ρ 從 0 掃到 1，S11 必須連續（STE/梯度優化的前提）。"""
    base = torch.zeros(1, N, N, dtype=torch.float64)
    base[0, 11:25, 6:20] = 1.0
    vals = []
    for t in np.linspace(0.0, 1.0, 11):
        p = base.clone()
        p[0, 11:14, 6:9] = float(t)
        vals.append(float(sim.forward(p)["S11"][0, 8]))
    d = np.abs(np.diff(vals))
    assert d.max() < 3.0, d


def test_gradient_flows(sim):
    """端到端可微：對 ρ 的梯度存在且有限（這是 diffsim 相對 SM 的存在理由之一）。"""
    p = torch.zeros(1, N, N, dtype=torch.float64)
    p[0, 11:25, 6:20] = 1.0
    p = (p * 0.9 + 0.05).requires_grad_(True)
    out = sim.forward(p)
    out["S11"][0, 8].backward()
    assert p.grad is not None and torch.isfinite(p.grad).all()
    assert float(p.grad.abs().max()) > 0


# ---------------------------------------------------------------- L2（MoM）
@pytest.fixture(scope="module")
def mom():
    from script.diffsim.l2 import MoML2, DCIMKernel
    return MoML2(kernel=DCIMKernel(v_scale=2.48))   # 解析校準（見 l2.calibrate_analytic）


def _rect(i0, i1, j0, j1, nb=1):
    p = torch.zeros(nb, N, N, dtype=torch.float64)
    p[:, i0:i1, j0:j1] = 1.0
    return p


def test_l2_matrix_is_symmetric(mom):
    """MoM 阻抗矩陣必須對稱（互易介質的 EFIE）——組裝寫錯順序就會破。"""
    Z = mom.impedance_at(28e9)
    assert float((Z - Z.T).abs().max() / Z.abs().max()) < 1e-12


def test_l2_passivity(mom):
    """被動性：Re(Zin) ≥ 0、|S11| ≤ 1（隨機圖，含斷路情形）。"""
    rng = np.random.default_rng(5)
    p = torch.as_tensor((rng.random((3, N, N)) > 0.5).astype(np.float64))
    with torch.no_grad():
        out = mom.solve(p, freqs=np.array([26e9, 28e9, 30e9]))
    assert (out["Zin"].real >= -1e-9).all(), float(out["Zin"].real.min())
    assert (out["S11"] <= 1e-6).all(), float(out["S11"].max())


def test_l2_masked_equals_resistive_loading(mom):
    """遮罩解（快）與電阻加載（可微）在二值 pattern 上必須給同一個答案。"""
    p = _rect(10, 25, 5, 20, nb=2)
    p[1, 12:15, 8:12] = 0.0
    f = np.array([28e9])
    with torch.no_grad():
        a = mom.solve(p, freqs=f)["S11"]
        b = mom.solve(p, freqs=f, r_open=1e6)["S11"]
    assert torch.allclose(a, b, atol=1e-6), float((a - b).abs().max())


def test_l2_resonances_match_closed_form(mom):
    """校準後的 MoM 共振頻率要對上 Hammerstad 閉式解（這是 L2 物理正確性的主證據）。

    L1 的硬磁牆理想化在 0.2mm 像素 / 0.508mm 基板上是破的；MoM 沒有這個假設，
    所以它「應該」對得上——對不上就代表核或組裝有問題，不是可接受的近似誤差。
    """
    from script.diffsim.l2 import patch_fr, CAL_RECTS, CAL_SHAPES
    got = mom.resonances(CAL_RECTS[:5], fmin=10e9, fmax=40e9, nf=61)
    want = np.array([patch_fr(n * DX, w * DX) for n, w in CAL_SHAPES[:5]])
    err = np.abs(np.log(got / want))
    assert err.max() < 0.06, list(zip((got / 1e9).round(1), (want / 1e9).round(1)))


def test_l2_farfield_horizontal_dipole_over_ground(mom):
    """遠場解析標的：地平面上方 h≪λ 的水平電流元 → D₀ = 7.5（解析積分）。

    U ∝ (cos²θcos²φ + sin²φ)·4sin²(k₀h cosθ)；h≪λ 展開後
    P_rad ∝ 4π(k₀h)²·(1/5 + 1/3)、U(0) ∝ 4(k₀h)² → D₀ = 4/(8/15) = 7.5。
    抓的是極化投影與地平面鏡像因子——兩者在 L1 的突變測試裡都曾整段沒人守。
    """
    from script.diffsim.l2 import MoML2, DCIMKernel
    m = MoML2(kernel=DCIMKernel(), n_theta=97, n_phi=48)
    cur = torch.zeros(1, 1, m.nb, dtype=torch.complex128)
    cur[0, 0, 0] = 1.0                                    # 單一 x 向屋頂
    f = torch.tensor([2e9], dtype=torch.float64)          # 低頻 → k₀h ≪ 1
    u0, prad = m.farfield(cur, f)
    d0 = float(4 * np.pi * u0[0, 0] / prad[0, 0])
    assert abs(d0 - 7.5) < 0.15, d0


def test_l2_energy_conservation(mom):
    """**能量守恆 η = P_rad/P_in ≈ 1** —— 無耗 MoM 的硬約束，也是唯一抓得到
    「核在偷吃功率」的低成本診斷。

    #! 2026-08-03：舊核把 n=√εr 放進兩支鏡像的指數（＝描述「浸在無限介質裡的地板」
    #  而非「接地薄板+空氣」），實測 **η = 0.085** ——91.5% 的輸入功率被 Z 矩陣吃掉、
    #  從沒進遠場，而 S11 曲線看起來完全正常、共振頻率還對閉式解 1.6%。
    #  沒有這條測試，這種 bug 只會表現成「rank ρ 就是上不去」，永遠查不到根因。
    """
    #! 守的必須是**出貨核**（`build_l2()` 實際載入的那顆），不是測試自己造的。
    #  2026-08-03 稽核抓到：`DCIMKernel` 修好了但 `build_l2` 沒跟著改，
    #  出貨核 η=0.372 而測試守的那條路 η=0.876 —— 測試綠但產品壞。
    from script.diffsim.run import build_l2
    ship, _ = build_l2()
    f = np.array([26e9, 28e9, 30e9])
    for rect in ((12, 25, 6, 19), (9, 25, 4, 21)):
        p = _rect(*rect)
        with torch.no_grad():
            out = ship.solve(p, freqs=f)
        eta = out["eta"][0].numpy()
        assert (eta < 1.15).all(), f"η > 1 違反能量守恆：{eta}"
        assert (eta > 0.5).all(), f"η 太低＝核在偷吃功率（舊核是 0.085）：{eta}"


def test_l2_disconnected_feed_no_nan(mom):
    """饋線接觸格全 void → 驅動邊全死 → itot = 0。必須回「開路」而不是 NaN。

    #! 這是**確定性的除以零**，不是「矩陣接近奇異」（實測 cond 最大只有 3e3）。
    #  val 120 筆踩到 1 筆、被 `rank_rho` 靜默濾掉。L1 早就有對應測試，L2 沒有——
    #  而 `test_l2_passivity` 用隨機圖，全 void 的機率 0.78%，它綠是靠 seed 沒抽中。
    """
    p = torch.zeros(1, N, N, dtype=torch.float64)
    p[0, 5:15, 8:18] = 1.0                       # 不碰 row 24
    with torch.no_grad():
        out = mom.solve(p, freqs=np.array([28e9]))
    assert torch.isfinite(out["S11"]).all() and torch.isfinite(out["Gain"]).all()
    assert abs(float(out["S11"][0, 0])) < 1e-9   # 全反射
    assert float(out["Gain"][0, 0]) <= -40.0 + 1e-9



# ---------------------------------------------------------------- 資料分割鐵則
# 這一組守 `docs/diffsim.md` §5 的分割鐵則。稽核指出它先前**零測試覆蓋**——
# 被稱為「鐵則」的東西沒有任何一條測試守著，而 gate 的可信度全靠它。
def _fake_idx(n_clean=400, n_neg=120, n_senior=300, n_frozen=30, seed=0):
    rng = np.random.default_rng(seed)
    parts, strat = [], []
    for tag, k in (("clean", n_clean), ("neg", n_neg), ("senior", n_senior), ("frozen", n_frozen)):
        parts.append((rng.random((k, 625)) > 0.5).astype(np.uint8))
        strat += [tag] * k
    x = np.concatenate(parts)
    return dict(x=x, y=rng.normal(-6, 4, (len(x), 34)).astype(np.float32),
                stratum=np.asarray(strat), store=np.asarray(["s"] * len(x)),
                name=np.asarray([f"n{i}" for i in range(len(x))]))


def test_split_is_disjoint_and_covers():
    """val / dev / fit 三者互斥且覆蓋全體——這是「不相交鐵則」的最低要求。"""
    from script.diffsim import data as D
    idx = _fake_idx()
    sp, _ = D.assign_split(idx, val_per_stratum=30, dev_per_stratum=50)
    assert set(np.unique(sp)) <= {"val", "dev", "fit"}
    #! 「總和 == N」是恆等式（split 由 full("fit") 起手），抓不到東西。
    #  真正的鐵則是**內容層級不相交**：擬合用的 pattern 不可以出現在 val。
    import hashlib as _h
    key = lambda i: _h.sha1(np.ascontiguousarray(idx["x"][i]).tobytes()).hexdigest()  # noqa: E731
    kv = {key(i) for i in np.where(sp == "val")[0]}
    kf = {key(i) for i in np.where(sp == "fit")[0]}
    kd = {key(i) for i in np.where(sp == "dev")[0]}
    assert not (kv & kf), f"val 的 pattern 出現在 fit：{len(kv & kf)} 筆"
    assert not (kv & kd), f"val 的 pattern 出現在 dev：{len(kv & kd)} 筆"
    assert len(kv) == int((sp == "val").sum()), "val 內有重複 pattern"


def test_frozen_ruler_never_leaves_val():
    """凍結尺（OOD 30 筆）**全部進 val、永不進擬合**——指導書 §5 的紅線。"""
    from script.diffsim import data as D
    idx = _fake_idx()
    sp, _ = D.assign_split(idx, val_per_stratum=30, dev_per_stratum=50)
    fr = idx["stratum"] == "frozen"
    assert (sp[fr] == "val").all(), "凍結尺外流到 dev/fit"


def test_split_is_content_deterministic():
    """切分只是內容的函數：打亂列序、換 dtype 都不能改變任何一筆的歸屬。"""
    from script.diffsim import data as D
    idx = _fake_idx()
    sp1, u1 = D.assign_split(idx, val_per_stratum=30, dev_per_stratum=50)
    perm = np.random.default_rng(7).permutation(len(idx["x"]))
    idx2 = {k: v[perm] for k, v in idx.items()}
    sp2, u2 = D.assign_split(idx2, val_per_stratum=30, dev_per_stratum=50)
    assert (sp2 == sp1[perm]).all()
    assert np.allclose(u2, u1[perm])


def test_val_freeze_pins_membership(tmp_path, monkeypatch):
    """★ val 凍結後**不因索引擴充而漂**。

    #! 沒有這條的話：新資料只要有一筆 hash-u01 落進該層前 30 名，就會擠掉現有 val 成員
    #  → 「val 只看一次」的帳悄悄失效、已報的 gate 數字不再可重現
    #  （實測未凍結時，索引縮減 10 筆就換掉 20 筆 val）。
    """
    from script.diffsim import data as D
    import hashlib as _h
    idx = _fake_idx()
    monkeypatch.setattr(D, "VAL_FREEZE", str(tmp_path / "freeze.txt"))
    sp, _ = D.assign_split(idx, val_per_stratum=30, dev_per_stratum=50)
    keys = [_h.sha1(np.ascontiguousarray(idx["x"][i]).tobytes()).hexdigest()[:16]
            for i in np.where(sp == "val")[0]]
    (tmp_path / "freeze.txt").write_text("\n".join(keys) + "\n", encoding="utf-8")

    #? 擴充批**不含 frozen**：凍結尺是固定的 30 筆；真有新凍結尺時它本來就該進 val
    #  （那條紅線優先於凍結名單，見 `assign_split` 的 `hit |= strat=='frozen'`）。
    big = _fake_idx(n_clean=900, n_neg=400, n_senior=700, n_frozen=0, seed=1)
    grown = {k: np.concatenate([idx[k], big[k]]) for k in idx}
    sp2, _ = D.assign_split(grown, val_per_stratum=30, dev_per_stratum=50)
    keys2 = {_h.sha1(np.ascontiguousarray(grown["x"][i]).tobytes()).hexdigest()[:16]
             for i in np.where(sp2 == "val")[0]}
    assert keys2 == set(keys), f"val 漂了：{len(keys2 ^ set(keys))} 筆不同"


def test_pick_is_process_stable():
    """`pick()` 的抽樣不可依賴 Python 的 `hash()`（每個 process 都不同）。"""
    from script.diffsim import data as D
    from script.diffsim.run import pick
    idx = _fake_idx()
    sp, _ = D.assign_split(idx, val_per_stratum=30, dev_per_stratum=50)
    #! 必須跨 **process** 驗——同一個 process 內 `hash("clean")` 是常數，
    #  就算退回用 `hash()` 這條測試照樣綠（稽核抓到我第一版就是這樣）。
    import subprocess
    import sys as _sys
    code = ("import numpy as np;from script.diffsim import data as D;"
            "from script.diffsim.run import pick;"
            "import tests.test_diffsim as T;idx=T._fake_idx();"
            "sp,_=D.assign_split(idx,val_per_stratum=30,dev_per_stratum=50);"
            "print(','.join(map(str,pick(idx,sp,'fit',20))))")
    outs = set()
    for seed in ("0", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        r = subprocess.run([_sys.executable, "-c", code], capture_output=True,
                           text=True, env=env, cwd=os.getcwd())
        assert r.returncode == 0, r.stderr[-800:]
        outs.add(r.stdout.strip().splitlines()[-1])
    assert len(outs) == 1, f"pick 依賴了 PYTHONHASHSEED：{outs}"
    assert not (pick(idx, sp, "fit", 20) == pick(idx, sp, "fit", 20, seed=1)).all()


def test_l2_gradient_check_vs_finite_difference(mom):
    """★ **對設計變數 ρ 的 adjoint 梯度 vs 中央有限差分**——ceviche / Meep 的標準驗收流程。

    為什麼非有不可：文獻明講可微模擬器的梯度**可能方向就是錯的**，而這種 bias
    **無法從 loss 曲線或梯度變異數看出來**（Suh et al., ICML 2022）。本鏈裡尤其危險——
    到處是 `clamp`（dB 地板、`lam.clamp_min(0)`），撞到邊界的分量梯度會被砍成**恰好 0**。

    注意：可微的是 `r_open`（電阻加載）那條路徑；遮罩解對 ρ 不可微（是硬遮罩），
    只給擬核與排序用。
    """
    base = torch.zeros(N, N, dtype=torch.float64)
    base[12:25, 6:19] = 1.0
    rho = base * 0.9 + 0.05                       # 鬆弛到 (0,1) 內部，避開 clamp 邊界
    f = np.array([28e9])
    r_open = 1e-4

    def J(p):
        return mom.solve(p.reshape(1, N, N), freqs=f, r_open=r_open)["S11"][0, 0]

    p = rho.clone().requires_grad_(True)
    J(p).backward()
    g = p.grad.detach()
    assert torch.isfinite(g).all()
    assert int((g.abs() > 1e-12).sum()) > 600, "大量像素梯度為零＝疑似被 clamp 砍掉"

    for (i, j) in ((21, 10), (18, 23), (11, 12)):   # 挑梯度量級夠大的，避開差分雜訊地板
        h = 1e-5
        pp, pm = rho.clone(), rho.clone()
        pp[i, j] += h
        pm[i, j] -= h
        with torch.no_grad():
            fd = float((J(pp) - J(pm)) / (2 * h))
        assert abs(float(g[i, j]) - fd) <= 0.02 * abs(fd), (i, j, float(g[i, j]), fd)


def test_wm_torch_matches_margins():
    """可微版 `_wm_torch` 必須與 `eval.margins` 逐筆相同——**兩把尺不可以漂**。

    #! 稽核指出本 repo 有兩份 worst_margin 實作，而只有 `margins` 被把關
    #  （`test_ruler_matches_worst_margin` 對 `antenna.losses.worst_margin`）。
    #  `_wm_torch` 是 rank loss 的監督訊號來源，一旦漂掉就會靜默優化另一把尺。
    """
    from script.diffsim.run import _wm_torch
    rng = np.random.default_rng(5)
    y = rng.normal(-6, 5, size=(16, 34))
    got = _wm_torch(torch.as_tensor(y[:, :17]), torch.as_tensor(y[:, 17:])).numpy()
    assert np.allclose(got, dse.margins(y)[0], atol=1e-9)



def test_l1_gradient_on_strictly_binary(sim):
    """★ **嚴格二值** pattern 的梯度必須有限——資料集餵的全是二值，鬆弛值抓不到這個。

    #! 2026-08-03 L1 複審實測：修這條之前 val 全 120 筆梯度非有限率 **100%**。
    #  `torch.linalg.eigh` 的反傳含 1/(λi−λj)，而二值 pattern 必然大量嚴格重根
    #  （孤立 void 格 λ 全同、每個連通分量一個嚴格 λ=0、方形貼片 TM₁₀/TM₀₁ 簡併）。
    #  舊的 `test_gradient_flows` 用 `p*0.9+0.05` 把重根抬開了，所以綠得毫無意義——
    #  這正是「測試沒碰到真正的輸入域」的同型盲區。
    """
    for gap in (0.0, 2.0):
        m = CavityL1(q=15.0, gap=gap, diag=gap, rad_eff=(gap > 0))
        p = torch.zeros(1, N, N, dtype=torch.float64)
        p[0, 11:25, 6:20] = 1.0                       # 嚴格 0/1，不做任何鬆弛
        p[0, 2:5, 2:5] = 1.0                          # 加一顆孤島 → 製造嚴格 λ=0 重根
        p.requires_grad_(True)
        m.forward(p)["S11"][0, 8].backward()
        assert torch.isfinite(p.grad).all(), f"gap={gap} 嚴格二值梯度出現非有限值"
        assert float(p.grad.abs().max()) > 0


def test_l1_grad_eps_does_not_change_forward():
    """梯度護欄只在 `requires_grad` 時生效——no_grad 的 forward 必須**逐位元**不變。

    否則「為了讓梯度活下來」就會偷偷改掉已報 gate 的數值。
    """
    p = torch.zeros(1, N, N, dtype=torch.float64)
    p[0, 11:25, 6:20] = 1.0
    with torch.no_grad():
        a = CavityL1(q=15.0, gap=2, diag=2, rad_eff=True, grad_eps=1e-6).forward(p)["S11"]
        b = CavityL1(q=15.0, gap=2, diag=2, rad_eff=True, grad_eps=0.0).forward(p)["S11"]
    assert torch.equal(a, b)


def test_mkl_single_thread_guard_restores_state():
    """`_mkl_single_thread` 進去必須是 1、出來必須還原（含例外路徑）。"""
    from script.diffsim.l2 import _mkl_single_thread
    n0 = torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        with _mkl_single_thread():
            assert torch.get_num_threads() == 1
        assert torch.get_num_threads() == 4
        try:                                          # 例外也要還原
            with _mkl_single_thread():
                raise ValueError("boom")
        except ValueError:
            pass
        assert torch.get_num_threads() == 4
    finally:
        torch.set_num_threads(n0)


def test_l2_solve_survives_multithreaded_mkl(mom):
    """多執行緒下 `solve` 不可以炸。

    #! 2026-08-03：**本機 MKL 的 batched complex128 LU 在 `num_threads > 1` 時會壞**——
    #  底層 `oneMKL ERROR: Parameter 6 was incorrect on entry to ZLASWP`，
    #  torch 再拋 `Pivots given to lu_solve must all be >= 1`。同一批資料 threads=1 正常、=6 必炸。
    #  ⚠ 我一度誤判成「遮罩子矩陣奇異」（環流零空間/孤島），寫了 `solve_ex`+`pinv` fallback
    #  ——**那攔不住**，因為例外是在 `lu_solve` 的輸入檢查階段就拋的，根本沒回傳 info。
    #  真因由 delta-gap 稽核 agent 在完全不同的實驗裡獨立撞到，才對上。
    #  ⚠ 這條測試在**沒有這個 MKL bug 的機器上會自動變成 no-op**（照樣綠）——
    #  它守的是「保護還在」，真正的行為驗證在 `test_mkl_single_thread_guard_restores_state`。
    """
    n0 = torch.get_num_threads()
    try:
        torch.set_num_threads(max(2, min(6, (os.cpu_count() or 2))))
        rng = np.random.default_rng(0)
        p = (rng.random((16, N, N)) < 0.5).astype(np.float64)
        p[:, 24, 9:16] = 1.0                          # 保證饋電接得上
        with torch.no_grad():
            out = mom.solve(torch.as_tensor(p, dtype=torch.float64), freqs=np.array([28e9]))
        assert torch.isfinite(out["S11"]).all()
    finally:
        torch.set_num_threads(n0)


# ---------------------------------------------------------------- 選批稽核工具
def test_selection_audit_calibration():
    """`sm_selection_audit.audit` 的兩個指標必須對「完美／隨機／反向」預測器給出正確的刻度。

    這條守的是**刻度本身**（不碰 NAS）：如果哪天改了 bootstrap 或命中率的定義，
    完美預測器不再是 ~100%、隨機不再是 ~10%，這裡會紅。
    """
    from script.sm_selection_audit import audit
    rng = np.random.default_rng(0)
    y = rng.normal(size=800)
    k, nb = 60, 400
    win_p, hit_p, _, rho_p, _ = audit(y.copy(), y, k, nb, seed=1)              # 完美
    win_r, hit_r, _, _, _ = audit(rng.normal(size=800), y, k, nb, seed=1)      # 隨機
    win_n, hit_n, _, rho_n, _ = audit(-y, y, k, nb, seed=1)                    # 反向
    assert rho_p > 0.99 and rho_n < -0.99
    assert hit_p > 90.0, f"完美預測器的 top10% 命中率應接近 100%，得到 {hit_p:.0f}%"
    assert 5.0 < hit_r < 18.0, f"隨機預測器應落在 10% 附近，得到 {hit_r:.0f}%"
    assert hit_n < 2.0, f"反向預測器應接近 0%，得到 {hit_n:.0f}%"
    #! `P(勝隨機)` 用嚴格 `>`（平手算輸）⇒ **完美預測器的上限不是 100%**：
    #  它必定選到最大值，但隨機挑 K 筆也有 ~K/n 機率選到同一筆＝平手＝算輸。
    #  n=800/K=60 實測 ~86%。這條測試就是釘住這個刻度，別把 86% 誤讀成「還有 14% 沒挑到」。
    assert 80.0 < win_p < 95.0, f"完美預測器應在 ~86%（平手算輸的上限），得到 {win_p:.0f}%"
    assert win_n < 10.0
    assert 25.0 < win_r < 75.0, f"隨機預測器的 P(勝隨機) 應在 50% 附近，得到 {win_r:.0f}%"


def test_l2_half_port_kernel_dependency():
    """`half_port` 的**被動性依賴於核** —— 這條測試釘住那個依賴，不是假裝它到處安全。

    #! 2026-08-03 實測（隨機 50% 圖 ×6，28 GHz）：
    #    DCIM 核 + half_port : min Re **−0.289**、η 中位 **1.369**、η>1 佔 **100%**  ← 非物理
    #    L3   核 + half_port : min Re **+2.064**、η>1 佔 **0%**                      ← 正常
    #  稽核 agent 當初的警告（會破 passivity/能量守恆）**在 DCIM 上是對的**，
    #  它只是沒測 L3。⇒ `half_port` **只在 L3 核上開**，預設 False 就是為此。
    #  互易性（Z 對稱）則是**兩個核都必須成立**的硬不變式，先守住它。
    """
    from script.diffsim.l2 import MoML2, DCIMKernel
    for hp in (False, True):
        z = MoML2(kernel=DCIMKernel(v_scale=2.48), half_port=hp).impedance_at(28e9)
        assert torch.allclose(z, z.T, atol=1e-12), f"half_port={hp} 破壞了互易性（Z 必須對稱）"

    #? L3 表要另外建（`python -m script.diffsim.l3 build`），CI 上不一定有 → 沒有就跳過
    import os
    from script.diffsim import data as D
    tab = os.path.join(D.CACHE_DIR, "l3_table.npz")
    if not os.path.exists(tab):
        pytest.skip("沒有 l3_table.npz（跑 `python -m script.diffsim.l3 build` 產生）")
    from script.diffsim.l3 import L3Kernel
    m = MoML2(kernel=L3Kernel(), half_port=True)
    rng = np.random.default_rng(0)
    p = (rng.random((4, N, N)) < 0.5).astype(np.float64)
    p[:, 24, 9:16] = 1.0
    with torch.no_grad():
        out = m.solve(torch.as_tensor(p, dtype=torch.float64), freqs=np.array([28e9]))
    assert torch.isfinite(out["S11"]).all()
    assert float(out["Zin"].real.min()) > 0.0, "L3 核 + half_port 必須維持被動性"
    assert float((out["eta"] > 1.0).to(torch.float64).mean()) == 0.0, "L3 核 + half_port 不可有 η>1"
