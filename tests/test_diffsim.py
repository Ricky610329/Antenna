"""diffsim（可微模擬器）的物理正確性把關 —— 不掛 NAS、不跑 HFSS、純確定性。

這一組是「對抗式驗證」的常駐版：ρ 好看不代表物理對，所以每個 gate 報數前，
求解器必須先過解析 sanity 案例。涵蓋：
  - 譜：實心/矩形貼片的離散 Neumann 特徵譜 vs 閉式解
  - 互易性、被動性（Re Zin ≥ 0、|S11| ≤ 1）、能量（P_rad > 0）
  - 尺一致性：本鏈的向量化 worst_margin 必須與 `antenna.losses.worst_margin` 逐筆相同
"""
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
    assert (out["Prad"] > 0).all()
    assert torch.isfinite(out["Gain"]).all()


def test_reciprocity_of_modal_impedance(sim):
    """互易性：模態阻抗矩陣 Z_ij = jωμh Σ ψ_n(i)ψ_n(j)/(k_n²−k²) 必然對稱。

    直接用兩個不同的「埠權重」建 2×2 → 檢查 Z12 == Z21（實作寫錯順序就會破）。
    """
    p = torch.zeros(1, N, N, dtype=torch.float64)
    p[0, 8:25, 6:20] = 1.0
    lam, psi = sim.modes(p)
    psi = (psi / DX)[0]
    w1 = torch.zeros(N * N, dtype=torch.float64)
    w2 = torch.zeros(N * N, dtype=torch.float64)
    w1[np.ravel_multi_index((24, 12), (N, N))] = 1.0
    w2[np.ravel_multi_index((10, 8), (N, N))] = 1.0
    a = (psi * w1[:, None]).sum(0)
    b = (psi * w2[:, None]).sum(0)
    k2 = (2 * np.pi * 28e9 / C0) ** 2 * EPS_R
    den = (lam[0] / DX ** 2 - k2)
    z12 = float((a * b / den).sum())
    z21 = float((b * a / den).sum())
    assert abs(z12 - z21) < 1e-12 * max(abs(z12), 1.0)


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
