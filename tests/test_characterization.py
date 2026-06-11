"""
核心純函式的特徵化測試（重構安全網）。

這些函式在重構後「不應改變行為」。語意斷言 + golden 快照雙保險：
- 語意斷言：不需 magic number，直接驗證該有的性質（確定性、二值、梯度、邊界值）。
- golden 快照：把現況數值寫進 tests/golden.json，重構後若漂移即 fail。

特別重點：binarization 的 STE 與「顯式 tau 覆寫」行為——這是接下來
「tau 去耦合」步驟要保留的契約，先在此 pin 住。
"""
import torch

from antenna import AntennaPattern, AntennaResponse, MultiResponses
from antenna.losses import SpectralConnectivityLoss, GapClosingLoss, FeedReachability
from antenna.patch import custom_loss_minmax, interval_loss


def _fixed_binary_pattern(seed=0):
    """確定性的 25x25 二元 pattern。"""
    torch.manual_seed(seed)
    return (torch.rand(25, 25) > 0.5).float()


# ===================== STE 二值化 =====================

def test_binarization_forward_is_binary():
    """forward 必須輸出乾淨的 0/1（STE 的 hard 路徑）。"""
    torch.manual_seed(1)
    x = torch.randn(25, 25)
    out = AntennaPattern.binarization(x.clone(), tau=1.0).detach()
    uniq = set(out.unique().tolist())
    assert uniq.issubset({0.0, 1.0}), f"輸出非二元：{uniq}"
    assert out.shape == (25, 25)


def test_binarization_explicit_tau_matches_manual():
    """顯式 tau 必須等於『手算 STE(tau)』。tau 去耦合後 binarization 只用傳入的 tau (無全域 cls.tau)。"""
    torch.manual_seed(2)
    x = torch.randn(25, 25)
    tau = 0.5

    out = AntennaPattern.binarization(x.clone(), tau=tau).detach()

    # 手算現行邏輯：clamp→以(clamp後)均值為閾值→sigmoid(1/tau*(x-thr))→round
    xc = torch.clamp(x, -10.0, 10.0)
    thr = xc.mean()
    soft = torch.sigmoid((1.0 / tau) * (xc - thr))
    hard = torch.round(soft)
    assert torch.equal(out, hard), "顯式 tau 的 forward 值與手算不符"


def test_binarization_gradient_flows():
    """STE：backward 必須讓梯度流回輸入，且有限。"""
    x = torch.randn(25, 25, requires_grad=True)
    out = AntennaPattern.binarization(x, tau=1.0)
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0


def test_binarization_deterministic():
    """同輸入同 tau → 同輸出。"""
    torch.manual_seed(3)
    x = torch.randn(25, 25)
    a = AntennaPattern.binarization(x.clone(), tau=1.0).detach()
    b = AntennaPattern.binarization(x.clone(), tau=1.0).detach()
    assert torch.equal(a, b)


def test_binarization_only_soft_snapshot(golden):
    """only_soft 軟近似的數值快照。"""
    torch.manual_seed(4)
    x = torch.randn(25, 25)
    soft = AntennaPattern.binarization(x.clone(), tau=1.0, only_soft=True)
    golden.check("binarization_soft_mean", soft.mean().item())
    golden.check("binarization_soft_sum", soft.sum().item())


# ===================== pattern 正則化 loss =====================

def test_tv_loss_uniform_is_zero():
    """全金屬 pattern 無變化 → TV loss = 0。"""
    p = AntennaPattern(torch.ones(25, 25))
    assert p.total_variation_loss(weight=1.0).item() == 0.0


def test_island_suppression_uniform_snapshot(golden):
    """全金屬 pattern 的孤島 loss 並非 0：avg_pool2d 的 padding 補零，使邊界像素
    的局部平均被稀釋 (<1)，邊界金屬因此被罰。此處 pin 住該『邊界零填充效應』的現況值。"""
    p = AntennaPattern(torch.ones(25, 25))
    val = p.island_suppression_loss(weight=1.0).item()
    assert val > 0  # 邊界零填充導致非零（非預期中的 0）
    golden.check("island_uniform", val)


def test_tv_and_island_snapshot(golden):
    p = AntennaPattern(_fixed_binary_pattern())
    golden.check("tv_loss", p.total_variation_loss(weight=1.0).item())
    golden.check("island_loss", p.island_suppression_loss(weight=1.0).item())


def test_spectral_connectivity_snapshot(golden):
    """SC Loss（論文主方法）的數值快照；連通 pattern 的 1/λ2 應為有限正數。"""
    sc = SpectralConnectivityLoss()
    pat = _fixed_binary_pattern().unsqueeze(0).unsqueeze(0)  # (1,1,25,25)
    val = sc.forward(pat)
    assert torch.isfinite(val).all() and val.item() > 0
    golden.check("sc_loss", val.item())


def test_gap_closing_snapshot(golden):
    gc = GapClosingLoss()
    pat = _fixed_binary_pattern().unsqueeze(0).unsqueeze(0)
    val = gc.forward(pat)
    assert torch.isfinite(val).all() and val.item() >= 0
    golden.check("gap_closing_loss", val.item())


# ===================== FeedReachability（評估指標）=====================

def test_feed_reachability_full_metal_is_one():
    """全金屬 → 饋電點所在連通塊 = 全部 → R_feed = 1.0。"""
    r = FeedReachability.single_feed()
    rate = r(torch.ones(25, 25))
    assert abs(float(rate) - 1.0) < 1e-9


def test_feed_reachability_snapshot(golden):
    r = FeedReachability.single_feed()
    rate = r(_fixed_binary_pattern())
    assert 0.0 <= float(rate) <= 1.0
    golden.check("r_feed_single", float(rate))


# ===================== loss hooks =====================

def test_custom_loss_minmax_perfect_vs_bad():
    """method='low'：預測在目標最低點處『不高於目標』→ 0；偏高 → >0。"""
    target = AntennaResponse.target["S11"].response
    # 完美達標：直接用目標本身（最低點處 pred==target，不高於 → 0）
    loss_ok = custom_loss_minmax(target.clone(), target, method="low")
    assert loss_ok.item() == 0.0
    # 故意把整條拉高 → 最低點處偏高 → 受罰
    loss_bad = custom_loss_minmax(target + 5.0, target, method="low")
    assert loss_bad.item() > 0.0


def test_interval_loss_in_vs_out():
    """interval_loss：落在 [target-1, target+1] 內 → 0；超出 → >0。"""
    target = torch.zeros(17)
    inside = custom = interval_loss(torch.zeros(17), -1.0, 1.0, target=target)
    assert inside.item() == 0.0
    outside = interval_loss(torch.full((17,), 5.0), -1.0, 1.0, target=target)
    assert outside.item() > 0.0


def test_loss_hooks_snapshot(golden):
    target = AntennaResponse.target["Gain"].response
    torch.manual_seed(5)
    pred = target + torch.randn(target.shape)
    golden.check("minmax_gain_high", custom_loss_minmax(pred, target, method="high").item())
    golden.check("interval_gain", interval_loss(pred, -1.0, 1.0, target=target).item())
