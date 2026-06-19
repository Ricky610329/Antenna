"""
tests/test_beam_coverage_loss.py — 方向圖覆蓋損失的純函式單元測試 (CPU、無 HFSS、不碰 golden)。

beam_coverage_loss 是 Stage 1：把方向圖塑成「相對 boresight 的平頂 + 中央峰」。
本檔用手算過的小角度網格，逐項驗證 floor / boresight / 窗 / 單邊 / 可微 等行為。
"""
import pytest
import torch

from antenna.losses import beam_coverage_loss

#? 共用小網格：θ = ±60/±55/±30/0；window=55 → ±60 落在窗外 (用來驗窗)。
THETA = torch.tensor([-60.0, -55.0, -30.0, 0.0, 30.0, 55.0, 60.0])
KW = dict(window_deg=55.0, floor_db=3.0, boresight_weight=1.0)


def test_perfect_flat_top_is_zero():
    """窗內全部等於 boresight → floor、boresight 兩項都 0。"""
    rad = torch.tensor([5.0, 5, 5, 5, 5, 5, 5])
    assert beam_coverage_loss(rad, THETA, **KW).item() == pytest.approx(0.0)


def test_floor_penalty_value():
    """窗內某角度比 G0−floor_db 還低 → 只有 floor 項貢獻，值＝手算。"""
    #  G0=5, floor=2；窗內 [5,5,5,0,5]，deficit=[0,0,0,2,0]，mean/5 = 0.4。
    #  θ=±60 的 10 在窗外，應被忽略 (順帶驗窗)。
    rad = torch.tensor([10.0, 5, 5, 5, 0, 5, 10])
    assert beam_coverage_loss(rad, THETA, **KW).item() == pytest.approx(0.4)


def test_boresight_penalty_value():
    """窗內某角度高過 boresight → 只有 boresight 項貢獻，值＝手算。"""
    #  G0=5；窗內 [5,8,5,5,5]，excess=[0,3,0,0,0]，mean/5 = 0.6。
    rad = torch.tensor([5.0, 5, 8, 5, 5, 5, 5])
    assert beam_coverage_loss(rad, THETA, **KW).item() == pytest.approx(0.6)


def test_between_floor_and_boresight_is_free():
    """窗內介於 (G0−floor_db, G0) 之間 → 單邊，不罰 (這就是「越高越好」)。"""
    #  G0=5，窗內 [4,4,5,4,4]：floor relu(2-·)=0、boresight relu(·-5)=0 → 0。
    rad = torch.tensor([9.0, 4, 4, 5, 4, 4, 9])
    assert beam_coverage_loss(rad, THETA, **KW).item() == pytest.approx(0.0)


def test_out_of_window_dip_ignored():
    """只在窗外 (±60) 大幅下陷 → loss 不受影響 (= 0)。"""
    rad = torch.tensor([0.0, 5, 5, 5, 5, 5, 0])
    assert beam_coverage_loss(rad, THETA, **KW).item() == pytest.approx(0.0)


def test_boresight_weight_scales_only_boresight_term():
    """boresight_weight 線性縮放 boresight 項 (floor 項不受影響)。"""
    rad = torch.tensor([5.0, 5, 8, 5, 5, 5, 5])     # 純 boresight 違規 = 0.6
    loss = beam_coverage_loss(rad, THETA, window_deg=55, floor_db=3, boresight_weight=2.0)
    assert loss.item() == pytest.approx(1.2)


def test_two_phi_cuts_average_over_all_window_elements():
    """(n_phi, n_theta)：對兩切面所有窗內元素一起平均。"""
    #  cut0 有 excess [0,3,0,0,0]、cut1 全 0 → mean over 2×5=10 → 3/10 = 0.3。
    rad = torch.tensor([[5.0, 5, 8, 5, 5, 5, 5],
                        [5.0, 5, 5, 5, 5, 5, 5]])
    assert beam_coverage_loss(rad, THETA, **KW).item() == pytest.approx(0.3)


def test_1d_equals_2d_single_row():
    """(n_theta,) 與 (1, n_theta) 結果一致。"""
    rad1d = torch.tensor([5.0, 5, 8, 5, 5, 5, 5])
    rad2d = rad1d.unsqueeze(0)
    a = beam_coverage_loss(rad1d, THETA, **KW).item()
    b = beam_coverage_loss(rad2d, THETA, **KW).item()
    assert a == pytest.approx(b)


def test_reduction_sum():
    """reduction='sum'：floor 單一下陷 2 → 總和 2.0 (不除元素數)。"""
    rad = torch.tensor([10.0, 5, 5, 5, 0, 5, 10])
    assert beam_coverage_loss(rad, THETA, reduction="sum", **KW).item() == pytest.approx(2.0)


def test_differentiable_and_pushes_sidelobe_down():
    """可微：backward 後梯度有限；超出 boresight 的旁瓣梯度為正 (會被往下壓)。"""
    rad = torch.tensor([5.0, 5, 8, 5, 5, 5, 5], requires_grad=True)
    loss = beam_coverage_loss(rad, THETA, **KW)
    loss.backward()
    assert rad.grad is not None
    assert torch.isfinite(rad.grad).all()
    assert rad.grad[2].item() > 0          # θ=-30 的旁瓣 (值 8 > G0) 應被往下推


def test_window_excludes_all_raises():
    """窗內沒有任何取樣點 → 明確報錯，不靜默回 0。"""
    theta = torch.tensor([100.0, 200.0])
    rad = torch.tensor([1.0, 2.0])
    with pytest.raises(ValueError):
        beam_coverage_loss(rad, theta, window_deg=55)


def test_theta_length_mismatch_raises():
    """theta 長度與 rad_pred 的 n_theta 不符 → 報錯。"""
    rad = torch.tensor([5.0, 5, 5, 5, 5, 5, 5])     # n_theta = 7
    with pytest.raises(ValueError):
        beam_coverage_loss(rad, torch.tensor([0.0, 1, 2]), window_deg=55)


def test_bad_reduction_raises():
    rad = torch.tensor([5.0, 5, 5, 5, 5, 5, 5])
    with pytest.raises(ValueError):
        beam_coverage_loss(rad, THETA, reduction="median")
