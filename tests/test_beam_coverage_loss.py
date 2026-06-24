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


def test_flatness_weight_zero_is_noop():
    """flatness_weight 預設 0 → ③ 項完全不影響 (golden 安全)；帶內小起伏的 loss 仍是 0。"""
    #  G0=5，窗內 [4,4,5,4,4]：① ② 都 0（介於 floor 與 G0 之間）→ 不加 ③ 時 loss=0。
    rad = torch.tensor([9.0, 4, 4, 5, 4, 4, 9])
    assert beam_coverage_loss(rad, THETA, **KW).item() == pytest.approx(0.0)
    assert beam_coverage_loss(rad, THETA, flatness_weight=0.0, **KW).item() == pytest.approx(0.0)


def test_flatness_penalizes_deviation_from_boresight():
    """flatness_weight>0：對「① ② 都不罰」的帶內起伏，③ 仍依對 G0 的偏差平方給罰，值＝手算。"""
    #  G0=5，窗內 [4,4,5,4,4]，偏差平方 [1,1,0,1,1]，mean=4/5=0.8 → loss = 0+0+1.0·0.8。
    rad = torch.tensor([9.0, 4, 4, 5, 4, 4, 9])
    assert beam_coverage_loss(rad, THETA, flatness_weight=1.0, **KW).item() == pytest.approx(0.8)
    # 線性縮放
    assert beam_coverage_loss(rad, THETA, flatness_weight=0.5, **KW).item() == pytest.approx(0.4)


def test_flatness_zero_for_perfect_flat_top():
    """完全平頂 (窗內全 = G0)：③ 偏差平方全 0 → 不論 flatness_weight 多大，loss 都 0。"""
    rad = torch.tensor([5.0, 5, 5, 5, 5, 5, 5])
    assert beam_coverage_loss(rad, THETA, flatness_weight=10.0, **KW).item() == pytest.approx(0.0)


def test_flatness_weight_zero_inf_not_nan():
    """Bug 1 回歸：flatness_weight=0 + 窗內含 inf → 短路不算 (pred-g0)^2，不會 0*inf=NaN；
    值＝只跑 ①② 的行為＝inf (改動前此例會回 NaN)。"""
    rad = torch.tensor([5.0, 5, float("inf"), 5, 5, 5, 5])   # θ=-30 (窗內) = inf
    loss = beam_coverage_loss(rad, THETA, flatness_weight=0.0, **KW)
    assert not torch.isnan(loss)
    assert torch.isinf(loss)


def test_flatness_anchors_per_phi_cut():
    """flatness 每個 phi 切面各自錨自己的 G0 (非全域)；隔離 ③ (boresight_weight=0)。"""
    #  cut0 G0=10，偏差平方 [0,4,0,4,0]；cut1 全 2 → 全 0。mean over 2×5=10 = 8/10 = 0.8。
    rad = torch.tensor([[10.0, 12, 10, 8, 10],
                        [2.0, 2, 2, 2, 2]])
    theta = torch.tensor([-55.0, -30, 0, 30, 55])
    loss = beam_coverage_loss(rad, theta, window_deg=55, floor_db=3,
                              boresight_weight=0.0, flatness_weight=1.0)
    assert loss.item() == pytest.approx(0.8)


def test_flatness_theta_unsorted():
    """θ 為 HFSS 匯出序 (未排序)：g0 仍取 |θ| 最小欄、flatness 逐欄對位正確。"""
    #  g0=θ0(idx0)=5；窗內 idx0-4=[5,4,4,3,3]，偏差平方 [0,1,1,4,4]=10，mean/5=2.0 (①② 此例皆 0)。
    theta = torch.tensor([0.0, 30, -30, 55, -55, 60, -60])
    rad = torch.tensor([5.0, 4, 4, 3, 3, 9, 9])
    loss = beam_coverage_loss(rad, theta, window_deg=55, floor_db=3,
                              boresight_weight=1.0, flatness_weight=1.0)
    assert loss.item() == pytest.approx(2.0)


def test_flatness_reduction_sum():
    """reduction='sum' 也乘了 flatness_weight、不漏項。"""
    #  G0=5，窗內 [5,4,5,4,5]，偏差平方 [0,1,0,1,0] sum=2 (①②=0) → loss=2.0。
    rad = torch.tensor([5.0, 5, 4, 5, 4, 5, 5])
    loss = beam_coverage_loss(rad, THETA, reduction="sum", flatness_weight=1.0, **KW)
    assert loss.item() == pytest.approx(2.0)


def test_flatness_pulls_g0_toward_window_mean():
    """設計取捨明文化：flatness 對 g0 也微分 → 把 g0 往窗內均值拉 (與 ② 的『0°最高』對抗)。
    隔離 ③ (boresight_weight=0)：峰 (g0) 梯度為正 (被往下拉)、窗內低點梯度為負 (被往上拉)。"""
    theta = torch.tensor([-40.0, -20, 0, 20, 40])
    rad = torch.tensor([3.0, 3, 5, 3, 3], requires_grad=True)   # g0=idx2=5 為峰
    loss = beam_coverage_loss(rad, theta, window_deg=55, floor_db=3,
                              boresight_weight=0.0, flatness_weight=1.0)
    loss.backward()
    assert rad.grad[2].item() > 0       # 峰 (g0) 被往下拉
    assert rad.grad[1].item() < 0       # 低點被往上拉 → 趨平


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
