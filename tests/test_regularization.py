"""正則化損失的純單元測試 (不需 HFSS / wandb / 網路)。"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from antenna.core.pattern import AntennaPattern
from antenna.losses.regularization import (
    GapClosingLoss,
    SpectralConnectivityLoss,
    total_variation_loss,
)

# FeedReachability 需要 scipy，環境未必安裝
try:
    import scipy  # noqa: F401

    from antenna.losses.regularization import FeedReachability

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    FeedReachability = None  # type: ignore

requires_scipy = pytest.mark.skipif(not HAS_SCIPY, reason="FeedReachability 需要 scipy")


# ---------------------------------------------------------------------------
# Fixtures (inline，避免動到 conftest.py)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _setup_pattern_coord():
    """AntennaPattern 是 class-level 狀態，呼叫 total_variation_loss 前需設座標。"""
    AntennaPattern.setDefaultCoordinate((0, 8, 0, 8))
    yield


@pytest.fixture
def zeros_4d() -> torch.Tensor:
    return torch.zeros(1, 1, 8, 8)


@pytest.fixture
def ones_4d() -> torch.Tensor:
    return torch.ones(1, 1, 8, 8)


@pytest.fixture
def checker_4d() -> torch.Tensor:
    """8x8 棋盤。"""
    h, w = 8, 8
    board = np.indices((h, w)).sum(axis=0) % 2
    return torch.as_tensor(board, dtype=torch.float32).view(1, 1, h, w)


# ---------------------------------------------------------------------------
# total_variation_loss
# ---------------------------------------------------------------------------


def test_tv_loss_zero_on_uniform(zeros_4d, ones_4d):
    assert total_variation_loss(zeros_4d).item() == pytest.approx(0.0)
    assert total_variation_loss(ones_4d).item() == pytest.approx(0.0)


def test_tv_loss_positive_on_checker(checker_4d):
    assert total_variation_loss(checker_4d).item() > 0.0


def test_tv_loss_weight_linear(checker_4d):
    base = total_variation_loss(checker_4d, weight=1.0).item()
    scaled = total_variation_loss(checker_4d, weight=2.5).item()
    assert scaled == pytest.approx(2.5 * base, rel=1e-5)


def test_tv_loss_gradient_flows():
    img = torch.rand(2, 1, 8, 8, requires_grad=True)
    loss = total_variation_loss(img, weight=0.1)
    loss.backward()
    assert img.grad is not None
    assert torch.isfinite(img.grad).all()


def test_tv_loss_gradient_reduces_loss(checker_4d):
    """沿負梯度方向走一步，loss 應該變小（驗證梯度方向正確）。"""
    img = checker_4d.clone().detach().requires_grad_(True)
    loss0 = total_variation_loss(img, weight=1.0)
    loss0.backward()
    with torch.no_grad():
        stepped = img - 0.1 * img.grad
    loss1 = total_variation_loss(stepped, weight=1.0)
    assert loss1.item() < loss0.item()


def test_tv_loss_accepts_2d_tensor():
    """size_converter 應可把 2D 張量轉為 (B, 1, H, W)。"""
    img = torch.rand(8, 8)
    loss = total_variation_loss(img)
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# SpectralConnectivityLoss
# ---------------------------------------------------------------------------


def test_spectral_connectivity_shape_small():
    """小尺寸避免 625x625 大矩陣，加快測試。"""
    sc = SpectralConnectivityLoss(height=5, width=5)
    img = torch.ones(1, 1, 5, 5)
    loss = sc(img)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_spectral_connectivity_full_metal_low_loss():
    """整片金屬應有高連通性 (lambda_2 大)，因此 loss 接近 0。"""
    sc = SpectralConnectivityLoss(height=5, width=5)
    full = torch.ones(1, 1, 5, 5)
    empty = torch.zeros(1, 1, 5, 5)
    loss_full = sc(full).item()
    loss_empty = sc(empty).item()
    assert loss_full <= loss_empty + 1e-6


def test_spectral_connectivity_gradient_flow():
    sc = SpectralConnectivityLoss(height=4, width=4)
    img = torch.rand(1, 1, 4, 4, requires_grad=True)
    loss = sc(img)
    loss.backward()
    assert img.grad is not None
    assert torch.isfinite(img.grad).all()


def test_spectral_connectivity_batch():
    sc = SpectralConnectivityLoss(height=4, width=4)
    img = torch.rand(3, 1, 4, 4)
    loss = sc(img)
    assert loss.ndim == 0


def test_spectral_connectivity_device_consistency():
    sc = SpectralConnectivityLoss(height=4, width=4)
    img = torch.rand(1, 1, 4, 4)
    loss = sc(img)
    assert loss.device == img.device


# ---------------------------------------------------------------------------
# GapClosingLoss
# ---------------------------------------------------------------------------


def test_gap_closing_zero_on_uniform(zeros_4d, ones_4d):
    gc = GapClosingLoss()
    assert gc(zeros_4d).item() == pytest.approx(0.0)
    assert gc(ones_4d).item() == pytest.approx(0.0)


def test_gap_closing_positive_on_hole():
    """中間挖洞後，closing 會填滿洞，因此 loss > 0。"""
    img = torch.ones(1, 1, 5, 5)
    img[0, 0, 2, 2] = 0.0
    gc = GapClosingLoss()
    assert gc(img).item() > 0.0


def test_gap_closing_gradient_flow():
    gc = GapClosingLoss()
    img = torch.rand(1, 1, 5, 5, requires_grad=True)
    loss = gc(img)
    loss.backward()
    assert img.grad is not None
    assert torch.isfinite(img.grad).all()


def test_gap_closing_gradient_reduces_loss():
    """有裂縫的圖上，沿負梯度走一步應能降低 loss。"""
    img = torch.ones(1, 1, 5, 5)
    img[0, 0, 2, 2] = 0.0
    img = img.requires_grad_(True)
    gc = GapClosingLoss()
    loss0 = gc(img)
    loss0.backward()
    with torch.no_grad():
        stepped = img - 0.5 * img.grad
    loss1 = gc(stepped)
    assert loss1.item() < loss0.item()


# ---------------------------------------------------------------------------
# FeedReachability
# ---------------------------------------------------------------------------


@requires_scipy
def test_feed_reachability_empty_pattern():
    """全 0 pattern：饋電點沒金屬，應回 0.0 且不 crash。"""
    fr = FeedReachability([(0, 0)])
    pattern = np.zeros((5, 5), dtype=np.float32)
    rate = fr(pattern)
    assert rate == 0.0


@requires_scipy
def test_feed_reachability_full_metal_single_feed():
    """全 1 pattern + 單饋電點：整張圖都連通，rate = 1.0。"""
    fr = FeedReachability([(0, 0)])
    pattern = np.ones((5, 5), dtype=np.float32)
    rate = fr(pattern)
    assert rate == pytest.approx(1.0)


@requires_scipy
def test_feed_reachability_two_feeds_same_block():
    """兩個饋電點落在同一連通塊上。"""
    fr = FeedReachability([(0, 0), (4, 4)])
    pattern = np.ones((5, 5), dtype=np.float32)
    rate = fr(pattern)
    assert rate == pytest.approx(1.0)


@requires_scipy
def test_feed_reachability_two_feeds_separated():
    """兩個饋電點分屬於不同連通塊，rate 應為 0.0。"""
    fr = FeedReachability([(0, 0), (4, 4)])
    pattern = np.zeros((5, 5), dtype=np.float32)
    pattern[0, 0] = 1.0  # 左上孤立 pixel
    pattern[4, 4] = 1.0  # 右下孤立 pixel
    rate = fr(pattern)
    assert rate == 0.0


@requires_scipy
def test_feed_reachability_partial_connectivity():
    """饋電點在大塊中，但只佔總金屬的一半。"""
    fr = FeedReachability([(0, 0)])
    pattern = np.zeros((5, 5), dtype=np.float32)
    pattern[0:2, 0:2] = 1.0  # 4 pixels connected with feed
    pattern[4, 4] = 1.0  # 1 isolated pixel
    rate = fr(pattern)
    assert rate == pytest.approx(4.0 / 5.0)


@requires_scipy
def test_feed_reachability_out_of_bounds():
    """饋電點越界：回傳 0.0 而不是崩潰。"""
    fr = FeedReachability([(100, 100)])
    pattern = np.ones((5, 5), dtype=np.float32)
    rate = fr(pattern)
    assert rate == 0.0


@requires_scipy
def test_feed_reachability_accepts_tensor():
    """傳入 Tensor 應自動轉 ndarray。"""
    fr = FeedReachability([(0, 0)])
    pattern = torch.ones(5, 5)
    rate = fr(pattern)
    assert rate == pytest.approx(1.0)


@requires_scipy
def test_feed_reachability_record_accumulates():
    """record=True 時，每次呼叫應該都附加到 self.record。"""
    fr = FeedReachability([(0, 0)])
    pattern = np.ones((3, 3), dtype=np.float32)
    fr(pattern, record=True)
    fr(pattern, record=True)
    assert len(fr.record) == 2
    assert fr.r_feed_list == [1.0, 1.0]
    assert fr.rate_list == [100.0, 100.0]
    assert fr.r_feed_avg == pytest.approx(1.0)


@requires_scipy
def test_feed_reachability_record_default_off():
    fr = FeedReachability([(0, 0)])
    pattern = np.ones((3, 3), dtype=np.float32)
    fr(pattern)
    assert fr.record == []


@requires_scipy
def test_feed_reachability_mask_after_success():
    """成功呼叫後 mask 應為 ndarray 而非初始 None/0。"""
    fr = FeedReachability([(0, 0)])
    pattern = np.ones((3, 3), dtype=np.float32)
    fr(pattern)
    assert isinstance(fr.mask, np.ndarray)
    assert fr.mask.shape == pattern.shape


@requires_scipy
def test_feed_reachability_single_feed_factory():
    fr = FeedReachability.single_feed()
    assert len(fr.feed_positions) == 1


@requires_scipy
def test_feed_reachability_dual_feed_factory():
    fr = FeedReachability.dual_feed()
    assert len(fr.feed_positions) == 2


@requires_scipy
def test_feed_reachability_r_feed_avg_empty():
    """沒紀錄時取平均不該 crash。"""
    fr = FeedReachability([(0, 0)])
    assert fr.r_feed_avg == 0.0


@requires_scipy
def test_feed_reachability_r_feed_dict_groups_by_title():
    fr = FeedReachability([(0, 0)])
    pattern = np.ones((3, 3), dtype=np.float32)
    fr(pattern, record=True, title="A-{rate:.2%}")
    fr(pattern, record=True, title="A-{rate:.2%}")
    grouped = fr.r_feed_dict
    assert len(grouped) == 1
    # 唯一的 key 對應兩筆紀錄
    assert len(next(iter(grouped.values()))) == 2


@requires_scipy
def test_feed_reachability_r_feed_str_accessible_on_instance_and_class():
    """train_single.py 會透過 instance 讀取 r_feed_str，必須保持公開介面。"""
    fr = FeedReachability([(0, 0)])
    assert FeedReachability.r_feed_str == fr.r_feed_str
    assert "R_" in fr.r_feed_str
