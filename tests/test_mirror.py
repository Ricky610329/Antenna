"""``antenna.losses.mirror`` 的純單元測試（不需 HFSS / wandb / 網路）。"""

import pytest
import torch

from antenna.losses.mirror import FlipMode, gumbel_sinkhorn_rectangular, mirror

# ---------------------------------------------------------------------------
# mirror() 行為
# ---------------------------------------------------------------------------


def _is_vertically_symmetric(t: torch.Tensor) -> bool:
    """上下對稱：t == flip(t, dim=0)。"""
    return torch.allclose(t, torch.flip(t, dims=[0]))


def _is_horizontally_symmetric(t: torch.Tensor) -> bool:
    """左右對稱：t == flip(t, dim=1)。"""
    return torch.allclose(t, torch.flip(t, dims=[1]))


@pytest.mark.parametrize("shape", [(4, 4), (5, 5), (4, 6), (5, 7), (6, 5)])
def test_mirror_horizontal_returns_two_horizontally_symmetric_tensors(shape):
    x = torch.randn(shape)
    results = mirror(x, mode="|")

    assert len(results) == 2
    for r in results:
        assert r.shape == x.shape
        assert _is_horizontally_symmetric(r)


@pytest.mark.parametrize("shape", [(4, 4), (5, 5), (4, 6), (5, 7), (6, 5)])
def test_mirror_vertical_returns_two_vertically_symmetric_tensors(shape):
    x = torch.randn(shape)
    results = mirror(x, mode="-")

    assert len(results) == 2
    for r in results:
        assert r.shape == x.shape
        assert _is_vertically_symmetric(r)


@pytest.mark.parametrize("shape", [(4, 4), (5, 5), (4, 6), (5, 7), (6, 5)])
def test_mirror_both_returns_four_fully_symmetric_tensors(shape):
    x = torch.randn(shape)
    results = mirror(x, mode="*")

    assert len(results) == 4
    for r in results:
        assert r.shape == x.shape
        # 同時上下 / 左右對稱
        assert _is_horizontally_symmetric(r)
        assert _is_vertically_symmetric(r)


def test_mirror_combined_modes_accumulate_results():
    x = torch.randn(4, 4)
    # "-|*" 應該回傳 2 (vertical) + 2 (horizontal) + 4 (both) = 8 個 tensor
    results = mirror(x, mode="-|*")
    assert len(results) == 8


def test_mirror_combined_modes_deduplicates_characters():
    x = torch.randn(4, 4)
    # 重複的字元不該讓結果數量加倍
    assert len(mirror(x, mode="||")) == 2
    assert len(mirror(x, mode="--")) == 2
    assert len(mirror(x, mode="**")) == 4


def test_mirror_accepts_flipmode_enum():
    x = torch.randn(4, 4)
    results = mirror(x, mode=FlipMode.horizontal)
    assert len(results) == 2
    for r in results:
        assert _is_horizontally_symmetric(r)


def test_mirror_invalid_mode_raises():
    x = torch.randn(4, 4)
    with pytest.raises(ValueError):
        mirror(x, mode="x")


def test_mirror_preserves_gradient_flow():
    x = torch.randn(4, 4, requires_grad=True)
    results = mirror(x, mode="*")
    # 對所有結果求和後反傳
    loss = sum(r.sum() for r in results)
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


# ---------------------------------------------------------------------------
# gumbel_sinkhorn_rectangular()
# ---------------------------------------------------------------------------


def test_gumbel_sinkhorn_soft_assignment_is_doubly_stochastic():
    """Soft 輸出在 square 情況下應近似 doubly stochastic。"""
    torch.manual_seed(0)
    logits = torch.randn(6, 6)
    out = gumbel_sinkhorn_rectangular(logits, tau=1.0, n_iters=50, hard=False)

    assert out.shape == logits.shape
    # Sinkhorn 的最後一輪為沿 K (倒數第二軸) 正規化，對應每個欄位和為 1
    col_sums = out.sum(dim=-2)
    assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-4)
    # 對方陣，列和也應接近 1（Sinkhorn 收斂性質）
    row_sums = out.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-2)


def test_gumbel_sinkhorn_soft_output_in_unit_interval():
    logits = torch.randn(4, 4)
    out = gumbel_sinkhorn_rectangular(logits, tau=1.0, n_iters=20, hard=False)
    assert torch.all(out >= 0)
    assert torch.all(out <= 1)


def test_gumbel_sinkhorn_hard_returns_one_hot_rows():
    logits = torch.randn(5, 7)
    out = gumbel_sinkhorn_rectangular(logits, tau=1.0, n_iters=20, hard=True)

    assert out.shape == (5, 7)
    # 每一列恰好有一個 1
    assert torch.all(out.sum(dim=-1) == 1)
    assert torch.all((out == 0) | (out == 1))


def test_gumbel_sinkhorn_supports_batch_dim():
    logits = torch.randn(3, 4, 4)
    out = gumbel_sinkhorn_rectangular(logits, tau=1.0, n_iters=20, hard=False)
    assert out.shape == logits.shape
    col_sums = out.sum(dim=-2)
    assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-4)


def test_gumbel_sinkhorn_preserves_gradient_flow():
    logits = torch.randn(4, 4, requires_grad=True)
    out = gumbel_sinkhorn_rectangular(logits, tau=1.0, n_iters=10, hard=False)
    out.sum().backward()
    assert logits.grad is not None
    assert logits.grad.shape == logits.shape
