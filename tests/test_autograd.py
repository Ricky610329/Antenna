"""`antenna.models.autograd` 單元測試。

測試目標：
- `sign_f`：forward 為 sign()，backward 使用 STE（梯度在 |x|>1 時被 clip 為 0）
- `GumbelSigmoid`：forward 輸出在 [0, 1] 之間且溫度極低時趨近 {0, 1}，backward 有 gradient flow
- `BinarizeSTE`：forward 為 0/1，backward 為 `grad_output * mask`（mask-aware STE）
- 各函式在不同 input shape 下維持形狀一致且穩定
"""

from __future__ import annotations

import pytest
import torch

from antenna.models.autograd import (
    BinarizeSTE,
    GumbelSigmoid,
    _GumbelSigmoid,
    sign_f,
)

# --- sign_f ---------------------------------------------------------------


class TestSignF:
    """sign_f：forward 為 sign，backward 為 STE（|x|>1 時梯度為 0）。"""

    def test_forward_values(self):
        x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
        y = sign_f.apply(x)
        assert torch.equal(y, torch.tensor([-1.0, -1.0, 1.0, 1.0, 1.0]))

    def test_forward_preserves_shape(self):
        for shape in [(1,), (5,), (3, 4), (2, 3, 4)]:
            x = torch.randn(*shape)
            y = sign_f.apply(x)
            assert y.shape == x.shape
            assert torch.all((y == 1.0) | (y == -1.0))

    def test_backward_ste_in_range(self):
        """|x| <= 1 時梯度 pass-through。"""
        x = torch.tensor([-0.5, 0.0, 0.3, 0.9], requires_grad=True)
        y = sign_f.apply(x)
        # 使用非 1 的上游梯度以避免 in-place 修改 1 的 tensor
        grad_out = torch.tensor([2.0, 3.0, 4.0, 5.0])
        y.backward(grad_out)
        assert torch.allclose(x.grad, grad_out)

    def test_backward_ste_clipped_out_of_range(self):
        """|x| > 1 時梯度被 clip 為 0。"""
        x = torch.tensor([-2.0, -1.5, 0.0, 1.5, 2.0], requires_grad=True)
        y = sign_f.apply(x)
        grad_out = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
        y.backward(grad_out)
        expected = torch.tensor([0.0, 0.0, 30.0, 0.0, 0.0])
        assert torch.allclose(x.grad, expected)

    def test_backward_finite_diff_consistency(self):
        """在 |x|<1 範圍，STE 梯度應等於 forward 為 identity 時的 finite diff。

        嚴格 `gradcheck` 不適用 STE（forward 是 sign、backward 是 identity，兩者不匹配）。
        此處驗證「STE 在 linear 使用下會讓 downstream loss 對 x 的梯度 = downstream gradient」。
        """
        x = torch.tensor([-0.3, 0.1, 0.5], requires_grad=True)
        y = sign_f.apply(x)
        loss = (y * torch.tensor([1.0, 2.0, 3.0])).sum()
        loss.backward()
        assert torch.allclose(x.grad, torch.tensor([1.0, 2.0, 3.0]))


# --- GumbelSigmoid --------------------------------------------------------


class TestGumbelSigmoid:
    """GumbelSigmoid：forward 為 gumbel-sigmoid 軟採樣，backward 有 gradient flow。"""

    def test_forward_output_in_unit_interval(self):
        torch.manual_seed(0)
        logits = torch.randn(10, 10)
        tau = torch.tensor(1.0)
        y = GumbelSigmoid.apply(logits, tau)
        assert torch.all(y >= 0.0)
        assert torch.all(y <= 1.0)

    def test_forward_preserves_shape(self):
        tau = torch.tensor(1.0)
        for shape in [(4,), (3, 5), (2, 3, 4)]:
            logits = torch.randn(*shape)
            y = GumbelSigmoid.apply(logits, tau)
            assert y.shape == logits.shape

    def test_forward_low_tau_approximates_binary(self):
        """tau 極小時，輸出應趨近 0 或 1。"""
        torch.manual_seed(42)
        logits = torch.randn(100) * 5.0  # 放大讓 sign 穩定
        tau = torch.tensor(1e-3)
        y = GumbelSigmoid.apply(logits, tau)
        # 絕大多數應該接近 0 或 1
        near_binary = ((y < 0.05) | (y > 0.95)).float().mean().item()
        assert near_binary > 0.9

    def test_backward_has_gradient_flow(self):
        torch.manual_seed(0)
        logits = torch.randn(4, 4, requires_grad=True)
        tau = torch.tensor(1.0, requires_grad=True)
        y = GumbelSigmoid.apply(logits, tau)
        loss = y.sum()
        loss.backward()
        assert logits.grad is not None
        assert tau.grad is not None
        assert torch.isfinite(logits.grad).all()
        assert torch.isfinite(tau.grad).all()
        # logits 梯度應非零（至少大部分元素）
        assert (logits.grad.abs() > 0).any()

    def test_backward_logits_grad_formula(self):
        """logits 梯度 = grad_output * y*(1-y) / tau。"""
        torch.manual_seed(1)
        logits = torch.randn(3, 3, requires_grad=True)
        tau = torch.tensor(0.5, requires_grad=True)
        y = GumbelSigmoid.apply(logits, tau)
        grad_out = torch.ones_like(y)
        y.backward(grad_out)
        expected = grad_out * y.detach() * (1 - y.detach()) / tau.detach()
        assert torch.allclose(logits.grad, expected, atol=1e-6)

    def test_stochasticity(self):
        """相同 input 多次呼叫應產生不同輸出（由於 Gumbel noise）。"""
        logits = torch.zeros(50)
        tau = torch.tensor(1.0)
        y1 = GumbelSigmoid.apply(logits, tau)
        y2 = GumbelSigmoid.apply(logits, tau)
        assert not torch.allclose(y1, y2)


# --- _GumbelSigmoid -------------------------------------------------------


class TestPrivateGumbelSigmoid:
    """_GumbelSigmoid：另一個 Gumbel 實作（無 scale、eps 較大）。"""

    def test_forward_output_in_unit_interval(self):
        torch.manual_seed(0)
        logits = torch.randn(5, 5)
        tau = torch.tensor(1.0)
        y = _GumbelSigmoid.apply(logits, tau)
        assert torch.all(y >= 0.0)
        assert torch.all(y <= 1.0)

    def test_backward_has_gradient_flow(self):
        torch.manual_seed(0)
        logits = torch.randn(3, 3, requires_grad=True)
        tau = torch.tensor(1.0)
        y = _GumbelSigmoid.apply(logits, tau)
        loss = y.sum()
        loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()


# --- BinarizeSTE ----------------------------------------------------------


class TestBinarizeSTE:
    """BinarizeSTE：forward 為 (x >= 0.5) float；backward 為 grad_output * mask。"""

    def test_forward_binary_output(self):
        x = torch.tensor([0.0, 0.3, 0.5, 0.7, 1.0])
        y = BinarizeSTE.apply(x)
        assert torch.equal(y, torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0]))

    def test_forward_preserves_shape(self):
        for shape in [(1,), (5,), (3, 4), (2, 3, 4)]:
            x = torch.rand(*shape)
            y = BinarizeSTE.apply(x)
            assert y.shape == x.shape
            assert torch.all((y == 0.0) | (y == 1.0))

    def test_backward_masked_straight_through(self):
        """backward：grad_output * mask（mask = (x >= 0.5)）。"""
        x = torch.tensor([0.1, 0.4, 0.5, 0.8], requires_grad=True)
        y = BinarizeSTE.apply(x)
        grad_out = torch.tensor([1.0, 2.0, 3.0, 4.0])
        y.backward(grad_out)
        expected = torch.tensor([0.0, 0.0, 3.0, 4.0])
        assert torch.allclose(x.grad, expected)

    def test_backward_all_above_threshold_passes_through(self):
        """全部 x >= 0.5 時，梯度完全 pass-through。"""
        x = torch.tensor([0.5, 0.6, 0.9], requires_grad=True)
        y = BinarizeSTE.apply(x)
        grad_out = torch.tensor([1.5, 2.5, 3.5])
        y.backward(grad_out)
        assert torch.allclose(x.grad, grad_out)

    def test_backward_all_below_threshold_zero_grad(self):
        """全部 x < 0.5 時，梯度為 0。"""
        x = torch.tensor([0.1, 0.2, 0.4], requires_grad=True)
        y = BinarizeSTE.apply(x)
        grad_out = torch.tensor([1.0, 2.0, 3.0])
        y.backward(grad_out)
        assert torch.allclose(x.grad, torch.zeros_like(x))

    def test_various_shapes_stable(self):
        for shape in [(10,), (4, 5), (2, 3, 4)]:
            x = torch.rand(*shape, requires_grad=True)
            y = BinarizeSTE.apply(x)
            loss = y.sum()
            loss.backward()
            assert x.grad is not None
            assert x.grad.shape == x.shape
            assert torch.isfinite(x.grad).all()


# --- 綜合穩定性 -----------------------------------------------------------


@pytest.mark.parametrize("shape", [(1,), (8,), (3, 4), (2, 3, 4), (1, 1, 8, 8)])
def test_all_functions_handle_various_shapes(shape):
    """確保三個函式在常見 tensor shape 下都能正常 forward + backward。"""
    # sign_f：input 限制在 (-1, 1) 內以保留 STE 梯度
    x = (torch.rand(*shape) * 2 - 1) * 0.8
    x.requires_grad_(True)
    y = sign_f.apply(x)
    y.sum().backward()
    assert x.grad is not None and x.grad.shape == shape

    # GumbelSigmoid
    logits = torch.randn(*shape, requires_grad=True)
    tau = torch.tensor(1.0)
    y = GumbelSigmoid.apply(logits, tau)
    y.sum().backward()
    assert logits.grad is not None and logits.grad.shape == shape

    # BinarizeSTE
    x = torch.rand(*shape, requires_grad=True)
    y = BinarizeSTE.apply(x)
    y.sum().backward()
    assert x.grad is not None and x.grad.shape == shape
