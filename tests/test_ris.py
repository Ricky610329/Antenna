"""antenna.ris 模組的 smoke test。"""

import pytest

torch = pytest.importorskip("torch")


def test_import_public_api():
    """確認公開 API 可正常匯入。"""
    from antenna.ris import RISSimulator, custom_loss  # noqa: F401


def test_simulator_forward_shape():
    """RISSimulator 應輸出 {'response': 1D Tensor} 並保留梯度。"""
    from antenna.ris import RISSimulator

    element_num = 4  # 小尺寸避免長時間計算
    sim = RISSimulator(element_num)

    pattern = torch.zeros(element_num * element_num, requires_grad=True)
    out = sim(pattern)

    assert isinstance(out, dict)
    assert "response" in out
    response = out["response"]
    assert response.dim() == 1
    # 可反向傳播，確保 pattern 的梯度鏈未中斷
    response.sum().backward()
    assert pattern.grad is not None


def test_custom_loss_returns_scalar_with_grad():
    """custom_loss 應對任意 prediction/target 回傳帶梯度之純量。"""
    from antenna.ris import custom_loss

    target = torch.tensor([-20.0, 0.0, -20.0, 0.0])
    prediction = torch.tensor([-10.0, -5.0, -15.0, -1.0], requires_grad=True)
    loss = custom_loss(prediction, target)

    assert loss.dim() == 0
    loss.backward()
    assert prediction.grad is not None


def test_custom_loss_no_violation_still_has_grad():
    """目標全滿足時仍需保留梯度路徑（透過 dummy MSE）。"""
    from antenna.ris import custom_loss

    target = torch.tensor([-20.0, 0.0])
    # low 點預測更低、high 點預測更高，兩個 mask 都為空
    prediction = torch.tensor([-30.0, 10.0], requires_grad=True)
    loss = custom_loss(prediction, target)

    loss.backward()
    assert prediction.grad is not None
