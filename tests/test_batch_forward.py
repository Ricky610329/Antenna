"""
tests/test_batch_forward.py — batch 安全化先修的回歸保護（無 HFSS、純函式層級）。

後續「同批多候選 Z」(N(z*,σ²I) 抽 K 個一起在 SM 上 optimize) 需要 GEN/SM 的
forward 能吃 (K, N) 批次。本檔鎖住這次先修的「batch 契約」：
  1. BiScaleNorm 改 per-row（每候選各自正規化、不互相耦合）；1-D 與舊全域版同值。
  2. HFSSNet.forward / forward_rad 加 ndim 分支：1-D 保留舊形狀（golden）、2-D 保留批次維。

核心斷言：**批次 forward ≡ 逐張 forward 疊起來**（候選之間互不干擾，正是多候選設計的前提）。
"""
import pytest
import torch

from antenna.models.generators import BiScaleNorm
from antenna.models.surrogates import MLPSurrogate
from antenna.utils.store import SampleStore


def test_biscalenorm_per_row_independent():
    """BiScaleNorm 對 (K, N) 逐列正規化：第 k 列只看自己的 max/min，不被別列尺度影響。"""
    norm = BiScaleNorm()
    x = torch.tensor([[1.0, 2.0, -4.0],
                      [10.0, -5.0, 0.0]])
    out = norm(x)
    assert torch.allclose(out[0], norm(x[0]))   # 逐列 ≡ 各自單獨正規化
    assert torch.allclose(out[1], norm(x[1]))
    # 把第 1 列的尺度放大 1000 倍，第 0 列的結果不該變（per-row 解耦的關鍵）
    x2 = x.clone(); x2[1] *= 1000.0
    assert torch.allclose(norm(x2)[0], out[0])


def test_biscalenorm_1d_matches_global():
    """1-D 輸入：per-row(dim=-1) 等同舊全域 max/min（golden 不動）。"""
    norm = BiScaleNorm()
    x = torch.tensor([3.0, -6.0, 1.5, 0.0, -2.0])
    pos = torch.where(x > 0, x / x.max(), torch.zeros_like(x))           # 舊全域版手算
    neg = torch.where(x < 0, x / x.min().abs(), torch.zeros_like(x))
    assert torch.allclose(norm(x), pos + neg)


def test_hfssnet_forward_batch_equals_stacked(tmp_path):
    """SM forward：(K,625) → (K,2,17)，且等於逐張 forward 疊起來（候選互不干擾）。"""
    sm = MLPSurrogate(tmp_path / "ck", 625, (2, 17))
    sm.model.eval()
    pats = torch.rand(4, 625)
    with torch.no_grad():
        batch_out = sm.model(pats)
        stacked = torch.stack([sm.model(p) for p in pats])   # 每張 1-D → (2,17)
    assert batch_out.shape == (4, 2, 17)
    assert torch.allclose(batch_out, stacked, atol=1e-5, rtol=1e-4)


def test_hfssnet_forward_1d_unchanged(tmp_path):
    """SM forward：1-D 輸入維持 (2,17)（舊形狀，golden 路徑）。"""
    sm = MLPSurrogate(tmp_path / "ck2", 625, (2, 17))
    sm.model.eval()
    with torch.no_grad():
        out = sm.model(torch.rand(625))
    assert out.shape == (2, 17)


def test_forward_rad_batch_equals_stacked(tmp_path):
    """方向圖頭：(K,625) → (K,n_phi,n_theta)，且等於逐張疊起來；1-D 維持 (n_phi,n_theta)。"""
    sm = MLPSurrogate(tmp_path / "ckr", 625, (2, 17), rad_response=(2, 21))
    sm.model.eval()
    pats = torch.rand(3, 625)
    with torch.no_grad():
        batch_out = sm.model.forward_rad(pats)
        stacked = torch.stack([sm.model.forward_rad(p) for p in pats])
        one = sm.model.forward_rad(torch.rand(625))
    assert batch_out.shape == (3, 2, 21)
    assert one.shape == (2, 21)
    assert torch.allclose(batch_out, stacked, atol=1e-5, rtol=1e-4)


def test_train_by_datas_batch_size_gt1_runs(tmp_path):
    """順手修的既存 bug：舊 forward 寫死 reshape((C,L)) → train_by_datas(batch_size>1) 會炸 (B*51≠51)。
    ndim 分支讓 (B,625)→(B,*resp) 後可正常 batch 訓練。"""
    s = SampleStore(tmp_path / "ds", verbose=False)
    for _ in range(3):
        s.add(torch.rand(25, 25).round(), torch.rand(2, 17))
    sm = MLPSurrogate(tmp_path / "ck", 625, (2, 17), max_epoch=1)
    losses = sm.train_by_datas(s, epochs=1, batch_size=2, verbose=False)   # 改前 B=2 會 reshape 報錯
    assert isinstance(losses, list)


def test_forward_3d_raises(tmp_path):
    """3-D 輸入 (未來盲點)：ndim 分支只認 1-D/2-D → 3-D 明確 RuntimeError、不靜默錯算。"""
    sm = MLPSurrogate(tmp_path / "ck3d", 625, (2, 17), rad_response=(2, 21))
    with pytest.raises(RuntimeError):
        sm.model(torch.rand(2, 3, 625))
    with pytest.raises(RuntimeError):
        sm.model.forward_rad(torch.rand(2, 3, 625))


@pytest.mark.xfail(reason="BiScaleNorm 退化列 (max=0 或 min=0) 的 torch.where 除零 → 反向 grad NaN；"
                          "待 Z 多候選上線前加 eps 保護 (golden-neutral)。1-D 現況不觸發 (MLP 浮點精確=0 機率≈0)。",
                   strict=True)
def test_biscalenorm_degenerate_row_grad_finite():
    """退化列 (無正值且含 0 → max=0) 的反向梯度應有限。現況為 NaN (已知缺口，xfail 記錄；加 eps 後翻綠)。"""
    x = torch.tensor([[0.0, -2.0, -1.0]], requires_grad=True)   # max=0
    BiScaleNorm()(x).sum().backward()
    assert torch.isfinite(x.grad).all()
