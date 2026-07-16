# -*- coding: utf-8 -*-
"""
tests/test_cnn_surrogate.py — 影子 CNN（R32 影子對決挑戰者）的介面契約。

CNNNet 必須與 HFSSNet 的 forward 語義完全對齊（1-D 單張→(3,17)、2-D 批次→(B,3,17)、
批次 ≡ 逐張），SurrogateModel 外殼（train_by_datas/save/load）直接重用——
sm_reanchor 影子段與 analyze 雙模盲測都依賴這個契約。
"""
import numpy as np
import torch
from torch.utils.data import TensorDataset

from antenna.models.surrogates import CNNNet, CNNSurrogate


def test_cnnnet_forward_shapes():
    """1-D 輸入回 (3,17)（與 HFSSNet 同語義）；2-D (B,625) 回 (B,3,17)。"""
    torch.manual_seed(0)
    net = CNNNet(625, (3, 17))
    x1 = torch.rand(625)
    assert net(x1).shape == (3, 17)
    xb = torch.rand(4, 625)
    assert net(xb).shape == (4, 3, 17)


def test_cnnnet_batch_equals_single():
    """批次 forward ≡ 逐張疊起來（候選互不干擾；同 test_batch_forward 的契約）。"""
    torch.manual_seed(0)
    net = CNNNet(625, (3, 17))
    net.eval()
    xb = torch.rand(3, 625)
    with torch.no_grad():
        out_b = net(xb)
        out_s = torch.stack([net(xb[i]) for i in range(3)])
    assert torch.allclose(out_b, out_s, atol=1e-6)


def test_cnn_surrogate_train_save_load(tmp_path):
    """SurrogateModel 外殼重用：train_by_datas 跑得動、save/load 往返預測一致。"""
    torch.manual_seed(0)
    sm = CNNSurrogate(str(tmp_path), 625, (3, 17))
    X = torch.rand(8, 625)
    Y = torch.rand(8, 3, 17)
    losses = sm.train_by_datas(TensorDataset(X, Y), epochs=2, batch_size=4, verbose=False)
    assert len(losses) >= 1 and np.isfinite(losses[-1])
    f = tmp_path / "shadow_test.pth"
    sm.save_as(f)
    sm2 = CNNSurrogate(str(tmp_path), 625, (3, 17))
    sm2.pre_load_model(f, strict=True)
    sm.model.eval(); sm2.model.eval()
    with torch.no_grad():
        x = torch.rand(625)
        assert torch.allclose(sm.model(x), sm2.model(x), atol=1e-6)
