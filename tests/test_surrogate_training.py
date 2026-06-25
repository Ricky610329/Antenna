"""
tests/test_surrogate_training.py — SM 訓練入口的形狀健全性（無 HFSS、純函式層級）。

回歸保護這幾次踩到的「形狀廣播」類 bug：SM 預測與 label 形狀不一致時，
PyTorch 的 MSELoss 會「靜默廣播」並印警告（`target size ... different to the input size`），
數值可能不正確。這裡用真實 HFSSNet 跑一輪，斷言「不該出現該廣播警告」。

全域 spec（S11+Gain, x=n257 → 響應形狀 (2,17)）由 conftest 的 autouse fixture 安裝。
"""
import warnings

import torch

from antenna.models.surrogates import MLPSurrogate
from antenna.utils.store import SampleStore


def _store(tmp_path, n=3):
    s = SampleStore(tmp_path / "ds", verbose=False)
    for _ in range(n):
        s.add(torch.rand(25, 25).round(), torch.rand(2, 17))   # (pattern, S11/Gain 響應)
    return s


_NOISY = ("different to the input size",      # MSE 形狀廣播
          "To copy construct from a tensor")  # torch.tensor(已是tensor) 複製


def _broadcast_warnings(records):
    """挑出這幾次踩到的雜訊警告（形狀廣播 / 張量複製建構）。"""
    return [w for w in records if any(s in str(w.message) for s in _NOISY)]


def test_train_by_datas_no_broadcast_warning(tmp_path):
    """整批訓練：outputs 對齊 labels 形狀 → 不該出現廣播警告（修 (2,17) vs (1,2,17)）。"""
    sm = MLPSurrogate(tmp_path / "ck", 625, (2, 17), max_epoch=1)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        sm.train_by_datas(_store(tmp_path), epochs=1, verbose=False)
    assert not _broadcast_warnings(rec), \
        f"train_by_datas 觸發形狀廣播警告: {[str(w.message)[:80] for w in _broadcast_warnings(rec)]}"


def test_train_one_data_no_broadcast_warning(tmp_path):
    """單筆線上訓練：兩邊都 reshape 成 (-1, *response_shape) → 不該出現廣播警告（回歸保護）。"""
    sm = MLPSurrogate(tmp_path / "ck2", 625, (2, 17), max_epoch=2)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        sm.train_one_data(torch.rand(625), torch.rand(2, 17), max_epoch=2, verbose=False)
    assert not _broadcast_warnings(rec)


def test_train_one_data_nan_guard(tmp_path):
    """NaN 防護網：target 含 inf → loss 非有限 → 跳過該步、不 raise（不炸掉整個 HFSS run）。"""
    sm = MLPSurrogate(tmp_path / "ck3", 625, (2, 17), max_epoch=5)
    bad = torch.rand(2, 17)
    bad[0, 0] = float("inf")
    sm.train_one_data(torch.rand(625), bad, max_epoch=5, verbose=False)   # 不該丟例外


def test_train_by_datas_nan_guard(tmp_path):
    """NaN 防護網：某筆 response 含 inf → 跳過該 batch、不 raise。"""
    s = SampleStore(tmp_path / "ds_nan", verbose=False)
    s.add(torch.rand(25, 25).round(), torch.rand(2, 17))
    bad = torch.rand(2, 17); bad[1, 3] = float("inf")
    s.add(torch.ones(25, 25), bad)
    sm = MLPSurrogate(tmp_path / "ck4", 625, (2, 17), max_epoch=1)
    sm.train_by_datas(s, epochs=1, verbose=False)                         # 不該丟例外


def test_train_one_data_returns_finite_loss_list(tmp_path):
    """回傳逐步 loss 清單去掉 index0 的 inf 初值 → 全有限、末位=收斂後 loss (training 拿來當 rad_fit)。"""
    import math
    sm = MLPSurrogate(tmp_path / "ck5", 625, (2, 17), max_epoch=3)
    hist = sm.train_one_data(torch.rand(625), torch.rand(2, 17), max_epoch=3, verbose=False)
    assert isinstance(hist, list) and len(hist) >= 1
    assert all(math.isfinite(x) for x in hist)        # 不再含 inf 初值 (防 float(ret) 之類踩雷)
    assert hist[-1] == hist[-1]                        # 末位有限 (非 NaN) → 可當 rad_fit


def test_sm_requires_grad_false_freezes_params_but_keeps_input_grad(tmp_path):
    """審查後優化的不變式：smodel.requires_grad(False) 後，SM 反傳『不在 SM 參數上累積梯度』
    (GEN 不會誤更新 SM)，但梯度仍穿過 SM 流回輸入 (GEN 仍學得動)。eval 模式不被動到。"""
    sm = MLPSurrogate(tmp_path / "ck6", 625, (2, 17), max_epoch=1)
    sm.requires_grad(False)
    assert all(not p.requires_grad for p in sm.model.parameters())
    x = torch.rand(625, requires_grad=True)
    out = sm(x)                                                      # smodel(x) → response 物件
    out.criterion().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()       # 梯度穿過凍住的 SM 回到輸入
    assert all(p.grad is None for p in sm.model.parameters())        # SM 參數零梯度 (沒被 GEN 帶歪)
