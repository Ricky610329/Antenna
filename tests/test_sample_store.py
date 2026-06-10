"""
SampleStore (一筆一檔樣本庫) 的單元測試。

驗證新資料格式的核心性質：add O(1) 落地、hash 去重、跨實例持久化、
崩潰殘留清理、torch Dataset 介面 (DataLoader 可直接吃)。
"""
import torch
from torch.utils.data import DataLoader

from antenna.utils.store import SampleStore, fingerprint


def _xy(seed):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(25, 25, generator=g), torch.rand(2, 17, generator=g)


def test_add_len_getitem_roundtrip(tmp_path):
    s = SampleStore(tmp_path / "ds", verbose=False)
    x, y = _xy(0)
    assert s.add(x, y) is True
    assert len(s) == 1
    rx, ry = s[0]
    assert torch.equal(rx, x) and torch.equal(ry, y)


def test_dedup_by_content(tmp_path):
    """同內容 (即使是不同 tensor 物件) → 同 hash → 不重複落地。"""
    s = SampleStore(tmp_path / "ds", verbose=False)
    x, y = _xy(1)
    assert s.add(x, y) is True
    assert s.add(x.clone(), y.clone()) is False
    assert len(s) == 1
    x2, y2 = _xy(2)
    assert s.add(x2, y2) is True
    assert len(s) == 2


def test_persistence_across_instances(tmp_path):
    """新實例開同一個資料夾 → 看得到既有樣本 (glob 重建索引)。"""
    d = tmp_path / "ds"
    s1 = SampleStore(d, verbose=False)
    s1.add(*_xy(3)); s1.add(*_xy(4))
    s2 = SampleStore(d, verbose=False)
    assert len(s2) == 2
    assert s2.add(*_xy(3)) is False      # 跨實例去重也成立 (檔名即指紋)


def test_grad_detached_on_add(tmp_path):
    """掛在計算圖上的 tensor 也能存 (內部 detach)，載回不帶梯度。"""
    s = SampleStore(tmp_path / "ds", verbose=False)
    x = torch.rand(25, 25, requires_grad=True) * 2
    y = torch.rand(2, 17, requires_grad=True) + 1
    assert s.add(x, y) is True
    rx, ry = SampleStore(s.rootdir, verbose=False)[0]
    assert not rx.requires_grad and not ry.requires_grad


def test_leftover_tmp_cleaned(tmp_path):
    """崩潰殘留的 .tmp 在下次開啟時清掉，且不被算進樣本。"""
    d = tmp_path / "ds"; d.mkdir()
    (d / "deadbeef.tmp").write_bytes(b"partial")
    s = SampleStore(d, verbose=False)
    assert len(s) == 0
    assert not (d / "deadbeef.tmp").exists()


def test_dataloader_compatible(tmp_path):
    """torch Dataset 介面：DataLoader batch 出 (patterns, responses)。"""
    s = SampleStore(tmp_path / "ds", verbose=False)
    for i in range(4):
        s.add(*_xy(10 + i))
    patterns, responses = next(iter(DataLoader(s, batch_size=4, shuffle=False)))
    assert patterns.shape == (4, 25, 25)
    assert responses.shape == (4, 2, 17)


def test_fingerprint_shape_sensitive():
    """同 bytes 不同 shape → 不同指紋 (避免 reshape 撞名)。"""
    a = torch.zeros(4)
    b = torch.zeros(2, 2)
    y = torch.zeros(1)
    assert fingerprint(a, y) != fingerprint(b, y)
