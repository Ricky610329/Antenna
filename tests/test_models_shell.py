"""
Models 外殼的行為特徵測試 (characterization)。

TDD 流程：先對「現行」實作鎖定行為合約 (全綠)，再進行素化重寫 (去泛型)，
重寫後本檔必須維持全綠 = 行為不變的證明。
涵蓋：label 換檔、save/load 原子寫入與 title 把關、pre_load_model (含 NaN 防護)、
step、requires_grad/train 模式、checkpoint 結構。
"""
import pytest
import torch
from torch import nn

from antenna.models import Models
from antenna.utils import config


def _make(tmp_path, name="models_{label}", cls=nn.Linear):
    config.device = "cpu"
    model = cls(2, 2)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=10)
    return Models(name=name, rootdir=str(tmp_path), model=model,
                  optimizer=opt, scheduler=sch, criterion=nn.MSELoss())


def test_label_template_mode(tmp_path):
    """name 含 {label} → 須先 change() 才有檔名；change 後檔名帶入 label。"""
    m = _make(tmp_path)
    with pytest.raises(AssertionError):
        _ = m.model_file
    m.change(3)
    assert m.model_file.name == "models_3.pth"
    m.change(4)
    assert m.model_file.name == "models_4.pth"


def test_fixed_name_mode(tmp_path):
    """name 無 {label} → 檔名固定，change 不改檔名。"""
    m = _make(tmp_path, name="sm")
    assert m.model_file.name == "sm.pth"
    m.change(99)
    assert m.model_file.name == "sm.pth"


def test_save_load_roundtrip(tmp_path):
    """save 後改壞權重 → load 還原 (含 optimizer/scheduler 狀態)。"""
    m = _make(tmp_path); m.change(1)
    saved = {k: v.clone() for k, v in m.model.state_dict().items()}
    m.save()
    with torch.no_grad():
        for p in m.model.parameters():
            p.add_(99.0)
    m.load()
    for k, v in m.model.state_dict().items():
        assert torch.equal(v, saved[k])


def test_load_rejects_wrong_title(tmp_path):
    """title (架構簽名) 不符 → 拒載，避免把錯的權重灌進不相容架構。"""
    m1 = _make(tmp_path, name="x"); m1.save()
    m2 = _make(tmp_path, name="x", cls=lambda a, b: nn.Bilinear(a, a, b))
    with pytest.raises(RuntimeError):
        m2.load()


def test_change_save_then_load_other_label(tmp_path):
    """change(新label, save=True, load=True) 的真實語義 (rollback 的核心動作)：
    先把「現狀」存到「目前 (舊) label」的檔，再換名、載入新 label 的檔。"""
    m = _make(tmp_path)
    m.change(1)
    m.save()                                   # models_1.pth = w1
    w1 = {k: v.clone() for k, v in m.model.state_dict().items()}
    m.change(2)
    with torch.no_grad():
        for p in m.model.parameters(): p.add_(1.0)
    m.save()                                   # models_2.pth = w1+1
    w2 = {k: v.clone() for k, v in m.model.state_dict().items()}

    m.change(1, save=True, load=True)          # 現狀再存回 models_2.pth → 載回 models_1.pth
    for k, v in m.model.state_dict().items():
        assert torch.equal(v, w1[k])           # 回到 label 1 的權重
    assert (tmp_path / "models_2.pth").exists()


def test_pre_load_model_skips_title_check(tmp_path):
    """pre_load_model 只灌權重+優化器，不比對 title (預訓練檔來自不同包裝)。"""
    src = _make(tmp_path, name="src"); src.save()
    dst = _make(tmp_path, name="dst")
    dst.pre_load_model(tmp_path / "src.pth")
    for k, v in dst.model.state_dict().items():
        assert torch.equal(v, src.model.state_dict()[k])


def test_pre_load_model_rejects_nan(tmp_path):
    """預訓練檔含 NaN → 立刻擋下 (避免汙染閉迴路梯度)。"""
    src = _make(tmp_path, name="bad")
    with torch.no_grad():
        next(src.model.parameters()).fill_(float("nan"))
    src.save()
    dst = _make(tmp_path, name="dst")
    with pytest.raises(RuntimeError, match="NaN"):
        dst.pre_load_model(tmp_path / "bad.pth")


def test_step_updates_params(tmp_path):
    m = _make(tmp_path, name="s")
    before = next(m.model.parameters()).clone()
    out = m(torch.ones(2))
    out.sum().backward()
    m.step()
    assert not torch.equal(next(m.model.parameters()), before)


def test_requires_grad_and_train_mode(tmp_path):
    m = _make(tmp_path, name="g")
    assert m.requires_grad(False, train=False) is False
    assert all(not p.requires_grad for p in m.model.parameters())
    assert m.model.training is False
    assert m.requires_grad(True, train=True) is True
    assert m.model.training is True


def test_checkpoint_structure_and_atomic_save(tmp_path):
    m = _make(tmp_path, name="c")
    ckpt = m.checkpoint()
    assert set(ckpt) == {"title", "model_state_dict", "optimizer_state_dict",
                         "scheduler_state_dict", "device", "record_state_dict"}
    m.save()
    assert not (tmp_path / "c.pth.tmp").exists()   # 原子寫入不留殘檔
    assert (tmp_path / "c.pth").exists()
