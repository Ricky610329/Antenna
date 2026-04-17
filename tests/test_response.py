"""AntennaResponse、MultiResponses、TargetResponse 單元測試。"""

import numpy as np
import pytest
import torch

from antenna.core.response import AntennaResponse, MultiResponses, TargetResponse


def _clear_response_class_state():
    AntennaResponse.target = TargetResponse()
    for attr in ("_x", "labels"):
        if attr in AntennaResponse.__dict__:
            delattr(AntennaResponse, attr)


@pytest.fixture(autouse=True)
def _reset_response_state():
    """每個測試前後重置 class-level 狀態，避免相互影響。"""
    _clear_response_class_state()
    yield
    _clear_response_class_state()


# -------------------- TargetResponse.__call__ --------------------


def test_target_response_call_mask_shape_and_values():
    """以 width=(2,3,4,3,2) 生成的 mask 應為總長 14 的張量，端值=side、中段=center。"""
    tr = TargetResponse()
    width = (2, 3, 4, 3, 2)
    side, center = 0.0, 1.0
    mask = tr(side=side, center=center, width=width, label="response", add=False)

    assert isinstance(mask, torch.Tensor)
    assert mask.shape == (sum(width),)
    # 兩端值
    assert torch.allclose(mask[: width[0]], torch.full((width[0],), side))
    assert torch.allclose(mask[-width[4] :], torch.full((width[4],), side))
    # 中段平坦 center
    start = width[0] + width[1]
    end = start + width[2]
    assert torch.allclose(mask[start:end], torch.full((width[2],), center))


def test_target_response_call_linspace_segments():
    """上升段與下降段應為 side → center 與 center → side 的 linspace。"""
    tr = TargetResponse()
    width = (1, 5, 1, 5, 1)
    side, center = -10.0, 0.0
    mask = tr(side=side, center=center, width=width, add=False).cpu().numpy()

    up = mask[width[0] : width[0] + width[1]]
    down = mask[width[0] + width[1] + width[2] : width[0] + width[1] + width[2] + width[3]]
    np.testing.assert_allclose(up, np.linspace(side, center, width[1]))
    np.testing.assert_allclose(down, np.linspace(center, side, width[3]))


def test_target_response_call_invalid_width_length():
    """width 長度不等於 5 時應 raise ValueError。"""
    tr = TargetResponse()
    with pytest.raises(ValueError, match="Expected 5 width"):
        tr(side=0.0, center=1.0, width=(1, 2, 3))  # type: ignore[arg-type]


def test_target_response_unregistered_getitem_raises():
    """未註冊 label 取用應 raise RuntimeError。"""
    tr = TargetResponse()
    with pytest.raises(RuntimeError, match="is not registered"):
        _ = tr["not_exist"]


# -------------------- AntennaResponse.__init__ & _reshape2vertical --------------------


def test_antenna_response_init_1d_tensor():
    """1D Tensor 輸入應保留 response 一維、vertical 為 (1, N)。"""
    data = torch.arange(8, dtype=torch.float32)
    r = AntennaResponse(data)
    assert r.response.shape == (8,)
    assert r.vertical.shape == (1, 8)
    # _reshape2vertical 會設 requires_grad
    assert r.vertical.requires_grad


def test_antenna_response_init_2d_tensor():
    """2D Tensor 輸入應壓平為 response、保留原形為 vertical。"""
    data = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    r = AntennaResponse(data)
    assert r.response.shape == (12,)
    assert r.vertical.shape == (3, 4)


def test_antenna_response_init_invalid_type_raises():
    """非 Tensor/Dict 輸入應 raise TypeError。"""
    with pytest.raises(TypeError, match="Expected Tensor"):
        AntennaResponse(123)  # type: ignore[arg-type]


def test_antenna_response_dict_becomes_multiresponses():
    """傳入 dict 會觸發 __new__ 走 MultiResponses 分支。"""
    AntennaResponse.registerLabels("a", "b", x=(0, 1, 4))
    d = {"a": torch.zeros(4), "b": torch.ones(4)}
    obj = AntennaResponse(d)
    assert isinstance(obj, MultiResponses)
    assert len(obj) == 2


def test_antenna_response_invert_detaches():
    """~response 應為 detach 且在 cpu 上。"""
    data = torch.ones(4, requires_grad=True)
    r = AntennaResponse(data)
    inv = ~r
    assert inv.device.type == "cpu"
    assert not inv.requires_grad


# -------------------- registerLabels --------------------


def test_register_labels_basic():
    """registerLabels 後 target.labels 與 AntennaResponse.labels 應一致，且 _x 已設。"""
    AntennaResponse.registerLabels("s11", "s21", x=(0, 10, 11))
    assert list(AntennaResponse.target.labels) == ["s11", "s21"]
    assert AntennaResponse.labels == ("s11", "s21")
    assert AntennaResponse._x == (0, 10, 11)


def test_register_labels_preset_ris():
    """x='ris' 應對應 (0, 360, 361)。"""
    AntennaResponse.registerLabels("r", x="ris")
    assert AntennaResponse._x == (0, 360, 361)


def test_register_labels_preset_n257():
    """x='n257' 應對應 (24, 32, 17)。"""
    AntennaResponse.registerLabels("r", x="n257")
    assert AntennaResponse._x == (24, 32, 17)


def test_register_labels_empty_raises_on_size():
    """無 labels 時呼叫 size() 應 raise RuntimeError。"""
    # fixture 已重置 target
    with pytest.raises(RuntimeError, match="No labels registered"):
        AntennaResponse.size()


def test_x_classmethod_uses_linspace():
    """x() 應回傳 np.linspace(*_x)。"""
    AntennaResponse.registerLabels("a", x=(0, 4, 5))
    np.testing.assert_allclose(AntennaResponse.x(), np.linspace(0, 4, 5))


def test_x_classmethod_without_register_raises():
    """未註冊 _x 呼叫 x() 應 raise RuntimeError。"""
    with pytest.raises(RuntimeError, match="No x registered"):
        AntennaResponse.x()


# -------------------- registerTargetResponse --------------------


def test_register_target_response_requires_labels():
    """未註冊 labels 即呼叫 registerTargetResponse 應 raise。"""
    with pytest.raises(RuntimeError, match="No labels registered"):
        AntennaResponse.registerTargetResponse(side=0.0, center=1.0, width=(1, 1, 1, 1, 1))


def test_register_target_response_populates_metadata():
    """註冊後 metadata 應包含 side / center / width / note 與 response。"""
    AntennaResponse.registerLabels("resp", x=(0, 5, 6))
    AntennaResponse.registerTargetResponse(side=0.0, center=2.0, width=(1, 2, 3, 2, 1), label="resp")

    meta = AntennaResponse.target.metadata["resp"]
    assert meta["side"] == 0.0
    assert meta["center"] == 2.0
    assert meta["width"] == (1, 2, 3, 2, 1)
    assert "note" in meta
    # 取用已註冊 label 不應 raise
    assert AntennaResponse.target["resp"] is not None


# -------------------- 邊界：空 labels / 覆寫 --------------------


def test_target_response_overwrite_same_label():
    """同一 label 二次註冊（add=True）會覆寫 metadata。"""
    AntennaResponse.registerLabels("resp", x=(0, 5, 6))
    AntennaResponse.registerTargetResponse(side=0.0, center=1.0, width=(1, 1, 2, 1, 1), label="resp")
    AntennaResponse.registerTargetResponse(side=-1.0, center=5.0, width=(2, 1, 1, 1, 1), label="resp")

    meta = AntennaResponse.target.metadata["resp"]
    assert meta["side"] == -1.0
    assert meta["center"] == 5.0
    assert meta["width"] == (2, 1, 1, 1, 1)


def test_register_labels_empty_sequence():
    """傳入空 labels 後 size() 仍應 raise（labels 視為空）。"""
    AntennaResponse.registerLabels(x=(0, 1, 2))
    assert AntennaResponse.target.labels == []
    with pytest.raises(RuntimeError, match="No labels registered"):
        AntennaResponse.size()


# -------------------- MultiResponses shape --------------------


def test_multiresponses_stack_and_concat_shapes():
    """stack 與 concat 回傳的張量形狀應符合預期。"""
    AntennaResponse.registerLabels("a", "b", x=(0, 3, 4))
    mr = MultiResponses({"a": torch.zeros(4), "b": torch.ones(4)})
    stacked = mr.stack()
    concated = mr.concat()
    assert stacked.shape == (2, 4)
    assert concated.shape == (8,)
