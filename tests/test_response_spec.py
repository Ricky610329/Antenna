"""
響應規格 (自包含 TargetResponse) 與 AntennaResponse.use() 的測試。

驗證這次解耦的核心性質：
  - spec 是自包含實例，不安裝也能推維度 (建模不依賴全域)。
  - 建構 spec 不污染全域；use() 是唯一寫入點且為原子切換。
  - dual 的順序雙軌制保真：concat (GEN 輸入) = 加入順序、criterion 對齊 = labels 宣告順序。
"""
import os

import pytest

from antenna import AntennaResponse, TargetResponse
from antenna.training import load_config, setup_responses

FIX = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture
def _restore_ambient():
    """測試結束還原 session 的 spec (conftest 安裝的單埠規格)。"""
    old = AntennaResponse.target
    yield
    AntennaResponse.use(old)


def test_spec_self_contained():
    """不安裝也能算尺寸/x 軸 —— 建模只需要實例。"""
    spec = TargetResponse(labels=("A", "B"), x=(0, 1, 5))
    assert spec.size() == (2, 5)
    assert spec.size(flatten=True) == 10
    assert len(spec.x()) == 5


def test_building_spec_does_not_touch_ambient():
    """建構 + 加目標曲線，全程不碰 AntennaResponse 類別狀態。"""
    before = AntennaResponse.target
    spec = TargetResponse(labels=("X",), x="n257")
    spec(0, -10, (5, 0, 7, 0, 5), label="X", add=True)
    assert AntennaResponse.target is before


def test_use_atomic_switch(_restore_ambient):
    """use() 原子切換：兩組 spec 可在同一 process 來回切。"""
    dual = TargetResponse(labels=("S11", "S21", "S22"), x="n257")
    AntennaResponse.use(dual)
    assert AntennaResponse.size() == (3, 17)
    assert AntennaResponse.target is dual


def test_use_rejects_incomplete_spec():
    """缺 labels/x 的 spec 不能安裝 (避免裝進殘缺狀態)。"""
    with pytest.raises(ValueError):
        AntennaResponse.use(TargetResponse())


def test_dual_order_two_tracks(_restore_ambient):
    """dual quirk 保真：concat 排列=加入順序 (S11→S22→S21)，criterion 對齊=labels 順序。"""
    spec = setup_responses(load_config(os.path.join(FIX, "dual_test.yaml")))
    assert list(spec.responses.keys()) == ["S11", "S22", "S21"]   # GEN 輸入 concat 排列
    assert list(spec.labels) == ["S11", "S21", "S22"]             # criterion/SM 對齊順序
