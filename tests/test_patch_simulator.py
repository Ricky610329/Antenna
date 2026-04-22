"""
patch_simulator 模組 smoke test。

本模組透過 HFSS COM 介面驅動 Ansys HFSS，實際模擬需要 Windows + HFSS
授權，CI 環境無法執行。此處僅驗證：

1. 模組 import 不報錯（即使在沒有 HFSS 的環境）
2. 公開 API 類別 / 函式存在
3. ``_common`` 純 Python helper 在不呼叫 HFSS 的前提下行為正確
"""

from __future__ import annotations

import pytest
import torch


def test_patch_simulator_imports():
    """確認核心類別可正確匯入。"""
    from antenna.patch.patch_simulator import PatchSimulator, com_error
    from antenna.patch.patch_simulator.dual_port import DualPortSimulator
    from antenna.patch.patch_simulator.single_port import SinglePortSimulator

    assert PatchSimulator is not None
    assert SinglePortSimulator is not None
    assert DualPortSimulator is not None
    assert com_error is not None


def test_patch_package_reexports():
    """確認 ``antenna.patch`` 套件層級的向後相容 re-export。"""
    from antenna.patch import DualPortSimulator, SinglePortSimulator, com_error

    assert DualPortSimulator is not None
    assert SinglePortSimulator is not None
    assert com_error is not None


def test_simulators_hub_reexports():
    """確認 ``antenna.simulators`` hub 可匯出 Patch 模擬器。"""
    from antenna.simulators import DualPortSimulator, PatchSimulator, SinglePortSimulator

    assert PatchSimulator is not None
    assert issubclass(SinglePortSimulator, PatchSimulator)
    assert issubclass(DualPortSimulator, PatchSimulator)


def test_subclasses_implement_call():
    """確認兩個具體類別皆實作 ``__call__``（不再是抽象方法）。"""
    from antenna.patch.patch_simulator.dual_port import DualPortSimulator
    from antenna.patch.patch_simulator.single_port import SinglePortSimulator

    assert "__call__" in SinglePortSimulator.__dict__
    assert "__call__" in DualPortSimulator.__dict__


class _FakeEditor:
    """記錄 HFSS COM 呼叫以便驗證 ``_common`` helper 行為。"""

    def __init__(self):
        self.create_box_calls: list[tuple] = []
        self.unite_calls: list[str] = []
        self.assign_material_calls: list[tuple] = []
        self.import_calls: list = []
        self.change_property_calls: list = []

    def CreateBox(self, params, attrs):
        self.create_box_calls.append((params, attrs))

    def Unite(self, selections, params):
        # selections = ["NAME:Selections", "Selections:=", "Patch_1,Patch_2,..."]
        self.unite_calls.append(selections[2])

    def AssignMaterial(self, selections, attrs):
        self.assign_material_calls.append((selections, attrs))

    def Import(self, params):
        self.import_calls.append(params)

    def ChangeProperty(self, params):
        self.change_property_calls.append(params)


class _FakeDesign:
    def __init__(self):
        self.change_property_calls: list = []

    def ChangeProperty(self, params):
        self.change_property_calls.append(params)


def test_common_create_patch_pixels_counts_ones():
    """``create_patch_pixels`` 應為每個值為 1 的像素呼叫一次 CreateBox。"""
    from antenna.patch.patch_simulator import _common

    editor = _FakeEditor()
    pattern = torch.zeros(3, 3, dtype=torch.int64)
    pattern[0, 0] = 1
    pattern[1, 2] = 1
    pattern[2, 1] = 1

    _common.create_patch_pixels(editor, pattern)

    assert len(editor.create_box_calls) == 3


def test_common_create_patch_pixels_empty_matrix():
    """全零矩陣不應產生任何 CreateBox 呼叫。"""
    from antenna.patch.patch_simulator import _common

    editor = _FakeEditor()
    _common.create_patch_pixels(editor, torch.zeros(4, 4, dtype=torch.int64))

    assert editor.create_box_calls == []


def test_common_unite_row_patches_skips_singletons():
    """單一 patch 或全零的欄位不應觸發 Unite。"""
    from antenna.patch.patch_simulator import _common

    editor = _FakeEditor()
    # 兩欄皆僅有單一 patch -> 無 Unite
    pattern = torch.zeros(3, 2, dtype=torch.int64)
    pattern[1, 0] = 1
    pattern[2, 1] = 1

    _common.unite_row_patches(editor, pattern)

    assert editor.unite_calls == []


def test_common_unite_row_patches_joins_multiple():
    """同欄超過一個 patch 時應產生對應的 Unite 呼叫。"""
    from antenna.patch.patch_simulator import _common

    editor = _FakeEditor()
    # 第 0 欄三個 1（x=0,1,2），因此會產生 Unite("Patch_1,Patch_2")
    pattern = torch.zeros(3, 2, dtype=torch.int64)
    pattern[0, 0] = 1
    pattern[1, 0] = 1
    pattern[2, 0] = 1

    _common.unite_row_patches(editor, pattern)

    assert len(editor.unite_calls) == 1
    assert "Patch_1" in editor.unite_calls[0]
    assert "Patch_2" in editor.unite_calls[0]


def test_common_assign_pixel_variables_known_size():
    """已知 pixel_count（20/25/50）應觸發 ChangeProperty。"""
    from antenna.patch.patch_simulator import _common

    design = _FakeDesign()
    _common.assign_pixel_variables(design, 25)
    assert len(design.change_property_calls) == 1

    design = _FakeDesign()
    _common.assign_pixel_variables(design, 20)
    assert len(design.change_property_calls) == 1

    design = _FakeDesign()
    _common.assign_pixel_variables(design, 50)
    assert len(design.change_property_calls) == 1


def test_common_assign_pixel_variables_unknown_size():
    """未知 pixel_count 不應觸發任何 ChangeProperty。"""
    from antenna.patch.patch_simulator import _common

    design = _FakeDesign()
    _common.assign_pixel_variables(design, 99)

    assert design.change_property_calls == []


def test_common_assign_substrate_material():
    """基板材料指派應呼叫 AssignMaterial 與 ChangeProperty（Solve Inside）。"""
    from antenna.patch.patch_simulator import _common

    editor = _FakeEditor()
    _common.assign_substrate_material(editor)

    assert len(editor.assign_material_calls) == 1
    assert len(editor.change_property_calls) == 1


def test_common_assign_conductor_material_passes_selections():
    """導體材料指派應將 selections 字串原封不動寫入呼叫 payload。"""
    from antenna.patch.patch_simulator import _common

    editor = _FakeEditor()
    _common.assign_conductor_material(editor, "feed_line,GND")

    assert len(editor.assign_material_calls) == 1
    selections_payload, _ = editor.assign_material_calls[0]
    assert "feed_line,GND" in selections_payload


def test_common_import_substrate_uses_path():
    """匯入基板應把 sab_path 放進 SourceFile 參數。"""
    from antenna.patch.patch_simulator import _common

    editor = _FakeEditor()
    _common.import_substrate(editor, "C:/fake/path/dual_port.sab")

    assert len(editor.import_calls) == 1
    assert "C:/fake/path/dual_port.sab" in editor.import_calls[0]


@pytest.mark.parametrize(
    "module_name,cls_name",
    [
        ("antenna.patch.patch_simulator.single_port", "SinglePortSimulator"),
        ("antenna.patch.patch_simulator.dual_port", "DualPortSimulator"),
    ],
)
def test_default_sab_path_exists(module_name, cls_name):
    """預設 .sab 路徑應存在於套件內。"""
    import importlib
    import inspect

    cls = getattr(importlib.import_module(module_name), cls_name)
    sig = inspect.signature(cls.__init__)
    default_sab = sig.parameters["HFSS_sab_path"].default
    assert default_sab.exists(), f"{cls_name} 預設 .sab 檔不存在: {default_sab}"
