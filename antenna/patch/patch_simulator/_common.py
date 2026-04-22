"""
patch_simulator 共用 HFSS 腳本片段。

本模組集中 ``SinglePortSimulator`` 與 ``DualPortSimulator`` 之間
重複出現的 HFSS COM 呼叫，避免兩邊各維護一份同樣的 VBS 字典。
這些函式僅封裝 HFSS 腳本呼叫，無任何條件分支以外的邏輯；改動時請
同步檢查對應的 ``.sab`` 模型。
"""

from __future__ import annotations

from typing import Any

import numpy as np
from torch import Tensor

# HFSS 腳本中用到的硬編碼幾何 / 材料常數
# (搬到這邊是為了避免字串散落；實際值由 .sab 模型以及 patch 幾何決定)
_COPPER_H = "0.035mm"
_SUBSTRATE_Z = "0.508mm"  # 基板厚度，對應 .sab 模型
_SUBSTRATE_MATERIAL = '"Rogers RO4003 (tm)"'
_CONDUCTOR_MATERIAL = '"copper"'

# pixel_count -> (pixel_H, pixel_W) 尺寸對應表
# 目前 .sab 模型僅支援 20x20 / 25x25 / 50x50 三種
_PIXEL_SIZE_TABLE: dict[int, tuple[str, str]] = {
    20: ("0.25mm", "0.25mm"),
    25: ("0.2mm", "0.2mm"),
    50: ("0.1mm", "0.1mm"),
}


def assign_pixel_variables(oDesign: Any, pixel_count: int) -> None:
    """根據 pixel_count 設定 CooperH / pixel_H / pixel_W 三個 HFSS 區域變數。

    若尺寸未在 ``_PIXEL_SIZE_TABLE`` 中則維持 HFSS 原本的預設變數。
    """
    if pixel_count not in _PIXEL_SIZE_TABLE:
        return
    pixel_h, pixel_w = _PIXEL_SIZE_TABLE[pixel_count]
    oDesign.ChangeProperty(
        [
            "NAME:AllTabs",
            [
                "NAME:LocalVariableTab",
                ["NAME:PropServers", "LocalVariables"],
                [
                    "NAME:NewProps",
                    ["NAME:CooperH", "PropType:=", "VariableProp", "UserDef:=", True, "Value:=", _COPPER_H],
                    ["NAME:pixel_H", "PropType:=", "VariableProp", "UserDef:=", True, "Value:=", pixel_h],
                    ["NAME:pixel_W", "PropType:=", "VariableProp", "UserDef:=", True, "Value:=", pixel_w],
                ],
            ],
        ]
    )


def import_substrate(oEditor: Any, sab_path: str) -> None:
    """從 ``.sab`` 檔匯入基板幾何。"""
    oEditor.Import(
        [
            "NAME:NativeBodyParameters",
            "HealOption:=",
            0,
            "Options:=",
            "-1",
            "FileType:=",
            "UnRecognized",
            "MaxStitchTol:=",
            -1,
            "ImportFreeSurfaces:=",
            False,
            "GroupByAssembly:=",
            False,
            "CreateGroup:=",
            True,
            "STLFileUnit:=",
            "Auto",
            "MergeFacesAngle:=",
            0.02,
            "HealSTL:=",
            False,
            "ReduceSTL:=",
            False,
            "ReduceMaxError:=",
            0,
            "ReducePercentage:=",
            100,
            "PointCoincidenceTol:=",
            1e-06,
            "CreateLightweightPart:=",
            False,
            "ImportMaterialNames:=",
            True,
            "SeparateDisjointLumps:=",
            False,
            "SourceFile:=",
            sab_path,
        ]
    )


def assign_substrate_material(oEditor: Any) -> None:
    """將 ``Sub`` 物件設為 Rogers RO4003，並打開 Solve Inside。"""
    oEditor.AssignMaterial(
        [
            "NAME:Selections",
            "AllowRegionDependentPartSelectionForPMLCreation:=",
            True,
            "AllowRegionSelectionForPMLCreation:=",
            True,
            "Selections:=",
            "Sub",
        ],
        [
            "NAME:Attributes",
            "MaterialValue:=",
            _SUBSTRATE_MATERIAL,
            "SolveInside:=",
            True,
            "ShellElement:=",
            False,
            "ShellElementThickness:=",
            "nan ",
            "IsMaterialEditable:=",
            True,
            "UseMaterialAppearance:=",
            False,
            "IsLightweight:=",
            False,
        ],
    )
    oEditor.ChangeProperty(
        [
            "NAME:AllTabs",
            [
                "NAME:Geometry3DAttributeTab",
                ["NAME:PropServers", "Sub"],
                ["NAME:ChangedProps", ["NAME:Solve Inside", "Value:=", True]],
            ],
        ]
    )


def assign_conductor_material(oEditor: Any, selections: str) -> None:
    """將指定物件名稱（逗號分隔）設為 copper，不求解內部。"""
    oEditor.AssignMaterial(
        [
            "NAME:Selections",
            "AllowRegionDependentPartSelectionForPMLCreation:=",
            True,
            "AllowRegionSelectionForPMLCreation:=",
            True,
            "Selections:=",
            selections,
        ],
        [
            "NAME:Attributes",
            "MaterialValue:=",
            _CONDUCTOR_MATERIAL,
            "SolveInside:=",
            False,
            "ShellElement:=",
            False,
            "ShellElementThickness:=",
            "nan ",
            "ReferenceTemperature:=",
            "nan ",
            "IsMaterialEditable:=",
            True,
            "UseMaterialAppearance:=",
            False,
            "IsLightweight:=",
            False,
        ],
    )


def create_patch_pixels(oEditor: Any, pixel_matrix: Tensor) -> None:
    """依 ``pixel_matrix`` 中值為 1 的位置建立 copper patch 方塊。"""
    pixel_row, pixel_column = pixel_matrix.shape
    for y in range(pixel_row):
        for x in range(pixel_column):
            if pixel_matrix[x][y] != 1:
                continue
            # HFSS 支援 "0mm+pixel_H+pixel_H+..." 這種變數運算寫法
            # （保持與原腳本相同以避免破壞相容性）
            oEditor.CreateBox(
                [
                    "NAME:BoxParameters",
                    "XPosition:=",
                    "0mm" + "+pixel_H" * x,
                    "YPosition:=",
                    "0mm" + "+pixel_W" * y,
                    "ZPosition:=",
                    _SUBSTRATE_Z,
                    "XSize:=",
                    "pixel_H",
                    "YSize:=",
                    "pixel_W",
                    "ZSize:=",
                    "CooperH",
                ],
                [
                    "NAME:Attributes",
                    "Name:=",
                    "Patch",
                    "Flags:=",
                    "",
                    "Color:=",
                    "(255 0 0)",
                    "Transparency:=",
                    0,
                    "PartCoordinateSystem:=",
                    "Global",
                    "UDMId:=",
                    "",
                    "MaterialValue:=",
                    _CONDUCTOR_MATERIAL,
                    "SurfaceMaterialValue:=",
                    '""',
                    "SolveInside:=",
                    True,
                    "IsMaterialEditable:=",
                    True,
                    "UseMaterialAppearance:=",
                    False,
                    "IsLightweight:=",
                    False,
                ],
            )


def unite_row_patches(oEditor: Any, pixel_matrix: Tensor) -> None:
    """將同一欄相連的 Patch 方塊合併，以減少後續模擬工件數。"""
    pixel_column = pixel_matrix.shape[1]
    ones_buf = 0
    for i in range(pixel_column):
        column = pixel_matrix[:, i]
        ones_indices = np.where(column == 1)[0]
        # 只有 0 或 1 個 patch 時不需要 unite
        if ones_indices.shape[0] > 1:
            patch_names = [f"Patch_{u + ones_buf}" for u in range(ones_indices.shape[0]) if u + ones_buf != 0]
            patch_unite = ",".join(patch_names)

            # 經過首方塊 (Patch_0) 被過濾後，若僅剩單一元素則無需 unite
            if patch_unite and patch_unite != "Patch_1":
                oEditor.Unite(
                    ["NAME:Selections", "Selections:=", patch_unite],
                    ["NAME:UniteParameters", "KeepOriginals:=", False],
                )
        ones_buf += ones_indices.shape[0]


def create_open_region(oDesign: Any, freq: str = "28GHz") -> None:
    """建立指定頻率的輻射邊界開放區域。"""
    oModule = oDesign.GetModule("ModelSetup")
    oModule.CreateOpenRegion(["NAME:Settings", "OpFreq:=", freq, "Boundary:=", "Radiation", "ApplyInfiniteGP:=", False])


def insert_analysis_setup(oDesign: Any, freq: str = "28GHz") -> None:
    """插入 HfssDriven 分析設定 (Setup1) 與 24~32 GHz Frequency Sweep。"""
    oModule = oDesign.GetModule("AnalysisSetup")
    oModule.InsertSetup(
        "HfssDriven",
        [
            "NAME:Setup1",
            "SolveType:=",
            "Single",
            "Frequency:=",
            freq,
            "MaxDeltaS:=",
            0.02,
            "UseMatrixConv:=",
            False,
            "MaximumPasses:=",
            6,
            "MinimumPasses:=",
            5,
            "MinimumConvergedPasses:=",
            5,
            "PercentRefinement:=",
            30,
            "IsEnabled:=",
            True,
            ["NAME:MeshLink", "ImportMesh:=", False],
            "BasisOrder:=",
            1,
            "DoLambdaRefine:=",
            True,
            "DoMaterialLambda:=",
            True,
            "SetLambdaTarget:=",
            False,
            "Target:=",
            0.3333,
            "UseMaxTetIncrease:=",
            False,
            "PortAccuracy:=",
            2,
            "UseABCOnPort:=",
            False,
            "SetPortMinMaxTri:=",
            False,
            "UseDomains:=",
            False,
            "UseIterativeSolver:=",
            False,
            "SaveRadFieldsOnly:=",
            False,
            "SaveAnyFields:=",
            True,
            "IESolverType:=",
            "Auto",
            "LambdaTargetForIESolver:=",
            0.15,
            "UseDefaultLambdaTgtForIESolver:=",
            True,
            "IE Solver Accuracy:=",
            "Balanced",
        ],
    )
    oModule.InsertFrequencySweep(
        "Setup1",
        [
            "NAME:Sweep",
            "IsEnabled:=",
            True,
            "RangeType:=",
            "LinearStep",
            "RangeStart:=",
            "24GHz",
            "RangeEnd:=",
            "32GHz",
            "RangeStep:=",
            "0.5GHz",
            "Type:=",
            "Fast",
            "SaveFields:=",
            True,
            "SaveRadFields:=",
            False,
            "GenerateFieldsForAllFreqs:=",
            False,
            "ExtrapToDC:=",
            False,
        ],
    )


def configure_3d_rad_field(oDesign: Any) -> None:
    """設定 3D 遠場取樣解析度（theta/phi 2 度）。"""
    oModule = oDesign.GetModule("RadField")
    oModule.EditInfiniteSphereSetup(
        "3D",
        [
            "NAME:3D",
            "UseCustomRadiationSurface:=",
            False,
            "ThetaStart:=",
            "-180deg",
            "ThetaStop:=",
            "180deg",
            "ThetaStep:=",
            "2deg",
            "PhiStart:=",
            "-180deg",
            "PhiStop:=",
            "180deg",
            "PhiStep:=",
            "2deg",
            "UseLocalCS:=",
            False,
        ],
    )
