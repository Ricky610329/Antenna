"""單埠微帶貼片天線 HFSS 模擬器。"""

from . import *
from . import _common


class SinglePortSimulator(PatchSimulator):
    def __init__(
        self, record_path, HFSS_sab_path=Path(__file__).parent.joinpath("sab", "single_port.sab"), pixel_count: int = 25
    ):
        super().__init__(record_path, HFSS_sab_path, pixel_count)

    def __call__(self, pixel_matrix: Tensor):
        super().__call__(pixel_matrix)
        pixel_row = self.pixel_count
        pixel_column = self.pixel_count

        pixel_matrix = pixel_matrix.reshape(pixel_row, pixel_column).cpu()

        oDesign = self.oDesign
        oEditor = oDesign.SetActiveEditor("3D Modeler")

        # 設定 pixel_H / pixel_W / CooperH 等 HFSS 區域變數
        _common.assign_pixel_variables(oDesign, pixel_row)

        # 匯入基板幾何與設定材料
        _common.import_substrate(oEditor, self.HFSS_sab_path)
        _common.assign_substrate_material(oEditor)
        _common.assign_conductor_material(oEditor, "feed_line,GND")

        # 依 pattern 建立並合併 patch 方塊
        _common.create_patch_pixels(oEditor, pixel_matrix)
        _common.unite_row_patches(oEditor, pixel_matrix)

        # 設定邊界條件 — 單埠 LumpedPort
        oModule = oDesign.GetModule("BoundarySetup")
        oModule.AssignLumpedPort(
            [
                "NAME:1",
                "Objects:=",
                ["Rectangle1"],
                "DoDeembed:=",
                True,
                "RenormalizeAllTerminals:=",
                True,
                [
                    "NAME:Modes",
                    [
                        "NAME:Mode1",
                        "ModeNum:=",
                        1,
                        "UseIntLine:=",
                        True,
                        [
                            "NAME:IntLine",
                            "Start:=",
                            ["27.5mm", "2.5mm", "9.99200722162641e-17mm"],
                            "End:=",
                            ["27.5mm", "2.5mm", "0.508mm"],
                        ],
                        "AlignmentGroup:=",
                        0,
                        "CharImp:=",
                        "Zpi",
                        "RenormImp:=",
                        "50ohm",
                    ],
                ],
                "ShowReporterFilter:=",
                False,
                "ReporterFilter:=",
                [True],
                "Impedance:=",
                "50ohm",
            ]
        )

        # 輻射邊界、分析設定、頻率掃描與遠場取樣
        _common.create_open_region(oDesign)
        _common.insert_analysis_setup(oDesign)
        _common.configure_3d_rad_field(oDesign)

        # 開始模擬
        oDesign.AnalyzeAll()

        # 建立 S11 與 Realized Gain 報表
        oModule = oDesign.GetModule("ReportSetup")
        oModule.CreateReport(
            "S Parameter Plot 1",
            "Modal Solution Data",
            "Rectangular Plot",
            "Setup1 : Sweep",
            ["Domain:=", "Sweep"],
            ["Freq:=", ["All"]],
            ["X Component:=", "Freq", "Y Component:=", ["dB(S(1,1))"]],
        )
        oModule.CreateReport(
            "Realized Gain Plot 1",
            "Far Fields",
            "Rectangular Plot",
            "Setup1 : Sweep",
            ["Context:=", "3D"],
            ["Freq:=", ["All"], "Phi:=", ["0deg"], "Theta:=", ["0deg"]],
            ["X Component:=", "Freq", "Y Component:=", ["dB(RealizedGainTotal)"]],
        )

        # 匯出 CSV
        oModule.ExportToFile(
            "S Parameter Plot 1", self.path_result.joinpath(f"NN_patch_Sparameter_{self.num}_S11.csv"), False
        )
        oModule.ExportToFile(
            "Realized Gain Plot 1", self.path_result.joinpath(f"NN_patch_Gain_{self.num}_S11.csv"), False
        )

        # 讀回 CSV，取出後 17 個取樣點供後續 loss 計算
        Sparameter_dataframe = read_csv(self.path_result.joinpath(f"NN_patch_Sparameter_{self.num}_S11.csv"))
        Gain_dataframe = read_csv(self.path_result.joinpath(f"NN_patch_Gain_{self.num}_S11.csv"))

        S11 = Sparameter_dataframe.iloc[0:17, 1]
        Gain = Gain_dataframe.iloc[0:17, 3]

        return {
            "S11": tensor(S11.to_list()),
            "Gain": tensor(Gain.to_list()),
        }
