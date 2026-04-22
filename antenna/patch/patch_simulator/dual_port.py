"""雙埠微帶貼片天線 HFSS 模擬器。"""

from . import *
from . import _common


class DualPortSimulator(PatchSimulator):
    def __init__(
        self, record_path, HFSS_sab_path=Path(__file__).parent.joinpath("sab", "dual_port.sab"), pixel_count: int = 25
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
        _common.assign_conductor_material(oEditor, "feedline1,feedline2,GND")

        # 依 pattern 建立並合併 patch 方塊
        _common.create_patch_pixels(oEditor, pixel_matrix)
        _common.unite_row_patches(oEditor, pixel_matrix)

        # 設定邊界條件 — 雙埠 LumpedPort
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
                            ["12.5mm", "2.5mm", "9.99200722162641e-17mm"],
                            "End:=",
                            ["12.5mm", "2.5mm", "0.508mm"],
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

        oModule.AssignLumpedPort(
            [
                "NAME:2",
                "Objects:=",
                ["Rectangle2"],
                "DoDeembed:=",
                True,
                "RenormalizeAllTerminals:=",
                True,
                [
                    "NAME:Modes",
                    [
                        "NAME:Mode2",
                        "ModeNum:=",
                        1,  # 這邊不能改 2，因為要和上面的 lumport 組成一組
                        "UseIntLine:=",
                        True,
                        [
                            "NAME:IntLine",
                            "Start:=",
                            ["-7.5mm", "2.5mm", "9.99200722162641e-17mm"],
                            "End:=",
                            ["-7.5mm", "2.5mm", "0.508mm"],
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

        # 建立 S11 / S21 / S22 報表
        oModule = oDesign.GetModule("ReportSetup")
        plot_specs = (
            ("S Parameter Plot 1", "dB(S(1,1))", "S11"),
            ("S Parameter Plot 2", "dB(S(2,1))", "S21"),
            ("S Parameter Plot 3", "dB(S(2,2))", "S22"),
        )
        for plot_name, s_param, _ in plot_specs:
            oModule.CreateReport(
                plot_name,
                "Modal Solution Data",
                "Rectangular Plot",
                "Setup1 : Sweep",
                ["Domain:=", "Sweep"],
                ["Freq:=", ["All"]],
                ["X Component:=", "Freq", "Y Component:=", [s_param]],
            )

        # 匯出 CSV 並讀回
        csv_paths = {
            key: self.path_result.joinpath(f"NN_patch_Sparameter_{self.num}_{key}.csv") for _, _, key in plot_specs
        }
        for plot_name, _, key in plot_specs:
            oModule.ExportToFile(plot_name, csv_paths[key], False)

        return {key: tensor(read_csv(path).iloc[:, 1].to_list()) for key, path in csv_paths.items()}
