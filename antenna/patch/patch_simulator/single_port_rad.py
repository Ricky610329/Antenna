###* 單埠 + 輻射方向圖 (radiation pattern) 萃取 模擬器 ###
# 本檔是 SinglePortSimulator 的「不侵入」擴充：除了既有的 S11 / 正向 Gain，
# 額外把「方向圖」(gain vs theta 角度，固定 28GHz，phi=0° 與 90° 兩個主平面切面)
# 匯出成 CSV，供後續驗證 / 分析。配方完全照使用者提供的 HFSS 操作截圖：
#   物理量 dB(GainTotal)、X 軸 Theta、固定 Freq=28GHz、Phi∈{0°,90°}、Setup1:LastAdaptive。
#
# 設計刻意「不搞壞既有環境」：
#   1. 完全沿用父類別的建模 / 求解 / S11&Gain 匯出與讀回 (super().__call__)，
#      父類別程式碼一行未改；本類別只在「求解完成後」多做報表匯出。
#   2. __call__ 的回傳值與父類別「完全相同」(只有 S11/Gain)——方向圖「不」塞進回傳 dict。
#      原因：訓練端 criterion 依 spec.labels 用 zip 對齊 (antenna/response.py)，回傳多塞 key
#      會讓長度與 labels 對不上而「靜默錯位」。故方向圖只走獨立通道，零污染訓練管線。
#   3. 方向圖資料寫到 self.path_result 的 CSV (你之後在正式機跑就能驗證「有這個資料」)，
#      並暫存到 self.last_radiation 供獨立驗證腳本 (script/verify_radiation.py) 讀取。
#   4. 方向圖萃取整段包 try/except：就算報表匯出失敗，也「不」影響已取得的 S11/Gain 回傳，
#      只記 warning 並把 self.last_radiation 標記為失敗。確保不會反過來弄壞既有流程。
import numpy as np
from loguru import logger
from pandas import read_csv
from torch import Tensor, tensor

from .single_port import SinglePortSimulator


def _parse_rad_csv(df):
    """從 HFSS Far Fields 匯出的 CSV 取 (theta_deg, gain_db)，依「欄名」抓、不靠欄位順序。
    HFSS 匯出欄位是 [Freq, Phi, Theta, dB(GainTotal)] —— theta 不在第 0 欄 (第 0 欄是 Freq、
    全 28GHz)！寫死 iloc[:,0] 會把 Freq 當 theta、方向圖整個畫錯 (踩過的雷)。"""
    theta_col = next(c for c in df.columns if "Theta" in c)
    gain_col = next(c for c in df.columns if "GainTotal" in c)
    return (np.asarray(df[theta_col].values, dtype=float),
            np.asarray(df[gain_col].values, dtype=float))


class SinglePortRadSimulator(SinglePortSimulator):
    """單埠 HFSS 模擬器 + 方向圖萃取 (繼承自 SinglePortSimulator)。

    沿用父類別全部建模 / 求解 / S11&Gain 流程；只在求解後額外匯出方向圖：
    dB(GainTotal) vs Theta，固定 Freq=28GHz，phi∈{0°,90°}。
    回傳值與父類別相同 (S11/Gain)；方向圖另存 CSV + self.last_radiation。
    """

    #? 方向圖萃取參數 (對應操作截圖；要改解析度/頻率/切面改這裡即可)
    RAD_FREQ = "28GHz"                       #? 固定觀測頻率 (設計中心頻)
    RAD_PHIS = (0, 90)                       #? 兩個主平面切面 (phi=0° / 90°，即 E-plane / H-plane)
    RAD_SPHERE = "3D"                        #? 沿用父類別 __call__ 已建好的 3D 無限球面 (theta/phi step 2°)
    RAD_SOLUTION = "Setup1 : LastAdaptive"   #? 28GHz 單頻自適應解 (與截圖一致)
    #? 註：若你的設定在 LastAdaptive 取不到遠場，把上行改成 "Setup1 : Sweep" 即可
    #?    (父類別的 Gain 報表就是用 Sweep + 3D 球面取遠場的)。

    def __init__(self, record_path, *args, **kwargs):
        super().__init__(record_path, *args, **kwargs)
        #? 最近一次 __call__ 萃取到的方向圖 (供獨立驗證腳本讀取)；尚未跑過為 None。
        self.last_radiation = None

    def __call__(self, pixel_matrix: Tensor, only_create_project: bool = False):
        #? 先完整跑父類別流程 (建模 → 求解 → 匯出/讀回 S11、Gain)，取得既有回傳 dict。
        result = super().__call__(pixel_matrix, only_create_project=only_create_project)

        if only_create_project:
            #? 只建模、不求解 → 沒有場可取，直接回傳 (行為與父類別一致)。
            return result

        #? 求解已在 super().__call__ 內 AnalyzeAll 完成；oDesign 仍有效 (end() 尚未呼叫)。
        #? 在同一個設計上「多做」方向圖報表，整段包 try/except 確保不影響已取得的 S11/Gain。
        try:
            self.last_radiation = self._export_radiation()
        except Exception as e:
            #! 方向圖萃取失敗不該拖垮既有 S11/Gain；記 warning、標記失敗、照常回傳。
            logger.warning(f"方向圖萃取失敗 (Pattern {self.num})，略過: {e}")
            self.last_radiation = {"error": str(e)}

        return result

    def _export_radiation(self) -> dict:
        """匯出 phi=0°/90° 兩條 dB(GainTotal) vs Theta 曲線 → CSV + dict。

        回傳 {'theta': Tensor, 'phi0': Tensor, 'phi90': Tensor}；
        theta 為角度 (deg)，phiN 為該切面的 gain (dB)。
        """
        oModule = self.oDesign.GetModule("ReportSetup")
        radiation = {"theta": None}

        for phi in self.RAD_PHIS:
            report_name = f"Rad Gain phi{phi} {self.num}"   #? 含 num 避免同設計內重名

            #* Far Fields 報表：X=Theta、Y=dB(GainTotal)，固定 phi 與 28GHz，用 3D 無限球面。
            oModule.CreateReport(
                report_name, "Far Fields", "Rectangular Plot", self.RAD_SOLUTION,
                ["Context:=", self.RAD_SPHERE],
                [
                    "Theta:=", ["All"],            # 掃 theta 全角度 (3D 球面 -180~180, step 2°)
                    "Phi:=", [f"{phi}deg"],        # 固定此切面
                    "Freq:=", [self.RAD_FREQ],     # 固定 28GHz
                ],
                [
                    "X Component:=", "Theta",
                    "Y Component:=", ["dB(GainTotal)"],
                ])

            #* 匯出 CSV (檔名含 num 與 phi)；這就是你之後要驗證「有沒有資料」的檔案。
            csv_path = self.path_result.joinpath(
                f"NN_patch_RadGain_{self.num}_phi{phi}.csv"
            )
            oModule.ExportToFile(report_name, csv_path, False)

            #* 讀回：依欄名抓 Theta / dB(GainTotal) (見 _parse_rad_csv 的踩雷說明)。
            theta_vals, gain_vals = _parse_rad_csv(read_csv(csv_path))

            if radiation["theta"] is None:
                radiation["theta"] = tensor(theta_vals.tolist())
            radiation[f"phi{phi}"] = tensor(gain_vals.tolist())

        return radiation
