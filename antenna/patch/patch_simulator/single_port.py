###* 單埠 (Single Port) 天線模擬器 ###
# 本檔負責「反向設計閉迴路」中的 SIM 角色：
#   GEN(目標響應 → 25x25 二元 pattern) → SM(pattern → 預測響應) → SIM(本檔, 用 HFSS 做真實電磁模擬)
# SinglePortSimulator.__call__ 透過 Ansys HFSS 的 COM API 一步步把 25x25 的二元像素圖
# 建成 3D 微帶貼片天線 (microstrip patch antenna)，跑完整全波模擬後回傳該天線的
# 反射係數 S11 (回波損耗) 與正向 (boresight, theta=phi=0) Realized Gain 隨頻率變化曲線，
# 作為訓練 SM/GEN 的「真值 (ground truth)」。本模擬器只有「單一」訊號埠 (對比 dual_port.py 的雙埠)。
import numpy as np
from loguru import logger
from pandas import read_csv
from torch import Tensor, tensor   # 注意：torch.tensor (與 antenna.utils 的自訂 tensor 不同)
from ...utils import Path
from . import PatchSimulator

def get_penalty(expected_len:int=17):
    #? 當 __call__ 模擬流程在外層被例外攔截 (例如 HFSS COM 崩潰、幾何建模失敗、求解不收斂) 時，
    #? 改呼叫本函式回傳一組「懲罰響應」，讓訓練流程不致中斷，並把這個壞 pattern 導向遠離最佳解。
    #? S11 全 0 dB：代表能量完全被反射 (天線完全不匹配，最差情況)。
    #? Gain 全 -40 dB：代表幾乎沒有輻射 (極差增益)，給優化器一個強烈的負面訊號。
    #? expected_len 預設 17，須與 __call__ 內 np.linspace(24,32,17) 的頻率點數一致，回傳張量長度才能對齊。
    logger.warning("模擬發生未知錯誤，回傳懲罰值")
    return {'S11': tensor([0.0] * expected_len), 'Gain': tensor([-40.0] * expected_len)}

class SinglePortSimulator(PatchSimulator):
    """單埠微帶貼片天線的 HFSS 模擬器。

    繼承自 PatchSimulator (定義於本套件 __init__.py)，沿用其 open/start/end/save 等
    HFSS COM 生命週期管理；本類別只負責「單一回合」的建模、求解與結果讀取 (即 __call__)。
    回傳的響應字典含 'S11' (反射係數, dB) 與 'Gain' (正向 Realized Gain, dB)。
    """
    def __init__(self, record_path, HFSS_sab_path = Path(__file__).parent.joinpath('sab', 'single_port.sab'), pixel_count:int = 25):
        #? HFSS_sab_path 預設指向本套件 sab/single_port.sab：一塊已預先繪製好的「底板」幾何，
        #? 內含基板 (Sub)、地平面 (GND)、單一饋線 (feed_line) 與激勵面 (Rectangle1)。
        #? 像素貼片只需畫在這塊底板上方即可，省去每次重建固定結構的時間。
        #? pixel_count=25 → 預設 25x25 像素網格 (共 625 像素)。
        super().__init__(record_path, HFSS_sab_path, pixel_count)

    def __call__(self, pixel_matrix:Tensor, only_create_project:bool = False):
        #? only_create_project=True 時只建模 (匯入底板、畫貼片、設邊界與掃頻) 但不啟動求解，
        #? 可用於除錯/檢視幾何，或先批次產生專案再另行求解。
        super().__call__(pixel_matrix)  # 父類別會檢查 num 是否已設定 (start 過) 且 pattern 為二元 (僅含 0/1)
        pixel_row = self.pixel_count    # 像素列數 (Y 方向像素數)
        pixel_column = self.pixel_count # 像素行數 (X 方向像素數)；單埠為正方形網格故與 row 相同
        one_num = 0                     # 統計 pattern 中值為 1 (有金屬貼片) 的像素數量，供除錯/統計用

        # 把一維/任意形狀的 pattern 重塑成 (row, column) 二維矩陣，並移回 CPU
        # (後續用 numpy/索引運算，且 COM 不接受 GPU 張量)。
        pixel_matrix = pixel_matrix.reshape(pixel_row, pixel_column).cpu()

        oDesign = self.oDesign                          # 目前作用中的 HFSS 設計 (由父類別 start() 建立)
        oEditor = oDesign.SetActiveEditor("3D Modeler") # 取得 3D 建模器，後續所有幾何操作 (畫盒、聯集、匯入) 都透過它

        ###* 依「總像素數」設定可變幾何參數 (HFSS Local Variables) ###
        # 這三個 if 分支對應三種網格解析度，目的：不論網格切多細，整片貼片區域的「物理尺寸」維持一致
        #   (約 5mm x 5mm)，因此單一像素邊長 = 5mm / 每邊像素數。
        # HFSS 內以「具名變數」表示尺寸，後續畫盒時用變數名 (字串表達式) 而非寫死數值，方便整體縮放。
        #   CooperH  : 金屬貼片 (銅箔) 厚度，固定 0.035mm (約 1oz 銅箔常見厚度)。
        #   pixel_H  : 單一像素在 X 方向的邊長 (Height)。
        #   pixel_W  : 單一像素在 Y 方向的邊長 (Width)。
        if pixel_row*pixel_column == 400:
            # 20x20 = 400 像素：每邊像素數 20，故 pixel_H/W = 0.25mm (20 x 0.25 = 5mm)。
            # 設置初始變數值
            oDesign.ChangeProperty(
                [
                    "NAME:AllTabs",
                    [
                        "NAME:LocalVariableTab",
                        [
                            "NAME:PropServers",
                            "LocalVariables"
                        ],
                        [
                            "NAME:NewProps",
                            [
                                "NAME:CooperH",
                                "PropType:=", "VariableProp",
                                "UserDef:=", True,
                                "Value:=", "0.035mm"  # 銅箔厚度 (與網格無關，固定值)
                            ],
                            [
                                "NAME:pixel_H",
                                "PropType:=", "VariableProp",
                                "UserDef:=", True,
                                "Value:=", "0.25mm"   # 20x20 網格的單一像素 X 邊長
                            ],
                            [
                                "NAME:pixel_W",
                                "PropType:=", "VariableProp",
                                "UserDef:=", True,
                                "Value:=", "0.25mm"   # 20x20 網格的單一像素 Y 邊長
                            ]
                        ]
                    ]
                ])

        if pixel_row*pixel_column == 2500:
            # 50x50 = 2500 像素 (最細網格)：每邊 50 格，故 pixel_H/W = 0.1mm (50 x 0.1 = 5mm)。
            # 設置初始變數值
            oDesign.ChangeProperty(
                [
                    "NAME:AllTabs",
                    [
                        "NAME:LocalVariableTab",
                        [
                            "NAME:PropServers",
                            "LocalVariables"
                        ],
                        [
                            "NAME:NewProps",
                            [
                                "NAME:CooperH",
                                "PropType:=", "VariableProp",
                                "UserDef:=", True,
                                "Value:=", "0.035mm"  # 銅箔厚度 (固定值)
                            ],
                            [
                                "NAME:pixel_H",
                                "PropType:=", "VariableProp",
                                "UserDef:=", True,
                                "Value:=", "0.1mm"    # 50x50 網格的單一像素 X 邊長
                            ],
                            [
                                "NAME:pixel_W",
                                "PropType:=", "VariableProp",
                                "UserDef:=", True,
                                "Value:=", "0.1mm"    # 50x50 網格的單一像素 Y 邊長
                            ]
                        ]
                    ]
                ])

        if pixel_row*pixel_column == 625:
            # 25x25 = 625 像素 (本專案預設)：每邊 25 格，故 pixel_H/W = 0.2mm (25 x 0.2 = 5mm)。
            # 設置初始變數值
            oDesign.ChangeProperty(
                [
                    "NAME:AllTabs",
                    [
                        "NAME:LocalVariableTab",
                        [
                            "NAME:PropServers",
                            "LocalVariables"
                        ],
                        [
                            "NAME:NewProps",
                            [
                                "NAME:CooperH",
                                "PropType:=", "VariableProp",
                                "UserDef:=", True,
                                "Value:=", "0.035mm"  # 銅箔厚度 (固定值)
                            ],
                            [
                                "NAME:pixel_H",
                                "PropType:=", "VariableProp",
                                "UserDef:=", True,
                                "Value:=", "0.2mm"    # 25x25 網格的單一像素 X 邊長
                            ],
                            [
                                "NAME:pixel_W",
                                "PropType:=", "VariableProp",
                                "UserDef:=", True,
                                "Value:=", "0.2mm"    # 25x25 網格的單一像素 Y 邊長
                            ]
                        ]
                    ]
                ])

        ###* 匯入預製底板幾何 (基板 / 地平面 / 饋線 / 激勵面) ###
        # 從 self.HFSS_sab_path 指定的 .sab (ACIS 幾何格式) 檔匯入固定結構，做為貼片的承載平台。
        # FileType="UnRecognized" + Heal/Reduce 關閉：忠實匯入原始幾何，不做自動修補或簡化，
        # 以確保每次建模的底板完全一致 (可重現性)；ImportMaterialNames=True 一併帶入物件名稱與材質標籤。
        # 匯入底板
        oEditor.Import(
            [
                "NAME:NativeBodyParameters",
                "HealOption:=", 0,
                "Options:=", "-1",
                "FileType:=", "UnRecognized",
                "MaxStitchTol:=", -1,
                "ImportFreeSurfaces:=", False,
                "GroupByAssembly:=", False,
                "CreateGroup:=", True,
                "STLFileUnit:=", "Auto",
                "MergeFacesAngle:=", 0.02,
                "HealSTL:=", False,
                "ReduceSTL:=", False,
                "ReduceMaxError:=", 0,
                "ReducePercentage:=", 100,
                "PointCoincidenceTol:=", 1E-06,
                "CreateLightweightPart:=", False,
                "ImportMaterialNames:=", True,
                "SeparateDisjointLumps:=", False,
                "SourceFile:=", self.HFSS_sab_path
            ])

        #* 指定基板 "Sub" 的材質為 Rogers RO4003 (tm)。
        #  RO4003 是毫米波/微波天線常用的低損耗微波基板 (εr≈3.55, tanδ≈0.0027)，
        #  決定了介質波長與貼片諧振尺寸。SolveInside=True：基板為介電質，電磁場會穿透其內部，
        #  必須在其體積內求解 (對比下方的銅導體可只解表面)。
        oEditor.AssignMaterial(
            [
                "NAME:Selections",
                "AllowRegionDependentPartSelectionForPMLCreation:=", True,
                "AllowRegionSelectionForPMLCreation:=", True,
                "Selections:=", "Sub"
            ],
            [
                "NAME:Attributes",
                "MaterialValue:=", "\"Rogers RO4003 (tm)\"",
                "SolveInside:=", True,
                "ShellElement:=", False,
                "ShellElementThickness:=", "nan ",
                "IsMaterialEditable:=", True,
                "UseMaterialAppearance:=", False,
                "IsLightweight:=", False
            ])

        #* 指定饋線 feed_line 與地平面 GND 的材質為 copper (銅)。
        #  這兩者是良導體，SolveInside=False：場幾乎不進入金屬內部，故不必在其體積內求解，
        #  HFSS 改以表面阻抗/邊界近似處理，可大幅減少網格量與求解時間。
        oEditor.AssignMaterial(
            [
                "NAME:Selections",
                "AllowRegionDependentPartSelectionForPMLCreation:=", True,
                "AllowRegionSelectionForPMLCreation:=", True,
                "Selections:=", "feed_line,GND"
            ],
            [
                "NAME:Attributes",
                "MaterialValue:=", "\"copper\"",
                "SolveInside:=", False,
                "ShellElement:=", False,
                "ShellElementThickness:=", "nan ",
                "ReferenceTemperature:=", "nan ",
                "IsMaterialEditable:=", True,
                "UseMaterialAppearance:=", False,
                "IsLightweight:=", False
            ])

        #* 顯式把基板 Sub 的 "Solve Inside" 再設為 True。
        #  保險步驟：確保匯入流程或材質指定後此旗標確實開啟，避免介質內部漏解導致結果失真。
        oEditor.ChangeProperty(
            [
                "NAME:AllTabs",
                [
                    "NAME:Geometry3DAttributeTab",
                    [
                        "NAME:PropServers",
                        "Sub"
                    ],
                    [
                        "NAME:ChangedProps",
                        [
                            "NAME:Solve Inside",
                            "Value:=", True
                        ]
                    ]
                ]
            ])

        ###* 把 pattern 中為 1 的像素逐一畫成銅製 Box (微帶貼片的金屬部分) ###
        # 雙層迴圈掃過整個像素矩陣；每個值為 1 的像素都建一個獨立的長方體 (Box) 放在基板上表面，
        # 之後再用 Unite 把這些散落的小盒子聯集成單一連通的貼片導體。
        # 將Patch Pexil 畫上
        # Create PatchBlock
        patch_names = [] # 建立一個 List 來收集所有生成的 Patch 名稱
        patch_count = 1 # 迴圈生成 Box
        for y in range(0, pixel_row, 1):
            for x in range(0, pixel_column, 1):
                # if pixel_matrix[x][y] > 0:
                #     one_num = one_num + 1
                if pixel_matrix[x][y] == 1:  # 僅在像素=1 (該格有金屬) 時建立 Box
                    current_name = f"Patch_{patch_count}"  # 每個盒子唯一命名 Patch_1, Patch_2, ... 便於後續聯集選取
                    one_num = one_num + 1
                    actual_patch_name = oEditor.CreateBox(  # 回傳 HFSS 實際採用的物件名稱 (可能因重名自動加後綴)
                        [
                            "NAME:BoxParameters",
                            # 起點座標：以像素索引乘上像素邊長變數，拼成字串表達式 (如 "0mm+pixel_H+pixel_H")。
                            # str("+pixel_H" * x) 利用 Python 字串重複，x 格就重複 x 次 "+pixel_H"。
                            "XPosition:=", "0mm" + str("+pixel_H" * x),
                            "YPosition:=", "0mm" + str("+pixel_W" * y),
                            "ZPosition:=", "0.508mm",        # Z 起點 = 基板上表面高度 (基板厚 0.508mm)，貼片貼在介質頂面
                            "XSize:=", "pixel_H + 0.01mm", #! 避免產生奇異點
                            # 盒子尺寸故意比一格像素大 0.01mm：讓相鄰像素的盒子「重疊」而非僅邊對邊相接。
                            # 若僅共邊 (零厚度接觸)，Unite/網格化時會在接縫產生數值奇異點 (singularity) 而報錯，
                            # 微量重疊可保證聯集成真正連通的實體。
                            "YSize:=", "pixel_W + 0.01mm", #!
                            "ZSize:=", "CooperH"            # 盒子高度 = 銅箔厚度
                        ],
                        [
                            "NAME:Attributes",
                            "Name:=", current_name, #! 獨立命名
                            "Flags:=", "",
                            "Color:=", "(255 0 0)",  # 紅色，方便在 HFSS 介面中辨識貼片
                            "Transparency:=", 0,
                            "PartCoordinateSystem:=", "Global",
                            "UDMId:=", "",
                            "MaterialValue:=", "\"copper\"",  # 貼片為銅導體
                            "SurfaceMaterialValue:=", "\"\"",
                            "SolveInside:=", False, #! 導體內部求解 (Original: True)
                            # 設 False：銅為良導體，場不進入內部，只解表面 → 大幅省網格與時間。
                            # 原始版本曾設 True (在介質內求解)，此處刻意改為 False 以加速。
                            "IsMaterialEditable:=", True,
                            "UseMaterialAppearance:=", False,
                            "IsLightweight:=", False
                        ])
                    patch_names.append(actual_patch_name)  # 收集實際名稱，供分批聯集使用
                    patch_count += 1

        ###* 分批聯集 (Batch Unite)：把上百個小盒子合併成單一連通貼片，再接上饋線 ###
        #! 為何要「分批」而非一次全選聯集：HFSS 的 Unite 是 COM 同步呼叫，一次傳入過多 (數百個) 物件名稱
        #! 字串會讓 COM 介面與幾何核心過載，容易逾時或崩潰。改成每批 20 個分階段聯集，可避免 COM 過載。
        # Batch Global Unite (修正後的安全邏輯)
        if len(patch_names) >= 1:
            chunk_size = 20 # 可以稍微調高以增加運算速度
            united_chunk_names = []

            # 階段 1：將所有的 Patch 進行分批互相聯集 (不捲入 feed_line)
            #   每 chunk_size 個盒子為一批，批內聯集成一塊；Unite 結果沿用該批第一個物件名稱。
            for i in range(0, len(patch_names), chunk_size):
                chunk_patches = patch_names[i:i+chunk_size]
                if len(chunk_patches) > 1:  # 只有 1 個無法 (也不需) 聯集，直接略過 Unite 呼叫
                    selections_str = ",".join(chunk_patches)  # COM 以逗號分隔的名稱字串選取多物件
                    oEditor.Unite(
                        ["NAME:Selections", "Selections:=", selections_str],
                        ["NAME:UniteParameters", "KeepOriginals:=", False]  # 不保留原件，合併後僅留一個實體
                    )
                # Unite 預設會保留清單中的第一個名稱，將其收集起來
                united_chunk_names.append(chunk_patches[0])  # 記下每批合併後存活的代表名稱，供階段 2 使用

            # 階段 2：將分批聯集產生的幾個大區塊，再次全部聯集成單一 Patch 實體
            #   各批的代表名稱數量已大幅減少 (每 20 個變 1 個)，此時一次聯集通常安全。
            if len(united_chunk_names) > 1:
                selections_str = ",".join(united_chunk_names)
                oEditor.Unite(
                    ["NAME:Selections", "Selections:=", selections_str],
                    ["NAME:UniteParameters", "KeepOriginals:=", False]
                )

            # 階段 3：最終步驟，將完整融合的 Patch 實體與 feed_line 進行聯集
            #   讓貼片與饋線在電氣上連成一體 (同一導體)，訊號才能由埠經饋線進入貼片輻射。
            final_patch_body = united_chunk_names[0]  # 階段 2 聯集後存活的代表 = 整片貼片本體
            oEditor.Unite(
                ["NAME:Selections", "Selections:=", f"feed_line,{final_patch_body}"],
                ["NAME:UniteParameters", "KeepOriginals:=", False]
            )

        ###* 邊界條件：設定唯一的訊號激勵埠 (Lumped Port) ###
        # 集總埠 (Lumped Port) 是天線在模擬中與外部電路的「饋入點」，HFSS 由此注入功率並量測 S 參數。
        # 指定在預製幾何中的激勵面 Rectangle1 上 (位於饋線末端，跨接饋線與地)。
        # 設定邊界條件
        oModule = oDesign.GetModule("BoundarySetup")
        oModule.AssignLumpedPort(
            [
                "NAME:1",                       # 埠編號 1 (單埠天線只有這一個埠 → S 參數即 S(1,1))
                "Objects:=", ["Rectangle1"],    # 埠所在的激勵面 (底板預製的矩形片)
                "DoDeembed:=", True,            # 去嵌入：把參考面平移到埠面，扣除饋線段相位，得到貼片本體的純淨 S11
                "RenormalizeAllTerminals:=", True,
                [
                    "NAME:Modes",
                    [
                        "NAME:Mode1",
                        "ModeNum:=", 1,
                        "UseIntLine:=", True,   # 使用積分線定義模態電壓的方向與極性 (決定埠阻抗與激勵方向)
                        [
                            "NAME:IntLine",
                            # 積分線由地平面 (z≈0) 垂直指向基板上表面 (z=0.508mm)，跨越介質厚度，
                            # 對應微帶線「導體→地」的電場方向。x=27.5mm 為單埠饋入位置 (對比雙埠的 12.5mm)。
                            # 起點 z 用 9.99e-17mm (≈0 但非剛好 0)：避免端點落在物件邊界上造成拾取歧義。
                            "Start:=", ["27.5mm", "2.5mm",
                                        "9.99200722162641e-17mm"],
                            "End:=", ["27.5mm", "2.5mm", "0.508mm"]
                        ],
                        "AlignmentGroup:=", 0,
                        "CharImp:=", "Zpi",     # 特徵阻抗以功率-電流 (Zpi) 定義
                        "RenormImp:=", "50ohm"  # 將 S 參數歸一化到 50ohm 系統阻抗 (標準射頻參考)
                    ]
                ],
                "ShowReporterFilter:=", False,
                "ReporterFilter:=", [True],
                "Impedance:=", "50ohm"          # 埠參考阻抗 50ohm
            ])

        ###* 輻射邊界：建立開放區域 (Open Region) ###
        # 天線會把能量輻射到自由空間，模擬域邊界必須「吸收」外行波而非反射回來，否則結果失真。
        # CreateOpenRegion 以 28GHz (設計中心頻) 為基準，在模型外圍自動加一層輻射 (Radiation) 邊界，
        # 模擬無限大開放空間。ApplyInfiniteGP=False：不假設無限大地平面 (本天線地平面有限)。
        oModule = oDesign.GetModule("ModelSetup")
        oModule.CreateOpenRegion(
            [
                "NAME:Settings",
                "OpFreq:=", "28GHz",            # 輻射邊界依此頻率的波長決定離開模型的距離 (約 1/4 波長外)
                "Boundary:=", "Radiation",
                "ApplyInfiniteGP:=", False
            ])

        ###* 求解設定 (Analysis Setup)：自適應網格 + 收斂準則 ###
        # HFSS 採「自適應網格細化」：在中心頻 28GHz 反覆加密網格，直到 S 參數變化收斂。
        # 模擬設定
        oModule = oDesign.GetModule("AnalysisSetup")
        oModule.InsertSetup("HfssDriven",
            [
                "NAME:Setup1",
                "SolveType:=", "Single",
                "Frequency:=", "28GHz",         # 自適應細化所用的求解頻率 (設計中心頻)
                "MaxDeltaS:=", 0.02,            # 收斂門檻：相鄰兩次細化的 S 參數最大變化 < 0.02 即視為收斂
                "UseMatrixConv:=", False,
                "MaximumPasses:=", 6,           # 最多細化 6 次 (上限，防止無止盡細化)
                "MinimumPasses:=", 5,           # 至少細化 5 次 (確保網格足夠)
                "MinimumConvergedPasses:=", 5,  # 至少連續 5 次都收斂才算真收斂 (避免假性收斂)
                "PercentRefinement:=", 30,      # 每次細化新增約 30% 網格元素
                "IsEnabled:=", True,
                [
                    "NAME:MeshLink",
                    "ImportMesh:=", False
                ],
                "BasisOrder:=", 1,
                "DoLambdaRefine:=", True,
                "DoMaterialLambda:=", True,
                "SetLambdaTarget:=", False,
                "Target:=", 0.3333,
                "UseMaxTetIncrease:=", False,
                "PortAccuracy:=", 2,
                "UseABCOnPort:=", False,
                "SetPortMinMaxTri:=", False,
                "UseDomains:=", False,
                "UseIterativeSolver:=", False,
                "SaveRadFieldsOnly:=", False,
                "SaveAnyFields:=", True,
                "IESolverType:=", "Auto",
                "LambdaTargetForIESolver:=", 0.15,
                "UseDefaultLambdaTgtForIESolver:=", True,
                "IE Solver Accuracy:=", "Balanced"
            ])
        #* 頻率掃描 (Frequency Sweep)：在 24~32GHz 量測響應隨頻率的變化
        #  以 28GHz 收斂出的網格為基礎，掃出整個頻帶的 S 參數與遠場，得到 S11/Gain 曲線。
        oModule.InsertFrequencySweep("Setup1",
            [
                "NAME:Sweep",
                "IsEnabled:=", True,
                "RangeType:=", "LinearStep",
                "RangeStart:=", "24GHz",         # 掃頻起點
                "RangeEnd:=", "32GHz",           # 掃頻終點 (涵蓋 28GHz 中心頻的 5G n257/n258 毫米波段附近)
                "RangeStep:=", "0.5GHz",         # 步距 0.5GHz → 24~32GHz 共 17 點 (與 expected_len=17 對應)
                "Type:=", "Interpolating",       #! 掃頻演算法選擇 {EX: Fast, Interpolating(插值掃描)}
                # Interpolating：自適應挑少數頻點求解再內插出整條曲線，平滑且快；
                # 對比 dual_port.py 用 "Fast" (掃描式)。
                "SaveFields:=", True,
                "SaveRadFields:=", False,
                "GenerateFieldsForAllFreqs:=", False,
                "ExtrapToDC:=", False
            ])

        #* 遠場觀測球面 (Infinite Sphere)：定義計算 Realized Gain 的角度範圍
        #  在以天線為中心的無限遠球面上，依 theta/phi 取樣計算輻射場/增益。
        #  本檔僅需正向 (theta=phi=0) 增益，但仍設定完整 3D 球面 (步距 2deg) 供報表查詢任意方向。
        oModule = oDesign.GetModule("RadField")
        oModule.EditInfiniteSphereSetup("3D",
            [
                "NAME:3D",
                "UseCustomRadiationSurface:=", False,
                "ThetaStart:=", "-180deg",       # 仰角 theta 掃描 -180~180 度
                "ThetaStop:=", "180deg",
                "ThetaStep:=", "2deg",
                "PhiStart:=", "-180deg",         # 方位角 phi 掃描 -180~180 度
                "PhiStop:=", "180deg",
                "PhiStep:=", "2deg",
                "UseLocalCS:=", False
            ])

        if only_create_project:
            # only_create_project=True：到此為止只完成建模與設定，不求解、不讀結果，直接返回 (None)。
            # self.oProject.Save()
            return


        ###* 啟動全波電磁求解 ###
        # AnalyzeAll 觸發 HFSS 執行上面定義的 Setup1 (自適應網格 + 24~32GHz 掃頻)，為整段流程中最耗時的步驟。
        # 開始模擬
        oDesign.AnalyzeAll()


        ###* 建立結果報表 (Report)：S11 與 正向 Realized Gain 隨頻率變化 ###
        # 畫出結果
        oModule = oDesign.GetModule("ReportSetup")

        #* S11 報表：反射係數 dB(S(1,1)) vs Freq。
        #  S11 衡量天線匹配/回波損耗，越負 (越深的凹陷) 代表越多能量送入天線而非反射回去；
        #  其凹陷頻率即天線諧振頻率，是反向設計最關心的指標之一。
        # Create S11
        oModule.CreateReport("S Parameter Plot 1", "Modal Solution Data", "Rectangular Plot", "Setup1 : Sweep",
            [
                "Domain:=", "Sweep"
            ],
            [
                "Freq:=", ["All"]                       # 取全部掃頻點
            ],
            [
                "X Component:=", "Freq",
                "Y Component:=", ["dB(S(1,1))"]         # Y 軸為 S11 的 dB 值
            ])
        #* Realized Gain 報表：正向 (theta=0, phi=0, 即天線正上方) 的總實現增益 vs Freq。
        #  Realized Gain 已含反射/失配損耗，最貼近實際可用增益；固定 theta=phi=0 取「正向增益」做為單一純量指標。
        oModule.CreateReport("Realized Gain Plot 1", "Far Fields", "Rectangular Plot", "Setup1 : Sweep",
            [
                "Context:=", "3D"                       # 使用前面定義的 3D 無限球面遠場
            ],
            [
                "Freq:=", ["All"],
                "Phi:=", ["0deg"],                      # 方位角固定 0 度
                "Theta:=", ["0deg"]                     # 仰角固定 0 度 → boresight 正向
            ],
            [
                "X Component:=", "Freq",
                "Y Component:=", ["dB(RealizedGainTotal)"]
            ])
        # oModule.CreateReport("Realized Gain Plot 2", "Far Fields", "Rectangular Plot", "Setup1 : LastAdaptive",
        #     [
        #         "Context:=", "3D"
        #     ],
        #     [
        #         "Theta:=", ["All"],
        #         "Phi:=", ["90deg"],
        #         "Freq:=", ["28GHz"]
        #     ],
        #     [
        #         "X Component:=", "Theta",
        #         "Y Component:=", ["dB(RealizedGainTotal)"]
        #     ])

        #* 把兩張報表匯出成 CSV 檔，存到 self.path_result。
        #  檔名含 self.num (本回合 pattern 編號) 以區分不同設計；後續再讀回轉成張量。
        # Export csv
        oModule.ExportToFile(
            "S Parameter Plot 1",
            self.path_result.joinpath(f"NN_patch_Sparameter_{self.num}_S11.csv"),  # S11 曲線檔
            False
        )
        oModule.ExportToFile(
            "Realized Gain Plot 1",
            self.path_result.joinpath(f"NN_patch_Gain_{self.num}_S11.csv"),        # 正向 Gain 曲線檔 (檔名沿用 _S11 後綴)
            False
        )
        # oModule.ExportToFile("Realized Gain Plot 2", args.record_path+'/csv/NN_patch_Beamwidth_"+ str(Design_index) +"_Realized Gain Plot 2.csv", False)
        
        
        ###* 讀回 CSV 並整理成固定長度 (17 點) 的響應張量 ###
        # Read csv
        Sparameter_dataframe = read_csv(
            self.path_result.joinpath(f"NN_patch_Sparameter_{self.num}_S11.csv")
        )
        Gain_dataframe = read_csv(
            self.path_result.joinpath(f"NN_patch_Gain_{self.num}_S11.csv")
        )

        # 預期頻率點，這裡定義後，後續皆以 len(freqs_expected) 為準
        #? 24~32GHz 等分 17 點 (對應掃頻步距 0.5GHz)；訓練端 (SM/loss) 假設響應固定長度 17，
        #? 故此處強制把結果對齊到這 17 個頻點。
        freqs_expected = np.linspace(24, 32, 17)
        expected_len = len(freqs_expected)

        #* S11
        freqs_s11 = Sparameter_dataframe.iloc[:, 0].values  # 第 0 欄：頻率 (GHz)
        S11_vals = Sparameter_dataframe.iloc[:, 1].values   # 第 1 欄：dB(S(1,1))
        if len(S11_vals) != expected_len:
            #? 點數不符的成因：Interpolating 掃頻實際輸出的取樣點數未必剛好 17 (演算法自選頻點)。
            #? 用線性插值 np.interp 依原始 (頻率, 值) 把曲線重新取樣到固定的 17 個 freqs_expected 點，
            #? 確保回傳張量長度恆為 17，下游才能對齊比較。
            # logger.warning(f"HFSS S11 模擬點數異常 (Pattern {self.num})！預期 {expected_len} 點，實際取得 {len(S11_vals)} 點，將自動進行插值補齊。")
            S11_vals:np.ndarray = np.interp(freqs_expected, freqs_s11, S11_vals)

        #* Gain
        # 注意 Gain CSV 的欄位排列不同：因含 Phi/Theta context 欄，頻率落在第 2 欄、增益值落在第 3 欄。
        freqs_gain = Gain_dataframe.iloc[:, 2].values       # 第 2 欄：頻率 (GHz)
        Gain_vals = Gain_dataframe.iloc[:, 3].values        # 第 3 欄：dB(RealizedGainTotal)
        if len(Gain_vals) != expected_len:
            # 同理對 Gain 做插值補點，對齊到 17 個頻點。
            # logger.warning(f"HFSS Gain 模擬點數異常 (Pattern {self.num})！預期 {expected_len} 點，實際取得 {len(Gain_vals)} 點，將自動進行插值補齊。")

            Gain_vals:np.ndarray = np.interp(freqs_expected, freqs_gain, Gain_vals)

        #* 組成回傳字典：兩條長度 17 的曲線轉成 torch.Tensor，供訓練端計算 loss / 比對。
        _result = {
            'S11': tensor(S11_vals.tolist()),
            'Gain': tensor(Gain_vals.tolist()),
        }

        # 把 S11 與 Gain 串成單一一維陣列 (備用，目前未被回傳)。
        full_output = []
        full_parameter = np.append(np.append(full_output, S11_vals), Gain_vals)

        return _result