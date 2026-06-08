
# 從套件 __init__.py 匯入共用相依：win32com COM 介面、numpy、pandas.read_csv、
# torch.tensor/Tensor、Path/config、logger，以及抽象基底類別 PatchSimulator。
# 與 single_port.py 共用同一組 import，確保兩種模擬器行為一致。
from . import *


class DualPortSimulator(PatchSimulator):
    """雙埠 Patch 天線 HFSS 模擬器。

    本類別繼承 :class:`PatchSimulator`，封裝「以 HFSS COM API 驅動 Ansys 真實
    全波模擬 (SIM)」的完整流程。輸入為 25x25 二元像素矩陣 (pattern)，輸出為三條
    散射參數 (S-parameter) 響應。

    與單埠版 (:class:`SinglePortSimulator`) 的核心差異：

    * 底板 sab 檔為 ``dual_port.sab``，含「兩條饋線 feedline1 / feedline2」與兩個
      port 對應的耦合矩形 (Rectangle1 / Rectangle2)。
    * 同時指定「兩個 LumpedPort (集總埠)」，故 S 矩陣為 2x2，會輸出：
        - ``S11``：port1 反射係數 (回波損耗，反映 port1 阻抗匹配)。
        - ``S21``：port1→port2 傳輸係數，亦即「兩埠間的互耦 (mutual coupling) /
          隔離度」；單埠版沒有此量，是雙埠最關鍵的新增物理量。
        - ``S22``：port2 反射係數 (port2 的阻抗匹配)。
    * 單埠版額外輸出遠場 Realized Gain；雙埠版此處改為聚焦三條 S 參數，未輸出增益。
    """

    def __init__(self, record_path, HFSS_sab_path = Path(__file__).parent.joinpath('sab', 'dual_port.sab'), pixel_count:int = 25):
        # record_path：本次訓練/實驗的紀錄根目錄 (基底類別會在其下建立 HFSS/、result/、project/)。
        # HFSS_sab_path：雙埠底板幾何檔，預設指向與本檔同層 sab/ 目錄下的 dual_port.sab，
        #               內含基板 Sub、地 GND、兩條饋線 feedline1/feedline2 及兩個 port 用矩形。
        #               (單埠版預設為 single_port.sab，僅一條 feed_line。)
        # pixel_count：每邊像素數，預設 25 → 25x25=625 個像素的可佈線網格。
        super().__init__(record_path, HFSS_sab_path, pixel_count)

    def __call__(self, pixel_matrix:Tensor):
        """對單一 pattern 執行一次完整的雙埠 HFSS 模擬並回傳 S 參數。

        :param pixel_matrix: 形狀可攤平為 ``pixel_count^2`` 的二元張量 (0/1)，
            1 代表該像素要鋪上銅 (Patch)，0 代表留空。
        :return: 含 ``S11`` / ``S21`` / ``S22`` 三條響應的 dict，各值為 torch.Tensor。

        前置條件：呼叫前必須已先 ``start(num)`` 建立專案與設計 (見基底類別)。
        """
        # 先呼叫基底類別的 __call__ 進行共用檢查：
        #   1. 確認已呼叫過 start() (self.num 不為 None)。
        #   2. 確認輸入為純二元 (僅含 0/1)，否則無法對應「鋪銅/留空」。
        super().__call__(pixel_matrix)
        pixel_row = self.pixel_count        # 像素列數 (高度方向格數)
        pixel_column = self.pixel_count     # 像素行數 (寬度方向格數)；本設計為正方形網格故兩者相等
        one_num = 0                         # 統計鋪銅 (值為 1) 的像素總數，供除錯/檢視之用

        # 將輸入攤平/重塑成 row x column 二維矩陣，並搬回 CPU。
        # 後續以 numpy / Python 迴圈逐格存取，需在 CPU 上避免 GPU↔CPU 反覆搬移。
        pixel_matrix = pixel_matrix.reshape(pixel_row, pixel_column).cpu()

        oDesign = self.oDesign                              # 目前作用中的 HFSS 設計物件 (由 start() 建立)
        oEditor = oDesign.SetActiveEditor("3D Modeler")     # 取得 3D 建模器編輯介面，用來建立/操作幾何

        # 依「總像素數」設定每個像素方塊的實體尺寸 (pixel_H / pixel_W) 與銅厚 (CooperH)。
        # 不同網格密度需搭配不同像素邊長，才能讓整片 Patch 的物理面積維持一致 →
        #   25x25=625 → 0.2mm；50x50=2500 → 0.1mm；20x20=400 → 0.25mm。
        # CooperH=0.035mm 為銅箔厚度 (約 1oz 銅)，三種網格皆相同。
        # 這些 LocalVariable 之後在 CreateBox 中以變數名 (而非寫死數值) 引用，方便整體縮放。
        if pixel_row*pixel_column == 400:
            # 20x20 網格：較粗，單像素邊長 0.25mm。
            # 設置初始變數值
            oDesign.ChangeProperty(
                [
                    "NAME:AllTabs",
                    [
                        # LocalVariableTab：在「設計層級的區域變數」分頁上新增變數
                        "NAME:LocalVariableTab",
                        [
                            "NAME:PropServers",
                            "LocalVariables"
                        ],
                        [
                            "NAME:NewProps",
                            [
                                "NAME:CooperH",                 # 銅厚 (Z 方向)
                                "PropType:=", "VariableProp",
                                "UserDef:=", True,
                                "Value:=", "0.035mm"
                            ],
                            [
                                "NAME:pixel_H",                 # 單一像素的 X 邊長
                                "PropType:=", "VariableProp",
                                "UserDef:=", True,
                                "Value:=", "0.25mm"
                            ],
                            [
                                "NAME:pixel_W",                 # 單一像素的 Y 邊長
                                "PropType:=", "VariableProp",
                                "UserDef:=", True,
                                "Value:=", "0.25mm"
                            ]
                        ]
                    ]
                ])

        if pixel_row*pixel_column == 2500:
            # 50x50 網格：最細，單像素邊長 0.1mm (像素越密、邊長越小)。
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
                                "Value:=", "0.035mm"
                            ],
                            [
                                "NAME:pixel_H",
                                "PropType:=", "VariableProp",
                                "UserDef:=", True,
                                "Value:=", "0.1mm"
                            ],
                            [
                                "NAME:pixel_W",
                                "PropType:=", "VariableProp",
                                "UserDef:=", True,
                                "Value:=", "0.1mm"
                            ]
                        ]
                    ]
                ])

        if pixel_row*pixel_column == 625:
            # 25x25 網格：本模擬器預設值 (pixel_count=25)，單像素邊長 0.2mm。
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
                                "Value:=", "0.035mm"
                            ],
                            [
                                "NAME:pixel_H",
                                "PropType:=", "VariableProp",
                                "UserDef:=", True,
                                "Value:=", "0.2mm"
                            ],
                            [
                                "NAME:pixel_W",
                                "PropType:=", "VariableProp",
                                "UserDef:=", True,
                                "Value:=", "0.2mm"
                            ]
                        ]
                    ]
                ])

        # 匯入底板
        #? Import：將預先做好的 sab 幾何 (基板 Sub、地 GND、兩條饋線 feedline1/feedline2、
        #?   兩個 port 用矩形 Rectangle1/Rectangle2) 一次匯入目前設計。
        #?   ImportMaterialNames=True 會一併帶入 sab 內定義的材質名稱；其餘為 ACIS/sab
        #?   匯入的幾何修復與容差參數 (沿用 HFSS 預設，請勿更動)。
        #? SourceFile 取 self.HFSS_sab_path → 即建構子指定的 dual_port.sab。
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

        # 指定基板材質：Sub 物件 → Rogers RO4003 (低損耗微波介電基板，εr≈3.55)。
        # SolveInside=True：介電質內部要求解電磁場 (基板內確實有場分佈，須納入計算)。
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

        # 指定導體材質：兩條饋線 feedline1/feedline2 與地 GND → copper。
        #! 與單埠版差異：單埠只有一條 "feed_line"，此處為「feedline1,feedline2」兩條，
        #!   分別連到兩個 port，是雙埠結構的根本不同。
        # SolveInside=False：良導體內部視為理想導體 (場無法穿入)，不在金屬內部求解 →
        #   大幅節省網格與運算量。
        oEditor.AssignMaterial(
            [
                "NAME:Selections",
                "AllowRegionDependentPartSelectionForPMLCreation:=", True,
                "AllowRegionSelectionForPMLCreation:=", True,
                "Selections:=", "feedline1,feedline2,GND"
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

        # 再次明確把基板 Sub 的 "Solve Inside" 設為 True (與上方 AssignMaterial 一致)，
        # 確保介電基板內部納入求解，避免匯入流程造成屬性被覆寫。
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

        # 將Patch Pexil 畫上
        # Create PatchBlock
        # 逐格掃描像素矩陣，凡值為 1 的格子就在對應位置畫一個銅方塊 (Box)，
        # 最終這些小方塊聯集 (Unite) 起來即為神經網路設計出的不規則 Patch 金屬面。
        #! 與單埠版差異：單埠版會逐一收集 Box 名稱 (patch_names) 並在 XSize/YSize 加
        #!   0.01mm 偏移避免奇異點；本雙埠版沿用較早的寫法——方塊統一命名為 "Patch"，
        #!   尺寸剛好等於 pixel_H/pixel_W，後續以另一套 Unite 邏輯逐欄聯集。
        for y in range(0, pixel_row, 1):
            for x in range(0, pixel_column, 1):
                # if pixel_matrix[x][y] > 0:
                #     one_num = one_num + 1
                if pixel_matrix[x][y] == 1:                  # 該像素要鋪銅
                    one_num = one_num + 1                    # 累計鋪銅像素數
                    oEditor.CreateBox(
                        [
                            "NAME:BoxParameters",
                            # 以區域變數 pixel_H/pixel_W 推算方塊左下角座標：
                            # 字串 "+pixel_H" 重複 x 次 → "0mm+pixel_H+pixel_H..." 即第 x 格的 X 起點。
                            # (HFSS 會把這段運算式自行解析成數值，故能直接用變數定位。)
                            "XPosition:=", "0mm" + str("+pixel_H" * x),
                            "YPosition:=", "0mm" + str("+pixel_W" * y),
                            "ZPosition:=", "0.508mm",        # Z 起點疊在 0.508mm 厚基板上表面
                            "XSize:=", "pixel_H",            # 方塊 X 邊長 = 一個像素寬
                            "YSize:=", "pixel_W",            # 方塊 Y 邊長 = 一個像素寬
                            "ZSize:=", "CooperH"             # 方塊厚度 = 銅厚
                        ],
                        [
                            "NAME:Attributes",
                            # 所有方塊同名 "Patch"，HFSS 會自動加後綴成 Patch_1、Patch_2…
                            # 後續 Unite 即依此「Patch_<序號>」名稱選取聯集。
                            "Name:=", "Patch",
                            "Flags:=", "",
                            "Color:=", "(255 0 0)",          # 以紅色顯示 Patch，便於視覺辨識
                            "Transparency:=", 0,
                            "PartCoordinateSystem:=", "Global",
                            "UDMId:=", "",
                            "MaterialValue:=", "\"copper\"",
                            "SurfaceMaterialValue:=", "\"\"",
                            "SolveInside:=", True,
                            "IsMaterialEditable:=", True,
                            "UseMaterialAppearance:=", False,
                            "IsLightweight:=", False
                        ])

        # 逐欄將同一行 (column) 內鋪銅的小方塊聯集成連續金屬，降低物件數量、利於後續網格化。
        # ones_buf：跨欄累計的「已生成方塊總數」，用來推算每個方塊在 HFSS 的全域序號 (Patch_<n>)。
        ones_buf = 0
        for i in range(pixel_row):
            patch_unite = ""                            # 累積本欄要聯集的方塊名稱字串 ("Patch_a,Patch_b,...")
            E = pixel_matrix[:, i]                       # 取出第 i 欄所有像素值
            # 使用 numpy.where 找到值為 1 的位置
            ones_indices = np.where(E == 1)[0]           # 本欄中鋪銅像素的列索引
            # 因為只有一個不能unite
            # Unite 至少需要兩個物件；本欄只有 0 或 1 個方塊就無從聯集，直接略過。
            if ones_indices.shape[0] > 1:
                for u in range(ones_indices.shape[0]):
                    # u+ones_buf == 0 對應 HFSS 的 "Patch" 本體 (無後綴/序號 0)，
                    # 命名規則使其不會出現於 "Patch_<n>" 系列，故跳過以免選到不存在的名稱。
                    if u+ones_buf == 0:
                        continue
                    # 串接全域序號：本欄第 u 個方塊的全域編號 = u + ones_buf。
                    patch_unite = patch_unite + "Patch_" + str(u+ones_buf) + ","

                patch_unite = patch_unite[:len(patch_unite)-1]   # 去掉字串尾端多餘的逗號

                # 因為只有一個不能unite
                # 若整串只剩 "Patch_1" 單一物件 (邊界情況)，同樣無法聯集，更新計數後略過。
                if patch_unite == "Patch_1":
                    ones_buf = ones_buf + ones_indices.shape[0]
                    continue

                # 對本欄收集到的方塊執行聯集；KeepOriginals=False 表示聯集後不保留原始小方塊，
                # 直接融合成單一連續銅面 (預設保留清單中第一個物件的名稱)。
                oEditor.Unite(
                    [
                        "NAME:Selections",
                        "Selections:=", patch_unite
                    ],
                    [
                        "NAME:UniteParameters",
                        "KeepOriginals:=", False
                    ])
            # 不論本欄是否聯集，皆把本欄方塊數累加進全域計數，供下一欄推算正確序號。
            ones_buf = ones_buf + ones_indices.shape[0]

        # 設定邊界條件
        oModule = oDesign.GetModule("BoundarySetup")        # 取得邊界/激勵設定模組


        #! 與單埠版最大差異：此處要指定「兩個 LumpedPort (集總埠)」，激勵 S 矩陣才會是 2x2，
        #!   進而能輸出 S11/S21/S22。單埠版只 AssignLumpedPort 一次，只有 S11。
        # Port 1：建立在矩形 Rectangle1 上，對應 feedline1。
        # DoDeembed=True：去嵌入，把 port 平面前一段饋線的相位/寄生效應扣除，得到參考面上的純結果。
        # IntLine (積分線)：界定激勵電場的方向與正負，須由地 (z≈0) 指向訊號線 (z=0.508mm)，
        #   讓 HFSS 知道此 port 的電壓參考方向。
        # CharImp="Zpi"、RenormImp/Impedance="50ohm"：以 Zpi 計算特性阻抗，並重新歸一化到 50 歐姆系統。
        oModule.AssignLumpedPort(
            [
                "NAME:1",
                "Objects:=", ["Rectangle1"],
                "DoDeembed:=", True,
                "RenormalizeAllTerminals:=", True,
                [
                    "NAME:Modes",
                    [
                        "NAME:Mode1",
                        "ModeNum:=", 1,
                        "UseIntLine:=", True,
                        [
                            "NAME:IntLine",
                            # port1 積分線位於 x=12.5mm 處 (feedline1 端)，由基板底面拉到上表面。
                            "Start:=", ["12.5mm", "2.5mm",
                                        "9.99200722162641e-17mm"],
                            "End:=", ["12.5mm", "2.5mm", "0.508mm"]
                        ],
                        "AlignmentGroup:=", 0,
                        "CharImp:=", "Zpi",
                        "RenormImp:=", "50ohm"
                    ]
                ],
                "ShowReporterFilter:=", False,
                "ReporterFilter:=", [True],
                "Impedance:=", "50ohm"
            ])

        # Port 2：建立在矩形 Rectangle2 上，對應另一側的 feedline2。
        # 有了第二個 port，HFSS 才能算出兩埠間的傳輸/互耦 S21 與 port2 自身反射 S22。
        oModule.AssignLumpedPort(
            [
                "NAME:2",
                "Objects:=", ["Rectangle2"],
                "DoDeembed:=", True,
                "RenormalizeAllTerminals:=", True,
                [
                    "NAME:Modes",
                    [
                        "NAME:Mode2",
                        "ModeNum:=", 1,  #這邊不能改2，因為要和上面的lumport組成一組
                        "UseIntLine:=", True,
                        [
                            "NAME:IntLine",
                            # port2 積分線位於 x=-7.5mm 處 (feedline2 端)，方向同樣由底面指向上表面，
                            # 與 port1 保持一致的電壓參考方向，才能正確組成 2x2 S 矩陣。
                            "Start:=", ["-7.5mm", "2.5mm",
                                        "9.99200722162641e-17mm"],
                            "End:=", ["-7.5mm", "2.5mm", "0.508mm"]
                        ],
                        "AlignmentGroup:=", 0,
                        "CharImp:=", "Zpi",
                        "RenormImp:=", "50ohm"
                    ]
                ],
                "ShowReporterFilter:=", False,
                "ReporterFilter:=", [True],
                "Impedance:=", "50ohm"
            ])

        # 設定輻射開放邊界 (Open Region)：
        # CreateOpenRegion 會在結構外圍自動套用輻射邊界 (Radiation)，模擬天線向自由空間輻射，
        # 而非被金屬牆封閉。OpFreq=28GHz 為設定吸收邊界距離所用的中心頻率。
        # ApplyInfiniteGP=False：不假設無限大地平面 (本設計地為有限尺寸)。
        oModule = oDesign.GetModule("ModelSetup")
        oModule.CreateOpenRegion(
            [
                "NAME:Settings",
                "OpFreq:=", "28GHz",
                "Boundary:=", "Radiation",
                "ApplyInfiniteGP:=", False
            ])

        ###* 模擬設定 ###
        # 建立求解設定 Setup1：以 28GHz 為自適應網格 (adaptive mesh) 的求解頻率。
        # MaxDeltaS=0.02：相鄰兩次自適應 pass 間 S 參數變化量門檻，<0.02 即視為收斂。
        # MaximumPasses/MinimumPasses/MinimumConvergedPasses：自適應加密的上下限與最少連續收斂次數，
        #   在「精度」與「運算時間」間取得平衡。BasisOrder=1 為一階基底函數。
        oModule = oDesign.GetModule("AnalysisSetup")
        oModule.InsertSetup("HfssDriven",
            [
                "NAME:Setup1",
                "SolveType:=", "Single",
                "Frequency:=", "28GHz",
                "MaxDeltaS:=", 0.02,
                "UseMatrixConv:=", False,
                "MaximumPasses:=", 6,
                "MinimumPasses:=", 5,
                "MinimumConvergedPasses:=", 5,
                "PercentRefinement:=", 30,
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
        # 設定頻率掃描：24~32GHz、步進 0.5GHz → 共 17 個頻點 (與後續取 0:17 列、插值補 17 點一致)。
        #! 與單埠版差異：本雙埠版 Type 用 "Fast" 掃描法；單埠版改用 "Interpolating" (插值掃描)。
        #!   兩者皆只在少數頻點實際求解再重建頻率響應，差別在重建演算法。
        oModule.InsertFrequencySweep("Setup1",
            [
                "NAME:Sweep",
                "IsEnabled:=", True,
                "RangeType:=", "LinearStep",
                "RangeStart:=", "24GHz",
                "RangeEnd:=", "32GHz",
                "RangeStep:=", "0.5GHz",
                "Type:=", "Fast",
                "SaveFields:=", True,
                "SaveRadFields:=", False,
                "GenerateFieldsForAllFreqs:=", False,
                "ExtrapToDC:=", False
            ])

        # 設定遠場輻射球 (Infinite Sphere)：theta/phi 皆 -180~180deg、步進 2deg，
        # 供日後計算/檢視遠場輻射場型之用 (本雙埠流程最終未匯出增益，但保留此設定)。
        oModule = oDesign.GetModule("RadField")
        oModule.EditInfiniteSphereSetup("3D",
            [
                "NAME:3D",
                "UseCustomRadiationSurface:=", False,
                "ThetaStart:=", "-180deg",
                "ThetaStop:=", "180deg",
                "ThetaStep:=", "2deg",
                "PhiStart:=", "-180deg",
                "PhiStop:=", "180deg",
                "PhiStep:=", "2deg",
                "UseLocalCS:=", False
            ])

        # self.oProject.Save()

        # 開始模擬
        # AnalyzeAll：觸發 HFSS 求解全部設定 (自適應網格 + 頻率掃描)，此為最耗時的真實全波模擬步驟。
        oDesign.AnalyzeAll()

        # 畫出結果
        oModule = oDesign.GetModule("ReportSetup")          # 取得報表模組，用來建立可匯出的曲線

        #! 雙埠核心：建立三張 S 參數報表 (S11/S21/S22)，對應 2x2 S 矩陣的三個獨立量。
        #!   被動互易結構下 S21=S12，故只需取其一即可代表兩埠間互耦。
        # Create S11
        # S(1,1)：port1 反射係數 (回波損耗)，反映 port1 阻抗匹配優劣，越負代表越匹配。
        oModule.CreateReport("S Parameter Plot 1", "Modal Solution Data", "Rectangular Plot", "Setup1 : Sweep",
            [
                "Domain:=", "Sweep"
            ],
            [
                "Freq:=", ["All"]
            ],
            [
                "X Component:=", "Freq",
                "Y Component:=", ["dB(S(1,1))"]
            ])

        # S(2,1)：port1→port2 傳輸係數，即「兩埠互耦 / 隔離度」——單埠版完全沒有的物理量，
        #   是雙埠設計關注的重點 (例如兩天線間是否充分隔離)。
        oModule.CreateReport("S Parameter Plot 2", "Modal Solution Data", "Rectangular Plot", "Setup1 : Sweep",
            [
                "Domain:=", "Sweep"
            ],
            [
                "Freq:=", ["All"]
            ],
            [
                "X Component:=", "Freq",
                "Y Component:=", ["dB(S(2,1))"]
            ])

        # S(2,2)：port2 反射係數，反映 port2 自身的阻抗匹配 (結構通常不完全對稱，故與 S11 不一定相同)。
        oModule.CreateReport("S Parameter Plot 3", "Modal Solution Data", "Rectangular Plot", "Setup1 : Sweep",
            [
                "Domain:=", "Sweep"
            ],
            [
                "Freq:=", ["All"]
            ],
            [
                "X Component:=", "Freq",
                "Y Component:=", ["dB(S(2,2))"]
            ])

        #* Export csv
        # 將三張報表分別匯出成 CSV，檔名格式 NN_patch_Sparameter_{num}_S11/S21/S22.csv，
        # {num} 為當前 pattern 編號 (self.num)，供下方讀回與後續計算 loss。第三引數 False = 不只匯出可見區。
        oModule.ExportToFile("S Parameter Plot 1", self.path_result.joinpath(f"NN_patch_Sparameter_{self.num}_S11.csv"), False)
        oModule.ExportToFile("S Parameter Plot 2", self.path_result.joinpath(f"NN_patch_Sparameter_{self.num}_S21.csv"), False)
        oModule.ExportToFile("S Parameter Plot 3", self.path_result.joinpath(f"NN_patch_Sparameter_{self.num}_S22.csv"), False)


        #* Read csv
        # 把剛匯出的三個 CSV 讀回成 DataFrame：第 0 欄為頻率、第 1 欄為對應的 dB 值。
        Sparameter_dataframe_11 = read_csv(self.path_result.joinpath(f"NN_patch_Sparameter_{self.num}_S11.csv"))
        Sparameter_dataframe_21 = read_csv(self.path_result.joinpath(f"NN_patch_Sparameter_{self.num}_S21.csv"))
        Sparameter_dataframe_22 = read_csv(self.path_result.joinpath(f"NN_patch_Sparameter_{self.num}_S22.csv"))

        #  將數值取出 之後要算loss
        # 取前 17 列、第 1 欄 (數值欄)，對齊 24~32GHz / 17 頻點的固定長度，便於與目標響應計算 loss。
        #! 與單埠版差異：單埠版會檢查實際點數並用 np.interp 插值補足到 17 點；
        #!   本雙埠版此處直接以 iloc[0:17] 截取 (假設 HFSS 已輸出足量頻點)。
        S11 = Sparameter_dataframe_11.iloc[0:17, 1]
        S21 = Sparameter_dataframe_21.iloc[0:17, 1]
        S22 = Sparameter_dataframe_22.iloc[0:17, 1]

        full_output = []                                    # 起始空容器，供下方串接成單一向量

        # TODO
        # 回傳結果 dict：注意這裡取「完整」第 1 欄 (iloc[:, 1]) 轉成 tensor，而非上面截到 17 點的版本。
        # 三個 key 分別對應 port1 反射 / 兩埠互耦 / port2 反射，交給訓練流程做損失計算與紀錄。
        _result = {
            'S11': tensor(Sparameter_dataframe_11.iloc[:, 1].to_list()),
            'S21': tensor(Sparameter_dataframe_21.iloc[:, 1].to_list()),
            'S22': tensor(Sparameter_dataframe_22.iloc[:, 1].to_list())
        }


        # 將 S11/S21/S22 (各 17 點) 依序串接成單一一維向量；目前僅供除錯/後續擴充，未直接回傳。
        full_parameter = np.append(np.append(np.append(full_output, S11.to_numpy()), S21.to_numpy()),S22.to_numpy())
        # breakpoint()
        return _result