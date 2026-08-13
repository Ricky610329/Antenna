import numpy as np
from loguru import logger
from pandas import read_csv
from torch import Tensor, tensor   # 注意：torch.tensor (與 antenna.utils 的自訂 tensor 不同)
from ...utils import Path
from . import PatchSimulator
from .single_port import align_curve   #! 頻率對位共用同一份實作 (別另寫一份 → 兩把尺會漂)

#! 幾何儀器版本(dedust run 逐筆蓋章進 results.json 的 geom 欄):
#  p00=學長原版(逐欄 Unite/盒剛好像素尺寸→對角零厚度接觸=數值幻影,R69 定讞)
#  p01=2026-08-13 換代(follow 單埠:+0.01mm 重疊/全聯集含饋線/SolveInside=False)
#  跨代量測對「對角重 pattern」不可比;乾淨結構(無真對角接點)預期近乎不變(括號驗證)。
GEOM_VER = "p01"


def slot_boxes(slot_spec, pixel_count: int):
    """把 `slot_spec` 展開成「縫盒」幾何 (mm)。**純函式、不碰 COM** → 開發機可測。

    R60 亞像素耦合縫 (round-60 §1)：R59 證明 0.2mm 格距下耦合縫只有「直通/斷路」兩個
    量子態；本函式讓縫寬變成連續幾何參數 (bits 不變，模擬器在指定位置 Subtract 挖細縫)。

    **座標對應 (以 `__call__` 的 CreateBox 為準，別憑直覺)**::

        pixel_matrix[r][c]  →  XPosition = r * pixel_H,  YPosition = c * pixel_W

    也就是**第一維 (列 r) 走 HFSS X 軸、第二維 (欄 c) 走 HFSS Y 軸**。旁證：兩個饋墊
    `q[0:5, 10:15]` / `q[20:25, 10:15]` (`dedust.dual_pads`) → X∈[0,1]∪[4,5]mm、
    Y∈[2,3]mm，恰好對上兩個 port 的 IntLine (x=-7.5mm / x=+12.5mm，兩者 y=2.5mm)。
    ⇒ 兩埠沿 **X** 分居兩端，「切第 r 列」＝在直通路上豎一道牆 (R59 的 `r12 全切`)。

    格式 (整夾一組，來自 `hfss_setup.json`)::

        [{"rows": [11, 13], "cols": [10, 16], "width_mm": 0.05}, ...]

    * ``rows``：要開縫的像素列 r 清單；縫**厚度落在 X 軸**、中心＝第 r 列的中心線
      ``(r+0.5)*pmm``。故 ``width_mm → pmm`` 時退化成「整列清空」(括號自證的上端)。
    * ``cols``：``[c0, c1]`` 像素欄閉區間；縫**沿 Y 軸延伸** ``c0*pmm ~ (c1+1)*pmm``。
    * ``width_mm``：縫寬 (mm)，需 ``0 < w < pmm``。上下界都拒收——``w=0`` 請直接不給
      ``slot_spec``；``w >= pmm`` 請在 bits 上清列 (幾何上會把單像素體挖成空物件)。

    對稱性由呼叫端負責 (鏡像列自己寫進 rows)，本函式不自動鏡像。

    :return: 每條縫一個 dict：``row/c0/c1/w`` (像素域) + ``x0/y0/dx/dy`` (mm，盒左下角與邊長)。
    """
    pmm = 5.0 / pixel_count                       # 像素邊長 mm (25→0.2 / 50→0.1)；貼片實體固定 5mm 見方
    if isinstance(slot_spec, dict):
        raise ValueError("slot_spec 必須是 list（一條縫一個 dict），不是單一 dict")
    out = []
    for k, item in enumerate(slot_spec):
        tag = f"slot_spec[{k}]"
        if not isinstance(item, dict):
            raise ValueError(f"{tag} 必須是 dict（鍵 rows/cols/width_mm）")
        #! 鍵名嚴格比對（多一個少一個都擋）：打錯鍵最惡劣的下場是「靜默不挖」——HFSS 照跑完、
        #  S 參數照樣長得像回事，整夾白燒還以為量到了 w 的效應。
        if set(item) != {"rows", "cols", "width_mm"}:
            raise ValueError(f"{tag} 鍵不對 {sorted(item)}（只收且必須有 rows/cols/width_mm）")
        w = float(item["width_mm"])
        if not 0.0 < w < pmm:
            raise ValueError(f"{tag} width_mm={w} 超出 (0, {pmm})：w=0 請不要給 slot_spec，"
                             f"w>=像素邊長請直接在 bits 上清列")
        cols = list(item["cols"])
        if len(cols) != 2:
            raise ValueError(f"{tag} cols 必須是 [c0, c1] 閉區間")
        c0, c1 = int(cols[0]), int(cols[1])
        if not 0 <= c0 <= c1 < pixel_count:
            raise ValueError(f"{tag} cols={c0},{c1} 越界（需 0<=c0<=c1<{pixel_count}）")
        rows = list(item["rows"])
        if not rows:
            raise ValueError(f"{tag} rows 是空的")
        for r in rows:
            r = int(r)
            if not 0 <= r < pixel_count:
                raise ValueError(f"{tag} row={r} 越界（需 0<=r<{pixel_count}）")
            out.append(dict(row=r, c0=c0, c1=c1, w=w,
                            x0=(r + 0.5) * pmm - w / 2.0,   # 厚度方向＝X，中心對準第 r 列中心線
                            dx=w,
                            y0=c0 * pmm,                    # 延伸方向＝Y，跨 c0..c1 含
                            dy=(c1 - c0 + 1) * pmm))
    return out


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

    def __init__(self, record_path, HFSS_sab_path = Path(__file__).parent.joinpath('sab', 'dual_port.sab'), pixel_count:int = 25,
                 sweep_type: str = "Fast",
                 max_delta_s: float = 0.02, max_passes: int = 6, min_passes: int = 5, min_converged: int = 5,
                 slot_spec: list = None, diag_bridge_w: float = None):
        # record_path：本次訓練/實驗的紀錄根目錄 (基底類別會在其下建立 HFSS/、result/、project/)。
        # HFSS_sab_path：雙埠底板幾何檔，預設指向與本檔同層 sab/ 目錄下的 dual_port.sab，
        #               內含基板 Sub、地 GND、兩條饋線 feedline1/feedline2 及兩個 port 用矩形。
        #               (單埠版預設為 single_port.sab，僅一條 feed_line。)
        # pixel_count：每邊像素數，預設 25 → 25x25=625 個像素的可佈線網格。
        #? sweep_type: HFSS 掃頻演算法 {"Fast"(**本檔預設**;自適應網格頻點+快速重建) /
        #  "Interpolating"(single 的預設,自選頻點+有理擬合) / "Discrete"(17 點逐點硬解,慢但每點真解)}。
        #! 預設刻意留 "Fast"＝本檔歷來寫死的值——`harvest_dual` 一萬筆全是這個設定跑出來的,
        #  換掃頻法等於換分佈 (SM 冷啟動/暖啟價值打折,兩者的重建演算法不同)。要換先想清楚代價。
        self.sweep_type = str(sweep_type)
        #? 自適應網格收斂設定 (簽名與預設值同 single_port,方便批次線兩埠共用同一組旋鈕)。
        #  預設值=本檔歷來寫死的值 (0.02/6/5/5)——不帶參數的行為與所有既有 dual 真值同設定;
        #  覆蓋來源=批次線讀輸入夾 hfss_setup.json,不進 config、不進訓練管線。
        self.max_delta_s = float(max_delta_s)
        self.max_passes = int(max_passes)
        self.min_passes = int(min_passes)
        self.min_converged = int(min_converged)
        #? 亞像素耦合縫(2026-08-11 R60,round-60 §1):bits 不變、在指定位置挖指定寬度的細縫
        #  (Subtract);None=現行幾何(完全向後相容,整段不執行)。格式與座標對應見 slot_boxes()。
        #  覆蓋來源=批次線輸入夾的 hfss_setup.json 的 slot_spec 鍵(整夾一組)。
        #! 建構時就展開一次＝**發車前就炸**(不等 HFSS 開起來、跑一半才發現格式錯)。
        self.slot_spec = None if slot_spec is None else list(slot_spec)
        if self.slot_spec is not None:
            slot_boxes(self.slot_spec, pixel_count)
        #? R54 菱形對角橋移植(2026-08-13,Ricky 核「湯物種救援」實驗):正值=45° 菱形橋
        #  (寬 w mm,Unite 進導體)、負值=45° 挖空槽(切斷對角,Subtract)、None=不加料。
        #  站點計算共用單埠 diag_bridge_sites(同一份邏輯,兩把尺不漂);p01 幾何下對角
        #  本已是 0.01mm 真橋,此參數用於劑量法問「加寬到可蝕刻寬度(0.075)還活著嗎」。
        self.diag_bridge_w = None if diag_bridge_w is None else float(diag_bridge_w)
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
        #! 2026-08-13 儀器換代 GEOM_VER=p01:本段已改 follow 單埠版——+0.01mm 重疊、
        #!   全域分批聯集、與饋線合體、SolveInside=False。歷史(p00=逐欄 Unite/剛好尺寸)
        #!   的對角零厚度接觸被 R69 證為數值幻影(fill 探針 Δ-20.8/尺度依賴 Δ-3~-5),
        #!   換代理由與證據=docs/log/round-69 §4。跨代量測不可比(對角重 pattern)。
        #? box_names：依**建立順序**記下 HFSS 會給的名字(第 0 個叫 "Patch"、之後 "Patch_<序>")——
        #  與下方 Unite 的命名推算是同一套規則。R60 挖縫的 Subtract 需要知道「最後還活著哪些物件」，
        #  這份清單＋Unite 的吞併記帳就是答案（slot_spec=None 時只是純 Python append，幾何零影響）。
        box_names = []
        for y in range(0, pixel_row, 1):
            for x in range(0, pixel_column, 1):
                # if pixel_matrix[x][y] > 0:
                #     one_num = one_num + 1
                if pixel_matrix[x][y] == 1:                  # 該像素要鋪銅
                    one_num = one_num + 1                    # 累計鋪銅像素數
                    box_names.append("Patch" if not box_names else f"Patch_{len(box_names)}")
                    oEditor.CreateBox(
                        [
                            "NAME:BoxParameters",
                            # 以區域變數 pixel_H/pixel_W 推算方塊左下角座標：
                            # 字串 "+pixel_H" 重複 x 次 → "0mm+pixel_H+pixel_H..." 即第 x 格的 X 起點。
                            # (HFSS 會把這段運算式自行解析成數值，故能直接用變數定位。)
                            "XPosition:=", "0mm" + str("+pixel_H" * x),
                            "YPosition:=", "0mm" + str("+pixel_W" * y),
                            "ZPosition:=", "0.508mm",        # Z 起點疊在 0.508mm 厚基板上表面
                            "XSize:=", "pixel_H + 0.01mm",   #! 避免奇異點(2026-08-13 儀器換代 p01,follow 單埠)
                            # 盒子故意比一格像素大 0.01mm:相鄰像素「重疊」而非邊對邊零厚度相接。
                            # R69 實證:零厚度接觸(尤其對角共邊)的導通是求解器數值縫合,強度隨網格/
                            # 離散度漂移(fill 探針 Δ-20.8/50×50 括號崩 3-5dB)——重疊後對角=真實
                            # 0.01mm 橋,幾何定義良好、尺度不變(可製造性另由 R54 菱形橋劑量法處理)。
                            "YSize:=", "pixel_W + 0.01mm",   #!
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
                            "SolveInside:=", False,          #! 隨換代改 follow 單埠:良導體只解表面,省網格
                            "IsMaterialEditable:=", True,
                            "UseMaterialAppearance:=", False,
                            "IsLightweight:=", False
                        ])

        ###* 45° 菱形對角橋/挖空槽(R54 移植;self.diag_bridge_w=None 時整段不執行) ###
        diag_slot_names = []                              # 挖空模式(w<0):全聯集後從導體 Subtract
        if self.diag_bridge_w is not None:
            from .single_port import diag_bridge_sites    # 站點邏輯共用單埠(同一份,兩把尺不漂)
            _slot_mode = self.diag_bridge_w < 0
            _pmm = 5.0 / pixel_row
            _sites, _skipped = diag_bridge_sites((pixel_matrix.numpy() > 0.5), abs(self.diag_bridge_w), _pmm)
            logger.info(f"{'挖空槽(斷開)' if _slot_mode else '菱形對角橋'}: {len(_sites)} 座"
                        f"(w={self.diag_bridge_w}mm,跳過 {_skipped});Pattern {self.num}")
            for _k, (_cx, _cy, _w) in enumerate(_sites):
                _dn = f"{'DiagSl' if _slot_mode else 'DiagBr'}_{_k + 1}"
                #? 原點置中建盒 → 繞全域 Z 轉 45° → Move 到角點(單埠同式:HFSS Rotate 只繞全域軸)。
                _actual = oEditor.CreateBox(
                    ["NAME:BoxParameters",
                     "XPosition:=", f"{-_w / 2}mm", "YPosition:=", f"{-_w / 2}mm", "ZPosition:=", "0.508mm",
                     "XSize:=", f"{_w}mm", "YSize:=", f"{_w}mm", "ZSize:=", "CooperH"],
                    ["NAME:Attributes", "Name:=", _dn, "Flags:=", "", "Color:=", "(255 128 0)",
                     "Transparency:=", 0, "PartCoordinateSystem:=", "Global", "UDMId:=", "",
                     "MaterialValue:=", "\"copper\"", "SurfaceMaterialValue:=", "\"\"",
                     "SolveInside:=", False, "IsMaterialEditable:=", True,
                     "UseMaterialAppearance:=", False, "IsLightweight:=", False])
                oEditor.Rotate(
                    ["NAME:Selections", "Selections:=", _actual, "NewPartsModelFlag:=", "Model"],
                    ["NAME:RotateParameters", "RotateAxis:=", "Z", "RotateAngle:=", "45deg"])
                oEditor.Move(
                    ["NAME:Selections", "Selections:=", _actual, "NewPartsModelFlag:=", "Model"],
                    ["NAME:TranslateParameters", "TranslateVectorX:=", f"{_cx}mm",
                     "TranslateVectorY:=", f"{_cy}mm", "TranslateVectorZ:=", "0mm"])
                # 橋=進分批 Unite(與導體同體);槽=收著,等全聯集後整批 Subtract
                (diag_slot_names if _slot_mode else box_names).append(_actual)

        ###* 全域分批聯集(2026-08-13 儀器換代 p01,follow 單埠版)###
        #! 舊版=逐欄 Unite(欄間/對角=零厚度接觸→數值奇異點;R69 實證其導通強度隨網格/
        #!   離散度漂移=幻影)。新版照單埠三階段:分批(20/批)互聯 → 全區塊合一 →
        #!   與 feedline1/feedline2 合體=**整個導體同一個物件**(Ricky 裁定 2026-08-13)。
        #! +0.01mm 放大使貼片盒與饋線金屬重疊,不聯集會留下重疊物件炸網格——與饋線合體
        #!   是幾何必需,不只是風格;port 激勵面 Rectangle1/2 為獨立 sheet,不受影響。
        if box_names:
            chunk_size = 20                     # 一次 Unite 過多物件會讓 COM/幾何核心過載(同單埠註解)
            united_chunks = []
            for _i in range(0, len(box_names), chunk_size):
                _chunk = box_names[_i:_i + chunk_size]
                if len(_chunk) > 1:
                    oEditor.Unite(
                        ["NAME:Selections", "Selections:=", ",".join(_chunk)],
                        ["NAME:UniteParameters", "KeepOriginals:=", False])
                united_chunks.append(_chunk[0])  # Unite 保留清單首名
            if len(united_chunks) > 1:
                oEditor.Unite(
                    ["NAME:Selections", "Selections:=", ",".join(united_chunks)],
                    ["NAME:UniteParameters", "KeepOriginals:=", False])
            oEditor.Unite(
                ["NAME:Selections", "Selections:=", f"feedline1,feedline2,{united_chunks[0]}"],
                ["NAME:UniteParameters", "KeepOriginals:=", False])
            # 此後整個導體(貼片+兩饋線)存活名=feedline1;上下貼片區若無金屬相連,
            # 仍是同一物件的兩個 lobe(HFSS 允許),電性不受聯集影響。
            ###* 45° 挖空槽 Subtract(diag_bridge_w<0 才有;時機=全聯集後,同單埠) ###
            for _i in range(0, len(diag_slot_names), 20):  # 分批理由同 Unite:COM 選取字串過長易崩
                oEditor.Subtract(
                    ["NAME:Selections", "Blank Parts:=", "feedline1",
                     "Tool Parts:=", ",".join(diag_slot_names[_i:_i + 20])],
                    ["NAME:SubtractParameters", "KeepOriginals:=", False])

        ###* 亞像素耦合縫：挖縫盒 → Subtract（R60；self.slot_spec=None 時整段不執行＝歷史幾何 bit 級不變）###
        if self.slot_spec is not None:
            patch_bodies = ["feedline1"] if box_names else []   # 全聯集後整個導體=單一物件(p01 儀器換代)
            _slots = slot_boxes(self.slot_spec, pixel_row)
            _mat = pixel_matrix.numpy() > 0.5
            _names = []
            if not patch_bodies:
                logger.warning(f"slot_spec 指定了 {len(_slots)} 條縫，但這張 pattern 沒有任何金屬"
                               f"——整段跳過（Pattern {self.num}）")
                _slots = []                          # 沒東西可挖就連盒子都別建（免留一塊懸空的銅片在模型裡）
            for _s in _slots:
                #? 空操作偵測：縫若落在全空區域，幾何上等於沒挖 → 結果會與「無縫」不可分辨。
                #  這是括號自證(w→0 收斂)最危險的假陽性來源，故顯性告警而非靜默。
                _cover = int(_mat[_s["row"], _s["c0"]:_s["c1"] + 1].sum())
                _dn = f"Slot_{len(_names) + 1}"
                _actual = oEditor.CreateBox(
                    ["NAME:BoxParameters",
                     #! X＝厚度方向（列 r 的中心線 ± w/2）、Y＝延伸方向（欄 c0..c1）——別搞反，
                     #  搞反縫會開成縱向（與兩埠直通路平行＝完全不切耦合路徑）。考證見 slot_boxes()。
                     "XPosition:=", f"{_s['x0']:.6f}mm", "YPosition:=", f"{_s['y0']:.6f}mm",
                     "ZPosition:=", "0.498mm",               # =0.508-0.01：上下各留 0.01mm 餘裕，防與銅層共面害布林失敗
                     "XSize:=", f"{_s['dx']:.6f}mm", "YSize:=", f"{_s['dy']:.6f}mm",
                     "ZSize:=", "CooperH+0.02mm"],
                    ["NAME:Attributes", "Name:=", _dn, "Flags:=", "", "Color:=", "(0 128 255)",
                     "Transparency:=", 0, "PartCoordinateSystem:=", "Global", "UDMId:=", "",
                     "MaterialValue:=", "\"copper\"", "SurfaceMaterialValue:=", "\"\"",
                     "SolveInside:=", False, "IsMaterialEditable:=", True,
                     "UseMaterialAppearance:=", False, "IsLightweight:=", False])
                _names.append(_actual)
                logger.info(f"耦合縫 {_actual}: row {_s['row']} cols {_s['c0']}-{_s['c1']} w={_s['w']}mm"
                            f"（X {_s['x0']:.4f}+{_s['dx']:.4f} / Y {_s['y0']:.4f}+{_s['dy']:.4f}mm，"
                            f"覆蓋金屬 {_cover}px）；Pattern {self.num}")
                if _cover == 0:
                    logger.warning(f"{_actual} 覆蓋 0 個金屬像素＝幾何空操作（Pattern {self.num}）")
            #! 時機：必須在全聯集之後——對單一導體(存活名 feedline1)下刀。刀具(縫盒)分批 20 個,
            #  理由同 Unite:COM 選取字串過長易崩。p01 換代後導體含饋線,但縫盒幾何嚴格落在像素區
            #  (X∈[0,5]mm),搆不到饋線本體(僅 0.01mm 重疊薄片),不會誤切饋電路徑。
            for _i in range(0, len(_names), 20):
                oEditor.Subtract(
                    ["NAME:Selections", "Blank Parts:=", ",".join(patch_bodies),
                     "Tool Parts:=", ",".join(_names[_i:_i + 20])],
                    ["NAME:SubtractParameters", "KeepOriginals:=", False])

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
                "MaxDeltaS:=", self.max_delta_s,       # 收斂門檻：相鄰兩次細化的 S 參數最大變化 < 此值即視為收斂
                "UseMatrixConv:=", False,
                "MaximumPasses:=", self.max_passes,    # 最多細化次數 (上限，防止無止盡細化)
                "MinimumPasses:=", self.min_passes,    # 至少細化次數 (確保網格足夠)
                "MinimumConvergedPasses:=", self.min_converged,  # 至少連續 N 次都收斂才算真收斂 (避免假性收斂)
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
        # 設定頻率掃描：24~32GHz、步進 0.5GHz → 共 17 個頻點 (與後續 align_curve 對齊的 17 點網格一致)。
        #! 與單埠版差異：本雙埠版預設 Type 用 "Fast" 掃描法；單埠版預設 "Interpolating" (插值掃描)。
        #!   兩者皆只在少數頻點實際求解再重建頻率響應，差別在重建演算法 (故兩埠數值不可混為同分佈)。
        logger.info(f"FrequencySweep Type={self.sweep_type} (Pattern {self.num})")   # 掃頻型式可觀測
        oModule.InsertFrequencySweep("Setup1",
            [
                "NAME:Sweep",
                "IsEnabled:=", True,
                "RangeType:=", "LinearStep",
                "RangeStart:=", "24GHz",
                "RangeEnd:=", "32GHz",
                "RangeStep:=", "0.5GHz",
                "Type:=", self.sweep_type,       #! 掃頻演算法 (建構參數;預設 Fast = harvest_dual 同設定)
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
        #! 匯出前先刪同名舊檔 (照 single_port 的做法)：檔名只含批內編號 → 跨批共用工作目錄時,若本次匯出
        #  silently 失敗,讀回的會是「上一批同編號的殘留 CSV」＝無聲污染 (single 的 verify-discrete 實際踩到)。
        #  先刪掉 → 匯出失敗會在 read_csv 炸 FileNotFoundError,錯誤變成看得見的。
        for _stale in (self.path_result.joinpath(f"NN_patch_Sparameter_{self.num}_S11.csv"),
                       self.path_result.joinpath(f"NN_patch_Sparameter_{self.num}_S21.csv"),
                       self.path_result.joinpath(f"NN_patch_Sparameter_{self.num}_S22.csv")):
            if _stale.exists():
                _stale.unlink()
        oModule.ExportToFile("S Parameter Plot 1", self.path_result.joinpath(f"NN_patch_Sparameter_{self.num}_S11.csv"), False)
        oModule.ExportToFile("S Parameter Plot 2", self.path_result.joinpath(f"NN_patch_Sparameter_{self.num}_S21.csv"), False)
        oModule.ExportToFile("S Parameter Plot 3", self.path_result.joinpath(f"NN_patch_Sparameter_{self.num}_S22.csv"), False)


        #* Read csv
        # 把剛匯出的三個 CSV 讀回成 DataFrame：第 0 欄為頻率、第 1 欄為對應的 dB 值。
        Sparameter_dataframe_11 = read_csv(self.path_result.joinpath(f"NN_patch_Sparameter_{self.num}_S11.csv"))
        Sparameter_dataframe_21 = read_csv(self.path_result.joinpath(f"NN_patch_Sparameter_{self.num}_S21.csv"))
        Sparameter_dataframe_22 = read_csv(self.path_result.joinpath(f"NN_patch_Sparameter_{self.num}_S22.csv"))

        #  將數值取出 之後要算loss
        #? 24~32GHz 等分 17 點 (對應掃頻步距 0.5GHz)；訓練端 (SM/loss/margin) 假設響應固定長度 17。
        freqs_expected = np.linspace(24, 32, 17)

        #* 三條 S 參數一律用 align_curve **按頻率值對位**到 17 點網格 (與 single_port 同一份實作)。
        #! 修掉的舊 bug (2026-08-10)：舊碼上面用 iloc[0:17] 截取、但下面回傳的 dict 卻用 iloc[:, 1] 全長 →
        #  HFSS 只要吐出 ≠17 個頻點 (Fast 掃頻的重建點數不保證),回傳張量長度就錯,而且**不會報錯**
        #  (下游 reshape/loss 靜默錯位)。改成按頻率對位 + 長度斷言,錯誤變成看得見的。
        S11_vals = align_curve(Sparameter_dataframe_11.iloc[:, 0].values,
                               Sparameter_dataframe_11.iloc[:, 1].values, freqs_expected)
        S21_vals = align_curve(Sparameter_dataframe_21.iloc[:, 0].values,
                               Sparameter_dataframe_21.iloc[:, 1].values, freqs_expected)
        S22_vals = align_curve(Sparameter_dataframe_22.iloc[:, 0].values,
                               Sparameter_dataframe_22.iloc[:, 1].values, freqs_expected)
        assert len(S11_vals) == 17, f"S11 對齊後長度 {len(S11_vals)} != 17 (Pattern {self.num})"
        assert len(S21_vals) == 17, f"S21 對齊後長度 {len(S21_vals)} != 17 (Pattern {self.num})"
        assert len(S22_vals) == 17, f"S22 對齊後長度 {len(S22_vals)} != 17 (Pattern {self.num})"

        # 回傳結果 dict：三個 key 分別對應 port1 反射 / 兩埠互耦 / port2 反射，
        # 交給訓練流程做損失計算與紀錄 (各為長度 17 的 torch.Tensor)。
        _result = {
            'S11': tensor(S11_vals.tolist()),
            'S21': tensor(S21_vals.tolist()),
            'S22': tensor(S22_vals.tolist())
        }

        return _result