
from . import *

def get_penalty(expected_len:int=17):
    logger.warning("模擬發生未知錯誤，回傳懲罰值")
    return {'S11': tensor([0.0] * expected_len), 'Gain': tensor([-40.0] * expected_len)}

class SinglePortSimulator(PatchSimulator):
    def __init__(self, record_path, HFSS_sab_path = Path(__file__).parent.joinpath('sab', 'single_port.sab'), pixel_count:int = 25):
        super().__init__(record_path, HFSS_sab_path, pixel_count)

    def __call__(self, pixel_matrix:Tensor, only_create_project:bool = False):
        super().__call__(pixel_matrix)
        pixel_row = self.pixel_count
        pixel_column = self.pixel_count
        one_num = 0

        pixel_matrix = pixel_matrix.reshape(pixel_row, pixel_column).cpu()

        oDesign = self.oDesign
        oEditor = oDesign.SetActiveEditor("3D Modeler")

        if pixel_row*pixel_column == 400:
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
                                "Value:=", "0.25mm"
                            ],
                            [
                                "NAME:pixel_W",
                                "PropType:=", "VariableProp",
                                "UserDef:=", True,
                                "Value:=", "0.25mm"
                            ]
                        ]
                    ]
                ])

        if pixel_row*pixel_column == 2500:
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
        patch_names = [] # 建立一個 List 來收集所有生成的 Patch 名稱
        patch_count = 1 # 迴圈生成 Box
        for y in range(0, pixel_row, 1):
            for x in range(0, pixel_column, 1):
                # if pixel_matrix[x][y] > 0:
                #     one_num = one_num + 1
                if pixel_matrix[x][y] == 1:
                    current_name = f"Patch_{patch_count}"
                    one_num = one_num + 1
                    actual_patch_name = oEditor.CreateBox(
                        [
                            "NAME:BoxParameters",
                            "XPosition:=", "0mm" + str("+pixel_H" * x),
                            "YPosition:=", "0mm" + str("+pixel_W" * y),
                            "ZPosition:=", "0.508mm",
                            "XSize:=", "pixel_H + 0.01mm", #! 避免產生奇異點
                            "YSize:=", "pixel_W + 0.01mm", #!
                            "ZSize:=", "CooperH"
                        ],
                        [
                            "NAME:Attributes",
                            "Name:=", current_name, #! 獨立命名
                            "Flags:=", "",
                            "Color:=", "(255 0 0)",
                            "Transparency:=", 0,
                            "PartCoordinateSystem:=", "Global",
                            "UDMId:=", "",
                            "MaterialValue:=", "\"copper\"",
                            "SurfaceMaterialValue:=", "\"\"",
                            "SolveInside:=", True, #! 導體內部求解 (Original: True)
                            "IsMaterialEditable:=", True,
                            "UseMaterialAppearance:=", False,
                            "IsLightweight:=", False
                        ])
                    patch_names.append(actual_patch_name)
                    patch_count += 1

        # Batch Global Unite
        if len(patch_names) >= 1:
            chunk_size = 20
            # try:
            for i in range(0, len(patch_names), chunk_size):
                chunk_patches = patch_names[i:i+chunk_size]
                selections_str = "feed_line," + ",".join(chunk_patches)
                
                oEditor.Unite(
                    ["NAME:Selections", "Selections:=", selections_str],
                    ["NAME:UniteParameters", "KeepOriginals:=", False]
                )
            # except:
            #     return get_penalty()

        # 設定邊界條件
        oModule = oDesign.GetModule("BoundarySetup")
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
                            "Start:=", ["27.5mm", "2.5mm",
                                        "9.99200722162641e-17mm"],
                            "End:=", ["27.5mm", "2.5mm", "0.508mm"]
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

        oModule = oDesign.GetModule("ModelSetup")
        oModule.CreateOpenRegion(
            [
                "NAME:Settings",
                "OpFreq:=", "28GHz",
                "Boundary:=", "Radiation",
                "ApplyInfiniteGP:=", False
            ])

        # 模擬設定
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
        oModule.InsertFrequencySweep("Setup1",
            [
                "NAME:Sweep",
                "IsEnabled:=", True,
                "RangeType:=", "LinearStep",
                "RangeStart:=", "24GHz",
                "RangeEnd:=", "32GHz",
                "RangeStep:=", "0.5GHz",
                "Type:=", "Interpolating",       #! 掃頻演算法選擇 {EX: Fast, Interpolating(插值掃描)}
                "SaveFields:=", True,
                "SaveRadFields:=", False,
                "GenerateFieldsForAllFreqs:=", False,
                "ExtrapToDC:=", False
            ])

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

        if only_create_project:
            # self.oProject.Save()
            return
        

        # 開始模擬
        oDesign.AnalyzeAll()


        # 畫出結果
        oModule = oDesign.GetModule("ReportSetup")

        # Create S11
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
        oModule.CreateReport("Realized Gain Plot 1", "Far Fields", "Rectangular Plot", "Setup1 : Sweep",
            [
                "Context:=", "3D"
            ],
            [
                "Freq:=", ["All"],
                "Phi:=", ["0deg"],
                "Theta:=", ["0deg"]
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

        # Export csv
        oModule.ExportToFile(
            "S Parameter Plot 1", 
            self.path_result.joinpath(f"NN_patch_Sparameter_{self.num}_S11.csv"), 
            False
        )
        oModule.ExportToFile(
            "Realized Gain Plot 1", 
            self.path_result.joinpath(f"NN_patch_Gain_{self.num}_S11.csv"), 
            False
        )
        # oModule.ExportToFile("Realized Gain Plot 2", args.record_path+'/csv/NN_patch_Beamwidth_"+ str(Design_index) +"_Realized Gain Plot 2.csv", False)
        
        
        # Read csv
        Sparameter_dataframe = read_csv(
            self.path_result.joinpath(f"NN_patch_Sparameter_{self.num}_S11.csv")
        )
        Gain_dataframe = read_csv(
            self.path_result.joinpath(f"NN_patch_Gain_{self.num}_S11.csv")
        )

        # 預期頻率點，這裡定義後，後續皆以 len(freqs_expected) 為準
        freqs_expected = np.linspace(24, 32, 17)
        expected_len = len(freqs_expected)

        #* S11
        freqs_s11 = Sparameter_dataframe.iloc[:, 0].values
        S11_vals = Sparameter_dataframe.iloc[:, 1].values
        if len(S11_vals) != expected_len:
            # logger.warning(f"HFSS S11 模擬點數異常 (Pattern {self.num})！預期 {expected_len} 點，實際取得 {len(S11_vals)} 點，將自動進行插值補齊。")
            S11_vals:np.ndarray = np.interp(freqs_expected, freqs_s11, S11_vals)

        #* Gain 
        freqs_gain = Gain_dataframe.iloc[:, 2].values
        Gain_vals = Gain_dataframe.iloc[:, 3].values
        if len(Gain_vals) != expected_len:
            # logger.warning(f"HFSS Gain 模擬點數異常 (Pattern {self.num})！預期 {expected_len} 點，實際取得 {len(Gain_vals)} 點，將自動進行插值補齊。")

            Gain_vals:np.ndarray = np.interp(freqs_expected, freqs_gain, Gain_vals)

        _result = {
            'S11': tensor(S11_vals.tolist()),
            'Gain': tensor(Gain_vals.tolist()),
        }

        full_output = []
        full_parameter = np.append(np.append(full_output, S11_vals), Gain_vals)

        return _result