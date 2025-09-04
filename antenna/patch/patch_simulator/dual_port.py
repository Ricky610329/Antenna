
from . import *


class DualPortSimulator(PatchSimulator):
    def __init__(self, record_path, HFSS_sab_path = Path(__file__).parent.joinpath('sab', 'dual_port.sab'), pixel_count:int = 25):
        super().__init__(record_path, HFSS_sab_path, pixel_count)

    def __call__(self, pixel_matrix:Tensor):
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
        for y in range(0, pixel_row, 1):
            for x in range(0, pixel_column, 1):
                # if pixel_matrix[x][y] > 0:
                #     one_num = one_num + 1
                if pixel_matrix[x][y] == 1:
                    one_num = one_num + 1
                    oEditor.CreateBox(
                        [
                            "NAME:BoxParameters",
                            "XPosition:=", "0mm" + str("+pixel_H" * x),
                            "YPosition:=", "0mm" + str("+pixel_W" * y),
                            "ZPosition:=", "0.508mm",
                            "XSize:=", "pixel_H",
                            "YSize:=", "pixel_W",
                            "ZSize:=", "CooperH"
                        ],
                        [
                            "NAME:Attributes",
                            "Name:=", "Patch",
                            "Flags:=", "",
                            "Color:=", "(255 0 0)",
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

        ones_buf = 0
        for i in range(pixel_row):
            patch_unite = ""
            E = pixel_matrix[:, i]
            # 使用 numpy.where 找到值為 1 的位置
            ones_indices = np.where(E == 1)[0]
            # 因為只有一個不能unite
            if ones_indices.shape[0] > 1:
                for u in range(ones_indices.shape[0]):
                    if u+ones_buf == 0:
                        continue
                    patch_unite = patch_unite + "Patch_" + str(u+ones_buf) + ","

                patch_unite = patch_unite[:len(patch_unite)-1]

                # 因為只有一個不能unite
                if patch_unite == "Patch_1":
                    ones_buf = ones_buf + ones_indices.shape[0]
                    continue

                oEditor.Unite(
                    [
                        "NAME:Selections",
                        "Selections:=", patch_unite
                    ],
                    [
                        "NAME:UniteParameters",
                        "KeepOriginals:=", False
                    ])
            ones_buf = ones_buf + ones_indices.shape[0]

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
        
        oModule = oDesign.GetModule("ModelSetup")
        oModule.CreateOpenRegion(
            [
                "NAME:Settings",
                "OpFreq:=", "28GHz",
                "Boundary:=", "Radiation",
                "ApplyInfiniteGP:=", False
            ])

        ###* 模擬設定 ###
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
                "Type:=", "Fast",
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

        # self.oProject.Save()

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
        oModule.ExportToFile("S Parameter Plot 1", self.path_result.joinpath(f"NN_patch_Sparameter_{self.num}_S11.csv"), False)
        oModule.ExportToFile("S Parameter Plot 2", self.path_result.joinpath(f"NN_patch_Sparameter_{self.num}_S21.csv"), False)
        oModule.ExportToFile("S Parameter Plot 3", self.path_result.joinpath(f"NN_patch_Sparameter_{self.num}_S22.csv"), False)


        #* Read csv
        Sparameter_dataframe_11 = read_csv(self.path_result.joinpath(f"NN_patch_Sparameter_{self.num}_S11.csv"))
        Sparameter_dataframe_21 = read_csv(self.path_result.joinpath(f"NN_patch_Sparameter_{self.num}_S21.csv"))
        Sparameter_dataframe_22 = read_csv(self.path_result.joinpath(f"NN_patch_Sparameter_{self.num}_S22.csv"))
        
        #  將數值取出 之後要算loss
        S11 = Sparameter_dataframe_11.iloc[0:17, 1]
        S21 = Sparameter_dataframe_21.iloc[0:17, 1]
        S22 = Sparameter_dataframe_22.iloc[0:17, 1]
        
        full_output = []

        # TODO 
        _result = {
            'S11': tensor(Sparameter_dataframe_11.iloc[:, 1].to_list()),
            'S21': tensor(Sparameter_dataframe_21.iloc[:, 1].to_list()),
            'S22': tensor(Sparameter_dataframe_22.iloc[:, 1].to_list())
        }


        full_parameter = np.append(np.append(np.append(full_output, S11.to_numpy()), S21.to_numpy()),S22.to_numpy())
        # breakpoint()
        return _result