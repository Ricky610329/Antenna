from  win32com.client import Dispatch as  _dispatch #? pip install pywin32
from pywintypes import com_error # type: ignore
from script.kill import kill as _kill
import numpy as np
from pandas import read_csv
from abc import ABC, abstractmethod
from ...utils import Path, config
from time import sleep, time
from loguru import logger
from torch import tensor, Tensor, all, logical_or

class PatchSimulator(ABC):
    def __init__(self, record_path:str, HFSS_sab_path:str, pixel_count:int):
        self.path_record = Path(record_path).joinpath("HFSS").not_exist_create()
        self.HFSS_sab_path = str(HFSS_sab_path)
        self.pixel_count = pixel_count

        self.path_result = self.path_record.joinpath('result').not_exist_create()
        self.path_project = self.path_record.joinpath('project').not_exist_create()

        self.name_project = f"project_{config.ID}_" + "{num}"   #? self.name_project.format(num=num)
        self.name_design = "patch_design_{num}"

        
    def open(self):
        oAnsoftApp = _dispatch('AnsoftHFSS.HfssScriptInterface')
        self.oDesktop = oAnsoftApp.GetAppDesktop() # HFSS 軟體主程式的總管
        self.oDesktop.RestoreWindow()   # 如果 HFSS 被最小化，讓視窗恢復顯示
        # self.oProject = self.oDesktop.NewProject("Design_Patch_Antenna") # 建立一個新專案（回傳 oProject 物件）

    def quit(self):
        """不會等待"""
        self.oDesktop.QuitApplication() # 關閉整個 HFSS 軟體

    def save(self, name:str = None):
        # path = path.format(count=str(self.count))
        if name:
            #* 另存新檔
            #? SaveAs(filename, overwrite)
            self.oProject.SaveAs(
                str(self.path_project.joinpath(f"{name}.aedt")), True
            )
        else:
            #* 儲存專案（到目前的儲存位置）
            self.oProject.Save()
    
    # def recreateProject(self, name):
    #     self.oDesktop.CloseProject(name)    # 關閉指定的專案
    #     # self.oProject = self.oDesktop.NewProject("Design_Patch_Antenna")
    
    def reopen(self, project_keep_latest:int = 5):
        self.kill() # self.quit()
        self.clean(project_keep_latest)
        # sleep(7)
        self.open()

    def clean(self, project_keep_latest:int = 5):
        return self.path_project.manage_file_count("*", keep_latest=project_keep_latest)

    def kill(self):
        _kill("ansysedt.exe")

    def start(self, num:int):
        """
        Create a new project and save it, then insert the new design

        :param num: pattern number
        """
        assert hasattr(self, "oDesktop"), "Please use `open()` or `reopen()` first"
        self.start_time = time()
        self.num = num

        project_name = self.name_project.format(num=num)
        if project_name in self.oDesktop.GetProjects():
            self.oDesktop.CloseProject(project_name)

        self.oProject = self.oDesktop.NewProject(project_name) # 建立一個新專案（回傳 oProject 物件）

        self.save(project_name)

        #* 設定目前作用中的專案
        #? SetActiveProject(name)
        self.oProject = self.oDesktop.SetActiveProject(
            project_name
        )

        ###* 插入新設計 ###
        #? InsertDesign(type, name, solutionType, setupType)
        self.oProject.InsertDesign( 
            "HFSS", self.name_design.format(num=num), "DrivenModal", ""
        )

        #* 設定目前作用中的設計
        #? SetActiveDesign(name)
        self.oDesign = self.oProject.SetActiveDesign(
            self.name_design.format(num=num)
        )
        
        return self.oDesign
    
    def end(self) -> int:
        """
        Delete Design and close project.

        :return: Execution time
        """
        assert getattr(self, 'num', None) != None, "Please use `start()` first"
        self.save(
            self.name_project.format(num=self.num)
        )
        self.oProject.DeleteDesign(
            self.name_design.format(num=self.num)
        )
        self.oDesktop.CloseProject(
            self.name_project.format(num=self.num)
        )
        self.num = None

        return int(time()-self.start_time)


    @abstractmethod
    def __call__(self, pattern:Tensor, *args, **kwds):
        """
        Custom simulation will first check num and is_binary.

        # Example
        ```
        def __call__(self, pixel_matrix:Tensor):
            super().__call__(pixel_matrix)
        ```
        """
        is_binary = all(logical_or(
            pattern == 0, pattern == 1
        ))
        assert getattr(self, 'num', None) != None, "Please use `start()` first"
        assert is_binary, "The input must be binary"

    def __str__(self):
        return f"{self.__class__.__name__}(HFSS_sab_path={self.HFSS_sab_path})"