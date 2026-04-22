from  win32com.client import DispatchEx as  _dispatch, gencache #? pip install pywin32
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

    def save(self, name:str = None, rootpath=None):
        path = Path(rootpath) if rootpath else self.path_project
        if name:
            #* 另存新檔
            #? SaveAs(filename, overwrite)
            self.oProject.SaveAs(
                str(path.joinpath(f"{name}.aedt")), True
            )

        else:
            #* 儲存專案（到目前的儲存位置）
            self.oProject.Save()

        return name if name else self.name_project.format(num=self.num)

    # def recreateProject(self, name):
    #     self.oDesktop.CloseProject(name)    # 關閉指定的專案
    #     # self.oProject = self.oDesktop.NewProject("Design_Patch_Antenna")
    
    def reopen(self, project_keep_latest:int = 5):
        self.kill() # self.quit()
        if project_keep_latest: self.clean(project_keep_latest)
        # sleep(7)
        self.open()

    def restart(self, kill:bool=False):
        num = self.num
        self.end(save_project=False)
        if kill: self.reopen(None)
        self.start(num)

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

        existing_project_names = []
        try:
            # 取得所有專案物件
            projects = self.oDesktop.GetProjects()
            for proj in projects:
                try:
                    # 嘗試取得名稱，若該物件已失效(Zombie)，這裡會報錯但被捕獲
                    name = proj.GetName()
                    existing_project_names.append(name)
                except Exception:
                    # 忽略無法讀取名稱的壞掉物件，不中斷程式
                    continue
        except Exception as e:
            # 若連 GetProjects 都失敗，代表 HFSS 可能徹底卡死
            logger.error(f"Failed to retrieve project list: {e}")
        
        # 檢查是否需要關閉舊專案
        if project_name in existing_project_names:
            logger.warning(f"Closing {project_name}...")
            try:
                self.oDesktop.CloseProject(project_name)
            except Exception as e:
                logger.warning(f"Failed to close {project_name}, it might be already closed. Error: {e}")


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
    
    def end(self, name=None, save_project:bool = True) -> int:
        """
        Delete Design and close project.
        """
        assert getattr(self, 'num', None) != None, "Please use `start()` first"

        # 1. 嘗試儲存專案
        if save_project:
            try:
                self.save(self.name_project.format(num=self.num))
            except Exception as e:
                logger.warning(f"專案儲存失敗，略過: {e}")

        # 2. 嘗試刪除設計 (加上強制獵殺防護)
        try:
            self.oProject.DeleteDesign(self.name_design.format(num=self.num))
        except Exception as e:
            
            
            sleep(2) # 等待 2 秒讓系統釋放檔案鎖定
            try:
                self.oProject.DeleteDesign(self.name_design.format(num=self.num))
            except:
                # 直接呼叫我們之前寫好的重生機制 (它會 kill, sleep, 並重新連線 COM)
                self.reopen()
                
                # 重生完畢後，直接結束這回合，不用再執行後面的 CloseProject 了
                self.num = None
                return int(time()-self.start_time)

        # 3. 嘗試關閉專案
        try:
            self.oDesktop.CloseProject(name if name else self.name_project.format(num=self.num))
        except Exception as e:
            logger.error(f"專案關閉異常，HFSS 可能徹底當機，準備強制重啟核心: {e}")
            # 如果連專案都關不掉，代表 ansysedt.exe 已經壞了，啟動您的 kill() 大絕招
            self.reopen() 

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