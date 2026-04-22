"""
微帶貼片天線 HFSS 模擬器基底。

提供 :class:`PatchSimulator` 抽象基底類別，封裝 HFSS 專案生命週期
（open / start / end / quit / reopen）與檔案管理邏輯。
實際的單埠 / 雙埠幾何由 ``single_port.py`` / ``dual_port.py`` 定義。
"""

from abc import ABC, abstractmethod
from time import time

import numpy as np
from loguru import logger
from pandas import read_csv
from pywintypes import com_error  # type: ignore
from torch import Tensor, all, logical_or, tensor
from win32com.client import DispatchEx as _dispatch  # 需要 pip install pywin32

from script.kill import kill as _kill

from ...utils import Path, config

__all__ = [
    "PatchSimulator",
    "com_error",
    # 以下重新匯出，供 single_port.py / dual_port.py 以 `from . import *` 使用
    "Path",
    "Tensor",
    "all",
    "config",
    "logical_or",
    "np",
    "read_csv",
    "tensor",
]


class PatchSimulator(ABC):
    """貼片天線 HFSS 模擬器基底類別。

    子類別須實作 ``__call__(pixel_matrix)``，負責依 ``.sab`` 模型建立
    專案並回傳模擬結果。
    """

    def __init__(self, record_path: str, HFSS_sab_path: str, pixel_count: int):
        self.path_record = Path(record_path).joinpath("HFSS").not_exist_create()
        self.HFSS_sab_path = str(HFSS_sab_path)
        self.pixel_count = pixel_count

        self.path_result = self.path_record.joinpath("result").not_exist_create()
        self.path_project = self.path_record.joinpath("project").not_exist_create()

        self.name_project = f"project_{config.ID}_" + "{num}"
        self.name_design = "patch_design_{num}"

    def open(self):
        """啟動 HFSS 並取得 Desktop 物件。

        若 HFSS 未安裝或 COM 介面無法建立，``DispatchEx`` 會丟出
        ``pywintypes.com_error``；此處加上明確的錯誤訊息以利除錯。
        """
        try:
            oAnsoftApp = _dispatch("AnsoftHFSS.HfssScriptInterface")
        except com_error as err:
            logger.error(
                f"無法建立 HFSS COM 介面，請確認已安裝 Ansys HFSS 並註冊 AnsoftHFSS.HfssScriptInterface: {err}"
            )
            raise
        self.oDesktop = oAnsoftApp.GetAppDesktop()  # HFSS 軟體主程式的總管
        self.oDesktop.RestoreWindow()  # 如果 HFSS 被最小化，讓視窗恢復顯示

    def quit(self):
        """關閉整個 HFSS 軟體（不等待結束）。"""
        self.oDesktop.QuitApplication()

    def save(self, name: str = None):
        if name:
            # 另存新檔：SaveAs(filename, overwrite)
            self.oProject.SaveAs(str(self.path_project.joinpath(f"{name}.aedt")), True)
        else:
            # 儲存專案到目前的儲存位置
            self.oProject.Save()

    def reopen(self, project_keep_latest: int = 5):
        self.kill()
        self.clean(project_keep_latest)
        self.open()

    def clean(self, project_keep_latest: int = 5):
        return self.path_project.manage_file_count("*", keep_latest=project_keep_latest)

    def kill(self):
        _kill("ansysedt.exe")

    def start(self, num: int):
        """建立新專案並插入新設計。

        :param num: pattern 編號
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
                    # 嘗試取得名稱，若該物件已失效 (Zombie) 會在此處報錯
                    existing_project_names.append(proj.GetName())
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

        self.oProject = self.oDesktop.NewProject(project_name)
        self.save(project_name)

        # 設定目前作用中的專案
        self.oProject = self.oDesktop.SetActiveProject(project_name)

        # 插入新設計：InsertDesign(type, name, solutionType, setupType)
        self.oProject.InsertDesign("HFSS", self.name_design.format(num=num), "DrivenModal", "")

        # 設定目前作用中的設計
        self.oDesign = self.oProject.SetActiveDesign(self.name_design.format(num=num))

        return self.oDesign

    def end(self) -> int:
        """刪除 Design 並關閉專案。

        :return: 此次模擬耗時（秒）
        """
        assert getattr(self, "num", None) is not None, "Please use `start()` first"
        self.save(self.name_project.format(num=self.num))
        self.oProject.DeleteDesign(self.name_design.format(num=self.num))
        self.oDesktop.CloseProject(self.name_project.format(num=self.num))
        self.num = None

        return int(time() - self.start_time)

    @abstractmethod
    def __call__(self, pattern: Tensor, *args, **kwds):
        """
        子類別實作前先呼叫 ``super().__call__(pattern)`` 驗證輸入。

        範例::

            def __call__(self, pixel_matrix: Tensor):
                super().__call__(pixel_matrix)
                ...
        """
        is_binary = all(logical_or(pattern == 0, pattern == 1))
        assert getattr(self, "num", None) is not None, "Please use `start()` first"
        assert is_binary, "The input must be binary"

    def __str__(self):
        return f"{self.__class__.__name__}(HFSS_sab_path={self.HFSS_sab_path})"
