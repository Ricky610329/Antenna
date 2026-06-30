###* ────────────────────────────────────────────────────────────────────
###* PatchSimulator：天線「反向設計」閉迴路中的 SIM（Ground Truth 模擬器）抽象基類
###*
###* 在整套 pipeline 裡，這個模組扮演「真實電磁模擬」這一端：
###*   GEN（生成器）：目標響應 → pattern（像素矩陣，0/1 表示某格是否鋪銅）
###*   SM （代理模型）：pattern → 預測響應（快、可微，但只是近似）
###*   SIM（本檔）   ：pattern → 真實響應（慢、不可微，但是 ground truth）
###*
###* 本檔只負責「HFSS COM 生命週期管理」這個共通骨架，不含任何實際建模指令；
###* 真正把像素矩陣畫成 3D 幾何、設邊界、跑求解、匯出 S 參數/增益的程式碼，
###* 由 single_port.py / dual_port.py 兩個子類在 __call__ 內各自實作。
###*
###* 為什麼需要這麼多「容錯／重生」機制（reopen/restart/kill/重試）？
###*   HFSS 是透過 Windows COM（win32com）以行程外（out-of-process）方式驅動的，
###*   底層其實是去操控 ansysedt.exe 這支龐大的商用軟體。長時間批次模擬時，
###*   COM 介面非常容易出現：物件失效（Zombie）、檔案鎖定未釋放、SetActiveDesign
###*   呼叫卡死、甚至整支 ansysedt.exe 當掉而不回應。一旦卡死，單純的例外處理
###*   救不回來，唯一可靠的辦法就是「強制殺掉行程 → 重新連線 COM → 繼續下一回合」。
###*   因此本類別大量使用 try/重試/sleep/kill/reopen，把單一回合的失敗隔離掉，
###*   讓整個 training loop（動輒上千個 epoch）不會因為某次模擬卡死而整批中斷。
###* ────────────────────────────────────────────────────────────────────

from  win32com.client import DispatchEx as  _dispatch, gencache #? pip install pywin32
from pywintypes import com_error # type: ignore
from script.kill import kill as _kill   #? 強制殺掉 ansysedt.exe 行程（HFSS 卡死時的最後手段）
import numpy as np
from pandas import read_csv
from abc import ABC, abstractmethod
from ...utils import Path, config   #? Path：自製路徑工具（含 not_exist_create / manage_file_count）；config.ID：本次執行的唯一識別碼
from time import sleep, time
from loguru import logger
from torch import tensor, Tensor, all, logical_or   #? all/logical_or：用來檢查輸入 pattern 是否為純二元 0/1

class PatchSimulator(ABC):
    """
    HFSS（Ansys 電磁模擬器）的抽象基底類別，負責管理「COM 連線的整段生命週期」。

    在 pipeline 中的定位：這是反向設計閉迴路裡的 SIM（ground truth）。子類別
    （SinglePortSimulator / DualPortSimulator）只需專注於「如何把 pattern 畫成天線、
    跑模擬、回傳響應」，而所有跟 HFSS 連線、開關專案、容錯重生有關的瑣事都集中在這裡。

    典型使用流程（與 train_single.py / train_dual.py 對應）：
        sim = SinglePortSimulator(record_path=...)   # 設定路徑與像素數
        sim.open()                                   # 整個訓練開始前，連線一次 HFSS
        for epoch in ...:
            sim.start(epoch)                         # 本回合：建立並切換到新專案/設計
            result = sim(pattern)                    # __call__：實際建模 + 模擬，回傳 S 參數/增益
            sim.end()                                # 儲存/刪除設計/關專案（含失敗重試與重生）
            sim.clean()                              # 只保留最近數個專案檔，避免硬碟塞爆
        sim.quit()

    設計重點：open 與 quit 一輩子只各做一次（COM 連線昂貴），而 start/end 每個 epoch
    做一次（一個 pattern 一個獨立專案，互不汙染）。當任何環節偵測到 HFSS 卡死，
    就由 reopen()/restart()/kill() 把行程砍掉重練，讓 loop 能自動續跑。
    """
    def __init__(self, record_path:str, HFSS_sab_path:str, pixel_count:int):
        #* record_path：本次訓練的輸出根目錄；底下再開一層 HFSS/ 專門放模擬產物。
        #* not_exist_create()：路徑不存在就自動建立並回傳自己（鏈式呼叫用）。
        self.path_record = Path(record_path).joinpath("HFSS").not_exist_create()
        #* .sab 是 HFSS 的幾何底板檔（含底材 Sub、饋線 feed_line、GND 與埠用矩形 Rectangle）；
        #* 子類別 __call__ 會 Import 這個檔當基礎，再把 pattern 的金屬像素疊上去。轉 str 保險用。
        self.HFSS_sab_path = str(HFSS_sab_path)
        self.pixel_count = pixel_count   #? 像素邊長：pattern 會 reshape 成 (pixel_count, pixel_count) 的方陣

        #* 兩個固定子目錄：result/ 放匯出的 S 參數與增益 csv；project/ 放每回合的 .aedt 專案檔。
        self.path_result = self.path_record.joinpath('result').not_exist_create()
        self.path_project = self.path_record.joinpath('project').not_exist_create()

        #* 專案/設計命名模板。專案名嵌入 config.ID（本次執行的唯一碼）以避免跨次執行撞名；
        #* {num} 會在 start(num) 時用 .format(num=num) 填入 pattern 編號（通常就是 epoch）。
        self.name_project = f"project_{config.ID}_" + "{num}"   #? self.name_project.format(num=num)
        self.name_design = "patch_design_{num}"


    def open(self, attempts: int = 6, wait: float = 8.0):
        #* 建立／取得與 HFSS 的 COM 連線。整個訓練原則上只在開訓呼叫一次（連線成本高），
        #* 之後僅 reopen()（卡死重生）會再呼叫，重新抓一條乾淨的連線。
        #! 對「RPC server 未就緒/不可用」有韌性 (根因)：剛 kill 掉舊 ansysedt 後，新 ansysedt 的
        #! COM/RPC server 要數秒才起得來，單發 GetAppDesktop 會撞 com_error(-2147023174『RPC 伺服器
        #! 無法使用』) → 過去這會逃到 excepthook 帶走整個 run。故失敗就 kill 殘行程 → 等 wait 秒 →
        #! 重試，最多 attempts 次；真的連不上(~attempts×wait 秒都起不來)才往外拋、交給上層容錯。
        last = None
        for i in range(attempts):
            try:
                oAnsoftApp = _dispatch('AnsoftHFSS.HfssScriptInterface')   #? DispatchEx：每次啟動獨立的 ansysedt.exe 行程
                self.oDesktop = oAnsoftApp.GetAppDesktop()                 # HFSS 軟體主程式的總管
                self.oDesktop.RestoreWindow()                             # 視窗若被最小化則恢復
                return
            except Exception as e:
                last = e
                logger.warning(f"open() 連 HFSS 第 {i + 1}/{attempts} 次失敗（多半新 ansysedt 的 RPC "
                               f"server 還沒起來）：{type(e).__name__}: {e}；kill 殘行程 + 等 {wait}s 重試")
                self.kill()
                sleep(wait)
        logger.error(f"open() 連 HFSS {attempts} 次都失敗，放棄（交給上層容錯/excepthook）")
        raise last

    def quit(self):
        """不會等待"""
        #* 「優雅地」要求 HFSS 自行關閉（非同步、不等待它真的關完）。
        #* 注意：這跟 kill() 不同——quit 走 COM 正常流程，HFSS 沒卡死時用；
        #* 一旦行程已當掉、QuitApplication 也叫不動，就得改用 kill() 直接砍行程。
        self.oDesktop.QuitApplication() # 關閉整個 HFSS 軟體

    def save(self, name:str = None, rootpath=None):
        #* 統一的存檔入口：有給 name 就「另存新檔」到 project/ 目錄，沒給就「原地儲存」。
        #* start() 開新專案後會用它把專案落地成 .aedt；end() 收尾前也會再存一次。
        path = Path(rootpath) if rootpath else self.path_project
        if name:
            #* 另存新檔
            #? SaveAs(filename, overwrite)
            self.oProject.SaveAs(
                str(path.joinpath(f"{name}.aedt")), True   #? 第二個參數 True = 允許覆蓋同名檔
            )

        else:
            #* 儲存專案（到目前的儲存位置）
            self.oProject.Save()

        #* 回傳實際使用的檔名（沒指定 name 時回推本回合的專案名），方便呼叫端記錄。
        return name if name else self.name_project.format(num=self.num)

    # def recreateProject(self, name):
    #     self.oDesktop.CloseProject(name)    # 關閉指定的專案
    #     # self.oProject = self.oDesktop.NewProject("Design_Patch_Antenna")

    def reopen(self, project_keep_latest:int = 5):
        #! 「核心級重生」：當 HFSS 卡死、連 quit/CloseProject 都救不回來時的最後手段。
        #! 流程＝強制砍行程 → 清理舊專案檔 → 重新建立 COM 連線。
        #! 之所以用 kill 而非 quit：行程已當機時 COM 呼叫本身就會卡住，只能從 OS 層硬砍。
        self.kill() # self.quit()
        #* 重生時順手清掉過多的舊 .aedt，避免反覆當機累積一堆殘檔把硬碟撐爆。
        if project_keep_latest: self.clean(project_keep_latest)
        # sleep(7)
        self.open()   #? 重新連線：拿到一條全新、乾淨的 oDesktop

    def restart(self, kill:bool=False):
        #* 「回合級重來」：保留目前的 num，把這一回合的專案收掉後，用同一個 num 重新 start。
        #* 用於該回合建模/模擬出狀況、想原地重跑時。kill=True 則連 HFSS 行程一起重生（更徹底）。
        num = self.num
        self.end(save_project=False)   #? save_project=False：重來前不需要保存壞掉的中途結果
        if kill: self.reopen(None)     #? 傳 None 給 reopen → 跳過 clean，純粹重啟連線
        self.start(num)                #? 用同一個 num 重新開專案，對外像是沒發生過

    def clean(self, project_keep_latest:int = 5):
        #* 只保留 project/ 目錄中最新的數個專案檔，其餘按建立時間由舊到新刪除。
        #* 每個 epoch 都會產生一個 .aedt，不清理的話長時間訓練會塞滿磁碟。
        return self.path_project.manage_file_count("*", keep_latest=project_keep_latest)

    def kill(self):
        #! 直接從作業系統層級強制終止所有 ansysedt.exe 行程（不經 COM）。
        #! 這是整套容錯機制的地基：唯有先確定舊行程徹底死透，重新 open() 才能拿到乾淨連線。
        _kill("ansysedt.exe")

    def start(self, num:int):
        """
        Create a new project and save it, then insert the new design

        :param num: pattern number

        每個 epoch 開頭呼叫一次：為這一回合（編號 num，通常等於 epoch）建立一個全新、
        乾淨的專案與設計，並設為作用中（active）。一個 pattern 配一個獨立專案，
        是為了讓回合之間的幾何/求解互不汙染，也讓出狀況時可以單獨砍掉重來。
        """
        #* 前置檢查：必須先 open()/reopen() 拿到 oDesktop 才能開專案；否則直接擋下，避免後面整串 COM 呼叫噴錯。
        assert hasattr(self, "oDesktop"), "Please use `open()` or `reopen()` first"
        self.start_time = time()   #? 記錄起始時間，end() 會用它算出本回合耗時
        self.num = num             #? 記住本回合編號，save/end/__call__ 都靠它組檔名與防呆

        project_name = self.name_project.format(num=num)

        #* ── 防呆：開新專案前，先確認同名專案沒有殘留在 HFSS 裡 ──
        #* HFSS 不允許同名專案並存；若上一回合（同 num，如 restart 重來）沒收乾淨，這裡得先關掉。
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

        self.save(project_name)   #? 立刻另存成 .aedt 落地，讓專案有實體檔案（後續 DeleteDesign/CloseProject 才穩）

        #* 設定目前作用中的專案
        #? SetActiveProject(name)
        #* 重新抓一次 active 專案物件回填 self.oProject：用「具名」物件比 NewProject 直接回傳的 handle 更穩定可靠。
        self.oProject = self.oDesktop.SetActiveProject(
            project_name
        )

        ###* 插入新設計 ###
        #? InsertDesign(type, name, solutionType, setupType)
        #* "DrivenModal"：以「模態」方式求解的驅動式問題，適合算 S 參數的天線/微波結構。
        self.oProject.InsertDesign(
            "HFSS", self.name_design.format(num=num), "DrivenModal", ""
        )

        #* 設定目前作用中的設計
        #? SetActiveDesign(name)
        #* 把 oDesign 存起來：子類別 __call__ 全程都在這個 oDesign 上建模、設邊界、跑分析。
        self.oDesign = self.oProject.SetActiveDesign(
            self.name_design.format(num=num)
        )

        return self.oDesign   #? 回傳作用中設計，子類別可直接接手使用
    
    def end(self, name=None, save_project:bool = True) -> int:
        """
        Delete Design and close project.

        每個 epoch 結尾呼叫一次，把本回合的專案/設計收乾淨並回傳本回合耗時（秒）。
        這裡是整套容錯機制的關鍵：收尾的每一步（存檔→刪設計→關專案）都可能因為
        HFSS COM 卡死而失敗，因此採「逐級升高」的策略——能略過就略過、能重試就重試、
        真的救不回來就直接 reopen() 砍掉重練，務必讓 training loop 能繼續往下一回合跑。
        """
        #* 前置檢查：沒先 start() 就沒有 num，無從組檔名，直接擋下。
        assert getattr(self, 'num', None) != None, "Please use `start()` first"

        # 1. 嘗試儲存專案
        #* 存檔失敗影響不大（最多丟掉這回合的 .aedt 存檔），所以只警告、不中斷，繼續往下收尾。
        if save_project:
            try:
                self.save(self.name_project.format(num=self.num))
            except Exception as e:
                logger.warning(f"專案儲存失敗，略過: {e}")

        # 2. 嘗試刪除設計 (加上強制獵殺防護)
        #* 刪設計是回收記憶體/資源的關鍵步；HFSS 偶爾會因檔案鎖定而第一次刪不掉。
        try:
            self.oProject.DeleteDesign(self.name_design.format(num=self.num))
        except Exception as e:


            sleep(2) # 等待 2 秒讓系統釋放檔案鎖定
            #* 第二次重試：很多暫時性鎖定睡一下就會解除，重試即可成功。
            try:
                self.oProject.DeleteDesign(self.name_design.format(num=self.num))
            except:
                #! 重試仍失敗 → 判定 HFSS 已卡死，不再硬碰，直接走核心重生（kill→clean→重連 COM）。
                # 直接呼叫我們之前寫好的重生機制 (它會 kill, sleep, 並重新連線 COM)
                self.reopen()

                #! 重生後舊的 oProject/oDesign 都已失效，後面的 CloseProject 沒意義也會噴錯，
                #! 因此直接提前 return：清掉 num（標記回合結束）並回傳耗時，交回給 loop 進下一回合。
                # 重生完畢後，直接結束這回合，不用再執行後面的 CloseProject 了
                self.num = None
                return int(time()-self.start_time)

        # 3. 嘗試關閉專案
        #* 正常路徑的最後一步：關掉專案釋放資源。name 可指定要關的專案，否則用本回合的專案名。
        try:
            self.oDesktop.CloseProject(name if name else self.name_project.format(num=self.num))
        except Exception as e:
            #! 連專案都關不掉 = ansysedt.exe 大概已徹底當機，升級到最高層級——整支砍掉重生。
            logger.error(f"專案關閉異常，HFSS 可能徹底當機，準備強制重啟核心: {e}")
            # 如果連專案都關不掉，代表 ansysedt.exe 已經壞了，啟動您的 kill() 大絕招
            self.reopen()

        self.num = None   #? 標記本回合結束；下一回合必須再 start() 才會重設
        return int(time()-self.start_time)   #? 回傳本回合總耗時（秒），train 腳本用來記錄/監控模擬速度


    @abstractmethod
    def __call__(self, pattern:Tensor, *args, **kwds):
        """
        Custom simulation will first check num and is_binary.

        # Example
        ```
        def __call__(self, pixel_matrix:Tensor):
            super().__call__(pixel_matrix)
        ```

        抽象方法：實際的「pattern → 真實響應」模擬由子類別實作。
        子類別（SinglePortSimulator / DualPortSimulator）會先用 super().__call__(...)
        跑這裡的共通前置檢查，再接著做各自的事：Import .sab 底板 → 依 pattern 把為 1 的
        像素畫成銅塊（CreateBox）並聯集（Unite）→ 設埠/邊界/求解設定 → AnalyzeAll →
        匯出並讀回 S 參數（單埠額外含增益）→ 整理成 dict 回傳。

        為何在基類就先把這兩個檢查做掉（fail fast）：
          1. num 檢查——確保呼叫前已 start()，oDesign 才存在，避免在 HFSS 端才爆掉、難以收拾。
          2. 二元檢查——pattern 代表「每一格鋪不鋪銅」，物理上只能是 0 或 1。GEN 輸出常是
             連續機率/logits，必須在送進模擬前已被二值化；若混進非 0/1 值，畫出來的幾何
             毫無意義、白白浪費一次昂貴的 HFSS 求解，因此寧可在此直接擋下。
        """
        #* 檢查輸入是否「逐元素皆為 0 或 1」：對每個元素取 (==0 OR ==1)，再要求全部為真。
        is_binary = all(logical_or(
            pattern == 0, pattern == 1
        ))
        assert getattr(self, 'num', None) != None, "Please use `start()` first"   #? 必須先 start() 開好設計
        assert is_binary, "The input must be binary"   #? pattern 只能是 0/1，否則拒絕模擬

    def __str__(self):
        #* 友善的字串表示：印出類別名與使用的 .sab 底板路徑，方便 log 中辨識是單埠還是雙埠模擬器。
        return f"{self.__class__.__name__}(HFSS_sab_path={self.HFSS_sab_path})"