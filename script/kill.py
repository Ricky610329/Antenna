"""
script/kill.py — HFSS 行程強制終結工具

【模組角色】
在天線反向設計的閉迴路訓練中，PatchSimulator 透過 COM 介面驅動
Ansys HFSS (ansysedt.exe) 進行電磁場模擬。當 HFSS 卡死或當機時，
PatchSimulator 的容錯機制會呼叫本模組的 kill() 函式，強制終結殘留
的 HFSS 行程，之後再重新建立 COM 連線，確保上千 epoch 的訓練不因
單次模擬器崩潰而全盤中斷。

【使用方式】
- 程式內嵌呼叫：from script.kill import kill; kill()
- 命令列呼叫：python script/kill.py [--name <行程名稱>]
"""

from argparse import  ArgumentParser
from  psutil import (
    process_iter,      #? 列舉系統上所有執行中的行程
    NoSuchProcess,     #? 行程在操作途中已自行消失時拋出
    AccessDenied,      #? 沒有足夠權限終結目標行程時拋出
    ZombieProcess      #? 目標行程已成殭屍行程(僅剩 PID 條目)時拋出
)
from loguru import logger

def kill(process_name='ansysedt.exe'):
    """強制終結與指定名稱相符的所有系統行程。

    實作策略
    --------
    - 使用 psutil.process_iter() 遍歷當前系統的全部行程快照；
      每個行程只快取 ['pid', 'name'] 兩個欄位，降低開銷。
    - 以「不分大小寫的子字串比對」判斷行程名稱是否符合目標，
      例如 'ansysedt.exe' 可同時匹配 'ANSYSEDT.EXE'（Windows 慣例）。
    - 對每個符合的行程呼叫 proc.kill()，在 POSIX 系統上等同
      SIGKILL（不可忽略的強制終結），在 Windows 上呼叫
      TerminateProcess() — 均屬作業系統層級的強制終結，
      不走正常的「優雅關閉」路徑，確保卡死的 COM server 能被清除。
    - 本函式不遞迴追殺子行程；若 HFSS 衍生的子行程需一併清除，
      須在上層呼叫端自行處理或改用 proc.children(recursive=True)。

    容錯處理
    --------
    - NoSuchProcess  : 行程在 iter 到 kill 之間已自行結束，靜默略過。
    - AccessDenied   : 無權限終結(例如以較低權限執行本腳本)，靜默略過。
    - ZombieProcess  : 已成殭屍的行程無法 kill，靜默略過。
    上述例外以 pass 吞掉，避免單一行程的異常中斷後續行程的清理。

    Parameters
    ----------
    process_name : str, optional
        目標行程名稱(或其子字串)，預設為 'ansysedt.exe'(Ansys HFSS 主程式)。
        比對時不區分大小寫。
    """
    for proc in process_iter(['pid', 'name']):
        if process_name.lower() in proc.info['name'].lower():
            try:
                proc.kill()  # 結束進程
                logger.warning(f"Process {process_name} terminated.")
            except (NoSuchProcess, AccessDenied, ZombieProcess):
                pass   #! 行程已消失、無權限或殭屍行程，略過不中斷流程


if __name__ == "__main__":
    ### 命令列入口：允許在訓練腳本外部單獨呼叫，方便手動清除殘留的 HFSS 行程
    parser = ArgumentParser(
        description = "用來結束HFSS的工作階段"
    )

    parser.add_argument(
        "--name",
        type=str,
        default = r"ansysedt.exe",   #* 預設目標為 Ansys HFSS 主執行檔
        help = "This is process name."
    )

    args =parser.parse_args()
    kill(args.name)   #* 執行強制終結，結果透過 loguru logger 輸出警告訊息
