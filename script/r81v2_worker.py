"""R81 spec v2 正式機 worker 短入口：`script\\r81v2.cmd`（三台同一行）。

等同於在 v2 資料根跑原生 dedust worker，只把長路徑與參數收進來；額外參數照樣轉給 worker。
--stale 120 分鐘：profile timeout 1800 秒加 900 秒的下限是 45 分鐘，預設 45 會被拒。
worker 以 HANG_EXIT 退出（殺完 HFSS 仍凍結）時自動重啟，重啟後續跑自己的 claim；其他退出碼照常返回。
"""
import subprocess
import sys
import time

ROOT = r"\\140.123.106.219\temp\碩二_鄒穎麒's\antenna\experiments\r81_wide_filter_20261007\v2_v001\dataset"
HANG_EXIT = 75          # 與 script/dedust.py 的 HANG_EXIT 同值（這裡不 import dedust，免載 torch）
RESTART_WAIT = 30


def main(argv=None, call=subprocess.call, kill=None, sleep=time.sleep):
    extra = list(sys.argv[1:] if argv is None else argv)
    cmd = [sys.executable, "-m", "script.dedust", "--dataset-root", ROOT, "worker",
           "--scope", "symmetry_filter_20261007", "--selfgen", "0",
           "--poll", "60", "--stale", "120", *extra]
    while True:
        rc = call(cmd)
        if rc != HANG_EXIT:
            return rc
        print(f"worker 因 HFSS 凍結自我退出，{RESTART_WAIT}s 後重啟並續跑自己的 claim", flush=True)
        if kill is None:
            from script.kill import kill as kill
        kill()
        sleep(RESTART_WAIT)


if __name__ == "__main__":
    sys.exit(main())
