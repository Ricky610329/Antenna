"""R81 spec v2 正式機 worker 短入口：`script\\r81v2.cmd`（三台同一行）。

等同於在 v2 資料根跑原生 dedust worker，只把長路徑與參數收進來；額外參數照樣轉給 worker。
--stale 120 分鐘：profile timeout 1800 秒加 900 秒的下限是 45 分鐘，預設 45 會被拒。
"""
import subprocess
import sys

ROOT = r"\\140.123.106.219\temp\碩二_鄒穎麒's\antenna\experiments\r81_wide_filter_20261007\v2_v001\dataset"


def main(argv=None):
    extra = list(sys.argv[1:] if argv is None else argv)
    cmd = [sys.executable, "-m", "script.dedust", "--dataset-root", ROOT, "worker",
           "--scope", "symmetry_filter_20261007", "--selfgen", "0",
           "--poll", "60", "--stale", "120", *extra]
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
