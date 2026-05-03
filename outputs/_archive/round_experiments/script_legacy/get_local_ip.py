"""印出本機 IP 位址的 CLI 包裝；實作轉用 antenna.utils.web。"""

import sys
from os.path import dirname, join

# 允許直接 `python script/get_local_ip.py` 執行而不必設 PYTHONPATH
sys.path.append(join(dirname(__file__), ".."))

from antenna.utils.web import get_local_ip  # noqa: E402

if __name__ == "__main__":
    print(f"Local IP Address: {get_local_ip()}")
