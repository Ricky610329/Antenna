import os as _os

from torch import Tensor, nn

from antenna.types import *

from .data import size_converter
from .torch_utils import *
from .torch_utils import cTensor, tensor
from .utils import *
from .utils import Axes, Config, Figure, Path, Record, config, errorCallback, json, plot
from .web import Email, connect_default_drive, connect_network_drive, get_local_ip

# 結果輸出根目錄。優先順序：
# 1. 環境變數 ANTENNA_ROOTDIR
# 2. 環境變數 ANTENNA_NETWORK_DRIVE_LETTER + 專案子路徑（若指向網路磁碟工作區）
# 3. 退回本地 CWD，結果寫入 ./result
# 避免硬編碼特定使用者的網路磁碟資料夾（之前會污染共用 drive）。
_env_rootdir = _os.environ.get("ANTENNA_ROOTDIR")
if _env_rootdir:
    ROOTDIR = Path(_env_rootdir)
else:
    ROOTDIR = Path(_os.getcwd())
DATASET_PATH = ROOTDIR.joinpath("dataset")
