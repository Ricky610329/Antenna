"""
antenna.utils — 工具子套件匯出聚合層
======================================
本模組作為 utils 子套件的「單一入口」，將分散在各子模組的
常用符號集中 re-export（全部顯式列名，無萬用字元），讓使用端
    from antenna.utils import config, Record, ...
即可取得常用工具與實驗路徑常數。

子模組分工：
  utils.py       — 通用基礎工具（Config、Figure、Record、Path 等）
  web.py         — 網路 / 網路磁碟機 / Email 功能
  torch_utils.py — PyTorch 張量輔助工具（含 size_converter 形狀轉換）
  types.py       — 天線系統共用型別別名（GEN/SM/SIM 接口型別）
"""

###* ── utils.py ─────────────────────────────────────────────────────────────
#? errorCallback : 統一錯誤回呼函式，供訓練迴圈 try/except 區塊呼叫
from .utils import errorCallback
#? Path          : pathlib.Path 的包裝或直接 re-export，讓腳本免 import pathlib
from .utils import Path
#? plot          : 快速繪圖輔助函式，封裝 matplotlib 常用操作
from .utils import plot
#? Config        : 實驗設定類別 (class)，支援 .yaml 讀寫、屬性存取
from .utils import Config
#? config        : 全域預設 Config 實例，訓練腳本可直接存取共用參數
from .utils import config
#? Figure        : 封裝 matplotlib Figure，提供儲存/顯示快捷方法 (figure.py)
from .figure import Figure
#? Axes          : matplotlib Axes 的 re-export，供型別提示使用
from .utils import Axes
#? Record        : 訓練紀錄物件 (=TEMP)，斷點續跑/rollback 的核心 (record.py)
from .record import Record
#? logger        : loguru logger（全專案統一的日誌器）
from .utils import logger
#? Complete      : 訓練完成通知（log + 可選 Email）
from .utils import Complete
#? TQDM_*        : tqdm 進度條的統一樣式設定
from .utils import TQDM_BAR_SIMPLE, TQDM_CONFIG

###* ── web.py ────────────────────────────────────────────────────────────────
#? connect_network_drive : 掛載實驗室 NAS 網路磁碟機（對應 T:\ 路徑）；
#?                         訓練腳本在讀取 ROOTDIR 前應先呼叫確保磁碟可用
from .web import connect_network_drive
#? get_local_ip          : 取得本機 IP，用於多機分散式訓練或日誌標記
from .web import get_local_ip
#? Email                 : 封裝 SMTP 寄信功能，訓練完成 / 異常時發通知信
from .web import Email

###* ── torch_utils.py：size_converter ────────────────────────────────────────
#? size_converter : 通用張量形狀轉換器（(B,N)攤平 / (B,H,W)影像 / 批次維度間轉換）
from .torch_utils import size_converter

###* ── torch (標準庫) + torch_utils.py ────────────────────────────────────
#? nn     : torch.nn 模組 re-export，讓腳本免再 import torch
from torch import nn
#? Tensor : torch.Tensor 型別 re-export，供函式簽章型別提示使用
from torch import Tensor
#? tensor  : 建立 torch.Tensor 的輔助函式，可能含額外型別/裝置預設
from .torch_utils import tensor
#? cTensor : 複數張量 (complex Tensor) 輔助函式，用於 S-parameter 計算
from .torch_utils import cTensor

###* ── 實驗室 NAS 根路徑常數 ──────────────────────────────────────────────
#! ROOTDIR      : 本實驗工作區根目錄（T:\ 對應網路磁碟機）；
#!               所有模型 checkpoint、資料集、結果均存於此路徑下。
#!               使用前請確認 connect_network_drive() 已成功掛載 T:\。
#!  ◆ 已自學長樹 (碩二_吳維文's) 遷出，改用自己的工作區。學長過往真實模擬樣本
#!    已收割進 dataset/harvest_single|dual (見 script/harvest_legacy.py)。
ROOTDIR = Path(r"T:\碩二_鄒穎麒's\antenna")

#! DATASET_PATH : 天線資料集目錄，位於 ROOTDIR/dataset；
#!               訓練腳本以此為基礎路徑載入 .csv / .json 天線參數資料。
DATASET_PATH = ROOTDIR.joinpath('dataset')
