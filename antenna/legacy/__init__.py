"""
antenna.legacy — 隔離的舊資料層（學長既有 code 相容用）
========================================================
這裡只放「舊的、單一 pickle 容器」資料層，與主訓練 pipeline 物理隔離。

  data.py — Data / DataManager / make_hashable / dynamic_loss_filter
            （.dataset / .data 單一 pickle 容器；一存一筆就全量重寫）

★ 設計原則：antenna/ 核心 (pattern / response / training / models / optim / losses)
  一律「不」依賴本套件。只有以下「非核心」角色可以碰 legacy：
    - application/app.py        : 舊實驗回溯、手繪 pattern 編輯器
    - script/convert_dataset.py : 舊 pickle → SampleStore 轉檔工具
    - script/kuohung.py         : KuoHung 參考圖樣（用 Data 容器讀 .data 快取）
    - train.py load_dataset     : 萬一指到舊 .dataset 時的 lazy fallback

  新標準是 antenna.utils.store.SampleStore（一筆一檔）；NAS 上學長真實模擬樣本
  已收割進 dataset/harvest_single|dual（見 script/harvest_legacy.py）。

  注意：Record（utils/record.py）與 size_converter（utils/torch_utils.py）
  「不是」legacy —— 它們仍被核心 pipeline 使用，故留在 antenna/utils/。
"""
from .data import (
    Data,
    DataManager,
    make_hashable,
    dynamic_loss_filter,
)

__all__ = ["Data", "DataManager", "make_hashable", "dynamic_loss_filter"]
