---
name: project-data-dual-track
description: 資料已收割到自己 NAS 工作區、ROOTDIR 遷出學長樹、legacy 資料層隔離到 antenna/legacy/；Record 與 size_converter 是核心非 legacy
metadata: 
  node_type: memory
  type: project
  originSessionId: 2adb79f3-22b2-4395-bf20-4dc71c4ffd49
---

**2026-06-11 完成「資料自有化 + legacy 隔離」**（接續 2026-06-10 起的資料層雙軌 commit `3d9ef59`）：

**1. 收割 done**：`script/harvest_legacy.py` 掃學長 `result/` 55 個夾的 `online.dataset`（裸 `pickle.load` 唯讀，不用 DataManager 以免它在學長夾寫 .log），依響應通道數分流（y=(2,17)→single、y=(3,17)→dual），去重寫入自己 NAS：
  - `T:\碩二_鄒穎麒's\antenna\dataset\harvest_single`（**24189** 筆）
  - `T:\碩二_鄒穎麒's\antenna\dataset\harvest_dual`（**10023** 筆）
  - 0 去重（fingerprint 同時吃 pattern+response，跨 run 重模擬有 float 微差）。磁碟檔數/筆數/形狀/dB 值域全驗證通過。

**2. 工作區遷移**：`antenna/utils/__init__.py` 的 `ROOTDIR` 從 `碩二_吳維文's\...` 改成 `T:\碩二_鄒穎麒's\antenna`（脫離學長樹）。warm-start 資產（`old_sm.pth`/`patch_dual.pth`/`KuoHung Pattern/`）已複製到工作區 dataset/。11 個 configs 的 `offline_dataset` → `harvest_single`/`harvest_dual`（`pretrained` 維持 warm-start）。

**3. legacy 隔離**（使用者選「搬進 antenna/legacy/」）：
  - `antenna/legacy/data.py` = `Data`/`DataManager`/`make_hashable`/`dynamic_loss_filter`（舊單檔 pickle 容器）。importers 改指：`kuohung.py`/`app.py`/`convert_dataset.py`/`train.py`(lazy fallback)。
  - **`Record`（utils/record.py）與 `size_converter`（已從 data.py 抽到 utils/torch_utils.py）是核心，不是 legacy**——Record 被 scheduler+每個 checkpoint 用、size_converter 被 losses/response/pattern/training/surrogates 用。移動它們會層級顛倒。
  - 原則：`antenna/` 核心零 legacy 依賴；只有 script/ application/ train.py 可碰 legacy。
  - pickle 安全：Data/DataManager/Record 都只 pickle 純 payload（非自身物件），故搬類別不影響舊 .dataset/.record/checkpoint 讀取。
  - 全綠：81 tests + golden 零漂移 + pyflakes 無 undefined + import smoke（含 application.app）。**尚未 commit**。

**已過時的待辦**：`convert_dataset patch_single_mirror/patch_dual` 不再需要——收割已取得 result 夾的真實樣本，configs 也改用 harvest 了（那兩包 offline .dataset 留在學長夾，無人引用）。

相關：[[feedback-prefer-simplicity]]、[[project-config-driven-validated]]、[[reference-lab-pipeline-locations]]
