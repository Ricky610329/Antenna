# CLAUDE.md — 專案工作指引

> 用**繁體中文**對話與寫文件。

## 北極星：持續輕量化 + 解耦

這個 repo 本質很單純——**一個類 GAN（生成器 G ⇄ 代理 SM）＋ 一個真實 HFSS 模擬器做 online learning**。
學長（吳維文）原始碼把太多小工具、變體、監控、容錯全綁在一起，核心反而被埋住。

**每次經手都讓它更輕、更解耦一點**，把「錦上添花」從核心剝離：

- 核心要能單獨讀懂、單獨跑：`G(spec) → pattern → SM/SIM → loss → 反傳`。
- 監控、容錯、視覺化、舊格式相容 = **外接模組**，不該污染核心。
- 看到 over-design（不必要的泛型、抽象、間接層）就**簡化**；看到核心依賴周邊就**解耦**。
- 但——**先 audit 既有 codebase 再動，不平行重建**；能刪不要包，能扁平不要巢狀。

## 動手前的護欄（讓優化安全）

1. **golden 保真**：任何結構性改動，`python -m pytest tests/ -q`（在 repo 根目錄跑）必須全綠、golden 零漂移。這是重構安全網。
2. **不破壞舊檔相容**：`Data`/`DataManager`/`Record`/checkpoint 都只 pickle **純 payload**，搬類別 OK；但 `antenna/utils/utils.py` 的 `Path`（有 `__reduce__`，被烘進舊 checkpoint）**不可搬離**。
3. **誠實**：報告先講 scope 與 limitations，別把工具當成 production system；測試沒過就說沒過。
4. **CI**：pyflakes 擋 undefined name；golden 雙容差（本機絕對 1e-4 / CI 相對 1%）。改動別讓 CI 紅。

## 架構不變式（「解耦/輕量」對本專案的定義）

- **層級單向**：`antenna/` 核心（pattern/response/training/models/optim/losses/zoo）**零 legacy 依賴**。
  只有 `train.py`、`script/`、`application/` 可以碰 `antenna/legacy/`。
- **檔案制資料層、訓練路徑零 pickle**：
  - 資料集 = `SampleStore`（一筆一檔，hash 即檔名去重，`antenna/utils/store.py`）。
  - 訓練狀態 = `RunState`（`metrics.csv` + `patterns/`，`antenna/utils/runstate.py`）。
  - 監控 = TensorBoard（`monitor.py`）；運行管理 = `status.json` 心跳。兩者是兩件事。
- **核心 vs legacy 的界線**（別搞混）：
  - 核心：`Record`（ACP 排程器 + 每個 checkpoint 用）、`size_converter`（`utils/torch_utils.py`）。
  - legacy：`Data`/`DataManager`/`make_hashable`/`dynamic_loss_filter`（`antenna/legacy/`，只讀舊 `.dataset`）。
- **config 驅動**：一個 `configs/*.yaml` = 一組實驗；模型用名字選（`antenna/zoo.py`）；不改 code 加實驗。
- **型別註解＝輕量文件**：簡單註解歡迎；`TypeVar`/`Generic`/`ParamSpec`/`@overload` 一律不用。
- **tau 歸 ACP**：二值化是訓練管線固定一步，模型不碰 tau。

## 慣例

- **環境**：開發機 conda env `ant`（`/c/Users/Ricky/miniforge3/envs/ant/python.exe`）；正式機才有 HFSS。
- **測試**：`python -m pytest tests/ -q`（一定從 repo 根，否則 sys.path 不對）。
- **分支**：開發都在 `GAN`，全綠後 `main` fast-forward。commit 才 push、且只在使用者要求時。
- **NAS**：工作區 `ROOTDIR = T:\碩一_鄒穎麒's\antenna`（已遷出學長樹）。動學長 `碩二` 資料一律**唯讀、零刪除**。
- **實驗記錄（每次都要做）**：新增/修改 `configs/*.yaml` 或訓練腳本時，**同步更新 `configs/README.md`** 的對照表 —— 一個 config＝一行（測什麼、與 base 差在哪、舊編號）。產生實驗 config 前先掃 `configs/README.md` 避免重複。這是硬規則，不是順手做。

## 持續優化的候選方向（非強制，看到順手就做）

- `antenna/utils/utils.py`（~630 行）仍偏雜，可續拆/瘦身。
- 只服務實驗變體的死碼（無用的 GEN/SM 變體、`application/app.py` 的 `PathFixUnpickler` 等）可清。
- HFSS 容錯 watchdog/run_forever（使用者標記「先不急」，穩定時再說）。
- 論文版 DLF rollback filter（`filter(upper=平均loss)`）未移植，是已知架構落差，需要時另案補。
- 方向圖（radiation pattern）→ loss 尚未接：只完成資料萃取（`SinglePortRadSimulator`），SM 多輸出頭/相對平坦度 loss 等決定見 `docs/development.md` §4.6（學長 ±55°/3dB 是「他們的」規格，非定案）。
- **SM 單筆擬合過於激進（待優化）**：`train_one_data`/`train_one_data_rad` 把「這一筆」訓到 `loss<min_loss(0.1)`、最多 `max_epoch(20000)` 步——就地過擬合單一資料點，易不穩（曾觸發梯度爆炸/NaN，已加防護網但沒治本）。可優化方向：梯度裁剪（clip_grad_norm）、調降 `max_epoch`、用相對收斂門檻、或加正則。改前先想清楚對線上學習收斂速度的影響、保 golden。

## 更多

詳見 `docs/`：`quickstart.md`（跑起來）、`training.md`（config）、`architecture.html`（模組地圖 + 論文術語）、`development.md`（測試/golden/擴充）。
文件若與 code 不一致，**以 code 與 `tests/` 為準**。
