# CLAUDE.md — 專案工作指引

> 用**繁體中文**對話與寫文件。

## 北極星：核心精簡 + 解耦

這個 repo 本質很單純——**一個類 GAN（生成器 G ⇄ 代理 SM）＋ 一個真實 HFSS 模擬器做 online learning**。
學長（吳維文）原始碼把太多小工具、變體、監控、容錯全綁在一起，核心反而被埋住。

**核心 code 保持精簡、解耦**，把「錦上添花」從核心剝離（⚠ 輕量化針對**核心程式碼**；研究流程/文件——如 `docs/log/` 研究日誌——為了可追溯**刻意帶結構**、屬另一軸、不受此約束，但仍守「指向不複製」）：

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
- **SM 線上更新＝模式分派，不是單一寫法**：`sm_train.mode` ∈ `single`（學長原始單筆過擬合，golden 基準）／
  `replay`／`dlf`／`dlf_fit`（＝論文原版 DLF：全收＋累計均值重過濾 elite＋訓到收斂）／`refit`／
  `adaptive`／`adaptive_window`（`antenna/training.py:467` `_update_surrogate`，白名單 `SM_MODES`）。
- **回滾（rollback）已於 2026-06-28 移除**：三條理由（貪婪規則卡第一個山頭／退回舊 G 配當下變動的 SM
  本質矛盾／原實作有 off-by-one＋覆蓋最佳檔兩個 bug 實際 ≈ no-op）記在 `antenna/training.py:813`。
  探索改交給 K 候選＋SM 引導＋trust；最佳 pattern 仍在 `patterns/`（不可變）。

## 慣例

- **環境**：開發機 conda env `ant`（`/c/Users/Ricky/miniforge3/envs/ant/python.exe`）；正式機才有 HFSS。
- **測試**：`python -m pytest tests/ -q`（一定從 repo 根，否則 sys.path 不對）。
- **分支**：開發都在 `GAN`，全綠後 `main` fast-forward。commit 才 push、且只在使用者要求時。
- **NAS**：工作區 `ROOTDIR = T:\碩二_鄒穎麒's\antenna`（2026-08-04 升碩二改名;已遷出學長樹）。動學長(`碩二_吳維文's`)資料一律**唯讀、零刪除**(⚠ 2026-08-04 起本人夾也叫碩二_=`碩二_鄒穎麒's`,別搞混)。
- **備份（量測資料＝全專案最珍貴資產）**：一筆 HFSS ~100 秒、六萬多筆 ≈ 兩千小時機時,**重跑不回來**（code 有 git、權重能重訓、圖能重畫,只有量測不能）。NAS → 本機走 `/nas-backup`（增量 robocopy `/E /XO`、只讀 NAS、**永不用 `/MIR`**、跑完 `verify.py` 對帳）。備份時**不做聰明過濾**：`rad/` 是子夾、`results.json` 不是 `.pt`、`result/*/online/` 也藏著量測。本機備份根＝`C:\Users\Ricky\antenna_nas_backup`。
- **實驗記錄（每次都要做）**：新增/修改 `configs/*.yaml` 或訓練腳本時，**同步更新 `configs/README.md`** 的對照表 —— 一個 config＝一行（測什麼、與 base 差在哪、舊編號）。產生實驗 config 前先掃 `configs/README.md` 避免重複。這是硬規則，不是順手做。
- **研究日誌（每個 round 都要做）**：一個「假設→實驗→結論」＝一個 round。開新 round → `docs/log/` 開 `round-NN-<slug>.md`（用 `_TEMPLATE.md`，狀態 proposed）+ `configs/ONGOING.md` 加一行 🔵 指向它。跑完 → `python -m script.round_report --round NN --runs … --labels …` 產圖（落 `docs/log/assets/round-NN/`）+ markdown 數字貼進 round 檔 §4、補 §5 結論/§7 歸檔、`docs/log/README.md` 索引 +1 行、ONGOING 把該 round 移出 🔵（✅ 區留一行指標）。**四層別搞混**：`docs/` 設計文件＝為什麼（固定）/ `docs/log/`＝時間軸歷史（append-only）/ `configs/ONGOING.md`＝live 操作板 / `configs/README.md`＝config 全集；round 檔只連結、不複製。**更新 ONGOING 的 run 狀態前先跑 `python -m script.status`（掃 NAS 真相、別手動猜；`--md` 出可貼的表）**；重現診斷用 `script/analyze.py`。詳見 [[project_research_log]]。
- **批次 HFSS 驗證線（R7 起的主力實驗形式，與線上訓練並行）**：`script/dedust.py`——開發機 `select-*` 生輸入上 NAS → 正式機 `run --input X_input --store X`（可中斷續跑）→ 任一機 `report`。子命令地圖在檔頭 docstring；margin 同一把尺（`worst_margin`）、生成決定性、雜訊地板已公證 ≈0（跨機 bit 級一致）。round 檔照常開、判準發車前寫死。**選批發車前必跑 `check-dup --input X_input`**（批內＋全歷史交叉,自動掃描全部輸入夾；`notarize`/`repeat` 豁免；exit 1 就別發車）。**批次收檔 → Claude 主動 invoke `/batch-cycle <round> <batch>`**（主 runbook：判讀→公證→重錨→發車→補池→掛偵測→記帳；輪結束時內轉 `/close-round`），不等使用者下指令。批次線 skill 全家＝`/takeover`（新 session 接手先跑）·`/batch-cycle`·`/notarize`·`/new-round`·`/close-round`·`/gain-check`·`/stall-protocol`·`/reconcile`（切模型/長自動跑/宣稱換王前對帳 git·NAS·records,防「在未落地狀態上蓋」——2026-07-13 持久化事件教訓）；紀錄門檻機器真相源＝`docs/records.json`。**每輪硬上限 3 批**（Ricky 2026-07-13,見 docs/log/CLAUDE.md）；研究方法原則（價值軸=帶外主鍵·軸相關枯竭·探索型介入用效率評·自主續輪宣告制）定案在 `docs/discuss/decisions.md`。
- **討論記憶（兩層）**：`docs/discuss/scratch.md`（隨意層：半熟點子/觀察，Claude **隨手記、不主動報**）＋ `docs/discuss/decisions.md`（確定層：定案結論/方向，Claude **新增會主動說**）。用途＝讓「對話討論本身」有記憶、可跨 session 接續。與 `docs/log/` 分開：research round 的假設→實驗→結論走 `docs/log/`（正式、結構化）；這兩層是低門檻的討論便條，只指向不複製 docs/log。
  - **流動**：scratch（半熟）→ 熟了升 `configs/ONGOING.md` 🔜 候選區（一句 what＋why＋**觸發條件**）→ 觸發即開 round。decisions 只放「原則/定案」，**不放待辦**。
  - **生命週期**：升級/否決的 scratch 條目**不刪、標狀態**（`✅ 已升級→去處`／`❌ 否決＋為什麼`）留思路軌跡；**每次開新 round 檔時順手掃一遍 scratch**（熟的升、死的標）——回顧節奏掛在 round 邊界，不另設排程。

## 持續優化的候選方向（非強制，看到順手就做）

- `antenna/utils/utils.py`（~630 行）仍偏雜，可續拆/瘦身。
- 只服務實驗變體的死碼（無用的 GEN/SM 變體、`application/app.py` 的 `PathFixUnpickler` 等）可清。
- HFSS 容錯 watchdog/run_forever（使用者標記「先不急」，穩定時再說）。
- ~~論文版 DLF rollback filter 未移植~~ → **已解（2026-07-28 校正）**：DLF 本體已以 `sm_train.mode: dlf_fit`
  移植（見上節）；原本綁在 rollback 上的 `filter(upper=平均loss)` 隨 rollback 一起退場，不是落差。
- 方向圖（radiation pattern）→ loss 已接：`SinglePortRadSimulator` 萃取 + SM 平滑基底 rad head + `beam_coverage_loss`（Stage 1-2 完成、golden 零漂移）；Stage 3 冷啟動離線資料待做。規格＝窗 **±45°** / floor 3dB（學長後續討論定，原 ±55，可調），詳見 `docs/development.md` §4.6。
- **SM 單筆擬合過於激進（待優化）**：`train_one_data`/`train_one_data_rad` 把「這一筆」訓到 `loss<min_loss(0.1)`、最多 `max_epoch(20000)` 步——就地過擬合單一資料點，易不穩（曾觸發梯度爆炸/NaN，已加防護網但沒治本）。可優化方向：梯度裁剪（clip_grad_norm）、調降 `max_epoch`、用相對收斂門檻、或加正則。改前先想清楚對線上學習收斂速度的影響、保 golden。

## 更多

詳見 `docs/`：`quickstart.md`（跑起來）、`training.md`（config）、`architecture.html`（模組地圖 + 論文術語）、`development.md`（測試/golden/擴充）。
AI 協作記憶鏡像＝`docs/memory/`（harness memory 的導出快照，README 有接手閱讀順序與同步指令）。
文件若與 code 不一致，**以 code 與 `tests/` 為準**。
**子目錄各有局部 CLAUDE.md**（`docs/log/`、`docs/report/`、`script/`、`antenna/`、`tests/`、`configs/`）——該目錄的操作/撰寫規範在那裡，本檔只留原則；兩邊守「指向不複製」。
