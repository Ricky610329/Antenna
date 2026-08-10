# script/ 操作規範

> 腳本一覽見 [README.md](README.md)；執行一律從 repo 根 `python -m script.<名>`。

## 批次 HFSS 驗證線（dedust.py）鐵則

1. 流程：開發機 `select-*` 生輸入上 NAS → **`check-dup --input X_input` 必跑、單獨跑不接管線**
   （exit 1 就不發車;接管線會被尾端指令吃掉 exit code=安全閘靜默失效,2026-08-06 實犯）→
   正式機 `run --input X_input --store X`（可中斷續跑、error 條目重試）→ 任一機 `report`。
   **收檔判讀＝`analyze batch --round R --batch N`**（臂別/前瞻/紀錄候選+公證指令/→行動;
   含**影子 CNN 雙模盲測段**——必須在重錨前跑,重錨後本批進訓練集就不是盲測）;
   收檔偵測＝`dedust watch --stores ...`（Monitor 直接掛）;重錨＝`sm_reanchor train --add "..." --out vN.pth`
   （自動 append `configs/clean_stores.txt`;**自帶制度合訓**:rad_headNN+ens 2 顆+影子 sm_shadowNN
   〔尺1 落 docs/kpi_shadow.csv;--no-rad/--no-ens/--no-shadow 可關〕）。
   gen 戰術換錨＝`sm_invert gen --champ-anchors "id:tag,..."`（B 層泵等,免改 code）。
   整鏈 runbook＝`/batch-cycle` skill。
2. check-dup **自動掃描全部輸入夾**（2026-07-10 起免維護清單;舊 select 內建去重仍引用 HISTORY_INPUTS）；
   蓄意重複＝kind `notarize`/`repeat`（查重豁免，公證靠這個）。
3. 新 select 函式的 docstring 寫死：臂別、筆數、判準（與 round 檔 §1 一致）。
   **命名規範（R23 起,Ricky 2026-07-12）**：round 號一路貫穿——夾 `dedust_r<NN>b<批><夾>`、
   id `<臂字母><NN>b<批>_<序>_<親>`（如 `m23b1_003_r2_016`）、填空池 `r<NN>g*`、公證 `r<NN>n*`;
   **不再用跨 round 的全域批次計數**（R21 m5_→R22 m6_ 是反例,R22 就算了）。
4. **生成決定性**：select 內不用未 seed 的隨機（`np.random.default_rng(seed)`）；同 seed 同輸出。
5. 同一個 store **不可兩台機同時跑**（results.json 整份重寫會互踩）；跨機接力 OK（斷點續跑）。
   **資料工廠模式**下這由 `jobs_state/<store>.claim` 原子認領自動保證——正式機常駐
   `python -m script.dedust worker`,派工用 `jobs-add --input X_input --store X [--prio N]`;
   單筆 watchdog（預設 900s 殺 HFSS 標 error 續跑）＋**死亡判定三層**（2026-07-15 升級,216 教訓）：
   連 5 敗=熔斷→冷卻 `--cooldown`(600s) 重開再試,`--max-blowout`(3) 循環用盡才判死→
   `.fail`（JSON 記 machines 名單）停機→**別台 worker 名單無自己=自動接管重跑**
   （毒批收斂:全機敗過=永久 fail 等人工）＋批尾自動補測（殘留 error 殺重開 HFSS 重試
   `--retry-pass` 輪,預設 2;同筆 3 連敗=毒樣本嫌疑不再試）;停止=建 `jobs_state/STOP`。
   **工作目錄（`_dedust_<store>`）跑完自動刪＋worker 啟動清掃**（2026-07-15 起;216 磁碟滿事件——
   0x80070223 連環保險絲的真兇=78 個 job 暫存吃光 C 槽;`--out` 自訂路徑不刪）。
   **機況探針**：`dedust probe --machine <IP末段> [--action status|cleanup]`——經 NAS 白名單指令,
   worker 空閒 poll 輪回應（磁碟/殘留/git 版/HFSS 行程;跑 job 中佔線最長 ~70 分才答）。
6. 錨點注意：x00 是破對稱錨點（含 (4,18) 翻轉），x00 條目收尾用除塵不對稱化（`_finish` 語義），
   不要 `symmetrize`（round-16 §3 caveat）。
7. **幾何/儀器變體批**（R52 網格起,R54-55 定型）：輸入夾放 `hfss_setup.json`（只收白名單鍵:
   `max_delta_s/max_passes/min_passes/min_converged/timeout/diag_bridge_w/pixel_count`,run 自動存證進 store）;
   kind=`meshconv/diagbridge`=check-dup 豁免（豁免集合以 code 為準,僅 bits 不變的幾何變體;
   `symprobe` 等**改 bits 的變體不豁免、必跑 check-dup**）、id=`{parent}~<tag>` 親代綁定;
   **資料永不入鍋**（不進 clean_stores、重錨不 --add——漏一次=污染 SM 訓練集）;
   新 COM 動詞（Rotate/Subtract 級）首筆當 smoke+幾何渲染目檢。
8. **機台部署事件**（改 worker 端 code:dedust/single_port）：push → 逐台 pull+**重啟 worker**
   （光 pull 不重啟=舊 code 陷阱,已兩犯;從機台本機桌面終端啟動,HFSS 視窗才可見）→
   長駐行程（chain daemon）一併重啟 → 首筆 smoke+`jobs-ls` 驗佇列。

## 其他慣例

- 圖表腳本歸 `figs/`＋`figs/README.md` 索引 +1 行；報告圖規範見 `docs/report/CLAUDE.md`。
- 改完跑 `python -m pyflakes script/<檔>` 清 undefined name / unused import（CI 會擋）。
- 判讀/統計工具優先擴充 `analyze.py` 子命令，不另開一次性腳本（用過兩次才工具化）。
- NAS 路徑經 `DATASET_PATH`（`antenna/utils`），不硬編 `T:\`（路徑含 `'`:bash 單引號/heredoc 會炸,用雙引號）。
- 旗標「轉正」（實驗→常規）必同步改 parser 預設——靠指令記憶帶旗標已三犯（rad_head/train-two/ds-mode）。
