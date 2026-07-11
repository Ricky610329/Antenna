# script/ 操作規範

> 腳本一覽見 [README.md](README.md)；執行一律從 repo 根 `python -m script.<名>`。

## 批次 HFSS 驗證線（dedust.py）鐵則

1. 流程：開發機 `select-*` 生輸入上 NAS → **`check-dup --input X_input` 必跑**（exit 1 就不發車）→
   正式機 `run --input X_input --store X`（可中斷續跑、error 條目重試）→ 任一機 `report`。
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
   單筆 watchdog（預設 900s 殺 HFSS 標 error 續跑）＋連敗保險絲（5 筆→標 .fail 停機等人工）
   ＋批尾自動補測（殘留 error 殺重開 HFSS 重試 `--retry-pass` 輪,預設 2;同筆 3 連敗=毒樣本嫌疑不再試）;
   停止=建 `jobs_state/STOP`。
6. 錨點注意：x00 是破對稱錨點（含 (4,18) 翻轉），x00 條目收尾用除塵不對稱化（`_finish` 語義），
   不要 `symmetrize`（round-16 §3 caveat）。

## 其他慣例

- 圖表腳本歸 `figs/`＋`figs/README.md` 索引 +1 行；報告圖規範見 `docs/report/CLAUDE.md`。
- 改完跑 `python -m pyflakes script/<檔>` 清 undefined name / unused import（CI 會擋）。
- 判讀/統計工具優先擴充 `analyze.py` 子命令，不另開一次性腳本（用過兩次才工具化）。
- NAS 路徑經 `DATASET_PATH`（`antenna/utils`），不硬編 `T:\`。
