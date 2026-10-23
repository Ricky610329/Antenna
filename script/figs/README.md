# script/figs — 歸檔 round 的圖表生成腳本（可重現性存檔）

`docs/log/assets/round-NN/` 各圖的來源。**pattern 渲染鐵則:第一索引 i 朝下、j 朝右=饋線邊在圖下緣**
(94eba73 轉置事故;imshow origin="upper" 天然滿足,自畫 Rectangle 要轉座標)。
**色條鐵則(Ricky 2026-08-08):上端=深色=好、下端=淺色=差,全站統一**;無資料背景用斜線紋跟「值」區分。
一次性分析腳本、非模組（直接
`python script/figs/<name>.py` 跑,路徑寫死 repo 位置）；資料來自 NAS 的
dedust store / pool 快取。R6 的圖走 `python -m script.expected_best report`、analysis-05 的圖走
`python -m script.analyze regularity`（正式模組,不在此）。

| 腳本 | 產出 |
|---|---|
| `r8_figs.py` | round-08 四圖（除塵/補洞/SM 體檢/random 基線） |
| `r8_pattern_gallery.py` | round-08 除塵前後 pattern 實例 |
| `pool_families.py` | round-09 池 top-300 家族普查圖 |
| `r9_figs.py` | round-09 四圖（校正/探索全景/SM 體檢/冠軍 gallery） |
| `r9_champ_curves.py` | round-09 冠軍三傑三標曲線 |
| `r10_figs.py` | round-10 三圖（w17 血統/三標曲線/承重熱圖） |
| `report_r1r10_style.py` | R1-R10 成果報告圖共用風格（下三支 import） |
| `report_r1r10_online.py` | 報告 F1-F3（紀錄時間軸/迴圈示意/線上線五輪疊圖）→ `docs/report/assets/` |
| `report_r1r10_batch.py` | 報告 F4-F11（R6 期望邊界/分布、R7 除塵、歸因、R8 四臂、R9 家族/校正/s05） |
| `report_r1r10_champs.py` | 報告 F12-F16（血統/承重圖/八冠軍/曲線/+0.48 案例） |
| `report_r1r18.py` | 總報告 E1-E6（每輪最佳 gallery/血統鏈/紀錄時間軸/帶外 Pareto/分組答案/新王 a024）→ progress-r1-r18 |
| `champ_compare.py` | **通用新舊冠軍對比圖**（CLI:--new/--old 自動定位;紀錄易主收檔必渲染,規則見 docs/log/CLAUDE.md）。pattern 差異＝綠加銅/紅去銅、radiation＝極座標（helper 在 report_r1r10_style）；`--plain`＝素底圖、`--old-left`＝before→after 閱讀順序 |
| `report_rad_polar.py` | 報告用：指定冠軍的方向圖**極座標**渲染（CLI:--ids/--labels;主波束朝上+±45°窗+G0−3dB 圈） |
| `report_dist_vs_strategy.py` | 報告 §2：「分布 ≫ 策略」概念圖（輸 random 輸在分布不在搜尋;概念示意非實資料） |
| `report_online_profile.py` | 報告 §8.1：線上 278s/ep 階段佔比 vs 批次線 160s/筆（profiling ec774de） |
| `report_dip_schematic.py` | 報告 §3.1：規格→生成器→pattern→SM→loss 順向/反向示意（梯度落在 pattern＝DIP 教學圖，概念示意非實資料） |
| `report_rad_need.py` | 報告 §3.4（對外版）：兩個帶內達標設計的方向圖對比（b28b2_010 rad −4.63 ✗ vs rad 王 o23b1_007 +1.00 ✓；pattern 素色不做差異渲染） |
| `report_senior_trajs.py` | 報告 §4（對外版）：學長 41 條軌跡疊圖（灰=個別/橘=最好+0.38/藍=逐輪中位,<15 條截斷；資料=R6 快取 senior_curves.npz） |
| `report_wm_dist.py` | 報告 §7（對外版）：兩資料集 wm 分布疊圖（學長 24k/達標 18 vs 新 ~9.3k/達標 ~2k;現場掃 NAS 排公證夾,數字隨工廠成長） |
| `report_diversity.py` | 報告 §7.2（對外版）：款式目錄規格書式——前 4 大款式各一代表,每款一列 [pattern\|S11\|Gain\|rad 極座標]（≤10px 連通=款式,64 款;代表優先挑有 rad 檔者） |
| `report_occlusion.py` | 報告 §7.3（對外版）：s05 承重單色溫度圖（dedust_occl 5×5 遮蔽掃描,白→紅=掉幅,格內標 Δwm） |
| `report_champion.py` | 報告用：單一設計規格卡 [pattern\|S11\|Gain\|rad 極座標]（CLI:--id/--title/--sub;定位/響應沿用 champ_compare,餘裕讀 store results.json） |
| `report_sprint48.py` | R28-R29 48hr 衝刺總覽（王座演進/多樣性換血/adversarial 閉環/低側誠實面板）→ docs/report/assets/sprint48.png |
| `pattern_map.py` | **pattern 空間分布圖**（全史 PCA 2D:家族著色+低側 lo 著色;每輪重跑=大陸漂移追蹤;王讀 records.json,HITS 手動更新;R32 海峽診斷出處） |
| `wm_terrain.py` | **wm 地形圖**（同 PCA 基底;高度=KNN8 局部潛力上限,海平面=三標線 wm=0,無資料區留白;★王=最高峰/▲同框筆/◇海峽臂;2026-07-17 Ricky 點名工具） |
- `response_map.py` — response 空間 PCA 三面板（S11+Gain 攤平;wm/左側lo/右側hi 著色+合格圈;2026-07-23 左右側拆帳制配套）
- `data_map.py` — pattern×response 雙空間 PCA 資料分布地圖（wm 著色+王朝表型;2026-07-25）
- `sm_capability.py` — SM 能力四面板（held-out 軌跡/盲測對決/輸出多樣性/落點疊圖;2026-07-25）
- `pareto_front.py` — 帕累托前緣雙面板（主戰場 wm×oob/左側 wm×lo;2026-07-26）
- `methodology_figs.py` — `docs/methodology.md` 兩張示意圖（§0.1 四角色一條環＋三層時間尺度／§0.3 決策權分配＋decisions.md 署名分布）→ `docs/assets/`；純 schematic 無外部資料，署名數字現場統計 decisions.md 標題不寫死
- `diffsim_l1.py` — analysis-08 的 L1 診斷四圖（diffsim wm vs HFSS wm 分層散點／曲線實例／每頻點 ρ／通道歸因）→ `docs/log/assets/analysis-08/`；`--split val` 才是 gate 報數用的那張
- `diffsim_final.py` — analysis-08 最終對比圖（5 個模型 × 5 個分層的 ρ 全景，含 gate 線）；數字寫死在檔內（gate 是「只跑一次」的讀數，不該每次畫圖重跑 val）
- `symmetry_profile.py` — 具名 profile 凍結資料的幾何六角格與 S11/Gain 頻率分布；輸入 `symmetry_analysis profile` 的 hash-bound JSON/NPZ，固定幾何座標、無插值。加 `--sample-id <id>` 產生單一樣本的金屬像素／S11／Gain／28 GHz 極座標方向圖卡，沿用 `polar_rad_ax`（0°朝上、每圈5 dB、±45°窗、G0−3 dB圈），兩截面共用刻度；顯示下限與截點數寫入收據，保留單次量測與描述統計限制。產生輸入 JSON/NPZ 的 `script/symmetry_analysis.py` 已於 2026-10-12 移除（tag `pre-cleanup-20261012`），已凍結的資料仍可直接重畫。
- `meeting_*.py` — 2026-10 組會簡報（6–9 月成果）補圖，輸出到 `docs/進度報告/組會_2026-10/figs/`（該夾 gitignore，只在本機）；只讀 repo 內 md/json 與本機 tmp 快取，零 NAS。共用設定兩支：`meeting_common.py`（records_overview／relay_climb／ga_vs_random／f04 用）、`meeting_figkit.py`（dust_two_way／wing_waterfall／diag_grammar／diamond_vs_cut／mesh_convergence／diffsim_final／emforge_onboard 用）；spec_masks／phantom_evidence／gain_waterfall／budget_scatter 自帶設定；`meeting_square.py`（方格 pattern／響應）、`meeting_clean.py`（乾淨版分析圖）、`meeting_schematic.py`（講解示意圖，假資料不帶數字）也自帶設定。數字寫死處皆在檔內註明出處 round 檔。
- `r80_sym.py` — R80 對稱天線：性能爬升（R80 5,000 筆＋10/15 續探，依完成時間）、續探放大圖與最佳設計卡片（S11／Gain／pattern 含對角橋／28 GHz 兩切面方向圖）→ `docs/log/assets/round-80/`
- `r81_v6.py` — R81 學長 v6 規格（11–40 GHz、阻帶 ≤ −20）：性能爬升（補量＋b62 起各批）與目前最佳設計卡片（S21／S11／S22 各一格＋pattern）；另以 −15 版（v6a）尺回算一張爬升（右格放大 b96 起）與一張最佳卡片 → `docs/log/assets/round-81/r81v6_climb.png`、`r81v6a_climb.png`、`r81v6_best_card.png`、`r81v6a_best_card.png`
- `r81_best_gif.py` — R81 濾波器最佳演進 GIF：預設 v6a（11–40 GHz 量測起全以 −15 版評分），累計最佳每刷新一格；左 2×2 S21／S11／S22＋pattern、右爬坡（同 r81v6a_climb.png 左格，隨動畫推進）；每格 0.2 s；`--history all` 改全程主線各段當時規格 → `docs/log/assets/round-81/r81v6a_best_evolution.gif`
- `r81_v6_question.py` — R81 問學長用：v4x 最佳設計 b57_004 以 11–40 GHz 重量後，分別套 v4x 與學長 v6 spec 的 S21（標出 13.5–14 GHz 漏點）→ `docs/log/assets/round-81/r81_v6_question.png`
- `r81_reseam.py` — R81 任意切縫示意：v6 最佳斷開設計（直線交界）與三個鋸齒切縫重切變體（上下主體與浮島分色）→ `docs/log/assets/round-81/r81_reseam_demo.png`
- `r82_s2_trial.py` — R82 上線試用：同批爬山名額「原 SM 挑」vs「變化量模型挑」的預測 vs 實測（兩格各一模型）→ `docs/log/assets/round-82/r82_s2_trial.png`
- `r85_block_scan.py` — R85 區塊掃描：b92_016 每個 4×4 小塊反轉後的 Δ低阻／Δ高阻 熱圖（兩格）→ `docs/log/assets/round-85/r85_block_scan.png`
- `report_generations.py` — 2026-10 日月光「歷來實驗總整理」世代時間軸：四代各一條帶（代表方法＋口試／日月光會議／交付／量測／代表成果標記），上方一條日月光要求與學長傳達；2026/06 起時間軸放大 3.5 倍並以淺底標示。日期與數字寫死、抄自 `docs/進度報告/10月_歷來實驗總整理/實驗總表.md`，零 NAS → `docs/進度報告/10月_歷來實驗總整理/figs/timeline_generations.png`；同支另出簡版甘特圖 `timeline_simple.png`（線性時間軸、四條帶各一行白話主題、4 個日月光節點）
- `report_ase_climb.py` — 2026-10 日月光報告第四代四任務爬升圖（單埠天線／對稱天線／雙埠濾波器／寬頻濾波器）：橫軸累計 HFSS 模擬筆數、縱軸最差餘裕，淡點每筆、粗階梯累計最佳、3–5 個白話節點，圖內無標題與內部術語；單埠／雙埠讀本機 NAS 備份逐筆樣本檔時間（快取 `tmp/ase_climb_*.npz`，`--refresh` 重掃），對稱與寬頻沿用 `r80_sym`／`r81_v6` 的讀取；口徑與數字出處在 `docs/進度報告/10月_歷來實驗總整理/notes/gen4_爬升量化.md` → 同夾 `figs/gen4_ase/climb_*.png`
