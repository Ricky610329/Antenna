# Round 35 — 新節奏首輪：批 75 高頻迭代 × 多軌 SM 第一步 × 鏈制常駐

- **狀態**: running（2026-07-22 午開輪;Ricky 拍板「收縮 round 讓結果更快更新模型」+
  「多軌 SM=性能提升關鍵,循序漸進」）
- **提出 / 開跑 / 結論**: 2026-07-22 / 2026-07-22 / —
- **一句話問題**: 把 tier 1 也「鏈化」——批 150→**75**（重錨頻率×2）+重錨輕量化（隔批制）,
  高頻小步閉環能不能把批次線的單位產出拉向鏈制水準？同時:多軌 SM 第一步（註冊表+校正表）落地。
- **一句話結論 (TL;DR)**: 待跑
- **指向**: [round-34](round-34-second-bloodline.md)（tier 架構元年/鏈制毫米線）· decisions
  「多軌 SM 路線圖」「合作方法三修正」· chains/*.jsonl（c1d3 攻 0.04/c2rad 逼 rad 0）

## 1. 假設 (Propose)
- **證據**：①鏈制實證=小步高頻閉環有效（12hr 兩軸毫米線）vs 大批三輪零紀錄;②v50 凍結尺
  1.21 新低=高頻教材直接餵準 SM;③重錨輕量化可行（ens/影子判定都是連兩批口徑,隔批訓不損判準）。
- **判準（發車前寫死;Ricky 可隨時否決）**：
  - **主判準（節奏紅利）**：R35 三批（225 筆）的三標率與帕累托前緣增量/百筆 ≥ R34 同口徑
    （batch 75 沒有因統計變小而丟失產出效率）∧ 輪週期實測 ≤ R34 的 60%——兩者兼備=新節奏常駐;
    產出效率顯著掉=回 150。
  - **輕量重錨審計**：奇數版（v51/v53…）=主+rad 頭;偶數版=全訓（ens+影子）——影子對決/std 鍵
    照連兩批口徑跨版判定;凍結尺不因輕量版系統性惡化（|Δ|<0.1）。
  - **多軌第一步**：SM 註冊表（configs/sm_registry.json）上線+校正表記錄版（analyze batch 印
    各臂 pred−real bias 表,兩批數據後判是否進鍵）。
  - **asym 記錄鍵**：select 記 asym 值進 manifest（記錄版,照 std/CNN 先例;R36 判進鍵）。
  - 鏈制常駐:c1d3（攻 wm 0.04）/c2rad（逼 rad 0）續跑;接棒鏈帶 --expert（判準=連兩包專家半勝）;
    c1d3 若 wm≥0.15∧oob<9.0=**可用帶外紀錄候選→公證+推播**。
  - 紀錄門檻引 records.json;紀錄級一律公證;批數 ≤3;五軸面板;修訂留註記。
- **配額（每批 75=3 夾）**：G 30（free 24/oobp 6）／L 12（爬山錨組）／M 7（凍結對照,跨批合併讀）／
  O 4／I 8／D 6／W 6／C 2。

## 2. 實驗設計 (Design)
| 項 | 設計 | 判準 |
|---|---|---|
| 批 75 | 各臂減半,重錨每批（輕量隔批制） | 產出效率不掉∧週期 ≤60% |
| SM 註冊表 | configs/sm_registry.json（職責/量測/狀態/退役） | 上線即過 |
| 校正表 | analyze batch 各臂 bias 表（記錄版） | 兩批後判進鍵 |
| asym | select 記 manifest（記錄版） | R36 判進鍵 |

## 3. 執行紀錄 (Run)
```
# 發車（開發機,conda ant;輕量重錨=奇數版 --no-ens --no-shadow）:
python -m script.sm_invert gen --sm sm_reanchor<vN>.pth --rad-head rad_head<vN>.pth --out-dir tmp/invert_stage_r35bN --n-free 24 --n-surg 0 --n-champ 0 --n-oob 6 --seed <60+N>
python -m script.dedust select-r35 --batch N --sm sm_reanchor<vN>.pth --gstage tmp/invert_stage_r35bN --rad-head rad_head<vN>.pth --novelty
python -m script.dedust check-dup --input dedust_r35bNa_input   # a..c 分開跑
python -m script.dedust jobs-add --input dedust_r35bNa_input --store dedust_r35bNa --prio 3   # ×3
```
| 批 | 狀態 |
|---|---|
| 1（r35b1{a-c}） | ✅ 17:08 收（15:42 發車;批週期 **86 分**,零 error。v51 輕量首航〔~60 分;凍結 1.36=+0.15 ⚠超審計線〕;asym 記錄鍵首航;查重 0×3。tier 0:c1d3 收鏈〔8.95 局部頂〕→ c1d4 expert 試點;c2rad 十一連勝 −0.45） |
| 2（r35b2{a-c}） | 🔵 20:56 發車（v52 全訓〔進程中斷,ens/shadow 同鍋補訓後齊裝〕;--rad-key 續鍵〔b1 前瞻+0.357〕;誤差錨外掛 8;查重 0×3。tier 0:c1d5 expert〔sizer 修復後首個真專家鏈,錨 wm0.14/oob9.09 毫米線〕+c2rad3〔rad −0.43〕） |

## 4. 分析 (Analyze) — b1
- **三標 8/75（11%）;合格（wm≥0.15∧rad≥0）3 筆**：i35b1_003 +0.29/oob12.8、m35b1_002 +0.29、m35b1_004 +0.28/**oob10.35**（本批可用帶外最佳;紀錄 9.0 零推進）。帕累托 +0。
- **⚠ G 臂 free 帶 100% adversarial**（24 筆 pred 中位 +0.55 vs real −8.35,|Δ| 8.95）——與 v51 凍結尺 +0.15 超審計線指向同一嫌疑=輕量重錨失真;**v52 全訓=裁決**（審計判準寫死 |Δ|<0.1）。誤差錨 5 筆已自動入 error_anchors.json。
- **asym 首讀（記錄版）**：G 臂中位 **0.585** vs 其他臂 0.21-0.28（兩倍+）——與幾何分析 asym↔wm ρ−0.63 一致,G free 慘況同源;**R36 判進鍵**（候選用途:G free 預過濾）。缺口:denovo 臂 6 筆 asym=None（生成路徑沒算,待補）。
- 前瞻:M 臂 wm ρ−0.14(弱)/oob+0.36/rad 頭 +0.357→**續鍵**（下批保留 --rad-key）。影子對決:CNN ρ0.754>MLP 0.715（CNN 續贏排序尺）。
- 多樣性恆溫:近王 3%/王系血統根 **5%**/歷史最近鄰 38——**反王朝壓制持續生效**（R33 高峰期 40%+→5%）。
- **輕量重錨審計首讀（v50→v51→v52）**：凍結尺 1.212→1.356（輕量,+0.144 ⚠超線）→**1.286**（全訓,
  收回 0.070）——v51 惡化約半數=輕量化代價（全訓可收回）、半數=r35b1 資料段本身（v52 仍 +0.074 vs v50,
  線內）。單次對照不判死,照判準連兩批口徑:v53（輕量）/v54（全訓）再對照一輪;若輕量代價 ~0.07 重現,
  隔批制的 20 分節省 vs 準度代價交 R35 收輪判。
- **開發機進程中斷（18:2x）**：Claude Code 進程換代,背景任務全滅——v52 訓練殺在 ens 段（主模+rad 頭
  +kpi 行已落地;ens52×2+shadow52 事後同鍋補訓〔ens_fix 腳本+train-shadow〕）、c1d4/c2rad2 daemon 死
  （在飛包 c1d4_p03/c2rad2_p02 機器層不受影響,watch 接盯,收檔後手動判讀+新鏈名接棒）。廢稿夾
  c1d4_p02/c2rad2_p01（撞歷史未派工）已清。
- **c2rad 假性收鏈（chain 教訓④=進程管理）**：p12 首卡後 p13-p20 八連「查重撞歷史」燒完 max_packs
  ——非鄰域枯竭,是 daemon 進程**還在跑 bug② 修復前的舊 code**（包內 used;前 11 包連勝每包換錨故無害,
  首次同錨重抽即現形,每對包期望撞 25²/600≈1 筆與實測吻合）。教訓寫死:**修 daemon bug 後重啟所有在跑
  的 daemon**。處置:刪 p13-p20 未派工草稿夾,**c2rad2** 從終錨 −0.45 修復版續爬（goal rad 不帶 --expert
  ——expert 排序鍵只支援 wm/dual）。

## 5. 結論 (Conclude)
（待）

## 6. 後續決策 (Next)
- 域專家三顆試點（R35-36）;軸判別器（R36+）;asym 進鍵判定（R36）。
- **收輪必判:tier 再平衡階梯**（decisions 2026-07-22,Ricky）——`analyze tiers` 產出率比 ≥2 連兩輪
  （R34 首讀 2.03×,本輪=第二讀）→ R36 批 75→50;多樣性 KPI 塌縮則回退。

## 7. 歸檔指向 (Archive)
- 結果夾: `dataset/dedust_r35b*`;公證 `r35n*`;鏈帳 docs/chains/。
