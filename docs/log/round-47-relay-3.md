# Round 47 — 接力三輪：均衡起點驗天花板 × 批線常態

- **狀態**: running（2026-07-29 08:3x 開輪;自主續輪宣告制;R46 收輪接棒）
- **提出 / 開跑 / 結論**: 2026-07-29 / 2026-07-29 / —
- **一句話問題**: 均衡型深水原礦（rad≈0∧lo≤−4 起點）能不能突破 d/g 兩線都撞上的
  −2.7~−3.0 高原牆——「起點品質 vs 隨機血統結構性天花板」假說判定。
- **指向**: [round-46](round-46-relay-2.md)（g 系終點/高原條件①）· scratch 07-28 高原三條件

## 1. 假設 (Propose)
- **背景**：d 線（+2.07）與 g 線（+5.32）雙雙止步 −2.7~−3.0 帶;g 線末段 rad −0.28=綁束軸。
  假說 A=牆是「起點品質」造成（rad 負起點爬 wm 時被 rad 拖住）;假說 B=牆是隨機血統結構性天花板。
- **判準（發車前寫死;Ricky 可隨時否決）**：
  - **c47d1 鏈**（tri,group,max 20 包,dry2 終點）:錨=**g46b2_009_oobp_brdg_t0**
    （wm −6.78∧rad −0.02∧lo −5.70∧oob 19.92;R46 名單 19 筆中**非家譜血統**起點最高者——
    w46b3_005 rad +1.46 更漂亮但帶 o26b3 家譜,破牆會混淆血統因子,列備援並標混淆）。
    **破 −2.7**（超越 g 線終點 −2.72）=假說 A 成立（起點品質論);進作戰區（wm≥−1）=重大成果;
    合格=里程碑照公證鐵則;**又停 −2.7~−3.0 帶=第三血統撞牆→假說 B 增證,
    高原條件②測試（GNN bakeoff 提前）升優先,帶帳回報 Ricky**。
  - 首包全撞防呆:換備援錨 g46b1_008_oobp_brdg_t0（wm −8.94∧rad +0.19∧lo −4.87）。
  - **批線**（≤3 批,select-r47=r46 配置）:V 臂常駐;紀錄照公證鐵則;
    G 臂 staging 前置步必跑（R46 b3 教訓,§3 模板）。
- **配額**：批 60×≤3（seed 20260804）;鏈 25/包。
- ★ 修正（2026-07-29 13:0x;**誠實聲明:c47d2 p01/p02 已回**〔−8.05/−7.57,距判定帶尚遠〕,本註=事實更正非門檻更動,判準門檻本體一字不動）：
  ①背景訂正——g 線單一血統止於 −2.72;**d 線止於 wm −7.00（−7 帶,另一道牆,見 docs/chains/c45d1/d2.jsonl）**,「雙雙止步 −2.7~−3.0」為 R46 引入的誤植;
  ②「第三血統」訂正為「**第二條**嘗試該段的血統」（c47d1 廢鏈不計）;
  ③若 c47d2 停 −2.7~−3.0 帶,證據強度=n=2 案例級,**不足以單獨把 GNN bakeoff 優先級拉前——升優先與否併 Ricky 判**。

## 2. 實驗設計 (Design)
| 項 | 設計 | 判準 |
|---|---|---|
| c47d1 | 均衡錨續爬（非家譜,rad≈0） | 破 −2.7=假說 A;作戰區=重大;合格=里程碑公證;dry2+停同帶=假說 B 增證 |
| 批線 | 常態（old6/GDd4） | 五軸;名單續記（lo≤−4∧rad≥−1） |

## 3. 執行紀錄 (Run)
```
# v86 重錨（R46 收輪時已發動）:
python -m script.sm_reanchor train --add "dedust_r46b3a,dedust_r46b3b" --out sm_reanchor86.pth
python -m script.sm_reanchor train-two --out sm_reanchor86.pth
# 鏈線:
python -m script.dedust chain --name c47d1 --anchor g46b2_009_oobp_brdg_t0 \
  --source-input dedust_r46b2b_input --goal tri --anchor-score -6.93 \
  --mutator group --n 25 --prio 1
# 批線（seed 20260804;⚠ select 前必跑 G 臂 staging）:
python -m script.sm_invert gen --sm sm_reanchor86.pth --rad-head rad_head86.pth \
  --n-free 6 --n-surg 0 --n-champ 0 --n-oob 6 --seed 47<批號> --out-dir tmp/invert_stage_r47b<N>
python -m script.dedust select-r47 --batch <N> --sm sm_reanchor86.pth --gstage tmp/invert_stage_r47b<N>
# check-dup ×2 → jobs-add ×2 prio 3 → watch
# ⚠ b1 發車前置（fanout 審視 07-29）:①等 train-two 完成（two86/lohead86/shadow86 落地）再 select——
#   select_r22mix 按 --sm 版號字串配對配套模型,缺件=靜默停鍵（pred_wm_two/pred_lo/pred_wm_cnn）;
#   ②select-r47 必顯式帶 --rad-head rad_head86.pth（parser default=rad_head85,與 sm 86 版本錯配）。
```
| 批/包 | 狀態 |
|---|---|
| c47d1 | ★ 修正（07-29 10:1x,發生在 p02 結果前）:**廢鏈=擇錨口徑事故**——名單誤用 `contrast_lo`,tri 鍵實際門檻=`oob_gain_max_lo ≤ −2`;錨 g46b2_009 該欄位 **+3.49**（盆地外）→ p01 全包 −99。判準本體不變;鏈放任 dry2 自收（p01/p02 共 50 筆照入資料池）。教訓:凡「lo」口徑必寫明欄位 |
| v86 | 重錨完成（07-29 15:0x 全家族）:held-out 中位 **1.202 續創新低**;**凍結尺 1.195→1.10 降**（條件③觀察窗 v87-90 開局向「有斜率」走）;凍結遠 3.01→2.78 回落;rad ρ+0.31;two86 凍結 0.819;lohead86 ρ+0.831;ens86×2 0.915 |
| b1 | 發車（07-29 15:0x,審視防呆全帶:staging 前置+顯式 --rad-head rad_head86）:staging→select-r47→check-dup→prio 3→watch 一鏈 |
| c47d2 | 正錨重發（07-29 10:1x）:**g46b3_000_free_randf（wm −8.10∧lo(gain_max) −3.43∧rad +0.44,G 臂反演無家譜）**——起點深度≈g 線（−8.04）只差 rad 正負=假說最乾淨對照;anchor-score −8.25;正格名單（lo≤−2∧rad≥−1）17 筆/≤−4 8 筆 |

## 4. 分析 (Analyze)
（待）

## 5. 結論 (Conclude)
（待）

## 6. 後續決策 (Next)
- 工具修正候選:analyze batch 影子對決 adv 尺 n/a(0) 應判贏非判輸（R46 §6 帶入）。
- GNN bakeoff 觸發線（pot 唯一 24,959,線 ~30k）;獨立艙凍結續;鏡射 rad 旋鈕候選續。

## 7. 歸檔指向 (Archive)
- 結果夾 `dataset/dedust_r47b*`;鏈帳 docs/chains/c47d1.jsonl。
