# Round 18 — 帶外戰役二批：挖礦落地（舊藏公證＋低側家族救援）

- **狀態**: proposed  <!-- proposed | running | analyzing | concluded | archived -->
- **提出 / 開跑 / 結論**: 2026-07-09 / — / —
- **一句話問題**: 挖礦出土的三個方向——b20_k4 等舊藏是真的嗎？池頂低側家族能構造化救活嗎？c18 手術能破 9.04 嗎？
- **一句話結論 (TL;DR)**: 待跑
- **指向**: [analysis-03](analysis-03-history-mining.md)（方向來源）· `select-r18` · 前作 [round-17](round-17-oob-primary.md)

> 本檔只放**連結指向**其他層,不複製內容。

## 1. 假設 (Propose)
- 承 analysis-03 三發現：①舊藏未公證高價值（b20_k4 +0.32/9.56 等四筆,皆可製造）;②池頂低側家族
  （lo −1.7~−4.5、t09/t03 oob 7.2/7.4 破 9 地板）唯 rad/製造全滅;③c18_sm=三標內帶外紀錄 9.04（跨店雙響應公證）。
- **悲觀先驗（誠實預註冊）**：S 臂 9 錨點中 t07/t03/p00 的既有變體白撿數據顯示——對稱化**保低側殺帶內**
  （wm −15~−25）、除塵**殺低側**（lo→+0.3~+2.1）＝粉塵疑為低側乾淨的載體（R7 定律帶外版）。
  剩 6 錨點（t09/n09/t08/t04/t11/t14）未試,本批終審。
- **判準（發車前寫死）**：
  - **V 臂**：b20_k4 三次中位 wm ≥ +0.30 ⇒ margin 王挑戰成立（與 R17 的 a024/i12 公證結果同場對決）;
    vb43 兩次一致 oob ≤ 9.1 ⇒ 帶外榜首易主（vs c18_sm 9.04）。
  - **S 臂**：任一變體 wm ≥ −1 **且** lo ≤ +1 ⇒「低側可構造化」成立 → R19 rad 救援接手;
    全滅 ⇒ **低側乾淨=粉塵諧振本體**定案（negative result 收檔,低側正式定為 w17 體質+判準拆側議題升 decisions）。
  - **T 臂**：任一 c18 手術三標過且 oob < 9.04 ⇒ 帶外紀錄更新;手術 Δ 方向與 R17（c25/x00）一致 ⇒ 手術規則普適。

## 2. 實驗設計 (Design)
| 臂 | kind | 筆數 | 內容 | 對照 |
|---|---|---|---|---|
| V 舊藏公證 | notarize | 8 | b20_k4×3＋vpc18_f_d2×2＋vb43_a1_d0×2＋x20_a00k8×1 | 鐵則:紀錄級一律重複量測 |
| S 低側救援 | lowrescue | 20 | 池頂低側家族 9 錨 × {sym10, sym12, 純除塵}（7 個變體與歷史重複=白撿舊值,見 §1） | 錨點原始值（lo −1.7~−4.5）＋s05 劇本 |
| T c18 手術 | surgery | 6 | vslot{2,5,8}＋colcut{1,2}＋hslot{3} @c18_sm（對稱不變已驗證） | c18_sm 9.04＋R17 同手術 |
- **HFSS 預算**: 34 筆 ≈1.5–2hr @218。批次原則 ≥30 ✓。
- 查重 ✓（26 新筆零重複;notarize 8 豁免）。

## 3. 執行紀錄 (Run)
```
# 開發機(已做): python -m script.dedust select-r18 && python -m script.dedust check-dup --input dedust_r18_input
# 218: python -m script.dedust run --input dedust_r18_input --store dedust_r18
# 進度: python -m script.dedust report --input dedust_r18_input --store dedust_r18
```
| 臂 | 機器 | 狀態 | 結果夾 |
|---|---|---|---|
| V+S+T | 218 | 待發 | `dataset/dedust_r18` |

## 4. 分析 (Analyze)
（待收檔）

## 5. 結論 (Conclude)
（待分析）

## 6. 後續決策 (Next)
- S 臂活 → R19 rad 救援（對稱化/小塊 rad 旋鈕上低側生還者）;死 → 帶外線收斂到高側精修＋判準拆側討論。
- V 臂公證結果 → champions.md 重排（b20_k4 若成立=margin 共冠級）。

## 7. 歸檔指向 (Archive)
- 結果夾: `dataset/dedust_r18` · analysis-03 · ONGOING 動作:（收檔時補）
