# Round 28 — 塊內 rad 手術：half 半成品錨 × 網布凍結 × maximin 鍵

- **狀態**: running（2026-07-14 19:4x 開輪;R27 結構定案的直接應用,宣告制）
- **提出 / 開跑 / 結論**: 2026-07-14 / 2026-07-14 / —
- **一句話問題**: R27 留下五顆「wm＋低側達標、只卡 rad −2~−4」的半成品——在**網布凍結**（一根不動）
  的約束下,塊內手術（挖點/開槽/中帶清理）能不能把 rad 修回 ≥0？修回即三達標,可用帶外直接進
  7.x-8.x 級（half 錨 oob 8.6-12 起跳）。
- **一句話結論 (TL;DR)**: 待跑
- **指向**: [round-27](round-27-mesh-arch.md)（結構定案:塊內=自由度/網布=本體）· champions「學長碎片族
  結構定律」· rad 機理知識=hslot 旋鈕（R17）＋中帶乾淨（analysis-03）＋對稱度（R11）

## 1. 假設 (Propose)

- **證據**：①R27 定案「塊內=自由度」——手術空間合法;②rad 機理三則:hslot=rad 大旋鈕（+1.0~+1.7,R17）、
  高 rad 冠軍中帶乾淨（analysis-03）、對稱度=rad 旋鈕（R11）;③rad 頭復鍵中（前瞻 +0.42~+0.67）
  =手術變體可預篩;④ikpi 首讀 I−M +0.20（工具已落地,R24 三輪債清）。
- **判準（發車前寫死;Ricky 可隨時否決）**：
  - **主判準**：Y 臂任一樣本「rad ≥0 ∧ wm >0 ∧ 低側對比 ≥5」＝**三達標命中**（照鐵則公證;
    其 oob_bad 若 <9.0 即挑戰可用帶外紀錄）。
  - **劑量反應**：rad 改善量 vs 手術像素數 d——有單調帶=可導引,全平=塊內手術對 rad 無槓桿。
  - **停止線**：Y 臂連兩批 rad best 對 half 錨基線（−2.1~−4.1）零改善 → 「塊內手術判死」,帶帳本回報
    （屆時剩餘路=對稱化整局重排——動網布,需新一輪假設）。
  - **批數 ≤3**（回歸硬上限;R27 加厚是單輪特例）。
  - R26 延續帳:D 14（min sel 續帳）/I 18（**每批跑 `analyze ikpi` 記帳**——工具已落地）/
    oob 前瞻續觀察帶/F 8 最小席位（Ricky 裁決點仍開）。
  - 紀錄級一律公證;判準修訂只能在結果回來前＋留註記。
  - **修訂註記（2026-07-14 20:2x,b1 已發車、結果未回）**：Ricky 提「反自餵」——**b2 起加
    `--dyn-simcap 0.35`**（王系相似度稅,d_dyn<12 近王樣本 ≤35%;b1 無此旗標不受影響,決定性不破壞）;
    響應面配額（lo-active 推 50%）留 R29,本輪不再加旋鈕。詳 decisions「反自餵雙軸」。
- **配額（每批 150）**：**Y 36**／O 24／M 20（前瞻母體）／C 10／S 12／D 14／I 18／W 8／F 8。

## 2. 實驗設計 (Design)

- **Y 手術臂**（`select-r28 --surgery 36`,commit c13cd3d）：錨=五顆 half 半成品
  （n27b1_018_p00 +0.42/n27b1_020_t07 +0.35/n27b1_017_n09 +0.20/n27b1_021_t09 −0.02/n27b1_019_t03 −0.39）;
  算子=surg_carve（挖 1-6px）/surg_slot（塊內水平槽 3-8,hslot 知識）/surg_midband（中帶列 10-14 清 2-8px）;
  **全限 blk mask 內＋生成後驗證塊外像素零變動（違者棄）**;鍵=maximin(pred_wm, pred_rad)。
- 其餘沿 r22mix:root-cap 0.6/novelty 0.02/滾動吸收（R23..R27+自產）/--rad-key（復鍵中）/
  --d-sm sm_denovo2（對決四連敗,設計檢討仍欠——本輪閒時）。
- 判讀:analyze batch＋Y 臂「rad vs 錨基線」對照表＋劑量反應散點（收檔手動,工具化候選）。

## 3. 執行紀錄 (Run)
```
# 發車（開發機,conda ant;每批照 /batch-cycle）:
python -m script.dedust select-r28 --batch N --sm sm_reanchor<最新> --rad-head rad_head2.pth --rad-key --novelty --d-sm sm_denovo2.pth
python -m script.dedust check-dup --input dedust_r28bNa_input   # a..f,exit 1=停
python -m script.dedust jobs-add --input dedust_r28bNa_input --store dedust_r28bNa --prio 3   # ×6
# 收檔判讀: analyze batch --round 28 --batch N ＋ Y 對照表 ＋ analyze ikpi --round 28 --batch N --pre v<發車版> --post v<重錨版>
```
| 批 | 狀態 |
|---|---|
| 1（r28b1{a-f}） | 🔵 20:0x 發車（v30;錨點 647;Y36 目檢過=純挖除零新增〔網布凍結成立〕,maximin 鍵偏好 midband;查重 0×6） |
| ⚠ 216 復發 | 20:1x r28b1b 保險絲（認領後開頭連 5 筆 COM 例外 0x80070223,成功 0——**上次重開未治好**）;推播請 Ricky 重開整台;b1b 待機器健康後重派;37/218 正常 |

## 4. 分析 (Analyze)
（待收檔）

## 5. 結論 (Conclude)
（待）

## 6. 後續決策 (Next)
- **塊級承重圖（Ricky 2026-07-14 提議,b1 發車前記入）**：b2 搭載決定性探針——對骨架 8-10 塊逐塊
  消融/縮環/全槽（blk 內合法）,量 Δwm/Δrad/Δ低側=塊 × 三軸重要性表;定位「rad 傷在哪塊」→ b3 知情手術。
  錨=t07h/p00h,~20-40 筆。

## 7. 歸檔指向 (Archive)
- 結果夾: `dataset/dedust_r28b*`;公證 `r28n*`;填空池 `r28g*`。
