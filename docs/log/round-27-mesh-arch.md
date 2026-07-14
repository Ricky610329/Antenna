# Round 27 — 加厚雙主軸：N 網架臂（骨架＋網布）× R26 延續

- **狀態**: running（2026-07-14 開輪;Ricky 授權「27 可以做厚一點 網架和 26 的延續」）
- **提出 / 開跑 / 結論**: 2026-07-14 / 2026-07-14 / —
- **一句話問題**: 圖 4-4 家族實測=「8-10 密度分塊骨架＋網布」雙層結構且全家族共享一副骨架——
  低側選擇性（學長碎片族對比 6-8.7 dB vs 我方冠軍 0.4 dB）住在骨架還是網布？拆開後能不能
  「保低側＋修 rad」把可用帶外從 9.0 推進 7.x 級？
- **一句話結論 (TL;DR)**: 待跑
- **指向**: [round-26](round-26-oob-foresight.md)（F 收官=盲翻雙軸失敗→結構化修復接棒）·
  scratch 2026-07-14「骨架+網布」塊（分析圖+腳本）· decisions「低側雙判死★適用範圍註」·
  senior_reference_patterns.md（t07=圖4-4 本尊）

## 1. 假設 (Propose)

- **證據**：①密度分割實測（σ0.8×0.6）:池頂家族 8-10 分塊載 66% 金屬＋~110px 網布,t03/t09/n09/p00
  共享同一骨架（質量譜逐塊 ±1px）;②帶外拆側:學長族低側對比 6.1-8.7 dB vs 我方全冠軍 **0.4 dB
  （低側形同全通）**——兩家族各壓一側,t03/t09 oob 7.2-7.4 贏帶外王 8.61;③R18 除塵殺低側＋R26 F 臂
  「修 wm 丟帶外」=網布是低側載體的兩則舊證;④F 臂已證盲翻可把碎片錨 wm 修到 −0.18=修復量存在,
  缺的是結構知識。
- **判準（發車前寫死;Ricky 可隨時否決）**：
  - **主判準（H2 拆解）**：N 臂 solidify_full（刪網布）vs solidify_half（保網布）同錨對照——
    半凝聚低側對比顯著優於全凝聚（≥3 錨方向一致）＝「低側住網布」;反向=住骨架;
    兩式皆保 5+ dB 對比＝住粗質量分布（最好情境:直接可製造化）。
  - **H1（骨架承載）**：mesh_redust 變體 wm/oob 對原錨漂移中位 ≤1 dB＝像素實現是自由度。
  - **獎品線**：任一 N 臂樣本「低側對比 ≥5 且 wm>−1 且 rad>−1」＝修復路線開張（→定向修復批）;
    可用帶外新前緣照鐵則公證。
  - **R26 延續**：D 14 加碼（KPI=min sel 續帳,基線 51.7-78.2）;I 22（KPI 量測工具本輪必辦,不再滾）;
    rad 頭復鍵=續記前瞻連兩批 ≥0.3（R26b3 +0.668 已 1 批,R27b1 ≥0.3 即 b2 復鍵）;
    oob 前瞻續觀察帶（M 臂續記,跨輪帳）;F 0（退役,精神由 N 繼承——Ricky 可否決）。
  - **批數**：**≤5 批**（Ricky 加厚授權,2026-07-14;3 批上限他輪照舊）。
  - 停止/回報線:可用帶外連 3 批零推進→重驗屍（第二層已用掉:本輪就是換分布;再紅=第三層證天花板進議程）;
    一般臂連兩批 <6% 收臂;紀錄級一律公證;判準修訂只能在結果回來前＋留註記。
- **配額（每批 150）**：O 26／M 20（前瞻母體）／C 14／S 14／**N 24**／D 14／I 22／W 8／F 0。

## 2. 實驗設計 (Design)

- **N 網架臂**（`select-r27 --mesh 24`,commit f080c38 實裝+t07 目檢）：錨=FRAG 11 顆
  （D 產物 6＋池頂族 5）;四式=solidify_full（塊填實+刪網布,天然可製造）/solidify_half（塊實+網布留）
  /mesh_uniform（等距棋盤網布=密度 vs 實現）/mesh_redust×3（H1 同骨架重抽）;
  選拔=按錨輪流+變體優先序（full→half→uniform→redust,H2 對照先進場）;
  ⚠ 物理探測批不過可製造閘（塵=實驗變數,同 D 慣例）。
- 其餘臂沿 r22mix 機器:root-cap 0.6、novelty 0.02、滾動吸收（R23..R26+自產）、--d-sm sm_denovo2
  （對決三連敗,R27 內檢討設計:改 vs 現任+擴 held-out+降 over）、不帶 --rad-key。
- 判讀:N 臂加「同錨 full vs half 低側對比表」（analyze batch 通用臂別外,收檔時手動補拆側——
  工具化候選,用過兩次再收）。

## 3. 執行紀錄 (Run)
```
# 發車（開發機,conda ant;每批照 /batch-cycle）:
python -m script.dedust select-r27 --batch N --sm sm_reanchor<最新> --rad-head rad_head2.pth --novelty --d-sm sm_denovo2.pth
python -m script.dedust check-dup --input dedust_r27bNa_input   # a..f,exit 1=停
python -m script.dedust jobs-add --input dedust_r27bNa_input --store dedust_r27bNa --prio 3   # ×6
# 收檔: dedust watch;判讀: analyze batch --round 27 --batch N ＋ N 臂拆側對照
# 每批收檔後: sm_reanchor train --add ... --out vN+1（train-denovo 待對決設計檢討後再議）
```
| 批 | 狀態 |
|---|---|
| 1（r27b1{a-f}） | 🔵 待發車（等 v28＋denovo6 訓完） |

## 4. 分析 (Analyze)
（待收檔）

## 5. 結論 (Conclude)
（待）

## 6. 後續決策 (Next)
（待）

## 7. 歸檔指向 (Archive)
- 結果夾: `dataset/dedust_r27b*`;公證 `r27n*`;填空池 `r27g*`。
