# Round 54 — 菱形橋輪:45° 顯式對角橋 × 六點橋寬轉移曲線 × 批線常態

- **狀態**: proposed（2026-08-05 15:4x 開輪;計畫=Ricky 核准之 plan「45° 對角橋」;R53 收輪接棒）
- **提出 / 開跑 / 結論**: 2026-08-05 / — / —
- **一句話問題**: 把對角接點的 0.01mm 意外微橋換成可設計的 45° 菱形橋(w_d 可製造尺寸),
  電性保得住嗎——保住=左側/lo 票倉整個可製造化;保不住=顯式橋判死,低對角軸原路走。
- **指向**: 核准計畫(plans/bright-leaping-rain.md 頂部)·[round-52](round-52-anchor-assault.md)
  r52dx 探針(兩端點:斷角全滅/全邊橋毀)·diffsim §45(0.01mm 角落重疊=真導體)·
  [round-53](round-53-lowdiag-left.md)(rad 高原=牆)

## 1. 假設 (Propose)

**核心假設**:對角電性活在窄橋區間(0.01↔0.2mm 之間存在轉折點)——菱形階梯掃出轉折點位置,
可製造帶(0.075-0.10)在轉折點左側=大勝(對角改「顯式橋規範」),右側=重要負結果(顯式橋判死)。

- **判準(發車前寫死;細節=核准計畫,此處為判定條款)**:
  1. **橋寬階梯** w_d ∈ {0.05, 0.075, 0.10, 0.14}mm+既有端點 0.01(現況帳)/0.2(r52dx)=六點曲線。
  2. **「可製造對角」成立**=存在 w_d 使 |Δwm|≤0.3∧lo/rad 保帶,**且 0.075 與 0.10 兩點同時保帶**
     (蝕刻 ±0.02mm 公差內無懸崖);成立→decisions 低對角左側軸修訂(對角=顯式橋規範)。
  3. **全尺寸毀**=0.01mm 特有效應蓋章→顯式橋判死,低對角軸照舊。
  4. 親本 13(對角重明星 8+**t07 系列 5**〔Ricky 08-05 指示:x52t07_add25/add50·c53lot2p02_07/p03_20·t09_top——軸產線實際資產的可製造化測試〕);探針帶 S1 級網格+放寬 timeout
     (hfss_setup.json);kind=diagbridge 查重豁免;**資料零入鍋**(幾何變體域)。
  5. 碰撞規則:同空格多菱形間隙<0.05mm→同步縮至淨距,縮不下→跳過+manifest 記帳。
  6. 批線照常 ≤3 批(select-r54=r53 配置,seed 20260821);紀錄公證鐵則;3 批必收輪。
- **修訂紀律**:結果回來前+日期註記。

## 2. 實驗設計 (Design)

| 項 | 設計 | 判準 |
|---|---|---|
| 菱形探針 | ~10 親×4 尺寸(~40 sims,S1 網格,prio 4) | §1②③ 六點曲線判定 |
| 批線 ×3 | =r53 配置(正30/負20,diagb-pen 6.0) | 五軸 KPI |
| sanity | 王朝零對角親=幾何 no-op | 應≈原值 |

## 3. 執行紀錄 (Run)

```
# 前置:single_port.py(菱形)+dedust.py(diag_bridge_w 鍵/diagbridge 豁免/select-diagbridge)
#   → commit push → **三台 pull+重啟**(菱形探針發車前置;批線不等重啟照跑)
# 正式機首測:only_create_project 幾何目檢 1 筆 → 王朝 sanity 1 筆全跑 → 探針全批
# 探針: select-diagbridge --ids-file tmp/diagbridge_ids.txt --sizes 50,75,100,140
#   → 四夾(dedust_r54db{50,75,100,140})各帶 hfss_setup.json(diag_bridge_w+S1 網格+timeout 2400)
#   → check-dup ×4 → jobs-add prio 4 ×4
# 批線 b1: v108(train --add r53b3a→two)→staging seed 520→select-r54 --batch 1 --diagb-pen 6.0
#   →neg→查重→prio 3→Monitor
```

| 批/包 | 狀態 |
|---|---|
| 開輪 | 2026-08-05 15:4x(code 實作中) |
| code | ✅ 15:5x 全落地+390 tests 綠(菱形幾何/select-diagbridge/豁免/select-r54)commit push |
| 探針生成 | ✅ 16:3x 四夾重生成含 t07 系擴充:**13 親×4 尺寸=52 筆**(283 橋/夾;0.14 檔縮橋 128 座);待三台重啟發車 |
| v108+b1 | 16:0x 鏈啟動(train --add r53b3a→two→staging 520→select-r54 b1) |
| 探針發車 | ✅ 16:5x Ricky 核准:db100 prio 2 前導+db50/75/140 prio 3;幾何渲染目檢過(diamond_geometry.png,t07=23 座) |

## 4. 分析 (Analyze)

### smoke 首讀(2026-08-05 17:2x;S0 尺,單次;Ricky 手測機 2)
- **王 c48nq1p05_16+12 座菱形(w=0.10)**:wm +0.79→**+0.58**(Δ−0.21,判準帶 ≤0.3 內,仍合格)、
  rad +0.20→**+0.37(反升)**、lo 4.36→4.13——**顯式橋≈溫和擾動,非毀滅**(對照 r52dx 全邊橋 Δ−4~−25)。
- COM Rotate/Move 實戰通過(23 座建模零例外);S1 單筆 15-40 分證實(t 帶+菱形網格重)。
- **判準修訂(17:4x,結果回來前)**:加開 **S0 雙生四夾**(同 52 筆,S0 網格)——量「S0 差分 vs S1 差分」
  相關;相關高→大規模驗證(合格+近合格全體)改 S0 差分篩,S1 只用於終選/公證。
  理由=差分測量對網格噪音天然免疫(smoke 王單點示範)+S1 硬掃不可規模化(Ricky 17:3x 提出)。

### 全量驗證戰役(2026-08-05 18:1x;Ricky 指示「探索全停,過標+近過標全測至少 S0」)
- **探索暫停實體化**:v108+b1 鏈停(b1 未發車;v108 权重已落地,恢復時注意 clean_stores 已含 r53b3a
  勿重複 --add)、r51 橋池 25 夾移出佇列(快照=`jobs_state/PAUSED_r51pools.json`,jobs.json 備份同夾)、
  三台進行中橋池 claim 改持有人觸發優雅退出(讓位機制反用;tier2-prio 預設 8,prio 4 不自動讓位=根因)。
- **戰役名單**(margin 同一把尺,S0 全史帳):合格=wm≥0.15∧rad≥0 → **1208**、
  近合格=wm≥0∧rad≥−0.5 → **2111**;排除變體店(meshconv/db/smoke/s1base)、`~` id、探針 13 親;
  99.9% 含 ≥1 真對角(零接點免測僅 2 筆)。
- **發車**:`dedust_r54valq01..13`(合格,prio 3,wm 降序)+`dedust_r54valn01..22`(近合格,prio 4);
  id=`{parent}~db100`、kind=diagbridge、hfss_setup={w=0.10, timeout 1800, S0 網格};
  估 3319×~6min≈4-6 天(三台)。**S1 探針 db50/75/140 降 prio 6**(S0 差分為主判;
  S1 交叉檢查=db100〔隊外,機1 手測〕+s1base 4 親基線+t03/t05/t09 既有 S1 原值=7 對)。
- **s1base 夾**(4 親 S1 原值基線:t07_top/王/x52t07_add25/F18644,無菱形):prio 3,補 S1 Δ 對照缺口。

### S1 首筆(2026-08-05 18:0x;單次)
- `t07_top~db100`(S1,23 菱):wm −0.93/rad −2.82/lo −4.09——落在 t 帶 S1 正常水位
  (兄弟 t03/t05/t09 S1 原值 wm −0.66~−0.71),非毀滅型;t07 本人 S1 原值待 s1base。

## 5. 結論 (Conclude)
（待）

## 6. 後續決策 (Next)
- **★斷開態(Ricky 2026-08-05 發車時指示;18:1x 升級=戰役後下一棒)**:「有對角相鄰但不相連」=第三態——
  現況幾何強制相連(0.01mm 重疊),真斷開需角落開槽(Subtract 小盒切斷點接觸)。
  保留 `diag_bridge_w=0` 語義=斷開模式;消融=三態(斷開/現況 0.01/菱形)全有全無對照。
- **渲染債(Ricky 18:1x)**:S0 雙生/戰役收檔判讀腳本要出「舊 pixel pattern vs 菱形版幾何+response 疊圖」
  每親一版;HFSS 視窗不渲染=code 未變(DispatchEx+RestoreWindow 一直是帶視窗模式),
  最可能=worker 這次由遠端/非互動 shell 啟動,視窗活在看不見的 session——下次重啟時從機台本機桌面
  終端啟動 worker 可驗證(best-effort 結案)。
- 掛起=GEN 面板/rad 頭三連招/轉正制議案/尺政策——皆待 Ricky 訊號。

## 7. 歸檔指向 (Archive)
- 結果夾 `dedust_r54b*`/`dedust_r54db*`(幾何變體域,永不入鍋);kpi*.csv 續帳。
