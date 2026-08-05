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
  4. 親本 ~10(對角重明星+1-2 王朝零對角 sanity 對照);探針帶 S1 級網格+放寬 timeout
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

## 4. 分析 (Analyze)
（待）

## 5. 結論 (Conclude)
（待）

## 6. 後續決策 (Next)
（待;掛起=GEN 面板/rad 頭三連招/轉正制議案/尺政策——皆待 Ricky 訊號）

## 7. 歸檔指向 (Archive)
- 結果夾 `dedust_r54b*`/`dedust_r54db*`(幾何變體域,永不入鍋);kpi*.csv 續帳。
