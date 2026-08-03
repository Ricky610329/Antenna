# Round 52 — 錨銀行攻堅輪:學長錨×兩軸攻堅鏈 × two 轉正裁決 × 網格判讀

- **狀態**: running（2026-08-03 18:4x 發車;v102 凍結 0.47 平低/two 凍結 0.564/lohead ho ρ+0.826）
- **提出 / 開跑 / 結論**: 2026-08-03 / — / —
- **一句話問題**: 學長錨銀行的同尺複測值(rad +1.20/lo −3.66 兩張天賦牌)能不能用攻堅鏈變現——
  同時裁決 two 轉正(1/2)與 HFSS 網格收斂(真值天花板)。
- **指向**: [round-51](round-51-bridge.md)(lo 生成端沒貨/two 三尺全贏/收割期確認)·
  [round-50](round-50-morphology.md) §1⑤ 錨銀行·R48 兩段式(主流聲錨×px 有效)·
  [proposal-mesh-convergence](../discuss/proposal-mesh-convergence.md)

## 1. 假設 (Propose)

**核心假設**:學長錨=「異形態、同聲音」(雙空間圖:pattern 側 OOD 大陸/response 側嵌合格帶)
→ 依 R48 兩段式條件化定論(主流聲錨×px d1 有效),攻堅鏈應該打得動;lo 深水生成端沒貨(R51 判死)
的解法=直接從有貨的錨出發。

- **判準(發車前寫死;Ricky 可隨時否決)**:
  1. **攻堅鏈兩條**(prio 2,25 筆/包,dry2 收鏈,學費各 ≤3 包):
     - `c52rad1` 錨=e50b2_007_F6161(wm −0.55∧rad +1.20):**首包存活=≥1 筆 rad≥+0.5∧wm≥−0.3**;
       畢業=合格解(wm≥0.15∧rad≥0)且 rad>+0.5(=rad 高身合格,逼近 rad 王帳)。
     - `c52lo1` 錨=e50b2_000_F18644(wm −0.49∧lo −3.66):**首包存活=≥1 筆 lo≤−3.0∧wm≥−0.5**;
       畢業=合格∧lo≤−3.46(=usable_lo 紀錄帶,紀錄門檻見 records.json,公證後才算)。
     - 首包不過=換錨(名單消耗:rad 軸候補 F15032/F9609;lo 軸候補=無→收線回報);
       **兩鏈首包雙敗=回報線**。
  2. **批線 ≤3 批**(select-r52=r51 配置,seed 20260813,--sm/--rad-head 顯式當版;
     負片 20 分層 6 臂/學長消耗制 b1 10+b2 5=名單收尾)。
  3. **two 轉正裁決**:b1 影子對決=第二讀(判準沿 R40:連兩批三尺全贏→轉正換裝;
     現計數 1/2)。轉正=架構換代事件,記 memory。
  4. **網格收斂判讀**(S1/S2 收檔即判,判準=proposal §3.4 四情境表,發車前已寫死):
     S0→S1 |Δf_res| 中位 ≥0.3 GHz 且 S1→S2 變小=網格未收斂根因確認→**停下回報 Ricky**
     (牽動 records buffer/harvest 重放政策,不自作主張)。
  5. 紀錄級一律公證(/notarize);每輪硬上限 3 批;判準發車後修訂=結果回來前+日期註記。
- **修訂紀律**:同上。

## 2. 實驗設計 (Design)

| 項 | 設計 | 判準 |
|---|---|---|
| c52rad1 鏈 | F6161 錨×px 爬山(兩段式) | 首包 rad≥+0.5∧wm≥−0.3;畢業=合格∧rad>+0.5 |
| c52lo1 鏈 | F18644 錨×px 爬山 | 首包 lo≤−3.0∧wm≥−0.5;畢業=合格∧lo≤−3.46 |
| 批線 ×3 | =r51 配置(正30/負20/學長10→5) | 五軸 KPI;two 轉正第二讀在 b1 |
| 網格判讀 | S1/S2 收檔→§3.4 表 | 未收斂→停+回報 |

## 3. 執行紀錄 (Run)

```
# v102(R51 b3 正片入鍋)已於收 R51 時發動;b1 用 v102 配套。
# b1: staging seed 514 → select-r52 --batch 1 --sm sm_reanchor102.pth --rad-head rad_head102.pth
#     → select-neg --round 52 --batch 1 --n 20 --stratify --arms eng,grf_neg,grf_inv,grf_lab,bool_cut,bool_keep
#     → select-senior --round 52 --batch 1 --n 10 → check-dup ×3 → jobs-add prio 3 ×3
#     → echo dedust_r52b1b >> configs/neg_stores.txt → Monitor watch
# 攻堅鏈(goal 鍵發鏈前定案 08-03 18:3x:rad 鍵要求 lo≤−2/lo 鍵要求已合格,皆不合身——
#   F18644 用 tri〔左側會師,錨分 −1.12〕;F6161 註冊新鍵 radq=min(wm−0.15,rad−0.5)〔畢業判準化身,錨分 −0.70〕):
python -m script.dedust chain --name c52rad1 --anchor e50b2_007_F6161 --source-input dedust_r50b2c_input --goal radq --anchor-score -0.70
python -m script.dedust chain --name c52lo1  --anchor e50b2_000_F18644 --source-input dedust_r50b2c_input --goal tri --anchor-score -1.12
# 網格:S1/S2 由 216 釘選續跑(Monitor bx0kmllqe 盯);收檔→判讀腳本(f_res 拋物線內插)
```

| 批/包 | 狀態 |
|---|---|
| 開輪 | 2026-08-03 17:0x(v102 訓練中,完成即發 b1) |

## 4. 分析 (Analyze)
（待）

## 5. 結論 (Conclude)
（待）

## 6. 後續決策 (Next)
（待）

## 7. 歸檔指向 (Archive)
- 結果夾 `dedust_r52b*`/鏈夾 `dedust_c52*`;kpi*.csv 續帳;網格=`dedust_r51ms*`(跨輪寄宿)。
