# Round 30 — SM 準度輪 2：低側據點擴張 × 窮舉公證 × champ 變現帶

- **狀態**: running（2026-07-16 凌晨開輪;宣告制接棒 R29;戰略換軸第二輪）
- **提出 / 開跑 / 結論**: 2026-07-16 / 2026-07-16 / —
- **一句話問題**: R29 破冰的低側 gap 區（8+ 筆 lo −8~−2 真值,全 wm 爛）——鄰域擴張能不能把
  「lo 深」和「wm 活」拉到同一張 pattern 上？同時:可用帶外 9.0 零推進連 6 批——窮舉公證歷史
  top 候選,它是天花板還是漏網？
- **一句話結論 (TL;DR)**: 待跑
- **指向**: [round-29](round-29-gradient-inversion.md)（gap 破冰/可信半徑 d≤25）· decisions
  「戰略換軸」「反馬太四機制」· analysis-05（r_feed=帶外最強旋鈕 ρ−0.48）· kpi.csv（SM 準度曲線）

## 1. 假設 (Propose)
- **證據**：①R29b3 gap 區 8+ 據點（d29b3_006 lo −8.27/champ_exking −8.82/誤差錨子代 −6.82——
  但 wm −5~−15）;②analysis-05:r_feed（feed 主件金屬佔比）=帶外最強結構旋鈕（ρ −0.48）
  ＝lo 與 wm 的張力有已知的結構載體;③champ 帶連兩批 realized 三標=變現通道;
  ④v36 起 SM 首次見過 gap 區物理（低側導航可能性從 0 變非 0）。
- **判準（發車前寫死;Ricky 可隨時否決）**：
  - **主判準（低側據點擴張）**：L 臂（gap 據點鄰域變異）realized「lo ≤−2 ∧ wm ≥−2」≥2 筆/批
    ——「深低側+近活」的中繼點存在證明;全批 0 筆連兩批=gap 據點是孤島（結構不相容),
    低側戰役回歸碎片族路線。
  - **窮舉公證（X 臂,一次性）**：歷史可用帶外 9.0-9.5 段全部候選（估 4-8 筆）重測 ×2——
    全部復現 ≥9.0 → 「天花板 9.0」寫進 records 註記;任一 <9.0 → 新紀錄照鐵則處理。
  - **G 臂續帳**：champ 帶 realized 三標 ≥1/批（連三批帳）;帶別 adv 率照報（KPI①）;
    rad 頭 +0.621 續鍵（<0.3 連兩批退鍵,帳重計）。
  - **恆溫器回應**：G-free 24→28＋D 12→16（連兩批觸發的配額加碼;來源=O 12→8/C 6→4/S 6→2）。
  - **紀錄門檻**：引 `docs/records.json`（wm 0.56/inband 0.61/usable_oob 9.0）;紀錄級一律公證。
  - **批數 ≤3**;五軸 KPI 面板每批必報;判準修訂只能在結果回來前＋留註記。
- **配額（每批 150）**：**G 64**（free 28〔含碎片/塊 init〕/champ 24/surg 8/oobp 4）／**L 20**（gap
  據點鄰域,d≤15 微調+d16-40 中變異）／**X ~10**（窮舉公證,b1 一次性）／M 14／O 8／I 14／D 16／
  W 10／C 4／S 2（＝150±,b1 含 X;b2 起 X 席回 G/L）。
- **r_feed 分軸 key**（analysis-05 促成,select 端）：L 臂選拔鍵加 r_feed 高者優先（帶外/低側臂
  用高 r_feed=結構先驗第一次進 select）。

## 2. 實驗設計 (Design)
| 臂 | 配額 | 生成 | 判準 |
|---|---|---|---|
| L（低側據點） | 20 | gap 8+ 據點鄰域變異（d≤15 ×12+d16-40 ×8）,r_feed 高者優先 | lo≤−2∧wm≥−2 ≥2/批 |
| X（窮舉公證） | ~10 | 歷史 usable_oob 9.0-9.5 全候選 ×2 重測（kind=repeat,查重豁免） | 全復現→天花板定案 |
| G | 64 | 同 R29 工具鏈,v36 起錨;champ 24 主力 | champ 三標 ≥1/批 |
| M/O/I/D/W/C/S | 68 | 沿 r22mix（dyn-frac 0.4/simcap 0.12） | 對照+資訊帳 |
- 工具:L 臂=select 端新函式（gap 據點清單寫死進 select-r30）;X 臂=select-repeat 批量。
- 判讀:analyze batch（五軸自動）＋L 臂 lo×wm 散點（收檔手動）。

## 3. 執行紀錄 (Run)
```
# 發車（開發機,conda ant;每批照 /batch-cycle）:
python -m script.sm_invert gen --sm sm_reanchor<vN>.pth --out-dir tmp/invert_stage_r30bN --n-free 28 --n-surg 8 --n-champ 24 --n-oob 4 --seed <10+N>
python -m script.dedust select-r30 --batch N --sm sm_reanchor<vN>.pth --gstage tmp/invert_stage_r30bN --rad-key --novelty
python -m script.dedust check-dup --input dedust_r30bNa_input   # a..f 分開跑,exit 1=停
python -m script.dedust jobs-add --input dedust_r30bNa_input --store dedust_r30bNa --prio 3   # ×6
```
| 批 | 狀態 |
|---|---|
| X（r30x1{a-f}） | 🔵 01:0x 發車（窮舉公證 6 候選×2=12 筆,prio 3 先跑;★修訂註記:§1 估 4-8 筆→實掃 88 筆,收斂為「9.04-9.16 未公證前緣段 6 筆」〔發車前,結果未回〕:o25b3_031/m23g1_053/m25b3_011/m26b2_009/g1_048/m25g1_064） |
| 1（r30b1{a-f}） | 🔵 01:2x 發車（v36〔分層 held-out 遠域 2.77→2.53 首降〕;G64+**L20 首航**〔gap 7 錨鄰域,r_feed 鍵〕+恆溫加碼 D16;誤差錨 +8 第二輪;錨點 760;查重 0×6;prio 4 讓 X 先跑） |

## 4. 分析 (Analyze)
（待收檔）

## 5. 結論 (Conclude)
（待）

## 6. 後續決策 (Next)
- lo-active 50% 配額（gap 據點確立後）;sm_denovo 對決檢討（閒時）;帶內 0.61 rad 修復候選。

## 7. 歸檔指向 (Archive)
- 結果夾: `dataset/dedust_r30b*`;公證 `r30n*`。
