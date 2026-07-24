# Round 39 — 左側家族化輪：F 臂鄰域變異 × two 絕對值通道 × lo 判別器進鍵

- **狀態**: concluded（2026-07-25 05:4x 收輪,3 批滿）
- **提出 / 開跑 / 結論**: 2026-07-24 / 2026-07-24 / 2026-07-25
- **一句話問題**: 左側合格解首例（c8trip03_01）是**孤點還是一族**？——鄰域變異（R11 穩健化
  方法論）能不能產出 ≥5 筆合格變異體證明「族」成立,並讓 usable_lo/usable_oob 繼續壓？
- **指向**: [round-38](round-38-shadow-two.md)（里程碑/轉正）· records.json（7.78/−2.63）·
  MILESTONES 第八章

## 1. 假設 (Propose)
- **判準（發車前寫死;Ricky 可隨時否決）**：
  - **F 臂家族錨組 16 席**（錨=c8trip03_01〔首例〕/c10trip02_07〔oob 7.12〕/c6tri5p06_21/
    s38s1_18〔oob 6.82〕;d 1-25 梯度;半 ref 半 rej 續帳;diagb 方向性續）:
    **三批合格變異體累計 ≥5 = 「左側合格族」成立**（公證抽驗 1 筆);<2 = 孤點警訊回報。
  - **two 絕對值通道換裝**：F 臂打分已用 two;主 select 通道（pred_wm/LCB 基準）R39 期評估後換
    （ens 仍 MLP 家族=混口徑誠實註記,R40 ens 換代)。
  - **lo 判別器進鍵第二讀**：批前瞻 ρ≥0.5（b1/b2 任一批 + R38b3 的 0.756=連兩批）→ 進鍵
    （select 罰分或 F 臂 gate）。
  - 紀錄:usable_lo 0.5/格公證;usable_oob 續壓（現 7.78）;紀錄級一律公證。
  - selfgen 換種首讀（R38 欠帳）:b1 判讀時算 auto store 新增樣本王朝表型佔比。
  - tri 鏈群（c8/c10/c6tri5）續=家族挖掘機;批 ≤3;五軸面板;修訂註記。
- **配額（批 58=2 夾）**：G 12/F 16（家族）/I 8/M 5/O 3/K 2/D 6/W 6。

## 2. 實驗設計 (Design)
| 項 | 設計 | 判準 |
|---|---|---|
| F 臂家族 | 四錨 d1-25 變異×半ref半rej | 三批合格 ≥5=族成立 |
| two 換裝 | F 臂即用;主通道評估 | 凍結尺+批前瞻不劣化 |
| lo 進鍵 | 第二讀 ρ≥0.5 | 連兩批→進鍵 |
| selfgen 首讀 | b1 判讀時算 | 王朝佔比 <40% 目標 |

## 3. 執行紀錄 (Run)
```
# v63 輕量重錨（response）:
python -m script.sm_reanchor train --add "dedust_r38b3a,dedust_r38b3b" --out sm_reanchor63.pth --no-ens --no-shadow --ds-mode response
# train-two v63（F 臂打分用）:
python -m script.sm_reanchor train-two --out sm_reanchor63.pth
# 批線（seed 100+N）:
python -m script.sm_invert gen --sm sm_reanchor63.pth --rad-head rad_head63.pth --out-dir tmp/invert_stage_r39bN --n-free 6 --n-surg 0 --n-champ 0 --n-oob 6 --seed <100+N>
python -m script.dedust select-r39 --batch N --sm sm_reanchor63.pth --gstage tmp/invert_stage_r39bN --rad-head rad_head63.pth --novelty
# check-dup ×2 → jobs-add ×2 prio 3 → watch
```
| 批 | 狀態 |
|---|---|
| b1 | ✅ 判讀完（07-24）:F 臂合格 0;lohead 批前瞻 ρ 0.756/0.717→**進鍵**（F 臂 gate pred_lo≤−1,commit 757a9df）;two 三尺勝;批內 NN 8 警報→b2 D/W 8/8 恆溫 |
| b2 | ✅ 判讀完（07-25 01:27,62 筆 error 0）:F 臂合格 0（家族帳 1/5,孤點警訊逼近）;**two 四批連勝三尺**（誤差 1.52 vs mlp 2.01,ρ+0.786 vs +0.077,adv 0% vs mlp 73%）;lohead ρ+0.813（連三批）;lo gate 首航 36→22;多樣性連二警報（NN 32/10）→b3 --dyn-simcap 0.08;帕累托+1（w39b2_007）;紀錄零推進;**selfgen 換種首讀:換種後 132 筆王朝 0%**（判準<40%,✅ 換種生效） |
| b3 | ✅ 判讀完（07-25 05:2x,62 筆）:**F 臂合格 0——三批終局 1/5,孤點警訊成立**;two 第五批連勝誤差+ρ（1.32/+0.758 vs mlp 1.46/+0.182;adv 尺 two n/a(0)=零樂觀宣稱）;lohead ρ+0.780（連四批）;I 臂 2 合格（i39b3_006 wm+0.36/rad+0.26 本批王）;三標 7/合格 4;多樣性 NN 36/10（simcap 0.08 微效 32→36,警報未解）;帕累托+0;紀錄零推進 |

## 4. 分析 (Analyze) — 五軸面板
- **① SM 準度（主 KPI）**：held-out v63 1.409→v65 1.396（持平）;two 凍結 0.804→0.817;
  **two 盲測五批連勝誤差+前瞻 ρ**（b2 1.52/+0.786、b3 1.32/+0.758 vs mlp 2.01/+0.077、1.46/+0.182;
  mlp adv 誤報 73%/53% vs two 0%/n·a）;**lohead 批前瞻 ρ 0.756/0.717/0.813/0.780 連四批**;
  G 臂 free 帶 adv 100% 未解（誤差錨持續吸收）。
- **② 覆蓋**：近王 3%/王朝根 11-13%/無親新血 23%;批內 NN 8→32→36 三批警報未解
  （simcap 0.08 微效）;對歷史 NN 10（微調批口徑）。
- **③ 水位**：wm 中位 −3.63→−3.08,P90 +0.17/+0.11,作戰區 21-23/62;帕累托 +1（w39b2_007）。
- **④ 變現**：批線紀錄零推進（usable_oob 7.78/usable_lo −2.63 不動）;
  **鏈線第二筆左側合格解 c6tri6p03_09**（wm+0.18∧rad+0.07∧lo−2.83,單次,07-24）——族的證據來自鏈,非批。
- **⑤ 健康**：error 0/0/0;jobs.json 本輪零事故;selfgen 換種後王朝 0%（132 筆）。

## 5. 結論 (Conclude)
1. **左側家族判準終局 1/5 → 「批線孤點警訊」成立**：F 臂 48 席（四錨 d1-25×半ref半rej×lo gate）
   三批合格 0——**批次鄰域變異這把鏟子挖不出左側合格族**;但鏈線同期挖出第二筆（c6tri6p03_09,
   c6tri5→c6tri6 兩代 d1 爬山）→ 修正結論=**族存在但極窄,只有 d1 爬山搆得到,F 臂撤、家族擴張主力=鏈線**。
2. **two 換裝主通道條件成熟**（五批連勝誤差+ρ,累計 b1-b3+R38 兩批）→ R40 執行（MLP 降 ens 成員）。
3. **lohead 進鍵成功**（連四批 ρ≥0.717）→ 常駐鍵（select 續掛 pred_lo）。
4. selfgen 換種生效（王朝 0%）;多樣性批內 NN 警報三批未解 → 結構性解法=response 空洞反演
   （R40 新臂,decisions 2026-07-25「資料擴展主軸=response 空間」）。

## 6. 後續決策 (Next)
- R40：two 換裝主通道+V 臂（response 空洞反演）+F 臂撤;佇列原子化先行（R40 前欠帳）。
- t07 觸發檢查（radhead2 讀其鄰域 rad 梯度）;獨立艙凍結續;R40 ens 換代候選。
- c6tri8 毫米線續飛（錨 c6tri8p01_18 −0.05,lo −3.64 鄰域）。

## 7. 歸檔指向 (Archive)
- 結果夾 `dataset/dedust_r39b*`;公證 `r39n*`;鏈帳 docs/chains/。
