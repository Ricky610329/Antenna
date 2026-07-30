# Round 48 — 定向嫁接輪：左側引擎 × 王朝骨架 × 批線常態

- **狀態**: running（2026-07-30 17:1x 開輪;自主續輪宣告制;R47 收輪接棒;方向② 觸發條件=c47d2 收鏈已到）
- **提出 / 開跑 / 結論**: 2026-07-30 / 2026-07-30 / —
- **一句話問題**: 定向嫁接（深 lo 引擎件移植進王朝肥沃骨架）能不能把「換盆地」變成可重複的動作
  ——左側合格解從「36hr 戰役孤例」變「生產線」？
- **指向**: [round-47](round-47-relay-3.md)（盆地個體性定調/廣度勝深耕）· decisions「探索方向優先序」·
  analysis-07（深水帶合併橋接=頭獎=物理預驗）· scratch 組義字典（零件清單）

## 1. 假設 (Propose)
- **背景**：王朝合格解全是嫁接產物（樂透嫁接=被動);盆地個體性定調後,嫁接=主動換盆地。
  組義字典給了零件清單（王朝三位一體 92%/左側家族=共用配件換引擎）。
- **判準（發車前寫死;Ricky 可隨時否決）**：
  - **嫁接試點包 g48graft1**（25 筆,prio 2,tier0 式）:生成=組義槽組合——王朝三位一體骨架
    （主件+雙翼,簇質心±2 抖動）× 左側引擎件（從 usable_lo ≤−2 的合格/苗子解抽 XL 主件替換或
    對角引擎件加掛;間距鐵律 gap=2;零對角混入者標記）;決定性 seed;check-dup 必跑。
  - 判定四檔:①**lo 保留率**=嫁接體 oob_gain_max_lo ≤−2 佔比 **≥20%=嫁接機制活**（主判準）;
    ②wm 水位中位 ≥−5=骨架紅利實現（對照:左側家族原生=深水）;
    ③合格（wm≥0.15∧rad≥0∧lo≤−2）=**里程碑照公證鐵則**;
    ④lo 全 >−2=「lo 引擎不可移植」實錘（重要負結果,與可製造化推論互證）。
  - **批線**（≤3 批,select-r48=r47 配置）:V 臂常駐;紀錄照公證鐵則;G 臂 staging 前置必跑;
    --rad-head 顯式當版。
  - **v89 儀器對照**:凍結尺 vs v88=0.528,劣化 >0.06（>10%）→ 退回 40ep/pattern（30ep+response+GPU 三新版驗證）。
  - **速度紅利加倍（Ricky 拍板 07-30「SM 推理遠快於 HFSS」）**:①主漏斗各臂候選池 ×3（O+M+I 72×/C 48×/
    wild 90×;繼 2026-07-16 首次放大後二次）②V 臂池 600→2000 ③G 臂反演 --oversample 3→6。
    **擴散型介入=效率評 over ≥5 輪**（gain L2,decisions 口徑）,不做單批判決;LCB/std 護欄照舊。
    ④雲內爬升（worker 閒時 SM 引導虛擬爬山→只測 top）=R48/49 實作候選,動 worker 端另案。
- **配額**：批 60×≤3（seed 20260805）;嫁接試點 25。

## 2. 實驗設計 (Design)
| 項 | 設計 | 判準 |
|---|---|---|
| g48graft1 | 組義槽嫁接生成 25 筆全測 | lo 保留率 ≥20%=活;合格=里程碑公證;全滅=負結果實錘 |
| 批線 | 常態（old6/GDd4） | 五軸;I 臂強勢續觀察 |
| v89 | 30ep+response+GPU 首版 | 凍結尺對照 v88,劣化 >10% 退回 |

## 3. 執行紀錄 (Run)
```
# v89（R47 收輪時已發動）:
python -m script.sm_reanchor train --add "dedust_r47b3a,dedust_r47b3b" --epochs 30 --out sm_reanchor89.pth
python -m script.sm_reanchor train-two --epochs 30 --out sm_reanchor89.pth
# 嫁接生成器待實作（_graft 槽組合+select-graft 命令;v89 收檔後實作發車,判準先寫死如 §1）
# 批線（seed 20260805;staging 前置）: sm_invert gen --sm sm_reanchor89.pth --rad-head rad_head89.pth \
#   --n-free 6 --n-surg 0 --n-champ 0 --n-oob 6 --seed 48<批號> --out-dir tmp/invert_stage_r48b<N>
# select-r48 --batch <N> --sm sm_reanchor89.pth --rad-head rad_head89.pth --gstage tmp/invert_stage_r48b<N>
```
| 批/包 | 狀態 |
|---|---|
| v89 | 訓練中（07-30 17:0x 發動） |

## 4. 分析 (Analyze)
（待）

## 5. 結論 (Conclude)
（待）

## 6. 後續決策 (Next)
- 訓練框架二刀:複製式→WeightedRandomSampler（~6×;epoch 甜蜜點已定,動核心保 golden）。
- GNN bakeoff（pot 唯一 26,828/線 ~30k）;組圖轉換器準備;獨立艙凍結續;鏡射 rad 旋鈕候選續。

## 7. 歸檔指向 (Archive)
- 結果夾 `dataset/dedust_r48b*`、`dedust_g48graft*`。
