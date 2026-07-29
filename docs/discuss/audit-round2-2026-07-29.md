# Fanout 第二輪存檔（2026-07-29,收網輪:5 查證+4 執行,9×Opus）

> 第一輪=discuss/audit-fanout-2026-07-29.md。本輪=對第一輪高信心未驗證項的獨立查證+四個零 HFSS 分析的完整執行結果。

## Verify（5 條判定）

### lr-plateau → **partial**（severity medium）

【機制＝實錘（三重證據）】

① 程式碼位置（逐行核對成立）
- `C:/Users/Ricky/Documents/GitHub/Antenna/antenna/models/surrogates.py:124` 起是 batch 迴圈；`:144` 在迴圈**內**呼叫 `self.step(scheduler_param=loss)`。
- `C:/Users/Ricky/Documents/GitHub/Antenna/antenna/models/shell.py:190-192`：`step()` ＝ `optimizer.step()` ＋ `scheduler.step(scheduler_param)` → 兩者綁在同一次呼叫，scheduler 因此是 **per-batch**。
- epoch 級的 `avg_epoch_loss` 在 `surrogates.py:150` 才算出來，且**從未餵給 scheduler**。
- 三個 SM 工廠參數一致（`surrogates.py:453-455` mlp／`:503-505` cnn／`:553-555` cnn2）：`lr=1e-3, factor=0.5, patience=10, min_lr=1e-6` → 1e-3 砍到 1e-6 需 10 次減半、每次至少 `patience+1=11` 個 bad step ＝ **最少 110 個 batch** 就能觸底。

② NAS checkpoint 取證（唯讀，script＝scratchpad/inspect_ckpt.py）
| 檔 | optimizer lr | scheduler last_epoch |
|---|---|---|
| sm_harvest.pth（暖啟動起點） | **1e-3** | 0（best=inf，全新） |
| sm_reanchor84.pth | **1e-6** | 35880 ＝ 40×897 |
| sm_reanchor85.pth | **1e-6** | 94560 ＝ 40×2364 |
| sm_reanchor86.pth | **1e-6** | 94400 ＝ 40×2360 |
| sm_two85.pth (cnn2) | **1e-6** | 23360 |
| sm_shadow85.pth (cnn) | **1e-6** | 189120 ＝ 80×2364 |
`last_epoch` 全部＝「epochs × 每 epoch batch 數」→ 直接證實 `scheduler.step()` 是每個 batch 一次；6/6 落在 1e-6 地板。

③ 實跑復現（真資料、真暖啟動；scratchpad/lr_trace.py、lr_heldout.py，只寫 scratchpad、零 NAS 寫入）
資料＝NAS 9 個 clean store 去重 813 筆 ＋ harvest 重放 2000 筆，over=8，batch=64，起點 sm_harvest.pth（同 sm_reanchor train 路徑），CPU。
- **現況（per-batch）**：lr 於 **step 135～153** 首次撞 1e-6（我的設定 132 batch/epoch ≈ 第 1.0～1.2 個 epoch；**正式線 2360 batch/epoch → 落在第 1 個 epoch 的前 6%**，比主張的「1-2 epoch」更早）。
  epoch_loss（20 epoch）：11.325 → 9.459 → 9.445 → … → **9.307**（第 2 epoch 之後每輪只降 ~0.008）。
- **對照（改成 per-epoch step，其他不動）**：lr 全程維持 1e-3；epoch_loss 10.971 → 7.473 → … → **1.752**。
→ 20 epoch 後訓練 loss 差 **5.3 倍**；「地板之後訓練預算幾乎空轉」＝成立（後 19 個 epoch 貢獻 <2% 的可達進步）。

【「主 KPI 儀器被廢」＝反駁（這是主張被高估的一半）】
同一實跑，用**真 KPI 口徑**（held-out |wm 誤差|，決定性 hash 每 5 筆切分；harvest 另留 500 筆驗證遺忘）：
| | held-out 中位 | held-out p90 | harvest 驗證中位 |
|---|---|---|---|
| 起點 sm_harvest | 3.606 | 5.906 | 2.106 |
| 現況 per-batch | **0.561** | 4.981 | 2.388 |
| 修法 per-epoch | **0.580** | **3.724** | **1.963** |
中位數**沒有差異**（修法甚至略差 0.02）。也就是說：held-out 中位在**第 1 個 epoch 就飽和**，後面 39 個 epoch 本來就不是中位數的瓶頸。修法真正買到的是**尾部**（p90 −25%）與**抗遺忘**（harvest −18%）。

【kpi.csv 為何仍在動 → 解釋】
不是矛盾，兩件事同時成立：
1. 每一版重錨都 `pre_load_model(sm_harvest.pth, strict=True)`（`script/sm_reanchor.py:213`），而 sm_harvest 的 optimizer lr **是乾淨的 1e-3**、scheduler 物件是新建的（best=inf, last_epoch=0）→ **崩塌不會跨版累積**，每版都真的拿到「第一個 epoch」的有效訓練。
2. 那一個 epoch 已足夠把中位打到飽和（3.61→0.56 在我的復現裡發生在 epoch 1）。
→ 所以 kpi.csv 的每版數字是真的、儀器沒壞；但它量到的其實是「1 epoch 訓練 + 當版資料集」的結果，**KPI 的版間變動來自資料集成長與洗牌噪音，不是來自 40 epoch 的訓練預算**。也正因如此，`frozen_med` 從 v39 到 v86（近 50 版）在 1.0～1.4 之間橫盤、沒有斜率——而**修掉這個 bug 也不會讓 frozen_med 出現斜率**（實測中位不動）。停滯要另找原因，不能記在這條帳上。

【附帶觀察（未實測，僅程式碼閱讀）】
`surrogates.py:239`（`train_one_data`）與 `:322`（`train_one_data_rad`）也是每個內層迭代 step 一次 scheduler，且線上迴圈只在 `antenna/training.py:1267` 開跑前 `reset_online_lr()` 一次、之後每筆之間**不重置** → 線上路徑跑幾筆後 lr 同樣永久停在 1e-6。此路徑目前非生產主線（批次線在跑），我沒實測，只標記位置。

**修法**：最小侵入，只動一個檔、兩行（能刪不要包，不加參數、不加旗標）：

`antenna/models/surrogates.py`（`train_by_datas`）
1. `:144` 改成只走 optimizer：
   `self.step(scheduler_param=loss)` → `self.optimizer.step()`
   （`shell.step` 的 optimizer_param 傳給 Ranger 的是 closure，原本就是 None，語義不變。）
2. `:150-152` 之後（算完 `avg_epoch_loss`、寫入 `epoch_loss` 之後）補一行：
   `if self.scheduler: self.scheduler.step(avg_epoch_loss)`
   → patience=10 的單位回到 epoch，40 epoch 最多減半 3 次。

不必動的：`EnsembleSurrogate.train_by_datas`（`:619` 只是 fan-out 給成員，自動繼承）；三個工廠的 `patience/factor/min_lr` 維持原值；`reset_online_lr` 維持原樣（`train_one_data` 路徑不在本次範圍，另案）。

落地前要驗的兩件事（我沒改 repo，所以沒跑）：
- `python -m pytest tests/ -q` 全綠＋golden 零漂移。預期安全：`tests/fixtures/*.yaml` 的 `sm_train` 沒設 mode → 走 default `single`（只呼叫 `train_one_data`），golden loop 不經過 `train_by_datas`；但 `prepare_models` 的離線預訓路徑（`training.py:1257`）會經過，仍須實跑確認。
- 副作用：修好之後 epoch_loss 會持續下降 → `train_by_datas` 的 early_stop（patience=epochs/2）不再提早觸發，重錨會**跑滿 40 epoch**（v84 目前只跑了 ~15 epoch 就停）。開發機每版重錨的 wall-clock 會變長，發車前先確認排程吃得下。

期望收益（依實測，n=1 seed／20 epoch／813 筆 clean，勿外推）：held-out **中位不會改善**，收的是 p90（4.98→3.72）與 harvest 抗遺忘（2.39→1.96）。若目標是推動 `frozen_med`，這條不是解藥。

**侷限**：1. **n 小、單一 seed**：復現用 813 筆 clean（正式線 v86 是 6244 筆）、20 epoch（正式 40）、torch seed 固定 0 只跑一次、無密度反權重（`_build_ds` 的 reps）。held-out 中位 0.56 vs 正式 kpi 的 1.1~1.4 不可直接比（我的 held-out 只來自 9 個 store，考卷較簡單）。「p90 −25%／harvest −18%」是**單次觀測**，不是統計顯著結論；要當發車依據應至少 3 seed 重跑。
2. **未實測項**：(a) 修法在正式規模（2360 batch/epoch × 40 epoch）的實際 held-out 表現；(b) `train_one_data` 線上路徑的 lr 崩塌（只讀程式碼，沒跑）；(c) 修完的 golden 漂移狀況（依鐵則我沒改 repo 任何檔，無法就地驗）。
3. **反面可能性（誠實列出）**：per-epoch 版在 20 epoch 內 lr 完全沒退火（一路 1e-3），最終權重是「沒 anneal 過」的；正式線跑滿 40 epoch 後可能出現過擬合或末期震盪，屆時 held-out 中位有可能**變差**。這是把 patience 從 batch 換到 epoch 的真實風險，不能只看訓練 loss 降得漂亮。
4. **審視存檔那條的措辭要修正**：「主 KPI 的儀器被廢掉」不成立——儀器沒壞、每版讀數是真的；壞掉的是 39/40 的訓練預算與尾部品質。把 KPI 十幾版沒斜率歸因到這條，會導向錯誤的下一步。
5. 全程唯讀：repo 零修改、NAS 零寫入、無 git 操作；中間產物在 scratchpad（`inspect_ckpt.py`、`lr_trace.py`、`lr_heldout.py`、`lr_trace.json`、`lr_heldout.json`、`data_cache.pt`、`smcache/`）。`smcache/` 是 SurrogateModel 建構要求的 rootdir，全程沒呼叫 save，實際為空。

### rad-falsy → **partial**（severity low）

## 結論一句話
falsy 陷阱**確實存在且在 live 路徑上**（主張成立），但「漏掉貼線三標解」的**量級遠小於審視條目的語氣**——1.59% 錨池稀釋、17 筆、且無任何一筆是失落的紀錄候選。故判 partial：機制為真，衝擊被高估。

## 一、逐處判定（全 repo falsy 哨兵掃描）

| # | 位置 | 函式 | 0.0 是否合法 | 是否 live | 判定 |
|---|---|---|---|---|---|
| 1 | `script/dedust.py:2943` | `select_r22mix` | **是** | **是（R22–R47 全部）** | **REAL，唯一有實害** |
| 2 | `script/dedust.py:2712` | `select_r21harvest` | 是 | 否（硬寫死 `dedust_r21b*`） | 同型 bug，死路徑 |
| 3 | `script/dedust.py:2466` | `select_r20gen` | 是 | 否（R20 世代 GA） | 同型 bug，死路徑 |
| 4 | `script/analyze.py:825` | `_auto_scan` | 是 | 是 | **無行為差異**（見下） |
| 5 | `script/analyze.py:1082` | 紀錄候選 | 是 | 是 | **無行為差異** |
| 6 | `script/figs/report_sprint48.py:96` | 一次性圖 | 是 | 否（過期 sprint 圖） | 純美觀 |
| 7 | `script/dedust.py:4058/4064` | `pred_wm_cnn/two or -99` | 理論上是 | 是 | **實測 0 次**，見下 |
| 8 | `script/dedust.py:2944/2946` | `(oob_bad or 99)` | 是 | 是 | 實測 3 次，全部無關 |
| 9 | `script/dedust.py:5718` | `r.get("rad_margin",-1)` | — | 是 | **不是 bug**（預設只在缺鍵時生效） |

**live 路徑確認**：`grep` argparse 顯示 `select-r22` 到 `select-r47` 全部 `s.set_defaults(fn=select_r22mix, round=NN)`（dedust.py:5815…6694），R45/R46/R47 皆然 → 站點 #1 每次選批都會執行。

**#4/#5 為何無實害（必須誠實扣掉）**：`analyze.py:817` 的 tri 判定用的是**正確**的 `is not None` 寫法，825 在其下游；而它比的是 `rec["rad"]["value"]`，`docs/records.json` 現值 **rad = 1.0**。`-9 > 1.0` 與 `0.0 > 1.0` 同為 False → 兩種寫法輸出完全一致。1082 同理。這兩處是潛伏、不是現行缺陷，**不該當成修好的東西報**。

**#7 為何無實害**：`pred_wm_cnn`/`pred_wm_two` 確實也走 `_r()` 兩位小數（寫在 3909/3919），0.00 理論可達；但掃全部 manifest.json（26,260 筆）實測 **pred_wm_cnn 0/2310、pred_wm_two 0/1218 恰為 0.00**。（`pred_wm` 有 13/9914 個 0.00，但它沒有被 falsy 保護包住。）

**#8 為何無實害**：`oob_bad == 0.0` 全史 3 筆（`a216_00990`／`a218_00198`／`g35b1_009_free_randf`），wm 分別 −8.08／−8.41／−12.79，離 `wm ≥ 0` 前置閘十萬八千里。且該子句被 `wm>=0.15 or …` 短路保護。

## 二、0.00 確為合法值（不是事實上的哨兵）

`rad_margin = min(cuts.values())`（dedust.py:5099），cuts 每個都是 `_r(rad_window_margin(...))` = round 到 2 位（dedust.py:648）→ 落在 0.01 網格。全 NAS 掃描（586 店、25,529 筆有 `wm` 的 entry）：

- `rad_margin` 有值率 = **25,529 / 25,529 = 100%**（此鍵幾乎不會缺）
- `rad_margin` 恰 0.00 = **99 筆（0.388%）**
- 0 附近網格分布：`−0.06:98  −0.05:103  −0.04:105  −0.03:103  −0.02:114  −0.01:109  **0.00:99**  +0.01:116  +0.02:128  +0.03:135  +0.04:109  +0.05:116  +0.06:133`

**這是最關鍵的一項證據**：0.00 這一格的密度（99）完全落在鄰格帶（98–135）之內，Poisson 噪聲量級內無異常。也就是說 0.00 是一個平凡、完全可達的網格值，程式卻把它當「沒有資料」處理。

**額外一層陰險**：17 筆漏掉的裡有 12 筆值是 **`-0.0`**（`round()` 對極小負數會產出負零）。Python 裡 `-0.0 >= 0` 是 **True**（正確語義下算三標過），但 `-0.0` 同樣 falsy → 一樣被吃掉。

## 三、實際影響面（分層收斂）

99 筆 rad==0.00
→ 其中 **25 筆** 同時 `wm[2] >= 0`（正確語義=三標過，buggy 判 False）
→ 其中 **20 筆** 再過 ANCH 閘 `(wm>=0.15 or oob_bad<9.5)`
→ 落在 `select-r45` 實際掃描範圍內（`dedust_r23..r44*` + `dedust_auto*` + 硬寫死 r21/r22 清單，排除 `_input/_src` 與公證店 `r\d+n`）= **17 筆**

複刻 dedust.py:2920-2946 吸收邏輯（含真 `POISON = ("g2_029","t14","vg0795")`）實算：

| | buggy 錨池 | 正確語義錨池 | 漏 |
|---|---|---|---|
| select-r45 | 1050 | 1067 | **17（1.59%）** |
| select-r47 | 1069 | 1086 | **17（1.57%）** |

漏掉的 17 筆（store, id, wm, rad, oob_bad, oob_gain_max_lo）：
```
r21b2c m2_044_o1_045_vg039      0.07  0.0  9.26 3.73
r22b1d h6_005_a024              0.34 -0.0 12.21 4.01
r23b1b s23b1_012_a024           0.27 -0.0 12.47 3.98
r24b1a s24b1_013_a024           0.19  0.0 11.21 3.90
r25b1c k25b1_010_o23b2_009_s8   0.43 -0.0 15.35 4.10
r25b3d o25b3_009_o23b2_001_s8   0.38 -0.0 10.50 4.02
r25g1b m25g1_043_o1_045_vg039   0.07 -0.0  9.21 3.73
r26b1a s26b1_012_g16            0.25 -0.0 15.49 4.02
r26b1e m26b1_014_o1_003_g1_03   0.13 -0.0  9.44 3.82
r26g1b m26g1_040_o1_033_g1_03   0.15  0.0  9.70 3.82
r27b2e i27b2_004_i25b2_006_o2   0.33 -0.0 12.71 4.02
r31b1d m31b1_013_k24b1_017_s8   0.26  0.0 13.46 4.02
r33b1e i33b1_010_i30b2_005_i2   0.38 -0.0 10.71 4.08
r34b2e m34b2_008_i33b1_012_i3   0.40  0.0 14.99 4.21
r41b1b i41b1_006_i30b2_005_i2   0.48  0.0 11.18 4.19
r42b3b i42b3_008_i31b1_005_i3   0.40 -0.0 10.38 4.10
r44b1a m44b1_001_m7_003_g16     0.38  0.0 12.81 4.02
```

## 四、必須扣分的三點（為什麼是 partial 不是 real）

1. **沒有失落的紀錄**。17 筆全部 oob_bad 9.21–15.49（現行 oob 王 8.61）、wm 0.07–0.48（現行 margin 王 0.73）；`oob_gain_max_lo` 除一筆外全是 **+3.7~+4.2（右側）**，不在「左側主戰場」（門票 `lo ≤ −2`）。這些是普通的池中錨，不是被吃掉的冠軍。審視條目暗示的「貼線合格解被永久埋沒」在資料上撐不起來。
2. **資料沒有丟，只丟了「當錨」的資格**。`store.add(p, resp)`（dedust.py:5103）是無條件的，`clean_stores.txt` 是店層級清單（已確認含 `dedust_r41b1b`/`r44b1a` 等）→ 這 17 筆**照常進 SM 訓練集**。所以審視原文「等於永遠不會被拿來當錨開採」正確，但不等於資料損失。
3. **只有 1/3 的 dedust 站點是活的**。2466/2712 分屬 `select_r20gen`/`select_r21harvest`，已無任何 round 指向；把三處並列會誇大暴露面。

## 五、仍值得修的理由（不是為了 1.59%）
偏差是**有方向性的**：被吃掉的永遠是「rad 剛好貼在門檻上」那一類，而 rad 正是全系統的綁束軸（memory：左側家族 rad_margin +0.03~0.07 全靠弱側貼線）。一個佐證——現行 margin 王 `m42b1_003_o26b3_022_o2`（records.json wm 0.73）的 rad 是 **+0.01**，距離這個洞剛好一個 0.01 網格。

## 六、測試覆蓋
`tests/` 對此謂詞**零覆蓋**：`grep` 全 tests 無 `ANCH`／`select_r22mix`／錨點吸收相關斷言；`test_dedust.py`（380 行）只測到 `rad_window_margin` 本身（第 94/100/106 行），沒測下游 tri 閘。

**修法**：最小侵入，**只動 1 個檔、淨減行數**。（本次為唯讀查證，以下未套用。）

**① 抽出重複謂詞（3 份 → 1 份，是刪不是包）**
`script/dedust.py` 模組層（建議緊接 `sel_score` 之後，約 line 648 附近）加 3 行：
```python
def _tri(r):
    """三標過（wm≥0 ∧ rad≥0）。rad_margin 走 _r() 兩位網格,0.00/-0.0 是合法合格值——
    不可用 `or -9`（falsy 會把貼線解判成缺件）。"""
    rm = r.get("rad_margin")
    return r["wm"][2] >= 0 and (rm if rm is not None else -9) >= 0
```
然後 **dedust.py:2466 / 2712 / 2943** 三行各改成 `tri = _tri(r)`。淨效果：+4 行 −3 行，且消掉三份複製。不加類別、不加參數、不動呼叫端結構。

**② 同行的 `(r.get("oob_bad") or 99)`（2944 / 2946）**
順手改成 `(r.get("oob_bad") if r.get("oob_bad") is not None else 99)`。**但在 commit / round 檔裡不要宣稱這修好了什麼**——實測全史只有 3 筆 `oob_bad==0.0` 且 wm 皆 ≈ −8~−13，零行為差異。純粹是不留同型地雷。

**③ 2466 / 2712 是死碼——先問要不要刪**
`select_r20gen` / `select_r21harvest` 已無 argparse 指向。照 CLAUDE.md「能刪不要包」，**優先選項是整個函式刪掉**（連同 ①的兩處改動一起省掉）；若 Ricky 要留作歷史存根，才套用 `_tri`。這個決定請 Ricky 拍板，不要 Claude 自作主張刪 select 家族。

**④ 回歸測試（照 tests/CLAUDE.md「每修一個 bug 補一條」）**
`tests/test_dedust.py` 加一個參數化測試，純函式、零 NAS、零 HFSS：
```python
@pytest.mark.parametrize("rad, wm, want", [
    (0.0,   0.20, True),   # ★ 本 bug 的靶心：貼線合格
    (-0.0,  0.20, True),   # ★ round() 產出的負零（17 筆裡佔 12 筆）
    (0.01,  0.20, True),
    (-0.01, 0.20, False),
    (None,  0.20, False),  # 缺方向圖 → 不算過
    (0.0,  -0.01, False),  # rad 過但 wm 不過
])
def test_tri_treats_zero_rad_as_pass(rad, wm, want):
    from script.dedust import _tri
    r = {"wm": [0.0, 0.0, wm]}
    if rad is not None:
        r["rad_margin"] = rad
    assert _tri(r) is want

def test_tri_missing_rad_key():
    from script.dedust import _tri
    assert _tri({"wm": [0.0, 0.0, 0.5]}) is False
```

**⑤ `script/analyze.py:825` / `:1082` — 統一寫法可以，但別記帳成修 bug**
兩處都可換成 `_tri` 同款寫法保持一致，但已驗證在現行 `records.json`（rad 王 = 1.0）下輸出完全相同。commit message 誠實寫「統一 falsy 哨兵寫法（無行為變更）」。

**⑥ 不要做的事**
- 不要碰 `dedust.py:5718` 的 `r.get("rad_margin", -1)`——那個是對的。
- 不要碰 `4058/4064` 的 `pred_wm_* or -99`——實測 0/2310 與 0/1218，改它是零收益的噪音。
- 不要回頭重跑歷史批或補發錨點包燒 HFSS 機時：17 筆全是普通池中錨（無紀錄候選），且資料早已進 SM 訓練集。修完讓它從下一次 `select-rNN` 自然生效即可。

**⑦ 落地時機與護欄**
批次線生產中，建議搭在下一次 `/close-round` 的 commit-cycle 順手做，不單開一輪。改完跑 `python -m pytest tests/ -q`（repo 根）確認全綠、golden 零漂移——`_tri` 不進核心 `antenna/`，golden 不應有任何位移。

**侷限**：1. **唯讀執行**：全程未改 repo／NAS 任何檔，未跑 select／jobs-add／chain／train／git，未跑 pytest。fix_plan 是提案、未驗證套用後行為，也未實測 golden。中間產物只落在 scratchpad（`scan_rad0.py`／`scope_live.py`／`scan_side.py`）。

2. **錨池數字是複刻不是實跑**：1050 / 1067 / 17 來自我在 scratchpad 複刻 `select_r22mix` 的 `prev` 清單構造與吸收迴圈（含真 `POISON`），非呼叫 `select` 本體。已對齊的部分：掃描前綴（`dedust_r21b*`/`r21g*`/`r22b*`/`r22g1*` 硬寫死清單 + `rnd>=25` 的 `dedust_r{23..rnd-1}*`/`dedust_auto*` 滾動吸收）、`_input`/`_src`/公證店 `dedust_r\d+n` 排除、POISON 過濾、`if name not in P` 名字去重。**未複刻**：`loadp()` 實際載入（若某 `_input` 夾缺件，真實池會比我算的更小）、ELITE/SPEC/FRAG 等硬寫死錨組、以及各臂配額。所以 1.59% 應視為量級估計（±數個百分點的分母不確定性），**17 這個分子是硬數字**（直接來自 results.json 逐筆判定，可重跑驗證）。

3. **`select-r46/r47` 的 `--batch` 未知**：`prev` 含 `dedust_r{rnd}b{1..batch-1}*`，我固定用 batch=1..3（每輪硬上限 3 批）。若實際發車時 batch 較小，本輪內自產的部分會少算，但不影響 17 筆（全部來自往輪，不是本輪）。

4. **只查了 `rad_margin` 這一族 falsy**：我掃的是 `or -9`／`or -99`／`or 0.0`／`or -1e`／`or -inf` 與全部 `rad_margin` 出現處。**未系統掃描**其他可能的 falsy 陷阱型別（例如 `or 0`、`if not x`、`x or default` 用在其他 0 可達的量上，如 `contrast_*`、`rolloff_*`、`removed_px`、`pred_std`）。若要根治這一類，需要另開一次針對性掃描。

5. **`docs/chains/*.jsonl` 未納入影響評估**：`_chain_score`（dedust.py:4571-4573）用的是**正確**的 `r = -9.0 if r is None else r`，所以鏈線不受本 bug 影響；但審視存檔另有一條關於 `−99 哨兵語義過載` 的獨立發現（同一段程式的不同問題），**不在本次查證範圍**，我沒有驗證它。

6. **n 的誠實聲明**：靶心樣本 n=17（漏掉的錨）／n=25（被誤判的三標）／n=99（rad==0.00 全體）。「無失落紀錄」這個結論建立在對這 17 筆的 wm/oob/lo 逐筆檢視上，樣本小但**是母體全查不是抽樣**，故此結論可靠；而「這 17 筆若入池會不會產出更好的後代」則**無法從現有資料判定**——那需要反事實，我沒有下這個結論。

7. **`report_sprint48.py:96` 未追查是否仍在產線**：我依檔名（sprint48 一次性圖）判為過期，未驗證 `script/figs/` 是否有排程或被 `round_report` 呼叫。若它其實仍在跑，該圖的三標計數會系統性少報約 0.4%×（wm≥0 比例）。

### notarize-train → **partial**（severity low）

【方法】全普查,非抽樣。用 os.scandir 盤點 NAS 上 587 個 dedust_* store 共 25,419 個 .pt(檔名=sha1(x,y) 前 16 碼),再全量 torch.load 算 x-bytes / y-bytes 指紋,並在腳本內**逐行複刻** script/sm_reanchor.py:44 `_load_clean_stores()`＋:80 `_load_clean()` 的優先序與去重(檔內順序→glob dedust_auto*→glob dedust_c*;store 內 sorted(*.pt);key=x.tobytes() 首見即贏)。中間檔全部落在 scratchpad,repo/NAS 零寫入(不走 SampleStore,因其 __init__ 會 mkdir + unlink *.tmp)。複刻結果訓練集唯一 x = 24,497,與 docs/kpi.csv 最新列 heldout_n=6,244(≈24,497/5 + 凍結)吻合,可信。

【屬實 1 — 機制成立,0/23 勝出】NAS 上符合公證命名的 store 共 22 家(r20n,r22n1,r22n2o/h/w,r22n3,r23n1r/w,r23n2a,r24n1a,r24n2a/b,r25n1a,r26n1a,r28n1a/b,r29n1a,r36n1,r38n1,r41n1,r42n1,r42n2a),樣本共 23 筆(r20n 有 2 筆,其餘各 1 筆——因 3/3 重測的兩次 repeat 逐位元相同,store hash 去重後只剩一檔;已用 results.json 交叉確認,如 dedust_r42n2a 的 r00_rep/r01_rep 兩筆 wm 都是 [1.11,0.73,0.73])。**23 筆全部沒有一筆是去重勝出者(KEPT=0)**,每一筆的 x 都已被更前面的店先占。清單內那 8 家公證店對訓練集的實際貢獻各為 0 筆。

【屬實 2 — 清單漏接,且違反已寫明的 runbook】configs/clean_stores.txt 只有 8 家公證店(行 48/77/81-83/92/108-109),NAS 上另外 14 家(r23n2a 起到 r42n2a)完全不在清單。而 .claude/skills/batch-cycle/SKILL.md:32 白紙黑字「公證店(rNNnX)與收完的填空池也一併 --add」——這是流程違反,不是模糊地帶。同時 sm_reanchor.py:199-208 的 `--add` 確實只 append 檔尾(工作區未 commit 的 diff 就是 +dedust_r46b3a/b 追加在最後三行),而 r22b1a 在行 71、r22n1 在行 77,審視存檔說的「批次店緊接在公證店之前」屬實。

【推翻 1 — 「全數被原始單次量測蓋掉、對訓練零貢獻」高估了】23 筆裡 **15 筆的遮蔽者是逐位元相同的檔**(同一個 sha1(x,y) 檔名同時出現在公證店與批次/鏈店,如 r42n2a/9c8c5dbd = r42b1a/9c8c5dbd、r41n1 = c41grp2_p02、r24n1a = r24b1d…)。也就是公證重測復現了原量測的**完全相同 response**,誰先誰後對訓練集毫無差別,零資訊損失。真正「值不一樣」的只有 8 筆。

【推翻 2 — 那 8 筆的落差在重測雜訊等級,不是假象】逐筆用 antenna.losses.worst_margin 兩邊都算過(公證值 → 訓練實際採用值,Δ=採用−公證):
  r20n 0.386→0.386 (Δ0.000, |Δy|max 0.0038) / r22n1 0.136→0.028 (Δ−0.108, max 0.152) / r22n2h 0.351→0.431 (Δ+0.080, max 0.080) / r22n2o −0.003→−0.003 (Δ0.000, max 0.021) / r23n1r 0.090→0.090 (Δ0.000, max 0.009) / r26n1a 0.436→0.527 (Δ+0.091, max 0.091) / r29n1a 0.564→0.564 (Δ0.000, max 0.011) / r36n1 0.203→0.203 (Δ0.000, max 0.003)。
最大 wm 標籤誤差 0.108 dB,最大單點 response 誤差 0.152 dB;對照 SM 自身 wm_err_med≈1.20 dB(kpi.csv 2026-07-29 v86)。**8 筆 / 24,497 筆 = 0.03% 的資料,帶約 1/10 模型誤差的標籤噪音**。全 22 次公證裡沒有任何一次的重測值與原量測有大落差 → 這條路徑至今沒有把任何「假象」烘進訓練集。

【推翻 3 — 「不變式已失效」的範圍被誇大,舊世代仍然有效】那句程式碼註解(sm_reanchor.py:36-37)原本要解的是 ref2 的 Gain 污染,對應的 verify 店 dedust_ref2v/champ_disc/verify_interp/verify_disc2/w17rep 現在仍然穩坐清單**最前五名(優先序 0-4)**,而 dedust_ref2 在 12、ref1 在 10。實測:這五家參與的同 x 不同 y 衝突共 14 組,**verify 店 14/14 全勝**。所以「certified 先見先贏」對它當初寫來要防的世代**完全成立**;失效的只有 2026-07-13 之後 notarize 線新生的 rNNnX 世代。全域同 x 不同 y 的衝突共 49 組。

【推翻 4 — 審視存檔的行動②單獨做是純 no-op】23 筆公證樣本的 x **全部**已存在於清單內的其他店,沒有一筆是新 pattern(LOST-NEW=0)。所以「把漏掉的 14 家補進 clean_stores.txt」若不搭配前移,新增樣本數 = 0、訓練集完全不變;搭配前移後,唯一效果是翻轉上述 8 個標籤(≤0.11 dB)。同理,skill 現在要求的「公證店一併 --add」在現行載入邏輯下本身就是空操作——這是這條發現真正該修的點。

【caveat】dedust_tol_src 三個檔(a15_k4/c21_sm/w17_k8.pt)torch.load 失敗,但該 store 不在 CLEAN 清單,與本案無關。

**修法**：目標不是救那 8 個標籤(值不了 0.11 dB),而是讓「公證優先」這句話從註解變成事實,並消掉 skill 裡那條空操作——避免未來真的抓到大落差假象時被靜默覆蓋。三步,不加抽象層:

① script/sm_reanchor.py `_load_clean_stores()`,在 return 前(glob 之後)插兩行穩定前移:
    import re
    out.sort(key=lambda s: 0 if re.match(r"^dedust_r\d+n", s) else 1)   # 公證店前移;list.sort 穩定,其餘順序不動
放在 glob 之後可順便涵蓋未來補進清單的公證店;不要放在 glob 之前,否則 dedust_c*/auto* 的相對位置語意會變。

② 把 NAS 上漏接的 14 家補進 configs/clean_stores.txt(dedust_r23n2a, r24n1a, r24n2a, r24n2b, r25n1a, r26n1a, r28n1a, r28n1b, r29n1a, r36n1, r38n1, r41n1, r42n1, r42n2a),加一行註解標明是補登、非新資料。**若不做 ①,這步請直接跳過**——單獨做等於什麼都沒做。

③ 補一條回歸測試(tests/,monkeypatch DATASET_PATH,兩個暫存 store 同 pattern 不同 response,斷言 _load_clean() 取到公證店的 y),鎖住這個不變式。零 NAS 依賴。

不要做的事:審視存檔的行動④「修完重錨一版比對 kpi」——8/24,497 筆、≤0.11 dB 的標籤變動遠低於重錨的 run-to-run 抖動,任何 kpi 差異都是雜訊,拿它當修復生效的證據會製造假訊號。要驗收就驗 ③ 的測試綠。

若判斷連保險都不想留,替代的更小做法是反向:刪掉 clean_stores.txt 裡那 8 行無作用的公證店、刪掉 sm_reanchor.py:36-37 已不成立的那半句註解、把 SKILL.md:32 的「公證店一併 --add」拿掉。兩條路二選一,最差的是維持現狀(註解與 skill 都宣稱有效、實際無效)。

所有改動都碰 configs/clean_stores.txt 與 script/,依 docs/report/handoff-direction-doc.md:55 屬批次線 session 獨佔範圍,且批次線生產中——請由批次線 session 執行,不要由旁支 session 動。

**侷限**：① 本次為全普查(22 家公證店 / 23 筆樣本 / 25,419 檔全掃),不是抽樣,但 n 本身就小:真正有爭議的樣本只有 8 筆,任何以「這 8 筆」為基礎的統計推論都撐不住。② 我複刻了 _load_clean 的去重邏輯而非直接呼叫它(避免 SampleStore.__init__ 對 NAS 的 mkdir/unlink 寫入);複刻正確性的旁證是唯一 x 數 24,497 與 kpi.csv heldout_n=6,244 的 1/5+凍結關係吻合,但這是間接驗證,不是逐位元對拍。③ 「至今沒有假象被烘進訓練集」只涵蓋走 notarize 線、且 store 落在 NAS 上的 22 次公證;若有公證結果只寫進 records.json 而沒留 store,本次掃不到。④ 沒有實跑重錨驗證修復後的 kpi 差異(鐵則禁止,且如上所述那個實驗本身沒有鑑別力)。

### checkdup-exit → **partial**（severity medium）

機制主張=完全屬實;「歷史上真的發生過」=查證後否定(從未發生)。故判 partial。

【A. exit 語義混同——屬實,已實證】
C:/Users/Ricky/Documents/GitHub/Antenna/script/dedust.py:4225-4258 `check_dup`:
- 全函式**零 try/except**(對比 chain:4617、_preload_used:4650 兩處都有)。`load_folder` 的 `json.load(manifest)` 與 `torch.load(*.pt)` 全裸。
- 查到重複 → 4258 `raise SystemExit(1)`;讀檔失敗 → 未捕捉例外 → CPython 同樣 exit 1。
- 實測(唯讀,對不存在的夾跑,秒退不掃 NAS):`check-dup --input dedust_NO_SUCH_FOLDER_input` → **exit code = 1**,與「查到重複」逐位元相同。
- 全檔唯一的數字化 exit 就是 4258 那個 `SystemExit(1)`,其餘皆 `SystemExit("訊息")`(也是 1)——等於根本沒有 exit code 命名空間。

【B. chain 吞 stderr——屬實】
dedust.py:4757-4761:`subprocess.run(..., capture_output=True)`,而 `cd.stdout`/`cd.stderr` 在 6,894 行全檔 **零引用**(grep 只有 4759 的 `cd.returncode`)。非 0 一律印「⚠ 查重撞歷史——重抽下一包」→ `continue`。
`continue` **不加 dry**(dry 只在 4842 收檔後 +1),所以唯一剩下的停止條件是 `max_packs`(預設 20,6188)。4843 的收鏈訊息看起來與正常收鏈無異,且該包不寫 jsonl。

【C. 暴露面(量化)】
唯讀實測 NAS:611 個 `*_input` 夾、26,260 筆 manifest。torch.load 實測 **24.4 ms/筆 → 單次 check-dup ≈ 10.7 分鐘、約 26,900 次無防護的 SMB 讀**,而且**每包都跑一次**。(與 scratch.md:568 健檢③「check-dup/analyze 省 10-25 分/批」對得上。)

【D. 歷史上是否真的發生過 → 否。27 個鏈孤兒全部歸因完畢】
唯讀盤點(scratchpad/orphans.py):全 NAS 29 個孤兒 input 夾 / 692 筆 = 查重母體的 2.6% 從未量測;其中 27 個是鏈孤兒。
逐夾重演 check_dup 判定(scratchpad/orphan_probe.py + lineage_probe.py,只用 manifest.json mtime 早於該包的夾當歷史):
- c41grp p01-p20(2026-07-25 18:06→19:56,1h50m 燒完 max_packs,零 HFSS,無 jsonl):**每包都真的有跨夾重複**,且撞數隨包號單調成長 1→13——因為前面失敗包的孤兒夾留在 NAS,成為下一包的歷史,形成**自我維持的撞擊風暴**。commit b891250 的診斷(組級算子產 d1 等價、used 只擋 px 半)得到獨立證實。
- c41grp2 p04/p05:各 1、2 筆真重複。
- c10tri p01/p04、c5tri p03、c7tri p03:各**恰好 1 筆**,且全是「反向翻回自己的祖先」(如 `c10trip04_08 == c9trip02_18`、`c5trip03_04 == c2radp10_21`)。同錨鄰域比對抓不到是因為祖先的 source_id 不是本錨。
- c6tri5 p08:**零重複**——對應 scratch.md「2026-07-24 jobs.json 並發壞檔第二起…c6tri5 daemon 死」。那個崩潰在 **4769 行 `json.load(jobs.json)`(同樣無防護)**,不是 check-dup。
結論:26 個 = check_dup 真陽性,1 個 = 別條路的崩潰。**沒有任何一次是 check-dup 崩了被誤讀成撞歷史。**

【E. 但故障類別已在本系統實證過】
①NAS JSON 壞檔 2 起(scratch.md:jobs.json 並發壞檔,2026-07-22 與 07-24),其中一起**確實把 chain daemon 打死**;②「燒完 max_packs、零 HFSS、安靜收鏈」的結果形態已發生兩次(c2rad p13-p20,scratch.md:563;c41grp p01-p20),兩次都是事後才診斷出來。所以觸發條件與爆炸半徑都不是假想的,只是還沒落在 check_dup 這一行上。

主張裡「最壞情況…帳面上看起來像鄰域枯竭=一個錯誤的科學結論」的推論鏈成立;「找歷史上是否真的發生過」的答案是**沒有**。

**修法**：最小侵入:只改 chain 一處,不新增旗標、不動 exit code 編號、不需同步任何文件。

理由:審視存檔提的 `SystemExit(2)` 方案會把「exit 1 就別發車」的對外約定反轉,需連帶改 4 處文件(dedust.py:9 檔頭、script/CLAUDE.md §1、.claude/skills/batch-cycle/SKILL.md:40、docs/discuss/decisions.md:124,另 analyze.py:1258 也印這句)。更便宜的判別式已經現成:check_dup 在 4256 行**無條件**先印摘要行 `f"{args.input}: {len(new)} 筆,重複 {bad}"` 才 raise——**stdout 沒有這行就代表沒跑到結尾＝崩了**。

dedust.py:4757-4761 改為(淨 +3 行):

    cd = subprocess.run([sys.executable, "-m", "script.dedust", "check-dup",
                         "--input", store + "_input"], capture_output=True, text=True)
    if cd.returncode != 0:
        if f"{store}_input:" not in cd.stdout:     # 沒印摘要行＝check-dup 自己炸了,不是撞歷史
            raise SystemExit(f"{store} check-dup 異常退出 rc={cd.returncode}\n"
                             f"{cd.stdout[-500:]}\n{cd.stderr[-500:]}")
        print(f"⚠ {store} 查重撞歷史——重抽下一包（不發車）")
        continue

停鏈而非重試是對的:NAS 讀不到,後面 jobs-add／等收檔／讀 results 全都做不了。

同批順手(同一個故障類別,而且**這條已經真的炸過一次**——c6tri5 p08):4769 行 `json.load(jobs.json)` 同樣無防護,包成 try/except 後 `continue` 進下一次 `_try` 重試即可(該迴圈本來就是為 jobs.json 互踩設計的),1 行。優先級其實比 check-dup 那條高。

不建議做:在 check_dup 內對歷史夾加 try/except 跳過壞夾——那會讓查重在殘缺母體上「通過」,把大聲的崩潰換成安靜的漏放,比現況更糟。

**侷限**：1. **判定的證據強度不對稱**:機制(A/B/C)是 code 直讀＋實測 exit code,鐵證;「從未發生過」(D)是重建推論——重建的歷史母體用 manifest.json 的 mtime 當「當時已存在」的近似,且是拿**今天**的檔案內容重演,無法排除當時存在、後來被刪的夾(例如 c2rad p13-p20 的 8 個 input 夾今天已不在 NAS 上,無法重演,只能採信 scratch.md 的當時診斷)。
2. **同錨鄰域比對是充分不必要**:orphan_probe 只掃同錨衍生夾,找到撞位即可證真陽性;找不到的 5 個才用血統爬升補完。我沒有建立全史 hash 索引(起了背景掃描,實測 ~11 分鐘且與生產／其他 agent 搶 NAS,跑 20 分鐘仍卡在 0/611,已 TaskStop)。所以嚴格說是「27 個孤兒全部找到了真陽性解釋」,不是「證明了不存在任何反例」。
3. **順帶發現(不在委託範圍,但有直接證據)**:換錨後 `used = _preload_used(best_id)`(4837)不含「造出這個錨的那個 px」,鏈會反向翻回父 pattern → 必撞歷史。因為 check-dup 是**整包全有全無**,1 筆重複就報廢整包 25 張——c10tri p01/p04、c5tri p03、c7tri p03 四包(100 張生成)就是這樣沒的。修法很小(換錨時把 anchor 的來源 px 塞進 used),但屬另一條發現,未列入 fix_plan。
4. 嚴重度給 medium 不給 high:沒有資料汙染、沒有錯誤量測、沒燒掉 HFSS 機時(什麼都沒發車),且從未觸發;但 daemon 是無人看顧連跑數天、故障類別已實證、爆炸半徑(20 包全燒＋20 個孤兒夾永久進查重母體)也已實證。
5. 全程唯讀:repo 零改動、NAS 零寫入,中間產物都在 scratchpad(chain_audit.py／orphans.py／c41grp_probe.py／orphan_probe.py／lineage_probe.py)。唯一執行過的 repo 指令是對不存在夾名的 check-dup(4235 行即拋例外,不掃 NAS、不寫檔)。

### chain-allout → **real**（severity medium）

## 程式碼事實（script/dedust.py）

`_chain_score` 定義在 **4566-4590**；chain 收檔段在 **4780-4842**。關鍵三行：

- `4781-4783`：`scored = [(k, _chain_score(v, args.goal), v) …]` → `best_id, best_s, best_v = max(scored, key=lambda t: t[1])`。**全部同分 −99 時 `max()` 回傳迭代序第一筆**（Python max 取第一個極大值），且對「全出局」零判別。
- `4784`：`win = anchor_score is None or best_s > anchor_score` —— 沒有 `best_s > -98` 這一項。
- `4785-4789`：`rec` **無條件**寫 `best=best_id, wm=…, rad=…, oob=…, lo=…, hi=…`，沒有任何 `gated`/`out_of_basin` 欄位或告警。
- `6183-6184` CLI：`--anchor-score … help="錨的已知 score（首包 baseline;不給=首包必換錨）"` → 不給時 `win=True` 恆成立，錨會被換成一筆 −99 pattern。
- `tests/` 全域 grep `_chain_score` / `chain(` **零命中** → 零測試覆蓋，確認。

## 資料驗證（NAS 唯讀；我把 `_chain_score` 逐行複製到 scratchpad 重算）

**保真度先驗**：對 `docs/chains/*.jsonl` 全部 **142 包**重算，**142/142 都能反推出唯一 goal 並複現記錄的 `best_score`（誤差 <0.02）** → 我的複製版與線上版等價，以下數字可信。

**主張 (b)「全 −99 包被當正常包記帳」——確認，且比審視寫的更糟：**

| | c47d1 p01 | c47d1 p02 |
|---|---|---|
| n_scored | 25 | 25 |
| 相異分數集合 | `[-99.0]` | `[-99.0]` |
| `max()` 選中 | `c47d1p01_00`（＝迭代序第一筆） | `c47d1p02_00`（同） |
| jsonl 記的 best / wm | `c47d1p01_00` / −6.94 | `c47d1p02_00` / −7.15 |
| **該包真正最好的一筆**（若 gate 有過） | `c47d1p01_14`：wm **−6.52**、rad **+0.87** | `c47d1p02_01`：wm −6.57、rad −0.45 |

記進帳的 `c47d1p01_00` **在 wm 與 rad 兩軸上都輸給同包的 `_14`** —— 不只是「任意一筆」，是可證明的非最佳筆。

**全史發生率**：142 包中 `best_score <= -98` 者 **恰好 2 包**（皆 c47d1）＝1.4%。次高是 `c4lo` p01 **92% 出局**（23/25 −99，2 筆倖存 → 記到真 best）——貼著這個邊緣跑過。

**`--anchor-score` 缺省風險**：142 包中 `anchor_score=null` 者 **0 筆** → 該路徑至今**從未被觸發**，是潛在坑非已實現損害。

**下游消費者**：`docs/chains/*.jsonl` **無任何程式讀取**（grep script/ 只有 chain 自己寫入）。污染只透過人／agent 讀帳傳播——但本次 fanout 審視自己就跑了全帳統計（「140 包鏈帳」「17 對接棒鏈假停校準」「深/中/近 pack 勝率 0.61/0.53/0.38」），這類 meta 分析正是受害者。

## 主張「−99 語義過載」——成立但機制被講錯了一半

`-99` 確實同時代表兩件事：**gate 不過**（門檻外）與**gate 欄位缺件**（`oob_gain_max_lo` 為 None → tri/rad 直接 −99；`oob_gain_max_{lo,hi}` 為 None → lo/hi 直接 −99）。這半成立。

但審視寫的「`r = -9.0 if rad_margin is None` 會讓缺 rad 的樣本**一律** gate 成 −99」**不精確**：
- 只有 goal=**lo/hi** 成立（`r >= 0` 判 False → −99）。
- goal=**tri/rad** 缺 rad → 得到 **−9.0**，不是 −99。而 −9 **大於** −99 → 在一個全出局的包裡，一筆「rad 萃取失敗」的樣本會**贏下整包、成為新錨**（且 `win` 只要 anchor_score < −9 就成立）。這是比審視所述**更嚴重**的方向：不是「儀器壞了被記成軸枯竭」，是「儀器壞了被記成勝步」。
- goal=**wm** 缺 rad → 靜默扣 1.0 dB（把「缺件」當「rad 崩」）。

**實測發生率＝0**：全 NAS **25,531** 筆已評分結果中 `rad_margin is None` **0 筆**；`oob_gain_max_lo is None` 969 筆但**全在 17 個舊店**（r8/r9/ref1-3/wide/tol/occl…，該欄位上線前），**鏈店 0 筆**。所以缺件半是純程式路徑風險，零實例。

## c47d1 根因（順帶釘死，支撐 fix_plan Cut B）

實際發車指令（round-47 §3:40）是 `--anchor g46b2_009_oobp_brdg_t0 --source-input dedust_r46b2b_input --goal tri --anchor-score -6.93`。

該錨在 `dedust_r46b2b/results.json` 的真值：`wm −6.78 / rad −0.02 / oob_gain_max_lo **+3.49** / contrast_lo −5.70`。

- 對它跑 `_chain_score(row, "tri")` → **恰好 −99.0**（lo 閘門 +3.49 > −2）。
- 操作者手打的 −6.93 ＝ `min(−6.78−0.15, −0.02)` ＝ **假設閘門有過**的分數 → 證實是誤用 `contrast_lo −5.70` 當 lo 欄位。
- 對照組：c47d2 的錨 `g46b3_000_free_randf`（`dedust_r46b3a`，wm −8.10 / rad +0.44 / gain_max_lo −3.43）→ `_chain_score` ＝ **−8.25**，與 c47d2.jsonl 記錄的 `anchor_score: -8.25` **完全一致**。

→ 一支發車前用 `_chain_score` 算錨分的閘門，會**擋掉 c47d1（−99）且放行 c47d2（−8.25 且數值正確）**，零 HFSS、零誤殺。

**修法**：兩刀，**不是同一刀**（兩個 call site、兩種失效模式），但共用同一個述詞 `_chain_score(...) <= -98`。合計約 10 行，零新抽象層。

### Cut A —— 收檔端誠實記帳（治本次查證的主張；4 行淨增，dedust.py:4783-4789）
```python
best_id, best_s, best_v = max(scored, key=lambda t: t[1])
gated = sum(1 for _, s, _ in scored if s <= -98)      # −99 哨兵：出局筆數
oob = gated == len(scored)                            # 全包沒有一筆進盆地
win = (not oob) and (anchor_score is None or best_s > anchor_score)
rec = dict(pack=pack, n=len(scored), gated=gated, out_of_basin=bool(oob),
           anchor=anchor_id, best=(None if oob else best_id), best_score=_r(best_s),
           anchor_score=(None if anchor_score is None else _r(anchor_score)), win=bool(win),
           wm=(None if oob else _r(best_v["wm"][2])), rad=(None if oob else _r(best_v.get("rad_margin"))),
           oob_bad=(None if oob else _r(best_v.get("oob_bad"))),
           lo=(None if oob else _r(best_v.get("oob_gain_max_lo"))),
           hi=(None if oob else _r(best_v.get("oob_gain_max_hi"))))
```
外加 4832 的 print 在 `oob` 時改印「全包出局（0/25 進盆地,goal=… 門檻）」。
`win` 那行**一併關掉 `--anchor-score` 缺省的坑**（−99 不可能成為錨），不必另外補判斷。

### Cut B —— 發車端驗錨（治機時；5 行，插在 4626 `anchor_id, anchor_score = …` 之後）
```python
_st = args.source_input[:-6] if args.source_input.endswith("_input") else args.source_input
_rp = DATASET_PATH.joinpath(_st, "results.json")
_row = (json.load(open(str(_rp), encoding="utf-8")).get(args.anchor) or {}) if _rp.exists() else {}
if "wm" in _row and "error" not in _row:
    _as = _chain_score(_row, args.goal)
    if _as <= -98:
        raise SystemExit(f"錨 {args.anchor} 在 goal={args.goal} 的盆地外（score {_as};"
                         f"wm {_row['wm'][2]} rad {_row.get('rad_margin')} "
                         f"gain_max_lo {_row.get('oob_gain_max_lo')}）——換錨或改 goal")
    if anchor_score is None:
        anchor_score = _as        # 取代手打（c47d1 事故=手打值算錯欄位）
```
最後兩行是**刪掉**一個手打數字的來源（能刪不要包）——c47d1 的真正根因就是那個手打值。

### 是否同一刀
不是。Cut B 在**發車前**、讀**另一個檔**（錨所屬 store 的 results.json），只擋「起手就在盆地外」，成本 0 筆；Cut A 在**收檔後**，涵蓋「爬著爬著離開盆地」（Cut B 管不到），並且是唯一能修好帳本的一刀。c47d1 若只有 Cut B ＝ 省下全部 50 筆；只有 Cut A ＝ 燒 25 筆但帳本誠實、不會再有假 best 被後續 meta 分析吃進去。兩刀都上，共用一個 `_chain_score` 呼叫。

### 與審視原提案的**一處刻意分歧**
審視行動寫「全 −99 → 直接收鏈不再發下一包」。**建議不要加這個 abort**：Cut B 上線後，錨在發車時已保證在盆地內，那麼中途整包 −99 的唯一含意是「錨站在盆地的刀鋒邊緣」——那是真訊號，不是事故；而既有 `--dry`（預設 2）已把成本封在多 25 筆。為省 25 筆改控制流不划算。若 Ricky 仍要 abort，就是 `oob` 分支加一行 `dry = args.dry`（與 4680 / 4737 同慣用法）。

### 回歸測試（tests/CLAUDE.md「每修一個 bug 補一條」；純函式、零 NAS、`tests/fixtures/` 已存在）
1. 參數化 `_chain_score`：6 個 goal × {合格、閘門外、缺 `rad_margin`、缺 `oob_gain_max_lo`} 的期望值——**特別釘死「tri/rad 缺 rad → −9.0 而非 −99」這條現況**（見 caveats，是否要改成 −99 屬另案）。
2. 一條 chain 記帳測試：餵一個全 −99 的 `scored` list，斷言 `win is False`、`rec["best"] is None`、`rec["out_of_basin"] is True`、`rec["gated"] == n`。
3. 一條 Cut B 測試：monkeypatch `DATASET_PATH`，造一個 `oob_gain_max_lo=+3.49` 的錨 → `chain` 應 `SystemExit`；造 `-3.43` 的錨 → 應算出正確 `anchor_score`。（照 `test_jobs_add_concurrent_lock` 手法）

**侷限**：**判 medium 而非 high 的理由（誠實界定損害）**：①單次事故成本被 `--dry`（預設 2）封頂在 50 筆 HFSS ≈ 2.3 小時機時，不是無界；②全史發生率 2/142 包＝1.4%，且 `anchor_score=null` 那條加乘路徑 0/142 從未觸發；③鏈帳**無程式下游消費者**，不會自動污染 `docs/records.json` / champions / SM 訓練集——c47d1 那 50 筆本身是**有效量測**，照常進了資料池；④本次事故已被人工在 round-47 §4 手寫攔下。判 medium 而非 low，是因為它會靜默毒化一份 append-only 的帳，而專案自己的 meta 分析（含本輪 fanout 的 140 包統計）就吃這份帳。

**對原主張的兩點修正**：①「缺 rad 一律 gate 成 −99」只在 goal=lo/hi 成立；tri/rad 得到 **−9.0**（可勝出、可成為新錨——比原述更糟），wm 則是靜默扣 1.0 dB。②該缺件路徑**實測 0 實例**（25,531 筆全數有 `rad_margin`），屬純程式風險；鏈店也 0 筆缺 `oob_gain_max_lo`。所以「語義過載」判 partial：現象在，機制敘述要改。

**Cut B 的已知限制**：`<source_input>` 去掉 `_input` 推導 store 的慣例，全 NAS 611 個輸入夾中 **582（95%）** 有對應 `<store>/results.json`；另 5% 需退化成「找不到就照舊放行」（我給的程式碼已是這個行為），**不可 raise**，否則會擋掉合法的舊錨發車。此外我只在 c47d1/c47d2 兩個真實發車上做了端到端驗證（n=2），沒有回放全部 39 條鏈的發車指令（那些指令多數沒留在 repo 裡）。

**我沒做的事**：未跑 pytest（鐵則禁止改動與生產線干擾，且無程式變更可測）；未驗證 `_group_mutate`、`--expert` 分半記帳等鄰近路徑；`analyze.py` 是否有同型 −99 哨兵未查（審視另有一條 `analyze.py:825` 同型 `or` 寫法的獨立發現，不在本次 scope）。

## Execute（4 個分析結果）

### replay

**頭條**：SM 排序在 d=1 鏈鄰域沒有可用價值——包內 ρ 中位 0.068，top-5/top-10 勝錨保存率「輸給」同預算隨機挑（0.17 vs 0.20、0.26 vs 0.30）。鏈不能靠 SM 排序省一半 HFSS。機制已定位：改 1 px 真實 wm 擺動 7 dB，SM 只動 0.17 dB（複製 11% 真實變異），排的是自己的雜訊。

**數字**：
【前提修正】任務假設「鏈包 manifest 有 pred_wm*」→ 實測 0/171 個鏈 input 夾有任何 pred_* 欄位。鏈包＝chain daemon 的純隨機 d=1 變異，選批時根本不叫 SM；pred_* 只存在批次線（371 個店）。故改用 audit 建議的離線補打分（每包取其 select 時點前最新 sm_reanchorNN，防洩漏）。管線先驗證：批次線重算 vs 存檔 pred_wm，Spearman 0.997/0.998/0.999，22-23/25 完全吻合。

【覆蓋】142 鏈包 × ~25 = 3,538 筆 HFSS 真值，40 條鏈，29 個 checkpoint 版本，零 HFSS。

【① 包內 Spearman ρ（n=142 包）】
中位 0.068｜平均 0.058｜p25 −0.108 / p75 0.212｜ρ>0 61%｜ρ>0.3 17%｜ρ>0.5 2%
t 檢定 p=0.004，95% CI [0.019, 0.097] → 統計上非零，但上界 0.097 就是操作上的零。

【② top-k regret（wm 軸，vs 隨機的解析期望）】
              reg5中  reg5均  隨機5均 | reg10中 reg10均 隨機10均
總表(142)     0.070  0.140  0.177  |  0.010  0.085  0.095
tri系(100)    0.060  0.152  0.201  |  0.000  0.095  0.109
dual系(25)    0.100  0.100  0.117  |  0.030  0.056  0.057
rad系(16)     0.065  0.131  0.127  |  0.020  0.067  0.066
點名鏈：c45g2(tri,14包) ρ0.044 reg10均0.246/隨機0.268｜c45g3(4) ρ−0.061 0.037/0.050｜c41grp2(6) ρ0.089 0.048/0.041｜c45g1(3) ρ0.002 0.513/0.336｜c45d1(3) ρ0.101 0.057/0.130｜c47d2(3) ρ0.369 0.000/0.389｜c45d2(2) ρ−0.154 0.505/0.327

【③ 決定性數字：目標鍵勝錨保存率】（子集內用真值算 _chain_score，完全正確）
預算      全測25  SM top-k  同預算隨機  真勝者命中  機遇值
top-5      0.47    0.17      0.20      0.21     0.20
top-10     0.47    0.26      0.30      0.39     0.40
條件於「全測會勝錨」的 66 包：top-5 SM 0.36 vs 隨機 0.43（−0.068）；top-10 SM 0.56 vs 隨機 0.64（−0.081）。

【④ regret 對照「一包本來賺多少」】勝錨包目標鍵增益中位 +0.060 dB。
top-10 regret 均 0.083 ＝ 1.38× 一整包增益；top-5 均 0.197 ＝ 3.29×。→ 省 60% 機時＝丟掉比一整包還多的進度。

【⑤ 機制診斷（論文用）】
包內漢明距離中位 1.0 px（max 1，確認純 d=1）
包內真值 wm：sd 1.462 dB、全距 7.225 dB
包內 SM 預測：sd 0.172 dB → pred_sd/real_sd = 0.110（只複製 11% 真實變異）；pred_sd/自身殘差 = 0.113
尺度崩潰（同一顆 SM）：跨包混池 ρ=0.613（複現批線前瞻 ρ 0.59/0.56/0.64）vs 包內 ρ=0.068。

【⑥ 排除「鍵挑錯」】包內 ρ(真實wm, tri分數)=+0.873、ρ(真實rad, tri分數)=+0.403、ρ(真實wm, dual分數)=+0.832。tri 包 61% 由 wm 綁束、39% rad。→ 完美 wm 預測器排 tri 可達 ρ≈0.87，鍵沒問題，是 SM 在 1 px 尺度看不見東西。

【⑦ 三個穩健性檢查，結論不變】
洩漏上界（用已吃過這些包的最新 v86）：ρ 中位 +0.039，reg10 0.081 vs 隨機 0.095 → 沒變好。
生產通道（R40 起實權在 sm_two/cnn2，非 MLP；防洩漏重跑 79 包）：ρ 中位 +0.004，pred_sd/real_sd=0.050，reg10 0.117 vs 隨機 0.129 → 更平更糟。
真 HFSS A/B（6 個 --expert 包，包內 exp 12 筆 vs rand 13 筆）：exp 勝 4/6，Δ 中位 +0.020、平均 +0.016 → n=6，與零無法區分。

【⑧ 唯一有訊號處】依包內離散度四分位：Q1最平 ρ+0.014（reg10 0.161 vs 隨機 0.140，輸）→ Q4最崎嶇 ρ+0.132（0.028 vs 0.061，贏）。依 SM 誤差分層則無單調關係（E1 誤差最小 ρ+0.030 反低於 E4 的 +0.113）。

**方法**：零 HFSS 離線反事實重播，repo/NAS 全程唯讀，所有產物只落 scratchpad。
1) 掃 171 個鏈 input 夾確認無 pred_*（前提修正），改採離線補打分。
2) 防洩漏：每包取其 manifest.json mtime 之前最新的 sm_reanchorNN（29 個版本），確保 checkpoint 早於該包收檔＝不可能訓練過它。
3) 打分＝複刻 script/dedust.py:3827-3906 的 select 路徑：load_config(single_r5_explore.yaml) → SURROGATES["mlp"] → worst_margin(raw)[0]，與生產同尺。
4) 管線驗證：對三個有存 pred_wm 的批次線店重算比對（Spearman 0.997-0.999）。
5) 判準＝逐字複製 script/dedust.py:4566 _chain_score；每條鏈的 goal 用 docs/chains/*.jsonl 的 best_score/best 反推匹配（142 包全部匹配成功）。
6) 隨機對照用精確組合公式（E[max of random k] = Σ v_i·C(i−1,k−1)/C(n,k)；P(勝錨) 同法），非模擬，零抽樣雜訊。
7) 勝錨保存率：top-k 子集內用真值算目標鍵，與 jsonl 的 anchor_score 比 —— 這個反事實是精確的，不是估計。
8) 穩健性：洩漏上界（v86 全跑）、生產 cnn2 通道（sm_twoNN 防洩漏跑 79 包）、6 個 --expert 包的真 HFSS A/B。

**侷限**：1) n 的層級：142 是「包級」n；單鏈多為 2-7 包（僅 c45g2 有 14），逐鏈那些行不可單獨下結論——只有總表、分層、機制診斷有統計力。c45g3/c41grp3/c45d2/c45g4/c47d1 各 2-4 包，列出是為完整，不是判決。
2) 不是隨機對照實驗：這是離線反事實。SM 排序真上線會改變後續錨點軌跡（路徑相依），本重播模擬不了；它答的是「在既有 142 個實際發生的包上，SM 排序能不能保住 max」。
3) 只測 wm 通道排序（mlp + cnn2 兩條）。沒測「rad 頭 + lo 頭合成的多軸鍵」——那套頭沒有跨時間軸的完整版本（audit「標量 rad 判別器化石化」）。§6 的 ρ(真實wm, tri分數)=0.87 顯示鍵不是瓶頸，但多軸鍵未被直接證偽。
4) ρ 平均值統計上顯著 >0（p=0.004）——我沒有宣稱「SM 完全無訊號」，宣稱的是「訊號上界 0.097 在操作上等於零，且 top-k 實測輸給隨機」。這兩件事並存不矛盾。
5) --expert 那組 n=6 包，4/6 勝、Δ +0.02，撐不起任何結論，只能說「與零無法區分」。
6) goal 是從 jsonl 反推非讀 CLI 原始參數；2 包整包 −99（全 gated）已從目標鍵 regret 統計排除。
7) 打分管線每店有 2-3/25 筆與存檔 pred_wm 不吻合（打分後的除塵步驟改動 pattern），排序不受影響但非 bit 級重現。

**建議**：別把 SM 排序上鏈（--expert 標配也該撤）——改把這 142 包做成**凍結的「d=1 敏感度」離線考卷**，當任何新 SM 架構（GNN／局部感受野 CNN／殘差頭）的第一道零成本篩子：判準寫死「包內 ρ 中位 ≥ 0.30 且 pred_sd/real_sd ≥ 0.5」，過不了就別燒 HFSS 驗證，因為現有兩條通道（mlp 0.068／cnn2 0.004）證明「批線 ρ 0.61 好看」跟「鏈上排得動」是兩回事。

產物：C:/Users/Ricky/AppData/Local/Temp/claude/C--Users-Ricky-Documents-GitHub-Antenna/514acb31-4aa0-43ec-a3b3-98cdf8e2a623/scratchpad/replay_result.md（主結果表，8 節）
同目錄佐證：replay_per_pack.csv（每包一列：ρ/regret/勝錨/離散度/checkpoint）、replay_by_chain.csv（逐鏈彙總）、replay_raw.pkl（原始重播資料）、analyze2_out.txt、sanity_out.txt（漢明距離+洩漏上界）、two_channel.txt（cnn2 生產通道）、binding.txt（綁束軸分析）、gain_ctx.txt（regret vs 單包增益）
可重跑腳本：replay.py、analyze2.py、sanity.py、validate_pipeline.py（全部唯讀 NAS）

### frozen-slope

**頭條**：主張半對：frozen_med 全史（v39–v86）確實零斜率（+0.00058/版, p=0.44, R²=0.013），但「自 07-16 起」是空框——凍結欄第一筆就是 07-16 11:49，不存在前後對比；且尾端 v85/v86 已把近 20 版斜率壓到 −0.00498（p=0.043），規則餘裕從 51% 掉到 11%。條件③ 依可判定定義（K=20, x=0.010/版, 單邊 95%）在「有檢定力的兩把尺」上成立、自 v80 連 7 版，但三尺 AND 不成立（frozen_far 是檢定力失敗，非有斜率）。

**數字**：
【0】資料範圍事實（先破框）
frozen_med/frozen_far 第一筆 = v39 @ 2026-07-16 11:49，最後 = v86 @ 07-29 11:30，n=49。
→ 07-16 之前**沒有凍結尺資料**（只有 v35–v38 四筆 rolling）。「自 07-16 起無斜率」＝「凍結尺全史無斜率」，07-16 前後對比在凍結尺上不可做。

【1】全史 OLS（x=版號；負=在改善）
序列                    n    slope/版      95%CI              p       R²      MK p    Theil-Sen  殘差sd
frozen_med             49   +0.00058   [−0.0009,+0.0021]   0.444   0.013   0.623   +0.00045   0.0716
frozen_far             49   −0.00222   [−0.0056,+0.0011]   0.188   0.037   0.022   −0.00367   0.1596
two_frozen             25   −0.00116   [−0.0025,+0.0001]   0.079   0.128   0.088   −0.00085   0.0230
wm_err_med rolling     53   +0.00163   [+0.0003,+0.0030]   0.018   0.105   0.012   +0.00168   0.0723
  ↳ 同窗(v39起)         49   +0.00058   [−0.0008,+0.0020]   0.411   0.014   0.190   +0.00078
err_far rolling(陽性對照) 53  −0.02700   [−0.0297,−0.0243]  7.8e−26  0.887  <1e−4   −0.02728   0.1458
shadow frozen_med(對照) 37 下降期窗中位 slope −0.117~−0.129/版

→ 「真有斜率」長 R²=0.89；frozen_med 是 R²=0.013。差三個數量級，主張的核心成立。

【2】破紀錄檢查（凍結尺才可跨版比）
frozen_med  v86=1.097  全史排名 2/49   最佳 1.012@v41(07-16 20:49)  落後 +0.085，46 版未破
frozen_far  v86=2.776  全史排名 36/49  最佳 2.218@v40             落後 +0.558，47 版未破（比開場更差）
two_frozen  v85=0.811  全史排名 15/25  最佳 0.769@v70             落後 +0.042，15 版未破
wm_err_med  v86=1.202「新低」不成立 → 全史最低 v39 1.169（heldout_n 1772 vs 現 6244，尺不同）

【3】變點偵測（單變點分段線性；置換檢定修正「切點是搜出來的」）
序列          切點         maxF   naive p   置換 p(B=3000)   前段slope(p)      後段slope(p)
frozen_med   v77 07-27 05:50  5.40  0.0079   **0.147**      +0.00190(.058)  −0.01985(.025) n=10
frozen_far   v76             4.26  0.0201    0.287         −0.00079(.728)  +0.03602(.043) n=11
two_frozen   v71             4.57  0.0225    0.168
wm_err_med   v69             23.16 <1e−4    **0.0003**      +0.00471(.000)  −0.00874(.001)
err_far(對照) v62            23.02 <1e−4    **0.0003**
→ 凍結尺三條的變點在誠實 p 下全部不顯著；rolling 與陽性對照則是貨真價實的變點。

【4】尾端斜率穩健性（不同 K，避免挑 K 挑結論）
尺           K=6      K=8      K=10     K=12     K=15     K=20     K=25     K=30
frozen_med  −0.0357  −0.0160  −0.0198  −0.0118  −0.0083  −0.0050  −0.0044  −0.0013
frozen_far  +0.0456  +0.0595  +0.0351  +0.0287  +0.0059  −0.0061  −0.0020  −0.0012
two_frozen  +0.0019  +0.0016  −0.0009  +0.0012  +0.0001  +0.0002  −0.0012    --
→ frozen_med 每個 K 都負、且隨 K 增大單調收縮＝典型「只有尾端 2 點在動」。
   Mann-Whitney 近10版 vs 之前：frozen_med p=0.28、近15版 p=0.43（不顯著）。
   逐日尺：frozen_med 全史 +0.0027 dB/天(p=.33)，近15版 −0.0402 dB/天(p=.036)。
   frozen_far 反向惡化：v85=3.006＝全史最差；K=8 +0.0595/版(p=.058)。

【5】條件③ 操作化定義（可判定、可校準）
　定義：對每把凍結尺 M，取最近 K=20 版做 OLS(誤差 ~ 版號)；
　　　　若 slope 的**單邊 95% 下界 > −x**（x = 0.010 dB/版），
　　　　＝「可在 95% 信心下排除 ≥0.010 dB/版 的改善速率」→ 該尺判 flat。
　　　　條件③ 成立 ⇔ **frozen_med ∧ two_frozen 同時 flat**（frozen_far 只當觀察，理由見校準）。
　為什麼用 CI 下界而非 |slope|<x：點估計規則會「資料越少越容易宣告高原」；CI 版本在 K 不足時自動不成立。
　為什麼單邊：只需排除「還在改善」，不需排除「在變差」。

　x=0.010 的來源（實質顯著性，非統計顯著性）：frozen_med 現值 1.097 vs 決策層近域目標 0.80 → 缺口 0.297 dB；
　版節奏 49 版/13.0 天 = 3.8 版/天；x=0.010 ⇒ 20 版走 0.20 dB ⇒ 缺口需 ~30 版≈8 天。慢於此即「本衝刺內到不了目標線」。

　K=20 的來源（模擬，用各尺實測殘差 sd）：
　尺           sd      K=10           K=15          K=20           K=25
　frozen_med  0.072  FN 68.3%       FN 28.9%      FN **3.4%**    FN 0.1%
　two_frozen  0.023  FN 2.6%        FN 0.0%       FN 0.0%        FN 0.0%
　frozen_far  0.160  FN 86.5%       FN 74.3%      FN **53.9%**   FN 29.6%
　假陽 FP（真斜率恰為 −0.010 卻誤判高原）= 5.0~5.5%，各 K 恆定（單邊 95% 的設計值）。
　→ K=20 是 frozen_med 首次有檢定力的點（≈5.3 天）；frozen_far 在任何實務 K 都沒檢定力，故不入 AND。

　歷史校準（把規則放回真實序列滑窗，看它在「真的在改善」時誤 fire 多少）：
　陽性對照 shadow frozen 下降期：K=10 0/15、K=15 0/10、K=20 0/5 → fire 率 **0%**
　陽性對照 err_far rolling：     K=10 0/44、K=15 3/39(8%)、K=20 1/34(**3%**)
　→ 實測假陽率 0–3%（K=20），與模擬設計值 5% 一致。
　殘差 lag-1 自相關 frozen_med −0.149 / frozen_far +0.078 / two_frozen +0.016 → 無正自相關，OLS 的 SE 未被高估。

【6】照此定義，條件③ 現在成立嗎？（v86, K=20, x=0.010）
尺           n    slope/版     單邊95%下界   > −x?   判定
frozen_med  20   −0.00498    −0.00895      是     **FIRE**（餘裕 0.00105 = 門檻的 11%）
two_frozen  20   +0.00019    −0.00110      是     **FIRE**（餘裕 89%）
frozen_far  20   −0.00615    −0.01821      否     no ← 檢定力失敗(FN 53.9%)，不是「有斜率」的證據

　→ **雙尺 AND（frozen_med ∧ two_frozen）= 成立**，且自 v80(07-27 20:39) 起連續 7 版成立。
　→ 三尺 AND（含 frozen_far）= 不成立，且全史從未成立過（frozen_far 一路阻擋）。
　→ 若改用 K=15：frozen_med 下界 −0.0152 < −0.010 → 不成立。定義對 K 敏感。

【7】as-of 時間軸（規則在每版當下套用，K=20）
frozen_med：v58–v86 **29/29 版連續 fire**（凍結尺自有足夠窗以來，從沒有一次判過「在改善」）
two_frozen：v80–v86 7/7 fire（v80 前不足 20 版）
frozen_far：僅 v68–v76 間歇 fire，其餘皆 no
雙尺 AND：v80,81,82,83,84,85,86 連 7 版成立
餘裕衰減：frozen_med 下界 v84 −0.00491(餘裕 51%) → v85 −0.00638(36%) → v86 −0.00895(**11%**)

圖：C:\Users\Ricky\AppData\Local\Temp\claude\C--Users-Ricky-Documents-GitHub-Antenna\514acb31-4aa0-43ec-a3b3-98cdf8e2a623\scratchpad\frozen_slope.png（五panel：a 凍結vs漂移尺、b 遠域vs陽性對照、c 滾動K=20斜率+單邊下界、d K校準FP/FN、e as-of判定時間軸）

**方法**：唯讀讀入 docs/kpi.csv(v35–v86, n=53)、kpi_two.csv(n=25)、kpi_shadow.csv(n=37)，版號由檔名解析（58r 記為 58.5）。
統計：①OLS 斜率 + t 檢定 95%CI + R² + 殘差 sd；②Mann-Kendall + Theil-Sen（無母數，抗離群）；③單變點分段線性（各段 ≥6 點，窮舉切點最小化 SSE），因切點是搜出來的，naive F-p 過度樂觀，故再做 max-F 置換檢定（B=3000，在「單一線性趨勢」虛無下重排殘差）取誠實 p；④Mann-Whitney 尾端 vs 之前；⑤殘差 lag-1 自相關檢查 OLS SE 是否被高估。
條件③ 用等價檢定（non-inferiority / 單邊 TOST）而非點估計門檻。校準兩路並行：(a) 蒙地卡羅——用各尺實測殘差 sd 產生已知真斜率序列，量 FP(真 −0.010)/FN(真平)/FN(真 −0.002)，N=4000；(b) 歷史校準——把規則滑窗套在兩條「已知真的在改善」的真實序列（shadow frozen_med v60 後下降期、err_far rolling）當陽性對照，量實測誤 fire 率。最後做 as-of 重播：每版只用該版與其前 19 版重跑判定，看規則若一路在線會何時 fire。
腳本 frozen_slope.py / frozen_slope2.py / frozen_slope3.py 與文字報表 frozen_slope_report*.txt 全在 scratchpad。repo 與 NAS 零寫入。

**侷限**：1. **n 小且時間跨度只有 13 天**：凍結尺 n=49 版，two_frozen 只有 25 版、K=20 窗只剩 6 個獨立窗（as-of 只有 7 個判定點）。two_frozen 的「連 7 版成立」高度重疊，實質獨立資訊遠少於 7。
2. **「07-16 前後對比」做不到**——凍結欄從 07-16 11:49 才存在。任何宣稱「07-16 之後變平」的敘述都隱含一個不存在的對照期，只能說「凍結尺自誕生以來平坦」。
3. **尾端訊號建立在 2 個點上**：frozen_med 近 10 版 −0.0199/版(p=.025) 幾乎全由 v85(1.195)、v86(1.097) 驅動；變點誠實 p=0.147、Mann-Whitney p=0.28 → **不足以宣告已脫離高原**，但也足以讓「條件③ 成立」變成隨時會翻的判定（餘裕剩 11%）。
4. **同期 frozen_far 反向惡化**（v85=3.006 全史最差，近 8 版 +0.0595/版 p=.058）→ v85/v86 的 frozen_med 改善很可能是近/遠域**重分配**而非淨準度提升；fanout 審視已把「v85 跳檔 / 密度反權重離散化」列為嫌疑，此分析無法排除。這是不下「已破高原」結論的主因。
5. **frozen_far 不能參與條件③**：sd=0.160，訊號 0.010/版，K=20 的 FN=54%、K=25 仍 30%。它的「no」是檢定力失敗，不是證據；把它放進 AND 等於讓一把測不準的尺永久否決結論（實測：三尺 AND 全史從未成立過）。若要救它需 K≥40（≈11 天）或降噪。
6. **x=0.010 是決策參數不是資料推出來的**：它綁在「近域目標 0.80」與「20 版內走完缺口」這兩個外生設定上。真斜率 −0.005（門檻一半、100 版仍有 0.5 dB）時規則在 K=20 會 fire 52% → 這條規則對「慢但真實的改善」是刻意不敏感的，屬設計取捨而非缺陷，但引用時必須連同 x 一起講。
7. **版號當等距 x**：版間隔 2.5h–3.5 天不等、heldout_n 一路長（rolling 1595→6244）。逐日尺重算的結論方向一致（frozen_med 近15版 −0.0402 dB/天 p=.036），但版號軸與時間軸不完全等價。
8. **rolling 欄的全史斜率是統計陷阱**：wm_err_med 全史 +0.00163/版 p=0.018 看似「越訓越差」，但 held-out 集在同期擴張近 4 倍，尺本身在變 → 該數字不可解讀為準度退步；同理 v86 的 1.202「新低」在全史上不成立（v39=1.169）。
9. 未查證 kpi.csv 各版的訓練配方差異（反權重改版、資料混入、two/shadow 通道換裝），因此所有斜率皆為**觀察性**，不是任何介入的因果效果。

**建議**：把條件③ 寫死成「frozen_med ∧ two_frozen 近 K=20 版 OLS 斜率的單邊 95% 下界 > −0.010 dB/版（frozen_far 因 FN 54% 只列觀察）」，並記錄它**現在成立、已連 7 版，但 frozen_med 餘裕從 51% 掉到 11%**；因此不要據此宣告真高原，而是先花 v87–v90 四版（≈1 天，零額外 HFSS）確認 v85/v86 的 frozen_med 改善是真準度還是 v85 反權重改版造成的近/遠域重分配——判準：若 frozen_med 續降而 frozen_far 不再惡化（回到 ≤2.70）則視為真斜率、條件③ 撤銷；若 frozen_med 回彈至 ≥1.25 或 frozen_far 續留 ≥2.9，則條件③ 確立並可啟動 GNN bakeoff 之類的換軸決策。

產物：C:\Users\Ricky\AppData\Local\Temp\claude\C--Users-Ricky-Documents-GitHub-Antenna\514acb31-4aa0-43ec-a3b3-98cdf8e2a623\scratchpad\frozen_slope.png（主圖，五 panel）；同目錄另有 frozen_slope.py / frozen_slope2.py / frozen_slope3.py（可重跑）與 frozen_slope_report.txt / _report2.txt / _report3.txt（完整數字表，含 as-of 逐版判定明細）、frozen_slope_stats.json。repo 與 NAS 零寫入。

### dry2-calib

**頭條**：「dry2 假停率 76%」不成立——該數字量的是「人為換錨接棒後仍有增益」（17 對中 13 對同錨只佔 4 對）。用同錨連敗段重算，dry2 假停率落在 12%–50%（可判子集 n=6，3 例翻盤），且三例翻盤的增益只有 +0.01/+0.05/+0.06 dB。真正的儀器問題不是「停太早」，而是 dry 根本不量枯竭：收鏈時只採樣了 8.0% 的 d=1 鄰域，且連敗一包之後 hazard 就攤平在 ~1/3 不再衰減——所以「把假停率壓到 <20%」在任何有限 dry 值下都做不到，這個條件不能當高原判準的儀器用。

**數字**：
資料底：docs/chains/*.jsonl 全史 40 鏈 / 141 包 / 3,519 筆 HFSS；時序全部用 NAS results.json mtime（141/141 精確，零內插）；90 個相異錨 → 43 個「同錨連敗段」。

【A】複刻審視主張（完全對上，證明我讀的是同一份資料）
  審視口徑 n=17 對、13 對正增益 = 76%、中位 +0.04（含負值）、最大 +0.34（c6tri4→c6tri5）  ← 逐項重現
  但拆開類型： 同錨接棒 4 對 / 兄弟錨（人為換錨）接棒 13 對
    同錨 4 對增益：c1d3→c1d4 +0.01、c6tri8→c41grp2 +0.08、c45g2→c45g3 +0.12、c45g3→c45g4 +0.00
    最大值 +0.34 那對（c6tri4 終錨 c6tri4p03_06 → c6tri5 起錨 c6tri4p04_23）是換錨，不測 dry
  → 76% 是「換錨接棒的成功率」，不是假停率。

【B】我的假停定義：同錨連敗段（把全史包依錨 id ＋時間串起來；連敗 L 包後若同錨再被續測且出現 win，該次 dryD(D≤L) 收鏈即為假停）
  dry=2 觸發收鏈 26 段 → 其中同錨真的被續測 6 段 → 翻盤 3 段
    假停率（續測子集）= 3/6 = 50%  [Wilson 95% CI 19–81%]   ← 上界（續測只發生在主力線＝選擇偏誤朝上）
    假停率（下界，未續測一律當真停）= 3/26 = 12%
    → 誠實區間 12%–50%，可判 n=6。
  加增益門檻後：≥0.05 dB 2/6=33%[10–70] ｜ ≥0.10 dB 0/6=0%[0–39] ｜ ≥0.20 dB 0/6=0%
  dry=3：觸發 5 段、續測 3 段、翻盤 2 段 = 67%[21–94]（n=3，無鑑別力）

【C】「第 N 包才翻盤」分布（uncensored；需要 dry = L+1）
  L=1（dry2 已能接住）：14 例，增益 +0.02…+0.45，中位 +0.065
  L=2（需 dry3）：1 例  c45g2p12_24 +0.06
  L=3（需 dry4）：2 例  c6tri8p01_18 +0.05、c1d3p01_18 +0.01
  L≥4：0 例（1 段右設限：c45g3p02_06 連敗 4 包仍未翻盤）
  ⚠ 結構性設限：dry2 讓 L≥2 只能透過接棒觀測，全史只有 6 段有續測資料。

【D】hazard（同錨已連敗 j 包後，下一包勝率）——關鍵證據
  j=0   n=91  勝49  0.54 [0.44,0.64]   勝時中位 +0.050 / 均值 +0.204
  j=1   n=39  勝14  0.36 [0.23,0.52]   中位 +0.065 / 均值 +0.096
  j≥2   n= 9  勝 3  0.33 [0.12,0.65]   中位 +0.050 / 均值 +0.040
  Fisher: h(0) vs h(1) p=0.084；h(1) vs h(≥2) p=1.000
  → 第一包敗有訊號（邊緣顯著），第二包敗之後 hazard 攤平在 ~1/3，看不到任何額外衰減。
  常數 h=0.35 外插：收鏈時鄰域還剩 ~22 包沒抽 → P(還會再勝一次) = 1.00，對 D=2/3/4/6/10 全部 1.00
  → 「假停率 <20%」在任何有限 dry 值下不可達；只有把假停改成「帶增益門檻」才可能（≥0.10 dB 現行 dry2 已是 0/6）。

【E】dry 不量枯竭（儀器可靠性本體）
  收鏈時同錨累計取樣 = 2~4 包 × 25 = 50~100 / 624 個 d=1 變體 = 8.0%~16.0%
  敗包差距（錨分 − 包內最佳）n=73：中位 0.050 dB；<0.02 佔 27%、<0.05 佔 49%、<0.10 佔 73%
  → 近半數「無勝錨」判定是 0.05 dB 內的平手，不是枯竭證據。
  混淆失效：c47d1 p01/p02 兩包整包 −99（盆地外）被 dry 當成正常敗包 → 2/141 包 = 1.4%；1/28 條 dry2 收鏈的鏈 = 3.6% 的收鏈是儀器失效誤讀成軸枯竭。

【F】邊際經濟（決定 dry 該不該調的真正依據）
  停滯錨多發一包（j≥1）：0.35 × 均值 +0.086 = +0.031 dB / 25 筆 = 0.00123 dB per HFSS
    （剔最大離群後 +0.021 / 中位法 +0.021 → 三種估計 0.021~0.031）
  剛勝或新錨那一包（j=0）：0.54 × +0.204 = +0.110 dB / 25 筆 = 0.0044 dB per HFSS
    （剔最大離群後 +0.078 / 中位法 +0.027）
  比值 1.3×（中位法，最保守）~ 3.7×（截尾均值），方向三種估計一致：停滯錨的邊際包比較不值錢。
  dry2→dry3 全史成本：26 次觸發 × 1 包 = 650 筆 HFSS；已知能救回 3 次翻盤共 +0.12 dB；模型外插 9.1 次 / +0.78 dB。
  dry1→dry2 的同一算式：48 包 = 1,200 筆 HFSS 買到 17 次翻盤 / +1.47 dB = 0.00123 dB per HFSS ← 與 dry2→dry3 完全同量級，證實敗一包之後邊際報酬是平的、沒有拐點。

【G】對高原條件① 的直接結論
  同錨假停造成的終點低估上界 = +0.12 dB（三例中最大，c45g2→c45g3；下游整鏈亦僅 +0.12）
  → 審視原本的結論「偏誤量級小、撐不起 1.7 dB 缺口」成立，而且比它報的 ≤0.34 更緊（0.34 那筆是換錨、不屬於此偏誤）。
  → 但條件① 若寫成「dry 收鏈=軸枯竭」則不可用：dry 在 8% 覆蓋率、hazard 未衰減、含 3.6% 儀器失效的狀態下停鏈。

**方法**：口徑與作法（全程唯讀，repo/NAS 零寫入，中間產物只落 scratchpad）：
1. 讀 docs/chains/*.jsonl 全 40 檔 141 包（去重 c6tri5 手動代判的重複行 1 筆）。時序不靠檔案 mtime 猜——直接取 NAS `dedust_<鏈>_p<NN>/results.json` 的 mtime，141/141 全部命中，零內插。
2. 從 script/dedust.py:4671/4834-4842 確認 dry 語義：`win = best_s > anchor_score`（嚴格大於），敗 → dry+=1，勝 → dry=0 且換錨；`while dry < args.dry`（--dry 預設 2）。故「連敗 L 包後翻盤」對應「需要 --dry = L+1」。
3. 假停的可量測代理＝**同錨連敗段**：把全史包依 anchor id 分組、按 NAS 時間排序（同一錨會橫跨母鏈與接棒鏈），取連續敗包段。段被 win 終結 = uncensored（觀測到「其實還爬得動」）；段末無後續包 = 右設限（無資訊）。這比「鏈對鏈」正確，因為①接棒鏈的錨常常不是母鏈終錨②同一錨可被跨鏈重複開採③它把鏈內敗包也納入，樣本從 17 對變成 43 段/141 包。
4. 三種估計都報：續測子集（上界，含選擇偏誤）、全觸發下界、以及帶增益門檻的版本。比例一律附 Wilson 95% CI，兩兩比較用 Fisher exact。
5. 邊際經濟用「每包期望增益 = hazard × 勝時增益」，並同時給均值／截尾均值／中位法三種估計，因為 j=0 的增益分布有長尾（單筆 +2.91 = c45g2 p09 那次跳段）。

定義的侷限（先講清楚再看數字）：
- **可判樣本只有 n=6**。dry2 本身讓 L≥2 不可觀測，只有人類決定同錨接棒時才會產生續測資料；D≥3 的一切結論建立在 3 個案例上，不能當統計結論用。
- **選擇偏誤朝上**：續測發生在當時的主力線（c45g、c6tri8、c1d3、c12tri、c6tri2），研究者本來就認為值得再挖 → 50% 是上界，真值應更靠近 12%。
- **接棒不是純粹的延續**：後鏈可能換了 SM 版本、換了生成臂；`_preload_used`（dedust.py:4634-4652）只排除同 source_id 的 chain_d1/scope_d1 已測 px，重疊控制是近似不是精確。
- **增益只計「翻盤那一包」的即時增益**。不計血統複利（c45g2p12_24 那次 +0.06 翻盤，後續才長成 g 線現任終點 −2.72）——全歸給它會誇大、完全不計會低估，故兩邊都標出來。
- **雜訊不是問題**：HFSS 已公證決定性（跨機 bit 級一致），所以 win/敗判定不含量測雜訊，翻盤都是真的；但「敗包差距中位 0.05 dB」說明判定本身是平手級的，弱的是訊號量不是雜訊。
- 「假停率 <20%」這個目標在「任何正增益都算假停」的定義下**本質不可達**（鄰域還剩 22 包、hazard 0.35），這不是資料不足，是定義退化。要可達必須綁增益門檻或綁預算。

**侷限**：1. **n 小要說在前面**：整個 dry 校準的可判子集是 6 段連敗、3 次翻盤。50% 這個點估計的 95% CI 是 19–81%，實務上等於「介於少見與過半之間」。任何「dry 該調成 3」或「dry 該調成 1」的因果宣稱，這份資料都撐不起來；能撐起來的只有「76% 不是假停率」和「hazard 在 j≥1 之後是平的」。
2. h(≥2)=0.33 只有 n=9。我說「hazard 攤平」的依據是 h(1) vs h(≥2) 的 Fisher p=1.000，那是**檢定不出差異**，不是**證明沒有差異**（power 近乎零）。h(0) vs h(1) 的 p=0.084 也只是邊緣。
3. 全史 141 包不是同質母體：橫跨 c1d~c47d 七週、goal 有 wm/dual/tri/lo/rad 五種、錨深度從 −9 到 0，SM 從 v52 一路到 v86。我做了深度分層（深水 j=0 0.68 / j≥1 0.44；作戰區 0.85 / 0.50；刀鋒 0.41 / 0.32，各層 j≥1 的 n 分別是 9/2/37），方向一致但只有刀鋒層 n 夠看。goal 別、SM 版本別沒有分層——樣本不夠再切。
4. c47d1 的兩包 −99 我在 hazard 與差距統計中剔除（否則差距中位被 92.07 汙染），但在「儀器失效比率」中保留計數。這是我的判斷：那兩包量的是擇錨口徑事故，不是 dry 訊號。
5. 「dry2 收鏈時只採樣 8% 鄰域」是算術（50/624），不是實測——`_preload_used` 對接棒鏈會預載已測 px，所以連續接棒的錨（c45g3p02_06 累計 4 包）覆蓋率到 16%，但沒有任何錨接近枯竭。dedust.py:4671 附近的「鄰域枯竭」分支（avail < n）在全史 141 包中一次都沒觸發過。
6. 我沒有動 repo 任何檔、沒跑 select/chain/train/git；目前 git status 的 M/?? 是生產線自己的改動，與本次分析無關。

**建議**：不要把 dry 調到 3（650 筆 HFSS 換模型外插 +0.78 dB，單位報酬 0.0012 dB/HFSS，比同一包放在剛勝／新錨上差 1.3–3.7 倍）；把 dry 留在 2，改做兩件成本更低的事——①先修「整包 −99 判為 dry」的混淆（收檔端 out_of_basin 偵測，全史 3.6% 的收鏈是這樣來的），②高原條件① 的措辭從「dry 收鏈=軸枯竭」改成可算式「終點 wm ＋ 同錨續測殘餘 ≤ +0.12 dB（n=6 觀測上界）」，並停用「假停率 <20%」這個目標（在現行定義下對任何有限 dry 值都不可達）。

產物：中間產物（全在 scratchpad，未進 repo）：
C:\Users\Ricky\AppData\Local\Temp\claude\C--Users-Ricky-Documents-GitHub-Antenna\514acb31-4aa0-43ec-a3b3-98cdf8e2a623\scratchpad\
  mt.py / pack_mtimes.json     — NAS results.json mtime 抽取（141 包精確時序）
  an2.py / pairs.json          — 鏈層級接棒對（複刻審視 n=17/76%/+0.34 口徑，並標 same vs sibling）
  an3.py / runs.json           — 同錨連敗段抽取（43 段）＋ hazard ＋ 假停率 vs D ＋ 翻盤/設限明細
  an4.py                       — Wilson CI、增益門檻分層、邊際經濟、深度分層、常數 hazard 外插
  an5.py                       — 增益分布穩健性（截尾/中位法）、鄰域覆蓋率、−99 事故率、dry3 成本效益
  an6.py                       — Fisher exact 檢定、審視中位數 +0.04 口徑對帳
  anchor_recs.json             — 90 個錨的完整 W/L 序列與跨鏈歸屬
重跑：cd 上述目錄後依序 an2/an3/an4/an5/an6，python 用 C:\Users\Ricky\miniforge3\envs\ant\python.exe，需 PYTHONIOENCODING=utf-8。

### basin-calib

**頭條**：首包訊號對終局有真實但弱的預測力（ρ +0.42, n=39, 置換 p=0.009），且幾乎只存在於 9 條新錨鏈（ρ +0.79）；任何能省 20% 以上機時的「首包不肥沃就停損」規則，retrodict 全史都會殺掉現任 usable_lo 紀錄 −3.46 與左側合格解 #2——結論是首包可以當排序器，不能當開關。

**數字**：
資料：docs/chains/*.jsonl 全史 **40 條鏈 / 142 包 / ≈3,550 筆 HFSS**（快照 2026-07-29 13:31；c47d2 在飛，p03 於分析中落地）。排除 c47d1（擇錨事故，整包 −99）後 n=39。

【A. 預測力：首包 d1 → 後續爬升】（後續爬升 = max(pack≥2 best) − max(錨, 首包 best) = 停在第一包會放棄的增益）
| 子群 | n | ρ(d1,後續爬) | perm p | ρ(beat1,後續) | ρ(d1,總爬)※ |
|---|--:|--:|--:|--:|--:|
| 全部 | 39 | **+0.42** | 0.009 | +0.41 | +0.71 |
| 新錨鏈（獨立盆地） | 9 | **+0.79** | 0.016 | +0.74 | +0.87 |
| 接棒鏈 | 30 | +0.20 | 0.279 | +0.16 | +0.60 |
| 王朝 c6/c8/c41grp | 11 | +0.39 | 0.241 | +0.43 | +0.62 |
| 深水 c45d/g,c47d | 7 | +0.38 | 0.401 | +0.25 | +0.36 |
| 其他（c1–c12） | 21 | **+0.01** | 0.960 | +0.16 | +0.50 |
※「總爬」與 d1 機械耦合（首包自己算在內），只列作對照，不可當預測力。

【B. 二分】win1 × 後續爬升>0.1 dB：T 7/22、F 1/17 → 敏感度 88%(7/8)、特異度 52%(16/31)、**Fisher 雙尾 p=0.106（n=39 未達顯著）**。
後續爬升 中位/平均/最大：win1=T +0.04/+0.44/+4.70；win1=F +0.00/+0.02/+0.18。拿掉主導的 c45g2 後 T 仍 +0.23 平均 vs F +0.02。
逐包勝率：全史 66/142=46%；首包勝的鏈其後續包勝率 49%(n=77)，首包不勝的鏈 24%(n=25)。

【C. 停損規則 retrodict（含血統連鎖：停在 p1 ⇒ pack≥2 從未量測 ⇒ 以其為錨的下游鏈整條不存在）】
| 規則 | 直接停 | 連鎖失 | 省包 | 省 HFSS | % | 失去的里程碑 |
|---|--:|--:|--:|--:|--:|---|
| 首包不勝錨(win=F) | 12 | 14 | 61 | 1,525 | 43% | **3 項** |
| d1<0 | 11 | 14 | 59 | 1,475 | 42% | **3 項** |
| d1<−0.05 | 6 | 8 | 41 | 1,025 | 29% | **3 項** |
| d1<−0.15 | 3 | 6 | 29 | 725 | 20% | **3 項** |
| d1<−0.30 | 2 | 0 | 2 | 50 | 1% | 無 |
| 僅盆地外(−99) | 1 | 0 | 1 | 25 | 1% | 無 |
前緣是斷崖、無中間帶：能省 ≥20% 的門檻全殺同一批里程碑；保得住的只省 1%。斷點卡在 c6tri3（d1 −0.22）。
失去的 3 項＝usable_lo −3.46（c41grp2p02_02，該鏈 p1 輸）、左側合格解#4（c41grp2p06_11）、左側合格解#2（c6tri6p03_09）。保住的＝★左側合格解#1 / usable_oob 7.78（c8trip03_01，該鏈 p1 勝）。

【D. 兩個決定性反例】
- **c6tri3**：d1 −0.22（全史第 3 百分位）、0/25 勝錨、自身兩包全敗（總爬 −0.01）→ p02 產出 c6tri3p02_20 → c6tri4(+0.14) → c6tri5(+0.34) → c6tri6p03_09（左側合格解#2）→ c6tri7 → c6tri8 → **c41grp2p02_02＝現任 usable_lo −3.46**（六代）。
- **c45g2**：d1 +0.03（第 64 百分位＝中庸）、1/25 勝錨 → 14 包 **+4.70 dB**（全史單鏈最大），其中 p09 單包 +2.91、p12 才進苗子帶。前 8 包只累積 +0.84。

【E. 小探針檢定力（用首包 25 筆真值反推 n 筆子集至少 1 筆勝錨）】
beat1 中位：王朝 1/25、深水 2/25、其他 1/25。P(≥1)@6筆：王朝 18%、深水 50%、其他 22%；肥沃鏈(後續>0.1) 54% vs 貧瘠 18%（對比 3×，但絕對值太低）。與 fanout 審視第 83 條結論一致。

【F. 血統森林】40 條鏈只有 **10 個獨立起錨**；其中 22 條同屬 c2rad 一個血統（合成 +1.96）。合成爬升：c45g1 系 +5.32（4 鏈 23 包）、c45d1 系 +2.07、c2rad 系 +1.96、c47d2 +0.68、c1d2 系 +0.10。

**方法**：全程唯讀、零 HFSS。①解析 docs/chains/*.jsonl 40 檔（依 pack 去重，c6tri5 有重複 p07 行）；②對每包載 NAS dedust_<鏈>_p<NN>/results.json，用 script/dedust.py `_chain_score` 同源公式重算全部 25 筆分數，並以「哪個 goal 能複現 jsonl 的 best_score」反解每條鏈的 goal（wm/dual/tri/rad/lo/hi），得到 beat1（首包勝錨筆數）、med1、sd1；③終局量刻意用「後續爬升」而非「總爬升」——總爬升把首包自己納入 max，與 d1 機械耦合會把 ρ 從 +0.42 灌到 +0.71；④Spearman + 20,000 次置換檢定 p 值、Fisher 精確檢定；⑤停損 retrodict 用血統圖傳播：以 id 正則 `^(c\d+[a-z0-9]*)p(\d{2})_\d{2}$` 解析每條鏈的錨出自哪條鏈哪一包，區分「被停鏈」（只跑 p1，其 p1 產物仍存在）與「連鎖消失」（來源包從未發生，整條鏈不存在），迭代到不動點；⑥對 docs/records.json 的現存紀錄保有者逐筆檢查在反事實下是否還存在。腳本：scratchpad/basin_screen2.py（主表+相關性）、basin_screen3.py（血統+連鎖）、sweep.py（門檻掃描）。

**侷限**：1) **n 小且高度不獨立**：40 條鏈只有 10 個獨立起錨，22 條同屬一個血統，有效樣本量遠小於 40；主要 2×2 的 Fisher p=0.106 未達顯著，分層 ρ 的 p 值全部 >0.24（王朝 0.241 / 深水 0.401）。只有「全部」與「新錨鏈」兩格 p<0.02。
2) **任務要求的「預測分散度」在現有資料下不存在**：掃過 NAS 全部 169 個鏈 input 夾 manifest，零個含 pred_* 欄位（鏈包走 d=1 變異生成，不經 select 的預測路徑）。所有首包訊號都是量測後真值（d1/beat1/med1/sd1），不是發車前可得的前瞻訊號——這對「早篩」的可用性是硬限制：真正的早篩需要 pack1 跑完（25 筆 HFSS）才有訊號。
3) **「進苗子帶」只有深水層有定義**：其餘 33 條鏈的 wm 早已 ≫ −3；深水 7 條中 3 條進帶，win1 完全不分離（n=7，零結論）。
4) **retrodict ≠ 實驗**：反事實只假設「被砍鏈的 pack≥2 未發生」，沒有計入省下的機時拿去跑別的東西的收益。表中 42% 是機時上界，不是淨收益。
5) **跨層 dB 不可加總**：王朝層增益是 0.0x 級、深水是 dB 級，「放棄增益合計 +0.39 dB」只是各鏈獨立加總的記帳量，不是任何單一價值軸的總和。
6) **c47d1 被排除**（d1 −92.07 會單點主導所有係數）；但它反向支持一件事：盆地外偵測（錨分 −99）值得寫成 refuse 條件，那是另一個問題，不是肥沃度。
7) 快照期間 c47d2 p03 落地，142 包這個數字是 2026-07-29 13:31 的狀態。

**建議**：別把首包當停損開關——只把「d1 < −0.30」與「錨分掉出盆地（−99）」寫進 chain 的發車前 refuse 條件（retrodict 零里程碑損失、代價僅 1–2 包），首包訊號改當「同時開幾條鏈、誰優先續包」的排序器；若仍要做真早篩，先限定在深水層（唯一 6 筆探針有 50% 檢定力的地層）且判準必須跨軸（被錯殺的 c41grp2 首包 goal 鍵輸 −0.06，但同包 lo 已達 −3.42＝當時紀錄級深度）。

產物：https://claude.ai/code/artifact/454c28e4-6f73-4fbe-8a51-4c7d267a82c9

