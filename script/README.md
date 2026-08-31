腳本 (Script)
==========

正常情況下是不能直接執行的，所以可以用以下方式解決。

1. 在Python檔的開頭加入下列程式碼
    ```python
    from sys import path
    from os.path import dirname, join
    path.append(join(dirname(__file__),'..'))
    ```
2. 在終端機 (根目錄) 執行以下指令
    ```bash
    python -m script.<腳本名>

    # Example
    python -m script.kill
    ```
## 主要腳本一覽（2026-07-06）

| 腳本 | 用途 |
|---|---|
| `dedust.py` | **批次 HFSS 驗證線**（R7 起的研究主力）：select-* 生輸入 → run 燒 HFSS → report 看結果（子命令地圖見檔頭 docstring） |
| `dedust.py worker` / `jobs-add` | **資料工廠**（2026-07-10）:NAS 佇列派工＋單筆 watchdog＋連敗保險絲（詳 `script/CLAUDE.md`） |
| `sm_reanchor.py` | SM 乾淨區重錨（train/eval;產 NAS `sm_reanchor.pth`） |
| `status.py` | NAS run 狀態掃描（`--md` 貼 ONGOING;`--alert --notify-topic` 當 watchdog） |
| `expected_best.py` | R6 期望基準尺（每 round 收檔可重跑疊圖） |
| `pattern_anatomy.py` | 池結構特徵快取（`collect-pool` → `tmp/pattern_anatomy/pool.npz`,多個 select 依賴） |
| `round_report.py` / `analyze.py` / `benchmark_vs_random.py` | 線上 run 的歸檔圖表 / 重現診斷 / worst-margin 對標 |
| `sm_selection_audit.py` | **SM 的「選批能力」稽核**（唯讀）：把準度翻譯成「這一批能挑到多好的候選」——`P(勝隨機)` ＋ `top10% 命中率`，重抽候選池給 CI。樣本自動取 `fit` 分割中**不在 `CLEAN_STORES`**（587 店，含自動納入的 `dedust_auto*`/`dedust_c*`）的 OOS 樣本。**何時用**：改動只碰模型的一部分機制時（換架構／加特徵／換錨組／改 loss 的一項）——diffsim 實測過一個反例：修結構性 bug 讓 ρ 改善 +124% 而選批 P(勝隨機) 18%→17% 完全沒動（`docs/log/analysis-10` §7/§8）。`--versions 88,94,100 [--k 60] [--stratum clean\|neg]` |
| `diffsim/` | **可微模擬器**（`docs/diffsim.md`／log `analysis-08`→`09`→`10`）：`data`(NAS 索引+val/dev/fit 決定性切分)／`geom`(SAB 直解的幾何真相)／`l1`(腔模型，快、clean 層內 ρ +0.418、84ms/筆)／`l2`(rooftop MoM；**求解器登記表 `SOLVERS`＝`dcim`/`l3`/`l3fl`**，一個名字＝核+埠+遠場一整組)／`l3`(精確分層 Green's function，Michalski–Mosig formulation C；表以整數 d² 索引、格網無關；`python -m script.diffsim.l3 build` 建表)／`eval`(同一把尺+ρ)／`run`(fitscan・gate1・l2cal・l2eval・l2fit・gate2・head，`--solver` 選設定)。零 HFSS、只讀 NAS。⚠ 舊的「離散化天花板」結論**已於 analysis-10 §37 撤回**（真根因是埠模型）；`analysis-09` 的 G-L3a 判準未過仍成立 |
| `gnn_bakeoff.py` | 金屬像素圖 GNN bakeoff（規格 v3;build-cache/train/grid/exam=d=1 考卷;方向③） |
| `handoff_pack.py` | **送板交付包**（2026-08-31 學長送板需求;`index`/`report`/`pdf`/`export`/`collect`）：從「一開始就帶菱形橋量測」的 store 挑家族代表 → 出 single/dual 兩份 PDF 目錄（pattern＋橋位＋24-32GHz 響應＋指標＋交付檔名）。尺＝single `worst_margin`(門檻 R54 規則① ≥0.3)／dual `wm_mfg`=min(m1+2,m2+2,m3,m4+5)(規格 v2);家族＝`dedust.cluster_families`;橋位＝`single_port.diag_bridge_sites`(模擬器同一份)。**R54 規則② n_sites<17 只套 single**(dual 帶橋量測、稅已內含,見檔頭) |
| `sm_dual.py` | **dual-port 排序器**（R58;`train`/`eval`/`rank`）：幾千張候選 → 挑 top-N 送 HFSS。鍋＝`harvest_dual`+`dedust_r57*` 去重 10,239 筆,尺＝`worst_margin_dual`。**排序走 margin 頭（直接回歸 m1..m4）而非「先預測響應再取極值」**（held-out ρ +0.47 vs +0.39）。權重＝NAS `sm_dual_v1_s{0,1,2}.pth` |

接手導覽：先讀 `docs/log/README.md`（時間軸）→ `configs/ONGOING.md`（live）→ 該 round 檔。
