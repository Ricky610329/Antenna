# 研究主線時間軸（decision log）

> 每個 round = 一個假設的完整生命週期(propose → run → analyze → conclude → archive)。
> 一個 round 一檔(`round-NN-<slug>.md`,用 `_TEMPLATE.md`),這裡只放時間軸索引。
>
> - **設計理由 / 為什麼**:看 `../*.md`(exploration_roadmap / research_landscape / guided_search_design / senior_method)。
> - **config 全集**:看 `../../configs/README.md`。
> - **現在在跑什麼 / 候選池**:看 `../../configs/ONGOING.md`(live 操作板)。
> - **跨 session 接手**:先讀本檔(走到第幾 round)+ ONGOING.md(手邊在跑什麼)。

最後更新:2026-07-13

## 時間軸
| Round | 主題 | 狀態 | 結論(一句) | 檔 |
|---|---|---|---|---|
| 01 | SM 訓練量 A/B(dlf/dlf_fit/refit) | ✅ archived | **訓練量非 bottleneck**;dlf≈refit>dlf_fit、皆差 spec ~4dB、未收斂 | [round-01](round-01-sm-training-ab.md) |
| 02 | ensemble + trust(文獻治本) | ✅ archived(2026-07-01 停,②~417ep) | **治本微幅、未決定性**(②③微贏~0.3-0.5dB、①輸、皆未收斂;trust_t 卡低) | [round-02](round-02-ensemble-trust.md) |
| 03 | 探索 × DIP（factorial E/D/E+D） | ✅ archived（2026-07-02 停,E@189/D@101/E+D@132） | **E(lr↑)最佳 -3.63@89（¼ epoch 追平②）**;DIP 連通成功但停滯;三臂被 SM 欠訓汙染 → R4 修瓶頸重跑 | [round-03](round-03-explore-dip.md) |
| 04 | 自適應 SM 訓練量（修 R3 SM 欠訓瓶頸） | ✅ archived（2026-07-03 停,E@208/D@222/E+D@201） | **E+D 破紀錄 -2.89@154**（探索躍遷,+2.80 vs R3）；但主假設未驗證——探測自鎖 3-5ep、fit_loss 仍 8-11、trust 全鎖；E/D 輸 R3 ~0.9dB → R5 | [round-04](round-04-adaptive-sm.md) |
| 05 | 滑動視窗 SM 訓練量（修 R4 欠訓+探測自鎖） | ✅ archived（2026-07-10 收 E 臂） | **gap 1.24 史上最低=訓練量修好;best −3.65 未破紀錄、trust 未解鎖=瓶頸在泛化/搜尋**;線上線收束（37 筆/天 vs 批次 540=15×） | [round-05](round-05-window-sm.md) |
| 06 | 離線期望基準（每輪 HFSS 的期望 best；零 HFSS） | ✅ archived（2026-07-03 當日完成） | **期望爬升到不了 spec**（-9.18+0.75·ln k,躍遷主導）；**達標 pattern 已在池內（oracle +0.38）**；學長同預算贏 1-2dB；池抽樣等效預算 200-450× → **分布≫策略** | [round-06](round-06-offline-expected-best.md) |
| 07 | 除塵驗證（達標 pattern 拔 1-3px 粉塵 HFSS 重驗＋順收 rad） | ✅ archived（2026-07-03 當日完成,15 筆/45 分） | **粉塵=共振的一部分（4/5 崩 -4.7~-16.9dB）→ 乾淨解用搜的不能用修的**；例外 p03 整塊型近零代價＝可製造最佳已知點（-2.68,rad+0.24）；oracle 重驗真（p00 +0.44）；rad=獨立第三關、與可製造同向 | [round-07](round-07-dedust.md) |
| 08 | 乾淨子空間測繪（前緣真值/補洞因果/SM 校準/random 基線,97 筆） | ✅ archived（2026-07-05 收檔,斷電中斷一次） | **A 崩**（整塊型除塵 \|Δ\| 中位 1.17,通則不成立）/**B 敗**（補洞非因果,rad 全負）/**C 半亮**（SM 池內 1.5-2.4dB、池外 4-5.5）/**D 實錘**（uniform 輸池抽樣 ~5dB）＋⚠池值漂移警訊→R9；附圖報告 [round-08-report](round-08-report.md) | [round-08](round-08-clean-mapping.md) |
| 09 | 池頂端重驗＋乾淨前緣探索（過夜 162 筆,重驗 T/N/M＋探索 E/G/S） | ✅ archived（2026-07-06 全收 162/162＋可重複性公證 41 次） | **oracle 活著（8/18,t00 +0.44）**；漂移家族依賴（頂帶±0.4 可信）；**可製造紀錄 −2.68→−0.29**（s05=F2×10-5-10 對稱,S11✓）；F3=可製造沃土；SM 分布外排序仍有訊號（G 贏 E 2.4dB） | [round-09](round-09-pool-revalidation.md) |
| 10 | 精修 × 物理歸因（ref1/ref2/遮蔽/重錨/交叉驗證,共 ~350 筆） | ✅ archived（2026-07-07,八冠軍 certified） | **八個三標全過冠軍（best c21 +0.20/+0.12）,可製造紀錄一週 −2.68→+0.20**;對稱化規則過因果關;承重圖;+0.48 假象偵破→量測誠信體系;SM 作戰區 0.36dB;名鑑 [champions](../champions.md) | [round-10](round-10-refine-attribution.md)＋[report](round-10-report.md) |
| 11 | 冠軍公差穩健化 × 規則普適性（tol/occl2/ref3/probes/wide,~535 筆） | ✅ archived（2026-07-08） | 承重圖跨家族(通則);**新王 c25 +0.22(組數階梯)**;對稱度=rad旋鈕(因果);搭橋崩=懸浮件功能(反駁孤島抑制);穩健王 c21 50% | [round-11](round-11-robustness.md) |
| 12 | 收斂（穩健冠軍）× 破單一化（第二山頭） | ✅ archived（2026-07-08） | 8 候選全公證;**第二山頭否決=w17 特殊性確立**;穩健王 c21;製造冠軍待 bake-off | [round-12](round-12-consolidate-diversify.md) |
| 13 | 組數階梯系統對比（3/4/5/6 塊） | ✅ archived（2026-07-08） | 組數是真設計軸但報酬有取捨:4-5塊甜蜜點(5塊買rad/4塊買選擇性)、6塊遞減;margin天花板僅+0.20→+0.22;製造冠軍 x00(72%穩健) | [round-13](round-13-block-ladder.md) |
| 14 | 組件級軸：消融（有無）× 尺寸（大小） | ✅ archived（2026-07-08） | **翼=帶內引擎(+6dB,遞減)且付 rad/帶外**=三標張力機理定位;冠軍在尖銳最優(±1圈=懸崖);細旋鈕=小塊非圈;像素級退役 | [round-14](round-14-component-axis.md) |
| 15 | 對照組實驗：push-button(GA) vs 工具箱(知情) | ✅ archived（2026-07-09） | GA 至少打平=**空間即知識載體**;**換王 i02 +0.29(公證✓)**;g14 +0.40 帶內紀錄(rad未過);理論模板否證新盆地(帶外 4.1=0翼端點) | [round-15](round-15-pushbutton-vs-toolbox.md) |
| 16 | 添加收益圖（治先驗跨算子誤用） | ✅ archived（2026-07-09） | **單塊近全負(空間飽和),唯一正點=r9c11×3×3**(位置×尺寸交互);配對正交互=貪心不夠組合有紅利;g14 rad 兇手=g3;翼修邊/再分配假說雙雙因果否決;x00′ caveat | [round-16](round-16-addition-map.md) |
| 17 | 帶外主目標一批：低側裙擺攻堅（手術+分組/尺寸+公證） | ✅ archived（2026-07-09） | **換王 a024 +0.35(公證3/3)**;低側物理可動(hslot 劑量)但三標內不可負擔=張力非地板;分組=集中≫分散(3小塊−2.6);尺寸峰3×3;**hslot=rad 大旋鈕(+1.7)** | [round-17](round-17-oob-primary.md) |
| 24 | 降根計畫：根多樣性稅×池外梯度×誘因包（治軸相關打轉） | 🔵 running（2026-07-13 b1 重跑） | — | [round-24](round-24-root-diversity.md) |
| 23 | 價值軸主戰：sel_score 選批,壓「可用帶外」（基線 9.09） | ✅ archived（2026-07-13,四批+公證） | **sel 鍵增值成立(O>M 四批)+rad 頭進鍵;可用帶外逼近未穿 9.09=深血統打轉→R24;一輪四紀錄(margin 雙躍 +0.41→+0.49 c18 奪回/rad 王 +1.00/帶外王 8.61)=軸相關枯竭實證**;⚠ 2026-07-13 NAS 事件部分工作已復原(/reconcile) | [round-23](round-23-selectivity-axis.md) |
| 22 | 分布組合批：降王朝比例、六臂分散探索 | ✅ archived（2026-07-12,三批+三公證） | **分散假設成立=一天三公證紀錄（rad m5_054 +0.89/帶外 o6_001 8.61/帶內 h7_010 +0.46）＋26px 死區告破**;C 冷支=新主產線（28→34%）;S 槽鏈首批成立（+0.38 三標）;Q/H 判準收臂=低側蓋棺;oob 選鍵壽終→sel_score 交棒 | [round-22](round-22-distribution-portfolio.md) |
| 21 | 收割管線：隨機生成＋SM 帶外過濾（贏家配方量產） | ✅ archived（2026-07-12,五批 774 筆） | **量產成立（M 臂三標 17-27%）＋帶外王易主 o1_035 8.65（公證3/3）;但 SM 帶外過濾紅利=一次性（O 臂 22→3%）**;馬太效應確診（決勝批因果）＋探索稅止跌;W 死區 0/16;rad 候選 0.89 單次→R22 公證;制度=自癒補測/6夾切片/tier-2 搶佔/gain 儀表 | [round-21](round-21-harvest-pipeline.md) |
| 20 | 模型線終審：演化 vs 隨機＋碎片探索（三代,真值在迴圈） | ✅ archived（2026-07-11） | **SM 有效維度=帶外（③19:10）;①②隨機優+GA 逐代衰退=分布收窄;配方=隨機+帶外過濾**;F 碎片族蓋棺;**雙王易主 r2_016 +0.39/vg0338 8.84（c18 王朝每代1px）**;vg0258 雙穩態+HFSS 敵意血系 | [round-20](round-20-evolution-loop.md) |
| 19 | 模型線一批：王結構變異 806 筆＋SM v5 門檻 | ✅ archived（2026-07-10） | **門檻未過（wm ρ 0.493<0.5）＝作戰區飽和是本質非資料量**;帶外排序 0.334→**0.603** 可用;**rad 王易主 cc_r9s2 +0.62（公證2/2）**;vg0338_c18 帶外 8.84 單次破紀錄待公證 | [round-19](round-19-model-line.md) |
| 18 | 帶外二批：挖礦落地（舊藏公證+低側家族救援+c18 手術） | ✅ archived（2026-07-09） | **b20 假象(+0.32→−0.19,鐵則第二次救命)**;vb43/x20 真(帶外 9.15/9.24 入榜);**低側救援 0/20=粉塵諧振本體定案**;三標內帶外地板≈9.0;模型線觸發 | [round-18](round-18-oob-mining.md) |

## 離線分析（analysis-NN；不佔 round——**慣例 2026-07-03**：round 編號只給燒 HFSS 的實驗，round-06 為慣例前歸檔、不回改）
| # | 主題 | 狀態 | 結論(一句) | 檔 |
|---|---|---|---|---|
| 04 | SM 架構對決（CNN 主幹 × 直接多頭,零 HFSS 四臂） | ✅ archived（2026-07-13 當日完成） | **判準未過（oob ρ +0.048<+0.1）＝架構非瓶頸,資料軸定論第三證**;副產=前瞻波動主因=範圍限制+分布漂移（held-out 0.7+ vs 前瞻 0.1-0.5）;CNN+多頭全指標一致小勝=平靜期升級候補 | [analysis-04](analysis-04-sm-architecture.md) |
| 03 | 歷史真值挖礦：帶外拆側全量回算（1286 pattern） | ✅ archived（2026-07-09 當日完成） | **低側可壓——池頂族 lo −4.5/t09 oob 7.2 破 9 地板,被 rad 蓋牌**（張力拓撲=(wm+低側)↔rad）;舊藏出土 b20_k4 +0.32/9.56 等 4 筆待公證;三標內帶外紀錄修正=c18_sm 9.04;低側統計載體=外緣欄(+0.48)+頂列;★7/13 口徑刷新（`analyze oobnav`,n=5,510）:低側分布複驗成立、載體領跑改中帶、rad 粗特徵無讀數 | [analysis-03](analysis-03-history-mining.md) |
| 02 | 組件尺寸分布 vs 三標（Ricky「塊大小不均」觀察量化） | ✅ archived（2026-07-09 當日完成） | **尺寸不均度對三標全面有害**（cv:wm−0.26/rad−0.26/oob+0.18）;作戰區「主件佔比高傷 wm(−0.38)、翼夠大強利(+0.42)」=金屬分配>主件堆料;過線解=緊配方(main 235-250/翼 73-83)→generator 尺寸先驗;★7/13 口徑刷新（n=5,505）:方向保持強度下修、cv 害處重心移帶外(+0.25)、配方變寬(main 215-250) | [analysis-02](analysis-02-component-sizes.md) |
| 01 | pattern 解剖：地形隨機性 × S11/Gain 結構歸因 | ✅ archived（2026-07-03 當日完成） | **地形非抽獎**（翻1-2px 只動不相關水位的 12-16%,有效半徑~數十翻轉;跨區大跳≈重抽）；**S11←少組+feed連通、Gain←少洞、共同敵人=細碎**（配對: 同Gain下 S11 好=組數−9.5;同S11下 Gain 好=洞−5）→ 解 R3-D 之謎、先驗要分工;★7/13 A 部自家真值重算（`analyze terrain`,n=5,513）:d1-2=漸近線 3%,非抽獎更強 | [analysis-01](analysis-01-pattern-anatomy.md) |

## 研究脈絡（一句話串起來）
generator G = 單 pattern 超特徵 → 轉 **generator-free SM-guided 搜尋**(輸 random)→ **Round 1** 測「是不是 SM 訓練不足」→ **否**(訓飽反而過擬合)→ 病灶是 SM-guided 搜尋本身 → **Round 2** 上文獻治本(ensemble + trust)→ 治本微幅未決定性、且實測搜尋「凍住」(每 epoch 才翻 ~6 像素)→ **Round 3** factorial 測「探索(lr↑)× DIP(generator 帶回來連通先驗)」的效果與加乘 → 健檢發現三臂共同瓶頸＝**SM 欠訓**(dlf 每輪只訓 1 epoch → 樂觀、trust 鎖 0.05 不利用) → **Round 4** 上**自適應 SM 訓練量**(held-out fresh 點自調每輪重訓 epoch 數)修這個瓶頸、重跑 E/D/E+D 去 confound → 中檢:E+D **破紀錄 -2.89**(探索撞到)但實錘**深度欠訓**(fit_loss 仍 8-11)+探測自鎖(target 3-5) → **Round 5** 上**滑動視窗訓練量**(訓到視窗頂+argmin 貼邊 ×2/÷2,Ricky 設計)把 SM 真的訓起來。平行地,**Round 6**(離線、零 HFSS)把我們/學長/random 三塊歷史放同一把尺:期望爬升到不了 spec、達標 pattern 已在 harvest 池內、**分布≫策略** → 「池頂端 warm-start」升候選 → **R7/R8** 實錘「粉塵=共振一部分、乾淨解要構造不能修」＋池值漂移警訊 → **R9**(過夜 162 筆)裁決 oracle 活著(8/18)、**可製造紀錄 −2.68→−0.29**(F2×10-5-10 對稱化)＋F3 沃土＋SM 排序訊號 → 下一步:精修 round(s05 補 Gain/g24 補 wm)× R10 設計規律目錄。
