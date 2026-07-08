# docs/reference — 外部文獻索引

> 一篇一節：出處、核心方法、對本專案的用途。三篇皆 Princeton Sengupta 組系列作（2024→2026），
> 合讀＝該組從「像素化逆設計」走到「人機協作端到端合成」的完整軌跡。
> 對照本專案的行動候選在 `docs/discuss/scratch.md`（2026-07-08 塊）、敘事定調在 `docs/discuss/decisions.md`。

## 2024 MWSCAS — Deep Learning Enabled Design of RF/mmWave IC and Antennas

- **出處**：IEEE MWSCAS 2024, pp.769–772（邀請型 overview，整合該組 2022–2024 工作）。
- **方法**：像素化格點（12×12 天線／16×16 on-chip）＋ CNN 正向代理（結構影像→S 參數）＋
  雙保真 transfer learning（粗網格大批量預訓→細網格小批量 fine-tune）＋ GA 族群搜尋
  （4096×100 代全走 CNN，每個新目標只換 cost function、合成 <5 分鐘）。實測 SiGe PA 30–94 GHz。
- **對我們**：立場是**全自動 push-button**（明寫排除 manual human intervention）——當新敘事的**對照組/立靶**。
  可借：雙保真 warm-start 配方、「一個 forward model 多 task 只換 cost」的工具化論據、
  離散空間適配族群算法（GA baseline 引據）。無不確定性門控、無 active learning、無 human-in-the-loop。

## 2026 SSC Magazine — AI for RFIC Design: Early Advancements, Opportunities, and Challenges

- **出處**：IEEE Solid-State Circuits Magazine, Spring 2026, pp.52–68（Sengupta 組立場文/五年回顧）。
- **方法**：一次性離線大投資訓 forward 代理（250K 粗＋90K 細 transfer learning，訓完推論毫秒級）；
  逆設計三路線（代理+GA／tandem NN／controlled diffusion "DALL·EM" 用 prior 控制產物風格）；
  RL 端到端合成（ISSCC 2025 PA）；快速合成掃 PDK 近似 Pareto front。
- **對我們**：**敘事級資產（最重要的一篇）**。
  ①「metaheuristic 不會學習、每次從零」＝我們輸 random 的文獻級解釋，解法＝攤提式預訓＝harvest warm-start 放大版；
  ② controlled diffusion 的「結構 prior 收斂搜尋空間」與 R14 組件級軸殊途同歸；
  ③ **human-in-the-loop 是官方終局**：Fig 1(b) "Foundational Model driven flow with Human Inputs"、
  L1–L5 自主性光譜類比（p.55）、"AI is a collaborator, not merely a brute-force optimizer"（pp.65–66）、
  "current strengths: accelerating exploration, exposing hidden tradeoffs, offloading repetitive optimization"（p.66）。
  ⚠ 文中 "agent" 指 RL agent 非 LLM agent；其人機協作＝設計師操縱 prior/規格，非對話式迭代。

## 2026 ISSCC 13.2 — AI-Enabled End-to-End Design in RFICs with Controllable Architectural Style

- **出處**：ISSCC 2026, Session 13 "Circuits for AI and AI for Circuits", paper 13.2（3 頁 digest）。
- **方法**：spec→GDS 端到端 LNA 合成；**架構風格（classical t-line vs pixelated）作為設計者輸入**；
  PPO 做架構/電路搜尋（35 萬樣本、192 核 24hr 離線攤提）＋ EM 代理＋逆 EM 設計；
  Pareto canvassing；實測 24–160 GHz 兩顆晶片。
- **對我們**：**「風格軸」框架直接支撐組件級定位**——同一組 S 參數下 pixelated 與 classical 性能近乎等價
  （Fig 13.2.3，風格是自由度不是代價）；「spec→design 一對多，最終取捨由人的主觀因素決定，
  故架構演化可控性是 AI 設計被採用的前提」（p.1）＝敘事的 ISSCC 版原句。
  我們可加碼：少樣本 regime 下組件級**更有效**（R14 實證），比他們「等價」更強。
  dedust 批次線＝天線版 canvassing。⚠ 他們的 human-in-the-loop＝事前注入約束＋事後挑選，
  無互動迭代、無 "agent" 一詞——我們的 agent+human 迭代迴圈是往前一步（novelty 空間）。

## 三篇合讀的共同邊界（防 oversell / 防審稿人）

- 三篇全是**離線攤提 regime**（數十萬樣本、HPC 天級預訓，訓一次服務整個 PDK）；
  我們是 per-task 少樣本線上 regime（數百次 HFSS）——RL/pixel 級逆設計在我們預算下不成立，
  引用時主動劃清這條線（同時回答「為何不用 RL」與「niche 在哪」）。
- 三篇都**沒有**碰 surrogate 不確定性/信任域門控——他們用海量資料把代理訓到夠準、繞開「SM 何時可信」。
  我們的治本方向（不確定性門控＋active learning＋warm-start）不因這三篇改變。
