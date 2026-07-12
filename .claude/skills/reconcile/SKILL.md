---
name: reconcile
description: 狀態對帳——交叉驗證 git／NAS／records 三方一致,防「在未落地的狀態上繼續蓋」（2026-07-13 持久化事件教訓）。接手時、切模型/長自動跑後、宣稱換王前、感覺不對時跑
---

# 狀態對帳

> **教訓（2026-07-13）**：一段 Opus session 的 commit 沒進 reflog、T: 產物被回退,但工具當下都回報成功;
> 我在**未驗證**的狀態上繼續蓋了好幾層（margin 王→R23 收輪→R24 發車）才發現。
> **鐵律：關鍵操作（commit 換王/重錨/派工）後,先驗證真的落地,再往上蓋。不信工具回報,信 ground truth。**

## 對帳清單（任一項紅 → 停,先修再繼續）

### A. git 落地
```
git log -1 --oneline          # 最後 commit 是不是你剛做的那個?
git status --short            # 該 commit 的檔有沒有還掛在未追蹤/modified?
git log origin/GAN -1 --oneline   # 推上去了嗎?本地 HEAD == origin?
```
- 剛 commit 完但 `git log -1` 不是它 → **commit 沒落地,重做**。
- HEAD ≠ origin → push 沒成,重推。

### B. NAS 產物落地
```
ls -t "T:/…/dataset/sm_reanchor"*.pth | head -1   # 最新 SM 版 == round 檔 §3 剛用的?
python -m script.dedust jobs-ls                    # 剛派的批在不在佇列?
```
- 剛重錨 `sm_reanchorN.pth` 但檔案不存在 → **重錨沒落地,重跑**。
- 剛 jobs-add 的 store 不在 jobs-ls → **派工沒落地,重派**。

### C. records ↔ champions ↔ round 一致
```
cat docs/records.json                    # 現任王 id/value
grep "王" docs/champions.md | head -6    # 散文表同一批 id/value?
```
- records.json 的王 ≠ champions.md 頭銜表 → **記帳半途,補齊**（先改 records、散文跟上,見 /notarize）。
- 宣稱「公證 3/3」但公證 store（rNNnX）不存在 → 只剩單次,**重公證或改口徑**。

### D. 資料健檢
```
python -m script.analyze data
```
- 「未收全/壞損」「查重洩漏」非零 → 先查。唯一樣本數當 L0 常升目標基準。

## 何時跑
- **接手**（/takeover 內含）;**切模型／長自動跑後**;**宣稱換王/破紀錄前**（避免記到沒落地的數字）;
  **任何「工具回報成功但後續對不上」的直覺**。
- 批次線每 3-4 批順手跑一次 A+B（零成本,防狀態漂移累積）。

## 若發現不一致（復原原則）
1. **git 有的最權威**（reflog 不會騙;`git reflog` 找懸掛 commit 可 cherry-pick 救）。
2. **NAS 原始資料 > 記憶**：實驗數據在 store 裡,重讀勝過相信對話記憶。
3. **決定性可重生**：select-* 同 seed 同輸出、train 有 manual_seed——丟的產物多半可重跑補齊,不是不可逆。
4. 補齊時**先驗證每步落地**再下一步（別再一次疊未驗證狀態）。
