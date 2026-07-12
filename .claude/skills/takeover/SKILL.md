---
name: takeover
description: 新 session 接手定位儀式——五步搞清楚「現在在迴圈哪一步」再行動；任何模型接手先跑這個（用法：/takeover）
---

# 接手定位

> 前一個 session 的監看（Monitor）隨 session 死掉了,但 worker 是 NAS 自治的——機器沒停,
> 只是「腦」需要重新對位。**不要憑記憶行動,先跑完這五步。**

## 五步定位

1. **工廠真相**：
   ```
   python -m script.status --factory
   python -m script.dedust jobs-ls
   ```
   → 有 `.fail`？先處理（/batch-cycle 分支表）。有在跑的批？記下 round/batch。

2. **讀三份現況**（順序固定）：
   `configs/ONGOING.md`（live 板:哪輪在跑/候選區）→ 最新 `docs/log/round-NN-*.md`
   （§1 判準＋§3 執行表＝本輪唯一真相）→ `docs/discuss/scratch.md` 尾段（最近的半熟觀察）。

3. **對位迴圈狀態**（三選一）：
   | 狀態 | 判斷依據 | 動作 |
   |---|---|---|
   | 批在跑 | jobs-ls 有 claim 中的 rNNbN | 掛回偵測 `Monitor: python -m script.dedust watch --stores ...`,等收檔走 `/batch-cycle` |
   | 批已收、未判讀 | jobs_state 全 done 但 round 檔 §4 沒該批 | 直接 `/batch-cycle <R> <N>` |
   | 佇列空轉（只剩自產） | jobs-ls 無 tier-1 | 上一批判讀做了沒？做了→發下批（/batch-cycle ④⑤）;判準觸發回報線→`/stall-protocol` |

4. **檢查未完成的公證**：`dedust_r<NN>n*` 有 done 但 round 檔 §4 沒判定 → 走 `/notarize`。

5. **授權與紀律提醒**（讀 memory 即載,這裡只列鍵名）：自主續輪=宣告制（[[feedback-autonomous-rounds]]）、
   價值軸=帶外主鍵（[[feedback-value-axis-oob]]）、L0 常升目標必報（[[project-strategy-data-flywheel]]）、
   紀錄門檻=讀 `docs/records.json` 不憑記憶。

## 禁止
- 不重問使用者「現在要做什麼」——NAS＋round 檔就是答案。
- 不在定位完成前生成/派工/改 records。
