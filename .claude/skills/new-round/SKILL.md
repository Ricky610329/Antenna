---
name: new-round
description: 開新 research round 的清單——取號、開檔、判準寫死檢查表、命名規範、ONGOING/README/scratch 同步（用法：/new-round <一句假設>）
---

# 開新輪 $ARGUMENTS

> 前提：上一輪已 `/close-round` 結掉（過厚警訊：§4 超 ~3 批或主題漂出 §1＝先結輪）。
> 自主續輪授權（decisions 2026-07-12）＝宣告制：開輪不等核准，但假設與判準要在對話裡宣告，
> 使用者可隨時否決。

## 清單

1. **`date` 查時間**（所有時間戳用查到的值）。
2. **取號**：`docs/log/README.md` 時間軸表最大 NN ＋1（單一編號宇宙，嚴格遞增）。
3. **開檔**：`docs/log/round-NN-<slug>.md`，用 `_TEMPLATE.md` 骨架（批次線 round 可精簡 §2 的
   epoch 欄，改臂×判準表）。狀態 `proposed`（發車時改 `running`）。
4. **§1 判準寫死檢查表**（發車前全填，缺一不發車）：
   - [ ] 主指標＋基線值（紀錄門檻**引 `docs/records.json`，不抄死數字**——寫「wm 紀錄見 records.json」）
   - [ ] 停止/回報線（如「主指標連三批零推進 → 回報」）
   - [ ] 每條臂的存活/畢業判準（一般臂 6% 隨機基準；**分布外探索臂走學費預算制**——
         固定 N 批、KPI=進步趨勢、達標=畢業、預算盡=回報裁決，見 decisions 漸進式成長條款）
   - [ ] 紀錄級一律公證（鐵則,引 /notarize）
   - [ ] 判準發車後要修訂：只能在結果回來前＋留日期註記
5. **命名規範**（R23 起）：夾 `dedust_r<NN>b<批><夾>`、id `<臂字母><NN>b<批>_<序>_<親>`、
   公證 `r<NN>n*`、填空池 `r<NN>g*`——round 號貫穿一切產物。
6. **§2/§3**：臂×配額×判準表；發車指令 code block（含 check-dup、公證、補池——斷電接手唯一真相）。
7. **ONGOING**：新輪加 🔵 條目（一句 what＋判準指向）；上一輪確認已是 ✅。
8. **掃 scratch**（round 邊界例行）：`docs/discuss/scratch.md` 全檔——熟的升 ONGOING 🔜、
   死的標 ❌＋為什麼、已用的標 ✅ 去處。
9. **README 索引**：`docs/log/README.md` +1 行（🔵 running）；「最後更新」改今天。
10. **發車**＝走 `/batch-cycle` 的 ④⑤⑥（select → check-dup → jobs-add → 補池 → watch）。
11. commit（`docs(log): RNN 開輪 — <假設一句>`）＋push；對話裡宣告假設與判準（宣告制）。
