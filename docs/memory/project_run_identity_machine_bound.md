---
name: project_run_identity_machine_bound
description: "結果夾名把機器 IP 末段寫進 {device} → 同一 config 換機器跑會「重頭來」、不能跨機接力續跑"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2adb79f3-22b2-4395-bf20-4dc71c4ffd49
---

`get_result_path`（`antenna/utils/run_setup.py:53`）的結果夾名格式是
`[Patch-{port}-{device}-{hash_id}] {name}`，其中：
- `{device}` = `get_local_ip().split('.')[-1]`（**本機 IP 末段** → 每台機器不同）。
- `{hash_id}` = `get_shake_128(模板字串)`（對 config 名雜湊 → **跨機相同**）。

**後果（2026-06-20 實際踩到）**：A 機（IP .37、DESKTOP-OVOFT76）跑 `single_sc_rad` 到 epoch 57 後關機；
B 機（IP .216、6GAntennaPC）想接力 → 因為 device 從 .37 變 .216 → 夾名不同 → `Continue=False` → **從 epoch 1 重頭來**。
A 機的 epoch-57 進度沒丟、完整健康躺在 NAS（sm.pth 非有限=0、max|w|=0.308、55 個不同 pattern → GEN 真在探索），
只是 B 機看不到。**run 身分被綁死在機器 IP 上、無法跨機接力**。

**對「三台集中派工」的意義**：這是派工的前置障礙。要嘛續跑判斷拿掉 `{device}`（夾名只用 config hash），
要嘛用「誰先認領誰負責」的 handoff。臨時救援：把舊夾改名成新機 IP 段即可被接力。

順帶：A 機關機時 status.json 停在 `state=running`（沒寫成 crashed/finished）→ 這就是 app.py 健康偵測要靠
「心跳新鮮度」判 stale 的原因。相關 [[project_monitoring_tensorboard]]。

**2026-07-06 更新**:跨機重複公證(R9 §4 附錄)證明兩台正式機 HFSS 結果 bit 級一致 → 「換機不能接力」確定只是結果夾命名問題(夾名含 {device}),非物理問題;要跨機接力改夾名解析即可。
