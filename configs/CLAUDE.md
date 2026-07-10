# configs/ 規範

- **一個 `*.yaml`＝一組實驗**；模型用名字選（`antenna/zoo.py`）；不改 code 加實驗。
- **硬規則**：新增/修改任何 config 或訓練腳本 → **同步更新 [README.md](README.md) 對照表**
  （一行＝測什麼／與 base 差在哪／舊編號）；產生新實驗前先掃表防重複。這不是順手做，是硬規則。
- **[ONGOING.md](ONGOING.md)＝live 操作板**：只放「現在在跑/待跑」＋重啟指令；
  跑完結論進 `docs/log/` round 檔，這裡只留一行 ✅ 指標。
  **更新 run 狀態前先跑 `python -m script.status`**（掃 NAS 真相，別手動猜）。
- 候選/待排區（🔜）條目格式：一句 what＋why＋**觸發條件**；升格自 `docs/discuss/scratch.md`。
