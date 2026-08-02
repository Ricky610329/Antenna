"""script/diffsim/ — 可微模擬器（differentiable EM surrogate）。

定位見 `docs/diffsim.md`：**要輪廓不要復現**——目標是排名器 + 梯度產生器，真相仍由 HFSS 公證。
與 SM 同構，差別是外推力來自物理結構而非訓練集覆蓋。

模組：
    data.py   掃 NAS 建索引 + 決定性分割（驗證/擬合不相交鐵則）
    l1.py     L1 可微廣義腔模型（2D Helmholtz 特徵問題 + 周邊磁流遠場）
    eval.py   同一把尺的評分與 rank ρ 驗收
"""
