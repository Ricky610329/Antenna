"""
script/train_sm_offline.py — 用離線資料 (harvest_single) 訓一顆 SM 當「好的初始化」。
產出=sm_harvest.pth;後續乾淨區續錨走 `script/sm_reanchor.py`（同族譜:初訓→重錨）。

背景：量過 old_sm.pth 對我們的 harvest 資料預測 ≈ 隨機 (中位 MSE 38、且吐 +40/-88dB 亂值)，
等於沒暖啟動、線上學習每次從近乎隨機重學、浪費早期昂貴 HFSS 評估。這支在 harvest 上
minibatch 訓一顆 fits 我們資料的 SM，存成 pre_load_model 可載的檔，當新的 pretrained。

開發機即可跑 (不需 HFSS、純 compute)。直接 minibatch (繞過 forward 的單樣本 reshape，
用 fc_patch + 自己 reshape)，比閉迴路的單樣本擬合快很多。

用法：
    python -m script.train_sm_offline                              # harvest_single → sm_harvest.pth
    python -m script.train_sm_offline --n 0 --epochs 60            # 用全部資料
    python -m script.train_sm_offline --dataset harvest_dual --response 3,17 --out sm_harvest_dual.pth
"""
import argparse
import random
import time
import tempfile

import torch
import torch.nn.functional as F

from antenna.utils import config, DATASET_PATH
config.device = "cpu"

from antenna import AntennaPattern
from antenna.models.surrogates import MLPSurrogate
from antenna.utils.store import SampleStore


def main():
    ap = argparse.ArgumentParser(description="離線訓練 SM 初始化權重 (不需 HFSS)")
    ap.add_argument("--dataset", default="harvest_single", help="DATASET_PATH 下的 SampleStore 名稱")
    ap.add_argument("--out", default="sm_harvest.pth", help="輸出檔名 (存到 DATASET_PATH)")
    ap.add_argument("--n", type=int, default=8000, help="取樣筆數 (0=全部；多=更準但載入慢)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--response", default="2,17", help="響應形狀 (單埠 2,17 / 雙埠 3,17)")
    args = ap.parse_args()

    AntennaPattern.setDefaultCoordinate((0, 25, 0, 25))
    rs = tuple(int(v) for v in args.response.split(","))

    store = SampleStore(DATASET_PATH.joinpath(args.dataset), verbose=False)
    idx = list(range(len(store)))
    random.seed(0); random.shuffle(idx)
    if args.n and args.n < len(idx):
        idx = idx[:args.n]
    print(f"{args.dataset}: 共 {len(store)} 筆，取 {len(idx)} 筆載入記憶體 (NAS I/O)…")
    t0 = time.time()
    X = torch.stack([store[i][0].reshape(-1).float() for i in idx])     # (N, 625)
    Y = torch.stack([store[i][1].float() for i in idx])                 # (N, *rs)
    print(f"  載入完成 {time.time()-t0:.0f}s, X{tuple(X.shape)} Y{tuple(Y.shape)}")

    #? 品管：剔除非有限 / 絕對值過大 (壞樣本，dB 不該超過 ~100)
    mask = torch.isfinite(Y).all(dim=(1, 2)) & (Y.abs() < 100).all(dim=(1, 2))
    X, Y = X[mask], Y[mask]
    print(f"  品管後 {len(X)} 筆 (剔除 {int((~mask).sum())})")

    nval = max(1, len(X) // 20)                                         # 切 5% 當 val
    Xtr, Ytr, Xv, Yv = X[nval:], Y[nval:], X[:nval], Y[:nval]

    sm = MLPSurrogate(tempfile.mkdtemp(), 625, rs)                      # fresh HFSSNet + Ranger
    model, opt = sm.model, sm.optimizer

    print(f"訓練 {args.epochs} epochs (batch={args.batch})…")
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for b in range(0, len(Xtr), args.batch):
            bi = perm[b:b + args.batch]
            out = model.fc_patch(Xtr[bi]).reshape(-1, *rs)             # 繞過 forward 的單樣本 reshape → 可 minibatch
            loss = F.mse_loss(out, Ytr[bi])
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 5 == 0 or ep == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                vmse = F.mse_loss(model.fc_patch(Xv).reshape(-1, *rs), Yv).item()
            print(f"  ep{ep:>3}: val MSE = {vmse:.3f}")

    out_path = DATASET_PATH.joinpath(args.out)
    sm.save_as(out_path)
    model.eval()
    with torch.no_grad():
        vmse = F.mse_loss(model.fc_patch(Xv).reshape(-1, *rs), Yv).item()
    print(f"\n已存：{out_path}")
    print(f"新 SM val MSE = {vmse:.3f}   (對照：old_sm.pth 在同資料 ≈ 38、≈隨機)")
    print("→ config 用 `surrogate.pretrained: " + args.out + "` 即可換成這顆初始化 (A/B 對標 old_sm.pth)")


if __name__ == "__main__":
    main()
