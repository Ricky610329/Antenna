# -*- coding: utf-8 -*-
"""script/diffsim/head.py — 殘差頭：physics-anchored SM（`docs/diffsim.md` §4）。

把 diffsim 的輸出當「物理錨」餵給一個小網路，讓它只學殘差（diffsim → HFSS 的差）。
**公平比較的設計**：同一份訓練資料、同一個架構、同一組超參，唯一差別是有沒有吃 diffsim 那 34 維
（`use_phys`）。這樣「diffsim 有沒有加值」是被隔離出來的，不需要去讀正式機的 SM checkpoint
（那是另一個 session 的資產，本作業唯讀不碰）。

⚠ 這裡的「純 SM 對照」是**同架構同資料的對照組**，不是正式線上的 `sm_reanchor*.pth`。
正式 SM 的數字（held-out |Δwm| 中位 0.61dB、凍結尺 0.47–0.56dB、批次前瞻 ρ 0.7–0.9）
訓練資料量與口徑都不同，只能當座標、不能直接比大小。
"""
import numpy as np
import torch
import torch.nn as nn


class ResidualHead(nn.Module):
    """25×25 pattern（＋可選的 diffsim 34 維）→ 響應 34 維。

    :param use_phys: 吃不吃 diffsim 輸出。False ＝ 對照組（純資料驅動）。
    """

    def __init__(self, use_phys: bool = True, width: int = 64, dtype=torch.float32):
        super().__init__()
        self.use_phys = use_phys
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 12
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 6
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(2),
        )
        nin = 64 * 4 + (34 if use_phys else 0)
        self.mlp = nn.Sequential(nn.Linear(nin, 4 * width), nn.ReLU(),
                                 nn.Linear(4 * width, 4 * width), nn.ReLU(),
                                 nn.Linear(4 * width, 34))
        self.to(dtype)

    def forward(self, x, phys=None):
        h = self.conv(x.reshape(-1, 1, 25, 25)).flatten(1)
        if self.use_phys:
            h = torch.cat([h, phys], 1)
        out = self.mlp(h)
        #? 有物理錨時輸出的是**殘差**（加回 diffsim），沒有時就是直接回歸。
        return out + phys if self.use_phys else out


def train_head(x, phys, y, *, use_phys=True, epochs=60, batch=128, lr=1e-3,
               seed=0, val_frac=0.1, verbose=True):
    """回 (model, 內部 held-out 的 |Δwm| 中位)。x (n,625) / phys (n,34) / y (n,34)。"""
    from .eval import margins
    torch.manual_seed(seed)
    n = len(x)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    nv = max(32, int(n * val_frac))
    vi, ti = perm[:nv], perm[nv:]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = torch.as_tensor(x, dtype=torch.float32, device=dev)
    P = torch.as_tensor(phys, dtype=torch.float32, device=dev)
    Y = torch.as_tensor(y, dtype=torch.float32, device=dev)
    #? 物理輸入標準化（S11 與 Gain 的動態範圍差很多，不標準化會讓 MLP 只看得到一邊）
    mu, sd = P[ti].mean(0), P[ti].std(0).clamp_min(1e-3)
    Pn = (P - mu) / sd
    m = ResidualHead(use_phys=use_phys).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lossf = nn.HuberLoss(delta=3.0)
    for ep in range(epochs):
        m.train()
        idx = ti[torch.randperm(len(ti)).numpy()]
        for i in range(0, len(idx), batch):
            b = idx[i:i + batch]
            #? 殘差模式下網路加回的是**原始** phys（不是標準化後的），輸入才用標準化版
            pred = m(X[b], Pn[b] if use_phys else None)
            if use_phys:
                pred = pred - Pn[b] + P[b]
            loss = lossf(pred, Y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sch.step()
        if verbose and (ep % max(1, epochs // 6) == 0 or ep == epochs - 1):
            m.eval()
            with torch.no_grad():
                pv = m(X[vi], Pn[vi] if use_phys else None)
                if use_phys:
                    pv = pv - Pn[vi] + P[vi]
            e = np.abs(margins(pv.cpu().numpy())[0] - margins(y[vi])[0])
            print(f"    ep {ep:3d} loss {float(loss):.3f}  held-out |Δwm| 中位 {np.median(e):.3f}",
                  flush=True)
    m.eval()
    with torch.no_grad():
        pv = m(X[vi], Pn[vi] if use_phys else None)
        if use_phys:
            pv = pv - Pn[vi] + P[vi]
    err = float(np.median(np.abs(margins(pv.cpu().numpy())[0] - margins(y[vi])[0])))
    m._norm = (mu.cpu().numpy(), sd.cpu().numpy())
    return m, err


def predict_head(m, x, phys):
    dev = next(m.parameters()).device
    mu, sd = m._norm
    X = torch.as_tensor(x, dtype=torch.float32, device=dev)
    P = torch.as_tensor(phys, dtype=torch.float32, device=dev)
    Pn = (P - torch.as_tensor(mu, device=dev)) / torch.as_tensor(sd, device=dev)
    with torch.no_grad():
        out = m(X, Pn if m.use_phys else None)
        if m.use_phys:
            out = out - Pn + P
    return out.cpu().numpy()
