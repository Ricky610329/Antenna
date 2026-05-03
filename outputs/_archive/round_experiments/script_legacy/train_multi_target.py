"""Plan D — Multi-target generator training（繞過 Trainer 的 minimal 實作）。

**動機**：v1–v4 的 generator 都對所有 target 給相同 pattern，conditioning
failure。診斷根因為：原 Trainer 每 epoch 只用一個固定 target 做訓練 — generator
從未看過多樣輸入，當然學不會把 input 分類到對應 pattern。

**實作**：每個 train step 從 target pool 隨機抽一個 target，generator + RIS
simulator + tolerance loss 一條路反傳。surrogate 凍結（用結構化版 v4）。

用法：
    python script/train_multi_target.py --epochs 200 --device cuda:0
"""

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from torch import nn

from antenna.core.pattern import AntennaPattern
from antenna.core.response import AntennaResponse
from antenna.models.autograd import BinarySTE
from antenna.models.generators.biased_gumbel_sigmoid_gen import BiasedGumbelSigmoidGEN
from antenna.ris import RISSimulator, custom_loss_tolerance
from antenna.utils.config import config


def make_target_pool(n_targets: int = 32, response_size: int = 361, seed: int = 0) -> torch.Tensor:
    """產生 N 組變化 target — plateau 位置 / 寬度都不同。"""
    rng = np.random.default_rng(seed)
    targets = []
    for _ in range(n_targets):
        plateau_w = int(rng.uniform(20, 70))
        center_pos = int(rng.uniform(80, response_size - 80))
        left_w = max(0, center_pos - plateau_w // 2)
        right_w = max(0, response_size - left_w - plateau_w)
        # 梯形 mask: side(-20) → plateau(0) → side(-20)
        side, center = -20.0, 0.0
        t = np.concatenate([
            np.full(left_w, side, dtype=np.float32),
            np.full(plateau_w, center, dtype=np.float32),
            np.full(right_w, side, dtype=np.float32),
        ])[:response_size]
        if len(t) < response_size:
            t = np.pad(t, (0, response_size - len(t)), constant_values=side)
        targets.append(t)
    return torch.tensor(np.stack(targets))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--element_num", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--n_targets", type=int, default=32, help="target pool 大小")
    parser.add_argument("--binary_mode", action="store_true", default=False,
                        help="是否套 BinarySTE（預設 False = 連續訓練 + 後處理量化）")
    parser.add_argument("--cond_reg_weight", type=float, default=0.5,
                        help="conditional regularizer 權重 — 強迫不同 target 給不同 logits")
    parser.add_argument("--surrogate_path", type=str,
                        default="result/_pretrained_surrogate_v4/checkpoint/sm.pth")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="result/RIS-multi-target")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.device:
        config.device = args.device
    elif torch.cuda.is_available():
        config.device = "cuda:0"
    logger.info(f"裝置：{config.device}, binary_mode={args.binary_mode}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 必須先 setup pattern/response 才能初始化 generator
    AntennaPattern.setDefaultCoordinate((0, args.element_num, 0, args.element_num))
    AntennaResponse.registerLabels("response", x="ris")
    AntennaResponse.target(side=-20.0, center=0.0, width=(140, 0, 40, 0, 181), label="response", add=True)

    sim = RISSimulator(element_num=args.element_num)

    # Surrogate（凍結）— 用 OldSM 載入結構化版本
    from antenna.smodels import HFSSNet
    surrogate = HFSSNet(args.element_num * args.element_num, AntennaResponse.size())
    sm_path = Path(args.surrogate_path)
    if sm_path.exists():
        sd = torch.load(sm_path, map_location=config.device, weights_only=False)
        surrogate.load_state_dict(sd)
        logger.info(f"載入 surrogate: {sm_path}")
    else:
        logger.warning(f"找不到 {sm_path}, surrogate cold start")
    surrogate.to(config.device)
    for p in surrogate.parameters():
        p.requires_grad = False
    surrogate.eval()

    # Generator
    gen = BiasedGumbelSigmoidGEN().to(config.device)
    optimizer = torch.optim.Adam(gen.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Target pool
    target_pool = make_target_pool(args.n_targets, response_size=361, seed=args.seed).to(config.device)
    logger.info(f"target pool: {tuple(target_pool.shape)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoint"
    ckpt_dir.mkdir(exist_ok=True)

    history = {"step": [], "fake_loss": [], "real_loss": [], "supp_avg": []}
    best_supp = -np.inf

    for ep in range(1, args.epochs + 1):
        gen.train()
        # 隨機抽 target
        idx = int(np.random.randint(0, args.n_targets))
        target = target_pool[idx]

        soft = gen(target)
        if args.binary_mode:
            pat = BinarySTE.apply(soft).reshape(args.element_num, args.element_num)
        else:
            pat = soft.reshape(args.element_num, args.element_num)
        pat_flat = pat.flatten().unsqueeze(0)  # surrogate 吃 (B, P)

        # surrogate 預測 (用 surrogate 給 generator 梯度)
        resp_pred = surrogate(pat_flat).squeeze()
        # 確保 shape 對得上 target (361,)
        if resp_pred.dim() > 1:
            resp_pred = resp_pred.squeeze()

        loss = custom_loss_tolerance(
            resp_pred, target,
            sidelobe_threshold=-25.0,
            main_target=0.0,
            main_weight=5.0,
        )

        # binary_balance penalty 仍開
        loss = loss + 0.5 * (soft.mean() - 0.5).abs()

        # ── Conditional regularizer ──
        # 隨機抽另一個對比 target，要求兩者的 logits 距離夠大；penalize 1/(distance+ε)
        # 用 logits（pre-sigmoid）而非 soft 比，避免 sigmoid 飽和把訊號壓掉
        if args.cond_reg_weight > 0:
            idx2 = int(np.random.randint(0, args.n_targets))
            while idx2 == idx:
                idx2 = int(np.random.randint(0, args.n_targets))
            target2 = target_pool[idx2]
            _ = gen(target2)
            logits2 = gen.logits.detach()  # detach 避免雙倍 grad path
            # 重新 forward target1 取 logits1（gen.logits 被 target2 的 forward 蓋掉）
            _ = gen(target)
            logits1 = gen.logits
            l2_dist_sq = ((logits1 - logits2) ** 2).mean()
            # reward 距離：penalty = w / (dist + eps)
            cond_loss = args.cond_reg_weight / (l2_dist_sq + 0.1)
            loss = loss + cond_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history["step"].append(ep)
        history["fake_loss"].append(float(loss.item()))

        # 每 10 步用真 RIS 算 real_loss 監控；每 50 步算所有 target 的 avg suppression
        if ep % 10 == 0:
            with torch.no_grad():
                hard = (soft > 0.5).float().reshape(args.element_num, args.element_num)
                real_resp = sim(hard)["response"]
                real_loss = custom_loss_tolerance(
                    real_resp, target,
                    sidelobe_threshold=-25.0,
                    main_target=0.0,
                    main_weight=5.0,
                )
            history["real_loss"].append(float(real_loss.item()))

        if ep % 50 == 0:
            with torch.no_grad():
                gen.eval()
                supps = []
                for ti in range(args.n_targets):
                    t = target_pool[ti]
                    s = gen(t)
                    h = (s > 0.5).float().reshape(args.element_num, args.element_num)
                    r = sim(h)["response"].cpu().numpy()
                    main_mask = (t.cpu().numpy() == t.max().item())
                    side_mask = (t.cpu().numpy() == t.min().item())
                    if main_mask.any() and side_mask.any():
                        mp = float(r[main_mask].max())
                        sm = float(r[side_mask].max())
                        supps.append(mp - sm)
                avg_supp = float(np.mean(supps))
                history["supp_avg"].append(avg_supp)

                if avg_supp > best_supp:
                    best_supp = avg_supp
                    torch.save(gen.state_dict(), ckpt_dir / "best.pth")

                logger.info(
                    f"ep {ep:4d}/{args.epochs} | fake_loss={loss.item():.2f} | "
                    f"avg_suppression={avg_supp:+.2f} dB | best={best_supp:+.2f}"
                )

    # 結束評估：載入 best ckpt（不是 final），對 10 個 sample target 計算 suppression
    best_path = ckpt_dir / "best.pth"
    if best_path.exists():
        gen.load_state_dict(torch.load(best_path, map_location=config.device, weights_only=False))
        logger.info(f"評估時載入 best ckpt: avg_suppression={best_supp:+.2f} dB")
    gen.eval()
    rows = []
    sample_targets = make_target_pool(10, response_size=361, seed=args.seed + 999).to(config.device)
    for i in range(10):
        t = sample_targets[i]
        with torch.no_grad():
            s = gen(t)
            h = (s > 0.5).float().reshape(args.element_num, args.element_num)
            r = sim(h)["response"].cpu().numpy()
        main_mask = (t.cpu().numpy() == t.max().item())
        side_mask = (t.cpu().numpy() == t.min().item())
        if main_mask.any() and side_mask.any():
            mp = float(r[main_mask].max())
            sm = float(r[side_mask].max())
            rows.append((i + 1, float(h.mean()), mp, sm, mp - sm))

    summary = ["# Multi-target generator 評估（10 個 sample target）",
               "",
               "| # | on% | main_peak dB | side_max dB | suppression dB |",
               "|---|-----|--------------|-------------|----------------|"]
    for i, on, mp, sm, sup in rows:
        summary.append(f"| {i} | {on:.0%} | {mp:.2f} | {sm:.2f} | {sup:+.2f} |")
    sup_vals = [r[4] for r in rows]
    summary.append("")
    summary.append(f"**suppression mean**: {np.mean(sup_vals):+.2f}, "
                   f"min={np.min(sup_vals):+.2f}, max={np.max(sup_vals):+.2f}")
    summary.append(f"\n對照：v3 −2.21 / v4 phase1+postq −1.63 / direct GD upper +3.05")

    text = "\n".join(summary)
    (out_dir / "summary.md").write_text(text, encoding="utf-8")
    print(text)

    # 存最終 history
    with open(out_dir / "history.pkl", "wb") as f:
        pickle.dump(history, f)
    logger.success(f"訓練完成 → {out_dir}/  (best avg suppression: {best_supp:+.2f} dB)")


if __name__ == "__main__":
    main()
