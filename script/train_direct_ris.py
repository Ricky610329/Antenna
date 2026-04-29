"""Plan E — Multi-target generator training **直接用可微 RIS sim**（不經 surrogate）。

**動機**：v1-v6 的 fake_loss 全程不下降，hamming distance ~0%，根因是 surrogate
給 generator 的梯度方向錯誤（surrogate 預測誤差大，沒真正學會 directional beam）。

`antenna/ris/simulate_ris.py` 的 RISSimulator 是純 torch、完全可微 — 不需要
surrogate 作中介。直接 `loss(RISSimulator(generator(target)), target)` 就有
端到端正確梯度。這是曾俊瑋 113 學年論文「公式層取代 decoder」的真正含意。

用法：
    python script/train_direct_ris.py --epochs 500 --device cuda:0
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from antenna.core.pattern import AntennaPattern
from antenna.core.response import AntennaResponse
from antenna.models.generators.biased_gumbel_sigmoid_gen import BiasedGumbelSigmoidGEN
from antenna.ris import RISSimulator, custom_loss_tolerance
from antenna.utils.config import config

# 重用 train_multi_target.py 的 target pool
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("_mt", Path(__file__).parent / "train_multi_target.py")
_mt = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mt)
make_target_pool = _mt.make_target_pool


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--element_num", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--n_targets", type=int, default=32)
    parser.add_argument("--cond_reg_weight", type=float, default=0.5)
    parser.add_argument("--balance_weight", type=float, default=0.5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="result/RIS-direct-v8")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.device:
        config.device = args.device
    elif torch.cuda.is_available():
        config.device = "cuda:0"
    logger.info(f"裝置：{config.device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    AntennaPattern.setDefaultCoordinate((0, args.element_num, 0, args.element_num))
    AntennaResponse.registerLabels("response", x="ris")
    AntennaResponse.target(side=-20.0, center=0.0, width=(140, 0, 40, 0, 181), label="response", add=True)

    sim = RISSimulator(element_num=args.element_num)
    target_pool = make_target_pool(args.n_targets, response_size=361, seed=args.seed).to(config.device)
    logger.info(f"target pool shape: {tuple(target_pool.shape)}")

    gen = BiasedGumbelSigmoidGEN().to(config.device)
    optimizer = torch.optim.Adam(gen.parameters(), lr=args.lr, betas=(0.5, 0.999))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoint"
    ckpt_dir.mkdir(exist_ok=True)

    history = {"step": [], "real_loss": [], "supp_avg": [], "hamming_avg": []}
    best_supp = -np.inf

    for ep in range(1, args.epochs + 1):
        gen.train()
        idx = int(np.random.randint(0, args.n_targets))
        target = target_pool[idx]

        soft = gen(target)  # gumbel-sigmoid output ∈ [0, 1]
        pat = soft.reshape(args.element_num, args.element_num)

        # ★ 直接用可微 RIS sim 算響應（不經 surrogate）★
        resp = sim(pat)["response"]

        loss = custom_loss_tolerance(
            resp, target,
            sidelobe_threshold=-25.0,
            main_target=0.0,
            main_weight=5.0,
        )
        loss = loss + args.balance_weight * (soft.mean() - 0.5).abs()

        # Conditional regularizer
        if args.cond_reg_weight > 0:
            idx2 = int(np.random.randint(0, args.n_targets))
            while idx2 == idx:
                idx2 = int(np.random.randint(0, args.n_targets))
            target2 = target_pool[idx2]
            _ = gen(target2); logits2 = gen.logits.detach()
            _ = gen(target); logits1 = gen.logits
            l2_dist_sq = ((logits1 - logits2) ** 2).mean()
            cond_loss = args.cond_reg_weight / (l2_dist_sq + 0.1)
            loss = loss + cond_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history["step"].append(ep)
        history["real_loss"].append(float(loss.item()))

        # 每 50 步：跑全部 target，算 avg suppression + 平均 hamming distance
        if ep % 50 == 0:
            with torch.no_grad():
                gen.eval()
                supps = []
                patterns_hard = []
                for ti in range(args.n_targets):
                    t = target_pool[ti]
                    s = gen(t)
                    h = (s > 0.5).float().reshape(args.element_num, args.element_num)
                    patterns_hard.append(h)
                    r = sim(h)["response"].cpu().numpy()
                    main_mask = (t.cpu().numpy() == t.max().item())
                    side_mask = (t.cpu().numpy() == t.min().item())
                    if main_mask.any() and side_mask.any():
                        mp = float(r[main_mask].max())
                        sm = float(r[side_mask].max())
                        supps.append(mp - sm)
                avg_supp = float(np.mean(supps))

                # Hamming pairwise: 量 conditional 是否真的有效
                ham_pairs = []
                for i in range(len(patterns_hard)):
                    for j in range(i + 1, len(patterns_hard)):
                        ham_pairs.append((patterns_hard[i] != patterns_hard[j]).float().mean().item())
                avg_ham = float(np.mean(ham_pairs))

                history["supp_avg"].append(avg_supp)
                history["hamming_avg"].append(avg_ham)

                if avg_supp > best_supp:
                    best_supp = avg_supp
                    torch.save(gen.state_dict(), ckpt_dir / "best.pth")

                logger.info(
                    f"ep {ep:4d}/{args.epochs} | real_loss={loss.item():.2f} | "
                    f"avg_supp={avg_supp:+.2f} dB | best={best_supp:+.2f} | "
                    f"avg_hamming={avg_ham:.2%}"
                )

    # 結束評估：載入 best ckpt
    if (ckpt_dir / "best.pth").exists():
        gen.load_state_dict(torch.load(ckpt_dir / "best.pth", map_location=config.device, weights_only=False))
    gen.eval()

    rows = []
    sample_targets = make_target_pool(10, response_size=361, seed=args.seed + 999).to(config.device)
    patterns_hard = []
    for i in range(10):
        t = sample_targets[i]
        with torch.no_grad():
            s = gen(t)
            h = (s > 0.5).float().reshape(args.element_num, args.element_num)
            patterns_hard.append(h.cpu().numpy())
            r = sim(h)["response"].cpu().numpy()
        main_mask = (t.cpu().numpy() == t.max().item())
        side_mask = (t.cpu().numpy() == t.min().item())
        if main_mask.any() and side_mask.any():
            mp = float(r[main_mask].max())
            sm = float(r[side_mask].max())
            rows.append((i + 1, float(h.mean()), mp, sm, mp - sm))

    # Hamming on sample targets
    ham_pairs = [(patterns_hard[i] != patterns_hard[j]).mean()
                 for i in range(10) for j in range(i + 1, 10)]
    avg_ham_sample = float(np.mean(ham_pairs))

    summary = ["# Plan E (direct RIS sim) 評估（10 個 sample target）", ""]
    summary += [
        "| # | on% | main_peak dB | side_max dB | suppression dB |",
        "|---|-----|--------------|-------------|----------------|",
    ]
    for i, on, mp, sm, sup in rows:
        summary.append(f"| {i} | {on:.0%} | {mp:.2f} | {sm:.2f} | {sup:+.2f} |")
    sup_vals = [r[4] for r in rows]
    summary += [
        "",
        f"**suppression mean**: {np.mean(sup_vals):+.2f} dB",
        f"**avg pairwise hamming**: {avg_ham_sample:.2%} "
        f"(>0 = pattern 對不同 target 確實有差異)",
        "",
        "對照：v6 plan D suppression −0.46（hamming ~0.4%），direct GD upper +3.05",
    ]
    text = "\n".join(summary)
    (out_dir / "summary.md").write_text(text, encoding="utf-8")
    print(text.encode("ascii", "replace").decode())
    with open(out_dir / "history.pkl", "wb") as f:
        pickle.dump(history, f)
    logger.success(f"完成 → {out_dir}/  best avg_supp={best_supp:+.2f}, hamming={avg_ham_sample:.2%}")


if __name__ == "__main__":
    main()
