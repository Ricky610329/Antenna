"""預訓練 RIS surrogate (pattern → response)。

用途
====
RIS generator 訓練的關鍵假設：surrogate 必須能正確映射 ``binary pattern → response``，
generator 才會收到「在硬體上會發生的響應」對應的梯度。原 trainer 的 online learning
從零開始 surrogate，前期幾百個 epoch 都在學 surrogate 而不是 generator，效率差且
容易把 generator 帶偏。

本 script 一次離線把 surrogate 訓練到合理水準：

1. 產生 N 組隨機 binary pattern（每像素 50% 機率為 1）
2. 用 :class:`RISSimulator` 計算對應 dB 響應 — 即 ground truth
3. 用 :class:`OldSM` (HFSSNet) 訓練到 loss < ``min_loss``
4. 把 (pattern, response) dataset 存到 ``result/_pretrained_surrogate/dataset.pkl``
5. 把 surrogate 權重存到 ``result/_pretrained_surrogate/checkpoint/sm.pth``

trainer 之後可透過 ``cfg.surrogate.pretrained_path`` 載入。

用法
====
    python script/pretrain_surrogate.py --element_num 15 --n_samples 5000
    python script/pretrain_surrogate.py --element_num 15 --n_samples 5000 --device cuda:0
"""

import argparse
import pickle
from pathlib import Path

import torch
from loguru import logger

from antenna.core.pattern import AntennaPattern
from antenna.core.response import AntennaResponse
from antenna.models.surrogates.surrogate_model import OldSM
from antenna.ris import RISSimulator
from antenna.utils.config import config
from antenna.utils.data import DataManager


def generate_dataset(
    element_num: int,
    n_samples: int,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """產生 N 組 (binary pattern, RIS response) 對。

    Args:
        element_num: RIS 單邊元件數。
        n_samples: 樣本數。
        seed: 亂數種子。

    Returns:
        (patterns, responses)：patterns shape ``(N, element_num**2)``，
        responses shape ``(N, 361)``（RIS theta sweep）。
    """
    g = torch.Generator(device=config.device).manual_seed(seed)
    sim = RISSimulator(element_num=element_num)

    probs = torch.full((n_samples, element_num * element_num), 0.5, device=config.device)
    patterns = torch.bernoulli(probs, generator=g)

    responses = []
    logger.info(f"模擬 {n_samples} 筆 binary pattern → RIS response...")
    for i in range(n_samples):
        pat_2d = patterns[i].reshape(element_num, element_num)
        with torch.no_grad():
            r = sim(pat_2d)["response"]
        responses.append(r.detach())
        if (i + 1) % max(1, n_samples // 10) == 0:
            logger.info(f"  {i + 1}/{n_samples}")
    responses = torch.stack(responses, dim=0)
    return patterns.cpu(), responses.cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--element_num", type=int, default=15, help="RIS 單邊元件數（預設 15）")
    parser.add_argument("--n_samples", type=int, default=5000, help="訓練樣本數（預設 5000）")
    parser.add_argument("--epochs", type=int, default=200, help="訓練 epoch 上限（預設 200）")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size（預設 64）")
    parser.add_argument("--device", type=str, default=None, help="覆蓋 config.device，如 cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out_dir",
        type=str,
        default="result/_pretrained_surrogate",
        help="輸出目錄（預設 result/_pretrained_surrogate）",
    )
    args = parser.parse_args()

    if args.device:
        config.device = args.device
    elif torch.cuda.is_available():
        config.device = "cuda:0"
    logger.info(f"使用裝置: {config.device}")

    # ── 設定全域 pattern / response 規格 ──
    AntennaPattern.setDefaultCoordinate((0, args.element_num, 0, args.element_num))
    AntennaResponse.registerLabels("response", x="ris")
    AntennaResponse.target(side=-20.0, center=0.0, width=(140, 0, 40, 0, 181), label="response", add=True)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoint"
    ckpt_dir.mkdir(exist_ok=True)

    # ── 1. 產生 dataset ──
    patterns, responses = generate_dataset(args.element_num, args.n_samples, seed=args.seed)
    logger.info(f"patterns: {tuple(patterns.shape)} dtype={patterns.dtype}, on-rate={patterns.mean():.3f}")
    logger.info(f"responses: {tuple(responses.shape)} dB range=[{responses.min():.2f}, {responses.max():.2f}]")

    dataset_pkl = out_dir / "dataset.pkl"
    with open(dataset_pkl, "wb") as f:
        pickle.dump({"patterns": patterns, "responses": responses, "element_num": args.element_num}, f)
    logger.info(f"dataset → {dataset_pkl}  ({dataset_pkl.stat().st_size / 1024:.0f} KB)")

    # ── 2. 灌進 DataManager 給 train_by_datas 吃 ──
    dm = DataManager("pretrain", rootdir=out_dir)
    rows = [[pat.cpu(), resp.cpu()] for pat, resp in zip(patterns, responses)]
    dm.add_and_save(rows, mode="append")
    logger.info(f"DataManager 內樣本數: {len(dm)}")

    # ── 3. 訓練 surrogate ──
    config["HFSS.lr"] = 0.001
    config["HFSS.min_loss"] = 0.01
    config["HFSS.max_epoch"] = args.epochs
    sm = OldSM(checkpoint=str(ckpt_dir))
    losses = sm.train_by_datas(dm, epochs=args.epochs, batch_size=args.batch_size, verbose=True)
    if losses:
        logger.info(f"訓練完成，最終 epoch loss: {losses[-1]:.4e}（共 {len(losses)} epoch）")

    # ── 4. 存權重 ──
    sm_path = ckpt_dir / "sm.pth"
    torch.save(sm.model.state_dict(), sm_path)
    logger.info(f"surrogate weights → {sm_path}  ({sm_path.stat().st_size / 1024:.0f} KB)")

    # ── 5. metadata（給 trainer 驗證 element_num 一致用）──
    meta = {
        "element_num": args.element_num,
        "n_samples": args.n_samples,
        "epochs_used": len(losses),
        "final_loss": losses[-1] if losses else None,
        "pattern_size": args.element_num * args.element_num,
        "response_size": int(responses.shape[1]),
    }
    with open(out_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    logger.info(f"meta → {out_dir / 'meta.pkl'}")

    # 清掉 DataManager 暫存（add_and_save 的 dedup 機制每樣本要 ~600 KB 開銷，
    # 預訓練完成後 surrogate weights 與 dataset.pkl 才是長期保存的物件）。
    # log 檔可能仍被 DataManager 的 logger 鎖住，failure 不致命。
    for tmp in (out_dir / "pretrain.dataset", out_dir / "pretrain.dataset.log"):
        if tmp.exists():
            try:
                tmp.unlink()
                logger.info(f"清理暫存：{tmp}")
            except OSError as e:
                logger.warning(f"無法刪除 {tmp}（檔案被鎖住）：{e}")

    logger.success(f"預訓練完成。trainer 設定：surrogate.pretrained_path: {out_dir}")


if __name__ == "__main__":
    main()
