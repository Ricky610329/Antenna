"""產生 RIS 結構化 binary pattern（線性相位梯度 + 變體），混進 surrogate 預訓練資料。

**動機**：純隨機 binary pattern 的響應大多是雜訊（無定向 beam）。surrogate 從這種
資料學到的 mapping 是「random pattern → diffuse response」，當 generator 試圖
產生定向 beam 時 surrogate 沒有相關經驗可以提供有用梯度。

**解法**：用 RIS 教科書級 beam-steering 公式預先產出已知會 work 的 pattern：

    phase[m, n] = 2π × (m sin θ cos φ + n sin θ sin φ) × d / λ
    binary_phase[m, n] = ((phase mod 2π) > π).float()   # → 0 或 1

對 (θ, φ) 取 N 組角度，每組產一張 pattern + 模擬響應，併入 random pattern dataset。
這樣 surrogate 至少看過「linear phase gradient → directional beam」這個基礎範式。

用法：
    python script/generate_structured_patterns.py --element_num 15 --n_angles 200
        --out_dir result/_pretrained_surrogate
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from antenna.core.pattern import AntennaPattern
from antenna.core.response import AntennaResponse
from antenna.ris import RISSimulator
from antenna.utils.config import config


def linear_phase_gradient_pattern(
    element_num: int,
    theta_deg: float,
    phi_deg: float,
    freq_hz: float = 28e9,
    feed_distance_m: float = 500e-3,
) -> torch.Tensor:
    """以線性相位梯度產生指定角度的 binary RIS pattern。

    Args:
        element_num: 單邊元件數
        theta_deg, phi_deg: 目標反射方向（球座標角度）
        freq_hz: 工作頻率
        feed_distance_m: 饋源距離（用於補償入射相位）

    Returns:
        (element_num, element_num) shape，dtype float32，元素 ∈ {0, 1}
    """
    c = 3e8
    wavelength = c / freq_hz
    de = 0.5 * wavelength  # 元素間距 = 半波長（與 RISSimulator 預設一致）
    k = 2 * np.pi / wavelength

    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)

    # RIS 元素中心化座標
    low = -(element_num / 2 - 0.5)
    high = low + element_num
    x_idx, y_idx = np.mgrid[low:high, low:high]
    x = x_idx * de
    y = y_idx * de

    # 反射相位 = -k × (x sinθ cosφ + y sinθ sinφ)
    # （補償入射波相位即可使主瓣對到 (θ, φ)）
    phase_rad = -k * (x * np.sin(theta) * np.cos(phi) + y * np.sin(theta) * np.sin(phi))

    # 折回 [0, 2π) 區間
    phase_mod = np.mod(phase_rad, 2 * np.pi)

    # 二值量化：phase ∈ [π, 2π) → 1（對應 RIS 相位 π），其他 → 0
    binary = (phase_mod >= np.pi).astype(np.float32)

    return torch.tensor(binary, dtype=torch.float32)


def generate_structured_dataset(
    element_num: int,
    n_angles: int,
    *,
    theta_range: tuple[float, float] = (-60, 60),
    phi_range: tuple[float, float] = (0, 360),
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """對 (θ, φ) 撒點產生 N 組結構化 pattern + RIS 響應。

    Returns:
        (patterns, responses)：patterns (N, element_num²)、responses (N, 361)
    """
    rng = np.random.default_rng(seed)
    sim = RISSimulator(element_num=element_num)

    patterns = []
    responses = []
    logger.info(f"產生 {n_angles} 組線性相位梯度 pattern (θ ∈ {theta_range}, φ ∈ {phi_range})")

    for i in range(n_angles):
        theta_deg = rng.uniform(*theta_range)
        phi_deg = rng.uniform(*phi_range)
        pat_2d = linear_phase_gradient_pattern(element_num, theta_deg, phi_deg).to(config.device)

        with torch.no_grad():
            r = sim(pat_2d)["response"].detach()

        patterns.append(pat_2d.flatten().cpu())
        responses.append(r.cpu())

        if (i + 1) % max(1, n_angles // 10) == 0:
            on_rate = pat_2d.mean().item()
            peak_idx = int(r.argmax().item())
            logger.info(
                f"  {i + 1}/{n_angles}  θ={theta_deg:+.1f}°, φ={phi_deg:.1f}°, "
                f"on={on_rate:.0%}, peak idx={peak_idx} ({r.max():.2f} dB)"
            )

    return torch.stack(patterns), torch.stack(responses)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--element_num", type=int, default=15)
    parser.add_argument("--n_angles", type=int, default=200, help="結構化樣本數（預設 200）")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="result/_pretrained_surrogate",
        help="輸出 dataset.pkl 的目錄；若已存在 dataset.pkl 會合併",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device:
        config.device = args.device
    elif torch.cuda.is_available():
        config.device = "cuda:0"
    logger.info(f"裝置：{config.device}")

    AntennaPattern.setDefaultCoordinate((0, args.element_num, 0, args.element_num))
    AntennaResponse.registerLabels("response", x="ris")
    AntennaResponse.target(side=-20.0, center=0.0, width=(140, 0, 40, 0, 181), label="response", add=True)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 產生結構化 dataset
    s_patterns, s_responses = generate_structured_dataset(
        args.element_num, args.n_angles, seed=args.seed
    )
    logger.info(f"結構化 pattern on-rate 統計：mean={s_patterns.mean():.2%}, "
                f"std={s_patterns.std():.4f}")
    logger.info(f"響應 dB 範圍：[{s_responses.min():.2f}, {s_responses.max():.2f}]")

    # 合併進現有 dataset.pkl（pretrain_surrogate.py 產出的格式）
    structured_pkl = out_dir / "structured_dataset.pkl"
    with open(structured_pkl, "wb") as f:
        pickle.dump(
            {
                "patterns": s_patterns,
                "responses": s_responses,
                "element_num": args.element_num,
                "kind": "linear_phase_gradient",
                "n_angles": args.n_angles,
            },
            f,
        )
    logger.info(f"結構化 dataset → {structured_pkl}  ({structured_pkl.stat().st_size / 1024:.0f} KB)")

    # 若已有 random dataset，合併成 combined_dataset.pkl
    random_pkl = out_dir / "dataset.pkl"
    if random_pkl.exists():
        with open(random_pkl, "rb") as f:
            random_data = pickle.load(f)
        r_patterns = random_data["patterns"]
        r_responses = random_data["responses"]
        combined_patterns = torch.cat([r_patterns, s_patterns], dim=0)
        combined_responses = torch.cat([r_responses, s_responses], dim=0)
        combined_pkl = out_dir / "combined_dataset.pkl"
        with open(combined_pkl, "wb") as f:
            pickle.dump(
                {
                    "patterns": combined_patterns,
                    "responses": combined_responses,
                    "element_num": args.element_num,
                    "n_random": len(r_patterns),
                    "n_structured": len(s_patterns),
                },
                f,
            )
        logger.success(
            f"合併 dataset → {combined_pkl}  "
            f"({len(r_patterns)} random + {len(s_patterns)} structured = {len(combined_patterns)} 筆)"
        )
    else:
        logger.warning(f"找不到 {random_pkl}，僅輸出 structured_dataset.pkl")


if __name__ == "__main__":
    main()
