"""ris_core — 1-bit RIS per-task GD beamforming 工具箱（research tool, NOT production API）。

⚠️  SCOPE 警告（必讀）⚠️
========================
本模組是 R94->R156 per-task gradient descent 研究的蒸餾，**不是 lab 真實
production 路徑**。Lab 真實要的是 `G(spec) → pattern` 的 amortized 模型
（在 `antenna/training/trainer.py`, ms 級 inference）, 本模組是
per-task GD（每給一個 spec 跑 30 秒到 5 分鐘出一個 pattern）。

**該用本模組的場景**:
  - 對特定 spec 跑出 gold-standard pattern 當 supervised pretraining data
    供應給 lab amortized G
  - 當 validation oracle 比對 G(spec) 的品質
  - 研究 loss design / surrogate noise robustness 等 sub-problem

**不該用本模組的場景**:
  - 直接當 production inference path（per-task GD 不夠快）
  - 取代 lab 的 `antenna/training/trainer.py` pipeline（架構不一樣）
  - patch antenna 場景的直接 deploy（HFSS 無 closed-form, per-task GD 不可行）

詳細整合計劃見 `outputs/INTEGRATION_WITH_LAB_PIPELINE.md`。

========================
功能 (per-task GD scope)
========================
  - 用 4D 決策樹挑 recipe (R134/R135)
  - 跑 unified pipeline (selector + joint early-stop, R140/R141/R150)
  - 把解析模擬器換成 warm-start surrogate forward 取得加速 (R146/R147)
  - 對多頻率聯合最佳化 (R154)

使用範例見 `script/ris_demo.py`。

Reference rounds (見 outputs/EXPERIMENT_LESSONS.md / outputs/REPORT_R94_to_R156.md)：
  - R119  baseline 1-bit recipe (rw=2, lambda=1.0)
  - R129  wide-cap retune (12 < width <= 30)
  - R131  inc=0 mmWave rescue (n=51)
  - R133  n=71 inc=0 mmWave rescue
  - R134  selector validation (5/6 PASS on held-out combos)
  - R135  width transition zone (12-15 deg boundary fix)
  - R140  joint early-stop (binary metrics + flat-top filter)
  - R141  optimize_ris_1bit() wrapper
  - R146  warm-start surrogate (exact match, R^2 ~ 1.0)
  - R147  surrogate-in-the-loop (matches analytical, ~5x speedup)
  - R150  unified pipeline (analytical / surrogate 同一個 API)
  - R154  multi-frequency joint optimization (in-band BW)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from antenna.ris import RISSimulator
from antenna.utils.config import config

# ---------------------------------------------------------------------------
# 常數
# ---------------------------------------------------------------------------
DEFAULT_GD_STEPS = 1500
DEFAULT_EVAL_EVERY = 50
DEFAULT_LR = 0.05
DEFAULT_DEVICE = "cuda:0"
N_THETA = 361   # theta 從 -90 到 +90 step 0.5deg


# ---------------------------------------------------------------------------
# 平滑 min/max（softmin / softmax via logsumexp，beta 越大越接近 hard）
# ---------------------------------------------------------------------------
def soft_max(x: torch.Tensor, beta: float = 20.0) -> torch.Tensor:
    return (1.0 / beta) * torch.logsumexp(beta * x, dim=-1)


def soft_min(x: torch.Tensor, beta: float = 20.0) -> torch.Tensor:
    return -(1.0 / beta) * torch.logsumexp(-beta * x, dim=-1)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def steer_to_indices(center_deg: float, width_deg: float) -> tuple[int, int]:
    """把 (center, width) 轉成 theta 軸 (-90..+90, step 0.5) 上的 [lo, hi) index 範圍。"""
    lo = int(round((center_deg - width_deg / 2 + 90) / 0.5))
    hi = int(round((center_deg + width_deg / 2 + 90) / 0.5))
    return lo, hi


def quantize_1bit(params: torch.Tensor) -> torch.Tensor:
    """把連續 phase 參數 (0..2) 量化成 1-bit binary (0 or 1)。

    參數約定：params * pi 是真實相位；落在 (pi/2, 3pi/2) 區間的元件設為 1。
    """
    phase = (params * torch.pi) % (2 * torch.pi)
    return ((phase > torch.pi / 2) & (phase < 3 * torch.pi / 2)).float()


# ---------------------------------------------------------------------------
# 4D Recipe 選擇器（蒸餾自 R134 / R135 / R131 / R133）
# ---------------------------------------------------------------------------
def select_1bit_recipe(n: int, inc_deg: float, freq_hz: float, width_deg: float) -> dict:
    """根據 (n, inc, freq, width) 挑 1-bit beamforming recipe。

    回傳 dict 含：
      - rw           : ripple weight (loss 中 ripple 項的係數)
      - lambda_mean  : mean-sidelobe weight
      - tier         : 描述用的 tier 名稱
    """
    if width_deg > 30:
        raise ValueError(f"width={width_deg} 超過驗證範圍 (>30deg)")
    if n not in (31, 51, 71):
        raise ValueError(f"n={n} 不在驗證 set (31, 51, 71)")

    # n=71 高自由度 tier
    if n == 71:
        if inc_deg == 0 and freq_hz >= 50e9:
            return {"rw": 5.0, "lambda_mean": 0.3, "tier": "R133 n=71 inc=0 mmWave"}
        if width_deg <= 15:
            return {"rw": 5.0, "lambda_mean": 0.5, "tier": "n=71 narrow extrapolation"}
        return {"rw": 7.0, "lambda_mean": 0.5, "tier": "n=71 wide extrapolation"}

    # 寬主瓣 cap (R129)
    if width_deg > 12:
        if width_deg <= 20:
            return {"rw": 3.0, "lambda_mean": 1.0, "tier": "R129 wide cap (12-20)"}
        return {"rw": 3.0, "lambda_mean": 0.5, "tier": "R129 wide cap 30 (marginal)"}

    # broadside + mmWave 救援 (R131)
    if inc_deg == 0 and freq_hz >= 20e9:
        if freq_hz >= 50e9:
            raise ValueError("inc=0 + freq>=50GHz at n=51 -> 改用 n=71")
        if freq_hz >= 35e9:
            return {"rw": 2.0, "lambda_mean": 0.5, "tier": "R131 inc=0 38GHz rescue"}
        return {"rw": 2.0, "lambda_mean": 0.3, "tier": "R131 inc=0 28GHz rescue"}

    # 一般 baseline (R119)
    return {"rw": 2.0, "lambda_mean": 1.0, "tier": "R119 baseline"}


# ---------------------------------------------------------------------------
# Loss & metrics
# ---------------------------------------------------------------------------
def compute_loss(
    response: torch.Tensor,
    main_lo: int,
    main_hi: int,
    rw: float,
    lambda_mean: float,
) -> torch.Tensor:
    """連續代理梯度 loss（R119/R129 三項組合）。

      loss = -(soft_min(main) - soft_max(side))   <- worst-case margin
             + rw * (soft_max(main) - soft_min(main))    <- ripple penalty
             + lambda_mean * mean(side)            <- average sidelobe
    """
    main = response[main_lo:main_hi]
    side = torch.cat([response[:main_lo], response[main_hi:]])
    mm = soft_min(main)
    sx = soft_max(side)
    mx = soft_max(main)
    return -(mm - sx) + rw * (mx - mm) + lambda_mean * side.mean()


def eval_binary_metrics(
    response_np: np.ndarray, main_lo: int, main_hi: int
) -> dict:
    """從一條 binary response (dB array) 算 worst / ripple / flat-top 指標。"""
    main = response_np[main_lo:main_hi]
    side = np.delete(response_np, np.arange(main_lo, main_hi))
    return {
        "worst": float(main.min() - side.max()),
        "side_mean": float(side.mean()),
        "ripple": float(main.max() - main.min()),
        "flat_top": int(np.sum(main < -3)) == 0,
    }


def _eval_binary_on(
    eval_fn: Callable, params: torch.Tensor, main_lo: int, main_hi: int
) -> dict:
    with torch.no_grad():
        binary = quantize_1bit(params)
        resp = eval_fn(binary)["response"].cpu().numpy()
    return eval_binary_metrics(resp, main_lo, main_hi)


# ---------------------------------------------------------------------------
# Warm-start surrogate (R146 修正版：column-major flatten + (1-2x) amplitude)
# ---------------------------------------------------------------------------
class ContinuousWarmStartSurrogate(nn.Module):
    """1 層複數 linear 模擬陣列因子，從 RISSimulator.pre_calAF 拷貝權重後可逐位元
    精準重現解析模擬器 (R146 證實 R^2 ~ 1.0)。

    用途：在 surrogate-in-the-loop 流程中當作 forward_fn 取代解析模擬器以加速梯度
    計算 (R147 ~5x speedup, 維持 PASS 率)。

    輸入 x: (n, n) 或 (B, n, n)，值域 [0, 2]，phase = x * pi。連續可微，
    binary x in {0, 1} 時 (cos, sin) 會精準對應 (1-2x) 的解析模擬器輸出。
    """

    def __init__(self, n_elem: int, n_angles: int = N_THETA):
        super().__init__()
        self.n = n_elem
        self.real_lin = nn.Linear(n_elem * n_elem, n_angles, bias=False)
        self.imag_lin = nn.Linear(n_elem * n_elem, n_angles, bias=False)

    def forward(self, x: torch.Tensor) -> dict:
        single = x.dim() == 2
        if single:
            x = x.unsqueeze(0)
        phase = x * torch.pi
        # CRITICAL: sim 內部做 MPD.t().reshape(...)，所以這裡也要轉置再 flatten
        cos_p = torch.cos(phase).transpose(1, 2).contiguous().flatten(1)
        sin_p = torch.sin(phase).transpose(1, 2).contiguous().flatten(1)
        # complex multiply (W_re + i W_im) * (cos + i sin)
        re = self.real_lin(cos_p) - self.imag_lin(sin_p)
        im = self.real_lin(sin_p) + self.imag_lin(cos_p)
        amp = torch.sqrt(re * re + im * im + 1e-12)
        peak = amp.max(dim=1, keepdim=True).values
        out = 20.0 * torch.log10(torch.clamp(amp, min=1e-8) / torch.clamp(peak, min=1e-8))
        return {"response": out.squeeze(0) if single else out}


def build_warmstart_surrogate(
    n: int, freq_hz: float, inc_deg: float, device: str = DEFAULT_DEVICE
) -> ContinuousWarmStartSurrogate:
    """從一個對應規格的解析模擬器拷貝 broadside (phi=0) 權重，回傳 perfect-fit surrogate。"""
    sim = RISSimulator(element_num=n, freq_hz=freq_hz, inc_theta_deg=inc_deg)
    sur = ContinuousWarmStartSurrogate(n).to(device)
    af = sim.pre_calAF.detach()[0]  # broadside slice (n_theta, n_elem*n_elem) complex
    with torch.no_grad():
        sur.real_lin.weight.copy_(af.real.to(torch.float32).to(device))
        sur.imag_lin.weight.copy_(af.imag.to(torch.float32).to(device))
    return sur


# ---------------------------------------------------------------------------
# Unified pipeline (analytical 或 surrogate forward 同一個 API)
# ---------------------------------------------------------------------------
@dataclass
class RISOptimizationResult:
    recipe: dict
    best: dict                  # 最佳 seed 的 metrics
    n_flat_top: int             # 多少 seed 通過 flat-top
    n_restarts: int
    n_early_stop_used: int
    seed_results: list[dict]


def optimize_ris_1bit(
    n: int,
    inc_deg: float,
    freq_hz: float,
    width_deg: float,
    n_restarts: int = 5,
    gd_steps: int = DEFAULT_GD_STEPS,
    eval_every: int = DEFAULT_EVAL_EVERY,
    steering_center_deg: float = 0.0,
    forward_fn: Optional[Callable] = None,
    eval_fn: Optional[Callable] = None,
    lr: float = DEFAULT_LR,
    device: str = DEFAULT_DEVICE,
) -> RISOptimizationResult:
    """單一 deployment spec 的 1-bit RIS pipeline。

    流程：
      1. 用 (n, inc, freq, width) 透過 select_1bit_recipe 拿 (rw, lambda_mean)
      2. 跑 n_restarts 個 seed 的 Adam 連續 GD
      3. 每 eval_every 步 quantize 成 binary 並 eval；只記錄 flat-top + 更佳 worst 的 snapshot
      4. 回傳所有 seed 的結果與最佳 binary metrics

    forward_fn / eval_fn 可選：
      - 都 None：解析模擬器同時做梯度與 eval (R141 經典模式)
      - forward_fn=surrogate, eval_fn=None：surrogate-in-the-loop (R147), eval 仍用解析
      - 都給：完全替換 (例如 patch 場景用 HFSS surrogate)
    """
    config.device = device
    recipe = select_1bit_recipe(n, inc_deg, freq_hz, width_deg)
    rw, lam = recipe["rw"], recipe["lambda_mean"]
    main_lo, main_hi = steer_to_indices(steering_center_deg, width_deg)

    sim = RISSimulator(element_num=n, freq_hz=freq_hz, inc_theta_deg=inc_deg)
    if forward_fn is None:
        forward_fn = sim
    if eval_fn is None:
        eval_fn = sim

    seed_results: list[dict] = []
    best_overall: Optional[dict] = None

    for seed in range(n_restarts):
        torch.manual_seed(seed)
        params = nn.Parameter(torch.rand(n, n, device=device) * 2.0)
        opt = torch.optim.Adam([params], lr=lr)

        # joint early-stop: 只挑 flat-top 且 worst 更佳的 snapshot
        best_joint_worst = -1e9
        best_joint_state: Optional[torch.Tensor] = None

        for step in range(gd_steps):
            opt.zero_grad()
            resp = forward_fn(params)["response"]
            loss = compute_loss(resp, main_lo, main_hi, rw, lam)
            loss.backward()
            opt.step()

            if (step + 1) % eval_every == 0:
                m = _eval_binary_on(eval_fn, params, main_lo, main_hi)
                if m["flat_top"] and m["worst"] > best_joint_worst:
                    best_joint_worst = m["worst"]
                    best_joint_state = params.detach().clone()

        if best_joint_state is not None:
            metrics = _eval_binary_on(eval_fn, best_joint_state, main_lo, main_hi)
            metrics["used_early_stop"] = True
        else:
            metrics = _eval_binary_on(eval_fn, params, main_lo, main_hi)
            metrics["used_early_stop"] = False
        metrics["seed"] = seed
        seed_results.append(metrics)
        if best_overall is None or metrics["worst"] > best_overall["worst"]:
            best_overall = metrics

    return RISOptimizationResult(
        recipe=recipe,
        best=best_overall,
        n_flat_top=sum(1 for r in seed_results if r["flat_top"]),
        n_restarts=n_restarts,
        n_early_stop_used=sum(1 for r in seed_results if r["used_early_stop"]),
        seed_results=seed_results,
    )


# ---------------------------------------------------------------------------
# 多頻率聯合最佳化 (R154)
# ---------------------------------------------------------------------------
def optimize_ris_1bit_multifreq(
    n: int,
    inc_deg: float,
    freqs_hz: Sequence[float],
    width_deg: float,
    rw: float = 2.0,
    lambda_mean: float = 1.0,
    n_restarts: int = 5,
    gd_steps: int = DEFAULT_GD_STEPS,
    eval_every: int = DEFAULT_EVAL_EVERY,
    steering_center_deg: float = 0.0,
    lr: float = DEFAULT_LR,
    device: str = DEFAULT_DEVICE,
) -> dict:
    """多頻率聯合 1-bit 最佳化 (R154)。

    對每一頻率建一個 RISSimulator，loss = sum over freqs of compute_loss(...)。
    Joint early-stop criterion：所有頻率都 flat-top 且 min(worst across freqs) 更佳。

    回傳 dict：
      - per_seed : list of {seed, per_freq:list of metrics dict}
      - best_state : 最佳 seed 對應的連續 params (在 device 上)
      - freqs : 輸入 freqs_hz 的 list
    """
    config.device = device
    main_lo, main_hi = steer_to_indices(steering_center_deg, width_deg)
    sims = [RISSimulator(element_num=n, freq_hz=f, inc_theta_deg=inc_deg) for f in freqs_hz]

    per_seed: list[dict] = []
    best_state: Optional[torch.Tensor] = None
    best_min_worst = -1e9

    for seed in range(n_restarts):
        torch.manual_seed(seed)
        params = nn.Parameter(torch.rand(n, n, device=device) * 2.0)
        opt = torch.optim.Adam([params], lr=lr)

        seed_best_min_worst = -1e9
        seed_best_state: Optional[torch.Tensor] = None

        for step in range(gd_steps):
            opt.zero_grad()
            total = 0.0
            for sim in sims:
                resp = sim(params)["response"]
                total = total + compute_loss(resp, main_lo, main_hi, rw, lambda_mean)
            total.backward()
            opt.step()

            if (step + 1) % eval_every == 0:
                with torch.no_grad():
                    binary = quantize_1bit(params)
                    metrics_per_f = [
                        eval_binary_metrics(sim(binary)["response"].cpu().numpy(), main_lo, main_hi)
                        for sim in sims
                    ]
                if all(m["flat_top"] for m in metrics_per_f):
                    mw = min(m["worst"] for m in metrics_per_f)
                    if mw > seed_best_min_worst:
                        seed_best_min_worst = mw
                        seed_best_state = params.detach().clone()

        eval_state = seed_best_state if seed_best_state is not None else params.detach()
        with torch.no_grad():
            binary = quantize_1bit(eval_state)
            per_freq = [
                eval_binary_metrics(sim(binary)["response"].cpu().numpy(), main_lo, main_hi)
                for sim in sims
            ]
        per_seed.append({"seed": seed, "per_freq": per_freq})

        if seed_best_min_worst > best_min_worst:
            best_min_worst = seed_best_min_worst
            best_state = seed_best_state

    return {
        "freqs": list(freqs_hz),
        "per_seed": per_seed,
        "best_state": best_state,
        "best_min_worst": best_min_worst,
    }


__all__ = [
    "soft_max",
    "soft_min",
    "steer_to_indices",
    "quantize_1bit",
    "select_1bit_recipe",
    "compute_loss",
    "eval_binary_metrics",
    "ContinuousWarmStartSurrogate",
    "build_warmstart_surrogate",
    "optimize_ris_1bit",
    "optimize_ris_1bit_multifreq",
    "RISOptimizationResult",
]
