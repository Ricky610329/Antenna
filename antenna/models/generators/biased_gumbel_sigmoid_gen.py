"""帶反 collapse 初始化的 GumbelSigmoidGEN 變體。

**設計目標**：解決原 GumbelSigmoidGEN 在 binary RIS 訓練中 collapse 到 all-1
的問題（§8.4d、V12-V15 觀察）。

**關鍵修正（v2）**：
1. 移除最後的 ``BiScaleNorm`` — 該層會把 logits 壓到 [-1, 1]，使任何 bias 初始化
   的差異都失效（sigmoid([-1, 1]) ⊂ [0.27, 0.73]，分布太集中於 0.5）。
2. 取消後做校準：跑 ``CALIB_PASSES`` 次 random target 的 forward，把最後一層
   Linear 的 bias 整體加上常數，使 ``mean(logits) ≈ 0``，這樣硬二值化後初始
   on-rate ≈ 50%。
3. 同時把 ``BIAS_INIT_RANGE`` 縮成 1.0 — 範圍仍夠分散，但不會被 clamp(±5) 截掉。
"""

import torch
from torch import nn

from antenna.core.pattern import AntennaPattern
from antenna.core.response import AntennaResponse
from antenna.models.autograd import GumbelSigmoid
from antenna.models.generators.base import _build_fc_patch, _kaiming_init_
from antenna.utils.config import config


class BiasedGumbelSigmoidGEN(nn.Module):
    """無 BiScaleNorm + 啟動校準 + 對稱 bias init 的 GumbelSigmoid 生成器。

    forward 路徑：``Linear×N → 最後 Linear → clamp(±LOGITS_CLAMP) → GumbelSigmoid``
    （**不經過** BiScaleNorm，這是與其他 GumbelSigmoid 變體的關鍵差別）。
    """

    LOGITS_CLAMP = 5.0
    BIAS_INIT_RANGE = 1.0  # 最後一層 bias uniform[-1, 1]
    CALIB_PASSES = 32  # 校準階段跑幾次 forward 求平均

    def __init__(self):
        super().__init__()
        pattern_size = AntennaPattern.size(flatten=True)
        self.fc_patch = _build_fc_patch(
            input_dim=AntennaResponse.size(flatten=True),
            hidden_dims=(pattern_size, pattern_size * 2, pattern_size),
            output_dim=pattern_size,
            with_biscalenorm=False,  # ← 關鍵：保留 bias init 訊號
        )
        _kaiming_init_(self.fc_patch)

        # 找最後一個帶 bias 的 Linear 並隨機初始化（給 pixel-wise 多樣性）
        last_linear: nn.Linear | None = None
        for module in self.fc_patch.modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                last_linear = module
        if last_linear is not None:
            with torch.no_grad():
                last_linear.bias.uniform_(-self.BIAS_INIT_RANGE, self.BIAS_INIT_RANGE)

        self.tau = nn.Parameter(torch.tensor(5.0, requires_grad=True))
        self.tau_history: list[float] = []

        self.to(config.device)

        # 校準：用隨機 input 估 logits 的整體偏差，把最後一層 bias 全部減掉
        # 這個常數，讓初始 mean(logits) ≈ 0、初始 on-rate ≈ 50%。
        self._calibrate_bias_offset(last_linear)

    @torch.no_grad()
    def _calibrate_bias_offset(self, last_linear: nn.Linear | None) -> None:
        """跑 ``CALIB_PASSES`` 次隨機 forward，把最後一層 bias 整體偏移到平均 logit ≈ 0。"""
        if last_linear is None:
            return
        input_dim = AntennaResponse.size(flatten=True)
        device = next(self.parameters()).device
        sums = []
        for _ in range(self.CALIB_PASSES):
            rand_input = torch.randn(input_dim, device=device)
            out = self.fc_patch(rand_input)  # 此時尚未 clamp
            sums.append(out.mean().item())
        avg_logit = sum(sums) / len(sums)
        last_linear.bias.sub_(avg_logit)  # 平移整層 bias
        # 校準後再驗證
        verify = []
        for _ in range(8):
            rand_input = torch.randn(input_dim, device=device)
            out = self.fc_patch(rand_input)
            verify.append(out.mean().item())
        # 把校準資訊存起來（供 logger 檢視）
        self._init_logit_mean_before = avg_logit
        self._init_logit_mean_after = sum(verify) / len(verify)

    def forward(self, input):
        self.logits = torch.clamp(self.fc_patch(input), min=-self.LOGITS_CLAMP, max=self.LOGITS_CLAMP)
        x = GumbelSigmoid.apply(self.logits, self.tau)
        self.tau_history.append(self.tau.detach().cpu().item())
        return x

    def anneal_tau(self, min_tau: float = 0.1) -> None:
        self.tau = torch.clamp(self.tau, min=min_tau)
        self.tau_history.append(self.tau.detach().cpu().item())

    def binarize(self, threshold: float = 0.5) -> AntennaPattern:
        binarized_output = (torch.sigmoid(self.logits) > threshold).float()
        return AntennaPattern(binarized_output)
