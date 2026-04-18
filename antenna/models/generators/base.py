"""生成器共用建構子。

抽取 SigmoidGEN / OldGEN / GumbelSigmoidGEN 共用的 MLP 結構，
統一以 `_build_fc_patch()` 組裝 `Linear -> PReLU -> ... -> BiScaleNorm`。
"""

from typing import Sequence

from torch import nn

from antenna.models.components import BiScaleNorm


def _build_fc_patch(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    *,
    with_biscalenorm: bool = True,
) -> nn.Sequential:
    """建立 `[Linear, PReLU] * N + Linear [+ BiScaleNorm]` 的 MLP。

    Args:
        input_dim: 輸入維度（通常為 `AntennaResponse.size(flatten=True)`）。
        hidden_dims: 隱藏層維度序列。
        output_dim: 輸出維度（通常為 `AntennaPattern.size(flatten=True)`）。
        with_biscalenorm: 是否在最後附加 `BiScaleNorm` 作正負值縮放。

    Returns:
        `nn.Sequential` 構成的 MLP 主幹。
    """
    layers: list[nn.Module] = []
    prev = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.PReLU())
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    if with_biscalenorm:
        layers.append(BiScaleNorm())
    return nn.Sequential(*layers)


def _kaiming_init_(module: nn.Module, *, bias_const: float = 1.0, prelu_const: float = 0.25) -> None:
    """對模組內的 `Linear` 做 Kaiming 初始化，並重置 `PReLU` 權重。

    使用 `modules()` 遍歷以支援巢狀容器（不限於 `nn.Sequential`）。
    """
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, bias_const)
        elif isinstance(m, nn.PReLU):
            m.weight.data.fill_(prelu_const)
