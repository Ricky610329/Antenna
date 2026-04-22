"""自訂 `torch.autograd.Function` 集合。

提供給 generator 使用的 Straight-Through Estimator / Gumbel-Sigmoid 離散化 autograd。
- `sign_f`：forward 為 sign()，backward 在 |x|>1 時 clip 梯度（STE）
- `GumbelSigmoid`：Gumbel-sigmoid 軟採樣，對 logits 與 tau 皆可反傳
- `_GumbelSigmoid`：上者的早期實作（保留以便對照 / 回溯）
- `BinarizeSTE`：以 0.5 為門檻二值化，backward 為 mask-aware STE
"""

import torch
from torch.autograd.function import (
    BackwardCFunction,
    Function,
    FunctionCtx,
)
from torch.types import Tensor


class sign_f(Function):
    """sign function 搭配 STE 反傳（|x|>1 時梯度 clip 為 0）。"""

    @staticmethod
    def forward(ctx: FunctionCtx, inputs: Tensor):
        output = torch.where(inputs >= 0.0, 1.0, -1.0)
        ctx.save_for_backward(inputs)
        return output

    @staticmethod
    def backward(ctx: BackwardCFunction, grad_output: Tensor):
        (input_,) = ctx.saved_tensors
        # 使用 torch.where 回傳新 tensor，避免 in-place 修改 grad_output
        # （expanded tensor 進行 in-place 操作會觸發 UserWarning 並可能破壞上游 autograd 圖）
        grad_input = torch.where(input_.abs() <= 1.0, grad_output, torch.zeros_like(grad_output))
        return grad_input


class _GumbelSigmoid(Function):
    """Gumbel-Sigmoid 的早期實作（保留以對照 `GumbelSigmoid`）。"""

    @staticmethod
    def forward(ctx: FunctionCtx, logits: Tensor, tau_tensor: Tensor, eps: float = 1e-10):
        U = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(U + eps) + eps)
        y = torch.sigmoid((logits + gumbel_noise) / tau_tensor)

        # 保存為 backward 方法提供所需的變數
        ctx.save_for_backward(logits, y, gumbel_noise)
        ctx.tau = tau_tensor  # tau 非 tensor save 目標，放在 ctx attribute
        return y

    @staticmethod
    def backward(ctx: BackwardCFunction, grad_output: Tensor):
        logits, y, gumbel_noise = ctx.saved_tensors
        tau = ctx.tau

        sigmoid_grad = y * (1 - y)  # Sigmoid 梯度
        grad_input = grad_output * sigmoid_grad / tau  # logits 梯度
        grad_tau = (grad_output * sigmoid_grad * (logits + gumbel_noise)).sum() / tau**2

        return grad_input, grad_tau, None


class GumbelSigmoid(Function):
    """Gumbel-Sigmoid 軟採樣，forward 加入 scale=0.1 Gumbel noise 以降低變異。"""

    @staticmethod
    def forward(ctx: FunctionCtx, logits: Tensor, tau: Tensor, eps: float = 1e-20):
        U = torch.rand_like(logits)
        scale = 0.1  # 降低 Gumbel noise 強度，避免 tau 梯度爆炸
        gumbel_noise = -torch.log(-torch.log(U + eps) + eps) * scale
        y = torch.sigmoid((logits + gumbel_noise) / tau)

        ctx.save_for_backward(logits, y, gumbel_noise, tau)
        return y

    @staticmethod
    def backward(ctx: BackwardCFunction, grad_output: Tensor):
        logits, y, gumbel_noise, tau = ctx.saved_tensors

        sigmoid_grad = y * (1 - y)
        grad_input = grad_output * sigmoid_grad / tau
        grad_tau = (-grad_output * sigmoid_grad * (logits + gumbel_noise) / (tau**2)).sum()
        return grad_input, grad_tau, None


class BinarizeSTE(Function):
    """以 0.5 為門檻二值化；backward 為 mask-aware STE（只保留 mask 區域梯度）。"""

    @staticmethod
    def forward(ctx: FunctionCtx, input: Tensor):
        mask = (input >= 0.5).float()
        ctx.save_for_backward(mask)
        return mask

    @staticmethod
    def backward(ctx: BackwardCFunction, grad_output: Tensor):
        (mask,) = ctx.saved_tensors
        return grad_output * mask  # 只保留 mask 區域的梯度
