import torch
from torch.autograd.function import (
    BackwardCFunction,
    Function,
    FunctionCtx,
)
from torch.types import Tensor


class sign_f(Function):
    """
    sign function
    """

    @staticmethod
    def forward(ctx: BackwardCFunction, inputs: Tensor):
        output = inputs.new(inputs.size())
        output[inputs >= 0.0] = 1
        output[inputs < 0.0] = -1
        ctx.save_for_backward(inputs)
        return output

    @staticmethod
    def backward(ctx: BackwardCFunction, grad_output: Tensor):
        (input_,) = ctx.saved_tensors
        grad_output[input_ > 1.0] = 0
        grad_output[input_ < -1.0] = 0
        return grad_output


class _GumbelSigmoid(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, logits, tau_tensor, eps=1e-10):
        """
        Gumbel-Sigmoid采樣方法
        logits: 輸入的logits（可以是實數）
        tau: 溫度，控制離散度
        eps: 防止除以0的小常數
        """
        U = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(U + eps) + eps)
        y = torch.sigmoid((logits + gumbel_noise) / tau_tensor)

        # 保存為 backward 方法提供所需的變數
        ctx.save_for_backward(logits, y, gumbel_noise)
        ctx.tau = tau_tensor  # 保存 tau 以便在 backward 中使用
        return y

    @staticmethod
    def backward(ctx: BackwardCFunction, grad_output):
        # 讀取 forward 傳遞的變數
        logits, y, gumbel_noise = ctx.saved_tensors
        tau = ctx.tau  # 從 ctx 中讀取 tau

        # 計算 gradient
        sigmoid_grad = y * (1 - y)  # Sigmoid 梯度
        grad_input = grad_output * sigmoid_grad / tau  # 給 logits 的梯度

        # 計算 tau 的梯度
        # grad_tau = (grad_output * sigmoid_grad * (logits - y)).sum() / tau**2  # 給 tau 的梯度
        grad_tau = (grad_output * sigmoid_grad * (logits + gumbel_noise)).sum() / tau**2  # 給 tau 的梯度

        return grad_input, grad_tau, None


class GumbelSigmoid(Function):
    @staticmethod
    def forward(ctx: Function, logits, tau, eps=1e-20):
        # tau = max(0.1, ctx.tau - 0.001 * ctx.tau) if hasattr(ctx, 'tau') else tau
        U = torch.rand_like(logits)
        scale = 0.1  # 降低到 0.1
        gumbel_noise = -torch.log(-torch.log(U + eps) + eps) * scale
        y = torch.sigmoid((logits + gumbel_noise) / tau)

        ctx.save_for_backward(logits, y, gumbel_noise, tau)

        return y

    @staticmethod
    def backward(ctx: Function, grad_output):
        logits, y, gumbel_noise, tau = ctx.saved_tensors

        ###* Sigmoid 函數的梯度 ###
        sigmoid_grad = y * (1 - y)

        ###* logits 的梯度 ###
        grad_input = grad_output * sigmoid_grad / tau

        ###* tau 的梯度 ###
        grad_tau = -grad_output * sigmoid_grad * (logits + gumbel_noise) / (tau**2)
        grad_tau = grad_tau.sum()  # 總和作為標量梯度
        return grad_input, grad_tau, None


class BinarizeSTE(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, input: Tensor):
        mask = (input >= 0.5).float()
        ctx.save_for_backward(mask)
        return mask

    @staticmethod
    def backward(ctx: BackwardCFunction, grad_output):
        (mask,) = ctx.saved_tensors
        return grad_output * mask  # 只保留 mask 區域的梯度
