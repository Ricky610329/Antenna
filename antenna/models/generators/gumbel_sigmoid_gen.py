import torch
from torch import nn

from antenna.core.pattern import AntennaPattern
from antenna.core.response import AntennaResponse
from antenna.models.autograd import GumbelSigmoid
from antenna.models.components import BiScaleNorm
from antenna.utils.config import config


class GumbelSigmoidGEN(nn.Module):
    """
    Generator Model
    """

    def __init__(self):
        super().__init__()
        pattern_size = AntennaPattern.size(flatten=True)
        self.fc_patch = nn.Sequential(
            nn.Linear(AntennaResponse.size(flatten=True), pattern_size),
            nn.PReLU(),
            nn.Linear(pattern_size, pattern_size * 2),
            nn.PReLU(),
            nn.Linear(pattern_size * 2, pattern_size),
            nn.PReLU(),
            nn.Linear(pattern_size, pattern_size),
            BiScaleNorm(),
        )
        self.tau = nn.Parameter(torch.tensor(5.0, requires_grad=True))
        self.tau_history = []

        for m in self.fc_patch:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 1.0)
            if isinstance(m, nn.PReLU):
                m.weight.data.fill_(0.25)

        self.to(config.device)

    def forward(self, input, *, is_trainig: bool = True):
        """
        輸出 Gumbel-Sigmoid 處理過的結果
        """
        self.logits = torch.clamp(  # 防止梯度爆炸
            self.fc_patch(input), min=-5.0, max=5.0
        )
        # # 在訓練階段使用 Gumbel-Sigmoid 來保持梯度
        # if is_trainig:
        #     x = GumbelSigmoid.apply(x, tau)  # 訓練階段使用 Gumbel-Sigmoid 進行輸出
        # else:
        #     x = (x >= 0.5).float()  # 推論階段，硬性 binarize

        x = GumbelSigmoid.apply(self.logits, self.tau)  # 訓練階段使用 Gumbel-Sigmoid 進行輸出
        # self.anneal_tau()
        self.tau_history.append(self.tau.detach().cpu().item())

        # x = BinarizeSTE.apply(x)

        return x

    def anneal_tau(self, rate=0.995, min_tau=0.1):
        """
        Annealing (退火)

        在訓練初期，較大的 tau 值會使得輸出更為平滑，有利於模型探索不同的解空間。

        在訓練後期，較小的 tau 值會使輸出更接近離散的 0 和 1，從而幫助模型收斂到一個確定的離散解。
        """
        # self.tau = max(min_tau, self.tau * rate)
        self.tau = torch.clamp(self.tau, min=0.1)
        self.tau_recoed.append(self.tau.detach().cpu())

    def binarize(self, threshold=0.5):
        binarized_output = (torch.sigmoid(self.logits) > threshold).float()
        return AntennaPattern(binarized_output)
        return AntennaPattern((self.x >= threshold).float())
