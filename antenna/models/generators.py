"""
antenna/models/generators.py — 生成器 GEN (目標響應 → pattern logits)。

約定：建構簽名 (in_dim, out_dim, **架構參數)；forward 不做 STE 二值化、
不碰 tau (那是訓練管線的固定一步，tau 由 ACP 控制)。
新生成器照此約定寫好後，到 antenna/zoo.py 登記一行即可。
"""
import torch
from torch import Tensor, nn

from antenna.utils import config


#? 最絕對負值映到 -1，而 0 仍保持 0。這讓後續 sigmoid/STE 的分界更穩定、梯度尺度更一致。
class BiScaleNorm(nn.Module):
    def __init__(self):
        super(BiScaleNorm, self).__init__()

    def forward(self, input_vector):
        # 大於 0 的值的正規化
        #? 正半邊：每個正值除以全域最大值 → 落在 (0, 1]；非正值位置填 0。
        max_val = torch.max(input_vector)
        positive_normalized = torch.where(input_vector > 0, input_vector / max_val, torch.tensor(0.0, device=input_vector.device))

        # 小於 0 的值的正規化
        #? 負半邊：每個負值除以最小值的絕對值 → 落在 [-1, 0)；非負值位置填 0。
        min_val = torch.min(input_vector)
        negative_normalized = torch.where(input_vector < 0, input_vector / torch.abs(min_val), torch.tensor(0.0, device=input_vector.device))

        # 合併正規化結果
        #? 兩半邊在彼此互斥的位置上各填 0，相加即還原成完整向量 (值域 [-1, 1])。
        normalized_vector = positive_normalized + negative_normalized
        return normalized_vector

###* SigmoidGenerator — 主用生成器 (MLP + BiScaleNorm + AntennaPattern.binarization) ###
#? 本專案實際主用的生成器，入口 train_single.py / train_dual.py 即用它。
#? 結構：response → MLP 1024-1024 → BiScaleNorm，最後接
class SigmoidGenerator(nn.Module):
    """
    生成器 G (純模型)：目標響應向量 → pattern logits。

    刻意「不做」STE 二值化、不認識 tau —— 二值化是訓練管線的固定一步
    (run_training 用 ACP 的 tau 統一套用)，模型只負責映射；新模型照此約定
    寫好後在 antenna/zoo.py 登記一行即可。
    """
    def __init__(self, in_dim, out_dim, hidden=(1024, 1024)):
        super(SigmoidGenerator,self).__init__()
        #? 維度由訓練端傳入 (不碰 AntennaResponse/AntennaPattern 的全域註冊狀態)。
        #? hidden 可由 config 指定；每層 Linear→PReLU，末層 Linear→BiScaleNorm (不接 PReLU)。
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.PReLU()]
            prev = h
        layers += [nn.Linear(prev, out_dim), BiScaleNorm()]
        self.fc_patch = nn.Sequential(*layers)
        self.to(config.device)

    def forward(self, input) -> Tensor:
        return self.fc_patch(input)

