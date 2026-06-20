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


###* LatentGenerator — 可學習潛在向量版 (輸入是 nn.Parameter，不是目標響應) ###
#? 動機：單一 target 的優化裡，target 整個 run 不變、當輸入沒有意義 —— G 每個 epoch
#? 都吃同一個常數。改成優化一個可學習潛在向量 z：logits = MLP(z)。target 只進 loss、
#? 不進輸入，順帶讓 G 與響應維度解耦。沿用 SigmoidGenerator 的 MLP+BiScaleNorm 身體，
#? 只把「輸入」從傳入的目標向量換成自有的 nn.Parameter (隨 Adam 一起被優化)。
class LatentGenerator(SigmoidGenerator):
    """生成器 G：可學習潛在向量 z → pattern logits (forward 忽略傳入的目標)。

    in_dim 即 z 的維度 (build_generator 預設帶入 spec.size()，但已與 spec 解耦)。
    z 隨機初始化 → 天然支援「同一 target 跑多個 z restart」找更好的 pattern。
    """
    def __init__(self, in_dim, out_dim, hidden=(1024, 1024)):
        super().__init__(in_dim, out_dim, hidden=hidden)
        #? z：可學習潛在輸入，取代「目標當輸入」。是 nn.Parameter → 自動進 optimizer。
        self.z = nn.Parameter(torch.randn(in_dim, device=config.device))

    def forward(self, *args) -> Tensor:
        #? 忽略傳入的目標向量 (run_training 仍會餵 spec.concat()，這裡丟掉)，改用自有 z。
        return self.fc_patch(self.z)


###* MirrorGenerator — 左右鏡像對稱版 (對稱性做進「生成」而非資料增強) ###
#? 動機：single port 在底部中央 → 結構對垂直中線左右對稱。與其讓 G 自由輸出 625、靠運氣逼近
#? 對稱 (學長舊法是把鏡像做成資料增強，只「偏向」對稱、不保證)，不如直接讓 G「只決定半邊」，
#? 再以可微的 flip+cat 鏡像成完整圖樣 —— 天生精確對稱 + 搜尋空間砍半 (325 vs 625)。
#? 鏡像欄的梯度會從左右兩側自然累加回同一個半邊 logit (標準 autograd)。沿用 SigmoidGenerator
#? 的 MLP+BiScaleNorm 身體，只把輸出維度改成「半邊」、forward 多一步鏡像。
class MirrorGenerator(SigmoidGenerator):
    """生成器 G：左右鏡像對稱。MLP 只出半邊 logits (含中軸欄)，forward 鏡像成完整 pattern。

    對稱軸＝垂直中線；奇數寬 25 → 半邊 13 欄 (含中軸第 12 欄，鏡像時不重複)。
    out_dim 須為完全平方 (方形 pattern)。run_training 不必改：仍拿到 out_dim 個對稱 logits。
    """
    def __init__(self, in_dim, out_dim, hidden=(1024, 1024)):
        side = int(round(out_dim ** 0.5))
        if side * side != out_dim:
            raise ValueError(f"MirrorGenerator 目前只支援方形 pattern (out_dim 須為完全平方)，得到 {out_dim}")
        self.side = side                       #? 一邊的像素數 (25)
        self.half_cols = (side + 1) // 2        #? 半邊欄數，含中軸：奇數 25 → 13
        super().__init__(in_dim, side * self.half_cols, hidden=hidden)   #? MLP 只出半邊 (25×13=325)

    def forward(self, input) -> Tensor:
        half = self.fc_patch(input)                                          #? (..., side*half_cols)
        half = half.reshape(*half.shape[:-1], self.side, self.half_cols)     #? (..., side, half_cols)
        #? 鏡像欄＝半邊「不含中軸」的部分左右翻轉 (中軸欄不重複)，接到右側 → 完整對稱圖
        mirror = half[..., :self.side - self.half_cols].flip(dims=[-1])      #? (..., side, side-half_cols)
        full = torch.cat([half, mirror], dim=-1)                            #? (..., side, side)
        return full.reshape(*full.shape[:-2], -1)                           #? (..., side*side)

