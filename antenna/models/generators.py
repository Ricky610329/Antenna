"""
antenna/models/generators.py — 生成器 GEN (目標響應 → pattern logits)。

約定：建構簽名 (in_dim, out_dim, **架構參數)；forward 不做 STE 二值化、
不碰 tau (那是訓練管線的固定一步，tau 由 ACP 控制)。
新生成器照此約定寫好後，到 antenna/zoo.py 登記一行即可。
"""
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from antenna.utils import config

#? BiScaleNorm 除數下限：遠小於任何正常 logit 尺度 (O(0.1~10)) → 正常列不受影響 (golden-neutral)，
#  只在退化列 (該半邊極值=0) 防 0/0 反向 NaN。
_EPS = 1e-12


#? 最絕對負值映到 -1，而 0 仍保持 0。這讓後續 sigmoid/STE 的分界更穩定、梯度尺度更一致。
class BiScaleNorm(nn.Module):
    def __init__(self):
        super(BiScaleNorm, self).__init__()

    def forward(self, input_vector):
        # 大於 0 的值的正規化
        #? 正半邊：每個正值除以「該列」最大值 → 落在 (0, 1]；非正值位置填 0。
        #? per-row (dim=-1)：1-D 時等同全域 max (golden 不動)；(K, N) batch 時每個候選各自正規化、不互相耦合。
        #! clamp_min(_EPS)：退化列 (無正值 → max=0) 時除數會是 0，torch.where 未選分支的 0/0=NaN
        #  雖被 forward 遮罩 (值仍 0)，卻會沿反向污染整列梯度。夾一個極小下限 → 退化列梯度有限、
        #  正常列 max≫_EPS 不受影響 (golden-neutral)。多候選 batch 任一壞候選才不會拖垮整批。
        max_val = torch.max(input_vector, dim=-1, keepdim=True).values.clamp_min(_EPS)
        positive_normalized = torch.where(input_vector > 0, input_vector / max_val, torch.tensor(0.0, device=input_vector.device))

        # 小於 0 的值的正規化
        #? 負半邊：每個負值除以「該列」最小值的絕對值 → 落在 [-1, 0)；非負值位置填 0。
        min_val = torch.abs(torch.min(input_vector, dim=-1, keepdim=True).values).clamp_min(_EPS)
        negative_normalized = torch.where(input_vector < 0, input_vector / min_val, torch.tensor(0.0, device=input_vector.device))

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


###* BatchLatentGenerator — 同批多候選 (高斯雲繞可學中心 z*，reparam) ###
#? 動機：單 target 反向設計裡，與其順序跑多個 z restart，不如同一輪同時生成 K 個候選、在 SM 上
#? 平行評分、挑最好的一張送昂貴 HFSS —— 同樣 HFSS 預算下搜更廣、每次評估都花在 best-of-K。
#? latent 設計 (γ)：q(z)=N(z*, σ²I)，reparam 抽 K 個 z_k = z* + σ·ε_k (ε_k~N(0,I))。中心 z* 隨
#? Adam 一起被優化、σ 隨訓練退火 (前期探索大、後期收斂)。沿用 SigmoidGenerator 的 MLP+BiScaleNorm，
#? 只把輸入換成「一批 K 個潛在向量」→ forward 回 (K, out_dim)。需 BiScaleNorm per-row (每候選各自
#? 正規化) 與 SM forward 支援 batch (見 surrogates ndim 分支)。多候選的生成/選擇/聚合在 run_training。
class BatchLatentGenerator(SigmoidGenerator):
    """生成器 G：高斯雲繞可學中心 z* (reparam) → 同批 K 個候選 pattern logits (K, out_dim)。

    in_dim 即 z 的維度 (build_generator 預設帶 spec.size()，但已與 spec 解耦)。
    forward 忽略傳入目標 (沿用 LatentGenerator 約定)，改抽自有的 K 個潛在向量。
    """
    def __init__(self, in_dim, out_dim, hidden=(1024, 1024), num_candidates=8,
                 sigma=0.5, sigma_min=0.05):
        super().__init__(in_dim, out_dim, hidden=hidden)
        self.K = int(num_candidates)            #? 同批候選數
        self.sigma_start = float(sigma)         #? σ 退火起點 (探索半徑)
        self.sigma_min = float(sigma_min)       #? σ 退火終點
        self.sigma = float(sigma)               #? 當前 σ (由 anneal_sigma 更新)
        #? z*：可學中心，隨 Adam 一起優化 (取代「目標當輸入」)。
        self.z_center = nn.Parameter(torch.randn(in_dim, device=config.device))

    def anneal_sigma(self, frac: float):
        """frac = epoch/epochs ∈ [0,1]：σ 從 sigma_start 線性退火到 sigma_min (探索→收斂)。"""
        frac = min(1.0, max(0.0, frac))
        self.sigma = self.sigma_start + (self.sigma_min - self.sigma_start) * frac

    def forward(self, *args) -> Tensor:
        #? reparam：z_k = z* + σ·ε_k，ε_k~N(0,I)。ε 不可訓 (detach 天然，randn 無梯度)，
        #  梯度經 z 流回 z_center 與 σ (σ 目前非 Parameter，僅退火常數)。
        eps = torch.randn(self.K, self.z_center.numel(), device=config.device)
        z = self.z_center.unsqueeze(0) + self.sigma * eps     #? (K, in_dim)
        return self.fc_patch(z)                               #? (K, out_dim)


###* MultiScaleGenerator — 多尺度淺層加性生成器 (coarse-to-fine 先驗) ###
#? 動機 (cold-start 視角)：不靠歷史資料時，generator 不是在學映射，而是把 pattern 用網路「重參數化」、
#? 對單一 target 做梯度下降 (≈ deep image prior / neural reparam topology-opt)。此時架構＝先驗。
#? 多尺度加性：各尺度 (1×1 / 5×5 / 13×13) 各一個淺線性頭 → 上採樣到 side×side → 相加 (Laplacian-pyramid 式)。
#?   - 最細只到 13×13 → 輸出 band-limited、結構上偏「連通平滑」(減少對 SC/island 依賴)；
#?   - 參數少、層淺 → 每個 task 從零更快收斂 (cold-start 友善)；
#?   - 「先粗後細」天生是一種按尺度退火 (粗=全域、細=局部)，比盲目 tau 加熱有結構。
#? ⚠ 表達力下限：太細的單像素特徵可能擬不出 (capacity 換先驗)；不夠就把更大尺度加進 scales。
class MultiScaleGenerator(nn.Module):
    """生成器 G：spec → 各尺度淺頭 → 上採樣相加 → side×side logits (BiScaleNorm)。

    建構簽名照 zoo 約定 (in_dim, out_dim, **架構參數)；scales 可由 config 指定 (預設 (1,5,13))。
    forward 不做 STE 二值化、不認識 tau (那是訓練管線的固定一步)。
    """
    def __init__(self, in_dim, out_dim, scales=(1, 5, 13)):
        super().__init__()
        side = int(round(out_dim ** 0.5))
        if side * side != out_dim:
            raise ValueError(f"MultiScaleGenerator 只支援方形 pattern (out_dim 須完全平方)，得到 {out_dim}")
        self.side = side
        self.scales = tuple(int(s) for s in scales)
        #? 每尺度一個「淺」線性頭：in_dim → s*s (無隱藏層)；參數 = in_dim·Σ s² ≪ 主 MLP 的 ~1M。
        self.heads = nn.ModuleList([nn.Linear(in_dim, s * s) for s in self.scales])
        self.norm = BiScaleNorm()
        self.to(config.device)

    def forward(self, input) -> Tensor:
        lead = input.shape[:-1]                              #? () 單筆 / (B,) 批次
        acc = None
        for s, head in zip(self.scales, self.heads):
            grid = head(input).reshape(-1, 1, s, s)          #? (N,1,s,s)
            up = F.interpolate(grid, size=(self.side, self.side),
                               mode="bilinear", align_corners=False)   #? 平滑上採樣 → 連通先驗
            up = up.reshape(*lead, self.side, self.side)
            acc = up if acc is None else acc + up            #? 各尺度相加 (coarse + fine)
        return self.norm(acc.reshape(*lead, -1))             #? (..., side²) → BiScaleNorm (per-row)

