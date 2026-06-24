"""
antenna/zoo.py — 模型動物園：所有可用的 GEN / SM 架構都登記在這，一眼看完、一處新增。

config 用名字指定：
    generator: sigmoid          # 或 {name: sigmoid, hidden: [512, 512]} 微調架構
    surrogate: {name: mlp, ...} # name 可省略 → mlp

新增模型 = 寫一個純 nn.Module (或工廠)，在下面的 dict 加一行。

設計約定 (解耦的關鍵)：
- GEN 建構簽名 (in_dim, out_dim, **架構參數)；forward: spec 向量 → pattern logits。
  模型「不做」STE 二值化、不認識 tau —— 那是訓練管線的固定一步 (run_training)，
  tau 由 ACP (AdaptiveCyclicalScheduler) 單獨控制。
- SM 建構簽名 (checkpoint, in_dim, response_shape, lr, min_loss, max_epoch, **架構參數)，
  回傳 SurrogateModel；lr 與單筆訓練門檻來自 YAML 的 hfss 區段，由訓練端顯式傳入。
- 維度由訓練端推好傳入，模型不碰全域註冊狀態、不讀全域 config。
"""
from antenna.models import (
    MLPSurrogate, SigmoidGenerator, LatentGenerator, MirrorGenerator, BatchLatentGenerator,
)

GENERATORS = {
    "sigmoid": SigmoidGenerator,      # MLP + BiScaleNorm (預設 hidden=(1024, 1024))；輸入＝目標響應
    "latent":  LatentGenerator,       # 同上 MLP，但輸入改成可學習潛在向量 z (target 只進 loss)
    "mirror":  MirrorGenerator,       # 左右鏡像對稱：MLP 只出半邊→flip 成完整圖 (對稱+搜尋空間砍半)
    "batch_latent": BatchLatentGenerator,  # 同批 K 候選：高斯雲繞可學中心 z* (reparam+σ退火)；多候選分支見 training.py
}

SURROGATES = {
    "mlp": MLPSurrogate,               # HFSSNet 純 MLP + Ranger + MSE (預設 hidden=(2048,1024,512,128,64))
}
