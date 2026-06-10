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
- SM 建構簽名 (checkpoint, in_dim, response_shape, **架構參數)，回傳 SurrogateModel。
- 維度由訓練端推好傳入，模型不碰全域註冊狀態 (AntennaResponse/AntennaPattern)。
"""
from antenna.models import SigmoidGEN
from antenna.smodels import OldSM

GENERATORS = {
    "sigmoid": SigmoidGEN,      # MLP + BiScaleNorm (預設 hidden=(1024, 1024))
}

SURROGATES = {
    "mlp": OldSM,               # HFSSNet 純 MLP + Ranger + MSE (預設 hidden=(2048,1024,512,128,64))
}
