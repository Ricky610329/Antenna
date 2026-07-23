"""
antenna/models — 模型層：外殼 (shell) / 生成器 (generators) / 代理模型 (surrogates)。
可用架構的「選單」在 antenna/zoo.py。
"""
from .shell import Models
from .generators import (
    BiScaleNorm, SigmoidGenerator, LatentGenerator, MirrorGenerator,
    BatchLatentGenerator, MultiScaleGenerator, DirectPatternGenerator,
)
from .surrogates import (
    SurrogateModel, HFSSNet, MLPSurrogate, EnsembleSurrogate, EnsembleMLPSurrogate,
    CNNNet, CNNSurrogate, ResCNNNet, ResCNNSurrogate,
)
