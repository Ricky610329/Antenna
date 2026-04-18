"""Legacy re-export shim.

代理模型的實作已搬遷至 :mod:`antenna.models.surrogates`；本檔案保留向後相容，
僅作為 re-export hub 讓既有的 ``train_*.py`` 腳本繼續可用。

新程式請改用:

    from antenna.models.surrogates import SurrogateModel, HFSSNet, OldSM, UNetSM
"""

from antenna.models.surrogates import (
    DoubleConvWithDropout,
    EnhancedHFSSUNet,
    HFSSNet,
    OldSM,
    SelfAttention,
    SurrogateModel,
    UNetSM,
)

__all__ = [
    "HFSSNet",
    "SelfAttention",
    "DoubleConvWithDropout",
    "EnhancedHFSSUNet",
    "SurrogateModel",
    "OldSM",
    "UNetSM",
]
