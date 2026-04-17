"""代理模型 (Surrogate Model) 子模組的單元測試。

涵蓋：
- ``HFSSNet``：forward shape、device 搬移。
- ``EnhancedHFSSUNet`` + ``SelfAttention`` + ``DoubleConvWithDropout``：forward shape、device 搬移。
- ``antenna.smodels`` shim 能匯出所有原本的符號。
"""

import torch


# ---------------------------------------------------------------------------
# HFSSNet
# ---------------------------------------------------------------------------
def test_hfss_net_forward_shape():
    """``HFSSNet`` 的 forward 輸出應符合指定 response 形狀。"""
    from antenna.models.surrogates.hfss_net import HFSSNet

    num_pattern_pixel = 100
    num_response = (2, 5)
    model = HFSSNet(num_pattern_pixel=num_pattern_pixel, num_response=num_response)

    dummy = torch.zeros(num_pattern_pixel, device=next(model.parameters()).device)
    out = model(dummy)

    assert out.shape == num_response


def test_hfss_net_repr():
    """``__repr__`` 應包含重要的超參數。"""
    from antenna.models.surrogates.hfss_net import HFSSNet

    model = HFSSNet(num_pattern_pixel=64, num_response=(3, 4))
    r = repr(model)
    assert "HFSSNet" in r
    assert "64" in r
    assert "(3, 4)" in r


def test_hfss_net_device_move_cpu():
    """將模型搬到 CPU 後，參數 device 應更新。"""
    from antenna.models.surrogates.hfss_net import HFSSNet

    model = HFSSNet(num_pattern_pixel=16, num_response=(1, 2))
    model.to("cpu")
    for p in model.parameters():
        assert p.device.type == "cpu"


# ---------------------------------------------------------------------------
# U-Net 系列
# ---------------------------------------------------------------------------
def test_self_attention_forward_shape():
    """``SelfAttention`` 必須保留輸入的空間形狀。"""
    from antenna.models.surrogates.unet import SelfAttention

    in_channels = 16
    attn = SelfAttention(in_channels)
    x = torch.randn(2, in_channels, 8, 8)
    out = attn(x)
    assert out.shape == x.shape


def test_self_attention_small_channels():
    """通道數極少時 ``attention_channels`` 仍應 >=1。"""
    from antenna.models.surrogates.unet import SelfAttention

    attn = SelfAttention(3)  # 3 // 8 == 0 → 會被 clip 成 1
    # query_conv 的 out_channels 即為 attention_channels
    assert attn.query_conv.out_channels >= 1


def test_double_conv_forward_shape():
    """``DoubleConvWithDropout`` 應保留 H/W，僅改變通道數。"""
    from antenna.models.surrogates.unet import DoubleConvWithDropout

    block = DoubleConvWithDropout(in_channels=1, out_channels=8, dropout_prob=0.1)
    x = torch.randn(2, 1, 16, 16)
    out = block(x)
    assert out.shape == (2, 8, 16, 16)


def test_enhanced_hfss_unet_forward_shape():
    """``EnhancedHFSSUNet`` 的 forward 必須輸出 ``(B, *response_size)``。

    U-Net 使用三次 2x2 MaxPool，所以輸入大小必須能被 8 整除。
    """
    from antenna.core.pattern import AntennaPattern
    from antenna.core.response import AntennaResponse
    from antenna.models.surrogates.unet import EnhancedHFSSUNet

    AntennaPattern.setDefaultCoordinate((0, 16, 0, 16))
    AntennaResponse.registerLabels("response", x=(0, 1, 5))

    model = EnhancedHFSSUNet(base_channels=8, dropout_prob=0.0)
    model.eval()  # 避免 BN 在 batch=1 時報錯
    dummy = torch.randn(2, 16 * 16, device=next(model.parameters()).device)
    out = model(dummy)

    assert out.shape == (2, 1, 5)


def test_enhanced_hfss_unet_device_move_cpu():
    """``EnhancedHFSSUNet`` 搬到 CPU 後，參數 device 應更新。"""
    from antenna.core.pattern import AntennaPattern
    from antenna.core.response import AntennaResponse
    from antenna.models.surrogates.unet import EnhancedHFSSUNet

    AntennaPattern.setDefaultCoordinate((0, 16, 0, 16))
    AntennaResponse.registerLabels("response", x=(0, 1, 5))

    model = EnhancedHFSSUNet(base_channels=8, dropout_prob=0.0)
    model.to("cpu")
    for p in model.parameters():
        assert p.device.type == "cpu"


# ---------------------------------------------------------------------------
# antenna.smodels shim
# ---------------------------------------------------------------------------
def test_smodels_shim_exports():
    """``antenna.smodels`` 必須 re-export 所有原有的類別與工廠函數。"""
    import antenna.smodels as shim

    expected = {
        "HFSSNet",
        "SelfAttention",
        "DoubleConvWithDropout",
        "EnhancedHFSSUNet",
        "SurrogateModel",
        "OldSM",
        "UNetSM",
    }
    for name in expected:
        assert hasattr(shim, name), f"antenna.smodels 缺少 {name}"


def test_smodels_shim_all_symbol():
    """``antenna.smodels.__all__`` 必須涵蓋所有原本公開符號。"""
    import antenna.smodels as shim

    expected = {
        "HFSSNet",
        "SelfAttention",
        "DoubleConvWithDropout",
        "EnhancedHFSSUNet",
        "SurrogateModel",
        "OldSM",
        "UNetSM",
    }
    assert expected.issubset(set(shim.__all__))


def test_smodels_shim_is_same_class():
    """shim 匯出的類別應與 canonical 來源是同一個物件。"""
    from antenna.models.surrogates import HFSSNet as Canonical
    from antenna.smodels import HFSSNet as Shim

    assert Canonical is Shim


def test_models_reexports_surrogates():
    """``antenna.models`` 也應 re-export 代理模型相關符號。"""
    from antenna.models import HFSSNet, OldSM, SurrogateModel

    assert HFSSNet is not None
    assert OldSM is not None
    assert SurrogateModel is not None
