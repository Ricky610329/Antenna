"""U-Net 系列代理模型。

包含：
- ``SelfAttention``：自注意力層（bottleneck 加強用）
- ``DoubleConvWithDropout``：帶 BN + Dropout 的雙卷積 block
- ``EnhancedHFSSUNet``：整合 self-attention 的 U-Net 代理模型
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from antenna.core.pattern import AntennaPattern
from antenna.core.response import AntennaResponse


class SelfAttention(nn.Module):
    """簡化的自注意力層。

    通道縮減時以 ``max(1, in_channels // 8)`` 防止通道過少。
    """

    def __init__(self, in_channels: int):
        super().__init__()
        attention_channels = max(1, in_channels // 8)
        self.query_conv = nn.Conv2d(in_channels, attention_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, attention_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, C, width, height = x.size()
        # [B, C', W*H] -> [B, W*H, C']
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        # [B, C', W*H]
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        # [B, W*H, W*H]
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        # [B, C, W*H]
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)

        # 殘差連接
        out = self.gamma * out + x
        return out


class DoubleConvWithDropout(nn.Module):
    """``(Conv => BN => ReLU => Dropout) * 2``"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int | None = None,
        dropout_prob: float = 0.15,
    ):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.double_conv(x)


class EnhancedHFSSUNet(nn.Module):
    """增強版的 HFSSUNet，包含增加通道數、Dropout 與 Self-Attention。

    Args:
        base_channels: U-Net 第一層的基礎通道數，控制模型容量。
        dropout_prob: 應用於 DoubleConv 層的 Dropout 概率。
    """

    def __init__(self, base_channels: int = 64, dropout_prob: float = 0.15):
        super().__init__()

        # --- 自動獲取輸入/輸出大小 ---
        _pattern_size = AntennaPattern.size(flatten=False)  # (H, W)
        _response_size = AntennaResponse.size(flatten=False)  # (C, L)

        if len(_response_size) != 2:
            raise ValueError(f"AntennaResponse.size() 應返回二維形狀，但得到 {_response_size}")

        self.num_response = _response_size
        self.num_pattern_pixel = _pattern_size[0] * _pattern_size[1]
        self.input_dim_h, self.input_dim_w = _pattern_size

        self.base_channels = base_channels
        self.dropout_prob = dropout_prob

        n_channels_in = 1
        n_channels_out = base_channels // 2  # Decoder 最後輸出通道數

        # --- Encoder ---
        self.down1 = DoubleConvWithDropout(n_channels_in, base_channels, dropout_prob=dropout_prob)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConvWithDropout(base_channels, base_channels * 2, dropout_prob=dropout_prob)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConvWithDropout(base_channels * 2, base_channels * 4, dropout_prob=dropout_prob)
        self.pool3 = nn.MaxPool2d(2)

        # --- Bottleneck ---
        self.bottleneck = DoubleConvWithDropout(base_channels * 4, base_channels * 8, dropout_prob=dropout_prob)

        # --- Self-Attention ---
        self.attention = SelfAttention(base_channels * 8)

        # --- Decoder ---
        self.up1 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConvWithDropout(base_channels * 8, base_channels * 4, dropout_prob=dropout_prob)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConvWithDropout(base_channels * 4, base_channels * 2, dropout_prob=dropout_prob)
        self.up3 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConvWithDropout(base_channels * 2, n_channels_out, dropout_prob=dropout_prob)

        # --- Head ---
        self.head_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head_fc = nn.Sequential(
            nn.Linear(n_channels_out, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, self.num_response[0] * self.num_response[1]),
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"num_pattern_pixel={self.num_pattern_pixel}, "
            f"num_response={self.num_response}, "
            f"base_channels={self.base_channels}, "
            f"dropout_prob={self.dropout_prob})"
        )

    def forward(self, x: Tensor) -> Tensor:
        # 0. Reshape Input: (...) -> (B, 1, H, W)
        x = x.unsqueeze(0)
        if x.dim() > 2:
            x = torch.flatten(x, 1)
        x_img = x.view(-1, 1, self.input_dim_h, self.input_dim_w)

        # 1. Encoder
        x1 = self.down1(x_img)
        x2 = self.pool1(x1)
        x3 = self.down2(x2)
        x4 = self.pool2(x3)
        x5 = self.down3(x4)
        x6 = self.pool3(x5)

        # 2. Bottleneck
        bottle = self.bottleneck(x6)

        # 3. Attention
        attn_bottle = self.attention(bottle)

        # 4. Decoder
        u1 = self.up1(attn_bottle)
        u1 = F.interpolate(u1, size=x5.shape[2:], mode="bilinear", align_corners=True)
        c1 = self.up_conv1(torch.cat([x5, u1], dim=1))

        u2 = self.up2(c1)
        u2 = F.interpolate(u2, size=x3.shape[2:], mode="bilinear", align_corners=True)
        c2 = self.up_conv2(torch.cat([x3, u2], dim=1))

        u3 = self.up3(c2)
        u3 = F.interpolate(u3, size=x1.shape[2:], mode="bilinear", align_corners=True)
        c3 = self.up_conv3(torch.cat([x1, u3], dim=1))

        # 5. Head
        out_pool = self.head_pool(c3)
        out_flat = torch.flatten(out_pool, 1)
        out_fc = self.head_fc(out_flat)

        # 6. Final Reshape
        return out_fc.view(-1, self.num_response[0], self.num_response[1])
