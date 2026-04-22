"""條件變分自動編碼器 (Conditional VAE, CVAE) 生成器。"""

import torch
import torch.nn.functional as F
from loguru import logger
from torch import Tensor, nn

from antenna.core.pattern import AntennaPattern
from antenna.core.response import AntennaResponse
from antenna.types import Tensor_B_N
from antenna.utils.config import config
from antenna.utils.data import size_converter

# 數值穩定性相關魔術數字
_LOGVAR_CLAMP_MIN = -10.0
_LOGVAR_CLAMP_MAX = 10.0
_BCE_EPS = 1e-6


class CVAE(nn.Module):
    """條件變分自動編碼器 (CVAE)。

    以 ``[pattern, response]`` 為 encoder 輸入，``[z, response]`` 為 decoder 輸入，
    產生重建的 pattern logits；支援直接 logits BCE 或配合 STE 的二值化 BCE。
    """

    def __init__(
        self,
        latent_dim: int,
        pattern_size: int | None = None,
        response_size: int | None = None,
    ):
        """初始化 CVAE 模型。

        Args:
            latent_dim: 潛在空間 (z) 的維度。
            pattern_size: 天線圖案展平後的大小；預設從 ``AntennaPattern.size(flatten=True)`` 取。
            response_size: EM 響應展平後的大小；預設從 ``AntennaResponse.size(flatten=True)`` 取。
        """
        super().__init__()
        self.pattern_size = pattern_size or AntennaPattern.size(flatten=True)
        self.response_size = response_size or AntennaResponse.size(flatten=True)
        self.latent_dim = latent_dim
        hidden_dim = 256  # 經驗值；與過往訓練超參維持一致

        # Encoder: [Pattern + Response] -> [Latent Params]  (僅訓練時使用)
        self.encoder_fc = nn.Sequential(
            nn.Linear(self.pattern_size + self.response_size, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: [Latent z + Response] -> [Pattern Logits]
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim + self.response_size, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, self.pattern_size),
        )

        self.to(config.device)

    def encode(self, pattern: Tensor_B_N, response: Tensor_B_N) -> tuple[Tensor, Tensor]:
        """``(pattern, response) -> (mu, logvar)``。

        ``logvar`` 會 clamp 到 ``[_LOGVAR_CLAMP_MIN, _LOGVAR_CLAMP_MAX]`` 以避免
        ``exp(logvar)`` 溢位造成 NaN。
        """
        inputs = torch.cat([pattern, response], dim=-1)
        h = self.encoder_fc(inputs)
        logvar = torch.clamp(self.fc_logvar(h), min=_LOGVAR_CLAMP_MIN, max=_LOGVAR_CLAMP_MAX)
        return self.fc_mu(h), logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """重參數化技巧：``z = mu + epsilon * std``。"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor, response: Tensor_B_N) -> Tensor:
        """``(z, response) -> logits``。"""
        inputs = torch.cat([z, response], dim=-1)
        return self.decoder_fc(inputs)

    def forward(self, pattern: Tensor, response: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """``(pattern, response) -> (recon_logits, mu, logvar)``。"""
        pattern = size_converter(AntennaPattern, pattern, flatten=True, batch=True)
        response = size_converter(AntennaResponse, response, flatten=True, batch=True)
        mu, logvar = self.encode(pattern, response)
        z = self.reparameterize(mu, logvar)
        recon_pattern = self.decode(z, response)
        return recon_pattern, mu, logvar

    def inference(
        self,
        target_response: Tensor,
        z: Tensor | None = None,
        best: tuple[Tensor, Tensor] | None = None,
        noise_scale: float = 0.0,
    ) -> AntennaPattern:
        """推論：回傳以目標響應條件化的二值化 pattern。

        Args:
            target_response: 目標 EM 響應。
            z: 指定潛在向量；若與 ``best`` 皆為 None 則隨機採樣 ``N(0, 1)``。
            best: ``(best_pattern, best_response)`` — 若提供則在其 ``mu`` 附近加噪探索。
            noise_scale: ``best`` 微調時的高斯擾動強度。
        """
        self.eval()
        target_response = size_converter(AntennaResponse, target_response, flatten=True, batch=True)
        if best is not None:  # 基於歷史最佳解進行微調
            best_pattern, best_response = best
            with torch.no_grad():
                best_pattern = size_converter(AntennaPattern, best_pattern, flatten=True, batch=True)
                best_response = size_converter(AntennaResponse, best_response, flatten=True, batch=True)
                mu, _ = self.encode(best_pattern, best_response)
                z = mu + torch.randn_like(mu) * noise_scale

        elif z is None:  # 全域隨機探索
            z = torch.randn(target_response.size(0), self.fc_mu.out_features).to(target_response.device)

        return AntennaPattern.binarization(self.decode(z, target_response))

    def elbo_logits(self, recon_logits: Tensor, target: Tensor, mu: Tensor, logvar: Tensor, beta: float = 1) -> Tensor:
        """基於原始 logits 的 ELBO loss。

        使用 ``binary_cross_entropy_with_logits`` 內建 sigmoid，可防止 ``log(sigmoid(x))``
        的數值溢位。

        Args:
            recon_logits: Decoder 的直接輸出（未經 sigmoid）。
            target: 真實圖樣 (0 或 1)。
        """
        BCE = F.binary_cross_entropy_with_logits(recon_logits, target, reduction="sum")
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + beta * KLD

    def elbo_binarized(self, recon_bin: Tensor, target: Tensor, mu: Tensor, logvar: Tensor, beta: float = 1) -> Tensor:
        """基於 STE 二值化結果的 ELBO loss。

        採用 ``mean`` reduction 以避免 pattern 很大時 loss 數值膨脹。
        最後會偵測 inf/NaN，若發生則回傳帶梯度的 0 讓 optimizer 跳過該步。
        """
        recon_safe = torch.clamp(recon_bin, min=_BCE_EPS, max=1.0 - _BCE_EPS)
        target = target.float()

        BCE = F.binary_cross_entropy(recon_safe, target, reduction="mean")
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        total_loss = BCE + beta * KLD

        # 最後防線：若數值崩壞則回傳帶梯度的 0，避免 optimizer 崩潰
        if torch.isinf(total_loss) or torch.isnan(total_loss):
            logger.warning(f"CVAE elbo_binarized 偵測到 inf/NaN loss: BCE={BCE.item()}, KLD={KLD.item()}")
            return torch.tensor(0.0, requires_grad=True, device=total_loss.device)

        return total_loss

    def fit(
        self,
        data_source,
        optimizer: torch.optim.Optimizer,
        epochs: int = 100,
        batch_size: int = 32,
        beta: float = 1,
        use_ste: bool = True,
    ) -> dict:
        """封裝好的訓練迴圈。

        Args:
            data_source: ``Tuple(patterns, responses)`` 的 Tensors，或 PyTorch
                ``DataLoader`` / ``Dataset``。
            optimizer: 優化器 (例如 Adam)。
            epochs: 訓練輪數。
            batch_size: 當 ``data_source`` 為 Tensor/Dataset 時使用。
            beta: KL divergence 的權重 (Beta-VAE)。
            use_ste: ``True`` 使用 ``elbo_binarized``（STE 二值化），``False`` 使用 ``elbo_logits``。

        Returns:
            含 ``total_loss``, ``bce``, ``kld`` 歷史紀錄的 dict。
        """
        if hasattr(data_source, "__len__") and len(data_source) == 0:
            logger.warning("CVAE.fit: data source is empty, skipping training loop.")
            return {"total_loss": [], "bce": [], "kld": []}

        self.train()

        # 準備 dataloader
        if isinstance(data_source, torch.utils.data.DataLoader):
            dataloader = data_source
        elif isinstance(data_source, torch.utils.data.Dataset):
            dataloader = torch.utils.data.DataLoader(data_source, batch_size=batch_size, shuffle=True)
        elif isinstance(data_source, (tuple, list)) and len(data_source) == 2:
            patterns, responses = data_source
            if not (torch.is_tensor(patterns) and torch.is_tensor(responses)):
                raise TypeError("Data source tuple/list must contain PyTorch Tensors.")
            dataset = torch.utils.data.TensorDataset(patterns, responses)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        else:
            raise TypeError(
                f"Unsupported data_source type: {type(data_source)}. "
                "Expected DataLoader, Dataset, or (Pattern_Tensor, Response_Tensor)."
            )

        history: dict[str, list[float]] = {"total_loss": [], "bce": [], "kld": []}

        for _epoch in range(epochs):
            epoch_loss = 0.0
            epoch_bce = 0.0
            epoch_kld = 0.0
            steps = 0

            for batch_x, batch_c in dataloader:
                batch_x = size_converter(AntennaPattern, batch_x.to(config.device), flatten=True, batch=True)
                batch_c = size_converter(AntennaResponse, batch_c.to(config.device), flatten=True, batch=True)

                recon_logits, mu, logvar = self.forward(batch_x, batch_c)

                if use_ste:
                    recon_ste = AntennaPattern.binarization(recon_logits)
                    loss = self.elbo_binarized(recon_ste, batch_x, mu, logvar, beta)
                else:
                    loss = self.elbo_logits(recon_logits, batch_x, mu, logvar, beta)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                with torch.no_grad():
                    epoch_loss += loss.item()
                    # 拆解項僅供紀錄參考，不影響訓練
                    kld_val = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    epoch_kld += beta * kld_val.item()
                    epoch_bce += loss.item() - (beta * kld_val.item())
                    steps += 1

            if steps > 0:
                history["total_loss"].append(epoch_loss / steps)
                history["bce"].append(epoch_bce / steps)
                history["kld"].append(epoch_kld / steps)

        return history

    def generate(self, response: Tensor, n_samples: int = 1) -> Tensor:
        """反向設計生成：給定目標響應，隨機採樣 ``z`` 並解碼出 ``n_samples`` 個候選 pattern logits。

        Args:
            response: 目標 EM 響應 ``(1, response_size)``。
            n_samples: 要生成的候選圖案數量。

        Returns:
            形狀 ``(n_samples, pattern_size)`` 的候選 logits。
        """
        z = torch.randn(n_samples, self.latent_dim).to(config.device)
        # 將目標響應複製 n_samples 份以匹配 z 的 batch
        response_batch = response.repeat(n_samples, 1)
        return self.decode(z, response_batch)
