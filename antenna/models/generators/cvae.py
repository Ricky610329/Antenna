from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from antenna.core.pattern import AntennaPattern
from antenna.core.response import AntennaResponse
from antenna.types import Tensor_B_N
from antenna.utils.config import config
from antenna.utils.data import size_converter


class CVAE(nn.Module):
    """
    條件變分自動編碼器 (CVAE)
    """

    def __init__(
        self,
        latent_dim: int,
        pattern_size: int | None = None,
        response_size: int | None = None,
        binary_fn: Callable[..., Tensor] = AntennaPattern.binarization,
    ):
        """
        初始化 CVAE 模型。

        Args:
            pattern_size (int): 天線圖案展平後的大小
            response_size (int): EM 響應展平後的大小
            latent_dim (int): 潛在空間 (z) 的維度。
        """
        super().__init__()
        self.pattern_size = pattern_size or AntennaPattern.size(flatten=True)  # ? x
        self.response_size = response_size or AntennaResponse.size(flatten=True)  # ? c
        self.latent_dim = latent_dim
        hidden_dim = 256

        # * Encoder: [Pattern + Response] -> [Latent Params]
        #! Only Training
        self.encoder_fc = nn.Sequential(
            nn.Linear(self.pattern_size + self.response_size, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.fc_binary = binary_fn
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)  # 均值
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # 變異數對數

        # * Decoder: [Latent z + Response] -> [Pattern Logits]
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim + self.response_size, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, self.pattern_size),
        )

        self.to(config.device)
        # logger.info(f"CVAE Model Initialized: pattern_size={pattern_size}, response_size={response_size}, latent_dim={latent_dim}")

    def encode(self, pattern: Tensor_B_N, response: Tensor_B_N) -> tuple[Tensor, Tensor]:
        """
        (pattern, response) -> (mu, logvar)

        Args:
            pattern (Tensor): 批次的二進制天線圖案 (B, pattern_size)。
            response (Tensor): 批次的對應 EM 響應 (B, response_size)。

        Returns:
            Tuple[Tensor, Tensor]: 潛在空間的 (mu, logvar)。
        """
        inputs = torch.cat([pattern, response], dim=-1)
        h = self.encoder_fc(inputs)
        return self.fc_mu(h), torch.clamp(self.fc_logvar(h), min=-10.0, max=10.0)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        z = mu + epsilon * std

        :param mu: 潛在空間的平均。
        :param logvar: 潛在空間的 log 變異數。
        :return Tensor: 採樣出的潛在向量 z。
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # 從標準常態分佈中採樣雜訊
        return mu + eps * std

    def decode(self, z: Tensor, response: Tensor_B_N) -> Tensor:
        """
        (z, Response) -> Logits

        :param z: 批次的潛在向量 (B, latent_dim)。
        :param response: 批次的目標 EM 響應 (B, response_size)。
        :return Tensor: 重建圖案的 Logits (B, pattern_size)。
        """
        inputs = torch.cat([z, response], dim=-1)
        return self.decoder_fc(inputs)

    def forward(self, pattern: Tensor, response: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        (Encoder + Decoder) -> (recon_logits, mu, logvar)

        Args:
            pattern (Tensor): 輸入的真實圖案。
            response (Tensor): 輸入的真實響應。

        Returns:
            Tuple[Tensor, Tensor, Tensor]: (recon_logits, mu, logvar)
        """
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
        noise_scale=0.0,
    ):
        """
        推論
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
        """
        基於原始 Logits 計算。

        Args:
            recon_logits: Decoder 的直接輸出 (未經 Sigmoid)
            target: 真實圖樣 (0 或 1)
        """
        # BCEWithLogitsLoss 內部整合了 Sigmoid，能防止 log(sigmoid(x)) 的溢位問題
        BCE = F.binary_cross_entropy_with_logits(recon_logits, target, reduction="sum")
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + beta * KLD

    def elbo_binarized(self, recon_bin: Tensor, target: Tensor, mu: Tensor, logvar: Tensor, beta: float = 1) -> Tensor:
        """
        基於二值化後的結果計算。
        """
        eps = 1e-6
        recon_safe = torch.clamp(recon_bin, min=eps, max=1.0 - eps)

        # 2. 確保 target 也是浮點數 (雖然通常已經是)
        target = target.float()

        # 3. 計算 BCE
        # 建議改用 mean (平均)，避免因 pattern_size 很大導致 Loss 數值過大
        BCE = F.binary_cross_entropy(recon_safe, target, reduction="mean")
        # 如果您堅持用 sum，請務必加上梯度裁剪 (Clip Grad)
        # BCE = F.binary_cross_entropy(recon_safe, target, reduction='sum')

        # 4. 計算 KLD
        # 這裡您之前已經加了 clamp logvar，應該是安全的
        # 若上方改用 mean，這裡建議也改用 mean 以維持比例
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        total_loss = BCE + beta * KLD

        # 5. [最後防線] 檢查 NaN/Inf
        if torch.isinf(total_loss) or torch.isnan(total_loss):
            print(f"[Warning] Inf/NaN Loss Detected! BCE={BCE.item()}, KLD={KLD.item()}")
            # 回傳帶梯度的 0，避免程式崩潰，讓 Optimizer 跳過這一步
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
        ste_params: dict = {"tau": 1.0, "threshold": 0.0},
    ) -> dict:
        """
        封裝好的訓練迴圈。

        Args:
            data_source: 可以是 Tuple(patterns, responses) 的 Tensors，或是 PyTorch DataLoader。
            optimizer: 優化器 (e.g. Adam)。
            epochs (int): 訓練輪數。
            batch_size (int): 若 data_source 為 Tensor 時的批次大小。
            beta (float): KL Divergence 的權重 (Beta-VAE)。
            use_ste (bool): 是否啟用 STE 二值化優化 (True 使用 elbo_binarized, False 使用 elbo_logits)。
            ste_params (dict): 傳給 binarization 的參數 (僅在 use_ste=True 時有效)。

        Returns:
            dict: 包含 'total_loss', 'bce', 'kld' 的歷史紀錄 list。
        """
        if hasattr(data_source, "__len__") and len(data_source) == 0:
            print("[Warning] Data source is empty (0 items). Skipping training loop.")
            return {"total": [], "bce": [], "kld": []}

        # 1. 確保模型處於訓練模式
        self.train()

        # 2. 準備數據迭代器
        if isinstance(data_source, torch.utils.data.DataLoader):
            dataloader = data_source

        # 其次檢查是否為 Dataset (需封裝 Batch)
        elif isinstance(data_source, torch.utils.data.Dataset):
            dataloader = torch.utils.data.DataLoader(data_source, batch_size=batch_size, shuffle=True)

        # 最後檢查是否為原始 Tensor Tuple/List (需封裝 Dataset + Batch)
        elif isinstance(data_source, (tuple, list)) and len(data_source) == 2:
            patterns, responses = data_source

            # 安全檢查：確保內容物確實是 Tensor
            if not (torch.is_tensor(patterns) and torch.is_tensor(responses)):
                raise TypeError("Data source tuple/list must contain PyTorch Tensors.")

            dataset = torch.utils.data.TensorDataset(patterns, responses)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        else:
            raise TypeError(
                f"Unsupported data_source type: {type(data_source)}. "
                "Expected DataLoader, Dataset, or (Pattern_Tensor, Response_Tensor)."
            )

        history = {"total_loss": [], "bce": [], "kld": []}

        # 3. 訓練迴圈
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_bce = 0.0
            epoch_kld = 0.0
            steps = 0

            for batch_x, batch_c in dataloader:
                batch_x = size_converter(AntennaPattern, batch_x.to(config.device), flatten=True, batch=True)
                batch_c = size_converter(AntennaResponse, batch_c.to(config.device), flatten=True, batch=True)

                # --- Forward ---
                recon_logits, mu, logvar = self.forward(batch_x, batch_c)

                # --- Loss Calculation ---
                if use_ste:  # Binary Optimization (STE)
                    recon_ste = AntennaPattern.binarization(
                        recon_logits,
                        # tau=ste_params.get('tau', 1.0),
                        # threshold=ste_params.get('threshold', 0.0)
                    )
                    loss = self.elbo_binarized(recon_ste, batch_x, mu, logvar, beta)
                else:  # Logits Optimization (Standard)
                    loss = self.elbo_logits(recon_logits, batch_x, mu, logvar, beta)

                # --- Backward ---
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                # --- Record ---
                # 為了記錄方便，我們重新算一下單項 Loss (不含 backward graph)
                with torch.no_grad():
                    epoch_loss += loss.item()
                    # 簡單估算拆解項 (僅供參考)
                    kld_val = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    epoch_kld += beta * kld_val.item()
                    epoch_bce += loss.item() - (beta * kld_val.item())
                    steps += 1

            # 平均 Loss
            if steps > 0:
                history["total_loss"].append(epoch_loss / steps)
                history["bce"].append(epoch_bce / steps)
                history["kld"].append(epoch_kld / steps)

        return history

    def generate(self, response: Tensor, n_samples: int = 1) -> Tensor:
        """
        用於反向設計的生成函數。
        給定一個「目標響應」(條件)，從潛在空間隨機採樣 z, 並使用「解碼器」生成 n_samples 個候選圖案。

        Args:
            response (Tensor): 目標 EM 響應 (1, response_size)。
            n_samples (int): 要生成的候選圖案數量。

        Returns:
            Tensor: 生成的候選圖案 Logits (n_samples, pattern_size)。
        """
        # 從標準常態分佈 N(0, 1) 中隨機採樣 z
        z = torch.randn(n_samples, self.latent_dim).to(config.device)

        # 將 "目標響應" (條件) 複製 n_samples 次，以匹配 z 的批次大小
        response_batch = response.repeat(n_samples, 1)

        # 只使用解碼器生成圖案 logits
        logits = self.decode(z, response_batch)
        return logits
