"""CVAE — 條件變分自動編碼器。"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from antenna import AntennaPattern, AntennaResponse, config
from antenna.types import *
from antenna.utils.data import size_converter


class CVAE(nn.Module):
    """
    條件變分自動編碼器 (CVAE)
    """

    def __init__(
        self,
        latent_dim: int,
        pattern_size: Optional[int] = None,
        response_size: Optional[int] = None,
        binary_fn: Callable[..., Tensor] = AntennaPattern.binarization,
    ):
        super().__init__()
        self.pattern_size = pattern_size or AntennaPattern.size(flatten=True)
        self.response_size = response_size or AntennaResponse.size(flatten=True)
        self.latent_dim = latent_dim
        hidden_dim = 256

        self.encoder_fc = nn.Sequential(
            nn.Linear(self.pattern_size + self.response_size, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.fc_binary = binary_fn
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim + self.response_size, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, self.pattern_size),
        )

        self.to(config.device)

    def encode(self, pattern: Tensor_B_N, response: Tensor_B_N) -> Tuple[Tensor, Tensor]:
        inputs = torch.cat([pattern, response], dim=-1)
        h = self.encoder_fc(inputs)
        return self.fc_mu(h), torch.clamp(self.fc_logvar(h), min=-10.0, max=10.0)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor, response: Tensor_B_N) -> Tensor:
        inputs = torch.cat([z, response], dim=-1)
        return self.decoder_fc(inputs)

    def forward(self, pattern: Tensor, response: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        pattern = size_converter(AntennaPattern, pattern, flatten=True, batch=True)
        response = size_converter(AntennaResponse, response, flatten=True, batch=True)
        mu, logvar = self.encode(pattern, response)
        z = self.reparameterize(mu, logvar)
        recon_pattern = self.decode(z, response)
        return recon_pattern, mu, logvar

    def inference(
        self,
        target_response: Tensor,
        z: Optional[Tensor] = None,
        best: Optional[tuple[Tensor, Tensor]] = None,
        noise_scale=0.0,
    ):
        self.eval()
        target_response = size_converter(AntennaResponse, target_response, flatten=True, batch=True)
        if best is not None:
            best_pattern, best_response = best
            with torch.no_grad():
                best_pattern = size_converter(AntennaPattern, best_pattern, flatten=True, batch=True)
                best_response = size_converter(AntennaResponse, best_response, flatten=True, batch=True)
                mu, _ = self.encode(best_pattern, best_response)
                z = mu + torch.randn_like(mu) * noise_scale
        elif z is None:
            z = torch.randn(target_response.size(0), self.fc_mu.out_features).to(target_response.device)

        return AntennaPattern.binarization(self.decode(z, target_response))

    def elbo_logits(self, recon_logits: Tensor, target: Tensor, mu: Tensor, logvar: Tensor, beta: float = 1) -> Tensor:
        BCE = F.binary_cross_entropy_with_logits(recon_logits, target, reduction="sum")
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + beta * KLD

    def elbo_binarized(self, recon_bin: Tensor, target: Tensor, mu: Tensor, logvar: Tensor, beta: float = 1) -> Tensor:
        eps = 1e-6
        recon_safe = torch.clamp(recon_bin, min=eps, max=1.0 - eps)
        target = target.float()
        BCE = F.binary_cross_entropy(recon_safe, target, reduction="mean")
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = BCE + beta * KLD
        if torch.isinf(total_loss) or torch.isnan(total_loss):
            print(f"[Warning] Inf/NaN Loss Detected! BCE={BCE.item()}, KLD={KLD.item()}")
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
        if hasattr(data_source, "__len__") and len(data_source) == 0:
            print("[Warning] Data source is empty (0 items). Skipping training loop.")
            return {"total": [], "bce": [], "kld": []}

        self.train()

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

        history = {"total_loss": [], "bce": [], "kld": []}

        for epoch in range(epochs):
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
        z = torch.randn(n_samples, self.latent_dim).to(config.device)
        response_batch = response.repeat(n_samples, 1)
        logits = self.decode(z, response_batch)
        return logits
