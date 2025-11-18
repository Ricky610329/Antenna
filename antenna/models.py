from antenna.utils import *
from antenna.types import *
from antenna import *

import torch
from torch.autograd.function import (
    Function,
    FunctionCtx ,
    BackwardCFunction,
)
from torch.autograd import Variable


import numpy as np
from math import sqrt


from functools import partial


class Models(Generic[CustomModule, ModelParams, ReturnType, CustomOptimizer, CustomScheduler, LossParams]):
    def __init__(
            self, 
            name:str = "models_{label}", 
            rootdir:Optional[Union[str, Path]] = None, 
            model:Optional[ CustomModule | CallableModule[ModelParams, ReturnType]] = None, 
            optimizer:Optional[CustomOptimizer] = None, 
            scheduler:Optional[CustomScheduler] = None, 
            criterion:Optional[Callable[LossParams, Tensor]] = None,
            *, 
            load:bool = False,
            device = config.device
        ):
        if "{label}" in name:
            self._name = name
            self.name = None
        else:
            self._name = None
            self.name = name

        self._rootdir = rootdir or config.checkpoint_save_path

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.record = Record(self.__class__.__name__, rootdir=self._rootdir, load=load and self.model_file.exists())

        self.device = device
        if load: self.load()
    
    def __call__(self, *args:ModelParams.args, **kwargs:ModelParams.kwargs) -> ReturnType:
        return  self.model(*args, **kwargs)

    def __str__(self):
        _str = "{class_name}(Model={model}, Optimizer={optimizer}, Scheduler={scheduler}, Criterion={criterion})"
        return _str.format(
            class_name = self.__class__.__name__,
            model = self.model.__class__.__name__,
            optimizer = self.optimizer.__class__.__name__,
            scheduler = self.scheduler.__class__.__name__,
            criterion = self.criterion.__name__ if isinstance(self.criterion, FunctionType) else self.criterion.__class__.__name__
        )
    
    @property
    def model_file(self) -> Path:
        """The full path to the model archive."""
        assert self.name, "Please use `Models.change()` first."
        return Path(self._rootdir).joinpath(f"{self.name}.pth")

    @property
    def device(self):
        """Model parameters of the device."""
        return next(self.model.parameters()).device
    
    @device.setter
    def device(self, device):
        self.model.to(device = device)
    
    @property
    def FloatTensor(self):
        return torch.FloatTensor if str(self.device) == 'cpu' else torch.cuda.FloatTensor # type: ignore
    
    def change(self, label:str, *, load:bool=False, save:bool=False):
        """
        Change model label.

        :param label: models label. You can enter `{label}` in name.
        :param load: Load the changed models.
        :param save: Save the models before the change.
        """
        if save:
            self.save()
        if self._name:
            self.name = self._name.format(label=label)
        if load:
            self.load()
        return self.name
    
    def load(self, force:bool = False):
        checkpoint_loaded = self.checkpoint(load=True)
        if checkpoint_loaded['title'] == self.__str__() or force:
            self.device = checkpoint_loaded['device']

            self.model.load_state_dict(checkpoint_loaded['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint_loaded['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint_loaded['scheduler_state_dict'])
            self.record.load_state_dict(checkpoint_loaded['record_state_dict'])
            
        else:
            raise RuntimeError(f"Please use the correct model file.\nFile: {checkpoint_loaded['title']}\nCurrent: {self.__str__()}")

    def save(self) -> Path:
        temp_file = self.model_file.with_suffix(self.model_file.suffix + '.tmp')
        try:
            torch.save(self.checkpoint(load=False), temp_file)
            temp_file.replace(self.model_file)
        except Exception as e:
            if temp_file.exists():
                temp_file.unlink()
            raise
        return self.model_file
    
    def pre_load_model(self, path:Union[str, Path]):
        path = Path(path)
        checkpoint_loaded:Checkpoint = path.load_torch()
        self.model.load_state_dict(checkpoint_loaded['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint_loaded['optimizer_state_dict'])
        for name, param in self.model.state_dict().items():
            if not torch.all(torch.isfinite(param)):
                raise RuntimeError(f"!!! 在參數 '{name}' 中發現無效值 (NaN 或 inf) !!!")
        logger.success(f'Successfully loaded the pre-trained model. ({path})')

    def step(self, optimizer_param=None, scheduler_param=None):
        self.optimizer.step(optimizer_param)
        if self.scheduler: self.scheduler.step(scheduler_param)
    
    def checkpoint(self, load:bool = False) -> Checkpoint:
        if load:
            checkpoint:dict = self.model_file.load_torch()
        else:
            checkpoint = {
                "title": self.__str__(),
                'model_state_dict': None if not self.model else self.model.state_dict(),
                'optimizer_state_dict': None if not self.optimizer else self.optimizer.state_dict(),
                'scheduler_state_dict': None if not self.scheduler else self.scheduler.state_dict(),
                'device': self.device,
                'record_state_dict': self.record.state_dict()
            }
        return checkpoint

    def requires_grad(self, mode:bool = True, train:Optional[bool] = None):
        for param in self.model.parameters():
            param.requires_grad = mode 

        match train:
            case True:
                self.model.train()
            case False:
                self.model.eval()
            case _:
                pass

        return next(self.model.parameters()).requires_grad
    
class BiScaleNorm(nn.Module):
    def __init__(self):
        super(BiScaleNorm, self).__init__()

    def forward(self, input_vector):
        # 大於 0 的值的正規化
        max_val = torch.max(input_vector)
        positive_normalized = torch.where(input_vector > 0, input_vector / max_val, torch.tensor(0.0, device=input_vector.device))

        # 小於 0 的值的正規化
        min_val = torch.min(input_vector)
        negative_normalized = torch.where(input_vector < 0, input_vector / torch.abs(min_val), torch.tensor(0.0, device=input_vector.device))

        # 合併正規化結果
        normalized_vector = positive_normalized + negative_normalized
        return normalized_vector

class sign_f(Function):
    """
    sign function
    """
    @staticmethod
    def forward(ctx:BackwardCFunction, inputs:Tensor):
        output = inputs.new(inputs.size())
        output[inputs >= 0.] = 1
        output[inputs < 0.] = -1
        ctx.save_for_backward(inputs)
        return output

    @staticmethod
    def backward(ctx:BackwardCFunction, grad_output:Tensor):
        input_, = ctx.saved_tensors
        grad_output[input_>1.] = 0
        grad_output[input_<-1.] = 0
        return grad_output

class _GumbelSigmoid(Function):
    @staticmethod
    def forward(ctx:FunctionCtx, logits, tau_tensor, eps=1e-10):
        """
        Gumbel-Sigmoid采樣方法
        logits: 輸入的logits（可以是實數）
        tau: 溫度，控制離散度
        eps: 防止除以0的小常數
        """
        U = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(U + eps) + eps)
        y = torch.sigmoid((logits + gumbel_noise) / tau_tensor)

        # 保存為 backward 方法提供所需的變數
        ctx.save_for_backward(logits, y, gumbel_noise)
        ctx.tau = tau_tensor  # 保存 tau 以便在 backward 中使用
        return y

    @staticmethod
    def backward(ctx:BackwardCFunction, grad_output):
        # 讀取 forward 傳遞的變數
        logits, y, gumbel_noise = ctx.saved_tensors
        tau = ctx.tau  # 從 ctx 中讀取 tau

        # 計算 gradient
        sigmoid_grad = y * (1 - y)  # Sigmoid 梯度
        grad_input = grad_output * sigmoid_grad / tau  # 給 logits 的梯度
        
        # 計算 tau 的梯度
        # grad_tau = (grad_output * sigmoid_grad * (logits - y)).sum() / tau**2  # 給 tau 的梯度
        grad_tau = (grad_output * sigmoid_grad * (logits + gumbel_noise)).sum() / tau**2  # 給 tau 的梯度

        return grad_input, grad_tau, None

class GumbelSigmoid(Function):
    @staticmethod
    def forward(ctx:Function, logits, tau, eps=1e-20):
        # tau = max(0.1, ctx.tau - 0.001 * ctx.tau) if hasattr(ctx, 'tau') else tau
        U = torch.rand_like(logits)
        scale = 0.1  # 降低到 0.1
        gumbel_noise = -torch.log(-torch.log(U + eps) + eps) * scale
        y = torch.sigmoid((logits + gumbel_noise) / tau)
        
        ctx.save_for_backward(logits, y, gumbel_noise, tau)

        return y

    @staticmethod
    def backward(ctx:Function, grad_output):
        logits, y, gumbel_noise, tau = ctx.saved_tensors
        
        ###* Sigmoid 函數的梯度 ###
        sigmoid_grad = y * (1 - y)

        ###* logits 的梯度 ###
        grad_input = grad_output * sigmoid_grad / tau

        ###* tau 的梯度 ###
        grad_tau = -grad_output * sigmoid_grad * (logits + gumbel_noise) / (tau ** 2)
        grad_tau = grad_tau.sum()  # 總和作為標量梯度
        return grad_input, grad_tau, None
    
class BinarizeSTE(Function):
    @staticmethod
    def forward(ctx:FunctionCtx, input:Tensor):
        mask = (input >= 0.5).float()
        ctx.save_for_backward(mask)
        return mask

    @staticmethod
    def backward(ctx:BackwardCFunction, grad_output):
        mask, = ctx.saved_tensors
        return grad_output * mask  # 只保留 mask 區域的梯度

# %%
from .functions import gumbel_sinkhorn_rectangular
class SPGEN(nn.Module, Generic[CallableParam]):
    def __init__(self ,pattern_table:Tuple, size=40, gumbel_fn:Callable[CallableParam, Tensor]=gumbel_sinkhorn_rectangular, **gumbel_fn_kwargs):
        super(SPGEN,self).__init__()

        self.pattern_table = pattern_table
        self.pattern_table_tensor = self._to_tensor()
        self.num_patterns = len(pattern_table) # [Channels] How many small patterns.
        self.grid_size = size // self.patern_size # [big_h, big_w] Composition of small patterns (|===|---|---|---|)
        self.logits = nn.Parameter( 
            #? [batch, big_h, big_w, Channels]
            torch.randn(1, self.grid_size, self.grid_size, self.num_patterns),
            requires_grad=True
        )
        self.gumbel_fn:Callable[CallableParam, Tensor] =  partial(gumbel_fn,  **gumbel_fn_kwargs)

    def __str__(self):
        return f"SPGEN(total={self.patern_size*self.grid_size}(small[{self.patern_size}]xbig[{self.grid_size}))"
    
    def _to_tensor(self):
        """
        >>> torch.Size([Channels, small_h * small_w])
        """
        _reselt = []
        for pattern in self.pattern_table:
            self.patern_size: int = len(pattern)
            _reselt.append(np.array(pattern, dtype=np.int16).reshape(-1))

        return torch.tensor(
            np.stack(_reselt), 
            dtype = torch.float32
        )

    def forward(self, tau: float = 1.0, n_iters: int = 20, hard: bool = True):

        # 原始 logits 形狀: [1, grid_h, grid_w, num_patterns]
        batch_size, grid_h, grid_w, num_patterns = self.logits.shape
        
        # 1. 重塑 logits 以符合 gumbel_sinkhorn_rectangular 的輸入
        # 將 grid_h 和 grid_w 維度合併為一個「位置」維度
        # 新形狀: [1, grid_h * grid_w, num_patterns]
        num_positions = grid_h * grid_w
        reshaped_logits = self.logits.view(batch_size, num_positions, num_patterns)

        # 2. 使用新的 Gumbel-Sinkhorn 函式
        # 輸出 assignment_matrix 形狀: [1, grid_h * grid_w, num_patterns]
        # 注意：訓練時 hard 應為 False，推斷時可設為 True
        assignment_matrix = self.gumbel_fn(reshaped_logits, tau=tau, hard=hard)

        # pattern_table_tensor: [num_patterns, small_h * small_w]
        # 3. 執行矩陣乘法來選擇圖案
        # torch.matmul: [1, K, M] @ [M, S*S] -> [1, K, S*S]
        # K = num_positions, M = num_patterns, S = patern_size
        selected_patterns = torch.matmul(assignment_matrix, self.pattern_table_tensor) # [1, H*W, small_h*small_w]
        
        # 4. 將結果重塑回最終的圖像形狀
        # selected_patterns 現在的空間維度是攤平的，需要重新構建
        soft_output = selected_patterns.view(
            batch_size, self.grid_size, self.grid_size,  # batch, grid_h, grid_w
            self.patern_size, self.patern_size           # 小圖案大小 (small_h, small_w)
        ).permute(
            0, 1, 3, 2, 4  # (batch, grid_h, small_h, grid_w, small_w)
        ).reshape(
            batch_size,
            self.grid_size * self.patern_size,
            self.grid_size * self.patern_size
        )

        self.output_image = soft_output
        return self.output_image

    def save(self, nrowcol:tuple, result_path, pattern_dict:dict=None):
        with Figure("SPGEN Small Pattern", nrowcol, save=True, rootdir=result_path, size=(18, 12), default_axes_title_size=10, default_tick_size=6) as fig:
            if pattern_dict:
                for name, pattern in pattern_dict.items():
                    ax:Axes = fig.index(-1)
                    ax.axis('off') 
                    ax.set_title(f"{name}")
                    ax.imshow(pattern, cmap='viridis')
            else:
                for n in range(len(self)):
                    ax:Axes = fig.index(-1)
                    ax.axis('off') 
                    ax.set_title(f"Small Pattern {n+1}")
                    ax.imshow(self[n], cmap='viridis')

    def __getitem__(self, idx):
        return self.pattern_table[idx]
    
    def __len__(self):
        return len(self.pattern_table)
    

class GumbelSigmoidGEN(nn.Module):
    """
    Generator Model
    """
    def __init__(self):
        super(GumbelSigmoidGEN,self).__init__()
        pattern_size = AntennaPattern.size(flatten=True)
        self.fc_patch = nn.Sequential(
            nn.Linear(AntennaResponse.size(flatten=True), pattern_size),
            nn.PReLU(),
            nn.Linear(pattern_size, pattern_size*2),
            nn.PReLU(),
            nn.Linear(pattern_size*2, pattern_size),
            nn.PReLU(),
            nn.Linear(pattern_size, pattern_size),
            BiScaleNorm(),
        )
        self.tau = nn.Parameter(torch.tensor(5.0, requires_grad=True))
        self.tau_history = [] 

        for m in self.fc_patch:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 1.0)
            if isinstance(m, nn.PReLU):
                m.weight.data.fill_(0.25)

        self.to(config.device)

    def forward(self, input, *, is_trainig:bool = True):
        """
        輸出 Gumbel-Sigmoid 處理過的結果
        """
        self.logits  = torch.clamp( # 防止梯度爆炸
            self.fc_patch(input), min=-5.0, max=5.0
        )
        # # 在訓練階段使用 Gumbel-Sigmoid 來保持梯度
        # if is_trainig:
        #     x = GumbelSigmoid.apply(x, tau)  # 訓練階段使用 Gumbel-Sigmoid 進行輸出
        # else:
        #     x = (x >= 0.5).float()  # 推論階段，硬性 binarize

        x = GumbelSigmoid.apply(self.logits, self.tau)  # 訓練階段使用 Gumbel-Sigmoid 進行輸出
        # self.anneal_tau()
        self.tau_history.append(self.tau.detach().cpu().item())


        # x = BinarizeSTE.apply(x)

        return x
    
    def anneal_tau(self, rate=0.995, min_tau=0.1):
        """
        Annealing (退火)
        
        在訓練初期，較大的 tau 值會使得輸出更為平滑，有利於模型探索不同的解空間。

        在訓練後期，較小的 tau 值會使輸出更接近離散的 0 和 1，從而幫助模型收斂到一個確定的離散解。
        """
        # self.tau = max(min_tau, self.tau * rate)
        self.tau = torch.clamp(self.tau, min=0.1)
        self.tau_recoed.append(self.tau.detach().cpu())

    def binarize(self, threshold = 0.5):
        binarized_output = (torch.sigmoid(self.logits) > threshold).float()
        return  AntennaPattern(binarized_output)
        return  AntennaPattern((self.x >= threshold).float())
        
class OldGEN(nn.Module):
    """
    Generator Model
    """
    def __init__(self):
        super(OldGEN,self).__init__()
        self.fc_patch = nn.Sequential( # Can use BiScaleNorm or nn.PReLU, except the last layer.
            nn.Linear(AntennaResponse.size(flatten=True), 1024),
            nn.PReLU(),
            nn.Linear(1024, 1024),
            nn.PReLU(),
            nn.Linear(1024, AntennaPattern.size(flatten=True)),
            BiScaleNorm(),
        )

        self.r = sign_f.apply
        self.to(config.device)
    
    def forward(self, input) -> Tensor:
        x = self.fc_patch(input)
        x = self.r(x) / 2 + 0.5 # type: ignore
        return x

class SigmoidGEN(nn.Module):
    """
    Generator Model
    """
    def __init__(self):
        super(SigmoidGEN,self).__init__()
        self.fc_patch = nn.Sequential( # Can use BiScaleNorm or nn.PReLU, except the last layer.
            nn.Linear(AntennaResponse.size(flatten=True), 1024),
            nn.PReLU(),
            nn.Linear(1024, 1024),
            nn.PReLU(),
            nn.Linear(1024, AntennaPattern.size(flatten=True)),
            BiScaleNorm(),
        )
        self.to(config.device)
    
    def forward(self, input, tau:Optional[float] = None) -> Tensor:
        x = self.fc_patch(input)
        return AntennaPattern.binarization(x, tau)

class GradientEstimator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.PReLU(),
            nn.Linear(2048, 1024),
            nn.PReLU(),
            nn.Linear(1024, 512),
            nn.PReLU(),
            nn.Linear(512, 512),
            nn.PReLU(),
            nn.Linear(512, output_dim)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Flatten()
            
        )
        self.to(config.device)

    def forward(self, A:Tensor):
        A = A.unsqueeze(0) #? [batch, W, H]
        # print(A.shape)
        output = self.conv(A)
        output = self.net(output)
        return AntennaResponse(output)

class CVAE(nn.Module):
    """
    條件變分自動編碼器 (CVAE)
    """
    def __init__(self, latent_dim: int, pattern_size:Optional[int] = None, response_size:Optional[int] = None):
        """
        初始化 CVAE 模型。

        Args:
            pattern_size (int): 天線圖案展平後的大小
            response_size (int): EM 響應展平後的大小
            latent_dim (int): 潛在空間 (z) 的維度。
        """
        super(CVAE, self).__init__()
        self.pattern_size = pattern_size or AntennaPattern.size(flatten=True)
        self.response_size = response_size or AntennaResponse.size(flatten=True)
        self.latent_dim = latent_dim

        # --- 1. Encoder (Recognition Network) ---
        # [cite_start]論文: "encodes the RIS pattern and EM responses together" [cite: 12]
        self.encoder_input_dim = self.pattern_size + self.response_size
        self.encoder_fc1 = nn.Linear(self.encoder_input_dim, 512)
        self.encoder_fc2 = nn.Linear(512, 256)
        # 輸出潛在空間的參數
        self.fc_mu = nn.Linear(256, self.latent_dim) # 平均 (mu)
        self.fc_logvar = nn.Linear(256, self.latent_dim) # log 變異數 (logvar)

        # --- 2. Decoder (Reconstruction Network) ---
        self.decoder_input_dim = self.latent_dim + self.response_size
        self.decoder_fc1 = nn.Linear(self.decoder_input_dim, 256)
        self.decoder_fc2 = nn.Linear(256, 512)
        # 輸出 logits (用於 Gumbel-Sigmoid 或 BCEWithLogitsLoss)
        self.decoder_fc3 = nn.Linear(512, self.pattern_size) 
        
        self.to(config.device)
        logger.info(f"CVAE Model Initialized: pattern_size={pattern_size}, response_size={response_size}, latent_dim={latent_dim}")

    def encode(self, pattern: Tensor, response: Tensor) -> Tuple[Tensor, Tensor]:
        """
        (pattern, response) -> (mu, logvar)

        Args:
            pattern (Tensor): 批次的二進制天線圖案 (B, pattern_size)。
            response (Tensor): 批次的對應 EM 響應 (B, response_size)。

        Returns:
            Tuple[Tensor, Tensor]: 潛在空間的 (mu, logvar)。
        """
        inputs = torch.cat([pattern, response], dim=-1)
        h1 = F.relu(self.encoder_fc1(inputs))
        h2 = F.relu(self.encoder_fc2(h1))
        return self.fc_mu(h2), self.fc_logvar(h2)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization Trick: (mu, logvar) -> z

        從 (mu, logvar) 分佈中隨機採樣一個 z 向量，同時保持梯度可回傳。

        Args:
            mu (Tensor): 潛在空間的平均。
            logvar (Tensor): 潛在空間的 log 變異數。

        Returns:
            Tensor: 採樣出的潛在向量 z。
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) # 從標準常態分佈中採樣噪聲
        return mu + eps * std

    def decode(self, z: Tensor, response: Tensor) -> Tensor:
        """
        (z, Response) -> Logits

        Args:
            z (Tensor): 批次的潛在向量 (B, latent_dim)。
            response (Tensor): 批次的目標 EM 響應 (B, response_size)。

        Returns:
            Tensor: 重建圖案的 Logits (B, pattern_size)。
        """
        inputs = torch.cat([z, response], dim=-1)
        h1 = F.relu(self.decoder_fc1(inputs))
        h2 = F.relu(self.decoder_fc2(h1))
        # 輸出 logits (原始分數)，用於後續的 Gumbel-Sigmoid 或 BCEWithLogitsLoss
        return self.decoder_fc3(h2) 

    def forward(self, pattern: Tensor, response: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        CVAE 的標準前向傳播，用於訓練 CVAE 本身 (Encoder + Decoder)。

        Args:
            pattern (Tensor): 輸入的真實圖案。
            response (Tensor): 輸入的真實響應。

        Returns:
            Tuple[Tensor, Tensor, Tensor]: (recon_logits, mu, logvar)
        """
        mu, logvar = self.encode(pattern, response)
        z = self.reparameterize(mu, logvar)
        recon_logits = self.decode(z, response)
        return recon_logits, mu, logvar

    def loss_function(self, recon_logits: Tensor, pattern: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        計算 CVAE 的總損失 (ELBO Loss)。
        Loss = 重建損失 (BCE) + KL 散度 (KLD)。

        Args:
            recon_logits (Tensor): 解碼器重建的圖案 Logits。
            pattern (Tensor): 原始的真實圖案。
            mu (Tensor): 編碼器輸出的 mu。
            logvar (Tensor): 編碼器輸出的 logvar。

        Returns:
            Tensor: CVAE 的總損失。
        """
        # 1. 重建損失 (Reconstruction Loss)
        #    使用 BCEWithLogitsLoss 更穩定，它等同於 Sigmoid + BCELoss
        BCE = F.binary_cross_entropy_with_logits(recon_logits, pattern, reduction='sum')
        
        # 2. KL 散度 (Kullback-Leibler Divergence)
        #    衡量潛在空間分佈 N(mu, var) 與標準常態分佈 N(0, 1) 的差異
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return BCE + KLD

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

import torch
import torch.nn as nn
from torch.functional import F
from typing import Optional, Tuple, List

# 匯入您專案所需的模組
from antenna import AntennaPattern, AntennaResponse, MultiResponses, config
from antenna.functions import mirror

class MirrorCVAE(nn.Module, Generic[CustomSModel]):
    """
    結合了 CVAE 解碼器 (生成器) 與鏡像/評估/選擇機制的模組。

    forward() 方法會執行以下操作：
    1. 根據輸入的條件 (c) 和一個隨機採樣的潛在向量 (z) 生成一個 "基礎 pattern"。
    2. 使用 mirror() 函數 [cite: 331-332] 產生多個鏡像版本的 pattern 。
    3. 使用傳入的 surrogate model (smodel) 評估所有鏡像 pattern 。
    4. 找出 "最佳" (smodel 損失最低) 的 pattern 。
    5. 回傳這個最佳的 pattern 及其對應的 smodel 損失，兩者都帶有梯度，
       可直接用於反向傳播 。
    """
    def __init__(self,
                 latent_dim: int,
                 smodel: CustomSModel, # 您預先訓練好的 surrogate model
                 lower_pattern: AntennaPattern = None # 靜態的 'lower' pattern
                ):
        """
        初始化 MirrorCVAE 生成器。

        Args:
            latent_dim (int): CVAE 的潛在向量維度 (例如: 128)。
            smodel (nn.Module): 一個預先訓練好的代理模型 (Surrogate Model)。
                             這個模型將被用來 "評估" 鏡像 pattern 的好壞。
            lower_pattern (AntennaPattern): 要添加到每個 pattern 上的靜態 'lower' 部分。
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        condition_dim = AntennaResponse.size(flatten=True)
        pattern_dim = AntennaPattern.size(flatten=True)
        
        # --- 儲存外部模組 ---
        if not callable(smodel):
            raise TypeError("smodel 必須是一個可呼叫的 nn.Module")
        
        # 儲存 smodel，並凍結其參數 (如果 smodel 在別處訓練)
        self.smodel = smodel
        self.smodel.requires_grad(False)

        self.lower = lower_pattern
        
        # --- CVAE 解碼器 (生成器) ---
        # !!
        # !! 請將這個 self.decoder 替換為您自己的 CVAE 解碼器架構
        # !!
        # 這裡使用一個基於 OldGEN [cite: 405-407] 的範例架構
        self.decoder = nn.Sequential(
            # 輸入維度 = 潛在向量 + 條件
            nn.Linear(latent_dim + condition_dim, 1024),
            nn.PReLU(), 
            nn.Linear(1024, 1024),
            nn.PReLU(), 
            nn.Linear(1024, pattern_dim),
            nn.Sigmoid() # 確保輸出在 0 到 1 之間
        )
            

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """ 
        解碼器：將潛在向量 z 和條件 c 轉換為基礎 pattern 張量。
        """
        inputs = torch.cat([z, c], dim=0)
        base_pattern_tensor = self.decoder(inputs)
        return base_pattern_tensor

    def forward(self, 
                c: torch.Tensor, 
                z: Optional[torch.Tensor] = None
               ) -> List[ResultType]:
        """
        執行 "產生-鏡像-評估-選擇" 的前向傳播。

        Args:
            c (torch.Tensor): 
條件向量 (例如 AntennaResponse.target.concat())。
            z (Optional[torch.Tensor], optional): 一個固定的潛在向量 (用於可重現的生成)。
                                                  如果為 None，將隨機採樣。

        Returns:
            Tuple[AntennaPattern, torch.Tensor]:
            - best_pattern (AntennaPattern): 
評估後最佳的鏡像 pattern 物件。
                                           梯度會連結到這個物件。
            - best_fake_loss (torch.Tensor): 
來自 smodel 對 best_pattern 的評估損失。
                                            這是您應該呼叫 .backward() 的損失張量。
        """
        if z is None:
            # 如果未提供 z，則隨機採樣一個
            z = torch.randn(self.latent_dim, device=c.device)
        
        # 1. 解碼 (生成) 基礎 pattern 張量
        # 這個張量帶有來自解碼器的梯度
        base_pattern_tensor = self.decode(z, c)
        base_pattern_obj = AntennaPattern(base_pattern_tensor)
        

        # 2. 產生鏡像 patterns 
        # 這個鏡像操作 (cat, flip) 是可微分的 [cite: 331-343]
        mirrored_tensors = mirror(base_pattern_obj.merge(), mode='-|*')
        mirrored_patterns = [
            AntennaPattern(t) + self.lower
            if self.lower else AntennaPattern(t)
            for t in mirrored_tensors
        ]
        # 3. 評估所有鏡像 patterns 
        all_losses: List[torch.Tensor] = []
        all_results: List[MultiResponses] = []
        self.smodel.model.eval() # 確保 smodel 處於評估模式
        
        with torch.enable_grad(): # 確保在 smodel 內部計算時保留梯度
            for pattern in mirrored_patterns:
                # 讓梯度流經 smodel
                
                output_result = self.smodel(pattern.series)
                fake_loss = output_result.criterion()
                all_losses.append(fake_loss)
                all_results.append(output_result)

        # 4. 找出最佳損失的索引 
        # 我們在 .detach() 後的張量上執行 argmin，
        # 這樣 "選擇" 操作本身 (argmin) 就不會接收梯度。
        losses_tensor_detached = torch.stack([l.detach() for l in all_losses])
        best_loss_index = torch.argmin(losses_tensor_detached)
        
        # (可選) 填充您在 `train_single_mirror.py`  中使用的 results 列表，用於繪圖
        results: List[ResultType] = []
        for i, pattern in enumerate(mirrored_patterns):
            result_dict: ResultType = {
                "pattern": pattern,
                "real_result": None,
                "fake_result": all_results[i],
                "real_loss": None,
                "fake_loss": all_losses[i],
                "sm_loss": [], # sm_loss 在 HFSS 模擬後才更新
                "time": 0,     # time 在 HFSS 模擬後才更新
                "sort_key": all_losses[i].item(), 
                "is_best": (i == best_loss_index.item())
            }
            results.append(result_dict)

            results_sorted: List[ResultType] = sorted(results, key=lambda x: x["sort_key"])

        return results_sorted