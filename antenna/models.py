from antenna.utils import *
from antenna import *

import torch
from torch.autograd.function import (
    Function,
    FunctionCtx ,
    BackwardCFunction,
)
from torch.autograd import Variable
from torch.optim.optimizer import Optimizer

import numpy as np
from math import sqrt

from torch.optim.lr_scheduler import LRScheduler
class Models:
    def __init__(
            self, 
            name:str = "models_{label}", 
            rootdir:Union[str, Path] = ".", 
            model:Optional[nn.Module] = None, 
            optimizer:Optional[Optimizer] = None, 
            scheduler:Optional[LRScheduler] = None, 
            criterion:Optional[nn.Module] = None,
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

        self._rootdir = rootdir
        self.device = device

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.record = Record(self.__class__.__name__, rootdir=self._rootdir, load=load and self.model_file.exists())

        if load: 
            assert self.name, "Please use `Models.change()` first."
            self.load()
    
    def __call__(self, *args, **kwds):
        return self.model(*args, **kwds)

    def __str__(self):
        _str = "{class_name}(Model={model}, Optimizer={optimizer}, Scheduler={scheduler}, Criterion={criterion})"
        return _str.format(
            class_name = self.__class__.__name__,
            model = self.model.__class__.__name__,
            optimizer = self.optimizer.__class__.__name__,
            scheduler = self.scheduler.__class__.__name__,
            criterion = self.criterion.__class__.__name__
        )
    
    @property
    def model_file(self) -> Path:
        """The full path to the model archive."""
        return Path(self._rootdir).joinpath(f"{self.name}.pth")
    
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

    def to(self, *args, **kward):
        """
        Move and/or cast the parameters and buffers.
        """
        self.model = self.model.to(*args, **kward)
        self.device = kward['device'] or args[0]
        return self.model
    
    def load(self):
        checkpoint_loaded:dict = self.model_file.load_torch()
        self.model.load_state_dict(checkpoint_loaded['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint_loaded['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint_loaded['scheduler_state_dict'])

        self.device = checkpoint_loaded['device']

    def save(self) -> Path:
        checkpoint = {
            'model_state_dict': None if not self.model else self.model.state_dict(),
            'optimizer_state_dict': None if not self.optimizer else self.optimizer.state_dict(),
            'scheduler_state_dict': None if not self.scheduler else self.scheduler.state_dict(),
            'device': self.device
        }
        torch.save(checkpoint, self.model_file)
        return self.model_file
    
    def pre_load_model(self, path:Union[str, Path]):
        path = Path(path)
        self.model.load_state_dict(
            path.load_torch()['model_state_dict']
        )
        logger.success(f'Successfully loaded the pre-trained model. ({path})')

    def step(self, optimizer_param=None, scheduler_patam=None):
        self.optimizer.step(optimizer_param)
        if self.scheduler: self.scheduler.step(scheduler_patam)

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



class HFSSNet(nn.Module):

    def __init__(self, num_pattern_pixel = 625, num_response:tuple = (3, 17)):
        super(HFSSNet, self).__init__()
        self.num_response = num_response
        self.num_pattern_pixel = num_pattern_pixel

        self.fc_patch = nn.Sequential(
            nn.Linear(num_pattern_pixel, 2048),
            nn.PReLU(),
            nn.Linear(2048, 1024),
            nn.PReLU(),
            nn.Linear(1024, 512),
            nn.PReLU(),
            nn.Linear(512, 128),
            nn.PReLU(),
            nn.Linear(128, 64),
            nn.PReLU(),
            nn.Linear(64, num_response[0]*num_response[1])
        )
        self.to(config.device)
        
    def __repr__(self):
        return f"{self.__class__.__name__}(num_pattern_pixel={self.num_pattern_pixel}, num_response={self.num_response}"
    
    def forward(self, input):
        x = self.fc_patch(input)
        x = x.reshape(self.num_response)
        return x
    

# %%
from .functions import gumbel_sinkhorn_rectangular
class SPGEN(nn.Module):
    def __init__(self ,pattern_table:Tuple, size=40):
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
        assignment_matrix = gumbel_sinkhorn_rectangular(reshaped_logits, tau=tau, n_iters=n_iters, hard=hard)

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
        patttern_size = AntennaPattern.size(flatten=True)
        self.fc_patch = nn.Sequential(
            nn.Linear(AntennaResponse.size(flatten=True), 1024),
            nn.PReLU(),
            nn.Linear(1024, 1024),
            nn.PReLU(),
            nn.Linear(1024, AntennaPattern.size(flatten=True)),
            BiScaleNorm(),
        )

        self.r = sign_f.apply
        self.to(config.device)

    def forward(self, input):
        x = self.fc_patch(input)
        x = self.r(x) / 2 + 0.5 # type: ignore
        return x
        # return x

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

