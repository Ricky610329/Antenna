from abc import ABC, abstractmethod

from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from antenna import *
from antenna.models import *
from antenna.ranger import Ranger
from antenna.utils import *
from antenna.utils.data import DataManager


class SurrogateModel(
    Models[CustomModule, ModelParams, ReturnType, CustomOptimizer, CustomScheduler, LossParams],
    Generic[CustomModule, ModelParams, ReturnType, CustomOptimizer, CustomScheduler, LossParams],
):
    def __init__(
        self,
        model: CustomModule,
        criterion: Callable[CallableParam, Tensor],
        optimizer: CustomOptimizer,
        scheduler: Optional[CustomScheduler] = None,
        *,
        rootdir=None,
    ):
        """
        Global Variable
        ---------------
        ```
        config['HFSS.min_loss'] = ...
        config['HFSS.max_epoch'] = ...
        ```

        Parameters
        ----------
        progress_callback: function
            A callback function that will be called for every frame to notify
            the saving progress. It must have the signature ::

                def func(current_frame: int, total_frames: int) -> Any

            where *current_frame* is the current frame number and
            *total_frames* is the total number of frames to be saved.
            *total_frames* is set to None, if the total number of frames can
            not be determined. Return values may exist but are ignored.

            Example code to write the progress to stdout::

                progress_callback = lambda i, n: print(f'Saving frame {i}/{n}')
        """
        super().__init__(
            name="sm",
            rootdir=rootdir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            load=False,  # 避免呼叫父類別未覆寫的 load
        )

        self.device = config["device"]
        self.pattern_size = AntennaPattern.size
        self.response_size = AntennaResponse.size
        self.size_converter = size_converter

        self.epoch = 0

    def __call__(self, pattern) -> MultiResponses:
        self.epoch += 1
        return MultiResponses(self.model(pattern))

    def train_by_datas(
        self, dataset: DataManager, epochs: int = 100, batch_size: Optional[int] = None, *, verbose: bool = True
    ) -> List[float]:
        """
        Train the model using the provided dataset.

        Args:
            dataset (DataManager): Data set used for training.
            epochs (int): Total number of training cycles.
            batch_size (Optional[int]): Size of each batch.
            verbose (bool): Enable progress bar.

        Returns:
            List[float]: List of average losses per epoch.
        """
        self.requires_grad(True, train=True)
        self.record.reset()

        if dataset is None or len(dataset) <= 0:
            return []
        elif batch_size is None:
            pass
        else:
            batch_size = min(len(dataset), batch_size)

        dataloader = DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=config.device)
        )

        epoch_bar = tqdm(range(epochs), desc="Training...", disable=not verbose, **TQDM_CONFIG)
        for epoch in epoch_bar:
            for n, (patterns, real_responses) in enumerate(cast(tuple[Tensor, Tensor], dataloader)):
                patterns = self.size_converter(AntennaPattern, patterns, flatten=True, batch=True)
                real_responses = self.size_converter(AntennaResponse, real_responses, flatten=False, batch=True)

                inputs: Tensor = patterns.flatten(start_dim=1).to(config.device)
                labels: Tensor = real_responses.to(config.device)

                self.optimizer.zero_grad()
                outputs: Tensor = self.model(inputs)
                loss: Tensor = self.criterion(outputs, labels)

                loss.backward()
                self.step(scheduler_param=loss)

                self.record["loss"] = loss.item()

            avg_epoch_loss = self.record.average("loss")
            self.record.reset("loss", delete=True)
            self.record["epoch_loss"] = avg_epoch_loss

            epoch_bar.set_postfix({"Loss": f"{avg_epoch_loss:.4e}"})

            if self.record.early_stop("epoch_loss", int(epochs / 2)):
                logger.success(f"Early Stopping triggered at epoch {epoch + 1}!")
                break

        self.model.eval()
        return self.record["epoch_loss"]

    def train_one_data(
        self, pattern: Tensor, real_response: Tensor, min_loss=None, max_epoch=None, *, verbose: bool = True
    ):
        """
        The model is trained using a single set of data.

        Args:
            pattern (Tensor): Real antenna pattern
            real_response (Tensor): The real response of the antenna pattern
            min_loss: Minimum loss limit
            max_epoch: Maximum epoch limit
            verbose (bool): Enable progress bar.

        Returns:
            List[float]: List of average losses per epoch.
        """
        self.requires_grad(True, train=True)
        self.record.reset()

        self.record["loss"] = float("inf")
        self.record["epoch"] = 0

        input = tensor(pattern, requires_grad=True)
        label = tensor(real_response, requires_grad=True)

        min_loss = min_loss or config["HFSS.min_loss"]
        max_epoch = max_epoch or config["HFSS.max_epoch"]

        epoch_bar = tqdm(
            total=max_epoch, desc="Training one data", bar_format=TQDM_BAR_SIMPLE, disable=not verbose, **TQDM_CONFIG
        )
        while self.record("loss", 0) > min_loss and self.record("epoch", float("inf")) < max_epoch:
            self.optimizer.zero_grad()

            outputs_result: Tensor = self.model(input)

            loss: Tensor = self.criterion(
                outputs_result.reshape(-1, *AntennaResponse.size()), label.reshape(-1, *AntennaResponse.size())
            )

            loss.backward()
            self.step(scheduler_param=loss)

            self.record["loss"] = loss.item()
            self.record.add("epoch", 1)

            epoch_bar.update()
            epoch_bar.set_postfix({"loss": f"{self.record('loss'):.2f}/{min_loss}"})

        self.model.eval()
        return self.record["loss"]


class HFSSNet(nn.Module):
    def __init__(self, num_pattern_pixel=625, num_response: tuple = (3, 17)):
        super().__init__()
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
            nn.Linear(64, num_response[0] * num_response[1]),
        )
        self.to(config.device)

    def __repr__(self):
        return f"{self.__class__.__name__}(num_pattern_pixel={self.num_pattern_pixel}, num_response={self.num_response}"

    def forward(self, input):
        x = self.fc_patch(input)
        x = x.reshape(self.num_response)
        return x


class SelfAttention(nn.Module):
    """簡化的自注意力層"""

    def __init__(self, in_channels):
        super().__init__()
        # 使用 // 8 可能會導致通道數過少，特別是如果 in_channels 本身不大
        # 改用固定的 attention_channels 或 min(in_channels // 8, 某個固定值)
        attention_channels = max(1, in_channels // 8)  # 確保至少為 1
        self.query_conv = nn.Conv2d(in_channels, attention_channels, kernel_size=1)  #
        self.key_conv = nn.Conv2d(in_channels, attention_channels, kernel_size=1)  #
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)  #
        self.gamma = nn.Parameter(torch.zeros(1))  #
        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        batch_size, C, width, height = x.size()
        # [B, C', W*H] -> [B, W*H, C']
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)  #
        # [B, C', W*H]
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)  #
        # [B, W*H, C'] @ [B, C', W*H] -> [B, W*H, W*H]
        energy = torch.bmm(proj_query, proj_key)  #
        attention = self.softmax(energy)  # 在 W*H 維度上 softmax
        # [B, C, W*H]
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)  #

        # [B, C, W*H] @ [B, W*H, W*H] -> [B, C, W*H] (注意permute)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  #
        out = out.view(batch_size, C, width, height)  #

        out = self.gamma * out + x  # 殘差連接
        return out  #


# --- 2. 包含 Dropout 的 DoubleConv ---
class DoubleConvWithDropout(nn.Module):
    """(convolution => [BN] => ReLU => [Dropout]) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, dropout_prob=0.15):  # 增加 dropout 概率
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),  #
            nn.BatchNorm2d(mid_channels),  #
            nn.ReLU(inplace=True),  # ***修改點：使用 ReLU***
            nn.Dropout(dropout_prob),  # ***修改點：添加 Dropout***
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),  #
            nn.BatchNorm2d(out_channels),  #
            nn.ReLU(inplace=True),  # ***修改點：使用 ReLU***
            nn.Dropout(dropout_prob),  # ***修改點：添加 Dropout***
        )

    def forward(self, x):
        return self.double_conv(x)  #


# --- 3. 整合修改後的 EnhancedHFSSUNet ---
class EnhancedHFSSUNet(nn.Module):
    def __init__(self, base_channels=64, dropout_prob=0.15):
        """
        增強版的 HFSSUNet，包含增加通道數、Dropout 和 Self-Attention。

        Args:
            num_pattern_pixel (int): 輸入 Pattern 的像素總數 (應為平方數)。
            num_response (tuple): 輸出響應的形狀 (e.g., (3, 17))。
            base_channels (int): U-Net 第一層的基礎通道數，控制模型容量。
            dropout_prob (float): 應用於 DoubleConv 層的 Dropout 概率。
        """
        super().__init__()

        # --- 自動獲取大小 ---
        _pattern_size = AntennaPattern.size(flatten=False)  # (H, W)
        _response_size = AntennaResponse.size(flatten=False)  # (C, L) or (H, W)

        # 確保 response_size 是二維的
        if len(_response_size) != 2:
            raise ValueError(f"AntennaResponse.size() 應返回二維形狀，但得到 {_response_size}")

        self.num_response = _response_size
        self.num_pattern_pixel = _pattern_size[0] * _pattern_size[1]
        self.input_dim_h, self.input_dim_w = _pattern_size  # 分別獲取高和寬
        # --------------------

        self.base_channels = base_channels
        self.dropout_prob = dropout_prob

        # 不再需要檢查平方數，因為我們直接用 H 和 W
        # if self.input_dim * self.input_dim != self.num_pattern_pixel:
        #     raise ValueError("num_pattern_pixel 不是一個完美的平方數，無法轉換為 2D 圖像")

        n_channels_in = 1
        n_channels_out = base_channels // 2  # Decoder 最後輸出的通道數, 64 // 2 = 32

        # --- Encoder (通道數增加, 使用 DoubleConvWithDropout) ---
        self.down1 = DoubleConvWithDropout(n_channels_in, base_channels, dropout_prob=dropout_prob)  # 1 -> 64
        self.pool1 = nn.MaxPool2d(2)  #
        self.down2 = DoubleConvWithDropout(base_channels, base_channels * 2, dropout_prob=dropout_prob)  # 64 -> 128
        self.pool2 = nn.MaxPool2d(2)  #
        self.down3 = DoubleConvWithDropout(
            base_channels * 2, base_channels * 4, dropout_prob=dropout_prob
        )  # 128 -> 256
        self.pool3 = nn.MaxPool2d(2)  #

        # --- Bottleneck (通道數增加, 使用 DoubleConvWithDropout) ---
        self.bottleneck = DoubleConvWithDropout(
            base_channels * 4, base_channels * 8, dropout_prob=dropout_prob
        )  # 256 -> 512

        # --- Self-Attention (在 Bottleneck 之後) ---
        self.attention = SelfAttention(base_channels * 8)  # 輸入通道數 = 512

        # --- Decoder (通道數增加, 使用 DoubleConvWithDropout) ---
        self.up1 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)  # 512 -> 256
        self.up_conv1 = DoubleConvWithDropout(
            base_channels * 8, base_channels * 4, dropout_prob=dropout_prob
        )  # Skip:256 + Up:256 -> 256
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)  # 256 -> 128
        self.up_conv2 = DoubleConvWithDropout(
            base_channels * 4, base_channels * 2, dropout_prob=dropout_prob
        )  # Skip:128 + Up:128 -> 128
        self.up3 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)  # 128 -> 64
        self.up_conv3 = DoubleConvWithDropout(
            base_channels * 2, n_channels_out, dropout_prob=dropout_prob
        )  # Skip:64 + Up:64 -> 32

        # --- Head (包含 Dropout, 使用 ReLU) ---
        self.head_pool = nn.AdaptiveAvgPool2d((1, 1))  #
        self.head_fc = nn.Sequential(
            nn.Linear(n_channels_out, 128),  # 輸入通道數 32, 增加中間層大小
            nn.ReLU(inplace=True),  # ***修改點：使用 ReLU***
            nn.Dropout(0.25),  # ***修改點：在 Head 中加入 Dropout, 稍微提高比例***
            nn.Linear(128, self.num_response[0] * self.num_response[1]),  #
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"num_pattern_pixel={self.num_pattern_pixel}, "
            f"num_response={self.num_response}, "
            f"base_channels={self.base_channels}, "
            f"dropout_prob={self.dropout_prob})"
        )  #

    def forward(self, x: Tensor):
        # 0. Reshape Input: (B, num_pixels) -> (B, 1, H, W)
        x = x.unsqueeze(0)
        if x.dim() > 2:
            x = torch.flatten(x, 1)
        # if x.shape[1] != self.num_pattern_pixel:
        #      raise ValueError(f"Input has {x.shape[1]} features, but expected {self.num_pattern_pixel}")
        # 使用初始化時獲取的高和寬
        x_img = x.view(-1, 1, self.input_dim_h, self.input_dim_w)  #

        # 1. Encoder
        x1 = self.down1(x_img)
        x2 = self.pool1(x1)  #
        x3 = self.down2(x2)
        x4 = self.pool2(x3)  #
        x5 = self.down3(x4)
        x6 = self.pool3(x5)  #

        # 2. Bottleneck
        bottle = self.bottleneck(x6)  #

        # 3. Attention
        attn_bottle = self.attention(bottle)

        # 4. Decoder
        u1 = self.up1(attn_bottle)  # <-- 使用 attn_bottle
        u1 = F.interpolate(u1, size=x5.shape[2:], mode="bilinear", align_corners=True)  #
        cat1 = torch.cat([x5, u1], dim=1)  #
        c1 = self.up_conv1(cat1)  #

        u2 = self.up2(c1)  #
        u2 = F.interpolate(u2, size=x3.shape[2:], mode="bilinear", align_corners=True)  #
        cat2 = torch.cat([x3, u2], dim=1)  #
        c2 = self.up_conv2(cat2)  #

        u3 = self.up3(c2)  #
        u3 = F.interpolate(u3, size=x1.shape[2:], mode="bilinear", align_corners=True)  #
        cat3 = torch.cat([x1, u3], dim=1)  #
        c3 = self.up_conv3(cat3)  #

        # 5. Head
        out_pool = self.head_pool(c3)  #
        out_flat = torch.flatten(out_pool, 1)  #
        out_fc = self.head_fc(out_flat)  #

        # 6. Final Reshape
        out = out_fc.view(-1, self.num_response[0], self.num_response[1])  #

        return out


# --- 4. 更新 UNetSM 函數 ---
def UNetSM(
    checkpoint,
    base_channels=64,
    dropout_prob=0.15,
    learning_rate=1e-4,
    scheduler_patience=15,
    weight_decay=1e-4,
    loss_type="L1",
):
    """
    創建一個使用 EnhancedHFSSUNet 的 SurrogateModel 實例，並調整超參數。

    Args:
        checkpoint (str or Path): 儲存/載入模型權重的路徑。
        base_channels (int): U-Net 第一層的基礎通道數。
        dropout_prob (float): 應用於 DoubleConv 層的 Dropout 概率。
        learning_rate (float): 優化器的學習率。
        scheduler_patience (int): ReduceLROnPlateau 排程器的耐心值。
        weight_decay (float): 優化器的權重衰減值。
        loss_type (str): 使用的損失函數類型 ('L1' or 'MSE')。
    """
    # pattern_size 和 response_shape 會在 EnhancedHFSSUNet 內部自動獲取
    model_ge = EnhancedHFSSUNet(base_channels=base_channels, dropout_prob=dropout_prob)

    # ***修改點：選擇損失函數***
    if loss_type == "L1":
        criterion_ge = nn.L1Loss()  # 使用 L1 Loss
    elif loss_type == "MSE":
        criterion_ge = nn.MSELoss()  #
    else:
        raise ValueError("loss_type 必須是 'L1' 或 'MSE'")

    optimizer_ge = Ranger(  # 保持 Ranger 優化器
        params=model_ge.parameters(),
        lr=learning_rate,  # ***修改點：使用較低的學習率***
        weight_decay=weight_decay,  # ***修改點：明確設置權重衰減***
    )
    from antenna.functions import AdaptiveCyclicalScheduler

    scheduler_ge = AdaptiveCyclicalScheduler(  #
        optimizer_ge,
    )
    return SurrogateModel(  #
        model_ge, criterion_ge, optimizer_ge, scheduler_ge, rootdir=checkpoint
    )


def OldSM(checkpoint):
    """
    學長的做法
    """
    model_ge = HFSSNet(  # Pattern -> Response
        AntennaPattern.size(flatten=True), AntennaResponse.size()
    )
    criterion_ge = nn.MSELoss()
    optimizer_ge = Ranger(params=model_ge.parameters(), lr=config["HFSS.lr"])
    scheduler_ge = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_ge, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )
    return SurrogateModel(model_ge, criterion_ge, optimizer_ge, scheduler_ge, rootdir=checkpoint)


# def UNetSM(checkpoint):
#     model_ge = HFSSUNet( # Pattern -> Response
#         AntennaPattern.size(flatten=True), AntennaResponse.size()
#     )
#     criterion_ge = nn.MSELoss()
#     optimizer_ge = Ranger(
#         params=model_ge.parameters(), lr=config['HFSS.lr']
#     )
#     scheduler_ge = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer_ge, mode="min", factor=0.5, patience=10, min_lr=1e-6
#     )
#     return SurrogateModel(
#         model_ge, criterion_ge, optimizer_ge, scheduler_ge, rootdir=checkpoint
#     )
