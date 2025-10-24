from antenna.utils import *
from antenna.models import *
from antenna.ranger import Ranger
from antenna import *

from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from abc import ABC, abstractmethod
from tqdm import trange
from antenna.utils.data import DataManager

class SurrogateModel(
    Models[CustomModule, ModelParams, ReturnType, CustomOptimizer, CustomScheduler, LossParams],
    Generic[CustomModule, ModelParams, ReturnType, CustomOptimizer, CustomScheduler, LossParams]
):
    def __init__(self, model:CustomModule, criterion:Callable[CallableParam, Tensor], optimizer:CustomOptimizer, scheduler:Optional[CustomScheduler]=None, *, progress_callback = lambda i, n: None, rootdir="."):
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
            name='sm',
            rootdir=rootdir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            load=False # 避免呼叫父類別未覆寫的 load
        )
        
        self.to(device=config['device'])
        self.progress_callback = progress_callback

        self.epoch = 0

    def __call__(self, pattern) -> MultiResponses:
        self.epoch += 1
        return MultiResponses(self.model(pattern))

    def train_by_datas(self, dataset:DataManager, epocks:int):
        self.model.train()
        self.record.reset()

        bar = trange(epocks, desc='Train ...')
        for i in bar:
            self.model.train()
            self.record.reset('loss')
            for pattern, real_response in dataset:
                input = pattern.flatten().to(config.device)
                label = real_response.to(config.device)
                self.optimizer.zero_grad()
                
                outputs_result:Tensor = self.model(input)

                loss:Tensor = self.criterion(
                    outputs_result.reshape(-1, *AntennaResponse.size()),
                    label.reshape(-1, *AntennaResponse.size())
                )
                loss.backward()
                self.step(scheduler_patam=loss)
                self.record['loss'] = loss.item()
            
            avg = self.record.average('loss')
            self.record['epock_loss'] = avg
            bar.set_postfix({"Loss":avg})
            if self.record.early_stop('epock_loss'):
                logger.success(f'Early Stop!')
                break

        self.model.eval()
        return self.record['epock_loss']
    
    def train_one_data(self, pattern:Tensor, real_response:Tensor, min_loss=None, max_epoch=None):
        self.model.train()
        self.record.reset()
        
        self.record['loss'] = float('inf')
        self.record['epoch'] = 0

        input = tensor(pattern,  requires_grad=True)
        label = tensor(real_response,  requires_grad=True)

        min_loss = min_loss or config['HFSS.min_loss']
        max_epoch = max_epoch or config['HFSS.max_epoch']

        while self.record('loss', 0) > min_loss and self.record('epoch', float('inf')) < max_epoch:
            self.optimizer.zero_grad()

            outputs_result:Tensor = self.model(input)

            loss:Tensor = self.criterion(
                outputs_result.reshape(-1, *AntennaResponse.size()),
                label.reshape(-1, *AntennaResponse.size())
            )

            loss.backward()
            self.step(scheduler_patam=loss)

            self.record['loss'] = loss.item()
            self.record.add('epoch', 1)

        self.model.eval()
        return self.record['loss']


def OldSM(checkpoint):
    """
    學長的做法
    """
    model_ge = HFSSNet( # Pattern -> Response
        AntennaPattern.size(flatten=True), AntennaResponse.size()
    )
    criterion_ge = nn.MSELoss()
    optimizer_ge = Ranger(
        params=model_ge.parameters(), lr=config['HFSS.lr']
    )
    scheduler_ge = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_ge, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )
    return SurrogateModel(
        model_ge, criterion_ge, optimizer_ge, scheduler_ge, rootdir=checkpoint
    )

def UNetSM(checkpoint):
    model_ge = HFSSUNet( # Pattern -> Response
        AntennaPattern.size(flatten=True), AntennaResponse.size()
    )
    criterion_ge = nn.MSELoss()
    optimizer_ge = Ranger(
        params=model_ge.parameters(), lr=config['HFSS.lr']
    )
    scheduler_ge = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_ge, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )
    return SurrogateModel(
        model_ge, criterion_ge, optimizer_ge, scheduler_ge, rootdir=checkpoint
    )
