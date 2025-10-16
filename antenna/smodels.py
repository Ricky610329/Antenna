from antenna.utils import *
from antenna.models import *
from antenna.ranger import Ranger
from antenna import *

from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from abc import ABC, abstractmethod
from tqdm import trange


#%% Import By Device
FloatTensor = torch.FloatTensor if str(config.device) == 'cpu' else torch.cuda.FloatTensor # type: ignore

class SurrogateModel(Models[CustomModule, CustomOptimizer, CustomScheduler], ABC):
    def __init__(self, model, criterion, optimizer:Optimizer, scheduler:Optional[LRScheduler]=None, *, progress_callback = lambda i, n: None, rootdir="."):
        """
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
        
        self.progress_callback = progress_callback

        self.epoch = 0

    def __call__(self, pattern) -> MultiResponses:
        self.epoch += 1
        return MultiResponses(self.model(pattern))
         
    @abstractmethod
    def train(self, pattern):
        pass

class OldSM(SurrogateModel[CustomModule, CustomOptimizer, CustomScheduler]):
    """
    學長的做法
    """
    def __init__(self, checkpoint='.'):
        model_ge = HFSSNet( # Pattern -> Response
            AntennaPattern.getAllPixel(), AntennaResponse.size()
        )
        criterion_ge = nn.MSELoss()
        optimizer_ge = Ranger(
            params=model_ge.parameters(), lr=config['HFSS.lr']
        )
        scheduler_ge = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_ge, mode="min", factor=0.5, patience=10, min_lr=1e-6
        )
        super().__init__(model_ge, criterion_ge, optimizer_ge, scheduler_ge, rootdir=checkpoint)

    def train(self, pattern:Tensor, real_response:Tensor):
        self.model.train()
        pilotLoss_2 = []
        self.loss = float('inf')
        epoch_2 = 0
        
        input = tensor(pattern,  requires_grad=True)
        label = tensor(real_response,  requires_grad=True)
        # for epoch in range(num_epochs):
        while self.loss > config['HFSS.min_loss'] and epoch_2 < config['HFSS.max_epoch']:
            self.optimizer.zero_grad()

            outputs_result:Tensor = self.model(input)

            loss_R:Tensor = self.criterion(
                outputs_result.reshape(-1, *AntennaResponse.size()),
                label.reshape(-1, *AntennaResponse.size())
            )

            loss_R.backward()
            self.optimizer.step()
            self.scheduler.step(loss_R)

            pilotLoss_2.append(loss_R.item())
            self.loss = loss_R.item()
            self.progress_callback(epoch_2, 2000)

            epoch_2 = epoch_2 + 1
        self.model.eval()
        return pilotLoss_2
    
    def pre_train(self, dataset, n=100):
        
        self.record.reset()
        self.loss = float('inf')
        epoch_2 = 0

        bar = trange(n, desc='Pre Train ...')
        for i in bar:
            pilotLoss_2 = []
            self.model.train()
                
            for pattern, real_response in dataset:
                input = pattern.flatten().to(config.device)
                label = real_response.to(config.device)
                self.optimizer.zero_grad()
                
                outputs_result:Tensor = self.model(input)

                loss_R:Tensor = self.criterion(
                    outputs_result.reshape(-1, *AntennaResponse.size()),
                    label.reshape(-1, *AntennaResponse.size())
                )
                loss_R.backward()
                self.step(scheduler_patam=loss_R)

                pilotLoss_2.append(loss_R.item())
                self.loss = loss_R.item()

                epoch_2 = epoch_2 + 1
            
            if pilotLoss_2: # 避免 pilotLoss_2 為空時出錯
                avg = sum(pilotLoss_2) / len(pilotLoss_2)
                # logger.info(f'Pretrain...({i+1}/{n}), Loss: {avg}')
                self.record['loss'] = avg
                bar.set_postfix({"Loss":avg})
            if self.record.early_stop('loss'):
                logger.success(f'Early Stop!')
                break

        self.model.eval()
        return self.record['loss']

