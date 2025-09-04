from antenna.utils import *
from antenna.models import *
from antenna.ranger import Ranger
from antenna import *

from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from abc import ABC, abstractmethod


#%% Import By Device
FloatTensor = torch.FloatTensor if str(config.device) == 'cpu' else torch.cuda.FloatTensor # type: ignore

class SurrogateModel(ABC):
    def __init__(self, model, criterion, optimizer:Optimizer, scheduler:Optional[LRScheduler]=None, *, progress_callback = lambda i, n: None):
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
        self.FloatTensor = torch.FloatTensor if str(config.device) == 'cpu' else torch.cuda.FloatTensor # type: ignore
        self.epoch = 1
        self.model: nn.Module = model
        self.criterion: nn.Module = criterion
        self.optimizer: Optimizer = optimizer
        self.scheduler: Optional[LRScheduler] = scheduler
        
        self.progress_callback = progress_callback

    def save(self, rootdir):
        # path = Path(rootdir).joinpath(f"sm_{self.epoch}.pth")
        path = Path(rootdir).joinpath(f"sm.pth")
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': None if not self.scheduler else self.scheduler.state_dict(),
        }
        torch.save(checkpoint, path)
        return path

    def load(self, rootdir):
        path = Path(rootdir).joinpath(f"sm.pth")
        checkpoint:Dict = path.load_torch()
        self.epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'] or {})

        return self.epoch

    def __call__(self, pattern) -> MultiResponses:
        self.epoch += 1
        return MultiResponses(self.model(pattern))
    
    def __str__(self):
        _str = "{class_name}(Model={model}, Optimizer={optimizer}, Scheduler={scheduler}, Criterion={criterion})"
        return _str.format(
            class_name = self.__class__.__name__,
            model = self.model.__class__.__name__,
            optimizer = self.optimizer.__class__.__name__,
            scheduler = self.scheduler.__class__.__name__,
            criterion = self.criterion.__class__.__name__
        )
         
    @abstractmethod
    def train(self, pattern):
        pass

class OldSM(SurrogateModel):
    """
    學長的做法
    """
    def __init__(self):
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
        super().__init__(model_ge, criterion_ge, optimizer_ge, scheduler_ge)

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

