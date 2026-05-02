from antenna.utils.utils import Path, Record
from antenna.types import Tensor

class PatchTrainingRecord(Record):
    def __init__(self, name:str):
        result_dir = Path(r"\\140.123.106.219\Temp\碩二_吳維文's\Patch Antenna\Experiment\result")
        super().__init__("temp", rootdir=result_dir.joinpath(name).absolute(), load=True)
        self.rootdir = result_dir.joinpath(name).absolute()
        self.name = name
        self.best_results = self.best(
            mode = min,
            key = "real_loss",
            output_keys = ['epoch', 'patch_pattern_buf', 'patch_result_buf']
        )
    @property
    def best_epoch(self) -> int:
        return self.best_results[0]
    @property
    def best_pattern(self) -> Tensor:
        return self.best_results[1]
    @property
    def best_response(self) -> Tensor:
        return self.best_results[2]

    @property
    def real_loss(self) -> list[float]:
        return self['real_loss']
    
    @property
    def min_loss(self) -> list[float]:
        return self['min_loss']
    @property
    def fake_loss(self) -> list[float]:
        return self['fake_loss']
    @property
    def mutation(self) -> list[float]:
        return self['mutation']
    @property
    def tau(self) -> list[float]:
        return self['tau']
    @property
    def real_loss_average(self) -> list[float]:
        return self['real_loss_average']
    @property
    def patch_result_buf(self) -> list[Tensor]:
        return self['patch_result_buf']
    @property
    def patch_pattern_buf(self) -> list[Tensor]:
        return self['patch_pattern_buf']
    @property
    def time(self) -> list[float]:
        return self['time']
    @property
    def r_feed(self) -> list[float]:
        return self['r_feed']
    
    def pattern(self, epoch:int) -> Tensor:
        return self.find('epoch', epoch, 'patch_pattern_buf')
    
    def response(self, epoch:int) -> Tensor:
        return self.find('epoch', epoch, 'patch_result_buf')