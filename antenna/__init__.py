"""
It includes a microstrip patch antenna and a reconfigurable intelligent surface (RIS).

Example::

    from antenna import *
    config.device = "cuda:0"

    from antenna.utils import *
    from antenna.models import ...
    from antenna.smodels import ...

    #* Select according to actual application.
    from antenna.ris import ...
    from antenna.patch import ...

    #* Basic Config
    connect_network_drive("T:", r"\\140.123.106.219\temp", "user", "ailab120")
    RESULT_PATH, is_connect_run = get_result_path('[...][{device}] ...', rootdir=ROOTDIR)
    
    #* Set Antemma Pattern
    AntennaPattern.setDefaultCoordinate((0, n, 0, n))
    PATTERN_SIZE = AntennaPattern.size(flatten=True)
    simulator = ...
    AntennaPattern.register_simulator(simulator)

    #* Set Antenna Response
    AntennaResponse.registerLabels('response', ..., x = '...')
    x = AntennaResponse.x()
    RESPONSE_SIZE = AntennaResponse.size(flatten=True)

"""
from antenna.utils import *
from antenna.types import *
from antenna.utils.data import size_converter
# import numpy as np

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from loguru import logger #? pip3 install loguru
from functools import partial
import sys
from os.path import normpath
from time import time

def get_result_path(
    name:str = "{id}-{device}", *, 
    rootdir = None, 
    set_logger:bool = True, 
    generate_code:Optional[str] = None,
    excepthook_mode:Union[bool, Literal['only_hfss']] = 'only_hfss',
    enable_exception_handler: bool = False,
):
    """
    Args:
        name: Folder and log name, support {id}.
        set_logger: Whether to set the logger.
            EX: XXX.log
        generate_code: 
            EX: __file__
        excepthook_mode:
            If True, an email will be sent for any exception; if False, nothing will happen.
            [global_exception_handler(mode)]
        enable_exception_handler:
            config.enable_exception_handler = enable_exception_handler

    Examples: (equivalence)
    ```
    RESULT_PATH, EXISTS = get_result_path()
    RESULT_PATH, CONTINUE_RUN = get_result_path(
        "{id}-{device}", 
        rootdir = ROOTDIR, generate_code = __file__, enable_exception_handler = True
    )

    NAME = RESULT_PATH.stem
    ```
    """
    from script.process_files import FileProcessor
    _now = int(time())
    _device = get_local_ip().split('.')[-1]
    rootdir = Path(str(normpath(rootdir))) if rootdir else  Path(__file__).parent.parent
    result_path = rootdir.joinpath(
        "result", str(name.format(id = _now, device = _device))
    )
    exists  = result_path.exists()
    result_path.not_exist_create()

    if set_logger:
        logger.add(
            result_path.joinpath(f"{result_path.stem}.log"),
            format = "{time:YYYY-MM-DD HH:mm:ss} {level} {message}",
            level = "INFO",
        )
    if generate_code:
        FileProcessor(
            output_dir = result_path,
            project_name=result_path.stem,
            generated_by=generate_code,
            verbose = False
        ).run()

    # from .utils.utils import global_exception_handler
    config.excepthook = global_exception_handler(excepthook_mode)
    config.enable_exception_handler = enable_exception_handler

    config.NAME = result_path.stem
    config.RESULT_PATH = result_path
    config.CONTINUE_RUN = exists
    config.MAIN_PROGRAM = generate_code
    
    logger.info(f"The results will be saved in {result_path.absolute()} (Continue: {exists}, CUDA: {torch.cuda.is_available()})")
    return result_path, exists

def mult(_ob):
    _result = 1
    for i in _ob:
        _result *= i
    return _result

class MultiResponses:
    def __init__(self, responses:Union[dict, Tensor] = None):
        _responses:Dict[str, AntennaResponse] = {}
        if isinstance(responses, Dict):
            for key, response in responses.items():
                _responses[key] = AntennaResponse(response)
        elif isinstance(responses, Tensor):
            for n, response in enumerate(responses.reshape(AntennaResponse.size())):
                _responses[AntennaResponse.labels[n]] = AntennaResponse(response)
        elif responses is None:
            pass
        else:
            raise TypeError(f"Expected type `dict or Tensor`, but got type {type(responses)}")
        self.responses = _responses

    def __len__(self) -> int:
        return len(self.responses)

    def __str__(self):
        responses_str = " ".join([f"{k}[{v.response.shape.numel()}]" for k, v in self.responses.items()])
        return f"MultiResponses(num={self.__len__()}, key={responses_str})"

    def __getitem__(self, key) -> "AntennaResponse":
        if isinstance(key, int):
            if key >= self.__len__():
                raise IndexError(f"Expected size {self.__len__()} but got size {key}")
            key = list(self.responses.keys())[key]

        return self.responses[key]
    
    def __setitem__(self, key, value):
        self.responses[key] = AntennaResponse(value)

    def __delitem__(self, key):
        del self.responses[key]

    
    def __invert__(self):
        """Detach the response"""
        return self.stack().detach().cpu()
    
    def to_list(self):
        return [n.response for n in self.responses.values()]
    
    def stack(self) -> Tensor_W_H:
        return stack(self.to_list())
    
    def concat(self) -> Tensor_N:
        return concat(self.to_list())
    
    def size_converter(self, flatten: bool = False, batch: bool = False, output_shape = None) -> torch.Tensor:
        """:param output_shape: Priority use. (B, H, W, N) EX: "B, 1, H, W" or "B, N, 1" """
        return size_converter(
            AntennaResponse, self.to_list(),
            flatten = flatten, batch = batch, output_shape = output_shape
        )
    
    def criterion(self):
        """The loss will be calculated from the registered labels."""
        responses = {}
        for label, res in zip(AntennaResponse.labels, self.stack()):
            responses[label] = res

        loss = tensor(0.0, requires_grad=True)
        for key, value in responses.items():
            loss = loss + AntennaResponse(value).criterion(key)
        return loss

class TargetResponse(MultiResponses):
    def __init__(self):
        super().__init__(None)
        self._note = {}
        self.metadata:dict[str, dict] = defaultdict(dict)

    def __getitem__(self, key):
        """
        Target Response Design.

        Use `setTargetResponse()` before use, otherwise use the default value
        """
        if key not in self._note.keys():
            raise RuntimeError(
                f"The {key} of TargetResponse is not registered. " \
                "Please use `registerTargetResponse()` first."
            )
        return super().__getitem__(key)
    
    def __call__(self, side:float, center:float, width:Tuple[int,int,int,int,int], label:str = "response", add:bool = False) -> Tensor:
        """
        Target Response Design.

        :param side: The Y value at both ends of the response.
        :param center: The y value of the center point of the response.

        :return: AntennaResponse
        
        """
        if len(width) != 5:
            raise ValueError(f"Expected 5 width, but got {len(width)}")
        mask_up = np.concatenate([
            np.ones(width[0]) * side,
            np.linspace(side, center, width[1]),
            np.ones(width[2]) * center,
            np.linspace(center, side, width[3]),
            np.ones(width[4]) * side
        ])
        # expected_response = np.array(mask_up)#.reshape(-1, sum(_width))
        expected_response = tensor(np.array(mask_up), dtype=torch.float32, device=config.device)

        if add:
            self[label] = expected_response
            self._note[label] = f"side={side}, center={center}, width={width}"
            self.metadata[label].update({
                'response': expected_response,
                'side': side,
                'center': center,
                'width': width,
                'note': f"side={side}, center={center}, width={width}",
            })
     
        return expected_response
    
    def register_loss_fn(self, label, loss_fn, **loss_fn_param):
        self.metadata[label].update({
            'loss_fn': partial(loss_fn, **loss_fn_param),
            'loss_fn_name': loss_fn.__name__ if isinstance(loss_fn, FunctionType) else loss_fn.__class__.__name__
        })

    def loss_fn(self, label) -> Callable[..., Tensor]:
        return self.metadata[label]['loss_fn']
    
    @property
    def labels(self):
        return list(self.metadata.keys())
    
    @labels.setter
    def labels(self, labels:Iterable[str]):
        for label in labels:
            _ = self.metadata[label]
    
    def concat(self):
        _result = super().concat()
        if _result.size(0) != AntennaResponse.size(flatten = True):
            raise RuntimeError(
                'The concat size does not match the set size. ' \
                'Please check `AntennaResponse.registerLabels()`' \
                f'\n{_result.size(0)} != {AntennaResponse.size(flatten = True)}{AntennaResponse.size()}'
            )
        return _result
    
    def __str__(self):
       
        _ = " ".join(
            [
                f"{key}({value['note']}, loss={value.get('loss_fn_name', None)})" 
                for key, value in self.metadata.items()
            ]
        )
        return f"TargetResponse({_})"
    
class AntennaResponse(Generic[LossParams]):
    """
    Antenna Response Design.

    Attributes:
        response (Tensor): response
        target (TargetResponse): target response
    """
    x_patch_n257 = np.linspace(24, 32, 17) #? 26.5 - 28 - 29.5
    x_ris = np.linspace(0, 360, 361)

    target = TargetResponse()
    
    @overload
    def __new__(cls, response:Tensor) -> "AntennaResponse":...
    @overload
    def __new__(cls, responses:Dict) -> "MultiResponses":...

    def __new__(cls, response):
        if isinstance(response, cls):
            return response
        elif isinstance(response, Dict):
            return MultiResponses(response)
        else:
            return super(AntennaResponse, cls).__new__(cls)

    def __init__(self, response:Union[Tensor, Dict]):
        """
        Antenna Response Design.

        Args:
            response: Response of the antenna.
        
        Raises:
            TypeError: If the response is not a tensor.
        
        """
        if isinstance(response,(AntennaResponse, Dict) ):
            return
        elif isinstance(response, Tensor):
            response = response.to(config.device)
        else:
            raise TypeError("Expected Tensor, but got {}".format(type(response)))
        
        if len(response.shape) == 1:
            self.response = response
            self.vertical = self._reshape2vertical()
        else:
            self.response = response.reshape(-1)
            self.vertical = response

    def __str__(self):
        return f"AntennaResponse(size={self.response.size().numel()})"

    def __invert__(self):
        """Detach the response"""
        return self.response.detach().cpu()
    
    def _reshape2vertical(self):
        assert len(self.response.shape) == 1
        _v = self.response.reshape(1, self.response.shape[0])
        _v.requires_grad_(True)
        return _v

    def plot(self, label, axes:Optional[Axes] = None, show:bool = False):
        ax:Axes = plt.axes(axes) # type: ignore
        ax.set_title(f'Antenna Response')
        ax.plot(self.target[label].response.cpu().detach(), color='red', label='Target')
        ax.plot(self.response.cpu().detach(), color='blue', label='Simulation')
        ax.legend()
        if show: plt.show()
        return ax

    @classmethod
    def registerLabels(cls, *labels:str, x:Union[tuple[int, int, int], Literal['ris', 'n257']] = 'ris') -> Tensor:
        """
        The loss will be calculated from the registered labels.

        :param x: (start, stop, total)
        """
        match x:
            case 'ris':
                x = (0, 360, 361)
            case 'n257': #? 26.5 - 28 - 29.5
                x = (24, 32, 17) 
            case _:
                pass

        cls.target.labels = labels
        cls.labels = labels
        cls._x = x

    @classmethod
    def x(cls):
        """Get the x-axis value of this response."""
        if not hasattr(cls, '_x'):
            RuntimeError("No x registered. Please use `registerLabels()` first.")
        return np.linspace(*cls._x)

    @overload
    @classmethod
    def size(cls, flatten: Literal[True]) -> int: ...
    @overload
    @classmethod
    def size(cls, flatten: Literal[False]) -> Tuple[int, int]: ...
    @overload
    @classmethod
    def size(cls) -> Tuple[int, int]: ...
    @classmethod
    def size(cls, flatten:bool = False):
        """The number of labels used to calculate loss and the number of points in their labels."""
        if not cls.target.labels:
            raise RuntimeError("No labels registered. Please use `registerLabels()` first.")
        _ = (len(cls.target.labels), cls._x[2])
        return _[0] * _[1] if flatten else _
    
    @classmethod
    def to_str(cls):
        """Get response information and default values."""
        return f"AntennaResponse(size={cls.size()}, x={cls._x}, target={cls.target})"
    
    @classmethod
    def registerTargetResponse(cls, side:float, center:float, width:Tuple[int,int,int,int,int], label:str = "response") -> Tensor:
        """
        Target Response Design.

        :param side: The Y value at both ends of the response.
        :param center: The y value of the center point of the response.

        :return: AntennaResponse
        
        """
        if not cls.target.labels:
            raise RuntimeError(
                "No labels registered. Please use `registerLabels()` first."
            )
        is_add = label in cls.target.labels
        return cls.target(side, center, width, label = label, add = is_add)

    @classmethod
    def registerLossHook(cls, loss_hook:Callable[LossParams, Tensor], label:str = "response", **loss_hook_param:LossParams.kwargs):
        """
        Args:
        
            loss_hook: Used for `criterion()`

            ```
            def criterion(response, target_response, ...):...
            ```
        
        """
        cls.target.register_loss_fn(
            label, loss_hook, **loss_hook_param
        )

    def criterion(self, label:str = "response", **param:LossParams.kwargs) -> Tensor:
        """[Loss Function] Register LossHook using `registerLossHook()` before use."""
        if label not in self.target.labels:
            raise RuntimeError(f"The {label} of LossHook is not registered. Please use `registerLossHook()` first.")
        
        return self.target.loss_fn(label)(
            self.response, **param
        )

class AntennaPattern:
    _history_datas:List[List[torch.Tensor]] = []
    _best_loss = float('inf')

    tau:float = 1.0
    """
    The temperature parameter controls the steepness of the Sigmoid. 
    - A smaller tau (e.g., 0.1) makes the approximation closer to hard binarization. 
    - It must be > 0.
    """
    
    def __new__(cls, pattern:"AntennaPattern", *args) -> "AntennaPattern":
        if isinstance(pattern, AntennaPattern):
            return pattern
        else:
            return super(AntennaPattern, cls).__new__(cls)
    
    @overload
    def __init__(self, pattern:Tensor, coordinate:Optional[Tuple[int,int, int, int]] = None):
        """
        Example:
        ```
        AntennaPattern.setCoordinate((0, 25, 0, 25))
        ```
        """
    @overload
    def __init__(self, patterns:List[Tuple[Tensor, int, int, int, int]]):
        """
        Args:
            pattern: [(pattern, x1, x2, y1, y2), ...] >>> pattern is 2D
        """
    
    def __init__(self, pattern:Union[Tensor, List], coordinate:Optional[Tuple[int,int, int, int]] = None):
        
        if isinstance(pattern, AntennaPattern):
            return
        
        #* The core of this class.
        #? [(pattern, x1, x2, y1, y2), ...] >>> pattern is 2D
        self.patterns:List[Tuple[Tensor, int, int, int, int]] = [] 

        if isinstance(pattern, Tensor):
            self.input_tensor = torch.clamp(pattern.to(config.device), min=0.0, max=1.0)
            self.coordinate:Union[Tuple[int,int, int, int], Tuple] = coordinate or getattr(self, '_antenna_pattern_coordinate', None)

            self._check_input()

        elif isinstance(pattern, List):
            self.patterns = pattern
        
        else:
            raise TypeError(
                f"Expected type for pattern is Tensor or List, but got {type(pattern)}"
            )
    
    def _check_input(self):
        _dim = self.input_dim()
        _c = self.coordinate
        _input_tensor = self.input_tensor
        
        if not _c: raise ValueError(
            'Please enter the `coordinate` parameter or use `setDefaultCoordinate()` to set the default value.'
        )
        if _dim == 1:
            _input_tensor = _input_tensor.reshape((_c[1]-_c[0], _c[3]-_c[2]))
        elif _dim == 2:
            pass
        else:
            raise ValueError(f"Input pattern expected >1 dimension, but got {_dim} dimension")
        
        self.patterns.append(
            (
                _input_tensor, _c[0], _c[1], _c[2], _c[3]
            )
        )
        
    @property
    def series(self):
        """One-dimensional array after merge."""
        return self.merge().reshape(-1)
    
    @property
    def fill_rate(self) -> float:
        """計算並返回天線 pattern 的金屬填充率。"""
        merged_pattern = self.merge()
        if merged_pattern.numel() == 0:
            return 0.0
        return (torch.sum(merged_pattern) / merged_pattern.numel()).item()
    
    @classmethod
    def register_simulator(cls, simulator:Callable[[Tensor],Dict[str, Tensor]]):
        cls._simulator = simulator

    @classmethod
    def getAllPixel(cls):
        """
        TODO: 目前是取回所有的像素點，但實際上是取得大圖的像素點
        """
        x1, x2, y1, y2 = cast(Tuple[int,int, int, int], getattr(cls, '_antenna_pattern_coordinate', (0,0,0,0)))
        return (x2-x1)*(y2-y1)
    
    @overload
    @classmethod
    def size(cls, flatten: Literal[True]) -> int: ...
    @overload
    @classmethod
    def size(cls, flatten: Literal[False]) -> Tuple[int, int]: ...
    @overload
    @classmethod
    def size(cls) -> Tuple[int, int]: ...
    @classmethod
    def size(cls, flatten:bool = False):
        """The number of labels used to calculate loss and the number of points in their labels."""
        if not hasattr(cls, '_antenna_pattern_coordinate'):
            raise RuntimeError("Please use `setDefaultCoordinate()` first.")
        x1, x2, y1, y2 = cast(Tuple[int,int, int, int], getattr(cls, '_antenna_pattern_coordinate', (0,0,0,0)))

        return (x2-x1)*(y2-y1) if flatten else ((x2-x1), (y2-y1))

    def size_converter(self,flatten: bool = False, batch: bool = False, output_shape = None) -> torch.Tensor:
        """:param output_shape: Priority use. (B, H, W, N) EX: "B, 1, H, W" or "B, N, 1" """
        return size_converter(
            self, self.merge(),
            flatten = flatten, batch = batch, output_shape = output_shape
        )

    @classmethod
    def _getRandomPattern(cls, w=40, h=40):
        patterns = torch.randn(
            w, h, 
            dtype = torch.float32,
            device = config.device
        )
        binaries = (patterns > 0.5).float()
        return cls(binaries, (0, w, 0, h))
    
    @classmethod
    def getRandomPattern(cls, shape: tuple, fill_rate: float = 0.5) -> Self:
        """
        根據指定的填充率 (fill rate) 生成一個二元 (0/1) 的 pattern。

        Args:
            shape (tuple): Pattern 的形狀, 例如 (w, h)。
            fill_rate (float): 金屬填充的比例, 範圍在 0.0 到 1.0 之間。

        Returns:
            生成的二元 pattern。
        
        Exapmple::

            AntennaPattern.getRandomPattern((25, 25), fill_rate = np.random.uniform(0.1, 0.9))
        """
        w = shape[0]
        h = shape[1]
        total_pixels = w * h
        num_ones = int(total_pixels * fill_rate)
        
        # 生成一個扁平化的一維數組
        pattern_flat = np.zeros(total_pixels)
        pattern_flat[:num_ones] = 1
        
        # 隨機打亂
        np.random.shuffle(pattern_flat)
        
        # 重塑為目標形狀並轉換為 PyTorch Tensor
        pattern_tensor = torch.tensor(pattern_flat.reshape(shape), dtype=torch.float32, device = config.device)
        return cls(pattern_tensor, (0, w, 0, h))

    def __str__(self):
        _shape = self.merge().shape
        return f"AntennaPattern(Pattern_num={self.__len__()} Shape=[{_shape[0]}, {_shape[1]}] Size=[{_shape.numel()}])"
    
    def __getitem__(self, key) -> "AntennaPattern":
        if key >= self.__len__():
            raise IndexError(f"Expected size {self.__len__()} but got size {key}")
        pattern, x1, x2, y1, y2 = self.patterns[key]
        return AntennaPattern(pattern, (x1, x2, y1, y2))
    
    def __add__(self, other):
        if isinstance(other, AntennaPattern):
            antenna_pattern = self.copy()
            antenna_pattern.patterns = self.patterns + other.patterns
            antenna_pattern.coordinate = None
            antenna_pattern.input_tensor = None

            return antenna_pattern
        else:
            raise TypeError(
                "Unsupported operand type for +: 'AntennaPattern' and '{}'".format(type(other))
            )
    
    def __len__(self):
        return len(self.patterns)
    
    def __invert__(self):
        """Detach the response"""
        return self.merge().detach().cpu()
    
    def input_dim(self) -> int:
        if self.input_tensor is None:
            raise RuntimeError("This function is not for multilayer boards.")
        
        if len(self.input_tensor.shape) == 1 or self.input_tensor.shape[0] == 1:
            return 1
        else:
            return self.input_tensor.dim()   
            
    def copy(self):
        return AntennaPattern(self.patterns)

    @classmethod
    def setDefaultCoordinate(cls, _coordinate:Tuple[int, int, int, int]):
        """
        Coordinate Design.

        """
        if not isinstance(_coordinate, tuple):
            raise TypeError(f"Expected tuple, but got {type(_coordinate)}")

        if not len(_coordinate) == 4:
            raise ValueError(f"Expected tuple of length 4, but got {len(_coordinate)}")
        
        setattr(cls, '_antenna_pattern_coordinate', _coordinate)

    def binarize(self, threshold = 0.5):
        """Binarize and become gradient-free."""
        bi = (self.merge() >= threshold).float()
        return AntennaPattern(bi, (0, len(bi), 0, len(bi)))
    
    @classmethod
    def binarization(cls, pattern:Tensor, tau:Optional[float] = None, threshold = None):
        """
        Perform differentiable binarization using the STE technique.

        Args:
            pattern: The pattern to be binarized. Its requires_grad will be set to True.
            tau:   The temperature parameter controls the steepness of the Sigmoid. 
                    A smaller tau (e.g., 0.1) makes the approximation closer to hard binarization. 
                    It must be > 0.

        Returns:
            torch.Tensor: Binarized tensor
        """
        #* Gradient is required
        pattern.requires_grad_(True)
        cls.tau:float = tau or getattr(cls, 'tau', 1.0)

        if len(pattern.shape) == 1:
            pattern = pattern.reshape(*cls.size()) 
        
        #* Calculate threshold and steepness
        threshold = threshold or pattern.mean().detach() # avg
        steepness = 1/cls.tau

        #* Produces a "soft" approximation
        #  This is to provide a smooth gradient during "backward" propagation.
        soft_pattern = torch.sigmoid(steepness * (pattern - threshold))

        #* Produces a "hard" binarization result (0/1, not differentiable).
        #  This is to get the 0/1 result you want during "forward" propagation.
        hard_pattern = torch.round(soft_pattern)

        #* STE
        #  Forward(hard):   (hard - soft) + soft
        #  Backward(soft)： `.detach()` will block the gradient of hard_pattern
        binary_pattern = (hard_pattern - soft_pattern).detach() + soft_pattern
        
        return binary_pattern

    def binarization_(self, tau:Optional[float] = None, threshold = None):
        pattern = self.merge().clone()
        shape = pattern.shape
        self.patterns = [(
            AntennaPattern.binarization(pattern, tau=tau, threshold=threshold),
            0, shape[1], 0, shape[0]
        )]
        

    def merge(self) -> torch.Tensor:
        """
        將所有 pattern 合併成一個大的底層 pattern
        - 後加入的 pattern 會覆蓋前面的 pattern
        - 返回合併後的二維 tensor
        """
        if not self.patterns:
            raise ValueError("No patterns to merge")

        max_x = max(x2 for _, _, x2, _, _ in self.patterns)
        min_x = min(x1 for _, x1, _, _, _ in self.patterns)
        
        max_y = max(y2 for _, _, _, _, y2 in self.patterns)
        min_y = min(y1 for _, _, _, y1, _ in self.patterns)
        
        base_pattern = torch.zeros((max_y, max_x))
        for pattern, x1, x2, y1, y2 in self.patterns:
            base_pattern[y1:y2, x1:x2] = pattern  # 後面的 pattern 覆蓋前面的

        return base_pattern.to(config.device)[min_y:max_y, min_x:max_x]
    

    def simulate(self, no_grad:bool = True, **param):
        pattern = self.merge()
        result_response = {}
       
        if hasattr(self, "_simulator"):
            if no_grad:
                with torch.no_grad():
                    result:Dict[str, Tensor] = self._simulator(pattern.detach(), **param)
            else:
                result:Dict[str, Tensor]  = self._simulator(pattern, **param)
        else:
            raise RuntimeError(
                "Please use `register_simulator()` to register the simulator."
            )
        
        for key, value in result.items():
            result_response[key] = AntennaResponse(value)

        # TODO 
        # if not any([pattern.equal(p) for p, _ in self._history_datas]):
        AntennaPattern._history_datas.append(
            [pattern, result_response]
        )

        return AntennaResponse(result_response)

    
    def plot(self, axes:Optional[Axes] = None, show:bool = False, title:str = "Antenna Pattern {shape}"):
        ax:Axes = plt.axes(axes) # type: ignore
        ax.set_title(title.format(shape=self.size()))
        ax.imshow(self.merge().cpu().detach(), cmap='viridis')
        ax.axis('off')
        if show: plt.show()
        return ax
    
    def plot_individual(self, axes:Optional[Axes] = None, show:bool = False):
        if not self.patterns:
            raise ValueError("No patterns to merge")

        max_x = max(x2 for _, _, x2, _, _ in self.patterns)
        max_y = max(y2 for _, _, _, _, y2 in self.patterns)
        base_pattern = torch.zeros((max_x, max_y), dtype=self.input_tensor.dtype)
        _result = []
        for pattern, x1, x2, y1, y2 in self.patterns:
            _pattern = base_pattern.clone()
            _pattern[x1:x2, y1:y2] = pattern  
            _result.append(_pattern)

        ax:Axes = plt.axes(axes) # type: ignore
        ax.set_title("Antenna Pattern Individual")
        ax.imshow(torch.cat(_result, dim=1).cpu().detach(), cmap='viridis')
        if show: plt.show()
        return ax

    def mutate(self, rate):
        matrix = self.merge()
        total = matrix.numel()
        n = int(total * rate)
        indices = torch.randperm(total).tolist()
        selected_indices = indices[:n]
        
        for idx in selected_indices:
            i, j = divmod(idx, matrix.size(1))
            matrix[i, j] = 1 - matrix[i, j]
        return AntennaPattern(matrix)

def reshape(_tensor:torch.Tensor):
    _shape = _tensor.shape
    if len(_shape) == 1:
        return _tensor.reshape(1, _shape[0])
    else:
        return _tensor.reshape(_shape[0], 1)

if __name__ == "__main__":
    config.device = 'cpu'
    # ap = tensor(np.random.rand(40*40), dtype=torch.float32)
    # binary_ap = (ap >= 0.5).float()
    # response = AntennaResponse.getTargetResponse()
    # pattern = AntennaPattern(binary_ap, [(0, 40, 0,40)])
    # pattern.plot()
    # response.plot()
    response = AntennaResponse(torch.randn(361))
    response.plot()





