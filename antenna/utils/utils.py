from typing import (
    Tuple, List, Dict, Deque, # Can use the built-in.
    TypeVar, cast, Callable, Any, Optional, overload, Union, Sequence, Literal
)
from loguru import logger
import traceback
import torch
from torch import (
    __version__,
    Tensor,
    cuda,
    manual_seed as _manual_seed,
    load as _torch_load,
    save as _torch_save,
    device as _torch_device,
    # get_default_device,
    set_default_device,

    set_grad_enabled, is_grad_enabled # with no_grad():...
)
from numpy import (
    ndarray,
    random
)
from  pickle import (
    dump as _pickle_dump,
    load as _pickle_load
)
from json import (
    load as _json_load, 
    dump as _json_dump
)
from pandas import DataFrame
from tqdm import trange
from collections import defaultdict
from warnings import filterwarnings

from pathlib import Path as _Path
from os.path import getctime, exists

from sys import maxsize

import numpy as np
from copy import deepcopy
from datetime import datetime
from shutil import rmtree as _rmtree

#* Figure
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.axes._axes import Axes  # type: ignore

FIG_CONFIG = {
    "format": 'png',
    "bbox_inches": "tight",
    "pad_inches": 0.1,
    "dpi": 300,
    "transparent": True,
    "facecolor": "none", # white
    "edgecolor": "none",
}

def errorCallback(errorCallback:Optional[Callable[[str],Any]]=None, *errorCallbackArgs, **errorCallbackKwargs):
    """
    Error callback function.

    ## Usage
    ```python
    @errorCallback()
    def func():
        raise Exception("Error")
    ```
    """
    def decorator(func:Callable):
        def wrap(*args, **kwargs):
            try:
                return func(*args, **kwargs)   # print(func.__name__)
            except Exception as e:
                if errorCallback:
                    errorCallback(str(e), *errorCallbackArgs, **errorCallbackKwargs)
                else:
                    logger.exception(e)
        return wrap
    return decorator



    
class Path(type(_Path()), _Path): # type: ignore
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)
    
    def __init__(self, path: str, create:bool=False):
        """
        Path model.
       
        ## Usage
        ```python
        path = Path("./path/to/file.ext")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
        path.unlink()
        path.del_from_glob('*.pth')
        path.manage_file_count('*.pth', keep_latest=3)
        path.load_torch()
        path.not_exist_create(create_file=True)
        ```
        """
        # self.path = path
        
        if create: self.not_exist_create()

    def rmtree(self) -> bool:
        if self.is_dir():
            _rmtree(self)
            return True
        else:
            return False
    
    def not_exist_create(self, create_file:bool = False):
        """
        Create the path if it does not exist.

        :param create_file: Whether to create the file.
        :return: Whether the path does not exist.
        """
        if self.suffix:
            self.parent.mkdir(parents=True, exist_ok=True)
            if create_file: self.touch(exist_ok=True)
        else:  # No file extension, treated as a directory.
            self.mkdir(parents=True, exist_ok=True)
        return self

    def del_from_glob(self, pattern:str):
        """
        Delete all files matching the pattern.

        :param pattern: Patterns matching files, E.g., '*.pth'
        """
        if not self.suffix:
            paths = list(self.glob(pattern))
            for path in paths:
                path.unlink()
        else:
            self.unlink()

    def manage_file_count(self, pattern:str, keep_latest:int = 3):
        """
        Manage the number of archives and only keep the latest specified number.

        :param pattern: Patterns matching archives, E.g., '*.pth'
        :param keep_latest: Latest quantity to keep.
        """

        # Confirm that the target directory exists.
        if not self.exists():
            raise FileNotFoundError(f"The destination directory ({self.absolute()}) does not exist.")
        
        # Get all files matching the pattern.
        files_sorted = sorted(self.glob(pattern), key=getctime)

        # If the file exceeds the limit, delete the oldest file.
        if len(files_sorted) > keep_latest:
            for old_backup in files_sorted[:len(files_sorted)-keep_latest]:
                if not old_backup.rmtree():
                    old_backup.unlink()
            return True
        else:
            return False
    
    def load_torch(self, device = None):
        from antenna.models import config
        if __version__ >= "2.6.0":
            return _torch_load(self, weights_only=False, map_location=device or config.device)
        else:
            return _torch_load(self, map_location=device or config.device)

def plot(x,file_name:Optional[str] = None) -> None:
        """
        Plot the weight matrix on a 3D graph
        """
        # This part is for plotting the graph
        plt.clf()
        # plt.figure(figsize=(20, 10))
        plt.title(f'')
        plt.plot(x)
        plt.legend()

        plt.show()

        if file_name: plt.savefig(file_name, **FIG_CONFIG)

class Config(dict):
    def __init__(self):
        self.epochs = 10
        self.lr = 1e-3 # Learning Rate For Main Training Loop
        self.element_num = 40
        self['checkpoint_save_path'] = Path("./checkpoint")
        self['device'] = _torch_device(type='cpu')
    
    def __getattr__(self, name):
        # 該Class沒有此變數(name)會執行。
        return self[name]

    def __setattr__(self, name, value):
        # 該Class沒有此變數(name)會執行。
        self[name] = value

    def check_keys(self, *keys:str, only_warning:bool = False ):
        for key in keys:
            if key not in self:
                if only_warning:
                    logger.warning(f'{key} not set.')
                else:
                    raise KeyError(f'{key} not set.')
    @property
    def device(self):
        return self['device']
    
    @device.setter
    def device(self, device):
        device = device or _torch_device("cuda:0" if cuda.is_available() else "cpu")
        set_default_device(device)
        if device != "cpu":
            cuda.set_device(device)
        self['device'] = device
    
    @property
    def checkpoint_save_path(self) -> Path:
        """
        ```
        config.checkpoint_save_path.not_exist_create()
        ```
        """
        return self['checkpoint_save_path'].absolute()
        
    @checkpoint_save_path.setter
    def checkpoint_save_path(self, path):
        self['checkpoint_save_path'] = Path(path)

    def setRandomSeeds(self, seed = 0):
        _manual_seed(seed)
        cuda.manual_seed(seed)
        random.seed(seed)
    
    def setWarning(self, warning_type:str = "ignore"):
        return filterwarnings(warning_type) # type: ignore
    
    def save(self, name:str = 'config', rootdir:Optional[str] = None):
        """
        Only save the following types
        ```
        dict, list, tuple, str, int, float, bool, None
        ```
        If it is other, it will be automatically converted to a string using `str()`
        """
        path = Path(rootdir or "./", f"{name}.json")
        _save = {}
        self.update(vars(self))
        for key, value in self.items():
            if isinstance(value, (dict, list, tuple, str, int, float, bool)) or value is None:
                _save[key] = value
            else:
                _save[key] = str(value)
        with open(path,'w') as f:
            _json_dump(_save, f, indent = 4)
    
    def load(self, name:str = 'config', rootdir:Optional[str] = None):
        """
        Only load the following types
        ```
        dict, list, tuple, str, int, float, bool, None
        ```
        """
        # TODO
        path = Path(rootdir or "./", f"{name}.json")
        with open(path, 'r', encoding = 'utf-8') as f:
            self.update(_json_load(f))
    
    def __str__(self):
        _str = ", ".join(f"{k}={v}" for k, v in self.items())
        return f"{self.__class__.__name__}({_str})"

config = Config()



class Figure:
    def __init__(self, name:str, nrowcol:tuple = (1, 1), save:bool = False, show:bool = False, rootdir:Optional[str] = None, size = (18, 12), **kwargs):
        """
        :param size: Example: (18, 12), (18 * 2, 9 * 2)
        :param kwargs: All plt.figure() arguments

        ## Example
        ```
        with Figure("test_3_2", nrowcol=(2,2), save=True) as fig:
    
            ax1 = fig.index(1)
            ax1.set_title("test")
            ax1.plot([1, 2, 3, 4])

            fig.addAll()
            fig[2].set_title("test")
            fig[2].plot([1, 2, 3, 4])
        ```
        
        ## Set
        ```
        class:
            ...
            def plot(self, axes:Axes|None = None):
                ax:Axes = plt.axes(axes) # type: ignore
                ax.set_title("test")
                ax.plot([1, 2, 3, 4])
        ```

        ## 動畫
        ```
        line = {}
        epochs = 1500
        line = np.random.random((epochs))
            
        with Figure("line", rootdir=r'./') as fig:
            fig.addAll()
            def update(frame):
                fig[0].clear()
                fig[0].set_title("line")
                fig[0].set_xlim(0, epochs)
                fig[0].plot(line[:frame+1])

                fig.fig.tight_layout(pad=0.1)
                
                return fig
            fig.saveMP4(update, epochs, video_time=5)
        ```
        """
        fig = plt.figure(name, **kwargs)
        fig.set_size_inches(*size)
        fig.tight_layout(pad=0.1)
        # fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)

        self.fig = fig
        self.save = save
        self.show = show
        self.name = name
        self.nrowcol = nrowcol    
        self.current_index = 1
        self.rootdir = Path(rootdir or "./")
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, nrowcol={self.nrowcol}, save={self.save}, show={self.show}, rootdir={self.rootdir.absolute()}, size={self.fig.get_size_inches()})"


    def index(self, index:int = 1):
        """
        :param index: Support -1
        """
        index = self.current_index if index == -1 else index
        ax = self.fig.add_subplot(self.nrowcol[0], self.nrowcol[1], index)
        self.current_index += 1
        return ax
    
    def addAll(self):
        for i in range(self.__len__()) :
            self.index(i+1)

    def saveGIF(self, update:Callable, epochs:int = 10, dpi = 150):
        writer = PillowWriter(fps=30, metadata={"artist": "WeiWen Wu"})
        tqdm_iter = trange(epochs, desc="Plotting")
        ani = FuncAnimation(self.fig, update, frames=epochs)
        ani.save(f"{self.rootdir.joinpath(self.name)}.gif", writer=writer, dpi=dpi, progress_callback=lambda i, n: tqdm_iter.update())
    
    def saveMP4(self, update:Callable[[int], "Figure"], epochs:int = 10, dpi = 150, video_time = None, del_temp = False):
        from imageio_ffmpeg import get_ffmpeg_exe #? pip install imageio-ffmpeg
        metadata = {
            'title': f'{self.name}',
            "artist": "WeiWen Wu",
            'comment': "Provided by WeiWen's kit"
        }
        rcParams['animation.ffmpeg_path'] = get_ffmpeg_exe()
        
        path_video_temp = self.rootdir.joinpath('video_temp').not_exist_create()
        path_merges:list[Path] = []
        for n in trange(epochs, desc='Creating'):
            self.fig.clear()
            path_merges.append(
                update(n).saveIMG(
                    path_video_temp.joinpath(f'{n}.png')
                )
            )
        def _update(frame):
            plt.clf()
            plt.imshow(
                plt.imread(path_merges[frame])
            )
            plt.axis('off')
            plt.tight_layout(pad=0)
            return self
            
        fps = int(epochs/video_time) if video_time else 30
        writer = FFMpegWriter(fps=max(1, min(fps, 120)), metadata=metadata) # , bitrate=1800
        filename = self._ani_save(_update, epochs, writer, dpi)
        writer.finish()
        logger.info(f'Video creation completed. ({filename.absolute()}, fps: {fps})')
        if del_temp: path_video_temp.rmtree()
        
    
    def _ani_save(self, update: Callable[[int], Any], epochs, writer, dpi):
        tqdm_iter = trange(epochs, desc="Plotting")
        ani = FuncAnimation(self.fig, update, frames=epochs)
        filename = self.rootdir.joinpath(f"{self.name}.mp4")
        ani.save(
            filename, writer=writer, dpi=dpi, 
            progress_callback=lambda i, n: tqdm_iter.update(),
        )
        return filename
        
    
    def saveIMG(self, path = None):
        FIG_CONFIG = {
            "format": 'png',
            "bbox_inches": "tight",
            "pad_inches": 0.1,
            "dpi": 300,
            "transparent": True,
            "facecolor": "white", # white or none
            "edgecolor": "white", # white or none
        }
        # self.fig.set_size_inches(18, 12)
        path = path or self.rootdir.joinpath(f"{self.name}.png")
        plt.savefig(path, **FIG_CONFIG) 
        return path
        
    def __getitem__(self, index:int) -> Axes:
        """
        Use first
        ```
        fig.addAll()
        ```
        """
        return self.fig.get_axes()[index]

    def __len__(self) -> int:
        return self.nrowcol[0] * self.nrowcol[1]

    def __enter__(self):
        self.prev = is_grad_enabled()
        set_grad_enabled(False)

        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback):
        if not exc_type:
            if self.show: plt.show()
            if self.save: self.saveIMG()
        plt.close()
        set_grad_enabled(self.prev)

class Record:
    def __init__(self, name:str = "record", rootdir:Optional[str] = None, load:bool = False):
        self._data = defaultdict(list)
        self._history = defaultdict(list)
        self.name = name
        self.path = Path(rootdir or "./").joinpath(
            f"{name}.record"
        )

        if load: self.load()
    
    def __call__(self, key, default = None, *, append = False):
        """Get the last value of key."""
        return self.end(key, default, append = append)

    def __setitem__(self, key, value):
        self._data[key].append(value)
                                
    def __getitem__(self, key):
        """Get the complete array of keys."""
        if self.__contains__(key):
            return self._data[key]
        else:
            _keys = ', '.join(self._data.keys())
            raise KeyError(f"{key} does not exist. (Current key: {_keys})")
        
    def  __contains__(self, item:str):
        return item in self._data

    def end(self, key, default = None, *, append = False):
        if self.__contains__(key):
            return self.__getitem__(key)[-1]
        else:
            if append:
                self.__setitem__(key, default)
                return self.end(key)
            else:
                return default
        
    def add(self, key, num, default = None):
        """
        add('a', 1):
        a += 1
        """
        self.__setitem__(
            key, self.end(key, default) + num
        )
        

    
    def save(self, description:Optional[str] = None):
        self._history["time"].append(str(datetime.now()).split(".")[0])
        self._history["description"].append(description or "No description")
        self._history["len"].append(len(self))

        with open(str(self.path), "wb") as f:
            _pickle_dump(
                {
                    "_data": self._data,
                    "_history": self._history
                }, 
                file = f
            )
    
    def load(self):
        if not self.path.exists():
            self.save()
        with open(str(self.path), "rb") as f:
            _loaded = _pickle_load(f)
            self._data = _loaded["_data"]
            self._history = _loaded["_history"]
        
        return self._data
    
    def average(self, key):
        return sum(self._data[key]) / len(self._data)
    
    def index(self, key:str, value, *, start:int = 0, stop:int = maxsize) -> Optional[int]:
        """
        Find the index of `value` in `key`.
        
        Returns:
            Returns the index value, starting from 0. 

            If `value` is not in `key`, returns `None`.

        Example:
            ```
            temp = Record('temp')
            for epoch in range(1, 10+1):
                temp['epoch'] = epoch
            print(temp.index('epoch', 0)) # None
            print(temp.index('epoch', 1)) # 0
            ```
        """
        if isinstance(value, ndarray):
            _result = [
                np.array_equal(value, x) 
                for x in self[key][start:stop]
            ]
        elif isinstance(value, Tensor):
            import torch
            _result = [
                torch.equal(value, x) 
                for x in self[key][start:stop]
            ]
        else:
            if value in self[key]:
                return self[key].index(value, start, stop)
            else:
                return None
        
        if True in _result:
            return _result.index(True)
        else:
            return None
    
    @overload
    def find(self, key, value, other_keys:str, *, start=0, stop=maxsize) -> Optional[Any]:...
    @overload
    def find(self, key, value, other_keys:Tuple[str, ...] , *, start=0, stop=maxsize) -> Optional[List[Any]]:...

    def find(self, key, value, other_keys, *, start=0, stop=maxsize):
        """
        Find the `value` in `key` that corresponds to `other keys`.

        Returns:
            Returns the `value` corresponding to the `other key`.

            If `value` is not in `key`, returns `None`.

        Examples:
            ```
            temp = Record('temp')
            for epoch, (a, b) in enumerate(zip(
                ['a1', 'a2', 'a3'], ['b1', 'b2', 'b3']
            ), start = 1):
                temp['epoch'] = epoch
                temp['a'] = a
                temp['b'] = b

            print(temp.find('a', 'a1', "epoch"))    # 1
            print(temp.find('epoch', 3, ('a','b'))) # ['a3', 'b3']
            ```
        """
        _index = self.index(key, value, start=start, stop=stop)
        if _index is None:
            return None
        elif isinstance(other_keys, str):
            return self[other_keys][_index]
        else:
            _result = []
            for other_key in other_keys:
                _result.append(self[other_key][_index])
            return _result

    def early_stop(self, key: str, patience: int = 10, is_maximize: bool = False) -> bool:
        """
        根據指定 key 的歷史資料，決定是否應該 early stop。
        若最近 `patience` 次都沒有改善，回傳 True。
        Args:
            is_maximize: 若為 True, 則尋找最大值, 否則尋找最小值。
        """
        values = self._data[key]
        if len(values) < patience + 1:
            return False  # 數據不足，不應該停止

        # 根據是最大化還是最小化來決定如何判斷最佳值
        if is_maximize:
            best_func = max
            comparison_op = lambda current, best: current <= best # 對於最大化，如果當前值小於最佳值則視為退步
        else:
            best_func = min
            comparison_op = lambda current, best: current >= best # 對於最小化，如果當前值大於最佳值則視為退步

        # 'best_so_far' 應該是到目前為止，在 patience 視窗之前所見的整體最佳值
        best_so_far = best_func(values[:len(values) - patience])
        recent_values = values[len(values) - patience:]

        # 檢查所有最近的數值是否都比 best_so_far 差
        if all(comparison_op(v, best_so_far) for v in recent_values):
            return True
        return False

    def reset(self):
        self._data = defaultdict(list)
    
    def custom(self, key:str, fn:Callable, *, default = None):
        _ket_data = self._data[key]
        if _ket_data:
            return fn(_ket_data)
        return default
    
    @property
    def dataframe(self):
        processed_data = {}
        for key, values in self._data.items():
            processed_values = []
            for item in values:
                if isinstance(item, torch.Tensor):
                    # Move to CPU and detach to convert to a standard Python list/number
                    processed_values.append(item.cpu().detach().tolist())
                else:
                    processed_values.append(item)

            processed_data[key] = processed_values
        try:
            return DataFrame(processed_data)
        except ValueError as e:
            raise ValueError(f"{e}\n{repr(self)}")

    @property
    def history(self):
        return DataFrame(self._history)

    def __str__(self):
        return str(self.dataframe)
    
    def __repr__(self):
        _str = ''
        for key, value in self._data.items():
            _str += f"{key}[{len(value)}] "

        return f"Record({self.name}: {_str})"
    
    def __len__(self):
        return len(self.dataframe)
    
class json:
    """
    ### Example
    ```
    from utils.utils import json
    _json = json('static/config.json')
    print(_json('base/UPLOAD_FOLDER'))
    _json_data = _json.load()
    print(_json_data['success'])
    _json_data['success'] = False
    _json.dump(_json_data)
    ```
    """
    def __init__(self, path:str, create:bool = True) -> None:
        self.path = Path(path)
        
        if not self.path.exists():
            if create:
                self.path.touch()
                self.dump({})
            else:
                raise FileNotFoundError(f"JSON file '{path}' does not exist.")

    @overload
    def __call__(self, key:str) -> Any: 
        """
        Get the value of the specified key in the JSON file.

        ### Example
        >>> _json('base/UPLOAD_FOLDER')
        """
    ...
    @overload
    def __call__(self, key:str, value:Any) -> dict: 
        """
        Set the value of the specified key in the JSON file.

        ### Example
        >>> _json('base/UPLOAD_FOLDER', 'new/path')
        """
    ...
    def __call__(self, key:str, value = None):
        keys = key.split('/')
        if value is not None: 
            if value == "null": value = None
            if value in ["True", "true"]: value = True
            if value in ["False", "false"]: value = False
            result = self._set(keys, value)
            self.dump(result)
            return result
        else:
            return self._get(keys)
    def __getitem__(self, key):
        return self.__call__(key, value = None)
    def __setitem__(self, key, value):
        return self.__call__(key, value)
    
    def get(self, key:str, default = None):
        keys = key.split('/')
        try:
            return self._get(keys)
        except KeyError:
            result = self._set(keys, default)
            self.dump(result)
            return default

    def _set(self, keys:list, value:Any) -> dict:
        temp =  self.load().copy()
        _ = "temp"
        for i, k in enumerate(keys):
            if k == '': continue
            _ += f"['{k}']"

            if i == len(keys) - 1:
                exec(f"{_} = value")
            else:
                if k not in temp:
                    exec(f"{_} = {{}}")

        return temp
        
    def _get(self, keys:list) -> Any:

        self.data = self.load()
        result = self.data.copy()
        for k in keys:
            if k == '': continue
            result = result[k]
        return result
    def load(self) -> dict:
        with open(self.path, 'r', encoding='utf-8') as f:
            return _json_load(f)

    def dump(self, data:dict) -> bool:
        with open(self.path, 'w', encoding='utf-8') as f:
            _json_dump(data, f, ensure_ascii=False, indent=4)
        return True

    def delete(self, key:str) -> bool:
        keys = key.split('/')
        data = self.load()
        
        # Traversing through the keys
        temp = data
        for k in keys[:-1]:  # Get to the parent of the key to delete
            if k in temp:
                temp = temp[k]
            else:
                return False  # If the key doesn't exist, return False
        
        # Deleting the key
        if keys[-1] in temp:
            del temp[keys[-1]]
            self.dump(data)  # Save the updated data back to the file
            return True
        else:
            return False





if __name__ == "__main__":
    # print(Path("./checkpoint").manage_file_count("*.pth", keep_latest=1))
    # print(Path("./checkpoint/GEN_model_0.pth").load_torch())
    config.device = 'cpu'
    r = Record('Temp', load=True, rootdir=r"D:\patch_result\1750340068")

    print(r)
    print(r.history)
    # r.save()
    # loss = LossFunction(CustomLoss())
    # for i in range(10):
    #     loss(tensor([i]), tensor([i + i]))
    # loss.plot(show=True)
