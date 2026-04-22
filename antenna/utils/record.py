import sys
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
from pickle import dump as _pickle_dump
from pickle import load as _pickle_load
from typing import (
    Any,
    TypeVar,
    overload,
)

import numpy as np
import torch
from numpy import ndarray
from pandas import DataFrame
from torch import Tensor

from antenna.utils.path import Path

ReturnType = TypeVar("ReturnType")


class Record:
    def __init__(self, name: str = "record", rootdir: str | None = None, load: bool = False):
        self._data: dict[str, list] = defaultdict(list)
        self._history = defaultdict(list)
        self.name = name
        self.path = Path(rootdir or "./").joinpath(f"{name}.record")

        if load:
            self.load()

    def __call__(self, key, default=None, *, append=False):
        """Get the last value of key."""
        return self.end(key, default, append=append)

    def __setitem__(self, key, value):
        self._data[key].append(value)

    def __getitem__(self, key):
        """Get the complete array of keys."""
        if self.__contains__(key):
            return self._data[key]
        else:
            _keys = ", ".join(self._data.keys())
            raise KeyError(f"{key} does not exist. (Current key: {_keys})")

    def __delitem__(self, key):
        del self._data[key]

    def __contains__(self, item: str):
        return item in self._data

    def state_dict(self) -> dict[str, dict[str, list]]:
        """Return the state of the Record as a dict."""
        return {  # Convert to a normal dict.
            "_data": dict(self._data),
            "_history": dict(self._history),
        }

    def load_state_dict(self, state_dict: dict[str, dict[str, list]]):
        """Load the Record state."""
        loaded_data = state_dict.get("_data", {})
        loaded_history = state_dict.get("_history", {})

        self._data = defaultdict(list, loaded_data)
        self._history = defaultdict(list, loaded_history)

    def end(self, key, default=None, *, append=False):
        values = self._data.get(key)
        if values:
            return values[-1]
        if append:
            self._data[key].append(default)
            return default
        return default

    def add(self, key, num, default=None):
        """
        add('a', 1):
        a += 1
        """
        self.__setitem__(key, self.end(key, default) + num)

    def save(self, description: str | None = None):
        self._history["time"].append(str(datetime.now()).split(".")[0])
        self._history["description"].append(description or "No description")
        self._history["len"].append(len(self))

        current_state = self.state_dict()
        with open(str(self.path), "wb") as f:
            _pickle_dump(current_state, file=f)

    def load(self):
        if not self.path.exists():
            self.save()
        with open(str(self.path), "rb") as f:
            loaded_state = _pickle_load(f)
        self.load_state_dict(loaded_state)

        return self._data

    def average(self, key: str):
        _key_datas = self._data[key]
        _key_datas_len = len(_key_datas)
        if _key_datas_len > 0:
            return sum(_key_datas) / _key_datas_len
        else:
            return None

    def index(self, key: str, value, *, start: int = 0, stop: int = sys.maxsize) -> int | None:
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
        if key not in self._data:
            return None

        if isinstance(value, ndarray):
            _result = [np.array_equal(value, x) for x in self[key][start:stop]]
        elif isinstance(value, Tensor):
            _result = [torch.equal(value, x) for x in self[key][start:stop]]
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
    def find(self, key, value, other_keys: str, *, start=0, stop=sys.maxsize) -> Any | None: ...
    @overload
    def find(self, key, value, other_keys: tuple[str, ...], *, start=0, stop=sys.maxsize) -> list[Any] | None: ...

    def find(self, key, value, other_keys, *, start=0, stop=sys.maxsize):
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

        # 取 patience 視窗之前的整體最佳值，與最近一段比較。
        best_func = max if is_maximize else min
        split = len(values) - patience
        best_so_far = best_func(values[:split])
        recent_values = values[split:]

        # 最大化時「退步」代表 <= best；最小化時代表 >= best。
        if is_maximize:
            return all(v <= best_so_far for v in recent_values)
        return all(v >= best_so_far for v in recent_values)

    def reset(self, key: str | None = None, delete: bool = False):
        if key is not None:
            if delete:
                self._data.pop(key, None)
            else:
                self._data[key] = []
        else:
            self._data = defaultdict(list)

    def custom(self, key: str, fn: Callable[[list], ReturnType], *, default=None) -> ReturnType | None:
        _key_data = self._data[key]
        if _key_data:
            return fn(_key_data)
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
        _str = ""
        for key, value in self._data.items():
            _str += f"{key}[{len(value)}] "

        return f"Record({self.name}: {_str})"

    def __len__(self):
        return len(self.dataframe)
