"""AntennaResponse、MultiResponses、TargetResponse 核心類別。"""

from collections import defaultdict
from functools import partial
from types import FunctionType

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from antenna.types import *
from antenna.utils.config import config
from antenna.utils.data import size_converter
from antenna.utils.torch_utils import concat, cTensor, stack, tensor


def mult(_ob):
    _result = 1
    for i in _ob:
        _result *= i
    return _result


class MultiResponses:
    def __init__(self, responses: Union[dict, Tensor] = None):
        _responses: Dict[str, AntennaResponse] = {}
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

    def size_converter(self, flatten: bool = False, batch: bool = False, output_shape=None) -> torch.Tensor:
        """:param output_shape: Priority use. (B, H, W, N) EX: "B, 1, H, W" or "B, N, 1" """
        return size_converter(AntennaResponse, self.to_list(), flatten=flatten, batch=batch, output_shape=output_shape)

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
        self.metadata: dict[str, dict] = defaultdict(dict)

    def __getitem__(self, key):
        """
        Target Response Design.

        Use `setTargetResponse()` before use, otherwise use the default value
        """
        if key not in self._note.keys():
            raise RuntimeError(
                f"The {key} of TargetResponse is not registered. Please use `registerTargetResponse()` first."
            )
        return super().__getitem__(key)

    def __call__(
        self,
        side: float,
        center: float,
        width: Tuple[int, int, int, int, int],
        label: str = "response",
        add: bool = False,
    ) -> Tensor:
        """
        Target Response Design.

        :param side: The Y value at both ends of the response.
        :param center: The y value of the center point of the response.

        :return: AntennaResponse

        """
        if len(width) != 5:
            raise ValueError(f"Expected 5 width, but got {len(width)}")
        mask_up = np.concatenate(
            [
                np.ones(width[0]) * side,
                np.linspace(side, center, width[1]),
                np.ones(width[2]) * center,
                np.linspace(center, side, width[3]),
                np.ones(width[4]) * side,
            ]
        )
        expected_response = tensor(np.array(mask_up), dtype=torch.float32, device=config.device)

        if add:
            self[label] = expected_response
            self._note[label] = f"side={side}, center={center}, width={width}"
            self.metadata[label].update(
                {
                    "response": expected_response,
                    "side": side,
                    "center": center,
                    "width": width,
                    "note": f"side={side}, center={center}, width={width}",
                }
            )

        return expected_response

    def register_loss_fn(self, label, loss_fn, **loss_fn_param):
        self.metadata[label].update(
            {
                "loss_fn": partial(loss_fn, **loss_fn_param),
                "loss_fn_name": loss_fn.__name__ if isinstance(loss_fn, FunctionType) else loss_fn.__class__.__name__,
            }
        )

    def loss_fn(self, label) -> Callable[..., Tensor]:
        return self.metadata[label]["loss_fn"]

    @property
    def labels(self):
        return list(self.metadata.keys())

    @labels.setter
    def labels(self, labels: Iterable[str]):
        for label in labels:
            _ = self.metadata[label]

    def concat(self):
        _result = super().concat()
        if _result.size(0) != AntennaResponse.size(flatten=True):
            raise RuntimeError(
                "The concat size does not match the set size. "
                "Please check `AntennaResponse.registerLabels()`"
                f"\n{_result.size(0)} != {AntennaResponse.size(flatten=True)}{AntennaResponse.size()}"
            )
        return _result

    def __str__(self):
        _ = " ".join(
            [f"{key}({value['note']}, loss={value.get('loss_fn_name', None)})" for key, value in self.metadata.items()]
        )
        return f"TargetResponse({_})"


class AntennaResponse(Generic[LossParams]):
    """
    Antenna Response Design.

    Attributes:
        response (Tensor): response
        target (TargetResponse): target response
    """

    x_patch_n257 = np.linspace(24, 32, 17)  # ? 26.5 - 28 - 29.5
    x_ris = np.linspace(0, 360, 361)

    target = TargetResponse()

    @overload
    def __new__(cls, response: Tensor) -> "AntennaResponse": ...
    @overload
    def __new__(cls, responses: Dict) -> "MultiResponses": ...

    def __new__(cls, response):
        if isinstance(response, cls):
            return response
        elif isinstance(response, Dict):
            return MultiResponses(response)
        else:
            return super().__new__(cls)

    def __init__(self, response: Union[Tensor, Dict]):
        """
        Antenna Response Design.

        Args:
            response: Response of the antenna.

        Raises:
            TypeError: If the response is not a tensor.

        """
        if isinstance(response, (AntennaResponse, Dict)):
            return
        elif isinstance(response, Tensor):
            response = response.to(config.device)
        else:
            raise TypeError(f"Expected Tensor, but got {type(response)}")

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

    def plot(self, label, axes: Optional[Axes] = None, show: bool = False):
        ax: Axes = plt.axes(axes)  # type: ignore
        ax.set_title("Antenna Response")
        ax.plot(self.target[label].response.cpu().detach(), color="red", label="Target")
        ax.plot(self.response.cpu().detach(), color="blue", label="Simulation")
        ax.legend()
        if show:
            plt.show()
        return ax

    @classmethod
    def registerLabels(cls, *labels: str, x: Union[tuple[int, int, int], Literal["ris", "n257"]] = "ris") -> Tensor:
        """
        The loss will be calculated from the registered labels.

        :param x: (start, stop, total)
        """
        match x:
            case "ris":
                x = (0, 360, 361)
            case "n257":  # ? 26.5 - 28 - 29.5
                x = (24, 32, 17)
            case _:
                pass

        cls.target.labels = labels
        cls.labels = labels
        cls._x = x

    @classmethod
    def x(cls):
        """Get the x-axis value of this response."""
        if not hasattr(cls, "_x"):
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
    def size(cls, flatten: bool = False):
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
    def registerTargetResponse(
        cls, side: float, center: float, width: Tuple[int, int, int, int, int], label: str = "response"
    ) -> Tensor:
        """
        Target Response Design.

        :param side: The Y value at both ends of the response.
        :param center: The y value of the center point of the response.

        :return: AntennaResponse

        """
        if not cls.target.labels:
            raise RuntimeError("No labels registered. Please use `registerLabels()` first.")
        is_add = label in cls.target.labels
        return cls.target(side, center, width, label=label, add=is_add)

    @classmethod
    def registerLossHook(
        cls, loss_hook: Callable[LossParams, Tensor], label: str = "response", **loss_hook_param: LossParams.kwargs
    ):
        """
        Args:

            loss_hook: Used for `criterion()`

            ```
            def criterion(response, target_response, ...):...
            ```

        """
        cls.target.register_loss_fn(label, loss_hook, **loss_hook_param)

    def criterion(self, label: str = "response", **param: LossParams.kwargs) -> Tensor:
        """[Loss Function] Register LossHook using `registerLossHook()` before use."""
        if label not in self.target.labels:
            raise RuntimeError(f"The {label} of LossHook is not registered. Please use `registerLossHook()` first.")

        return self.target.loss_fn(label)(self.response, **param)
