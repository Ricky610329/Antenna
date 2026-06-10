"""
antenna/types.py — 專案共用型別定義（瘦身版）。

只保留「有消費者」的型別：形狀語意別名、兩個介面/結構型別、Tensor/Axes 轉手。
過去 250+ 行的 TypeVar/ParamSpec/Generic 體操已隨各模組去泛型化一併移除——
型別註解在本專案的定位是「輕量文件」，不做靜態型別檢查。
"""
from __future__ import annotations

from typing import (
    Hashable as _Hashable,
    Protocol, Tuple, TypeAlias, TypedDict, TypeVar, Union,
)

from numpy import ndarray
from torch import Tensor
from matplotlib.axes._axes import Axes  # type: ignore  (re-export，供繪圖相關型別標註)

# ── TypeVar（legacy data.py 的泛型容器仍使用）──────────────────────────────
DataType = TypeVar('DataType')                    # Data[DataType].data 的內容型別
Hashable = TypeVar('Hashable', bound=_Hashable)   # 可作 dict 鍵的可雜湊型別


# ── 介面：能回報自身尺寸的物件（AntennaPattern / AntennaResponse / spec）────
class Sizable(Protocol):
    """物件須提供 .size() 方法（size_converter 以此查詢資料的空間維度）。"""
    def size(self, flatten: bool = False) -> Union[int, Tuple[int, ...]]: ...


# ── Tensor 形狀語意別名（底層皆為 torch.Tensor，僅作文件提示）────────────────
# B=Batch, N=攤平特徵數, W=寬, H=高
Tensor_B_N: TypeAlias = Tensor      # (B, N)   批次攤平
Tensor_B_W_H: TypeAlias = Tensor    # (B, W, H) 批次影像
Tensor_N: TypeAlias = Tensor        # (N,)     單樣本攤平
Tensor_W_H: TypeAlias = Tensor      # (W, H)   單樣本影像


# ── 饋電可達性 (FeedReachability) 單次評估結果 ──────────────────────────────
class FeedReachabilityDictType(TypedDict):
    feed_positions: list
    """饋入點"""
    rate: float
    """電流導通率"""
    mask: ndarray
    """電流導通的遮罩"""
    pattern: ndarray
    title: str
