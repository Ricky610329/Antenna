"""資料管理與常用 tensor 轉換工具。

本模組將純函式（hash、tensor ↔ numpy、pickle I/O、備份）抽為 module-level helpers，
`Data` / `DataManager` class 只負責協調與狀態管理，業務邏輯下放至 helpers，便於單元測試。
"""

import pickle
import shutil
import sys
from datetime import datetime
from hashlib import md5
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset, Subset

from antenna.types import *

# ----------------------------------------------------------------------------
# Hash helpers
# ----------------------------------------------------------------------------


@overload
def make_hashable(item: Union[torch.Tensor, np.ndarray]) -> bytes: ...
@overload
def make_hashable(item: Union[list[Any], dict[Any, Any]]) -> tuple: ...
@overload
def make_hashable(item: set[Any]) -> frozenset: ...
@overload
def make_hashable(item: Hashable) -> Hashable: ...
def make_hashable(item: Any) -> Any:
    """將複雜資料結構遞迴地轉換為可 hash 的形式。

    - ``Tensor`` / ``ndarray`` → ``bytes``（透過 ``tobytes()``）
    - ``list`` / ``tuple`` → 遞迴後 ``tuple``
    - ``set`` / ``frozenset`` → 排序後 ``frozenset``
    - ``dict`` → 依 key 排序後的 ``tuple`` of ``(key, make_hashable(value))``
    """
    if isinstance(item, (int, float, str, bytes, type(None))):
        return item
    if isinstance(item, torch.Tensor):
        return item.detach().cpu().numpy().tobytes()
    if isinstance(item, np.ndarray):
        return item.tobytes()
    if isinstance(item, (list, tuple)):
        return tuple(make_hashable(i) for i in item)
    if isinstance(item, (set, frozenset)):
        return frozenset(sorted(make_hashable(i) for i in item))
    if isinstance(item, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in item.items()))
    try:
        hash(item)
        return item
    except TypeError:
        raise TypeError(f"物件 {type(item).__name__} 不可 hash，且未在 make_hashable 中處理。")


def compute_data_id(item: Any) -> str:
    """使用 MD5 產生資料的 HASH 值。純函式，適合做快取 key。"""
    return md5(pickle.dumps(make_hashable(item))).hexdigest()


# ----------------------------------------------------------------------------
# Tensor ↔ Numpy helpers
# ----------------------------------------------------------------------------


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """將 Tensor 安全地轉為 numpy 陣列（detach、移到 CPU）。"""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"tensor_to_numpy 只接受 Tensor，收到 {type(tensor).__name__}。")
    return tensor.detach().cpu().numpy()


def numpy_to_tensor(array: np.ndarray, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """將 numpy 陣列轉為 Tensor。若指定 ``dtype`` 會同時轉型。"""
    if not isinstance(array, np.ndarray):
        raise TypeError(f"numpy_to_tensor 只接受 np.ndarray，收到 {type(array).__name__}。")
    tensor = torch.from_numpy(array)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def ensure_tensor(value: Any, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """將 value 轉為 Tensor（若已是 Tensor 則直接使用）。"""
    if isinstance(value, torch.Tensor):
        tensor = value
    elif isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value)
    else:
        tensor = torch.tensor(value)

    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


# ----------------------------------------------------------------------------
# Pickle I/O helpers
# ----------------------------------------------------------------------------


def atomic_pickle_save(path: Union[Path, str], data: Any, temp_path: Optional[Union[Path, str]] = None) -> None:
    """原子性地將 ``data`` 以 pickle 格式存到 ``path``。

    先寫入 ``temp_path``（預設為 ``<path>.tmp``）再 ``move``，避免中斷時毀損原檔。
    """
    path = Path(path)
    temp_path = Path(temp_path) if temp_path is not None else path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(temp_path, "wb") as f:
            pickle.dump(data, f)
        shutil.move(temp_path, path)
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"儲存檔案 '{path}' 失敗") from e


def pickle_load(path: Union[Path, str]) -> Any:
    """從 ``path`` 讀回 pickle 資料。"""
    path = Path(path)
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"載入失敗：找不到檔案 '{path}'")
    except (OSError, pickle.UnpicklingError) as e:
        raise RuntimeError(f"載入失敗：無法讀取或解析檔案 '{path}'") from e


def timestamped_backup(
    src_path: Union[Path, str], name: Optional[str] = None, suffix: str = "dataset.bak"
) -> Optional[Path]:
    """將 ``src_path`` 備份為 ``<name>_<timestamp>.<suffix>``，回傳備份路徑。

    若 ``src_path`` 不存在則回傳 ``None``。
    """
    src_path = Path(src_path)
    if not src_path.exists():
        return None

    name = name if name is not None else src_path.stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = src_path.parent.joinpath(f"{name}_{timestamp}.{suffix}")
    shutil.copy2(src_path, backup_path)
    return backup_path


# ----------------------------------------------------------------------------
# Data / DataManager
# ----------------------------------------------------------------------------


class Data(Generic[DataType]):
    def __init__(
        self,
        data: DataType = None,
        *,
        name: str = "data",
        rootdir: Union[Path, str] = "./",
        suffix: str = "data",
        load=True,
    ):
        self.data: DataType = data
        self.data_set = set()
        self.rootdir = Path(rootdir)
        self.name = name
        self.suffix = suffix
        self.savepath = self.rootdir.joinpath(f"{name}.{suffix}")
        self.temppath = self.rootdir.joinpath(f"{name}.{suffix}.tmp")
        self.logpath = self.rootdir.joinpath(f"{name}.{suffix}.log")

        if load and self.savepath.exists():
            self.load()

    def save(self):
        atomic_pickle_save(self.savepath, self.data, temp_path=self.temppath)

    def load(self) -> DataType:
        self.data: DataType = pickle_load(self.savepath)
        return self.data

    def update(self, data, save: bool = False):
        self.data = data
        if save:
            self.save()

    def add_set(self, item):
        hashable_item = make_hashable(item)
        self.data_set.add(hashable_item)
        return hashable_item

    def clear(self, default=None):
        self.data = default
        self.data_set.clear()

    def __eq__(self, other: "Data") -> bool:
        if not isinstance(other, Data):
            return NotImplemented
        return self.make_hashable() == other.make_hashable()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', path='{self.savepath}')"

    def __str__(self):
        return str(self.data)

    def __hash__(self) -> int:
        return hash(self.make_hashable())

    def __contains__(self, item: "Data") -> bool:
        hashable_item = make_hashable(item)
        return hashable_item in self.data_set

    @property
    def id(self) -> str:
        """使用 MD5 產生資料的 HASH 值"""
        return compute_data_id(self.data)

    @property
    def has_data(self) -> bool:
        return self.data is not None

    def make_hashable(self):
        """將 ``self.data`` 遞迴地轉換為可 hash 的結構。"""
        return make_hashable(self.data)

    def backup(self):
        timestamped_backup(self.savepath, name=self.name, suffix="dataset.bak")


class DataManager(Data[list[tuple[Tensor, Tensor]]], Dataset):
    """
    一個整合了 loguru 日誌功能的通用資料管理類別。

    主要功能:
    1. 支援逐筆或整批添加資料，並可選擇附加(append)或覆寫(overwrite)。
    2. 自動將資料儲存/載入至 Pickle 檔案中，方便持久化。
    3. 執行如清除或覆寫等破壞性操作前，會自動建立時間戳備份。
    4. 透過 verbose 參數控制 loguru 的日誌輸出等級。
    5. 提供 info() 方法快速檢視資料集狀態與內容。
    6. 內建資料結構一致性檢查，避免混入格式錯誤的資料。
    7. 繼承自 PyTorch Dataset，可無縫接軌 DataLoader 進行模型訓練。
    """

    def __init__(self, name, *, rootdir=".", transform=None, verbose=True):
        """
        初始化 DataManager。

        Args:
            name (str): 資料集的名稱，將作為檔名 (例如: 'training_data')。
            rootdir (str, optional): 儲存資料檔案的根目錄。預設為目前目錄。
            transform (callable, optional): 應用於樣本的轉換函式 (通常來自 torchvision.transforms)。
            verbose (bool, optional): 是否印出詳細的操作訊息 (INFO 等級以上)。預設為 True。
        """
        super().__init__(data=[], name=name, rootdir=Path(rootdir), suffix="dataset", load=True)

        self.transform = transform

        # 設定 loguru
        self.logger = logger
        log_level = "INFO" if verbose else "WARNING"
        self.logger.remove()
        self.logger.add(sys.stdout, level=log_level)
        self.logger.add(self.logpath, level="INFO", rotation="10 MB")

        # 根據已載入的 self.data 初始化輔助屬性
        for item in self.data:
            self.add_set(item)
        self.data_structure = self._detect_structure(self.data)

        self.logger.info(f"DataManager '{self.name}' 初始化完成，共 {len(self.data)} 筆資料。")

    @staticmethod
    def _detect_structure(data: list) -> Optional[tuple]:
        """偵測資料集首筆資料的結構，回傳 ``(type, len)`` 或 ``None``。"""
        if data and isinstance(data[0], (list, tuple)):
            return (type(data[0]), len(data[0]))
        return None

    def load_data(self):
        """從 pickle 檔案載入資料，並建立一次性的查找 set。"""
        try:
            self.load()
            self.data_structure = self._detect_structure(self.data)
            self.data_set = {make_hashable(item) for item in self.data}
            self.logger.success(f"成功從 '{self.savepath}' 載入 {len(self.data)} 筆資料，索引建立完成。")
        except Exception as e:
            self.logger.exception(f"載入資料時發生錯誤：{e}")

    def filter(self, filter_func: Callable[[tuple[Tensor, Tensor]], bool], *args, **kwargs) -> Subset:
        """
        根據過濾條件建立並回傳一個資料子集 (Subset)。
        此方法不會改變 DataManager 本身的狀態。

        Args:
            filter_func (Callable): 接受單筆資料 (sample, label) 並回傳 bool 的函式。

        Returns:
            torch.utils.data.Subset: 包含符合條件資料的子集物件。
        """
        if not self.data:
            self.logger.warning("資料集為空，回傳空子集。")
            return Subset(self, [])

        self.logger.info("正在計算過濾條件並建立子集...")
        try:
            indices = [i for i, item in enumerate(self.data) if filter_func(item, *args, **kwargs)]
            return Subset(self, indices)
        except Exception as e:
            self.logger.error(f"建立過濾子集時發生錯誤：{e}")
            return Subset(self, [])

    def add_and_save(self, new_data: list, mode="append"):
        """
        添加新資料到資料集並更新 pickle 檔案。

        Args:
            new_data (list or tuple):
                - 單筆資料: 格式為 [x, y] 或 (x, y)。
                - 多筆資料: 格式為 [[x1, y1], [x2, y2], ...]。
            mode (str, optional): 寫入模式，'append' (附加) 或 'overwrite' (覆寫)。預設為 'append'。
                在 'append' 模式下會高效過濾重複的資料。
        """
        if not isinstance(new_data, list):
            self.logger.error("輸入資料必須是 list。")
            return
        if not new_data:
            self.logger.warning("輸入的資料是空的，不執行任何操作。")
            return

        is_batch = isinstance(new_data[0], (list, tuple))
        items_to_process: list = new_data if is_batch else [new_data]

        # --- 檢查資料結構一致性 ---
        if not self._validate_structure(items_to_process):
            return

        # --- 處理 Tensor detach ---
        if isinstance(items_to_process[0][0], Tensor):
            items_to_process = [[t.detach() for t in inner_list] for inner_list in items_to_process]

        # --- 處理 mode ---
        if mode == "overwrite":
            self.logger.warning("使用 'overwrite' 模式，將會清除所有現有資料。")
            self.backup()
            self.clear([])
        elif mode != "append":
            self.logger.error(f"無效的模式 '{mode}'。請使用 'append' 或 'overwrite'。")
            return

        # --- 去重 ---
        unique_new_data = []
        for item in items_to_process:
            if item not in self:
                unique_new_data.append(item)
                self.add_set(item)

        num_duplicates = len(items_to_process) - len(unique_new_data)
        if num_duplicates > 0:
            if not unique_new_data:
                self.logger.info(f"所有 ({num_duplicates} 筆) 待加入的資料皆已存在，不執行任何操作。")
                return
            self.logger.info(f"發現 {num_duplicates} 筆重複資料，將予以忽略。")

        num_added = len(unique_new_data)
        self.data.extend(unique_new_data)

        try:
            self.save()
            self.logger.success(f"成功添加 {num_added} 筆新資料並儲存！ (目前共 {len(self.data)} 筆資料)")
        except RuntimeError as e:
            self.logger.error(f"儲存資料時發生錯誤：{e}")

    def _validate_structure(self, items: list) -> bool:
        """檢查 items 的結構是否全部一致且符合 self.data_structure。"""
        for item in items:
            if not isinstance(item, (list, tuple)):
                self.logger.error(f"資料項必須是 list 或 tuple，但收到了 {type(item)}。")
                return False
            current_structure = (type(item), len(item))
            if self.data_structure is None and not self.data:
                self.data_structure = current_structure
                self.logger.debug(
                    f"偵測到資料結構為：{self.data_structure[0].__name__} of length {self.data_structure[1]}。"
                )
            elif self.data_structure != current_structure:
                self.logger.error(f"結構不符：新資料的結構 {current_structure} 與現有結構 {self.data_structure} 不符。")
                return False
        return True

    def clear_all_data(self):
        """清空記憶體中和檔案中的所有資料。執行前會先備份。"""
        self.logger.warning(f"即將刪除所有資料及檔案 '{self.savepath}'...")
        if self.savepath.exists():
            self.backup()

        self.clear([])
        self.data_structure = None
        self.save()
        self.logger.info("所有資料已清除。")

    def info(self):
        """印出資料集的摘要資訊。"""
        print("\n--- DataManager Info ---")
        print(f"Name:          {self.name}")
        print(f"File Path:     {self.savepath}")
        print(f"Total Items:   {len(self)}")
        print(f"Hashed Items:  {len(self.data_set)}")
        if self.data_structure:
            struct_type = self.data_structure[0].__name__
            struct_len = self.data_structure[1]
            print(f"Data Structure: {struct_type} of length {struct_len}")
        else:
            print("Data Structure: 尚未確定 (資料集為空)")
        if self.data:
            sample, _ = self.data[0]
            print(f"First Sample:  {type(sample)}")
            if hasattr(sample, "shape"):
                print(f"  - Shape:     {sample.shape}")
        print("--------------------------\n")

    def __len__(self):
        """Return the length of our dataset."""
        return len(self.data)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        """Working for indexing and automatically converted to PyTorch Tensors."""
        if not self.data:
            raise IndexError("資料集是空的，請先使用 .add_and_save() 添加資料。")

        try:
            sample, label = self.data[idx]
        except (ValueError, TypeError) as e:
            raise ValueError(f"索引 {idx} 的資料結構不符預期，應為 (sample, label) 的形式。錯誤: {e}")

        # 使用者定義的轉換（主要用於圖像增強等複雜操作）
        if self.transform:
            sample = self.transform(sample)

        # 確保 sample 是 Tensor（特徵通常需要 float32）
        if not isinstance(sample, torch.Tensor):
            sample = ensure_tensor(sample, dtype=torch.float32)

        # 確保 label 是 Tensor；依 dtype 自動對應 float32（迴歸）或 int64（分類）
        if not isinstance(label, torch.Tensor):
            label = ensure_tensor(label)
        if label.is_floating_point():
            label = label.to(dtype=torch.float32)
        else:
            label = label.to(dtype=torch.int64)

        return sample, label

    def __repr__(self):
        return f"<DataManager name='{self.name}' items={len(self)} path='{self.savepath}'>"


# ----------------------------------------------------------------------------
# size_converter
# ----------------------------------------------------------------------------


@overload
def size_converter(
    sizer: Sizable, tensor: torch.Tensor, flatten: Literal[True], batch: Literal[True]
) -> Tensor_B_N: ...
@overload
def size_converter(sizer: Sizable, tensor: torch.Tensor, flatten: Literal[True], batch: Literal[False]) -> Tensor_N: ...
@overload
def size_converter(
    sizer: Sizable, tensor: torch.Tensor, flatten: Literal[False], batch: Literal[True]
) -> Tensor_B_W_H: ...
@overload
def size_converter(
    sizer: Sizable, tensor: torch.Tensor, flatten: Literal[False], batch: Literal[False]
) -> Tensor_W_H: ...
@overload
def size_converter(sizer: Sizable, tensor: torch.Tensor, output_shape: str) -> torch.Tensor: ...


def size_converter(
    sizer: Sizable, tensor: torch.Tensor, flatten: bool = False, batch: bool = False, output_shape: Optional[str] = None
) -> torch.Tensor:
    """
    General-purpose tensor size converter.

    Mode 1 (output_shape is None):
        Transformation using the `flatten` and `batch` parameters.

    Mode 2 (output_shape is a string):
        Ignore the `flatten` and `batch` parameters,
        and perform the transformation exactly as indicated by the `output_shape` string.

    Args:
        tensor (torch.Tensor): The input tensor to be transformed.
        sizer (Sizable): An object or class that has a `.size(flatten: bool)` method

        flatten (bool): Only output_shape is None.
            True - 輸出形狀為 (B, N) 或 (N,)。
            False - 輸出形狀為 (B, H, W) 或 (H, W)。
        batch (bool): Only output_shape is None.
            True - 強制輸出為批次形式 (B, ...)，即使 B=1。
            False - 如果計算出的 B=1，則移除批次維度 (...,)。
        output_shape (Optional[str]): [B, H, W, N] Priority use. EX: "B, 1, H, W" or "B, N, 1"

    Returns:
        torch.Tensor: The reshaped tensor.
    """
    try:
        N_per_sample = sizer.size(flatten=True)
        components = sizer.size(flatten=False)
        H_comp = components[0]
        W_comp = components[1]
    except Exception as e:
        raise ValueError(f"Unable to obtain size information from sizer({sizer})\n{e}")

    total_input_numel = tensor.numel()
    if total_input_numel % N_per_sample != 0:
        raise ValueError(
            f"The total number of elements in the input tensor ({total_input_numel}) "
            f"must be an integer multiple of {N_per_sample}."
        )

    # * Batch size
    B_calc = total_input_numel // N_per_sample

    # * Use the string output_shape
    if output_shape is not None:
        return _reshape_by_output_spec(
            tensor,
            output_shape=output_shape,
            B_calc=B_calc,
            N_per_sample=N_per_sample,
            H_comp=H_comp,
            W_comp=W_comp,
            total_input_numel=total_input_numel,
        )

    # * Use the flatten and batch parameters
    target_shape_per_sample = (N_per_sample,) if flatten else components
    final_shape = (B_calc, *target_shape_per_sample)
    output_tensor = tensor.reshape(final_shape)

    if batch:
        return output_tensor
    if B_calc == 1:
        return output_tensor.squeeze(dim=0)
    raise ValueError(f"輸入的計算批次大小為 {B_calc}, 但請求了 'batch=False' (非批次輸出)。無法壓縮非單例的批次維度。")


def _reshape_by_output_spec(
    tensor: torch.Tensor,
    *,
    output_shape: str,
    B_calc: int,
    N_per_sample: int,
    H_comp: int,
    W_comp: int,
    total_input_numel: int,
) -> torch.Tensor:
    """依字串規格（B, N, H, W, 數字）重塑 tensor。"""
    mapping = {"B": B_calc, "N": N_per_sample, "H": H_comp, "W": W_comp}
    try:
        shape_parts = [part.strip() for part in output_shape.split(",")]
        final_shape_list = []
        has_batch_dim = False
        for part in shape_parts:
            if part in mapping:
                final_shape_list.append(mapping[part])
                if part == "B":
                    has_batch_dim = True
            elif part.isdigit():
                final_shape_list.append(int(part))
            else:
                raise ValueError(f"'{part}'")
    except ValueError as e:
        raise ValueError(
            f"The string `output_shape` contains an unrecognized component: {e}."
            "Please only use 'B', 'N', 'H', 'W', or numbers."
        )

    if not has_batch_dim and B_calc > 1:
        raise ValueError(
            f"輸入的計算批次大小為 {B_calc}, 但 output_shape '{output_shape}' 中未包含 'B'。 無法壓縮非單例的批次維度。"
        )

    target_numel = 1
    for dim in final_shape_list:
        target_numel *= dim
    if target_numel != total_input_numel:
        raise ValueError(
            f"Output shape '{output_shape}' (解析為 {final_shape_list}) "
            f"的總元素量 ({target_numel}) 與 "
            f"輸入張量的總元素量 ({total_input_numel}) 不匹配。"
        )

    return tensor.reshape(final_shape_list)


def dynamic_loss_filter(
    datas: Tuple[Tensor, Tensor],
    lower: float = float("inf"),
    upper: float = float("-inf"),
) -> bool:
    """
    Example::

        DataManager.filter(minmax_filter, lower=TEMP('smaller', float('inf')), upper=TEMP('bigger', float('-inf')))
    """
    from antenna import MultiResponses

    _pattern, _response = datas
    _response = MultiResponses(_response)
    _loss = _response.criterion().item()

    return _loss > upper or _loss < lower
