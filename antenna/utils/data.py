
import os
import pickle
import shutil
import sys
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from torch.utils.data import Dataset, Subset
from loguru import logger
from hashlib import md5
from antenna.types import *

@overload
def make_hashable(item: Union[torch.Tensor, np.ndarray]) -> bytes: ...
@overload
def make_hashable(item: Union[list[Any], dict[Any, Any]]) -> tuple: ...
@overload
def make_hashable(item: set[Any]) -> frozenset: ...
@overload
def make_hashable(item: Hashable) -> Hashable: ...
def make_hashable(item: Any) -> Any:
    """將複雜資料結構遞迴地轉換為可 hash 的形式。"""
    if isinstance(item, (int, float, str, bytes, type(None))): return item
    if isinstance(item, torch.Tensor): return item.detach().cpu().numpy().tobytes()
    if isinstance(item, np.ndarray): return item.tobytes()
    if isinstance(item, (list, tuple)): return tuple(make_hashable(i) for i in item)
    if isinstance(item, (set, frozenset)): return frozenset(sorted(make_hashable(i) for i in item))
    if isinstance(item, dict): return tuple(sorted((k, make_hashable(v)) for k, v in item.items()))
    try:
        hash(item)
        return item
    except TypeError:
        raise TypeError(f"物件 {type(item).__name__} 不可 hash，且未在 make_hashable 中處理。")
    

class Data(Generic[DataType]):
    def __init__(self, data:DataType=None, *, name:str="data", rootdir:Union[Path, str]="./", suffix:str="data", load=True):
        self.data:DataType = data
        self.data_set = set()
        self.rootdir = Path(rootdir)
        self.name = name
        self.suffix = suffix
        self.savepath = self.rootdir.joinpath(f"{name}.{suffix}")
        self.temppath = self.rootdir.joinpath(f"{name}.{suffix}.tmp")
        self.logpath = self.rootdir.joinpath(f"{name}.{suffix}.log")

        if load and self.savepath.exists(): self.load()
    
    def save(self):
        self.rootdir.mkdir(parents=True, exist_ok=True)
        try:
            # 1. 將資料寫入一個暫存檔
            with open(self.temppath, 'wb') as f:
                pickle.dump(self.data, f)

            # 2. 用暫存檔覆蓋正式檔 (這一步操作非常快，幾乎不可能中斷)
            shutil.move(self.temppath, self.savepath)
        except Exception as e:
            if self.temppath.exists():
                self.temppath.unlink()
            raise RuntimeError(f"儲存檔案 '{self.savepath}' 失敗") from e
    
    def load(self) -> DataType:
        try:
            with open(self.savepath, 'rb') as f:
                self.data:DataType = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"載入失敗：找不到檔案 '{self.savepath}'")
        except (pickle.UnpicklingError, IOError) as e:
            raise RuntimeError(f"載入失敗：無法讀取或解析檔案 '{self.savepath}'") from e
        return self.data
        
    def update(self, data, save:bool=False):
        self.data = data
        if save: self.save()

    def add_set(self, item):
        hashable_item = make_hashable(item)
        self.data_set.add(hashable_item)
        return hashable_item
    
    def clear(self, default = None):
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
        data_bytes = pickle.dumps(self.make_hashable())
        return md5(data_bytes).hexdigest()
    
    @property
    def has_data(self) -> bool:
        return self.data is not None
    
    def make_hashable(self):
        """
        將一個可能包含不可 hash 型別的複雜資料結構，遞迴地轉換為
        一個完全可 hash 的結構。
        """
        return make_hashable(self.data)

    def backup(self):
        if not self.savepath.exists(): return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.rootdir.joinpath(f"{self.name}_{timestamp}.dataset.bak")
        shutil.copy2(self.savepath, backup_path)

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
        self.data_structure = None
        if self.data and isinstance(self.data[0], (list, tuple)):
            self.data_structure = (type(self.data[0]), len(self.data[0]))
        
        self.logger.info(f"DataManager '{self.name}' 初始化完成，共 {len(self.data)} 筆資料。")

    def load_data(self):
        """從 pickle 檔案載入資料，並建立一次性的查找 set。"""
        try:
            self.load()
            
            if self.data and isinstance(self.data[0], (list, tuple)):
                self.data_structure = (type(self.data[0]), len(self.data[0]))

            self.data_set = {make_hashable(item) for item in self.data}

            self.logger.success(f"成功從 '{self.savepath}' 載入 {len(self.data)} 筆資料，索引建立完成。")

        except Exception as e:
            self.logger.exception(f"載入資料時發生錯誤：{e}")

    def filter(self, filter_func: Callable[[Any], bool], *args, **kwargs) -> Subset:
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
            # 找出符合條件的索引
            # self.data[i] 是一個 (sample, label) 元組
            indices = [ 
                i for i, item in enumerate(self.data) 
                if filter_func(item, *args, **kwargs)
            ]
            
            subset = Subset(self, indices)
            # self.logger.success(f"子集建立完成。包含 {len(indices)} / {len(self.data)} 筆資料。")
            return subset

        except Exception as e:
            self.logger.error(f"建立過濾子集時發生錯誤：{e}")
            # 發生錯誤時回傳空子集或拋出異常，視需求而定
            return Subset(self, [])

    def add_and_save(self, new_data: list, mode='append'):
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
        items_to_process:list[Union[list[Tensor,Tensor],list[Any,Any]]] = new_data if is_batch else [new_data]

        # --- 檢查資料結構一致性 ---
        for item in items_to_process:
            if not isinstance(item, (list, tuple)):
                self.logger.error(f"資料項必須是 list 或 tuple，但收到了 {type(item)}。")
                return
            current_structure = (type(item), len(item))
            if self.data_structure is None and not self.data:
                self.data_structure = current_structure
                self.logger.debug(f"偵測到資料結構為：{self.data_structure[0].__name__} of length {self.data_structure[1]}。")
            elif self.data_structure != current_structure:
                self.logger.error(f"結構不符：新資料的結構 {current_structure} 與現有結構 {self.data_structure} 不符。")
                return

        if isinstance(items_to_process[0][0], Tensor):
            detached_data = [
                [t.detach() for t in inner_list]
                for inner_list in items_to_process
            ]
            items_to_process = detached_data

        if mode == 'overwrite':
            self.logger.warning(f"使用 'overwrite' 模式，將會清除所有現有資料。")
            self.backup()
            self.clear([])
        elif mode != 'append':
            self.logger.error(f"無效的模式 '{mode}'。請使用 'append' 或 'overwrite'。")
            return
        
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
            else:
                self.logger.info(f"發現 {num_duplicates} 筆重複資料，將予以忽略。")

            
        num_added = len(unique_new_data)
        self.data.extend(unique_new_data)
        
        try:
            self.save()
            self.logger.success(f"成功添加 {num_added} 筆新資料並儲存！ (目前共 {len(self.data)} 筆資料)")
        except RuntimeError as e:
            self.logger.error(f"儲存資料時發生錯誤：{e}")

    def clear_all_data(self):
        """清空記憶體中和檔案中的所有資料。執行前會先備份。"""
        self.logger.warning(f"即將刪除所有資料及檔案 '{self.savepath}'...")
        if self.savepath.exists(): self.backup()
        
        self.clear([])
        self.data_structure = None
        self.save() # 儲存空的列表
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
            if hasattr(sample, 'shape'):
                print(f"  - Shape:     {sample.shape}")
        print("--------------------------\n")

    def __len__(self):
        """Return the length of our dataset."""
        return len(self.data)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        """Working for indexing and automatically converted to PyTorch Tensors."""
        if not self.data:
            raise IndexError("資料集是空的，請先使用 .add_and_save() 添加資料。")
        
        # 1. 從列表中獲取原始資料
        try:
            sample, label = self.data[idx]
        except (ValueError, TypeError) as e:
            raise ValueError(f"索引 {idx} 的資料結構不符預期，應為 (sample, label) 的形式。錯誤: {e}")
        
        # 2. 應用使用者定義的轉換 (主要用於圖像增強等複雜操作)
        if self.transform:
            sample = self.transform(sample)
        
        # 3. 確保 sample 是 Tensor
        if not isinstance(sample, torch.Tensor):
            if isinstance(sample, np.ndarray):
                # 對於 numpy 陣列，使用 from_numpy 更高效
                sample = torch.from_numpy(sample)
            else:
                # 對於 list, int, float 等，使用 torch.tensor
                sample = torch.tensor(sample)
            
            # 通常特徵資料需要是浮點數類型
            sample = sample.to(dtype=torch.float32)

        # 4. 確保 label 是 Tensor
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label)
        
        # 根據標籤的類型，賦予合適的 dtype
        # 迴歸任務的標籤通常是 float32
        if label.is_floating_point():
            label = label.to(dtype=torch.float32)
        # 分類任務的標籤需要是 long (int64)，以用於 CrossEntropyLoss 等
        else:
            label = label.to(dtype=torch.int64)
            
        return sample, label


    def __repr__(self):
        return f"<DataManager name='{self.name}' items={len(self)} path='{self.savepath}'>"

@overload
def size_converter(
    sizer: Sizable,tensor: torch.Tensor, flatten:Literal[True], batch: Literal[True]
) -> Tensor_B_N: ...
@overload
def size_converter(
    sizer: Sizable,tensor: torch.Tensor, flatten:Literal[True], batch: Literal[False]
) -> Tensor_N: ...
@overload
def size_converter(
    sizer: Sizable,tensor: torch.Tensor, flatten:Literal[False], batch: Literal[True]
) -> Tensor_B_W_H: ...
@overload
def size_converter(
    sizer: Sizable,tensor: torch.Tensor, flatten:Literal[False], batch: Literal[False]
) -> Tensor_W_H: ...
@overload
def size_converter(
    sizer: Sizable,tensor: torch.Tensor, output_shape: str
) -> torch.Tensor: ...

def size_converter(
    sizer: Sizable,
    tensor: torch.Tensor, 
    flatten: bool = False, 
    batch: bool = False,
    output_shape: Optional[str] = None
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
    
    #* Batch size
    B_calc = total_input_numel // N_per_sample

    #* Use the string output_shape
    if output_shape is not None: 
        try:
            shape_parts = [part.strip() for part in output_shape.split(',')]
            final_shape_list = []
            has_batch_dim = False

            for part in shape_parts:
                if part == 'B':
                    final_shape_list.append(B_calc)
                    has_batch_dim = True
                elif part == 'N':
                    final_shape_list.append(N_per_sample)
                elif part == 'H':
                    final_shape_list.append(H_comp)
                elif part == 'W':
                    final_shape_list.append(W_comp)
                elif part.isdigit():
                    final_shape_list.append(int(part))
                else:
                    raise ValueError(f"'{part}'")
        
        except ValueError as e:
            raise ValueError(
                f"The string `output_shape` contains an unrecognized component: {e}."
                "Please only use 'B', 'N', 'H', 'W', or numbers."
            )
        
        #* Validate: batch dimension
        if not has_batch_dim and B_calc > 1:
            raise ValueError(
                f"輸入的計算批次大小為 {B_calc}, 但 output_shape "
                f"'{output_shape}' 中未包含 'B'。 "
                "無法壓縮非單例的批次維度。"
            )

        #* Validate: the final total number of elements
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

    else: #* Use the flatten and batch parameters
        if flatten:
            target_shape_per_sample = (N_per_sample,)
        else:
            target_shape_per_sample = components # (H_comp, W_comp)

        final_shape = (B_calc, *target_shape_per_sample)
        output_tensor = tensor.reshape(final_shape)

        if not batch:
            if B_calc == 1:
                #? (1, H, W) -> (H, W) or (1, N) -> (N,)
                return output_tensor.squeeze(dim=0)
            else:
                # B > 1, 但要求 non-batch output
                raise ValueError(
                    f"輸入的計算批次大小為 {B_calc}, 但請求了 'batch=False' "
                    "(非批次輸出)。無法壓縮非單例的批次維度。"
                )
        else: 
            #? (B, H, W) -> (B, H, W)
            return output_tensor
        
def dynamic_loss_filter(
    datas:Tuple[Tensor, Tensor], 
    lower: float = float('inf'), 
    upper: float = float('-inf'), 
) -> bool:
    """
    Example::

        DataManager.filter(minmax_filter, lower=TEMP('smaller', float('inf')), upper=TEMP('bigger', float('-inf')))
    """
    from antenna import MultiResponses
    _pattern, _response = datas
    # _pattern = AntennaPattern(_pattern)
    _response = MultiResponses(_response)
    _loss = _response.criterion().item()

    return _loss > upper or _loss < lower