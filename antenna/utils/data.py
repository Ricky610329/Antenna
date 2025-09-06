
import os
import pickle
import shutil
import sys
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from torch.utils.data import Dataset
from loguru import logger

def make_hashable(item):
        """
        將資料項 (list/tuple) 遞迴轉換為完全可雜湊的 tuple。
        - torch.Tensor 和 np.ndarray 會被轉換成 bytes。
        - list 會被遞迴轉換成 tuple。
        """
        if isinstance(item, torch.Tensor):
            return item.cpu().numpy().tobytes()
        if isinstance(item, (tuple, list)):
            return tuple(make_hashable(i) for i in item)
        if isinstance(item, np.ndarray):
            return item.tobytes()
        return item

class DataManager(Dataset):
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
        self.name = name
        self.rootdir = Path(rootdir)
        self.rootdir.mkdir(parents=True, exist_ok=True) # 確保根目錄存在
        self.pickle_path = self.rootdir.joinpath(f"{self.name}.dataset")
        self.temp_path = self.rootdir.joinpath(f"{self.name}.dataset.tmp")
        self.transform = transform
        
        self.data = []
        self.data_set = set() # 用於快速重複檢查的集合
        self.data_structure = None

        self.logger = logger
        log_level = "INFO" if verbose else "WARNING"
        self.logger.remove()
        self.logger.add(sys.stdout, level=log_level)
        self.logger.add(self.rootdir.joinpath(f"{name}.dataset.log"), level="INFO", rotation="10 MB")

        if self.pickle_path.exists():
            self.logger.info(f"找到已存在的檔案，正在從 '{self.pickle_path}' 載入資料...")
            self.load_data()
        else:
            self.logger.info(f"資料檔案 '{self.pickle_path}' 尚不存在，將在首次添加資料時建立。")

    def load_data(self):
        """從 pickle 檔案載入資料，並建立一次性的查找 set。"""
        try:
            with open(self.pickle_path, 'rb') as f:
                self.data = pickle.load(f)
            
            if self.data and isinstance(self.data[0], (list, tuple)):
                self.data_structure = (type(self.data[0]), len(self.data[0]))

            self.data_set = {make_hashable(item) for item in self.data}

            self.logger.success(f"成功從 '{self.pickle_path}' 載入 {len(self.data)} 筆資料，索引建立完成。")

        except Exception as e:
            self.logger.exception(f"載入資料時發生錯誤：{e}")

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
        items_to_process = new_data if is_batch else [new_data]

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

        if mode == 'overwrite':
            self.logger.warning(f"使用 'overwrite' 模式，將會清除所有現有資料。")
            self.backup()
            self.data = []
            self.data_set.clear()
        elif mode != 'append':
            self.logger.error(f"無效的模式 '{mode}'。請使用 'append' 或 'overwrite'。")
            return
        
        unique_new_data = []
        for item in items_to_process:
            hashable_item = make_hashable(item)
            if hashable_item not in self.data_set:
                unique_new_data.append(item)
                self.data_set.add(hashable_item)
        
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
            self.save_data(self.data)
            self.logger.success(f"成功添加 {num_added} 筆新資料並儲存！ (目前共 {len(self.data)} 筆資料)")
        except RuntimeError as e:
            self.logger.error(f"儲存資料時發生錯誤：{e}")

    def save_data(self, datas):
        """使用安全的原子寫入方式儲存資料。"""
        try:
            # 1. 將資料寫入一個暫存檔
            with open(self.temp_path, 'wb') as f:
                pickle.dump(datas, f)

            # 2. 用暫存檔覆蓋正式檔 (這一步操作非常快，幾乎不可能中斷)
            shutil.move(self.temp_path, self.pickle_path)
        except Exception as e:
            if self.temp_path.exists():
                self.temp_path.unlink()
            raise RuntimeError(f"儲存檔案失敗: {e}")

    def clear_all_data(self):
        """清空記憶體中和檔案中的所有資料。執行前會先備份。"""
        self.logger.warning(f"即將刪除所有資料及檔案 '{self.pickle_path}'...")
        if self.pickle_path.exists():
            self.backup()
            self.pickle_path.unlink()
        self.data = []
        self.data_set.clear()
        self.data_structure = None
        self.logger.info("所有資料已清除。")

    def backup(self):
        """備份目前的資料檔案。"""
        if not self.pickle_path.exists():
            self.logger.info("找不到資料檔案，無需備份。")
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.rootdir.joinpath(f"{self.name}_{timestamp}.dataset.bak")
        try:
            shutil.copy2(self.pickle_path, backup_path)
            self.logger.info(f"資料已成功備份至 '{backup_path}'")
        except Exception as e:
            self.logger.error(f"備份時發生錯誤：{e}")
    
    def info(self):
        """印出資料集的摘要資訊。"""
        print("\n--- DataManager Info ---")
        print(f"Name:          {self.name}")
        print(f"File Path:     {self.pickle_path}")
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
        return f"<DataManager name='{self.name}' items={len(self)} path='{self.pickle_path}'>"