
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
        self.data_structure = None # 用於檢查資料結構 ([特徵, 標籤] -> tuple, 2)

        self.logger = logger
        log_level = "INFO" if verbose else "WARNING"

        self.logger.remove()
        self.logger.add(sys.stdout, level=log_level)
        self.logger.add(
            self.rootdir.joinpath(f"{name}.dataset.log"),
            level = "INFO",
        )

        # 初始化時，如果 pickle 檔案已存在，就直接載入
        if self.pickle_path.exists():
            self.logger.info(f"找到已存在的檔案，正在從 '{self.pickle_path}' 載入資料...")
            self.load_data()
        else:
            self.logger.info(f"資料檔案 '{self.pickle_path}' 尚不存在，將在首次添加資料時建立。")

    def add_and_save(self, new_data:list, mode='append'):
        """
        添加新資料到資料集並更新 pickle 檔案。

        Args:
            new_data (list or tuple):
                - 單筆資料: 格式為 [x, y] 或 (x, y)。
                - 多筆資料: 格式為 [[x1, y1], [x2, y2], ...]。
            mode (str, optional): 寫入模式，'append' (附加) 或 'overwrite' (覆寫)。預設為 'append'。
        """
        if not isinstance(new_data, list):
            self.logger.error("輸入資料必須是 list。")
            return
        if not new_data:
            self.logger.warning("輸入的資料是空的，不執行任何操作。")
            return

        is_batch = isinstance(new_data[0], (list, tuple))
        items_to_check = new_data if is_batch else [new_data]
        
        # --- 檢查資料結構一致性 ---
        for item in items_to_check:
            if not isinstance(item, (list, tuple)):
                 self.logger.error(f"資料項必須是 list 或 tuple，但收到了 {type(item)}。")
                 return
            
            current_structure = (type(item), len(item))
            if self.data_structure is None and not self.data:
                self.data_structure = current_structure
                # 使用 DEBUG 等級，因為這通常只在第一次設定時重要
                self.logger.debug(f"偵測到資料結構為：{self.data_structure[0].__name__} of length {self.data_structure[1]}。")
            elif self.data_structure != current_structure:
                self.logger.error(f"結構不符：新資料的結構 {current_structure} 與現有結構 {self.data_structure} 不符。")
                return

        if mode == 'overwrite':
            self.logger.warning(f"使用 'overwrite' 模式，將會清除所有現有資料。")
            self.backup()
            self.data = []
        elif mode != 'append':
            self.logger.error(f"無效的模式 '{mode}'。請使用 'append' 或 'overwrite'。")
            return
        
        num_added = len(new_data) if is_batch else 1
        # self.logger.info(f"正在添加 {num_added} 筆資料...")
        
        if is_batch:
            self.data.extend(new_data)
        else:
            self.data.append(new_data)
        
        # self.logger.info(f"正在將全部 {len(self.data)} 筆資料儲存回 '{self.pickle_path}'...")
        try:
            self.save_data(self.data)
            self.logger.success(f"添加 {num_added} 筆資料與儲存完成！ (目前共 {len(self.data)} 筆資料)")
        except RuntimeError as e:
            self.logger.error(f"儲存資料時發生錯誤：{e}")
    def save_data(self, datas):
        """
        使用安全的原子寫入方式儲存資料。
        """
        try:
            # 1. 將資料寫入一個暫存檔
            with open(self.temp_path, 'wb') as f:
                pickle.dump(datas, f)

            # 2. 備份舊的正式檔 (可選，但建議)
            # if os.path.exists(self.pickle_path):
            #     # 這部分可以整合你的 backup 邏輯
            #     backup_path = self.pickle_path + ".old" 
            #     shutil.copy2(self.pickle_path, backup_path)

            # 3. 用暫存檔覆蓋正式檔 (這一步操作非常快，幾乎不可能中斷)
            shutil.move(self.temp_path, self.pickle_path)
            

        except Exception as e:
            if os.path.exists(self.temp_path):
                os.remove(self.temp_path)
            raise RuntimeError(e)

    def load_data(self):
        """從 pickle 檔案載入資料到 self.data。"""
        try:
            with open(self.pickle_path, 'rb') as f:
                self.data = pickle.load(f)
            # 載入後，設定資料結構參考
            if self.data and isinstance(self.data[0], (list, tuple)):
                self.data_structure = (type(self.data[0]), len(self.data[0]))
            self.logger.success(f"成功從 '{self.pickle_path}' 載入 {len(self.data)} 筆資料。")
        # except (FileNotFoundError, EOFError):
        #     self.logger.warning(f"找不到檔案 '{self.pickle_path}' 或檔案為空 ({len(self.data)}，將建立新的資料集。")
        except Exception as e:
            self.logger.exception(f"載入資料時發生錯誤：{e}")

    def clear_all_data(self):
        """清空記憶體中和檔案中的所有資料。執行前會先備份。"""
        self.logger.warning(f"即將刪除所有資料及檔案 '{self.pickle_path}'...")
        if self.pickle_path.exists():
            self.backup()
            self.pickle_path.unlink() # 使用 pathlib 的 unlink 取代 os.remove
        self.data = []
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
        """印出資料集的摘要資訊 (此處使用 print 是為了格式化輸出報告，而非記錄事件)。"""
        print("\n--- DataManager Info ---")
        print(f"Name: {self.name}")
        print(f"File Path: {self.pickle_path}")
        print(f"Total Items: {len(self)}")
        if self.data_structure:
            print(f"Data Structure: {self.data_structure[0].__name__} of length {self.data_structure[1]}")
        else:
            print("Data Structure: 尚未確定 (資料集為空)")
        if self.data:
            sample, _ = self.data[0]
            print(f"First Sample Type: {type(sample)}")
            if hasattr(sample, 'shape'):
                print(f"First Sample Shape: {sample.shape}")
        print("--------------------------\n")

    def __len__(self):
        """Return the length of our dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Working for indexing and automatically converted to PyTorch Tensors."""
        if not self.data:
            raise IndexError("資料集是空的，請先使用 .add_and_save() 添加資料。")
        
        try:
            sample, label = self.data[idx]
        except (ValueError, TypeError) as e:
            raise ValueError(f"索引 {idx} 的資料結構不符預期，應為 (sample, label) 的形式。錯誤: {e}")
        
        # --- 自動轉為 Tensor ---
        if not isinstance(sample, torch.Tensor):
            # 處理 numpy, list 等常見格式
            if isinstance(sample, np.ndarray):
                sample = torch.from_numpy(sample).float()
            else:
                try:
                    # 對 list 等結構也適用
                    sample = torch.tensor(sample, dtype=torch.float32)
                except (TypeError, ValueError):
                    # 如果是 PIL Image 等無法直接轉換的格式，交由 transform 處理
                    pass
        
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.float32)

        if self.transform:
            sample = self.transform(sample)

        return sample, label

    def __repr__(self):
        return f"<DataManager name='{self.name}' items={len(self)} path='{self.pickle_path}'>"