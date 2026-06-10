# ==============================================================================
# record.py — Record (★)：訓練狀態的「時序記錄器」(run_training 裡的 TEMP)
# ------------------------------------------------------------------------------
# 從 utils.py 拆出 (純搬家，行為不變)。對外經 antenna.utils facade 取用：
#     from antenna.utils import Record
# 序列化相容：Record.save() 落地的是 state_dict() 純 dict (不含類別參照)，
# 故搬家不影響舊 .record 檔的讀取 (app.py 依賴的格式不變)。
# ==============================================================================
import sys
from collections import defaultdict
from datetime import datetime
from pickle import dump as _pickle_dump, load as _pickle_load
from typing import Any, Callable, List, Optional, Tuple, TypeVar, overload

import numpy as np
from numpy import ndarray
from pandas import DataFrame
import torch
from torch import Tensor

from .utils import Path

ReturnType = TypeVar('ReturnType')

class Record:
    """
    訓練狀態的「時序記錄器」—— 即 train_single.py / train_dual.py 裡的 ★TEMP★。

    核心心智模型：它是一個「每個 key 都對應一條歷史序列 (list)」的容器。
        record['loss'] = 0.5   # 不是覆寫，而是 append 到 'loss' 這條序列尾端
        record('loss')         # 取 'loss' 序列的「最後一筆」(目前值)
        record['loss']         # 取整條 'loss' 序列 (歷史全紀錄)

    為什麼這樣設計 / 在訓練迴圈中的角色：
      - 斷點續跑：每個 epoch 都把 loss / epoch / pattern / 結果等 append 進來，
        再 save() 成 .record (pickle)；重啟時 load(=True) 即可從上次 epoch 接著跑。
      - 繪圖：因為每個 key 是完整序列，直接丟給 matplotlib 就是整段訓練曲線。
      - rollback / early stop：
          * early_stop('real_loss', patience) → 連續 patience 次沒進步就回傳 True，
            觸發把 GEN 回滾到歷史最佳 epoch。
          * find('real_loss', min_loss, 'epoch') → 反查「最佳 loss 出現在哪個 epoch」，
            據此 generator.change(該 epoch, load=True) 把權重載回最佳狀態。
      - pattern 去重：index('patch_pattern_buf', 某 pattern) 找這張 pattern 是否模擬過，
        命中就跳過昂貴的 HFSS、直接用 find 取回先前結果 (省時的關鍵)。

    _data 存「時序資料」；_history 存「每次 save 的中繼資料 (時間/描述/長度)」當存檔日誌。
    """
    def __init__(self, name:str = "record", rootdir:Optional[str] = None, load:bool = False):
        # defaultdict(list)：存取任一新 key 時自動建立空序列，因此 record['x']=v 永遠是 append。
        self._data:dict[str, list] = defaultdict(list)
        self._history = defaultdict(list)  # 存檔日誌：每次 save() 追加一筆 (time/description/len)
        self.name = name
        self.path = Path(rootdir or "./").joinpath(
            f"{name}.record"  # 存成 <name>.record (pickle 檔)
        )

        if load: self.load()  # load=True：建構時就從磁碟載回上次狀態 (斷點續跑入口)

    def __call__(self, key, default = None, *, append = False):
        """Get the last value of key."""
        # record(key)：取該序列「最後一筆」= 目前值。這是訓練迴圈最常用的讀法。
        # append=True 時，若 key 還沒有值就先把 default 寫進去再回傳 (確保曲線從第一個 epoch 起就有點)。
        return self.end(key, default, append = append)

    def __setitem__(self, key, value):
        # record[key] = value：把 value「追加」到該序列尾端 (不是覆寫!)，形成時序。
        self._data[key].append(value)

    def __getitem__(self, key):
        """Get the complete array of keys."""
        # record[key]：取「整條序列」(歷史全紀錄)，常直接餵給 matplotlib 畫曲線。
        if self.__contains__(key):
            return self._data[key]
        else:
            # 故意不回空 list 而是報錯並列出現有 key，避免打錯字導致畫出空圖卻無感。
            _keys = ', '.join(self._data.keys())
            raise KeyError(f"{key} does not exist. (Current key: {_keys})")

    def __delitem__(self, key):
        del self._data[key]

    def  __contains__(self, item:str):
        # 'key' in record：判斷該序列是否存在 (去重時先確認 buffer 鍵已建立)。
        return item in self._data

    def state_dict(self) -> dict[str, dict[str, list]]:
        """Return the state of the Record as a dict."""
        # 仿 PyTorch state_dict 介面：把可序列化的完整狀態打包成普通 dict 供 save() pickle。
        return {    # Convert to a normal dict.
            '_data': dict(self._data),
            '_history': dict(self._history)
        }

    def load_state_dict(self, state_dict: dict[str, dict[str, list]]):
        """Load the Record state."""
        # 從 state_dict 還原；用 .get 容錯舊檔缺欄位，並包回 defaultdict(list) 維持 append 語義。
        loaded_data = state_dict.get('_data', {})
        loaded_history = state_dict.get('_history', {})

        self._data = defaultdict(list, loaded_data)
        self._history = defaultdict(list, loaded_history)

    def end(self, key, default = None, *, append = False):
        # 取某序列的最後一筆 (即「目前值」)。__call__ 就是轉呼叫這裡。
        if self.__contains__(key) and len(self.__getitem__(key)) > 0:
            return self.__getitem__(key)[-1]
        else:
            # 序列還空：append=True → 先把 default 寫入再回傳 (遞迴一次取出)；否則僅回傳 default。
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
        # 累加器：以「目前值 + num」作為新的一筆 append 進去 (仍保留每一步的歷史)。
        # 訓練裡的 de (距上次刷新最佳的 epoch 數) 就靠 TEMP.add('de', 1) 累加。
        self.__setitem__(
            key, self.end(key, default) + num
        )
        

    
    def save(self, description:Optional[str] = None):
        # 訓練每個 epoch 結尾呼叫 (TEMP.save(f"{epoch} times"))：這就是斷點續跑的「存檔點」。
        # 先在 _history 追加一筆存檔日誌 (時間/描述/當前長度)，再把整個狀態 pickle 寫檔。
        self._history["time"].append(str(datetime.now()).split(".")[0])  # 去掉微秒，只留到秒
        self._history["description"].append(description or "No description")
        self._history["len"].append(len(self))

        current_state = self.state_dict()
        with open(str(self.path), "wb") as f:
            _pickle_dump(
                current_state,
                file = f
            )

    def load(self):
        # 從 .record 載回 (建構時 load=True 會走這裡)。
        if not self.path.exists():
            self.save()  # 首次執行還沒有存檔 → 先存一份空的，避免讀檔失敗
        with open(str(self.path), "rb") as f:
            loaded_state = _pickle_load(f)
        self.load_state_dict(loaded_state)

        return self._data

    def average(self, key:str):
        # 回傳整條序列的平均值 (空序列回 None)。
        # 訓練用它做兩件事：(1) 判斷本筆 real_loss 是否優於歷史平均 → 決定是否收進線上資料集；
        #                  (2) 在圖上標 r_feed / time 的平均值。
        _key_datas = self._data[key]
        _key_datas_len = len(_key_datas)
        if _key_datas_len > 0:
            return sum(_key_datas) / _key_datas_len
        else:
            return None
        
    def index(self, key:str, value, *, start:int = 0, stop:int = sys.maxsize) -> Optional[int]:
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
        # 在某序列中找出 value 第一次出現的位置 (找不到回 None)。
        # ★ pattern 去重的核心：訓練用 index('patch_pattern_buf', 這張 pattern) 判斷
        #   這張圖樣是否模擬過 —— 非 None 代表命中快取，可省下一次昂貴的 HFSS 模擬。
        if key not in self._data:
            return None

        # numpy/torch 張量不能用 in / list.index (== 會逐元素比較、語義不對)，
        # 必須改用 array_equal / torch.equal 逐筆做「整體相等」比對。
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
            # 一般可雜湊/可比較的值：直接用 list.index (含 start/stop 範圍)。
            if value in self[key]:
                return self[key].index(value, start, stop)
            else:
                return None

        # 張量情形：回傳第一個 True 的位置 (即第一筆相等的索引)。
        if True in _result:
            return _result.index(True)
        else:
            return None
    
    @overload
    def find(self, key, value, other_keys:str, *, start=0, stop=sys.maxsize) -> Optional[Any]:...
    @overload
    def find(self, key, value, other_keys:Tuple[str, ...] , *, start=0, stop=sys.maxsize) -> Optional[List[Any]]:...

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
        # 「同一時間步、跨序列查表」：先用 index 在 key 序列找到 value 的位置 _index，
        # 再回傳同一位置上 other_keys 序列的值 (因所有序列以 epoch 同步對齊)。
        # 訓練兩大用途：
        #   (1) rollback：find('real_loss', min_loss, 'epoch') → 反查最佳 loss 是哪個 epoch。
        #   (2) 去重命中：find('patch_pattern_buf', 此 pattern, ('patch_result_buf','real_loss'))
        #       → 取回先前同一張 pattern 的模擬結果與 loss，免再跑 HFSS。
        _index = self.index(key, value, start=start, stop=stop)
        if _index is None:
            return None
        elif isinstance(other_keys, str):
            return self[other_keys][_index]            # 單一 key → 回傳單值
        else:
            _result = []
            for other_key in other_keys:
                _result.append(self[other_key][_index])  # 多個 key → 回傳對齊的值清單
            return _result

    def best(self, mode: Callable = min, key:str = "real_loss", output_keys:list[str] = ['epoch', 'patch_pattern_buf', 'patch_result_buf']) -> list:
        # 找出某指標的「最佳那一筆」並一次帶回對應欄位。
        # 預設 mode=min、key='real_loss'：即「真實 loss 最小的那個 epoch」的 epoch/pattern/結果。
        if key not in self._data or not self._data[key]:
            return None

        # 取得目標 key 中的最佳數值 (Best value)
        best_value = mode(self._data[key])

        # 呼叫現有的 find 方法回傳對應的 output_keys
        return self.find(key, best_value, output_keys)

    def early_stop(self, key: str, patience: int = 10, is_maximize: bool = False) -> bool:
        """
        根據指定 key 的歷史資料，決定是否應該 early stop。
        若最近 `patience` 次都沒有改善，回傳 True。
        Args:
            is_maximize: 若為 True, 則尋找最大值, 否則尋找最小值。
        """
        # ★ 注意：在本專案訓練迴圈中，回傳 True 並非「停止訓練」，而是「觸發 rollback」——
        #   即把 GEN 載回歷史最佳 epoch、並用線上資料集重訓 SM，藉此跳出停滯/局部最佳。
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

    def reset(self, key:Optional[str]=None, delete:bool = False):
        # 清空紀錄：給 key → 只清該序列 (delete=True 連鍵一起移除，否則清成空 list)；
        # 不給 key → 整個 _data 重置 (重新開始記錄)。
        if key is not None:
            if delete:
                self._data.pop(key, None)
            else:
                self._data[key] = []
        else:
            self._data = defaultdict(list)


    def custom(self, key:str, fn:Callable[[list], ReturnType], *, default = None) -> Optional[ReturnType]:
        # 對整條序列套用任意彙總函式並回傳結果 (空序列回 default)。
        # 例：訓練結尾 TEMP.custom('real_loss', min) 取整段最小 real_loss 當最終戰績。
        _key_data = self._data[key]
        if _key_data:
            return fn(_key_data)
        return default

    @property
    def dataframe(self):
        # 把所有時序資料轉成 pandas DataFrame (每個 key 一欄)，方便檢視/匯出/印出。
        processed_data = {}
        for key, values in self._data.items():
            processed_values = []
            for item in values:
                if isinstance(item, torch.Tensor):
                    # 張量先搬回 CPU、detach 脫離計算圖再轉成原生 list/數值，
                    # 否則 DataFrame 無法妥善容納帶梯度的 GPU 張量。
                    # Move to CPU and detach to convert to a standard Python list/number
                    processed_values.append(item.cpu().detach().tolist())
                else:
                    processed_values.append(item)

            processed_data[key] = processed_values
        try:
            return DataFrame(processed_data)
        except ValueError as e:
            # DataFrame 要求各欄等長；長度不一時把目前各序列長度 (repr) 一併拋出，方便定位哪一欄漏記。
            raise ValueError(f"{e}\n{repr(self)}")

    @property
    def history(self):
        # 以 DataFrame 呈現存檔日誌 (每次 save 的時間/描述/長度)，可快速回顧續跑歷程。
        return DataFrame(self._history)

    def __str__(self):
        # print(record) 直接顯示整張資料表。
        return str(self.dataframe)

    def __repr__(self):
        # 精簡摘要：列出每個 key 及其序列長度，例如 Record(temp: epoch[10] real_loss[10] ...)。
        _str = ''
        for key, value in self._data.items():
            _str += f"{key}[{len(value)}] "

        return f"Record({self.name}: {_str})"

    def __len__(self):
        # len(record)：以 DataFrame 列數為準 = 已記錄的 epoch 數 (各序列同步成長)。
        return len(self.dataframe)

