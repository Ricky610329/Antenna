
###* ============================================================================
###* antenna/utils/data.py
###*   反向設計閉迴路 (GEN→SM→SIM) 的「資料層」工具集。
###*
###*   本檔提供四個彼此相依的元件：
###*     1. make_hashable  : 把 Tensor / ndarray / 巢狀結構遞迴轉成「可 hash」的形式，
###*                          是整套去重 (deduplication) 機制的地基。
###*     2. Data           : 可持久化的 pickle 容器，封裝 save/load/backup/id 等基礎能力。
###*     3. dynamic_loss_filter : 依「樣本當下的 loss 是否落在 [lower, upper]」決定去留的過濾器。
###*     4. DataManager    : ★核心★ 同時繼承 Data 與 PyTorch Dataset，
###*                          是訓練腳本中 online_dataset / data_manager 的真身。
###*     5. size_converter : ★通用★ 張量形狀轉換器，GEN / SM / 正則化反覆呼叫。
###*
###*   在 pipeline 裡的位置：
###*     - SM (pattern→預測響應) 是 HFSS 的可微分替身，需要「資料集」才能訓練；
###*       線上學習過程中 SIM 用真實 HFSS 算出 (pattern, response) 樣本，
###*       透過 DataManager.add_and_save() 持續累積到 online_dataset。
###*     - 因為同一個 pattern 可能被 GEN 重複生成，去重至關重要，否則資料集會被
###*       近乎相同的樣本灌爆 → 這正是 make_hashable + data_set 存在的理由。
###* ============================================================================

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

###* ----------------------------------------------------------------------------
###* make_hashable：去重機制的地基
###*   Tensor / ndarray 本身「不可 hash」(無法直接放進 set / 當 dict key)，
###*   因此要先轉成不可變且可 hash 的代理物 (bytes / tuple / frozenset)。
###*   #! 關鍵：轉換必須是「確定性 (deterministic)」的 ——
###*      相同內容的兩個物件，務必映射到同一個 hash 值，去重才會正確。
###* ----------------------------------------------------------------------------
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
    #* 1) 純量 / 字串 / bytes / None：本身已可 hash，原樣回傳。
    if isinstance(item, (int, float, str, bytes, type(None))): return item
    #* 2) Tensor：先 detach() 切斷 autograd 圖、搬到 CPU、轉 numpy，再取原始位元組。
    #!   tobytes() 取的是「數值內容」的位元組 → 內容相同即 hash 相同，達成內容去重。
    #!   陷阱：dtype / shape 不同的 Tensor 可能產生相同位元組序列；本專案資料結構固定，
    #!         一般不會踩到，但若日後混入異質資料需留意。
    if isinstance(item, torch.Tensor): return item.detach().cpu().numpy().tobytes()
    #* 3) ndarray：同理取位元組。
    if isinstance(item, np.ndarray): return item.tobytes()
    #* 4) list / tuple：遞迴處理每個元素後包成 tuple(不可變→可 hash)。
    #!   list 與 tuple 都映射到 tuple → 兩者在去重時被視為等價。
    if isinstance(item, (list, tuple)): return tuple(make_hashable(i) for i in item)
    #* 5) set / frozenset：先排序再凍結，確保「同集合不同插入順序」得到相同結果。
    if isinstance(item, (set, frozenset)): return frozenset(sorted(make_hashable(i) for i in item))
    #* 6) dict：把每個 (k, v) 遞迴後依 key 排序 → 與插入順序無關，確定性可 hash。
    if isinstance(item, dict): return tuple(sorted((k, make_hashable(v)) for k, v in item.items()))
    #* 7) 其他型別：只要本身可 hash 就直接放行；不可 hash 則明確報錯，避免「靜默漏掉去重」。
    try:
        hash(item)
        return item
    except TypeError:
        raise TypeError(f"物件 {type(item).__name__} 不可 hash，且未在 make_hashable 中處理。")
    

###* ----------------------------------------------------------------------------
###* Data：可持久化的 pickle 容器 (DataManager 的父類別之一)
###*   提供「以 pickle 落地 / 載入 + 去重輔助 set + 內容指紋 id + 備份」的通用基礎。
###*   泛型 DataType 讓子類別能標註自己裝的是什麼 (DataManager 裝的是
###*   list[tuple[Tensor, Tensor]]，即 (pattern, response) 樣本清單)。
###* ----------------------------------------------------------------------------
class Data(Generic[DataType]):
    def __init__(self, data:DataType=None, *, name:str="data", rootdir:Union[Path, str]="./", suffix:str="data", load=True):
        self.data:DataType = data                 #* 實際裝載的資料(會被 pickle 落地的就是它)
        self.data_set = set()                     #* 去重用的查找表：存放 make_hashable(item) 後的指紋
        self.rootdir = Path(rootdir)
        self.name = name
        self.suffix = suffix
        #* 三條路徑：正式檔 / 暫存檔 / 日誌檔，皆由 name+suffix 推導，命名一致方便管理。
        self.savepath = self.rootdir.joinpath(f"{name}.{suffix}")
        self.temppath = self.rootdir.joinpath(f"{name}.{suffix}.tmp")
        self.logpath = self.rootdir.joinpath(f"{name}.{suffix}.log")

        #* 建構時若磁碟上已有舊檔，預設直接載回 → 達成跨程序/跨 run 的持久化。
        if load and self.savepath.exists(): self.load()
    
    def save(self):
        #* 採「寫暫存檔 → 原子搬移覆蓋」的寫法，是為了避免寫到一半被中斷而毀損正式檔。
        #!   線上學習會頻繁呼叫 save()，若直接覆寫原檔，一旦中途崩潰就會留下半截的壞檔，
        #!   下次載入即拋例外、整個訓練資料報廢；故先寫 .tmp 再 move(近乎瞬間且不可中斷)。
        self.rootdir.mkdir(parents=True, exist_ok=True)
        try:
            # 1. 將資料寫入一個暫存檔
            with open(self.temppath, 'wb') as f:
                pickle.dump(self.data, f)

            # 2. 用暫存檔覆蓋正式檔 (這一步操作非常快，幾乎不可能中斷)
            shutil.move(self.temppath, self.savepath)
        except Exception as e:
            #* 失敗時清掉殘留的暫存檔，避免污染目錄；再包成 RuntimeError 往外拋。
            if self.temppath.exists():
                self.temppath.unlink()
            raise RuntimeError(f"儲存檔案 '{self.savepath}' 失敗") from e

    def load(self) -> DataType:
        #* 從 pickle 正式檔還原 self.data；把兩類常見錯誤轉成語意清楚的例外訊息。
        try:
            with open(self.savepath, 'rb') as f:
                self.data:DataType = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"載入失敗：找不到檔案 '{self.savepath}'")
        except (pickle.UnpicklingError, IOError) as e:
            raise RuntimeError(f"載入失敗：無法讀取或解析檔案 '{self.savepath}'") from e
        return self.data

    def update(self, data, save:bool=False):
        #* 整批替換 self.data；save=True 時順手落地。
        #!   注意：這裡只換 data，不會同步重建 data_set → 去重表會與資料脫節，
        #!         所以 DataManager 的線上累積流程走的是 add_and_save() 而非 update()。
        self.data = data
        if save: self.save()

    def add_set(self, item):
        #* 把單筆資料的指紋登記進 data_set，回傳該指紋。
        #*   add_and_save() 每加入一筆「新」資料就呼叫它，使去重表與 data 同步成長。
        hashable_item = make_hashable(item)
        self.data_set.add(hashable_item)
        return hashable_item

    def clear(self, default = None):
        #* 同時清空資料本體與去重表，兩者必須一起清以保持一致。
        self.data = default
        self.data_set.clear()

    def __eq__(self, other: "Data") -> bool:
        #* 相等性以「內容指紋」判定，而非物件位址 → 兩個裝著相同資料的 Data 視為相等。
        if not isinstance(other, Data):
            return NotImplemented
        return self.make_hashable() == other.make_hashable()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', path='{self.savepath}')"

    def __str__(self):
        return str(self.data)

    def __hash__(self) -> int:
        #* 與 __eq__ 對齊：同內容→同 hash，使 Data 物件本身也能放進 set / dict。
        return hash(self.make_hashable())

    def __contains__(self, item: "Data") -> bool:
        #* 支援 `item in data_manager` 語法 → 這是 add_and_save() 去重的核心查詢。
        #!   注意參數名雖標註為 Data，實務上傳入的是「單筆樣本」(如 [pattern, response])；
        #!   先轉成指紋，再到 O(1) 的 data_set 裡查有無 → 故去重是常數時間，不需掃整個 list。
        hashable_item = make_hashable(item)
        return hashable_item in self.data_set

    
    @property
    def id(self) -> str:
        """使用 MD5 產生資料的 HASH 值"""
        #* 把整份資料的指紋再經 pickle→MD5 壓成一個 32 字元字串，
        #*   可當「這份資料集當下內容」的版本指紋(內容變→id 變)，便於辨識/紀錄。
        data_bytes = pickle.dumps(self.make_hashable())
        return md5(data_bytes).hexdigest()

    @property
    def has_data(self) -> bool:
        #* 注意：判斷的是 `is not None`，空 list ([]) 仍算「有 data 物件」→ 回傳 True。
        return self.data is not None

    def make_hashable(self):
        """
        將一個可能包含不可 hash 型別的複雜資料結構，遞迴地轉換為
        一個完全可 hash 的結構。
        """
        #* 對整份 self.data 套用模組級 make_hashable，供 __eq__ / __hash__ / id 共用。
        return make_hashable(self.data)

    def backup(self):
        #* 在破壞性操作(overwrite / clear)前先把現有正式檔複製一份帶時間戳的 .bak，
        #*   萬一新資料有問題仍可回溯。copy2 會連同檔案 metadata 一起複製。
        if not self.savepath.exists(): return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.rootdir.joinpath(f"{self.name}_{timestamp}.dataset.bak")
        shutil.copy2(self.savepath, backup_path)

###* ----------------------------------------------------------------------------
###* dynamic_loss_filter：依「樣本品質(loss)」決定去留的過濾器
###*   傳給 DataManager.filter() 當判斷函式；對每筆 (pattern, response) 計算其響應的
###*   criterion (即 loss / 設計目標差距)，只保留落在 [lower, upper] 區間者。
###*
###*   ★在 rollback(回滾) 流程的角色★：
###*     當線上更新讓 SM 變差需要回滾時，腳本會以 filter(upper=平均loss) 取出
###*     「loss 低於平均」的好樣本子集，只拿這批乾淨資料重訓 SM，
###*     避免被高 loss 的壞樣本繼續污染。lower 預設 -inf、upper 預設 +inf → 不設限即全收。
###* ----------------------------------------------------------------------------
def dynamic_loss_filter(
    datas:Tuple[Tensor, Tensor],
    lower: float = float('-inf'),
    upper: float = float('inf'),
) -> bool:
    """
    Example::

        DataManager.filter(minmax_filter, lower=TEMP('smaller', float('-inf')), upper=TEMP('bigger', float('inf')))
    """
    #* 延遲匯入 MultiResponses，避免 antenna 套件初始化時的循環匯入(circular import)。
    from antenna import MultiResponses
    _pattern, _response = datas              #* 拆出單筆樣本：pattern 此處用不到，只評估 response。
    # _pattern = AntennaPattern(_pattern)
    _response = MultiResponses(_response)     #* 包成響應物件以取得其 criterion()。
    _loss = _response.criterion().item()      #* criterion() 回傳該響應對設計目標的 loss(越小越好)。

    #* 區間判定：lower <= loss <= upper 才保留 → 回傳 True 代表「這筆留下」。
    return lower <= _loss <= upper


###* ============================================================================
###* DataManager：★整個資料層的核心★
###*   多重繼承 = Data(可持久化 pickle 容器 + 去重) + torch Dataset(可餵 DataLoader)。
###*   訓練腳本裡的 online_dataset / data_manager 就是它。
###*
###*   ◆ 在「線上學習(online learning)」中的角色：
###*     1) SIM 用真實 HFSS 產生新的 (pattern, response) 樣本；
###*     2) add_and_save() 自動「去重 + 落地」把樣本累積進線上資料集；
###*     3) DataLoader 直接吃這個 DataManager，__getitem__ 即時把每筆轉成 float/long Tensor，
###*        用來持續(online)更新 SM，使這顆「HFSS 可微分替身」越來越準。
###*
###*   ◆ 去重邏輯(為什麼需要)：
###*     GEN 在相近目標下可能反覆生成幾乎相同的 pattern；若不去重，資料集會被
###*     近乎重複的樣本灌爆，造成 SM 過擬合到少數模式、且浪費昂貴的 HFSS 計算。
###*     去重靠繼承自 Data 的 data_set(指紋集合) + `item in self`(O(1) 查詢)。
###*
###*   ◆ filter() 與 rollback 的關係：
###*     filter() 不改自身狀態，只回傳符合條件的 torch Subset；
###*     rollback 時以 filter(upper=平均loss) 取出好樣本子集重訓 SM(見 dynamic_loss_filter)。
###* ============================================================================
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
        #* 呼叫 Data.__init__：固定 suffix="dataset"、預設空 list、load=True 自動載回舊資料集。
        super().__init__(data=[], name=name, rootdir=Path(rootdir), suffix="dataset", load=True)

        self.transform = transform                #* 選用的樣本轉換(如資料增強)，於 __getitem__ 套用。

        # 設定 loguru
        #!   這裡 self.logger.remove() 會移除「全域」logger 的所有 handler，
        #!   故多個 DataManager 並存時後建立者會覆蓋前者的日誌設定 → 屬全域副作用，須留意。
        self.logger = logger
        log_level = "INFO" if verbose else "WARNING"
        self.logger.remove()
        self.logger.add(sys.stdout, level=log_level)
        self.logger.add(self.logpath, level="INFO", rotation="10 MB")

        # 根據已載入的 self.data 初始化輔助屬性
        #* 父類 load() 只還原 self.data，不會重建 data_set；故這裡逐筆 add_set() 補建去重表，
        #*   確保「續跑(resume)」時舊樣本也納入去重，不會被當成新樣本重複加入。
        for item in self.data:
            self.add_set(item)
        #* data_structure 記錄「每筆樣本的 (容器型別, 長度)」，作為後續寫入的一致性護欄。
        self.data_structure = None
        if self.data and isinstance(self.data[0], (list, tuple)):
            self.data_structure = (type(self.data[0]), len(self.data[0]))

        self.logger.info(f"DataManager '{self.name}' 初始化完成，共 {len(self.data)} 筆資料。")

    def load_data(self):
        """從 pickle 檔案載入資料，並建立一次性的查找 set。"""
        #* 相對於 __init__ 的逐筆 add_set，這裡用「集合生成式」一次重建整張去重表，
        #*   適合手動重新載入磁碟資料(例如外部更新了 .dataset 檔後想刷新記憶體狀態)。
        try:
            self.load()                            #* 還原 self.data。

            #* 重新推斷資料結構護欄。
            if self.data and isinstance(self.data[0], (list, tuple)):
                self.data_structure = (type(self.data[0]), len(self.data[0]))

            #* 直接整批重建 data_set(覆蓋舊的)，保證去重表與剛載入的 data 完全同步。
            self.data_set = {make_hashable(item) for item in self.data}

            self.logger.success(f"成功從 '{self.savepath}' 載入 {len(self.data)} 筆資料，索引建立完成。")

        except Exception as e:
            self.logger.exception(f"載入資料時發生錯誤：{e}")

    def filter(self, filter_func: Callable[[tuple[Tensor, Tensor]], bool] = dynamic_loss_filter, *args, **kwargs) -> Subset:
        """
        根據過濾條件建立並回傳一個資料子集 (Subset)。
        此方法不會改變 DataManager 本身的狀態。

        Default:: 
            
            # DataManager.filter(lower=..., upper=...)
            def dynamic_loss_filter(
                datas:Tuple[Tensor, Tensor], 
                lower: float = float('-inf'),
                upper: float = float('inf'),
            ) -> bool:
                return lower <= loss <= upper



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
            #* 對每筆原始資料(尚未轉 Tensor)套用 filter_func；*args/**kwargs 即 lower/upper 等門檻。
            #!   回傳的是「索引清單」而非資料本身 → 包成 Subset 後，DataLoader 取用時才經
            #!   __getitem__ 轉 Tensor，因此 filter 階段是對「原始 (pattern, response)」做判斷。
            indices = [
                i for i, item in enumerate(self.data)
                if filter_func(item, *args, **kwargs)
            ]

            #* Subset 只持有「父資料集 + 索引」的輕量視圖，不複製資料 → 省記憶體；
            #*   且不改動 self.data，符合 docstring 「不改變 DataManager 本身狀態」的承諾。
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

        #* 判斷傳進來的是「單筆」還是「一批」：
        #!   靠 new_data[0] 是否為 list/tuple 來推斷 →
        #!   單筆 [x, y] 會因 x 本身常是 Tensor(非 list/tuple) 被判為非 batch，包成 [[x, y]] 統一處理；
        #!   一批 [[x1,y1],[x2,y2]] 的首元素是 list → 判為 batch。
        #!   邊界陷阱：若單筆樣本的 x 本身就是 list/tuple，會被誤判成 batch，故樣本特徵通常用 Tensor。
        is_batch = isinstance(new_data[0], (list, tuple))
        items_to_process:list[Union[list[Tensor,Tensor],list[Any,Any]]] = new_data if is_batch else [new_data]

        # --- 檢查資料結構一致性 ---
        #* 逐筆核對 (型別, 長度) 是否與既有資料一致，避免把格式不符的資料混進同一個資料集，
        #*   否則 __getitem__ 解包 (sample, label) 時會在訓練中途才爆炸，難以追查。
        for item in items_to_process:
            if not isinstance(item, (list, tuple)):
                self.logger.error(f"資料項必須是 list 或 tuple，但收到了 {type(item)}。")
                return
            current_structure = (type(item), len(item))
            #* 資料集尚空 → 以第一筆建立「基準結構」；之後所有寫入都要符合它。
            if self.data_structure is None and not self.data:
                self.data_structure = current_structure
                self.logger.debug(f"偵測到資料結構為：{self.data_structure[0].__name__} of length {self.data_structure[1]}。")
            elif self.data_structure != current_structure:
                self.logger.error(f"結構不符：新資料的結構 {current_structure} 與現有結構 {self.data_structure} 不符。")
                return

        #* 若樣本是 Tensor，先 detach() 切斷計算圖再存。
        #!   極重要：線上學習中這些樣本常直接來自 GEN/SIM 的前向輸出，仍掛在 autograd 圖上；
        #!   不 detach 就 pickle，會把整張計算圖也序列化進去 → 記憶體爆炸甚至無法 pickle。
        if isinstance(items_to_process[0][0], Tensor):
            detached_data = [
                [t.detach() for t in inner_list]
                for inner_list in items_to_process
            ]
            items_to_process = detached_data

        #* overwrite：先備份再清空(含去重表)，等同「整批重置」資料集 —— rollback 重建資料集時可用。
        if mode == 'overwrite':
            self.logger.warning(f"使用 'overwrite' 模式，將會清除所有現有資料。")
            self.backup()
            self.clear([])
        elif mode != 'append':
            self.logger.error(f"無效的模式 '{mode}'。請使用 'append' 或 'overwrite'。")
            return

        #* ★去重關鍵迴圈★：逐筆問「item not in self」(走 __contains__ → O(1) 指紋查詢)，
        #*   只有不存在的才收進 unique_new_data，並同步 add_set() 登記指紋。
        #!   這確保線上學習反覆生成的重複 (pattern, response) 不會被重複落地。
        unique_new_data = []
        for item in items_to_process:
            if item not in self:
                unique_new_data.append(item)
                self.add_set(item)

        #* 重複數 = 送進來的總筆數 − 實際新增筆數。
        num_duplicates = len(items_to_process) - len(unique_new_data)
        if num_duplicates > 0:
            #* 全部都是重複 → 直接 return，連 save() 都省，避免無謂的磁碟 I/O。
            if not unique_new_data:
                self.logger.info(f"所有 ({num_duplicates} 筆) 待加入的資料皆已存在，不執行任何操作。")
                return
            else:
                self.logger.info(f"發現 {num_duplicates} 筆重複資料，將予以忽略。")


        num_added = len(unique_new_data)
        self.data.extend(unique_new_data)     #* 把去重後的新樣本接到資料尾端。

        #* 落地：成功才記 success；save() 內部已是「暫存→原子搬移」，失敗會丟 RuntimeError。
        try:
            self.save()
            self.logger.success(f"成功添加 {num_added} 筆新資料並儲存！ (目前共 {len(self.data)} 筆資料)")
        except RuntimeError as e:
            self.logger.error(f"儲存資料時發生錯誤：{e}")

    def clear_all_data(self):
        """清空記憶體中和檔案中的所有資料。執行前會先備份。"""
        self.logger.warning(f"即將刪除所有資料及檔案 '{self.savepath}'...")
        if self.savepath.exists(): self.backup()    #* 破壞前先備份，可回溯。

        self.clear([])                #* 清空 self.data 與 data_set(去重表)。
        self.data_structure = None    #* 連結構護欄一併重置，之後第一筆資料會重新定義結構。
        self.save() # 儲存空的列表
        self.logger.info("所有資料已清除。")
    
    def info(self):
        """印出資料集的摘要資訊。"""
        #* 快速健檢工具：可順手比對 Total Items vs Hashed Items —— 兩者應相等，
        #*   若不等代表資料本體與去重表脫節(例如曾用 update() 繞過去重)，是潛在 bug 訊號。
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
        #* DataLoader 靠它決定可索引範圍 = 目前資料筆數。
        return len(self.data)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        """Working for indexing and automatically converted to PyTorch Tensors."""
        #* ★Dataset 介面核心★：DataLoader 取每筆時呼叫它，
        #*   把「儲存時的原始形態 (可能是 Tensor/ndarray/list/純量)」即時標準化成訓練可用的 Tensor。
        #!   設計重點：去重/落地階段刻意保留原始資料(可 detach 的 Tensor)，
        #!             型別/dtype 的統一延到「取用時」才做，讓儲存層與訓練層解耦。
        if not self.data:
            raise IndexError("資料集是空的，請先使用 .add_and_save() 添加資料。")

        # 1. 從列表中獲取原始資料
        try:
            sample, label = self.data[idx]    #* 預期每筆都是 (sample, label) 二元組，否則報結構錯誤。
        except (ValueError, TypeError) as e:
            raise ValueError(f"索引 {idx} 的資料結構不符預期，應為 (sample, label) 的形式。錯誤: {e}")

        # 2. 應用使用者定義的轉換 (主要用於圖像增強等複雜操作)
        if self.transform:
            sample = self.transform(sample)

        # 3. 確保 sample 是 Tensor
        #* 已是 Tensor 就原樣保留(不強制改 dtype)；否則依來源型別選最有效率的轉法並轉 float32。
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
        #* 依「label 是否為浮點」自動分流 dtype：
        # 迴歸任務的標籤通常是 float32
        if label.is_floating_point():
            label = label.to(dtype=torch.float32)
        # 分類任務的標籤需要是 long (int64)，以用於 CrossEntropyLoss 等
        else:
            label = label.to(dtype=torch.int64)

        return sample, label


    def __repr__(self):
        return f"<DataManager name='{self.name}' items={len(self)} path='{self.savepath}'>"

###* ============================================================================
###* size_converter：★通用張量形狀轉換器★
###*   GEN / SM / 正則化在 pipeline 中反覆需要把「一團展平或半成形的張量」整回正確形狀。
###*   它不靠呼叫端硬編形狀，而是向 `sizer`(具 .size() 的物件，描述單一樣本的 N 與 (H, W))
###*   詢問每筆樣本的尺寸，再據此自動推出 batch 大小 B 並 reshape。
###*
###*   兩種模式(由 output_shape 是否為 None 決定)：
###*     ◆ 模式 1（output_shape=None，用 flatten / batch 兩個旗標）：
###*         flatten=True  → 每筆攤平成 (N,)，整體 (B, N) 或去批次後 (N,)
###*         flatten=False → 每筆還原成 (H, W)，整體 (B, H, W) 或 (H, W)
###*         batch=True    → 強制保留批次維 (B, ...)，即使 B=1
###*         batch=False   → B=1 時擠掉批次維；B>1 卻要非批次輸出則直接報錯(無法壓非單例維)。
###*     ◆ 模式 2（output_shape 是字串，如 "B,1,H,W" 或 "B,N,1"）：
###*         忽略 flatten/batch，逐段把 'B'/'N'/'H'/'W'/數字 翻成實際維度，精準塑形，
###*         並驗證總元素量一致、未含 'B' 時不可有 B>1。常用於要插入 channel 維度的情境(如 1)。
###*   #! 共同前提：input 的總元素數必須是「單筆 N」的整數倍，否則無法推出乾淨的 B。
###* ============================================================================
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
    #* 先向 sizer 問清「單筆樣本」的尺寸：
    #*   N_per_sample = 攤平後元素數(N)；components = (H, W) 二維形狀。
    #!   sizer 介面是這個轉換器的關鍵抽象 —— 只要物件提供 .size(flatten)，就能被它整形，
    #!   因此 GEN/SM 的 pattern 與 response 都能共用同一支函式而不必各寫 reshape。
    try:
        N_per_sample = sizer.size(flatten=True)
        components = sizer.size(flatten=False)
        H_comp = components[0]
        W_comp = components[1]
    except Exception as e:
        raise ValueError(f"Unable to obtain size information from sizer({sizer})\n{e}")

    #* 用「總元素數 ÷ 單筆 N」反推批次大小；不能整除代表輸入根本不是這種樣本的整數倍 → 報錯。
    total_input_numel = tensor.numel()
    if total_input_numel % N_per_sample != 0:
        raise ValueError(
            f"The total number of elements in the input tensor ({total_input_numel}) "
            f"must be an integer multiple of {N_per_sample}."
        )

    #* Batch size
    B_calc = total_input_numel // N_per_sample    #* 推算出的批次大小，後續所有塑形都以它為準。

    #* Use the string output_shape
    #* ===== 模式 2：字串塑形 =====
    #*   逐段解析 output_shape，把符號翻成實際維度數值，組成 final_shape_list。
    if output_shape is not None:
        try:
            shape_parts = [part.strip() for part in output_shape.split(',')]
            final_shape_list = []
            has_batch_dim = False              #* 記錄字串裡是否出現 'B'，供後續批次合法性檢查。

            for part in shape_parts:
                if part == 'B':                #* 'B' → 推算出的批次大小
                    final_shape_list.append(B_calc)
                    has_batch_dim = True
                elif part == 'N':              #* 'N' → 單筆攤平長度
                    final_shape_list.append(N_per_sample)
                elif part == 'H':              #* 'H' → 單筆高度
                    final_shape_list.append(H_comp)
                elif part == 'W':              #* 'W' → 單筆寬度
                    final_shape_list.append(W_comp)
                elif part.isdigit():           #* 純數字 → 固定維度(常見如插 channel 維 "1")
                    final_shape_list.append(int(part))
                else:
                    raise ValueError(f"'{part}'")    #* 不認得的符號 → 觸發下方統一錯誤訊息。

        except ValueError as e:
            raise ValueError(
                f"The string `output_shape` contains an unrecognized component: {e}."
                "Please only use 'B', 'N', 'H', 'W', or numbers."
            )
        
        #* Validate: batch dimension
        #!   護欄一：B>1 卻沒在字串裡放 'B'，等於要把多個樣本硬塞進不含批次維的形狀 →
        #!           會默默把 batch 揉進其他維度造成資料錯亂，故直接擋下。
        if not has_batch_dim and B_calc > 1:
            raise ValueError(
                f"輸入的計算批次大小為 {B_calc}, 但 output_shape "
                f"'{output_shape}' 中未包含 'B'。 "
                "無法壓縮非單例的批次維度。"
            )

        #* Validate: the final total number of elements
        #!   護欄二：reshape 前先確認目標形狀的總元素量與輸入完全一致，
        #!           提早給出清楚錯誤，而不是讓 torch.reshape 丟出較難讀的訊息。
        target_numel = 1
        for dim in final_shape_list:
            target_numel *= dim

        if target_numel != total_input_numel:
            raise ValueError(
                f"Output shape '{output_shape}' (解析為 {final_shape_list}) "
                f"的總元素量 ({target_numel}) 與 "
                f"輸入張量的總元素量 ({total_input_numel}) 不匹配。"
            )

        return tensor.reshape(final_shape_list)    #* 通過兩道護欄後才真正塑形。

    else: #* Use the flatten and batch parameters
        #* ===== 模式 1：flatten / batch 旗標塑形 =====
        #*   先決定「每筆樣本」的目標形狀，再前綴 B 組成完整形狀。
        if flatten:
            target_shape_per_sample = (N_per_sample,)      #* 攤平 → (N,)
        else:
            target_shape_per_sample = components # (H_comp, W_comp)    #* 還原二維 → (H, W)

        final_shape = (B_calc, *target_shape_per_sample)   #* 一律先 reshape 成含批次維的形狀。
        output_tensor = tensor.reshape(final_shape)

        if not batch:
            #* 要求非批次輸出：唯有 B=1 才能安全擠掉批次維。
            if B_calc == 1:
                #? (1, H, W) -> (H, W) or (1, N) -> (N,)
                return output_tensor.squeeze(dim=0)
            else:
                # B > 1, 但要求 non-batch output
                #!   B>1 又要 batch=False 是矛盾請求(會遺失樣本維度) → 報錯而非靜默壓縮。
                raise ValueError(
                    f"輸入的計算批次大小為 {B_calc}, 但請求了 'batch=False' "
                    "(非批次輸出)。無法壓縮非單例的批次維度。"
                )
        else:
            #? (B, H, W) -> (B, H, W)
            #* batch=True：保留批次維直接回傳(即使 B=1 也維持 (1, ...) 形狀)。
            return output_tensor
        
