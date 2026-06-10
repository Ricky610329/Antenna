# ============================================================
# antenna/patch/record.py
#
# 【用途】事後分析工具（非訓練迴圈本身使用）
#   train_single.py / train_dual.py 訓練時，會透過名為 TEMP 的
#   Record 物件逐 epoch 將各指標序列寫入 NAS 上的 temp.record。
#   本模組的 PatchTrainingRecord 在「訓練結束後」載入該檔案，
#   提供便捷屬性與查詢方法，讓研究者快速取得最佳結果與歷史曲線。
#
# 【設計原則】
#   - 繼承 antenna.utils.utils.Record，利用 load=True 參數在
#     __init__ 時自動從磁碟反序列化 temp.record。
#   - 所有 @property 皆為唯讀包裝，直接委派給父類別的 __getitem__，
#     不建立額外副本，確保記憶體效率。
# ============================================================

from antenna.utils.utils import Path
from antenna.utils.record import Record
from antenna.types import Tensor

class PatchTrainingRecord(Record):
    """
    既有訓練結果的事後分析封裝。

    用法範例（於 Notebook 或分析腳本中）::

        rec = PatchTrainingRecord("exp_20250101_001")
        print(rec.best_epoch)           # 最低 real_loss 出現的 epoch
        print(rec.best_pattern.shape)   # 對應天線幾何參數
        pattern_at_50 = rec.pattern(50) # 取第 50 epoch 的 pattern

    注意：此類別不參與訓練迴圈，請勿在 Trainer 內部實例化。
    """

    ###* ── 建構與初始化 ──────────────────────────────────────
    def __init__(self, name:str):
        # ── NAS 路徑說明 ───────────────────────────────────────
        # 硬編碼指向實驗室 NAS (IP: 140.123.106.219) 上的結果根目錄。
        # 若 NAS 離線或網路不通，Path 物件仍可建立，但 super().__init__
        # 的 load=True 會在嘗試讀取 temp.record 時拋出 FileNotFoundError。
        # 若需在不同機器分析，請在呼叫前確認 NAS 已掛載且路徑可存取。
        result_dir = Path(r"\\140.123.106.219\Temp\碩二_吳維文's\Patch Antenna\Experiment\result")

        # 以 "temp" 為 Record 名稱載入 result_dir/name/temp.record
        # load=True：立即從磁碟反序列化，所有 epoch 序列皆可取用
        super().__init__("temp", rootdir=result_dir.joinpath(name).absolute(), load=True)

        self.rootdir = result_dir.joinpath(name).absolute() # 實驗資料夾絕對路徑（供外部查詢）
        self.name = name                                     # 實驗名稱，對應 NAS 上的子資料夾名稱

        ###? best() 呼叫說明：
        #   mode=min      → 尋找使 key 值最小的 epoch（min_loss 模式）
        #   key='real_loss' → 依 TEMP 中 real_loss 序列（模擬器回傳的真實損耗，單位 dB）排序
        #   output_keys   → 同時取出對應 epoch 的三個欄位：
        #     'epoch'            → epoch 編號（整數）
        #     'patch_pattern_buf'→ 該 epoch 最佳天線幾何參數（Tensor）
        #     'patch_result_buf' → 該 epoch 模擬器回傳的頻率響應曲線（Tensor）
        #   回傳值為長度 3 的 list，依序對應 output_keys 的順序
        self.best_results = self.best(
            mode = min,
            key = "real_loss",
            output_keys = ['epoch', 'patch_pattern_buf', 'patch_result_buf']
        )

    ###* ── 最佳結果快捷屬性 ─────────────────────────────────
    @property
    def best_epoch(self) -> int:
        """real_loss 最小值所在的 epoch 編號（對應 TEMP['epoch']）。"""
        return self.best_results[0]  # best() output_keys[0] = 'epoch'

    @property
    def best_pattern(self) -> Tensor:
        """best_epoch 對應的天線幾何參數（對應 TEMP['patch_pattern_buf']）。"""
        return self.best_results[1]  # best() output_keys[1] = 'patch_pattern_buf'

    @property
    def best_response(self) -> Tensor:
        """best_epoch 對應的模擬器頻率響應曲線（對應 TEMP['patch_result_buf']）。"""
        return self.best_results[2]  # best() output_keys[2] = 'patch_result_buf'

    ###* ── 各 epoch 序列屬性（對應 TEMP 中的記錄鍵值） ─────
    @property
    def real_loss(self) -> list[float]:
        """每 epoch 模擬器計算的真實損耗序列（TEMP['real_loss']）。
        此為訓練目標，數值越低代表天線越符合規格。"""
        return self['real_loss']

    @property
    def min_loss(self) -> list[float]:
        """每 epoch 截至當下的歷史最低 real_loss（TEMP['min_loss']）。
        單調遞減曲線，用於觀察收斂趨勢。"""
        return self['min_loss']

    @property
    def fake_loss(self) -> list[float]:
        """每 epoch 鑑別器（surrogate / GAN discriminator）預測的損耗
        （TEMP['fake_loss']）。可與 real_loss 比較觀察代理模型品質。"""
        return self['fake_loss']

    @property
    def mutation(self) -> list[float]:
        """每 epoch 演化演算法的突變率序列（TEMP['mutation']）。"""
        return self['mutation']

    @property
    def tau(self) -> list[float]:
        """每 epoch 溫度/閾值參數序列（TEMP['tau']）。
        用於控制 Gumbel-softmax 或退火策略的軟化程度。"""
        return self['tau']

    @property
    def real_loss_average(self) -> list[float]:
        """每 epoch real_loss 的滑動/批次平均值（TEMP['real_loss_average']）。
        較 real_loss 平滑，適合繪製收斂曲線。"""
        return self['real_loss_average']

    @property
    def patch_result_buf(self) -> list[Tensor]:
        """每 epoch 最佳候選的模擬器頻率響應曲線序列（TEMP['patch_result_buf']）。
        shape 通常為 [num_freq_points]。"""
        return self['patch_result_buf']

    @property
    def patch_pattern_buf(self) -> list[Tensor]:
        """每 epoch 最佳候選的天線幾何參數序列（TEMP['patch_pattern_buf']）。
        shape 通常為 [num_params]。"""
        return self['patch_pattern_buf']

    @property
    def time(self) -> list[float]:
        """每 epoch 耗時（秒）序列（TEMP['time']）。
        可用於估算完整訓練所需時間或偵測效能瓶頸。"""
        return self['time']

    @property
    def r_feed(self) -> list[float]:
        """每 epoch 饋電點電阻序列（TEMP['r_feed']）。
        對應模擬器回傳的輸入阻抗實部，用於阻抗匹配分析。"""
        return self['r_feed']

    ###* ── 依 epoch 查詢單筆資料 ──────────────────────────
    def pattern(self, epoch:int) -> Tensor:
        """查詢指定 epoch 的天線幾何參數。

        委派給父類別 Record.find()，以 TEMP['epoch'] 欄位定位，
        再取出對應位置的 TEMP['patch_pattern_buf'] 值。

        Args:
            epoch: 欲查詢的 epoch 編號（需存在於 TEMP['epoch'] 序列中）。

        Returns:
            該 epoch 的幾何參數 Tensor。
        """
        return self.find('epoch', epoch, 'patch_pattern_buf')

    def response(self, epoch:int) -> Tensor:
        """查詢指定 epoch 的模擬器頻率響應曲線。

        委派給父類別 Record.find()，以 TEMP['epoch'] 欄位定位，
        再取出對應位置的 TEMP['patch_result_buf'] 值。

        Args:
            epoch: 欲查詢的 epoch 編號（需存在於 TEMP['epoch'] 序列中）。

        Returns:
            該 epoch 的頻率響應 Tensor。
        """
        return self.find('epoch', epoch, 'patch_result_buf')
