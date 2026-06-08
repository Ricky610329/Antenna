###* ============================================================================
###* KuoHung.py — 學長參考天線模組
###*
###* 功能定位：
###*   本模組封裝「已知良好參考天線」(俗稱「學長(KuoHung)天線」)的圖樣(pattern)
###*   與對應電磁響應(response)，專門供 SM(Surrogate Model，代理模型)在訓練開始時
###*   進行「單筆暖身微調(warm-up fine-tune)」，使 SM 在正式線上學習前就對
###*   優秀圖樣附近的 pattern→response 映射有較準確的初始預測。
###*
###* SM 暖身流程角色(見 train_single.py)：
###*   1. 呼叫 KuoHung.load(name) → 取得 (KuoHung_tensor, response_tensor)
###*   2. 對 SM 執行 train_one_data(pattern.series, response, ...)
###*      → SM 僅用這一筆資料做微調，不大幅偏離預訓練權重
###*   3. 後續正式閉迴路(GEN/SM/SIM)訓練中，SM 對「好圖樣區域」預測更穩定
###*
###* 資料格式說明：
###*   - 圖樣(pattern) : 形狀 (25, 25) 的 0/1 Tensor，代表像素化天線蝕刻圖樣
###*                     座標範圍設定為 (0, 25, 0, 25)(行 15)
###*   - 響應(response): 依 port 模式不同，包含 S11/Gain(Single) 或 S11/S21/S22(Dual)
###*                     頻段為 n257；回傳的 response 是 Tensor 或 Tensor list/tuple，
###*                     具體格式與 simulate() 存檔時的 Data.data 結構一致(pickle)
###*
###* 資料來源與儲存路徑：
###*   - 掛載遠端 NAS：\\140.123.106.219\temp (T:)，帳號 user/ailab120
###*   - 讀寫目錄：DATASET_PATH / 'KuoHung Pattern'/
###*   - 檔案名稱：KuoHung-{name}.data (pickle，由 Data 類別管理)
###*   - KuoHung-1 對應 single_1() 定義的圖樣，KuoHung-2 對應 single_2()
###* ============================================================================
from antenna import *
config.device = "cpu"

from antenna.utils import *
from antenna.utils.data import Data

#* Select according to actual application.
from antenna.patch import SinglePortSimulator, DualPortSimulator
# AntennaPattern.setDefaultCoordinate((0, 200, 0, 1))

class KuoHung:
    """
    學長(KuoHung)參考天線管理類別。

    主要用途有二：
      (A) 互動式(物件實例) — 呼叫 simulate() 跑模擬器產生 .data 檔，再用 draw() 視覺化。
      (B) 靜態載入(load)   — train_single.py 等訓練腳本直接呼叫 KuoHung.load(name)
                             取得 (pattern_tensor, response_tensor)，無需建立實例。

    設計原則：
      - simulate() 只在尚無快取時才呼叫模擬器(開銷大)；有快取時 __call__ 直接 draw()。
      - load() 純讀快取，不觸發模擬；若快取不存在則拋 FileNotFoundError。
    """
    def __init__(self, name:str, port:Literal['Single', 'Dual']):
        #* Basic Config
        connect_network_drive("T:", r"\140.123.106.219\temp", "user", "ailab120")
        AntennaPattern.setDefaultCoordinate((0, 25, 0, 25))
        # PATTERN_SIZE = AntennaPattern.size(flatten=True)
        # RESPONSE_SIZE = AntennaResponse.size(flatten=True)

        self.name = f"KuoHung-{name}"              #* 最終的資料檔名前綴，例如 "KuoHung-1"
        self.result_path = DATASET_PATH.joinpath('KuoHung Pattern')  #* 存放 .data 快取的目錄

        self.data = Data(name=self.name, rootdir=self.result_path)  #* 負責 pickle 讀寫的 Data 物件

        match port:
            case 'Single':
                #* 單埠模式：S11 + Gain，圖表排版 1 行 × 3 欄(圖樣 + 2 條響應曲線)
                self.simulator = SinglePortSimulator(self.result_path)
                self.nrowcol = (1, 3)
                self.label = ('S11', 'Gain')

                AntennaResponse.registerLabels(*self.label, x = 'n257')

            case 'Dual':
                #* 雙埠模式：S11 + S21 + S22，圖表排版 2 行 × 2 欄
                self.simulator = DualPortSimulator(self.result_path)
                self.nrowcol = (2, 2)
                self.label = ('S11', 'S21', 'S22')

                AntennaResponse.registerLabels(*self.label, x = 'n257')

            case _:
                raise ValueError(f"{port}")

        AntennaPattern.register_simulator(self.simulator)
        self.x = AntennaResponse.x()        #* 頻率軸向量，用於繪圖 x 座標

    def __call__(self):
        if self.data.savepath.exists():
            #* 快取已存在 → 直接繪圖，跳過耗時的模擬步驟
            self.draw()
        else:
            #* 快取不存在 → 先模擬產生資料再繪圖
            self.simulate()
            self.draw()

    @staticmethod
    def load(name:str) -> tuple[Tensor, Tensor]:
        """
        靜態載入方法 — 訓練腳本的主要呼叫入口。

        從 DATASET_PATH/'KuoHung Pattern'/KuoHung-{name}.data 讀取
        已預先模擬好的 (pattern, response) 快取，直接回傳供 SM 暖身使用。

        Args:
            name (str): 參考天線編號，例如 '1' 或 '2'，
                        對應 single_1()/single_2() 所定義的圖樣。

        Returns:
            tuple[Tensor, Tensor]:
                - 第一項: pattern Tensor，形狀 (25, 25)，像素化天線蝕刻圖
                - 第二項: response Tensor(或 Tensor 序列)，包含 S11/Gain 等響應曲線
                  (與存檔時 Data.data 的格式一致，取決於建構時的 port 模式)

        Note:
            此方法不建立 KuoHung 實例，也不連 NAS、不開模擬器，純讀磁碟快取。
            若快取尚未產生，請先執行 `python KuoHung.py` 建立之。
        """
        return Data(name=f"KuoHung-{name}", rootdir=DATASET_PATH.joinpath('KuoHung Pattern')).load()

    def __str__(self):
        return f"<KuoHung name={self.name} file={self.data.savepath}>"

    def simulate(self, pattern:Optional[AntennaPattern] = None):
        """
        呼叫電磁模擬器，對指定圖樣進行全頻帶模擬並將結果存成 .data 快取。

        Args:
            pattern: 欲模擬的 AntennaPattern；若為 None 則使用 self.pattern
                     (需先呼叫 single_1() 或 single_2() 設定)。

        Side effect:
            在 self.result_path 寫入 {self.name}.data (pickle)，
            供後續 load() / draw() 直接讀取，避免重複開模擬器。
        """
        pattern = pattern or self.pattern
        self.simulator.open()           #* 啟動外部電磁模擬器(HFSS 或同類工具)
        self.simulator.start(self.name) #* 在模擬器中建立以 self.name 命名的專案
        response = pattern.simulate()   #* 執行模擬，取得 S 參數與增益等響應
        self.simulator.end()            #* 關閉模擬器，釋放資源

        #* ~pattern → 展平為 1D Tensor；~response → 展平後的響應向量
        #* 打包成 [pattern_flat, response_flat] 並落地為 pickle 快取
        data_in = Data([~pattern, ~response], name=self.name, rootdir=self.result_path)
        data_in.save()

    def draw(self):
        """
        從 .data 快取讀出 pattern 與 response，繪製天線圖樣與各響應曲線並存圖。

        繪圖佈局：
          - fig[0]  : AntennaPattern 像素圖
          - fig[1+] : 各響應曲線(S11/Gain 或 S11/S21/S22)，一曲線一子圖
        """
        data_result = Data(name=self.name, rootdir=self.result_path)
        KuoHung, responses = data_result.load()  #* KuoHung: pattern Tensor；responses: 響應序列

        with Figure(self.name, self.nrowcol, save=True, rootdir=self.result_path, size=(18, 9)) as fig:
            fig.addAll()

            AntennaPattern(KuoHung).plot(fig[0])  #* 子圖 0：像素化天線蝕刻圖

            for n, (label, response) in enumerate(zip(self.label, responses), start=1):
                #* 子圖 n：對應頻帶的響應曲線(S11 dB / Gain dB)
                fig[n].plot(self.x, response)
                fig[n].set_title(label)

    def single_1(self):
        """
        定義參考天線 #1 的圖樣：上半部全零(空白)、下半部中間 23 列導體。

        圖樣描述(25×25 像素)：
          - 上 12 行全 0 (無蝕刻)
          - 下 13 行：兩側各 1 列為 0，中間 23 列為 1(導體)
        """
        #* 13 * (0, 12, 1, 12, 0)
        upper_part = torch.zeros(12, 25)
        lower_part = torch.cat((torch.zeros(13, 1), torch.ones(13, 23), torch.zeros(13, 1)), dim=1)
        self.pattern = AntennaPattern(
            torch.cat((upper_part, lower_part), dim=0)
        )


    def single_2(self):
        """
        定義參考天線 #2 的圖樣：上半部全零(空白)、下半部全導體。

        圖樣描述(25×25 像素)：
          - 上 12 行全 0 (無蝕刻)
          - 下 13 行全 1 (完整導體，比 single_1 兩側少兩條空白)
        """
        #* 13 * (0, 12, 1, 12, 0)
        upper_part = torch.zeros(12, 25)
        lower_part = torch.ones(13, 25)
        self.pattern = AntennaPattern(
            torch.cat((upper_part, lower_part), dim=0)
        )

if __name__ == "__main__":
    base1 = KuoHung('1', port='Single')
    base2 = KuoHung('2', port='Single')

    pattern1, responses1 = base1.data.load()
    pattern2, responses2 = base2.data.load()


    with Figure('Base', (2,3), show=True, size=(18, 9), default_axes_title_size=16, default_tick_size=14) as fig:

            # #* Base 1
            # title1 = f"Base-1"

            # pattern1_ax = fig.index(-1)
            # AntennaPattern(pattern1).plot(pattern1_ax)
            # pattern1_ax.set_title(title1)

            # s1_ax = fig.index(-1)
            # s1_ax.plot(base1.x, responses1[0])
            # s1_ax.set_title(f"S11 ({title1})")

            # gain1_ax = fig.index(-1)
            # gain1_ax.plot(base1.x, responses1[1])
            # gain1_ax.set_title(f"Gain ({title1})")

            # #* Base 2
            # title2 = f"Base-2"

            # pattern2_ax = fig.index(-1)
            # AntennaPattern(pattern2).plot(pattern2_ax)
            # pattern2_ax.set_title(title2)

            # s2_ax = fig.index(-1)
            # s2_ax.plot(base1.x, responses2[0])
            # s2_ax.set_title(f"S11 ({title2})")

            # gain2_ax = fig.index(-1)
            # gain2_ax.plot(base2.x, responses2[1])
            # gain2_ax.set_title(f"Gain ({title2})")

            fig.addAll()

            for n, (base, responses) in enumerate([base1.data.load(), base2.data.load()], 0):
                title = f"Base-{n+1}"
                AntennaPattern(base).plot(fig[3*n])
                fig[3*n].set_title(title)


                fig[3*n+1].plot(base1.x, responses[0])
                fig[3*n+1].set_title(f"S11 ({title})")
                fig[3*n+1].set_xlabel("Frequency (GHz)")
                fig[3*n+1].set_ylabel("dB")

                fig[3*n+2].plot(base1.x, responses[1])
                fig[3*n+2].set_title(f"Gain ({title})")
                fig[3*n+2].set_xlabel("Frequency (GHz)")
                fig[3*n+2].set_ylabel("dB")
