# ==============================================================================
# figure.py — matplotlib Figure 的 with-context 包裝
# ------------------------------------------------------------------------------
# 從 utils.py 拆出 (純搬家，行為不變)。對外經 antenna.utils facade 取用：
#     from antenna.utils import Figure
# ==============================================================================
from typing import Any, Callable, Optional, TypeVar

from loguru import logger
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.figure import Figure as _Figure
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.axes._axes import Axes  # type: ignore
from torch import set_grad_enabled, is_grad_enabled
from tqdm import trange

from .utils import Path

ReturnType = TypeVar('ReturnType')

class Figure:
    """
    matplotlib Figure 的 with-context 包裝。

    解決兩個訓練常見痛點：
      1. 樣板繁瑣：自動處理 figure 尺寸、字型大小、子圖網格 (index/addAll)，
         離開 with 區塊時依 save/show 自動存圖或顯示、並 plt.close() 釋放記憶體
         (長訓練若不關 figure 會記憶體洩漏)。
      2. 梯度誤算：進入 context 時依 requires_grad 暫停 autograd，離開時還原；
         避免「畫圖時順手做的張量運算」誤入計算圖、污染 GEN/SM 的梯度。

    另提供 saveGIF / saveMP4 把每個 epoch 的圖串成動畫，呈現訓練演進。
    """
    def __init__(
            self,
            name:str,
            nrowcol:tuple = (1, 1),
            ncols:tuple = (0, 0),
            save:bool = False, show:bool = False,
            rootdir:Optional[str] = None,
            size:tuple = (18, 12),
            default_font_size = 12,
            default_axes_title_size = 20,
            default_tick_size:int = 18,
            requires_grad:bool = False,
            **kwargs
        ):
        """
        :param size: Example: (18, 12), (18 * 2, 9 * 2)
        :param kwargs: All plt.figure() arguments
        :param ncols: (total, cols) -> nrowcol=(total/cols, cols)

        ## Example
        ```
        with Figure("test_3_2", nrowcol=(2,2), save=True) as fig:
    
            ax1 = fig.index(1)
            ax1.set_title("test")
            ax1.plot([1, 2, 3, 4])

            fig.addAll()
            fig[2].set_title("test")
            fig[2].plot([1, 2, 3, 4])
        ```
        
        ## Set
        ```
        class:
            ...
            def plot(self, axes:Axes|None = None):
                ax:Axes = plt.axes(axes) # type: ignore
                ax.set_title("test")
                ax.plot([1, 2, 3, 4])
        ```

        ## 動畫
        ```
        line = {}
        epochs = 1500
        line = np.random.random((epochs))
            
        with Figure("line", rootdir=r'./') as fig:
            fig.addAll()
            def update(frame):
                fig[0].clear()
                fig[0].set_title("line")
                fig[0].set_xlim(0, epochs)
                fig[0].plot(line[:frame+1])

                fig.fig.tight_layout(pad=0.1)
                
                return fig
            fig.saveMP4(update, epochs, video_time=5)
        ```
        """
        from math import ceil
        fig = plt.figure(name, **kwargs)  # 以 name 當 figure 識別字 (相同 name 會復用同一張)
        fig.set_size_inches(*size)
        fig.tight_layout(pad=0.1)
        # 統一字型/標題/刻度大小，讓多張輸出圖風格一致 (便於並排比較或放進論文)。
        plt.rcParams.update({
            'font.size': default_font_size,
            'axes.titlesize': default_axes_title_size,
            'xtick.labelsize': default_tick_size,
            'ytick.labelsize': default_tick_size,
            'axes.labelsize': default_tick_size,
        })
        # fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)

        self.fig = fig
        self.save = save                 # 離開 context 時是否存圖
        self.show = show                 # 離開 context 時是否 plt.show()
        self.name = name
        # 子圖網格：若用 ncols=(總數, 欄數) 指定，則自動換算列數 (ceil)；否則直接用 nrowcol。
        self.nrowcol = (ceil(ncols[0] / ncols[1]), ncols[1]) if ncols[0] > 0 else nrowcol
        self.current_index = 1           # index() 自動遞增的子圖游標
        self.rootdir = Path(rootdir or "./")  # 圖檔輸出目錄
        self.requires_grad = requires_grad    # context 內是否保留梯度 (預設關閉)

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, nrowcol={self.nrowcol}, save={self.save}, show={self.show}, rootdir={self.rootdir.absolute()}, size={self.fig.get_size_inches()})"


    def index(self, index:int = 1, title:Optional[str] = None):
        """
        :param index: Support -1
        """
        # 在網格上取得/新增一個子圖 Axes 並回傳。傳 index=-1 表「沿用目前游標」，
        # 因此 train_single/dual 內常連續呼叫 fig.index(-1) 一格一格往下擺子圖。
        self.current_index = self.current_index if index == -1 else index
        ax = self.fig.add_subplot(self.nrowcol[0], self.nrowcol[1], self.current_index)
        self.current_index += 1  # 取完即前進，下一次 -1 會落到下一格

        if title is not None:
            ax.set_title(title)
        return ax

    def addAll(self):
        # 一次把整個網格 (nrow*ncol) 的子圖全部建出來，之後即可用 fig[i] 索引存取。
        for i in range(self.__len__()) :
            self.index(i+1)
            
    def convert_to(self, fn:Callable[[_Figure], ReturnType]) -> ReturnType:
        """
        Convert to the specified type.

        :param fn: Convert function. Ex: wandb.Image

        Example::

            fig.conver_to(wandb.Image)
        """
        # 把內部 matplotlib Figure 交給轉換函式 (例如 wandb.Image) 並回傳其結果，
        # 方便把圖直接上傳到實驗追蹤平台。
        return fn(self.fig)

    def saveGIF(self, update:Callable, epochs:int = 10, dpi = 150):
        # 用 PillowWriter 把 update(frame) 逐格畫出的內容串成 GIF，存到 rootdir/name.gif。
        # progress_callback 掛 tqdm，讓動畫輸出也有進度條。
        writer = PillowWriter(fps=30, metadata={"artist": "WeiWen Wu"})
        tqdm_iter = trange(epochs, desc="Plotting")
        ani = FuncAnimation(self.fig, update, frames=epochs)
        ani.save(f"{self.rootdir.joinpath(self.name)}.gif", writer=writer, dpi=dpi, progress_callback=lambda i, n: tqdm_iter.update())

    def saveMP4(self, update:Callable[[int], "Figure"], epochs:int = 10, dpi = 150, video_time = None, del_temp = False):
        # 產出 MP4 影片。策略是「先把每一格畫成 PNG 暫存，再把 PNG 串成影片」——
        # 比直接用 FuncAnimation 重畫複雜圖更穩 (避免狀態殘留)，代價是多一次磁碟暫存。
        from imageio_ffmpeg import get_ffmpeg_exe #? pip install imageio-ffmpeg
        metadata = {
            'title': f'{self.name}',
            "artist": "WeiWen Wu",
            'comment': "Provided by WeiWen's kit"
        }
        rcParams['animation.ffmpeg_path'] = get_ffmpeg_exe()  # 指向 imageio 內附的 ffmpeg

        # 第一階段：逐 epoch 呼叫 update(n) 重畫，並把該格存成 PNG 暫存檔。
        path_video_temp = self.rootdir.joinpath('video_temp').not_exist_create()
        path_merges:list[Path] = []
        for n in trange(epochs, desc='Creating'):
            self.fig.clear()
            path_merges.append(
                update(n).saveIMG(
                    path_video_temp.joinpath(f'{n}.png')
                )
            )
        # 第二階段：動畫實際只是「把第 frame 張 PNG 讀進來貼滿畫面」。
        def _update(frame):
            plt.clf()
            plt.imshow(
                plt.imread(path_merges[frame])
            )
            plt.axis('off')
            plt.tight_layout(pad=0)
            return self

        # 由 video_time(秒) 反推 fps，並夾在 1~120 之間避免極端值；video_time 未給則用 30。
        fps = int(epochs/video_time) if video_time else 30
        writer = FFMpegWriter(fps=max(1, min(fps, 120)), metadata=metadata) # , bitrate=1800
        filename = self._ani_save(_update, epochs, writer, dpi)
        writer.finish()
        logger.info(f'Video creation completed. ({filename.absolute()}, fps: {fps})')
        if del_temp: path_video_temp.rmtree()  # 視需要清掉中途的 PNG 暫存目錄


    def _ani_save(self, update: Callable[[int], Any], epochs, writer, dpi):
        # saveGIF/saveMP4 共用的底層存檔：FuncAnimation 逐格呼叫 update 並寫出，
        # progress_callback 掛 tqdm 顯示輸出進度。回傳最終檔名供記 log。
        tqdm_iter = trange(epochs, desc="Plotting")
        ani = FuncAnimation(self.fig, update, frames=epochs)
        filename = self.rootdir.joinpath(f"{self.name}.mp4")
        ani.save(
            filename, writer=writer, dpi=dpi,
            progress_callback=lambda i, n: tqdm_iter.update(),
        )
        return filename
        
    
    def saveIMG(self, path = None):
        # 把目前 figure 存成 PNG。注意這裡的 FIG_CONFIG 是區域變數，刻意用白底
        # (facecolor/edgecolor='white') 覆蓋模組頂端那份透明設定 —— 因為這些圖會被
        # saveMP4 讀回貼進影片，透明背景會變黑，故統一改白底。
        FIG_CONFIG = {
            "format": 'png',
            "bbox_inches": "tight",
            "pad_inches": 0.1,
            "dpi": 300,
            "transparent": True,
            "facecolor": "white", # white or none
            "edgecolor": "white", # white or none
        }
        # self.fig.set_size_inches(18, 12)
        path = path or self.rootdir.joinpath(f"{self.name}.png")  # 未指定就用 rootdir/name.png
        plt.savefig(path, **FIG_CONFIG)
        return path  # 回傳實際存檔路徑 (saveMP4 靠它收集每格 PNG)

    def __getitem__(self, index:int) -> Axes:
        """
        Use first
        ```
        fig.addAll()
        ```
        """
        # fig[i] 取第 i 個已建立的子圖 Axes；前提是已先 addAll() 把子圖都建出來。
        return self.fig.get_axes()[index]

    def __len__(self) -> int:
        # 子圖總數 = 列 × 欄；addAll() 與 fig[i] 邊界都以此為準。
        return self.nrowcol[0] * self.nrowcol[1]

    def __enter__(self):
        # 進入 with：記下目前的 autograd 開關，再切到本 Figure 指定的 requires_grad。
        # 預設關閉梯度，確保畫圖時的張量運算不會被記進計算圖。
        self.prev = is_grad_enabled()
        set_grad_enabled(self.requires_grad)

        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback):
        # 離開 with：唯有「沒有發生例外」時才 show/save (避免存出半成品的錯誤圖)。
        if not exc_type:
            if self.show: plt.show()
            if self.save: self.saveIMG()
        plt.close()                  # 一律關閉 figure，釋放記憶體 (長訓練必要)
        set_grad_enabled(self.prev)  # 還原進入前的 autograd 狀態

