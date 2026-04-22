from collections.abc import Callable
from math import ceil
from typing import (
    Any,
    TypeVar,
)

import matplotlib.pyplot as plt
from loguru import logger

# * Figure
from matplotlib import rcParams
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.axes._axes import Axes  # type: ignore
from matplotlib.figure import Figure as _Figure
from torch import (
    is_grad_enabled,  # with no_grad():...
    set_grad_enabled,
)
from tqdm import trange

from antenna.utils.path import Path

ReturnType = TypeVar("ReturnType")

FIG_CONFIG = {
    "format": "png",
    "bbox_inches": "tight",
    "pad_inches": 0.1,
    "dpi": 300,
    "transparent": True,
    "facecolor": "none",  # white
    "edgecolor": "none",
}
TQDM_CONFIG = {"unit": "epoch", "unit_scale": True, "mininterval": 1.0, "dynamic_ncols": True}
TQDM_BAR_SIMPLE = "{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}"


def plot(x, file_name: str | None = None) -> None:
    """
    Plot the weight matrix on a 3D graph
    """
    # This part is for plotting the graph
    plt.clf()
    # plt.figure(figsize=(20, 10))
    plt.title("")
    plt.plot(x)
    plt.legend()

    plt.show()

    if file_name:
        plt.savefig(file_name, **FIG_CONFIG)


class Figure:
    def __init__(
        self,
        name: str,
        nrowcol: tuple = (1, 1),
        ncols: tuple = (0, 0),
        save: bool = False,
        show: bool = False,
        rootdir: str | None = None,
        size: tuple = (18, 12),
        default_font_size=12,
        default_axes_title_size=20,
        default_tick_size: int = 18,
        requires_grad: bool = False,
        **kwargs,
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
        fig = plt.figure(name, **kwargs)
        fig.set_size_inches(*size)
        fig.tight_layout(pad=0.1)
        plt.rcParams.update(
            {
                "font.size": default_font_size,
                "axes.titlesize": default_axes_title_size,
                "xtick.labelsize": default_tick_size,
                "ytick.labelsize": default_tick_size,
                "axes.labelsize": default_tick_size,
            }
        )
        # fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)

        self.fig = fig
        self.save = save
        self.show = show
        self.name = name
        self.nrowcol = (ceil(ncols[0] / ncols[1]), ncols[1]) if ncols[0] > 0 else nrowcol
        self.current_index = 1
        self.rootdir = Path(rootdir or "./")
        self.requires_grad = requires_grad

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, nrowcol={self.nrowcol}, save={self.save}, show={self.show}, rootdir={self.rootdir.absolute()}, size={self.fig.get_size_inches()})"

    def index(self, index: int = 1, title: str | None = None):
        """
        :param index: Support -1
        """
        self.current_index = self.current_index if index == -1 else index
        ax = self.fig.add_subplot(self.nrowcol[0], self.nrowcol[1], self.current_index)
        self.current_index += 1

        if title is not None:
            ax.set_title(title)
        return ax

    def addAll(self):
        for i in range(self.__len__()):
            self.index(i + 1)

    def convert_to(self, fn: Callable[[_Figure], ReturnType]) -> ReturnType:
        """
        Convert to the specified type.

        :param fn: Convert function. Ex: wandb.Image

        Example::

            fig.conver_to(wandb.Image)
        """
        return fn(self.fig)

    def saveGIF(self, update: Callable, epochs: int = 10, dpi=150):
        writer = PillowWriter(fps=30, metadata={"artist": "WeiWen Wu"})
        tqdm_iter = trange(epochs, desc="Plotting")
        ani = FuncAnimation(self.fig, update, frames=epochs)
        ani.save(
            f"{self.rootdir.joinpath(self.name)}.gif",
            writer=writer,
            dpi=dpi,
            progress_callback=lambda i, n: tqdm_iter.update(),
        )

    def saveMP4(self, update: Callable[[int], "Figure"], epochs: int = 10, dpi=150, video_time=None, del_temp=False):
        from imageio_ffmpeg import get_ffmpeg_exe  # ? pip install imageio-ffmpeg

        metadata = {"title": f"{self.name}", "artist": "WeiWen Wu", "comment": "Provided by WeiWen's kit"}
        rcParams["animation.ffmpeg_path"] = get_ffmpeg_exe()

        path_video_temp = self.rootdir.joinpath("video_temp").not_exist_create()
        path_merges: list[Path] = []
        for n in trange(epochs, desc="Creating"):
            self.fig.clear()
            path_merges.append(update(n).saveIMG(path_video_temp.joinpath(f"{n}.png")))

        def _update(frame):
            plt.clf()
            plt.imshow(plt.imread(path_merges[frame]))
            plt.axis("off")
            plt.tight_layout(pad=0)
            return self

        fps = int(epochs / video_time) if video_time else 30
        writer = FFMpegWriter(fps=max(1, min(fps, 120)), metadata=metadata)  # , bitrate=1800
        filename = self._ani_save(_update, epochs, writer, dpi)
        writer.finish()
        logger.info(f"Video creation completed. ({filename.absolute()}, fps: {fps})")
        if del_temp:
            path_video_temp.rmtree()

    def _ani_save(self, update: Callable[[int], Any], epochs, writer, dpi):
        tqdm_iter = trange(epochs, desc="Plotting")
        ani = FuncAnimation(self.fig, update, frames=epochs)
        filename = self.rootdir.joinpath(f"{self.name}.mp4")
        ani.save(
            filename,
            writer=writer,
            dpi=dpi,
            progress_callback=lambda i, n: tqdm_iter.update(),
        )
        return filename

    def saveIMG(self, path=None):
        save_config = {**FIG_CONFIG, "facecolor": "white", "edgecolor": "white"}
        path = path or self.rootdir.joinpath(f"{self.name}.png")
        plt.savefig(path, **save_config)
        return path

    def __getitem__(self, index: int) -> Axes:
        """
        Use first
        ```
        fig.addAll()
        ```
        """
        return self.fig.get_axes()[index]

    def __len__(self) -> int:
        return self.nrowcol[0] * self.nrowcol[1]

    def __enter__(self):
        self.prev = is_grad_enabled()
        set_grad_enabled(self.requires_grad)

        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback):
        if not exc_type:
            if self.show:
                plt.show()
            if self.save:
                self.saveIMG()
        plt.close()
        set_grad_enabled(self.prev)
