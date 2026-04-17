from os.path import getctime
from pathlib import Path as _Path
from shutil import rmtree as _rmtree

from torch import __version__
from torch import (
    load as _torch_load,
)


class Path(type(_Path())):  # type: ignore
    def __new__(cls, *args, **kwargs):
        kwargs.pop("create", None)
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, *args, create: bool = False, **kwargs):
        """
        Path model.

        ## Usage
        ```python
        path = Path("./path/to/file.ext")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
        path.unlink()
        path.del_from_glob('*.pth')
        path.manage_file_count('*.pth', keep_latest=3)
        path.load_torch()
        path.not_exist_create(create_file=True)
        ```
        """
        # Python 3.12+：PurePath 狀態（_raw_paths 等）在 __init__ 建立，需呼叫 super。
        super().__init__(*args, **kwargs)

        if create:
            self.not_exist_create()

    def __reduce__(self):
        return (self.__class__, (str(self),))

    def rmtree(self) -> bool:
        if self.is_dir():
            _rmtree(self)
            return True
        else:
            return False

    def not_exist_create(self, create_file: bool = False):
        """
        Create the path if it does not exist.

        :param create_file: Whether to create the file.
        :return: Whether the path does not exist.
        """
        if self.suffix:
            self.parent.mkdir(parents=True, exist_ok=True)
            if create_file:
                self.touch(exist_ok=True)
        else:  # No file extension, treated as a directory.
            self.mkdir(parents=True, exist_ok=True)
        return self

    def del_from_glob(self, pattern: str):
        """
        Delete all files matching the pattern.

        :param pattern: Patterns matching files, E.g., '*.pth'
        """
        if not self.suffix:
            paths = list(self.glob(pattern))
            for path in paths:
                path.unlink()
        else:
            self.unlink()

    def manage_file_count(self, file: str, keep_latest: int | None = 3):
        """
        Manage the number of archives and only keep the latest specified number.

        :param file: Patterns matching archives, E.g., '*.pth'
        :param keep_latest: Latest quantity to keep.
        """
        if keep_latest is None:
            return False

        # Confirm that the target directory exists.
        if not self.exists():
            raise FileNotFoundError(f"The destination directory ({self.absolute()}) does not exist.")

        # Get all files matching the pattern.
        files_sorted = sorted(self.glob(file), key=getctime)

        # If the file exceeds the limit, delete the oldest file.
        if len(files_sorted) > keep_latest:
            for old_backup in files_sorted[: len(files_sorted) - keep_latest]:
                if not old_backup.rmtree():
                    old_backup.unlink()
            return True
        else:
            return False

    def load_torch(self, device=None):
        # Deferred import 避免循環相依。
        from antenna.utils.config import config

        if __version__ >= "2.6.0":
            return _torch_load(self, weights_only=False, map_location=device or config.device)
        else:
            return _torch_load(self, map_location=device or config.device)
