from json import dump as _json_dump
from json import load as _json_load
from typing import (
    Any,
    overload,
)

from antenna.utils.path import Path


class json:
    """
    ### Example
    ```
    from utils.utils import json
    _json = json('static/config.json')
    print(_json('base/UPLOAD_FOLDER'))
    _json_data = _json.load()
    print(_json_data['success'])
    _json_data['success'] = False
    _json.dump(_json_data)
    ```
    """

    def __init__(self, path: str, create: bool = True) -> None:
        self.path = Path(path)

        if not self.path.exists():
            if create:
                self.path.touch()
                self.dump({})
            else:
                raise FileNotFoundError(f"JSON file '{path}' does not exist.")

    @overload
    def __call__(self, key: str) -> Any:
        """
        Get the value of the specified key in the JSON file.

        ### Example
        >>> _json('base/UPLOAD_FOLDER')
        """

    ...

    @overload
    def __call__(self, key: str, value: Any) -> dict:
        """
        Set the value of the specified key in the JSON file.

        ### Example
        >>> _json('base/UPLOAD_FOLDER', 'new/path')
        """

    ...

    def __call__(self, key: str, value=None):
        keys = key.split("/")
        if value is not None:
            if value == "null":
                value = None
            if value in ["True", "true"]:
                value = True
            if value in ["False", "false"]:
                value = False
            result = self._set(keys, value)
            self.dump(result)
            return result
        else:
            return self._get(keys)

    def __getitem__(self, key):
        return self.__call__(key, value=None)

    def __setitem__(self, key, value):
        return self.__call__(key, value)

    def get(self, key: str, default=None):
        keys = key.split("/")
        try:
            return self._get(keys)
        except KeyError:
            result = self._set(keys, default)
            self.dump(result)
            return default

    def _set(self, keys: list, value: Any) -> dict:
        temp = self.load().copy()
        _ = "temp"
        for i, k in enumerate(keys):
            if k == "":
                continue
            _ += f"['{k}']"

            if i == len(keys) - 1:
                exec(f"{_} = value")
            else:
                if k not in temp:
                    exec(f"{_} = {{}}")

        return temp

    def _get(self, keys: list) -> Any:

        self.data = self.load()
        result = self.data.copy()
        for k in keys:
            if k == "":
                continue
            result = result[k]
        return result

    def load(self) -> dict:
        with open(self.path, encoding="utf-8") as f:
            return _json_load(f)

    def dump(self, data: dict) -> bool:
        with open(self.path, "w", encoding="utf-8") as f:
            _json_dump(data, f, ensure_ascii=False, indent=4)
        return True

    def delete(self, key: str) -> bool:
        keys = key.split("/")
        data = self.load()

        # Traversing through the keys
        temp = data
        for k in keys[:-1]:  # Get to the parent of the key to delete
            if k in temp:
                temp = temp[k]
            else:
                return False  # If the key doesn't exist, return False

        # Deleting the key
        if keys[-1] in temp:
            del temp[keys[-1]]
            self.dump(data)  # Save the updated data back to the file
            return True
        else:
            return False
