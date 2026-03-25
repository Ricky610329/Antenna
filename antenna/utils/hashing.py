from time import time
from typing import (
    Literal,
    overload,
)


class TID:
    """
    TID (Time-based ID) Generator
    支援輸出格式：Base62 字串 或 Integer (偏移數值)
    """

    import string

    # 設定基準時間 (Epoch): 2001-09-28 00:00:00 UTC
    CUSTOM_EPOCH = 1001635200

    # Base62 字元集
    ALPHABET = string.digits + string.ascii_lowercase + string.ascii_uppercase
    BASE = len(ALPHABET)

    @classmethod
    @overload
    def generate(cls, timestamp: int | None = None, as_int: Literal[False] = False) -> str: ...
    @classmethod
    @overload
    def generate(cls, timestamp: int | None = None, as_int: Literal[True] = ...) -> int: ...

    @classmethod
    def generate(cls, timestamp: int = None, as_int: bool = False) -> str | int:
        """
        產生 TID。
        :param timestamp: 指定時間戳，若無則使用當前時間
        :param as_int: True 回傳整數 (偏移值); False 回傳 Base62 字串 (預設)
        """
        if timestamp is None:
            timestamp = int(time())

        # 計算偏移量 (ID 本體)
        delta = timestamp - cls.CUSTOM_EPOCH

        if delta < 0:
            raise ValueError("時間早於基準點 2001-09-28，無法產生 ID")

        # 若使用者想要 Int，直接回傳偏移後的數值
        if as_int:
            return delta

        # 若為 0 的邊界情況
        if delta == 0:
            return cls.ALPHABET[0]

        # 進行 Base62 編碼
        arr = []
        num = delta
        while num:
            num, rem = divmod(num, cls.BASE)
            arr.append(cls.ALPHABET[rem])

        arr.reverse()
        return "".join(arr)

    @classmethod
    def decode(cls, tid: str | int) -> int:
        """
        將 TID (字串或整數) 還原為原始 Unix Timestamp
        """
        # 如果傳入的是整數 (Offset Int)，直接加回 Epoch
        if isinstance(tid, int):
            return tid + cls.CUSTOM_EPOCH

        # 如果是字串，先解 Base62
        num = 0
        for char in tid:
            if char not in cls.ALPHABET:
                raise ValueError(f"非法字元: {char}")
            num = num * cls.BASE + cls.ALPHABET.index(char)

        return num + cls.CUSTOM_EPOCH


def get_shake_128(text: str, length: int = 6) -> str:
    """Generate a shake_128 ID."""
    from hashlib import shake_128

    return shake_128(text.encode()).hexdigest(length // 2 + 1)[:length]
