# -*- coding: utf-8 -*-
"""
script/convert_dataset.py — 舊資料集 (DataManager 單一 pickle) → 新格式 (SampleStore 一筆一檔)。

用法 (正式機，確認後才執行)：
    python -m script.convert_dataset patch_single_mirror
    python -m script.convert_dataset patch_dual --rootdir "T:\\...\\dataset"

行為：
- 讀 <rootdir>/<name>.data (舊 pickle)，逐筆寫到 <rootdir>/<name>/ (新資料夾)。
- 「不刪除」舊檔 —— 學長既有 code 仍可用；轉換後 train.py 的 load_dataset
  會優先偵測到資料夾而自動改走新格式。
- 可重複執行 (hash 去重，跑兩次結果相同)。
"""
import sys
from os.path import dirname, join
sys.path.append(join(dirname(__file__), '..'))

from argparse import ArgumentParser

from antenna.utils import DATASET_PATH, logger
from antenna.legacy import DataManager
from antenna.utils.store import SampleStore


def convert(name: str, rootdir) -> None:
    old = DataManager(name, rootdir=rootdir, verbose=True)
    if len(old) == 0:
        logger.warning(f"'{name}' 是空的，不轉換。")
        return
    new = SampleStore(rootdir.joinpath(name))
    added = sum(new.add(x, y) for x, y in old)
    logger.success(f"轉換完成：{name} 共 {len(old)} 筆 → 新增 {added} 筆 "
                   f"(重複 {len(old) - added})，新格式位於 {new.rootdir}")
    logger.info(f"舊檔 {name}.data 保留未動 (學長 code 仍可用)。")


if __name__ == "__main__":
    parser = ArgumentParser(description="舊資料集 pickle → 一筆一檔 SampleStore")
    parser.add_argument("name", help="資料集名稱 (如 patch_single_mirror)")
    parser.add_argument("--rootdir", default=None, help="資料集根目錄 (預設 DATASET_PATH)")
    args = parser.parse_args()
    from antenna.utils import Path
    convert(args.name, Path(args.rootdir) if args.rootdir else DATASET_PATH)
