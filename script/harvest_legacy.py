# -*- coding: utf-8 -*-
"""
script/harvest_legacy.py — 收割學長 result/ 各 run 夾的 online.dataset，
依響應形狀分成 single / dual 兩包新格式 (SampleStore 一筆一檔)。

來源：每個 run 夾的 online.dataset 是學長 DataManager 落地的 list[(pattern, response)]，
      都是真實 HFSS 模擬過的樣本 —— 最貴、最值得重用。

分類：以「響應通道數」判定，與夾名無關 (更穩)：
      y.shape[0] == 2  -> single (S11 + Gain)
      y.shape[0] == 3  -> dual   (S11 + S21 + S22)

去重：SampleStore 以「內容 hash 即檔名」自動去重 —— 同一張 pattern 被多個 run
      重複模擬只留一份；跨 run 收割可重複執行，結果一致 (冪等)。

★安全鐵則★ (學長資料絕不可動)：
  - 來源 (--src) 全程「唯讀」：用裸 pickle.load，不碰 DataManager
    (DataManager 建構時會在 rootdir 寫 .log → 會污染學長夾，故刻意不用)。
  - 目的地 (--dst) 必須在使用者自己的 NAS；若解析後路徑落在「碩二」樹下，直接中止。
  - 只新增、零刪除、零覆寫學長任何檔案。

用法：
    python -m script.harvest_legacy            # 用下方預設路徑
    python -m script.harvest_legacy --src "<學長 result>" --dst "<你的 dataset 夾>"
"""
import sys
import pickle
from os.path import dirname, join
from argparse import ArgumentParser

sys.path.append(join(dirname(__file__), ".."))

import torch  # noqa: 讓 pickle 還原 tensor
from loguru import logger

from antenna.utils import Path
from antenna.utils.store import SampleStore

# 預設路徑：學長 result/ (讀) → 使用者自己的 NAS (寫)
DEFAULT_SRC = Path(r"T:\碩二_吳維文's\Patch Antenna\Experiment\result")
DEFAULT_DST = Path(r"T:\碩一_鄒穎麒's\antenna\dataset")

FORBIDDEN = "碩二"  # 目的地若含此字串 → 拒絕寫入 (保護學長資料)


def load_raw(path: Path):
    """唯讀載入 online.dataset (裸 pickle，不經 DataManager)。"""
    with open(path, "rb") as f:
        return pickle.load(f)


def as_tensor(o):
    return o if isinstance(o, torch.Tensor) else torch.as_tensor(o)


def harvest(src: Path, dst: Path) -> None:
    # --- 安全鎖：目的地不得落在學長樹下 ---
    if FORBIDDEN in str(dst.resolve()):
        raise SystemExit(f"[拒絕] 目的地 {dst} 位於「{FORBIDDEN}」樹下，不允許寫入學長資料夾。")

    logger.info(f"來源 (唯讀): {src}")
    logger.info(f"目的地 (只新增): {dst}")
    single = SampleStore(dst / "harvest_single")
    dual = SampleStore(dst / "harvest_dual")

    folders = sorted([d for d in src.iterdir() if d.is_dir()])
    n_skip = n_err = n_odd = 0
    add_s = add_d = 0

    for d in folders:
        od = d / "online.dataset"
        if not od.exists():
            n_skip += 1
            continue
        try:
            data = load_raw(od)
        except Exception as e:
            n_err += 1
            logger.error(f"載入失敗 {d.name}: {type(e).__name__}: {e}")
            continue
        if not isinstance(data, (list, tuple)) or not data:
            n_skip += 1
            continue

        f_s = f_d = 0
        for pair in data:
            x, y = pair
            x, y = as_tensor(x), as_tensor(y)
            ch = y.shape[0] if y.ndim >= 1 else None
            if ch == 2:
                f_s += single.add(x, y)
            elif ch == 3:
                f_d += dual.add(x, y)
            else:
                n_odd += 1
        add_s += f_s
        add_d += f_d
        logger.info(f"{d.name:60s} n={len(data):4d} → single+{f_s} dual+{f_d}")

    logger.success("=" * 70)
    logger.success(f"single 收割完成：{len(single)} 筆 (本次新增 {add_s}) → {single.rootdir}")
    logger.success(f"dual   收割完成：{len(dual)} 筆 (本次新增 {add_d}) → {dual.rootdir}")
    if n_odd:
        logger.warning(f"略過 {n_odd} 筆非 2/3 通道的異常響應。")
    if n_err:
        logger.warning(f"{n_err} 個夾載入失敗 (見上)。")
    logger.info(f"略過 {n_skip} 個夾 (無 online.dataset)。學長來源全程未寫入。")


if __name__ == "__main__":
    p = ArgumentParser(description="收割學長 online.dataset → single/dual 兩包 SampleStore")
    p.add_argument("--src", default=None, help="學長 result/ 根目錄 (唯讀)")
    p.add_argument("--dst", default=None, help="輸出 dataset 根目錄 (你自己的 NAS)")
    a = p.parse_args()
    harvest(Path(a.src) if a.src else DEFAULT_SRC,
            Path(a.dst) if a.dst else DEFAULT_DST)
