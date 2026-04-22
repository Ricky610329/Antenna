import json
import os
import pathlib
import pickle
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

# 在 import pyplot 前設定 matplotlib backend，避免 GUI backend 問題
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import torch
from flask import Flask, abort, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from antenna.utils import Record, config
from antenna.utils.data import Data, DataManager

# --- 模式與路徑設定 ---
# 透過 '-dev' 參數啟用開發模式
DEV_MODE = "-dev" in sys.argv
if DEV_MODE:
    print("----- Running in Development Mode (debug=True) -----")
else:
    print("----- Running in Production Mode (debug=False) -----")

# 實驗結果與資料集目錄（皆指向網路磁碟）
RESULT_DIR = Path(r"T:\碩二_吳維文's\Patch Antenna\Experiment\result")
DATASET_DIR = Path(r"T:\碩二_吳維文's\Patch Antenna\Experiment\dataset")

# 本地暫存目錄（用於產生下載檔）
TEMP_DIR = Path(os.getcwd()).joinpath("temp_downloads").absolute()
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# 強制使用 CPU，檢視器不需 CUDA
config.device = "cpu"

app = Flask(__name__)


# 舊版 pickle 中的 Path 類別位於 antenna.utils.utils.Path，
# 此 Unpickler 將其重新導向至標準 pathlib.Path，以便載入舊記錄。
class PathFixUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "Path" and "antenna.utils" in module:
            return pathlib.Path
        return super().find_class(module, name)


# --- 線上使用者追蹤 ---
ACTIVE_USERS = {}  # {ip: last_seen_datetime}
TIMEOUT_MINUTES = 5


@app.before_request
def track_active_users():
    """更新請求來源 IP 的最後活躍時間戳。"""
    try:
        # 若位於反向代理後方，優先採用 X-Forwarded-For 第一個 IP
        forwarded = request.headers.getlist("X-Forwarded-For")
        ip = forwarded[0].split(",")[0].strip() if forwarded else request.remote_addr
        ACTIVE_USERS[ip] = datetime.now()
    except Exception:
        pass


def get_active_ip_list():
    """回傳最近 TIMEOUT_MINUTES 分鐘內活躍過的 IP 列表，並清理過期項目。"""
    cutoff = datetime.now() - timedelta(minutes=TIMEOUT_MINUTES)
    expired_ips = [ip for ip, ts in ACTIVE_USERS.items() if ts < cutoff]
    for ip in expired_ips:
        del ACTIVE_USERS[ip]
    return sorted(ACTIVE_USERS.keys())


# --- 快取 ---
RECORD_CACHE = {}  # {record_id: {'key': (rec_mtime, pic_mtime), 'data': record_dict}}

# 僅允許字母、數字、底線、連字號、點；用於驗證 URL 中的 record/dataset 名稱，避免路徑遍歷
_SAFE_NAME_RE = re.compile(r"^[\w\-.]+$")


def _validate_name(name: str) -> bool:
    """檢查名稱是否僅含安全字元（非 .. 且無路徑分隔符）。"""
    return bool(name) and name not in (".", "..") and _SAFE_NAME_RE.match(name) is not None


def natural_sort_key(s):
    """自然排序鍵（1, 2, 10 而非 1, 10, 2），接受 Path 或字串。"""
    name = s.name if hasattr(s, "name") else s
    return [int(text) if text.isdigit() else text.lower() for text in re.split("([0-9]+)", name)]


@app.route("/")
def index():
    """提供主頁骨架，實際資料由 /api/main-page-data 非同步載入。"""
    return render_template("index.html")


def _build_record_entry(d: Path) -> dict:
    """建立單一 record 目錄的資料 dict（使用 mtime 做快取鍵以減少重算）。"""
    rec_mtime = d.stat().st_mtime
    pic_dir = d / "pic"
    pic_mtime = pic_dir.stat().st_mtime if pic_dir.exists() else 0

    cache_key = (rec_mtime, pic_mtime)
    cached = RECORD_CACHE.get(d.name)
    if cached and cached["key"] == cache_key:
        return cached["data"]

    best_image = None
    if pic_dir.exists():
        images = sorted(pic_dir.glob("*.png"), key=natural_sort_key)
        if images:
            best_candidates = [img for img in images if "best" in img.name.lower()]
            best_image = (best_candidates[-1] if best_candidates else images[-1]).name

    stat = d.stat()
    record_data = {"id": d.name, "mtime": stat.st_mtime, "ctime": stat.st_ctime, "best_image": best_image}
    RECORD_CACHE[d.name] = {"key": cache_key, "data": record_data}
    return record_data


@app.route("/api/main-page-data")
def get_main_page_data():
    """主頁一次取得 records / datasets / active_ips。"""
    records = []
    if RESULT_DIR.exists():
        records = [_build_record_entry(d) for d in RESULT_DIR.iterdir() if d.is_dir()]
        records.sort(key=lambda x: x.get("mtime", 0), reverse=True)

    datasets = []
    if DATASET_DIR.exists():
        for f in DATASET_DIR.glob("*.dataset"):
            stat = f.stat()
            datasets.append({"id": f.stem, "name": f.name, "mtime": stat.st_mtime, "size": stat.st_size})
        datasets.sort(key=lambda x: x.get("mtime", 0), reverse=True)

    return jsonify({"records": records, "datasets": datasets, "active_ips": get_active_ip_list()})


@app.route("/generator")
def generator():
    """提供 pattern 產生器頁面。"""
    return render_template("generator.html")


@app.route("/api/save_generated_pattern", methods=["POST"])
def save_generated_pattern():
    """將使用者自訂 pattern 儲存為 .data 檔並回傳下載。"""
    try:
        payload = request.json or {}
        grid = payload.get("grid")
        raw_filename = payload.get("filename", "custom_pattern")

        if not grid:
            return "No grid data provided", 400

        filename = secure_filename(raw_filename) or "custom_pattern"

        tensor_data = torch.from_numpy(np.array(grid, dtype=np.float32))
        data_obj = Data(tensor_data, name=filename, rootdir=TEMP_DIR, suffix="data", load=False)
        data_obj.save()

        return send_from_directory(directory=str(TEMP_DIR), path=f"{filename}.data", as_attachment=True)

    except Exception as e:
        print(f"Exception in save_generated_pattern: {e}")
        return f"{type(e).__name__}: {str(e)}", 500


def _to_list(tensor_or_arr):
    """將 tensor / ndarray 轉為 list，便於 JSON 序列化。"""
    if tensor_or_arr is None:
        return None
    if hasattr(tensor_or_arr, "tolist"):
        return tensor_or_arr.tolist()
    return tensor_or_arr


@app.route("/dataset/<dataset_name>")
def view_dataset(dataset_name):
    """檢視特定 dataset 內容（前 50 個樣本）。"""
    if not _validate_name(dataset_name):
        abort(400, description="Invalid dataset name")
    try:
        dm = DataManager(dataset_name, rootdir=DATASET_DIR, verbose=False)
        total_items = len(dm)

        # 限制前 50 個樣本以免瀏覽器記憶體爆炸
        limit = 50
        samples = []
        for i in range(min(total_items, limit)):
            item = dm[i]
            if isinstance(item, (list, tuple)):
                inp, outp = item[0], item[1]
            else:
                inp, outp = item, None
            samples.append({"id": i, "input": _to_list(inp), "output": _to_list(outp)})

        return render_template("dataset.html", dataset_name=dataset_name, total_items=total_items, samples=samples)
    except Exception as e:
        return f"Error loading dataset: {str(e)}", 500


@app.route("/record/<record_id>")
def view_record(record_id):
    """檢視特定 record 的骨架頁（真實資料由 /api/record 取得）。"""
    if not _validate_name(record_id):
        abort(400, description="Invalid record id")
    record_path = RESULT_DIR / record_id
    if not record_path.exists():
        abort(404, description="Record not found")

    return render_template("record.html", record_id=record_id)


@app.route("/api/record/<record_id>")
def get_record_data(record_id):
    """取得特定 record 的訓練指標、矩陣資料、影像與設定。"""
    if not _validate_name(record_id):
        abort(400, description="Invalid record id")
    record_path = RESULT_DIR / record_id
    if not record_path.exists():
        abort(404, description="Record not found")

    data_dict: dict = {}
    matrix_dict: dict = {}
    history_list: list = []
    images: list = []
    config_data: dict = {}
    error_msg = None
    record_name_loaded = None

    # 1. 載入 .record 檔（優先順序：temp.record > {record_id}.record > 其他）
    try:
        record_files = list(record_path.glob("*.record"))
        target_file = None
        if record_files:
            file_map = {f.name: f for f in record_files}
            target_file = file_map.get("temp.record") or file_map.get(f"{record_id}.record") or record_files[0]

        if target_file:
            record_name_loaded = target_file.name
            record = Record(target_file.stem, rootdir=record_path, load=False)
            with open(target_file, "rb") as f:
                state = PathFixUnpickler(f).load()
            record.load_state_dict(state)

            df = record.dataframe.where(pd.notnull(record.dataframe), None)
            for key, values in df.to_dict(orient="list").items():
                first_val = next((v for v in values if v is not None), None)
                if first_val is None:
                    continue
                # list/ndarray 視為矩陣；scalar 視為單值序列
                target = matrix_dict if isinstance(first_val, list) else data_dict
                target[key] = values

            history_df = record.history.where(pd.notnull(record.history), None)
            history_list = history_df.to_dict(orient="records")
        else:
            error_msg = "No .record file found in this directory."

    except Exception as e:
        error_msg = f"Failed to load record data: {str(e)}"
        print(error_msg)

    # 2. 取得影像列表（位於 pic/ 子目錄）
    pic_dir = record_path / "pic"
    if pic_dir.exists():
        images = sorted((img.name for img in pic_dir.glob("*.png")), key=natural_sort_key)

    # 3. 載入 config.json
    config_path = record_path / "config.json"
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                config_data = json.load(f)
        except Exception as e:
            print(f"Failed to load config.json: {e}")

    return jsonify(
        record_id=record_id,
        record_name=record_name_loaded,
        data=data_dict,
        matrix_data=matrix_dict,
        history=history_list,
        images=images,
        config=config_data,
        error=error_msg,
    )


@app.route("/result/<path:filename>")
def serve_result(filename):
    """提供 result/ 下的靜態檔案（例如影像）；send_from_directory 會阻擋路徑遍歷。"""
    return send_from_directory(RESULT_DIR, filename)


if __name__ == "__main__":
    if not RESULT_DIR.exists():
        print(f"Warning: Result directory '{RESULT_DIR}' does not exist.")
        try:
            RESULT_DIR.mkdir(parents=True, exist_ok=True)
            print(f"Created '{RESULT_DIR}'.")
        except Exception as e:
            print(f"Error creating result directory: {e}")

    app.run(debug=DEV_MODE, host="0.0.0.0", port=5000)
