import json
import os
import pathlib
import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from flask import Flask, abort, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

# --- Mode and Path Configuration ---
# Check for '-dev' flag to enable development mode
DEV_MODE = "-dev" in sys.argv
if DEV_MODE:
    print("----- Running in Development Mode (debug=True) -----")
else:
    print("----- Running in Production Mode (debug=False) -----")

# Define Result Directory (always use production path)
RESULT_DIR = Path(r"T:\\碩二_吳維文's\\Patch Antenna\\Experiment\\result")
# We assume dataset directory is sibling to result or explicitly defined
DATASET_DIR = Path(r"T:\\碩二_吳維文's\\Patch Antenna\\Experiment\\dataset")

# Temp dir for downloads is always local
TEMP_DIR = Path(os.getcwd()).joinpath("temp_downloads").absolute()
TEMP_DIR.mkdir(parents=True, exist_ok=True)


# Set environment variable to avoid some issues with matplotlib GUI backends
import matplotlib

matplotlib.use("Agg")

# Import antenna utils
# Ensure PYTHONPATH is set in the shell script, or sys.path.append here if needed
sys.path.append(os.getcwd())

import shutil

import numpy as np
import torch

from antenna.utils import Record, config
from antenna.utils.data import Data, DataManager

# Force CPU to avoid CUDA errors in the viewer
config.device = "cpu"

app = Flask(__name__)


# Custom Unpickler to handle older Path objects
class PathFixUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Redirect antenna.utils.utils.Path to standard pathlib.Path
        # or WindowsPath if on Windows, but pathlib.Path usually auto-resolves.
        if name == "Path" and "antenna.utils" in module:
            return pathlib.Path
        return super().find_class(module, name)


# --- IP Tracking Logic ---
ACTIVE_USERS = {}  # {ip: last_seen_datetime}
TIMEOUT_MINUTES = 5


@app.before_request
def track_active_users():
    """Update the last seen timestamp for the requesting IP."""
    try:
        # If behind a proxy, you might need request.headers.get('X-Forwarded-For')
        if request.headers.getlist("X-Forwarded-For"):
            # 取第一個 IP，通常是真實客戶端 IP
            ip = request.headers.getlist("X-Forwarded-For")[0].split(",")[0].strip()
        else:
            # 如果沒有 Header，回退到 remote_addr (例如本地測試時)
            ip = request.remote_addr
        ACTIVE_USERS[ip] = datetime.now()
    except Exception:
        pass  # Don't break the app if tracking fails


def get_active_ip_list():
    """Return a list of IPs active within the last TIMEOUT_MINUTES."""
    cutoff = datetime.now() - timedelta(minutes=TIMEOUT_MINUTES)
    # Filter active users and return the list of IPs
    # Also clean up the dictionary
    expired_ips = [ip for ip, ts in ACTIVE_USERS.items() if ts < cutoff]
    for ip in expired_ips:
        del ACTIVE_USERS[ip]

    return sorted(list(ACTIVE_USERS.keys()))


# --- Caching ---
RECORD_CACHE = {}  # {record_id: {'key': (rec_mtime, pic_mtime), 'data': record_dict}}


def natural_sort_key(s):
    """Helper for natural sort (e.g. 1, 2, 10). s can be a Path object or string."""
    import re

    # If s is a Path object, use s.name, else use s string
    name = s.name if hasattr(s, "name") else s
    return [int(text) if text.isdigit() else text.lower() for text in re.split("([0-9]+)", name)]


@app.route("/")
def index():
    """Serves the skeleton of the index page. Data is fetched asynchronously."""
    return render_template("index.html")


@app.route("/api/main-page-data")
def get_main_page_data():
    """API endpoint to fetch all data needed for the main index page."""
    # 1. Records
    if not RESULT_DIR.exists():
        records = []
    else:
        records = []
        current_record_ids = set()
        for d in RESULT_DIR.iterdir():
            if d.is_dir():
                current_record_ids.add(d.name)
                rec_mtime = d.stat().st_mtime
                pic_dir = d / "pic"
                pic_mtime = 0
                if pic_dir.exists():
                    pic_mtime = pic_dir.stat().st_mtime

                cache_key = (rec_mtime, pic_mtime)

                cached = RECORD_CACHE.get(d.name)
                if cached and cached["key"] == cache_key:
                    records.append(cached["data"])
                    continue

                stat = d.stat()
                best_image = None
                if pic_dir.exists():
                    images = list(pic_dir.glob("*.png"))
                    if images:
                        images.sort(key=natural_sort_key)
                        best_candidates = [img for img in images if "best" in img.name.lower()]
                        if best_candidates:
                            best_image = best_candidates[-1].name
                        else:
                            best_image = images[-1].name

                record_data = {"id": d.name, "mtime": stat.st_mtime, "ctime": stat.st_ctime, "best_image": best_image}

                RECORD_CACHE[d.name] = {"key": cache_key, "data": record_data}
                records.append(record_data)

        records.sort(key=lambda x: x.get("mtime", 0), reverse=True)

    # 2. Datasets
    datasets = []
    if DATASET_DIR.exists():
        for f in DATASET_DIR.glob("*.dataset"):
            stat = f.stat()
            datasets.append({"id": f.stem, "name": f.name, "mtime": stat.st_mtime, "size": stat.st_size})
        datasets.sort(key=lambda x: x.get("mtime", 0), reverse=True)

    # 3. Get active users
    active_ips = get_active_ip_list()

    return jsonify({"records": records, "datasets": datasets, "active_ips": active_ips})


@app.route("/generator")
def generator():
    """Render the antenna pattern generator page."""
    return render_template("generator.html")


@app.route("/api/save_generated_pattern", methods=["POST"])
def save_generated_pattern():
    """Save the generated pattern using Data class and return file."""
    try:
        payload = request.json
        grid = payload.get("grid")
        raw_filename = payload.get("filename", "custom_pattern")

        if not grid:
            return "No grid data provided", 400

        # Secure filename
        filename = secure_filename(raw_filename)
        if not filename:
            filename = "custom_pattern"

        # Convert to Tensor (assuming 0/1 float32)
        # grid is list of lists
        arr = np.array(grid, dtype=np.float32)
        tensor_data = torch.from_numpy(arr)

        # Use utils/data.py Data class
        # Data(data=..., name=..., rootdir=..., suffix='data')
        # It saves to rootdir/name.suffix
        data_obj = Data(tensor_data, name=filename, rootdir=TEMP_DIR, suffix="data", load=False)
        data_obj.save()

        # File path
        file_path = data_obj.savepath

        # Debugging prints
        print(f"Saved generated pattern to: {file_path}")

        if not file_path.exists():
            print(f"Error: File not found at {file_path}")
            return "Failed to save file", 500

        # Send file
        # send_from_directory expects directory string and filename
        return send_from_directory(directory=str(TEMP_DIR), path=f"{filename}.data", as_attachment=True)

    except Exception as e:
        print(f"Exception in save_generated_pattern: {e}")
        # Return the actual error type for better debugging
        return f"{type(e).__name__}: {str(e)}", 500


@app.route("/dataset/<dataset_name>")
def view_dataset(dataset_name):
    """View details of a specific dataset."""
    try:
        # Load dataset using DataManager
        # Note: DataManager expects name without extension if it manages suffixes,
        # but here we might need to be careful. DataManager(name, rootdir) looks for name.dataset
        dm = DataManager(dataset_name, rootdir=DATASET_DIR, verbose=False)

        # Metadata
        total_items = len(dm)

        # Prepare samples for visualization (limit to first 50 to avoid browser crash)
        limit = 50
        samples = []

        # DataManager items are usually (input, output) tuples of tensors
        # We need to convert them to lists for JSON serialization
        for i in range(min(total_items, limit)):
            item = dm[i]
            # Handle list/tuple of tensors
            if isinstance(item, (list, tuple)):
                inp, outp = item[0], item[1]
            else:
                inp, outp = item, None

            # Convert to list helper
            def to_list(tensor_or_arr):
                if tensor_or_arr is None:
                    return None
                if hasattr(tensor_or_arr, "tolist"):
                    return tensor_or_arr.tolist()
                return tensor_or_arr

            samples.append({"id": i, "input": to_list(inp), "output": to_list(outp)})

        return render_template("dataset.html", dataset_name=dataset_name, total_items=total_items, samples=samples)
    except Exception as e:
        return f"Error loading dataset: {str(e)}", 500


@app.route("/record/<record_id>")
def view_record(record_id):
    """View details of a specific record."""
    # This route now only serves the skeleton page.
    # The actual data will be fetched by a JavaScript call to /api/record/<record_id>
    record_path = RESULT_DIR / record_id
    if not record_path.exists():
        abort(404, description="Record not found")

    return render_template("record.html", record_id=record_id)


@app.route("/api/record/<record_id>")
def get_record_data(record_id):
    """API endpoint to fetch data for a specific record."""
    record_path = RESULT_DIR / record_id
    if not record_path.exists():
        abort(404, description="Record not found")

    # Initialize variables to avoid UnboundLocalError
    data_dict = {}
    matrix_dict = {}
    history_list = []
    error_msg = None
    record_name_loaded = None
    images = []
    config_data = {}

    # 1. Load Record Data
    try:
        # Search for .record files
        record_files = list(record_path.glob("*.record"))
        target_file = None

        # Priority: temp.record > {record_id}.record > any .record
        if record_files:
            file_map = {f.name: f for f in record_files}
            if "temp.record" in file_map:
                target_file = file_map["temp.record"]
            elif f"{record_id}.record" in file_map:
                target_file = file_map[f"{record_id}.record"]
            else:
                target_file = record_files[0]

        if target_file:
            record_name_loaded = target_file.name
            # Initialize with the found name
            record = Record(target_file.stem, rootdir=record_path, load=False)

            # Manual load
            with open(target_file, "rb") as f:
                state = PathFixUnpickler(f).load()
            record.load_state_dict(state)

            # Convert DataFrame to simple dict for JSON serialization
            df = record.dataframe
            df = df.where(pd.notnull(df), None)
            all_data = df.to_dict(orient="list")

            # Separate scalars and matrices
            for key, values in all_data.items():
                # Find first non-none value to determine type
                first_val = next((v for v in values if v is not None), None)
                if first_val is None:
                    continue

                if isinstance(first_val, list):
                    # Assume list implies array/matrix data
                    matrix_dict[key] = values
                else:
                    # Assume scalar
                    data_dict[key] = values

            # History
            history_df = record.history
            history_df = history_df.where(pd.notnull(history_df), None)
            history_list = history_df.to_dict(orient="records")
        else:
            error_msg = "No .record file found in this directory."

    except Exception as e:
        error_msg = f"Failed to load record data: {str(e)}"
        print(error_msg)

    # 2. Get Images
    # Images are in 'pic' subdirectory
    pic_dir = record_path / "pic"
    if pic_dir.exists():
        # Get all png images
        for img_path in pic_dir.glob("*.png"):
            images.append(img_path.name)

        # Sort images naturally (e.g. 1.png, 2.png, 10.png)
        def natural_sort_key(s):
            import re

            return [int(text) if text.isdigit() else text.lower() for text in re.split("([0-9]+)", s)]

        images.sort(key=natural_sort_key)

    # 3. Load Config
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
    """Serve static files from the result directory (e.g., images)."""
    return send_from_directory(RESULT_DIR, filename)


if __name__ == "__main__":
    # Ensure result directory exists for the app to not crash immediately
    # This is more critical in production. In dev, dirs are created above.
    if not RESULT_DIR.exists():
        print(f"Warning: Result directory '{RESULT_DIR}' does not exist.")
        # In prod, we might not want to create it automatically, but for convenience:
        try:
            RESULT_DIR.mkdir(parents=True, exist_ok=True)
            print(f"Created '{RESULT_DIR}'.")
        except Exception as e:
            print(f"Error creating result directory: {e}")

    app.run(debug=DEV_MODE, host="0.0.0.0", port=5000)
