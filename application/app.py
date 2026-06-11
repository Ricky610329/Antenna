import os
import json
import pickle
import pathlib
from pathlib import Path
from flask import Flask, render_template, send_from_directory, abort, jsonify, request
from werkzeug.utils import secure_filename
import pandas as pd
from datetime import datetime, timedelta
import sys

# --- Mode and Path Configuration ---
# Check for '-dev' flag to enable development mode
DEV_MODE = '-dev' in sys.argv
if DEV_MODE:
    print("----- Running in Development Mode (debug=True) -----")
else:
    print("----- Running in Production Mode (debug=False) -----")

# RESULT_DIR / DATASET_DIR 改為跟隨 antenna.utils 的工作區 (ROOTDIR/DATASET_PATH)，
# 不再硬編學長路徑；實際定義在下方 antenna 匯入之後。

# Temp dir for downloads is always local
TEMP_DIR = Path(os.getcwd()).joinpath('temp_downloads').absolute()
TEMP_DIR.mkdir(parents=True, exist_ok=True)


# Set environment variable to avoid some issues with matplotlib GUI backends
import matplotlib
matplotlib.use('Agg')

# Import antenna utils
# Ensure PYTHONPATH is set in the shell script, or sys.path.append here if needed
sys.path.append(os.getcwd())

from antenna.utils import config, ROOTDIR, DATASET_PATH
from antenna.legacy import DataManager, Data
import yaml
import subprocess
import torch
import numpy as np

# Force CPU to avoid CUDA errors in the viewer
config.device = 'cpu'

# 結果夾 / 資料集目錄預設跟隨工作區 (antenna.utils.ROOTDIR)；可用環境變數臨時覆寫，
# 方便還沒有自己的 run 時，先指向別的 result/ (例如學長的) demo 看效果。
RESULT_DIR = Path(os.environ.get('ANTENNA_RESULT_DIR', str(ROOTDIR / 'result')))
DATASET_DIR = Path(os.environ.get('ANTENNA_DATASET_DIR', str(DATASET_PATH)))

app = Flask(__name__)

# Custom Unpickler to handle older Path objects
class PathFixUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Redirect antenna.utils.utils.Path to standard pathlib.Path
        # or WindowsPath if on Windows, but pathlib.Path usually auto-resolves.
        if name == 'Path' and 'antenna.utils' in module:
            return pathlib.Path
        return super().find_class(module, name)

# --- IP Tracking Logic ---
ACTIVE_USERS = {} # {ip: last_seen_datetime}
TIMEOUT_MINUTES = 5

@app.before_request
def track_active_users():
    """Update the last seen timestamp for the requesting IP."""
    try:
        # If behind a proxy, you might need request.headers.get('X-Forwarded-For')
        if request.headers.getlist("X-Forwarded-For"):
            # 取第一個 IP，通常是真實客戶端 IP
            ip = request.headers.getlist("X-Forwarded-For")[0].split(',')[0].strip()
        else:
            # 如果沒有 Header，回退到 remote_addr (例如本地測試時)
            ip = request.remote_addr
        ACTIVE_USERS[ip] = datetime.now()
    except Exception:
        pass # Don't break the app if tracking fails

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
RECORD_CACHE = {} # {record_id: {'key': (rec_mtime, pic_mtime), 'data': record_dict}}

def natural_sort_key(s):
    """Helper for natural sort (e.g. 1, 2, 10). s can be a Path object or string."""
    import re
    # If s is a Path object, use s.name, else use s string
    name = s.name if hasattr(s, 'name') else s
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', name)]

@app.route('/')
def index():
    """Serves the skeleton of the index page. Data is fetched asynchronously."""
    return render_template('index.html')

def _read_run_summary(d):
    """讀單一 run 夾的總覽 (新格式)：status.json + metrics.csv + config.yaml + summary.png。"""
    info = {'id': d.name, 'mtime': d.stat().st_mtime}
    sj = d / 'status.json'
    if sj.exists():
        try:
            s = json.loads(sj.read_text(encoding='utf-8'))
            info.update(state=s.get('state'), machine=s.get('machine'),
                        updated_at=s.get('updated_at'), epoch=s.get('epoch'))
        except Exception:
            pass
    cy = d / 'config.yaml'
    if cy.exists():
        try:
            c = yaml.safe_load(cy.read_text(encoding='utf-8')) or {}
            info['name'] = c.get('name')
            info['port'] = c.get('port')
        except Exception:
            pass
    mc = d / 'metrics.csv'
    if mc.exists():
        try:
            df = pd.read_csv(mc)
            if len(df):
                # 兼容新舊欄名 (best_loss←min_loss)
                best_col = next((c for c in ('best_loss', 'min_loss') if c in df.columns), None)
                if best_col:
                    info['best_loss'] = round(float(df[best_col].min()), 4)
                if 'r_feed' in df.columns:
                    info['r_feed'] = round(float(df['r_feed'].iloc[-1]), 4)
                if 'epoch' in df.columns:
                    info['epoch'] = int(df['epoch'].iloc[-1])
                info['rows'] = int(len(df))
        except Exception:
            pass
    if (d / 'summary.png').exists():
        info['summary'] = 'summary.png'
    info['has_tb'] = (d / 'tb').exists()
    return info


@app.route('/api/main-page-data')
def get_main_page_data():
    """首頁資料：所有 run 的總覽 (新格式 status.json + metrics.csv) + 資料集清單。"""
    records = []
    if RESULT_DIR.exists():
        for d in RESULT_DIR.iterdir():
            if not d.is_dir():
                continue
            cache_key = d.stat().st_mtime           # live run 心跳會更新 mtime → 自動刷新
            cached = RECORD_CACHE.get(d.name)
            if cached and cached['key'] == cache_key:
                records.append(cached['data'])
                continue
            data = _read_run_summary(d)
            RECORD_CACHE[d.name] = {'key': cache_key, 'data': data}
            records.append(data)
        records.sort(key=lambda x: x.get('mtime', 0), reverse=True)

    datasets = []
    if DATASET_DIR.exists():
        for f in DATASET_DIR.glob('*.dataset'):      # 舊 legacy 單檔 pickle
            st = f.stat()
            datasets.append({'id': f.stem, 'name': f.name, 'kind': 'legacy',
                             'mtime': st.st_mtime, 'size': st.st_size})
        for d in DATASET_DIR.iterdir():              # 新 SampleStore 資料夾 (有 .pt 即是)
            if d.is_dir() and next(d.glob('*.pt'), None) is not None:
                datasets.append({'id': d.name, 'name': d.name, 'kind': 'SampleStore',
                                 'mtime': d.stat().st_mtime, 'size': 0})
        datasets.sort(key=lambda x: x.get('mtime', 0), reverse=True)

    return jsonify({'records': records, 'datasets': datasets,
                    'active_ips': get_active_ip_list()})

@app.route('/generator')
def generator():
    """Render the antenna pattern generator page."""
    return render_template('generator.html')

@app.route('/api/save_generated_pattern', methods=['POST'])
def save_generated_pattern():
    """Save the generated pattern using Data class and return file."""
    try:
        payload = request.json
        grid = payload.get('grid')
        raw_filename = payload.get('filename', 'custom_pattern')
        
        if not grid:
            return "No grid data provided", 400

        # Secure filename
        filename = secure_filename(raw_filename)
        if not filename:
            filename = 'custom_pattern'

        # Convert to Tensor (assuming 0/1 float32)
        # grid is list of lists
        arr = np.array(grid, dtype=np.float32)
        tensor_data = torch.from_numpy(arr)
        
        # Use utils/data.py Data class
        # Data(data=..., name=..., rootdir=..., suffix='data')
        # It saves to rootdir/name.suffix
        data_obj = Data(tensor_data, name=filename, rootdir=TEMP_DIR, suffix='data', load=False)
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

@app.route('/dataset/<dataset_name>')
def view_dataset(dataset_name):
    """瀏覽資料集前 50 筆：資料夾 → SampleStore 一筆一檔；否則 → 舊 DataManager (.dataset)。
    SampleStore 走「直接 glob 早停」只抓前 50 個 .pt，避免在 NAS 上枚舉數萬檔造成頁面卡死。"""
    PREVIEW = 50

    def to_list(t):
        if t is None:
            return None
        return t.tolist() if hasattr(t, 'tolist') else t

    try:
        path = DATASET_DIR / dataset_name
        samples = []
        if path.is_dir():                              # 新格式 SampleStore：只抓前 50 檔
            files = []
            for pf in path.glob('*.pt'):
                files.append(pf)
                if len(files) >= PREVIEW:
                    break
            for i, pf in enumerate(files):
                x, y = torch.load(pf, weights_only=True)
                samples.append({'id': i, 'input': to_list(x), 'output': to_list(y)})
            total_items = f"{len(samples)}+（一筆一檔，預覽前 {len(samples)} 筆）"
        else:                                          # 舊 legacy 單檔 .dataset
            dm = DataManager(dataset_name, rootdir=DATASET_DIR, verbose=False)
            total = len(dm)
            for i in range(min(total, PREVIEW)):
                item = dm[i]
                inp, outp = (item[0], item[1]) if isinstance(item, (list, tuple)) else (item, None)
                samples.append({'id': i, 'input': to_list(inp), 'output': to_list(outp)})
            total_items = total

        return render_template('dataset.html', dataset_name=dataset_name,
                               total_items=total_items, samples=samples)
    except Exception as e:
        return f"Error loading dataset: {str(e)}", 500

@app.route('/record/<record_id>')
def view_record(record_id):
    """View details of a specific record."""
    # This route now only serves the skeleton page.
    # The actual data will be fetched by a JavaScript call to /api/record/<record_id>
    record_path = RESULT_DIR / record_id
    if not record_path.exists():
        abort(404, description="Record not found")

    return render_template('record.html', record_id=record_id)

@app.route('/api/record/<record_id>')
def get_record_data(record_id):
    """API：讀新格式 RunState — metrics.csv (純量曲線) + patterns/ (pattern/response 演化)
    + config.yaml + summary.png。取代舊的 temp.record/Record。"""
    record_path = RESULT_DIR / record_id
    if not record_path.exists():
        abort(404, description="Record not found")

    data_dict, matrix_dict, history_list, config_data, images = {}, {}, [], {}, []
    error_msg = record_name_loaded = None

    # 1. metrics.csv → 純量序列 (data) + pattern/response 隨 epoch 演化 (matrix)
    mc = record_path / 'metrics.csv'
    if mc.exists():
        record_name_loaded = 'metrics.csv'
        try:
            df = pd.read_csv(mc)
            for col in df.columns:
                if col != 'pattern_hash':
                    data_dict[col] = df[col].tolist()

            if 'pattern_hash' in df.columns:
                pdir = record_path / 'patterns'
                hashes = df['pattern_hash'].astype(str).tolist()
                stride = max(1, len(hashes) // 40)   # 等距取樣 ~40 步，避免從 NAS 讀太多 .pt
                pats, resps = [], []
                for i in range(0, len(hashes), stride):
                    pf = pdir / f"{hashes[i]}.pt"
                    if not pf.exists():
                        continue
                    try:
                        pattern, response, _ = torch.load(pf, weights_only=True)
                        pats.append(pattern.tolist())
                        resps.append(response.tolist())
                    except Exception:
                        pass
                if pats:
                    matrix_dict['pattern'] = pats
                if resps:
                    matrix_dict['response'] = resps
        except Exception as e:
            error_msg = f"讀 metrics.csv 失敗: {e}"
    else:
        error_msg = ("找不到 metrics.csv —— 這個 run 不是新格式 (RunState)。"
                     "舊實驗請改用 TensorBoard 或學長封存查看。")

    # 2. summary.png (新格式放在 run 根目錄，非 pic/)
    if (record_path / 'summary.png').exists():
        images.append('summary.png')

    # 3. config.yaml → dict
    cy = record_path / 'config.yaml'
    if cy.exists():
        try:
            config_data = yaml.safe_load(cy.read_text(encoding='utf-8')) or {}
        except Exception as e:
            print(f"讀 config.yaml 失敗: {e}")

    return jsonify(
        record_id=record_id, record_name=record_name_loaded,
        data=data_dict, matrix_data=matrix_dict, history=history_list,
        images=images, config=config_data, error=error_msg,
    )

@app.route('/result/<path:filename>')
def serve_result(filename):
    """Serve static files from the result directory (e.g., images)."""
    return send_from_directory(RESULT_DIR, filename)

if __name__ == '__main__':
    if not RESULT_DIR.exists():
        try:
            RESULT_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Error creating result directory: {e}")

    # 自動起 TensorBoard 指向 result/ → 首頁「開啟 TensorBoard」按鈕 (另開 :6006)，
    # 一個 TB 實例即可疊圖比較所有 run。失敗不影響 app 本身。
    try:
        subprocess.Popen(
            [sys.executable, "-m", "tensorboard.main",
             "--logdir", str(RESULT_DIR), "--port", "6006", "--bind_all"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"TensorBoard 啟動中：logdir={RESULT_DIR} port=6006")
    except Exception as e:
        print(f"TensorBoard 啟動失敗 (可手動 tensorboard --logdir): {e}")

    app.run(debug=DEV_MODE, host='0.0.0.0', port=5000)
