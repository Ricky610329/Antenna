import os
import json
import pickle
import pathlib
from pathlib import Path
from flask import Flask, render_template, send_from_directory, abort, jsonify, request
import pandas as pd
from datetime import datetime, timedelta

# Set environment variable to avoid some issues with matplotlib GUI backends
import matplotlib
matplotlib.use('Agg')

# Import antenna utils
# Ensure PYTHONPATH is set in the shell script, or sys.path.append here if needed
import sys
sys.path.append(os.getcwd())

from antenna.utils import Record, config

# Force CPU to avoid CUDA errors in the viewer
config.device = 'cpu'

app = Flask(__name__)

# Define Result Directory
RESULT_DIR = Path(r"T:\\碩二_吳維文's\\Patch Antenna\\Experiment\\result")

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

@app.route('/')
def index():
    """List all available records."""
    if not RESULT_DIR.exists():
        records = []
    else:
        # List directories in result/
        records = []
        for d in RESULT_DIR.iterdir():
            if d.is_dir():
                # Get basic stats
                stat = d.stat()
                records.append({
                    'id': d.name,
                    'mtime': stat.st_mtime,
                    'ctime': stat.st_ctime
                })
        
        # Sort by modification time, newest first
        records.sort(key=lambda x: x['mtime'], reverse=True)
    
    # Get active users
    active_ips = get_active_ip_list()
        
    return render_template('index.html', records=records, active_ips=active_ips)

@app.route('/record/<record_id>')
def view_record(record_id):
    """View details of a specific record."""
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
        record_files = list(record_path.glob('*.record'))
        target_file = None
        
        # Priority: temp.record > {record_id}.record > any .record
        if record_files:
            file_map = {f.name: f for f in record_files}
            if 'temp.record' in file_map:
                target_file = file_map['temp.record']
            elif f'{record_id}.record' in file_map:
                target_file = file_map[f'{record_id}.record']
            else:
                target_file = record_files[0]
        
        if target_file:
            record_name_loaded = target_file.name
            # Initialize with the found name
            record = Record(target_file.stem, rootdir=record_path, load=False)
            
            # Manual load
            with open(target_file, 'rb') as f:
                state = PathFixUnpickler(f).load()
            record.load_state_dict(state)
            
            # Convert DataFrame to simple dict for JSON serialization
            df = record.dataframe
            df = df.where(pd.notnull(df), None)
            all_data = df.to_dict(orient='list')
            
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
            history_list = history_df.to_dict(orient='records')
        else:
            error_msg = "No .record file found in this directory."
        
    except Exception as e:
        error_msg = f"Failed to load record data: {str(e)}"
        print(error_msg)

    # 2. Get Images
    # Images are in 'pic' subdirectory
    pic_dir = record_path / 'pic'
    if pic_dir.exists():
        # Get all png images
        for img_path in pic_dir.glob('*.png'):
            images.append(img_path.name)
        
        # Sort images naturally (e.g. 1.png, 2.png, 10.png)
        def natural_sort_key(s):
            import re
            return [int(text) if text.isdigit() else text.lower()
                    for text in re.split('([0-9]+)', s)]
        
        images.sort(key=natural_sort_key)

    # 3. Load Config
    config_path = record_path / 'config.json'
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
        except Exception as e:
            print(f"Failed to load config.json: {e}")

    return render_template(
        'record.html', 
        record_id=record_id, 
        record_name=record_name_loaded,
        data=data_dict, 
        matrix_data=matrix_dict,
        history=history_list, 
        images=images,
        config=config_data,
        error=error_msg
    )

@app.route('/result/<path:filename>')
def serve_result(filename):
    """Serve static files from the result directory (e.g., images)."""
    return send_from_directory(RESULT_DIR, filename)

if __name__ == '__main__':
    # Ensure result directory exists for the app to not crash immediately (optional)
    if not RESULT_DIR.exists():
        print(f"Warning: Result directory '{RESULT_DIR}' does not exist.")
        try:
            RESULT_DIR.mkdir()
            print(f"Created '{RESULT_DIR}'.")
        except:
            pass
            
    app.run(debug=True, host='0.0.0.0', port=5000)