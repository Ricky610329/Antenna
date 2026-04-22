"""以 waitress 啟動 production server。"""
import logging

from waitress import serve

from application.app import app

# 抑制 waitress 的 access log，只顯示錯誤
logging.getLogger("waitress").setLevel(logging.ERROR)

print("----- Starting production server with Waitress -----")
print("----- Access logs are disabled. Server is running at http://0.0.0.0:5000 -----")
serve(app, host="0.0.0.0", port=5000)
