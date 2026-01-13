@echo off
:: 設定編碼為 UTF-8，解決亂碼問題
chcp 65001 >nul

set PYTHONPATH=%CD%

echo --- 正在檢查更新 (Git Pull) ---
git pull

echo.
echo --- 正在啟動 Conda 環境: patch ---
:: 這裡使用 call 確保環境切換後繼續執行
call conda activate patch

echo.
echo Starting Antenna Visualizer...

python application/app.py

pause