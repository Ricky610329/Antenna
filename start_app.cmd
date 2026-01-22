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

:menu
cls
echo Select an environment to start:
echo.
echo   1. Production Mode (by Waitress)
echo   2. Development Mode
echo   3. Production Mode (by Flask Built-in Server)
echo.

choice /c 123 /n /m "Please enter your choice (1=Waitress, 2=Development, 3=Flask Prod): "

if errorlevel 3 (
    echo Starting in Production Mode (Flask Built-in Server)...
    python application/app.py
    goto end
)

if errorlevel 2 (
    echo Starting in Development Mode...
    python application/app.py -dev
    goto end
)

if errorlevel 1 (
    echo Starting in Production Mode (Waitress)...
    python application/run_waitress.py
    goto end
)

:end

pause