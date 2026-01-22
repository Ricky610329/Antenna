@echo off
:: 設定編碼為 UTF-8
chcp 65001 >nul

set PYTHONPATH=%CD%

echo --- 正在檢查更新 (Git Pull) ---
git pull

echo.
echo --- 正在啟動 Conda 環境: patch ---
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

:: 這裡使用 choice 指令
choice /c 123 /n /m "Please enter your choice (1=Waitress, 2=Development, 3=Flask Prod): "

:: --- 修正重點：使用 GOTO 跳轉，不要用括號包住指令 ---
:: 注意：errorlevel 判斷必須從大到小 (3 -> 2 -> 1)

if errorlevel 3 goto ModeFlask
if errorlevel 2 goto ModeDev
if errorlevel 1 goto ModeWaitress

:: 如果意外穿透，跳回選單
goto menu

:: --- 獨立的執行區塊 ---

:ModeFlask
echo Starting in Production Mode (Flask Built-in Server)...
python application/app.py
goto end

:ModeDev
echo Starting in Development Mode...
python application/app.py -dev
goto end

:ModeWaitress
echo Starting in Production Mode (Waitress)...
python application/run_waitress.py
goto end

:end
echo.
echo [程式執行結束]
:: 加上 pause 讓視窗不會立刻消失，方便你看錯誤訊息
pause