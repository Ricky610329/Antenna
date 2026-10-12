@echo off
cd /d "%~dp0.."
python -m script.r81v2_worker %*
exit /b %errorlevel%
