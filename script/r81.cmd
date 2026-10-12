@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0r81.ps1" %*
exit /b %errorlevel%
