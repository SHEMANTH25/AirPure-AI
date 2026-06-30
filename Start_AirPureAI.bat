@echo off
title AirPure AI Dashboard

echo Starting AirPure AI...
echo.

call venv\Scripts\activate

start cmd /k "python app.py"

timeout /t 5 >nul

start http://127.0.0.1:5000

exit