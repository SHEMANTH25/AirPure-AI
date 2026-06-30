@echo off

title AirPure AI

echo ======================================
echo      AirPure AI Starting...
echo ======================================

REM Check virtual environment
if not exist venv\Scripts\activate.bat (
    echo.
    echo ERROR:
    echo Virtual Environment not found.
    echo.
    pause
    exit
)

call venv\Scripts\activate.bat

echo.
echo Starting Flask Server...
echo.

start "" cmd /k python app.py

timeout /t 6 > nul

start http://127.0.0.1:5000

exit