@echo off
REM Development startup script for Ghibli Image Generator (Windows)
REM This script starts both Django backend and React frontend

echo Starting Ghibli Image Generator Development Environment
echo ==========================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed. Please install Python 3.9+ first.
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo Node.js is not installed. Please install Node.js 18+ first.
    exit /b 1
)

REM Start Django backend
echo Starting Django backend...
start "Django Backend" cmd /k "python manage.py migrate && python manage.py runserver 8000"
echo Django backend started at http://localhost:8000
echo.

REM Wait a moment for Django to start
timeout /t 3 /nobreak >nul

REM Start React frontend
echo Starting React frontend...
cd frontend

REM Install dependencies if node_modules doesn't exist
if not exist "node_modules\" (
    echo Installing frontend dependencies...
    call npm install
)

start "React Frontend" cmd /k "npm run dev"
echo React frontend started at http://localhost:5173
echo.

cd ..

echo ==========================================================
echo Both servers are running!
echo.
echo    Frontend: http://localhost:5173
echo    Backend:  http://localhost:8000
echo    API:      http://localhost:8000/api/
echo.
echo Close the terminal windows to stop the servers
echo ==========================================================

pause