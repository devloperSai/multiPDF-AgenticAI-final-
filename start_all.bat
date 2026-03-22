@echo off
title Multi-PDF QA — Starting All Services
echo.
echo ========================================
echo   Multi-PDF QA System
echo   Starting all services...
echo ========================================
echo.

:: Start Redis check first
echo [1/3] Checking Redis...
redis-cli ping >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: Redis not running. Start Redis manually if Celery fails.
) else (
    echo Redis is running.
)
echo.

:: Start Celery in new window
echo [2/3] Starting Celery Worker...
start "Celery Worker" cmd /k "cd /d "C:\Multi-PDF Question Answering System\backend" && call venv\Scripts\activate && celery -A celery_config.celery_app worker --loglevel=info --pool=solo"
timeout /t 3 /nobreak >nul

:: Start FastAPI in new window
echo [3/3] Starting FastAPI Backend...
start "FastAPI Backend" cmd /k "cd /d "C:\Multi-PDF Question Answering System\backend" && call venv\Scripts\activate && uvicorn main:app --reload"
timeout /t 3 /nobreak >nul

:: Start Frontend in new window
echo [4/3] Starting Frontend...
start "Frontend" cmd /k "cd /d "C:\Multi-PDF Question Answering System\pdf-genius-main" && npm run dev"

echo.
echo ========================================
echo   All services started!
echo.
echo   Frontend:  http://localhost:5173
echo   Backend:   http://localhost:8000
echo   API Docs:  http://localhost:8000/docs
echo ========================================
echo.
echo This window can be closed.
pause
