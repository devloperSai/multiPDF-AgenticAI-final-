@echo off
title FastAPI Backend — Multi-PDF QA
cd /d "C:\Multi-PDF Question Answering System\backend"
call venv\Scripts\activate
echo.
echo ========================================
echo   Starting FastAPI Backend...
echo   URL: http://localhost:8000
echo ========================================
echo.
uvicorn main:app --reload
pause
