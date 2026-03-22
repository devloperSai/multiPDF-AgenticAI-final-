@echo off
title Celery Worker — Multi-PDF QA
cd /d "C:\Multi-PDF Question Answering System\backend"
call venv\Scripts\activate
echo.
echo ========================================
echo   Starting Celery Worker...
echo ========================================
echo.
celery -A celery_config.celery_app worker --loglevel=info --pool=solo
pause
