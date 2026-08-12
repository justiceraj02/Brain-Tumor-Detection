@echo off
title Brain Tumor Detection - BrainScan AI
echo ============================================
echo   Brain Tumor Detection - BrainScan AI
echo ============================================
echo.
echo Starting server...
echo Open http://127.0.0.1:62000 in your browser
echo Press Ctrl+C to stop the server
echo.
cd /d "%~dp0"
python app.py
pause
