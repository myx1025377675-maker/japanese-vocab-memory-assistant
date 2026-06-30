@echo off
cd /d "%~dp0"
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8502" ^| findstr "LISTENING"') do (
    taskkill /PID %%a /F >nul 2>nul
)
timeout /t 1 /nobreak >nul
python -m streamlit run app.py --server.port 8502 --server.headless false
pause
