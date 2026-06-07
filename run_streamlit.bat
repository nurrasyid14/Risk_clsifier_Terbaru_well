@echo off
REM ====================================================================
REM Credit Risk Classifier - Streamlit Launcher
REM ====================================================================
REM Script ini menjalankan aplikasi dengan Python 3.11 (credit_env)
REM yang memiliki PyCaret terinstall untuk prediksi optimal
REM ====================================================================

echo.
echo ====================================================================
echo   Credit Risk Classifier - NexBank Decision Engine
echo ====================================================================
echo.
echo [INFO] Menggunakan Python 3.11 dari credit_env environment...
echo [INFO] Model: XGBoost (PyCaret) - AUC Score: 0.9785
echo.

REM Cek apakah Python 3.11 ada
if not exist "C:\Users\User\miniconda3\envs\credit_env\python.exe" (
    echo [ERROR] Python 3.11 (credit_env) tidak ditemukan!
    echo [ERROR] Silakan install atau aktifkan environment terlebih dahulu.
    echo.
    pause
    exit /b 1
)

echo [INFO] Memulai Streamlit server...
echo [INFO] Aplikasi akan terbuka di browser secara otomatis.
echo [INFO] Tekan Ctrl+C untuk menghentikan server.
echo.
echo ====================================================================
echo.

"C:\Users\User\miniconda3\envs\credit_env\python.exe" -m streamlit run app.py

echo.
echo [INFO] Streamlit server dihentikan.
pause
