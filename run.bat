@echo off
REM ─────────────────────────────────────────────────────────────
REM  ChickpeaAI — Windows Run Script
REM  Usage:
REM    run.bat           → full pipeline (generate + train + serve)
REM    run.bat generate  → only generate dataset
REM    run.bat train     → only train
REM    run.bat api       → only start API
REM ─────────────────────────────────────────────────────────────

SET BACKEND=%~dp0backend
SET DATASET=%~dp0dataset
SET MODELS=%~dp0models

IF "%1"=="generate" GOTO generate
IF "%1"=="train" GOTO train
IF "%1"=="api" GOTO api

:all
echo.
echo ╔══════════════════════════════════════════════╗
echo ║     ChickpeaAI — Chickpea Disease Detection  ║
echo ╚══════════════════════════════════════════════╝

:generate
echo.
echo [1/3] Generating synthetic dataset...
python "%BACKEND%\dataset_generator.py" "%DATASET%" 200
IF ERRORLEVEL 1 (echo ERROR: Dataset generation failed & pause & exit /b 1)
echo Dataset generated!

:train
echo.
echo [2/3] Training Hybrid CNN-RNN model...
python "%BACKEND%\train.py" ^
  --data_dir "%DATASET%" ^
  --model_dir "%MODELS%" ^
  --epochs 50 ^
  --batch_size 16 ^
  --lr 0.001 ^
  --patience 8
IF ERRORLEVEL 1 (echo ERROR: Training failed & pause & exit /b 1)
echo Training complete!

:api
echo.
echo [3/3] Starting Flask API...
echo.
echo   Open browser: http://localhost:5000
echo   Press Ctrl+C to stop
echo.
python "%BACKEND%\app.py"
pause
