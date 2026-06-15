@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

echo ============================================
echo   Chinese MFA Pipeline - One-Click Setup
echo ============================================
echo.

:: Step 0: Check conda
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] conda not found.
    echo   Download Miniconda: https://docs.conda.io/en/latest/miniconda.html
    echo   Install with: "Add Miniconda to PATH" checked.
    pause
    exit /b 1
)
echo [OK] conda found.

:: Find conda base installation
for /f "tokens=*" %%i in ('conda info --base') do set "CONDA_BASE=%%i"
set "CONDA_ACTIVATE=%CONDA_BASE%\Scripts\activate.bat"

:: Step 1: Create conda environment from environment.yml
echo.
echo [1/3] Creating mfa_chinese environment from environment.yml...
echo   (This may take 10-20 minutes on first run)
call "%CONDA_ACTIVATE%"
conda env create -f environment.yml --force 2>&1 | findstr /v "^$"
if %errorlevel% neq 0 (
    echo [WARNING] Some packages may have failed to install.
    echo   Try: conda activate mfa_chinese ^&^& pip install -r requirements.txt
)

:: Step 2: Convert pinyin dict to IPA format
echo.
echo [2/3] Generating MFA IPA dictionary...
call "%CONDA_ACTIVATE%" mfa_chinese
python "%~dp0scripts\convert_dict_to_ipa.py"
if %errorlevel% neq 0 (
    echo [WARNING] IPA dict generation failed.
    echo   Run manually: python scripts\convert_dict_to_ipa.py
)

:: Step 3: Download MFA Chinese models
echo.
echo [3/3] Downloading MFA Chinese models...
set "MFA_ROOT_DIR=%~dp0models\mfa"
call "%CONDA_ACTIVATE%" mfa_chinese

mfa model download acoustic mandarin_mfa 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Acoustic model 'mandarin_mfa' download failed.
    echo   Run manually: mfa model download acoustic mandarin_mfa
)

mfa model download dictionary mandarin_china_mfa 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Dictionary download failed (non-critical).
)

echo.
echo ============================================
echo   Setup Complete!
echo.
echo   Environment:  mfa_chinese
echo   Configuration: config.yaml
echo   Dictionary:    dict/mfa_ipa.dict
echo   Models:        models/mfa/
echo.
echo   Quick start:
echo     1. Edit config.yaml - set 'workspace' path
echo     2. conda activate mfa_chinese
echo     3. python scripts\run_pipeline.py --data-dir E:\path\to\audio
echo.
echo   See README.md for detailed usage.
echo ============================================
pause
