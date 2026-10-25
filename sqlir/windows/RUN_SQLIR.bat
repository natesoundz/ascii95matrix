@echo off
setlocal
if exist "%~dp0..\..\sqlir\compile.py" (
  cd /d "%~dp0..\.."
) else if exist "%~dp0..\compile.py" (
  cd /d "%~dp0.."
) else (
  echo Could not locate sqlir\compile.py relative to this runner.
  pause
  exit /b 1
)
set "DB=%~1"
if "%DB%"=="" set "DB=C:\Users\idz9m\Downloads\webster_full_lexicon.sqlite"
if not exist "%DB%" (
  echo Database not found: %DB%
  echo Usage: RUN_SQLIR.bat "C:\path\to\webster_full_lexicon.sqlite"
  pause
  exit /b 1
)
set "PY=python"
where python >nul 2>nul || set "PY=py -3"
set "WEIGHTS=direct"
if /I "%SQLIR_WEIGHTS%"=="rubik" set "WEIGHTS=rubik"
echo Weight modes: direct is the default; Rubik uses SQLIR_WEIGHTS=rubik,
echo SQLIR_RUBIK_STORE and SQLIR_RUBIK_MODULE.
if /I "%WEIGHTS%"=="rubik" if "%SQLIR_RUBIK_STORE%"=="" (
  echo SQLIR_RUBIK_STORE and SQLIR_RUBIK_MODULE are required for Rubik mode.
  pause
  exit /b 1
)
if /I "%WEIGHTS%"=="rubik" if "%SQLIR_RUBIK_MODULE%"=="" (
  echo SQLIR_RUBIK_STORE and SQLIR_RUBIK_MODULE are required for Rubik mode.
  pause
  exit /b 1
)
if exist results rmdir /s /q results
mkdir results
set "LOG=results\run_log.txt"

echo === [1/5] Python + NumPy ===
%PY% --version > "%LOG%" 2>&1
%PY% -m pip install --quiet numpy >> "%LOG%" 2>&1
%PY% -c "import numpy, sqlite3, sys; print('python', sys.version); print('numpy', numpy.__version__); print('sqlite', sqlite3.sqlite_version)" >> "%LOG%" 2>&1
type "%LOG%"

echo === [2/5] Compile (reads the whole DB once; can take a long time) ===
echo COMPILE START %date% %time% >> "%LOG%"
powershell -NoProfile -Command "$ErrorActionPreference='Continue'; & %PY% -u -m sqlir.compile --db '%DB%' --workbook sqlir\conical437_real_workbook.json --out model 2>&1 | ForEach-Object { \"$_\" } | Tee-Object -FilePath '%LOG%' -Append"
echo COMPILE END %date% %time% >> "%LOG%"

echo === [3/5] Verify (re-runs live SQL and the original script; also slow) ===
echo VERIFY START %date% %time% >> "%LOG%"
powershell -NoProfile -Command "$ErrorActionPreference='Continue'; & %PY% -u -m sqlir.verify --model model --db '%DB%' --text Earth 2>&1 | ForEach-Object { \"$_\" } | Tee-Object -FilePath '%LOG%' -Append"
echo VERIFY END %date% %time% >> "%LOG%"

echo === [4/5] Sample calls (weights: %WEIGHTS%; direct is the default) ===
if /I "%WEIGHTS%"=="rubik" (
  %PY% -m sqlir.serve --model model --weights rubik --rubik-store "%SQLIR_RUBIK_STORE%" --rubik-module "%SQLIR_RUBIK_MODULE%" < sqlir\windows\requests.jsonl > results\chat_samples.jsonl 2>> "%LOG%"
) else (
  %PY% -m sqlir.serve --model model --weights direct < sqlir\windows\requests.jsonl > results\chat_samples.jsonl 2>> "%LOG%"
)

echo === [5/5] Collecting results ===
dir model > results\model_sizes.txt 2>&1
dir "%DB%" >> results\model_sizes.txt 2>&1
if exist model\manifest.json copy /y model\manifest.json results\ >nul
if exist model\verify_report.json copy /y model\verify_report.json results\ >nul
powershell -NoProfile -Command "Compress-Archive -Force -Path results\* -DestinationPath sqlir_results.zip"
echo.
echo Done. Send this file back to Devin:
echo   %CD%\sqlir_results.zip
pause
