@echo off
setlocal
cd /d "%~dp0"
set PIP_NO_CACHE_DIR=1

where py >nul 2>nul
if %errorlevel%==0 (
    set "PYTHON_CMD=py -3"
) else (
    set "PYTHON_CMD=python"
)

if not exist ".venv\Scripts\python.exe" (
    echo Creando entorno virtual...
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 goto :error
)

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
if errorlevel 1 goto :error
python -m pip install -r requirements.txt
if errorlevel 1 goto :error
python ejecutar_proyecto.py
if errorlevel 1 goto :error

echo.
echo Ejecucion completada. Revise outputs\diagnostico_estudiantes.csv.
pause
exit /b 0

:error
echo.
echo La ejecucion se detuvo por un error. Revise el mensaje anterior.
pause
exit /b 1
