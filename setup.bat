@echo off
REM Crear el entorno virtual
python -m venv .venv

REM Activar el entorno y actualizar pip
call .\.venv\Scripts\activate && python -m pip install --upgrade pip

REM Instalar las dependencias del proyecto
if exist requirements.txt (
    pip install -r requirements.txt
    echo.
    echo ¡Entorno virtual creado y dependencias instaladas con exito!
) else (
    echo.
    echo Entorno creado, pero no se encontro el archivo requirements.txt.
)

pause
