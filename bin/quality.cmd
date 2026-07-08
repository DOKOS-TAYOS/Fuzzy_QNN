@echo off
setlocal
set "PYTHON_EXE=%~dp0..\.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"
"%PYTHON_EXE%" -m ruff check . --fix
if errorlevel 1 exit /b %errorlevel%
"%PYTHON_EXE%" -m ruff format .
if errorlevel 1 exit /b %errorlevel%
"%PYTHON_EXE%" -m pytest
exit /b %errorlevel%
