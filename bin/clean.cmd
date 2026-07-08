@echo off
setlocal
for /d /r %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
if exist ".pytest_cache" rd /s /q ".pytest_cache"
if exist ".ruff_cache" rd /s /q ".ruff_cache"
exit /b 0
