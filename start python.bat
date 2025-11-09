@echo off
setlocal
set DIR=%~dp0
"%DIR%\.venv\Scripts\python.exe" "%DIR%\Python.py" %*
pause