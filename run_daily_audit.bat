@echo off
REM ------------------------------------------------------------------
REM Daily accuracy audit — the file Windows Task Scheduler runs.
REM
REM A .bat wrapper rather than pointing the scheduler straight at
REM python.exe, because the job must start in the project directory:
REM the SQLite path and the `app` package import are both relative to it.
REM ------------------------------------------------------------------

cd /d "%~dp0"
".venv\Scripts\python.exe" -m scripts.daily_audit --quiet
exit /b %ERRORLEVEL%
