@echo off
REM Daily refresh of the Macro Risk Range report.
REM Recomputes every level from the latest completed session, then optionally emails.
REM Run by Windows Task Scheduler; see install_task.ps1.

setlocal
cd /d "%~dp0.."

REM %DATE% formatting is locale-dependent, so ask PowerShell for the stamp instead.
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd"') do set TODAY=%%i
if not exist "fractal\out\logs" mkdir "fractal\out\logs"
set LOG=fractal\out\logs\daily_%TODAY%.log

echo ================================================== >> "%LOG%" 2>&1
echo Run started %TODAY% %TIME% >> "%LOG%" 2>&1

python run_daily.py >> "%LOG%" 2>&1
if errorlevel 1 (
  echo REFRESH FAILED with exit code %errorlevel% >> "%LOG%" 2>&1
  endlocal
  exit /b 1
)

REM Publish the freshly rendered dashboard to GitHub Pages.
if not exist "docs" mkdir "docs"
copy /Y "fractal\out\etf_dashboard.html" "docs\index.html" >nul
git rev-parse --is-inside-work-tree >nul 2>&1
if not errorlevel 1 (
  git add docs/index.html >nul 2>&1
  git diff --cached --quiet || git commit -q -m "daily: levels off the latest close" >> "%LOG%" 2>&1
  git push -q >> "%LOG%" 2>&1
)

REM Email only when FRACTAL_MAIL_TO is set. publish.py skips duplicate sends itself.
if defined FRACTAL_MAIL_TO (
  python -m fractal.app.publish --send >> "%LOG%" 2>&1
) else (
  echo FRACTAL_MAIL_TO not set - skipping email >> "%LOG%" 2>&1
)

echo Run finished %TIME% >> "%LOG%" 2>&1
endlocal
