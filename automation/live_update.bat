@echo off
REM Intraday re-pricing of the dashboard.
REM Levels are NOT recomputed - they are fixed for the session by the previous
REM close. Only spot moves, so this re-prices against live quotes and rewrites the
REM dashboard. Cheap enough to run every few minutes.

setlocal
cd /d "%~dp0.."
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd"') do set TODAY=%%i
if not exist "fractal\out\logs" mkdir "fractal\out\logs"
set LOG=fractal\out\logs\live_%TODAY%.log

python -m fractal.app.etf_report --live --push --sync >> "%LOG%" 2>&1
if errorlevel 1 (
  echo LIVE REFRESH FAILED %TIME% >> "%LOG%" 2>&1
  endlocal
  exit /b 1
)

REM GitHub Pages serves docs\index.html, so the freshly rendered dashboard has to be
REM copied there. Regenerating fractal\out alone never reaches the site.
if not exist "docs" mkdir "docs"
copy /Y "fractal\out\etf_dashboard.html" "docs\index.html" >nul

REM Publish to GitHub Pages when this is a git repo with a remote.
git rev-parse --is-inside-work-tree >nul 2>&1
if not errorlevel 1 (
  git add docs >nul 2>&1
  git diff --cached --quiet || git commit -q -m "live: dashboard %TODAY% %TIME%" >> "%LOG%" 2>&1
  git push -q >> "%LOG%" 2>&1
)
endlocal
