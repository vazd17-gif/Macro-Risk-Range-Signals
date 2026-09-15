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

REM TWO DECISION WINDOWS. Backtests showed syncing the book every 10 min churns it
REM (names flicker across TRADE and round-trip). Acting at just two windows a day --
REM ~10:30 ET (one hour into the open) and the close -- beat both every-10-min and
REM close-only on return AND Sharpe at every cost level. So SYNC the book only in the
REM 10:30-ET window (~15:30 local, since the task starts 14:30 = US open in BST); the
REM close window is the 21:20 settle. Every other run re-prices for display only.
for /f %%h in ('powershell -NoProfile -Command "Get-Date -Format HH"') do set HH=%%h
for /f %%m in ('powershell -NoProfile -Command "Get-Date -Format mm"') do set MM=%%m
set SYNC=
if "%HH%"=="15" if %MM% GEQ 30 if %MM% LSS 40 set SYNC=--sync

python -m fractal.app.etf_report --live --push %SYNC% >> "%LOG%" 2>&1
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
