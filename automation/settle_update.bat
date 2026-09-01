@echo off
REM End-of-day settle. Rebuilds the dashboard from the CLOSE once the session is
REM over, so the site does not sit on the last intraday snapshot all night.
REM
REM Without this the 21:00 live build stays up until noon the next day -- fifteen
REM hours of a page reading "live" and showing intraday crossings in the present
REM tense. By 21:15 the day's bar is in the feed, so this rolls the levels to the
REM next session, which is the correct view once trading is done.
REM
REM No --live and no --push: nothing is re-priced and nobody's phone rings after
REM the close. The book is synced, because by now the close IS the decision.

setlocal
cd /d "%~dp0.."
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd"') do set TODAY=%%i
if not exist "fractal\out\logs" mkdir "fractal\out\logs"
set LOG=fractal\out\logs\settle_%TODAY%.log

echo ================================================== >> "%LOG%" 2>&1
echo Settle started %TODAY% %TIME% >> "%LOG%" 2>&1

python -m fractal.app.etf_report --sync >> "%LOG%" 2>&1
if errorlevel 1 (
  echo SETTLE FAILED %TIME% >> "%LOG%" 2>&1
  endlocal
  exit /b 1
)

if not exist "docs" mkdir "docs"
copy /Y "fractal\out\etf_dashboard.html" "docs\index.html" >nul
git rev-parse --is-inside-work-tree >nul 2>&1
if not errorlevel 1 (
  git add docs fractal\data\portfolio.csv >nul 2>&1
  git diff --cached --quiet || git commit -q -m "settle: dashboard off the %TODAY% close" >> "%LOG%" 2>&1
  git push -q >> "%LOG%" 2>&1
)

echo Settle finished %TIME% >> "%LOG%" 2>&1
endlocal
