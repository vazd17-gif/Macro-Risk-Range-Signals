@echo off
REM End-of-day settle. Rebuilds the dashboard from the CLOSE once the session is
REM over, so the site does not sit on the last intraday snapshot all night.
REM
REM Without this the 21:00 live build stays up until noon the next day -- fifteen
REM hours of a page reading "live" and showing intraday crossings in the present
REM tense. This rolls the levels to the
REM next session, which is the correct view once trading is done. It waits for
REM the day's bar rather than assuming it has landed -- see await_bars.
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

REM Wait for the day's bars before settling anything. On 2026-09-03 this job ran
REM 20 minutes after the close, the bars had not published, and it settled off the
REM PREVIOUS close while the P&L track emailed "0 positions, +0.00%" as a real
REM session. Waiting on the data beats guessing a later clock time.
python -m fractal.app.await_bars --minutes 90 >> "%LOG%" 2>&1
if errorlevel 1 (
  echo SETTLE SKIPPED - bars never published; intraday dashboard left up >> "%LOG%" 2>&1
  endlocal
  exit /b 0
)

python -m fractal.app.etf_report --settle --sync >> "%LOG%" 2>&1
if errorlevel 1 (
  echo SETTLE FAILED %TIME% >> "%LOG%" 2>&1
  endlocal
  exit /b 1
)

REM P&L track, once the book has been squared against the close. Goes to the owner
REM ONLY -- this is the private performance record, not the newsletter, and it must
REM never reach the distribution list. The address is hard-coded here rather than
REM read from FRACTAL_MAIL_TO precisely so that adding a newsletter recipient can
REM never quietly add them to this.
python -m fractal.app.track --send-to vazd17@gmail.com >> "%LOG%" 2>&1
if errorlevel 1 echo TRACK FAILED %TIME% >> "%LOG%" 2>&1

if not exist "docs" mkdir "docs"
copy /Y "fractal\out\etf_dashboard.html" "docs\index.html" >nul
git rev-parse --is-inside-work-tree >nul 2>&1
if not errorlevel 1 (
  git add docs fractal\data\portfolio.csv fractal\data\pnl_track.csv >nul 2>&1
  git diff --cached --quiet || git commit -q -m "settle: dashboard off the %TODAY% close" >> "%LOG%" 2>&1
  git push -q >> "%LOG%" 2>&1
)

echo Settle finished %TIME% >> "%LOG%" 2>&1
endlocal
