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
REM await_bars exits non-zero for TWO very different reasons, and this used to
REM treat them as one: the session's bars genuinely never published (benign, wait
REM for tomorrow), or the process CRASHED. On 16 Oct 2026 it died on `import
REM pandas` with a MemoryError, the log said "bars never published", the batch
REM exited 0, and Task Scheduler recorded a success -- while the book never
REM synced, the P&L track never sent and the dashboard sat on an intraday build.
REM A crash that reports itself as a quiet skip is the worst kind.
REM
REM Retry once in case the machine was briefly short of memory, then tell the two
REM apart by looking for a traceback, and exit NON-ZERO on a crash so it shows as
REM a failure in Task Scheduler instead of being swallowed.
REM
REM Written flat with GOTO on purpose. The obvious nested-parenthesis version is
REM wrong: inside a parenthesised block an `if errorlevel` after another command
REM does not see the value you expect, and a tested version of this silently
REM returned 0 down the crash path -- which would have defeated the whole point.
python -m fractal.app.await_bars --minutes 90 >> "%LOG%" 2>&1
if not errorlevel 1 goto bars_ok
echo await_bars failed, retrying once in 120s %TIME% >> "%LOG%" 2>&1
powershell -NoProfile -Command "Start-Sleep -Seconds 120" >nul 2>&1
python -m fractal.app.await_bars --minutes 30 >> "%LOG%" 2>&1
if not errorlevel 1 goto bars_ok
findstr /C:"Traceback" /C:"MemoryError" "%LOG%" >nul 2>&1
if errorlevel 1 goto bars_notready
echo SETTLE ABORTED - await_bars CRASHED, see the traceback above. The book was >> "%LOG%" 2>&1
echo NOT synced and the track was NOT sent; re-run the settle by hand. >> "%LOG%" 2>&1
endlocal
exit /b 1
:bars_notready
echo SETTLE SKIPPED - bars never published; intraday dashboard left up >> "%LOG%" 2>&1
endlocal
exit /b 0
:bars_ok

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
