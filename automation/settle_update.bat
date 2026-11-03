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
REM RETRY POLICY, rewritten 3 Nov 2026 after this failed a SECOND time. The old
REM version retried exactly once, 120 seconds later -- enough for a momentary blip
REM and nothing else. On 16 Oct and again on 30 Oct the machine was under a
REM pagefile squeeze that outlasted the pause: `import pandas` died with "DLL load
REM failed while importing timezones: The paging file is too small", the retry hit
REM the same wall two minutes later, and the settle aborted. Two sessions of marks
REM lost to a condition that clears on its own in a few minutes.
REM
REM So the two failure modes now get different treatment, decided per attempt
REM rather than once at the end:
REM
REM   CRASH   - died on import, in seconds, having consumed none of its waiting
REM             window. Back off 2, then 4, then 8 minutes and run again with the
REM             full window. Four attempts, 14 minutes of added wait in the worst
REM             case, which both incidents would have survived.
REM   LATE    - genuinely watched the feed for the whole window and no bar
REM             appeared. The data is late, not the machine, so more memory will
REM             not help. One short extra look, as before, then leave the intraday
REM             dashboard up and exit 0.
REM
REM The two paths keep SEPARATE counters on purpose. Sharing one would let three
REM crashes eat the allowance for the single late-feed retry, so a run that
REM recovered from a memory squeeze and then met a slow vendor would give up
REM without the second look it is entitled to.
REM
REM Each attempt writes to its own file so the crash test reads THIS attempt
REM rather than a traceback an earlier one left in the log; the file is appended
REM to the main log either way, so the record is still complete and in order.
REM
REM --session is pinned to %TODAY%. await_bars defaults to date.today() evaluated
REM when Python starts, so an attempt that crossed midnight would quietly begin
REM waiting for TOMORROW's bars -- which never arrive, turning a recoverable late
REM feed into a guaranteed timeout at the one moment the retries are working.
REM
REM Still written flat with GOTO on purpose. The obvious nested-parenthesis
REM version is wrong: inside a parenthesised block an `if errorlevel` after
REM another command does not see the value you expect, and a tested version of
REM this silently returned 0 down the crash path -- which would have defeated the
REM whole point. ERRORLEVEL is captured into RC on the line after the command for
REM the same reason: the `type` that follows would otherwise clobber it.
set CRASHES=0
set LATES=0
set MAX_CRASH=4
set BACKOFF=120
set WINDOW=90
set ATTLOG=fractal\out\logs\settle_attempt.tmp

:try_bars
python -m fractal.app.await_bars --session %TODAY% --minutes %WINDOW% > "%ATTLOG%" 2>&1
set RC=%ERRORLEVEL%
type "%ATTLOG%" >> "%LOG%" 2>&1
if "%RC%"=="0" goto bars_ok
findstr /C:"Traceback" /C:"MemoryError" "%ATTLOG%" >nul 2>&1
if errorlevel 1 goto bars_late
set /a CRASHES+=1
if %CRASHES% GEQ %MAX_CRASH% goto bars_crashed
echo await_bars CRASHED (%CRASHES% of %MAX_CRASH%); backing off %BACKOFF%s %TIME% >> "%LOG%" 2>&1
powershell -NoProfile -Command "Start-Sleep -Seconds %BACKOFF%" >nul 2>&1
set /a BACKOFF*=2
goto try_bars

:bars_late
if %LATES% GEQ 1 goto bars_notready
set /a LATES+=1
echo bars not published after %WINDOW% min; one more look in 120s %TIME% >> "%LOG%" 2>&1
powershell -NoProfile -Command "Start-Sleep -Seconds 120" >nul 2>&1
set WINDOW=30
goto try_bars

:bars_crashed
echo SETTLE ABORTED - await_bars CRASHED on all %MAX_CRASH% attempts, see the >> "%LOG%" 2>&1
echo tracebacks above. The book was NOT synced and the track was NOT sent; >> "%LOG%" 2>&1
echo re-run the settle by hand. >> "%LOG%" 2>&1
del "%ATTLOG%" >nul 2>&1
endlocal
exit /b 1

:bars_notready
echo SETTLE SKIPPED - bars never published; intraday dashboard left up >> "%LOG%" 2>&1
del "%ATTLOG%" >nul 2>&1
endlocal
exit /b 0

:bars_ok
del "%ATTLOG%" >nul 2>&1

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
