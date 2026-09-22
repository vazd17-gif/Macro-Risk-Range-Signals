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

REM THE LIVE JOB NO LONGER TOUCHES THE BOOK -- it re-prices for display only.
REM
REM It used to SYNC in the 10:30-ET window, on the two-window cadence result.
REM That result was measured BEFORE the book could exit on state rather than on
REM a still-fresh break, and the two interact badly: a name opened at 10:30 on an
REM intraday reclaim that fails by the close is now exited at that same close, so
REM the pair manufactures same-day round trips. Re-run over 2 years of hourly
REM data (495 sessions, net of 10bp), sell-high deleted in every variant:
REM
REM   two-window + state exit        + 99.7%  Sharpe 2.17  -7.8%DD  91% dep  223 RTs
REM   close-confirmed + state exit   +103.1%  Sharpe 2.47  -7.1%DD  84% dep    0 RTs
REM   two-window, no state exit      + 96.2%  Sharpe 2.12  -8.4%DD
REM   close-confirmed, no state exit + 95.9%  Sharpe 2.31  -7.3%DD
REM
REM Close-confirmed wins on return, Sharpe AND drawdown while deploying less, and
REM drops round trips from 223 to zero. An entry now has to hold the close to be
REM taken at all. The book changes once a session, at the 21:20 settle.
REM (At 0bp two-window edges it +117.4 vs +116.5 -- it only loses once cost is
REM real, which is exactly what 223 round trips buys you.)

python -m fractal.app.etf_report --live --push >> "%LOG%" 2>&1
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
