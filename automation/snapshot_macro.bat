@echo off
REM Point-in-time snapshot of the macro model.
REM
REM The regime overlay cannot be backtested today because the workbook holds only
REM the CURRENT state plus a vintage or two -- running today's revised forecast
REM against 2023 prices is look-ahead, since the forecast has since been corrected
REM with knowledge of what happened. This builds the missing history forward: one
REM immutable copy per month, never revised.
REM
REM It lands in fractal\reference\, which is gitignored in full. The repo is
REM PUBLIC and this is proprietary work -- it must never be committed.

setlocal
cd /d "%~dp0.."
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd"') do set TODAY=%%i
set SRC=%USERPROFILE%\OneDrive\Desktop\Macro Model.xlsx
set DST=fractal\reference\macro_snapshots\macro_model_%TODAY%.xlsx

if not exist "%SRC%" (
  echo %TODAY% SOURCE MISSING: "%SRC%" >> fractal\out\logs\macro_snapshots.log
  endlocal
  exit /b 1
)
if exist "%DST%" (
  echo %TODAY% snapshot already exists, leaving it alone >> fractal\out\logs\macro_snapshots.log
  endlocal
  exit /b 0
)
copy /Y "%SRC%" "%DST%" >nul
echo %TODAY% snapshotted to %DST% >> fractal\out\logs\macro_snapshots.log
endlocal
