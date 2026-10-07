@echo off
REM TRADE/TREND/RANGE charts for the IBKR book, mailed to the owner.
REM
REM Takes the window label as %1 ("market open" / "market close") so the two
REM scheduled triggers share one script and the subject line says which run it was.
REM
REM The name list comes from fractal\data\ibkr_names.txt, NOT from IBKR directly:
REM the brokerage connector is an MCP tool Claude uses in a session, and a scheduled
REM process has no access to it. Ask Claude to refresh the file when positions change.

setlocal
cd /d "%~dp0.."
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd"') do set TODAY=%%i
if not exist "fractal\out\logs" mkdir "fractal\out\logs"
set LOG=fractal\out\logs\charts_%TODAY%.log

set LABEL=%~1
if "%LABEL%"=="" set LABEL=update

echo ================================================== >> "%LOG%" 2>&1
echo Charts (%LABEL%) started %TODAY% %TIME% >> "%LOG%" 2>&1

python -m fractal.app.position_charts --label "%LABEL% - %TODAY%" >> "%LOG%" 2>&1
if errorlevel 1 (
  echo CHARTS FAILED %TIME% >> "%LOG%" 2>&1
  endlocal
  exit /b 1
)
echo Charts finished %TIME% >> "%LOG%" 2>&1
endlocal
