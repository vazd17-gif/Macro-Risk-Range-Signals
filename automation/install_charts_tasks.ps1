# IBKR position charts, twice a day.
#
#   powershell -ExecutionPolicy Bypass -File automation\install_charts_tasks.ps1
#
# Open run is five minutes after the US open so quotes are live. The close run is
# deliberately AFTER the 21:20 settle, not at 21:00: the session bar has to publish
# before the levels roll, and charting mid-publication draws a half-formed day.
#
# Times are local, like every other job here, so they drift by an hour when the UK
# and US change clocks on different weekends. Same caveat as the live/settle jobs.

param(
    [string]$OpenAt  = "14:35",
    [string]$CloseAt = "21:45",
    [string]$OpenTask  = "FractalIbkrChartsOpen",
    [string]$CloseTask = "FractalIbkrChartsClose"
)

$root   = Split-Path -Parent $PSScriptRoot
$script = Join-Path $PSScriptRoot "ibkr_charts.bat"
if (-not (Test-Path $script)) { throw "not found: $script" }

$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -RunOnlyIfNetworkAvailable `
              -WakeToRun -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
              -ExecutionTimeLimit (New-TimeSpan -Minutes 20) -MultipleInstances IgnoreNew

foreach ($job in @(@{N=$OpenTask; T=$OpenAt; L="market open"},
                   @{N=$CloseTask; T=$CloseAt; L="market close"})) {
    $action  = New-ScheduledTaskAction -Execute $script -Argument ('"' + $job.L + '"') `
                 -WorkingDirectory $root
    $trigger = New-ScheduledTaskTrigger -Weekly `
                 -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At $job.T
    Register-ScheduledTask -TaskName $job.N -Action $action -Trigger $trigger `
        -Settings $settings -Description ("IBKR TRADE/TREND/RANGE charts - " + $job.L) -Force | Out-Null
    Write-Host ("Registered " + $job.N + " for " + $job.T + " on weekdays.")
}
