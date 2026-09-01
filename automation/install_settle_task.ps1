# End-of-day settle: rebuild the dashboard from the close once the session is over.
#
#   powershell -ExecutionPolicy Bypass -File automation\install_settle_task.ps1
#
# Runs after the live window closes at 21:00. The day's bar is in the feed by then,
# so the levels roll to the next session and the site stops showing an intraday
# snapshot overnight.

param(
    [string]$At = "21:20",
    [string]$TaskName = "FractalRiskRangeSettle"
)

$root   = Split-Path -Parent $PSScriptRoot
$script = Join-Path $PSScriptRoot "settle_update.bat"
if (-not (Test-Path $script)) { throw "not found: $script" }

$action   = New-ScheduledTaskAction -Execute $script -WorkingDirectory $root
$trigger  = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At $At
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -RunOnlyIfNetworkAvailable `
              -WakeToRun -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
              -ExecutionTimeLimit (New-TimeSpan -Minutes 15) -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Settings $settings -Description "Rebuild the Risk Range dashboard from the close" -Force | Out-Null
Write-Host "Registered $TaskName for $At on weekdays."
