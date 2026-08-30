# Registers the daily refresh with Windows Task Scheduler.
#
#   powershell -ExecutionPolicy Bypass -File automation\install_task.ps1
#
# Runs weekdays at 12:00 London. This machine is on London time, so the trigger is
# simply 12:00 local. At that hour the most recent completed session is the previous
# day's US close (16:00 ET = 21:00 London), so the newsletter is a recap of it and
# carries the levels that apply to the session opening at 14:30 London.
#
# It also runs if the machine was asleep at the scheduled time.

param(
    [string]$Time = "12:00",
    [string]$TaskName = "FractalRiskRangeDaily"
)

$root   = Split-Path -Parent $PSScriptRoot
$script = Join-Path $PSScriptRoot "daily_update.bat"

if (-not (Test-Path $script)) { throw "not found: $script" }

$action    = New-ScheduledTaskAction -Execute $script -WorkingDirectory $root
$trigger   = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At $Time
$settings  = New-ScheduledTaskSettingsSet -StartWhenAvailable -RunOnlyIfNetworkAvailable `
                -ExecutionTimeLimit (New-TimeSpan -Hours 1) -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Settings $settings -Description "Daily Macro Risk Range refresh" -Force | Out-Null

Write-Host "Registered '$TaskName' - weekdays at $Time"
Write-Host "  run now:   Start-ScheduledTask -TaskName $TaskName"
Write-Host "  check:     Get-ScheduledTaskInfo -TaskName $TaskName"
Write-Host "  remove:    Unregister-ScheduledTask -TaskName $TaskName -Confirm:`$false"
Write-Host "  logs:      fractal\out\logs\"
