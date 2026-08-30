# Intraday dashboard refresh, every N minutes while the US market is open.
#
#   powershell -ExecutionPolicy Bypass -File automation\install_live_task.ps1
#
# Default window is 14:30-21:00 London (US cash session during BST). The levels do
# not change intraday, so this only re-prices spot - it is light work.

param(
    [int]$EveryMinutes = 10,
    [string]$Start = "14:30",
    [string]$End   = "21:00",
    [string]$TaskName = "FractalRiskRangeLive"
)

$root   = Split-Path -Parent $PSScriptRoot
$script = Join-Path $PSScriptRoot "live_update.bat"
if (-not (Test-Path $script)) { throw "not found: $script" }

$span     = New-TimeSpan -Hours ([int]($End.Split(':')[0]) - [int]($Start.Split(':')[0])) `
                         -Minutes ([int]($End.Split(':')[1]) - [int]($Start.Split(':')[1]))
$action   = New-ScheduledTaskAction -Execute $script -WorkingDirectory $root
$trigger  = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At $Start
$trigger.Repetition = (New-ScheduledTaskTrigger -Once -At $Start `
                         -RepetitionInterval (New-TimeSpan -Minutes $EveryMinutes) `
                         -RepetitionDuration $span).Repetition
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -RunOnlyIfNetworkAvailable `
              -ExecutionTimeLimit (New-TimeSpan -Minutes 9) -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Settings $settings -Description "Intraday Macro Risk Range re-pricing" -Force | Out-Null

Write-Host "Registered '$TaskName' - every $EveryMinutes min, $Start to $End, weekdays"
Write-Host "  run now: Start-ScheduledTask -TaskName $TaskName"
Write-Host "  remove:  Unregister-ScheduledTask -TaskName $TaskName -Confirm:`$false"
