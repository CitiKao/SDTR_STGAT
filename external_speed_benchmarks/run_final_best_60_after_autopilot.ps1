param(
    [string]$PythonExe = "C:\Users\WMNL\anaconda3\envs\STDR\python.exe",
    [string]$RepoRoot = "D:\Citi\STDR\STDR_STGAT",
    [string]$CampaignRoot = "",
    [int]$WaitForPid = 0,
    [int]$FinalBestRepeatEpochs = 60,
    [int]$FinalBestRepeats = 2,
    [int]$PollSeconds = 60
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($CampaignRoot)) {
    throw "CampaignRoot is required."
}

$logPath = Join-Path $CampaignRoot "final_best_60epoch_waiter.log"
$scriptPath = Join-Path $RepoRoot "external_speed_benchmarks\run_pems_bay_autopilot.ps1"

function Write-WaiterLog {
    param([string]$Message)
    Add-Content -Path $logPath -Value ("[" + (Get-Date -Format o) + "] " + $Message) -Encoding utf8
}

Write-WaiterLog ("waiter started | wait_for_pid=" + $WaitForPid + " | repeats=" + $FinalBestRepeats + " | epochs=" + $FinalBestRepeatEpochs)

if ($WaitForPid -gt 0) {
    while ($true) {
        $proc = Get-Process -Id $WaitForPid -ErrorAction SilentlyContinue
        if ($null -eq $proc) {
            break
        }
        Write-WaiterLog ("waiting for autopilot pid " + $WaitForPid + " to finish")
        Start-Sleep -Seconds $PollSeconds
    }
}

Write-WaiterLog "base autopilot finished; launching final-best 60epoch resume pass"

& powershell -ExecutionPolicy Bypass -File $scriptPath `
    -PythonExe $PythonExe `
    -RepoRoot $RepoRoot `
    -CampaignRoot $CampaignRoot `
    -ResumeExisting `
    -FinalBestRepeatEpochs $FinalBestRepeatEpochs `
    -FinalBestRepeats $FinalBestRepeats

$exitCode = $LASTEXITCODE
Write-WaiterLog ("final-best 60epoch resume pass exited with code " + $exitCode)
exit $exitCode
