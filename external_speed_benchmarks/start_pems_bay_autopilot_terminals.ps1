param(
    [string]$PythonExe = "C:\Users\WMNL\anaconda3\envs\STDR\python.exe",
    [string]$RepoRoot = "D:\Citi\STDR\STDR_STGAT",
    [string]$CampaignRoot = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($CampaignRoot)) {
    throw "CampaignRoot is required."
}

$statusProc = Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-ExecutionPolicy", "Bypass",
    "-File", (Join-Path $RepoRoot "external_speed_benchmarks\watch_sensor_autopilot_status_cn.ps1"),
    "-CampaignRoot", $CampaignRoot
) -PassThru

$gpuProc = Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-ExecutionPolicy", "Bypass",
    "-File", (Join-Path $RepoRoot "external_speed_benchmarks\watch_training_processes.ps1")
) -PassThru

$breakthroughProc = Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-ExecutionPolicy", "Bypass",
    "-File", (Join-Path $RepoRoot "external_speed_benchmarks\watch_autopilot_breakthroughs.ps1"),
    "-CampaignRoot", $CampaignRoot
) -PassThru

@{
    status_monitor_pid = $statusProc.Id
    process_monitor_pid = $gpuProc.Id
    breakthrough_monitor_pid = $breakthroughProc.Id
    campaign_root = $CampaignRoot
} | ConvertTo-Json -Depth 4
