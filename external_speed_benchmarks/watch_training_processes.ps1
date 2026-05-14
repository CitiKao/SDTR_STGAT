param(
    [int]$IntervalSeconds = 15
)

$ErrorActionPreference = "Continue"

while ($true) {
    Clear-Host
    Write-Host "ACTIVE TRAINING PROCESSES"
    Get-CimInstance Win32_Process |
        Where-Object { $_.Name -eq "python.exe" -and $_.CommandLine -like "*train_sensor_speed.py*" } |
        Select-Object ProcessId, CommandLine |
        Format-List
    Write-Host ""
    Write-Host "NVIDIA-SMI"
    nvidia-smi
    Start-Sleep -Seconds $IntervalSeconds
}
