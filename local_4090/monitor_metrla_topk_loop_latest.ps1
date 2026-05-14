param(
    [string]$RootDir = "D:\Citi\STDR\STDR_STGAT",
    [int]$PollSeconds = 30
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $RootDir

function Get-LatestLoopRoot {
    param([string]$BaseDir)

    Get-ChildItem (Join-Path $BaseDir "runs\external_speed\sweeps") -Directory |
        Where-Object { $_.Name -like "METR-LA_topk_16_19_20_forever_*" } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
}

while ($true) {
    Clear-Host

    $latest = Get-LatestLoopRoot -BaseDir $RootDir
    if ($null -eq $latest) {
        Write-Host "No METR-LA continuous TopK loop found yet." -ForegroundColor Yellow
        Start-Sleep -Seconds $PollSeconds
        continue
    }

    $statusPath = Join-Path $latest.FullName "status.txt"
    $leaderboardPath = Join-Path $latest.FullName "leaderboard.csv"

    Write-Host "==> METR-LA TopK loop monitor" -ForegroundColor Cyan
    Write-Host ("loop_root = {0}" -f $latest.FullName)
    Write-Host ("updated   = {0}" -f $latest.LastWriteTime)
    Write-Host ""

    if (Test-Path -LiteralPath $statusPath) {
        Write-Host "-- status --" -ForegroundColor Yellow
        Get-Content $statusPath
        Write-Host ""
    }

    if (Test-Path -LiteralPath $leaderboardPath) {
        Write-Host "-- current best by TopK --" -ForegroundColor Green
        Import-Csv $leaderboardPath |
            Select-Object topk, test_rmse, best_val_rmse, best_epoch, rmse15, rmse30, rmse60, cycle, run_index |
            Format-Table -AutoSize
    }
    else {
        Write-Host "leaderboard.csv not ready yet." -ForegroundColor DarkGray
    }

    Start-Sleep -Seconds $PollSeconds
}
