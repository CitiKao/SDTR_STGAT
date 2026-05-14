param(
    [string]$RootDir = "D:\Citi\STDR\STDR_STGAT",
    [int]$PollSeconds = 20
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $RootDir

function Get-LatestMatrixRoot {
    param([string]$BaseDir)

    Get-ChildItem (Join-Path $BaseDir "runs\external_speed\sweeps") -Directory |
        Where-Object { $_.Name -like "METR-LA_topk_seed_matrix_ep*" } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
}

while ($true) {
    Clear-Host

    $latest = Get-LatestMatrixRoot -BaseDir $RootDir
    if ($null -eq $latest) {
        Write-Host "No METR-LA TopK x Seed matrix run found yet." -ForegroundColor Yellow
        Start-Sleep -Seconds $PollSeconds
        continue
    }

    $statusPath = Join-Path $latest.FullName "status.txt"
    $overallPath = Join-Path $latest.FullName "leaderboard_overall.csv"
    $topkPath = Join-Path $latest.FullName "leaderboard_by_topk.csv"

    Write-Host "==> METR-LA TopK x Seed matrix monitor" -ForegroundColor Cyan
    Write-Host ("matrix_root = {0}" -f $latest.FullName)
    Write-Host ("updated     = {0}" -f $latest.LastWriteTime)
    Write-Host ""

    if (Test-Path -LiteralPath $statusPath) {
        Write-Host "-- status --" -ForegroundColor Yellow
        Get-Content $statusPath
        Write-Host ""
    }

    if (Test-Path -LiteralPath $overallPath) {
        Write-Host "-- overall ranking --" -ForegroundColor Green
        Import-Csv $overallPath |
            Select-Object -First 12 topk, seed, test_rmse, best_val_rmse, best_epoch, rmse15, rmse30, rmse60 |
            Format-Table -AutoSize
        Write-Host ""
    }
    else {
        Write-Host "leaderboard_overall.csv not ready yet." -ForegroundColor DarkGray
        Write-Host ""
    }

    if (Test-Path -LiteralPath $topkPath) {
        Write-Host "-- best seed per topk --" -ForegroundColor Magenta
        Import-Csv $topkPath |
            Select-Object topk, seed, test_rmse, best_val_rmse, best_epoch, rmse15, rmse30, rmse60 |
            Format-Table -AutoSize
    }
    else {
        Write-Host "leaderboard_by_topk.csv not ready yet." -ForegroundColor DarkGray
    }

    Start-Sleep -Seconds $PollSeconds
}
