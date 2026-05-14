param(
    [string]$PythonExe = "C:\Users\WMNL\anaconda3\envs\STDR\python.exe",
    [string]$RepoRoot = "D:\Citi\STDR\STDR_STGAT",
    [string]$CampaignRoot = "",
    [int]$IntervalSeconds = 15
)

$ErrorActionPreference = "Continue"

if ([string]::IsNullOrWhiteSpace($CampaignRoot)) {
    throw "CampaignRoot is required."
}

$campaignStatusPath = Join-Path $CampaignRoot "campaign_status.json"

while ($true) {
    Clear-Host
    Write-Host "PEMS-BAY AUTOPILOT STATUS"
    Write-Host "CampaignRoot: $CampaignRoot"
    Write-Host ""

    $campaignStatus = $null
    if (Test-Path $campaignStatusPath) {
        $campaignStatus = Get-Content -Path $campaignStatusPath -Raw | ConvertFrom-Json
        Write-Host ("active_stage: " + [string]$campaignStatus.active_stage)
        Write-Host ("completed_batches: " + [string]$campaignStatus.completed_batches)
        Write-Host ("current_best_rmse: " + [string]$campaignStatus.current_best_rmse)
        Write-Host ("current_best_run_id: " + [string]$campaignStatus.current_best_run_id)
        Write-Host ("prepared_dataset_dir: " + [string]$campaignStatus.prepared_dataset_dir)
    } else {
        Write-Host "waiting for campaign_status.json ..."
    }

    Write-Host ""
    $latestBatch = Get-ChildItem -Path $CampaignRoot -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like "batch_*" } |
        Sort-Object Name |
        Select-Object -Last 1

    if ($null -ne $latestBatch) {
        Write-Host ("latest_batch: " + $latestBatch.FullName)
        $queueStatusPath = Join-Path $latestBatch.FullName "queue_status.json"
        if (Test-Path $queueStatusPath) {
            $queueStatus = Get-Content -Path $queueStatusPath -Raw | ConvertFrom-Json
            Write-Host ("batch_active_run_id: " + [string]$queueStatus.active_run_id)
            Write-Host ("batch_runs_finished: " + [string]$queueStatus.runs_finished)
            Write-Host ("batch_runs_failed: " + [string]$queueStatus.runs_failed)
            Write-Host ("batch_best_rmse: " + [string]$queueStatus.session_best_test_rmse)
            Write-Host ("batch_best_run_id: " + [string]$queueStatus.session_best_run_id)

            $activeRunDir = [string]$queueStatus.active_run_dir
            if (-not [string]::IsNullOrWhiteSpace($activeRunDir) -and (Test-Path $activeRunDir)) {
                Write-Host ""
                Write-Host ("active_run_dir: " + $activeRunDir)
                & $PythonExe (Join-Path $RepoRoot "external_speed_benchmarks\monitor_training_progress.py") --run-dirs $activeRunDir --once
                $liveLog = Join-Path $activeRunDir "live.stdout.log"
                if (Test-Path $liveLog) {
                    Write-Host ""
                    Write-Host "live.stdout.log tail:"
                    Get-Content -Path $liveLog -Tail 5
                }
            }
        }
    }

    Start-Sleep -Seconds $IntervalSeconds
}
