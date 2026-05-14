param(
    [string]$SweepRoot,
    [int]$StartTopK = 10,
    [int]$EndTopK = 24,
    [int]$PollSeconds = 30
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($SweepRoot)) {
    throw "SweepRoot is required."
}

if (-not (Test-Path -LiteralPath $SweepRoot)) {
    throw "SweepRoot does not exist: $SweepRoot"
}

$leaderboardCsvPath = Join-Path $SweepRoot "leaderboard.csv"
$leaderboardMdPath = Join-Path $SweepRoot "leaderboard.md"
$statusPath = Join-Path $SweepRoot "status.txt"

function Get-BestEpoch {
    param([string]$RunDir)

    $runSummary = Join-Path $RunDir "run_summary.json"
    $stgatMeta = Join-Path $RunDir "stgat_meta.json"

    if (Test-Path -LiteralPath $runSummary) {
        try {
            $summary = Get-Content -Path $runSummary -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($null -ne $summary.best_epoch) {
                return [int]$summary.best_epoch
            }
        }
        catch {
        }
    }

    if (Test-Path -LiteralPath $stgatMeta) {
        try {
            $meta = Get-Content -Path $stgatMeta -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($null -ne $meta.best_epoch) {
                return [int]$meta.best_epoch
            }
        }
        catch {
        }
    }

    foreach ($path in @($runSummary, $stgatMeta)) {
        if (-not (Test-Path -LiteralPath $path)) {
            continue
        }
        try {
            $match = Select-String -Path $path -Pattern '"best_epoch"\s*:\s*(\d+)' -AllMatches
            if ($match -and $match.Matches.Count -gt 0) {
                return [int]$match.Matches[0].Groups[1].Value
            }
        }
        catch {
        }
    }

    return $null
}

function Get-RunMetrics {
    param([string]$RunDir, [int]$TopK)

    $metricsPath = Join-Path $RunDir "predictor_test_metrics.json"
    if (-not (Test-Path -LiteralPath $metricsPath)) {
        return $null
    }

    $metrics = Get-Content -Path $metricsPath -Raw -Encoding UTF8 | ConvertFrom-Json
    [pscustomobject]@{
        TopK = $TopK
        BestValRMSE = [double]$metrics.selected_checkpoint_metric
        BestEpoch = Get-BestEpoch -RunDir $RunDir
        TestRMSE = [double]$metrics.raw_metrics.speed.rmse
        RMSE15 = [double]$metrics.raw_metrics_report.speed.'15min'.rmse
        RMSE30 = [double]$metrics.raw_metrics_report.speed.'30min'.rmse
        RMSE60 = [double]$metrics.raw_metrics_report.speed.'60min'.rmse
        CompletedEpochs = [int]$metrics.training_control.completed_epochs
        RunDir = $RunDir
    }
}

function Get-CurrentProgress {
    param([string]$RunDir)

    $logPath = Join-Path $RunDir "live.stdout.log"
    if (-not (Test-Path -LiteralPath $logPath)) {
        return "waiting_for_log"
    }

    $tail = Get-Content -Path $logPath -Tail 30
    $interesting = $tail | Where-Object {
        $_ -match "\[Ep\s+\d+\]" -or $_ -match "Early stopping" -or $_ -match "TestV="
    }

    if ($interesting) {
        return ($interesting[-1]).Trim()
    }

    return "log_started"
}

while ($true) {
    $rows = @()
    $completedCount = 0
    $currentTopK = $null
    $currentProgress = $null

    for ($topk = $StartTopK; $topk -le $EndTopK; $topk++) {
        $runDir = Join-Path $SweepRoot ("topk_{0}" -f $topk)
        $metrics = Get-RunMetrics -RunDir $runDir -TopK $topk
        if ($null -ne $metrics) {
            $rows += $metrics
            $completedCount += 1
            continue
        }

        if ($null -eq $currentTopK -and (Test-Path -LiteralPath $runDir)) {
            $currentTopK = $topk
            $currentProgress = Get-CurrentProgress -RunDir $runDir
        }
    }

    $csvLines = @("topk,best_val_rmse,best_epoch,test_rmse,rmse15,rmse30,rmse60,completed_epochs,run_dir")
    foreach ($row in ($rows | Sort-Object TestRMSE, BestValRMSE)) {
        $csvLines += ("{0},{1:F6},{2},{3:F6},{4:F6},{5:F6},{6:F6},{7},{8}" -f `
            $row.TopK, $row.BestValRMSE, $row.BestEpoch, $row.TestRMSE, $row.RMSE15, $row.RMSE30, $row.RMSE60, $row.CompletedEpochs, $row.RunDir)
    }
    $csvLines | Set-Content -Path $leaderboardCsvPath

    $mdLines = @(
        "# METR-LA AdaptiveTopK Sweep",
        "",
        ("- Sweep root: {0}" -f $SweepRoot),
        ("- Completed: {0}/{1}" -f $completedCount, ($EndTopK - $StartTopK + 1))
    )
    if ($null -ne $currentTopK) {
        $mdLines += ("- Running now: TopK={0}" -f $currentTopK)
        $mdLines += ("- Latest progress: {0}" -f $currentProgress)
    }
    else {
        $mdLines += "- Running now: none"
    }
    $mdLines += ""
    $mdLines += "| Rank | TopK | Best Val RMSE | Best Epoch | Test RMSE | 15 min | 30 min | 60 min |"
    $mdLines += "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"

    $rank = 1
    foreach ($row in ($rows | Sort-Object TestRMSE, BestValRMSE)) {
        $mdLines += ("| {0} | {1} | {2:F4} | {3} | {4:F4} | {5:F4} | {6:F4} | {7:F4} |" -f `
            $rank, $row.TopK, $row.BestValRMSE, $row.BestEpoch, $row.TestRMSE, $row.RMSE15, $row.RMSE30, $row.RMSE60)
        $rank += 1
    }
    $mdLines | Set-Content -Path $leaderboardMdPath

    $statusLines = @(
        ("timestamp={0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss")),
        ("completed={0}" -f $completedCount),
        ("total={0}" -f ($EndTopK - $StartTopK + 1)),
        ("running_topk={0}" -f $(if ($null -eq $currentTopK) { "" } else { $currentTopK })),
        ("progress={0}" -f $(if ($null -eq $currentProgress) { "" } else { $currentProgress }))
    )
    $statusLines | Set-Content -Path $statusPath

    if ($completedCount -ge ($EndTopK - $StartTopK + 1)) {
        break
    }

    Start-Sleep -Seconds $PollSeconds
}
