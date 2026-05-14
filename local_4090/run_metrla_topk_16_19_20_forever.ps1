param(
    [string]$PythonExe = "C:\Users\WMNL\anaconda3\envs\STDR\python.exe",
    [string]$RootDir = "D:\Citi\STDR\STDR_STGAT",
    [ValidateSet("auto", "bf16", "fp32")]
    [string]$Precision = "bf16",
    [int[]]$TopKs = @(16, 19, 20),
    [int]$Epochs = 20,
    [int]$Seed = 77,
    [int]$HiddenDim = 32,
    [double]$Lr = 0.001,
    [double]$WeightDecay = 1e-5,
    [int]$PollSecondsBetweenRuns = 3
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $RootDir

function Get-BestEpoch {
    param([string]$RunDir)

    $runSummary = Join-Path $RunDir "run_summary.json"
    $stgatMeta = Join-Path $RunDir "stgat_meta.json"

    foreach ($path in @($runSummary, $stgatMeta)) {
        if (-not (Test-Path -LiteralPath $path)) {
            continue
        }

        try {
            $obj = Get-Content -Path $path -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($null -ne $obj.best_epoch) {
                return [int]$obj.best_epoch
            }
        }
        catch {
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
    param(
        [string]$RunDir,
        [int]$TopK,
        [int]$Cycle,
        [int]$RunIndex
    )

    $metricsPath = Join-Path $RunDir "predictor_test_metrics.json"
    if (-not (Test-Path -LiteralPath $metricsPath)) {
        throw "predictor_test_metrics.json not found: $metricsPath"
    }

    $metrics = Get-Content -Path $metricsPath -Raw -Encoding UTF8 | ConvertFrom-Json
    [pscustomobject]@{
        cycle = $Cycle
        run_index = $RunIndex
        topk = $TopK
        best_val_rmse = [double]$metrics.selected_checkpoint_metric
        best_epoch = Get-BestEpoch -RunDir $RunDir
        test_rmse = [double]$metrics.raw_metrics.speed.rmse
        rmse15 = [double]$metrics.raw_metrics_report.speed.'15min'.rmse
        rmse30 = [double]$metrics.raw_metrics_report.speed.'30min'.rmse
        rmse60 = [double]$metrics.raw_metrics_report.speed.'60min'.rmse
        completed_epochs = [int]$metrics.training_control.completed_epochs
        run_dir = $RunDir
    }
}

function Write-HistoryFiles {
    param(
        [string]$LoopRoot,
        [System.Collections.ArrayList]$History
    )

    $historyPath = Join-Path $LoopRoot "history.csv"
    $leaderboardCsvPath = Join-Path $LoopRoot "leaderboard.csv"
    $leaderboardMdPath = Join-Path $LoopRoot "leaderboard.md"

    $history | Export-Csv -Path $historyPath -NoTypeInformation -Encoding UTF8

    $bestByTopk = $History |
        Group-Object -Property topk |
        ForEach-Object {
            $_.Group | Sort-Object test_rmse, best_val_rmse | Select-Object -First 1
        } |
        Sort-Object test_rmse, best_val_rmse

    $bestByTopk | Export-Csv -Path $leaderboardCsvPath -NoTypeInformation -Encoding UTF8

    $mdLines = @(
        "# METR-LA Continuous TopK Loop",
        "",
        ("- Loop root: {0}" -f $LoopRoot),
        ("- Total finished runs: {0}" -f $History.Count),
        ("- Candidate TopK: {0}" -f (($TopKs | ForEach-Object { "$_" }) -join ", ")),
        "",
        "| Rank | TopK | Best Test RMSE | Best Val RMSE | Best Epoch | 15 min | 30 min | 60 min | Cycle | Run Index |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )

    $rank = 1
    foreach ($row in $bestByTopk) {
        $mdLines += ("| {0} | {1} | {2:F4} | {3:F4} | {4} | {5:F4} | {6:F4} | {7:F4} | {8} | {9} |" -f `
            $rank, $row.topk, $row.test_rmse, $row.best_val_rmse, $row.best_epoch, $row.rmse15, $row.rmse30, $row.rmse60, $row.cycle, $row.run_index)
        $rank += 1
    }

    $mdLines | Set-Content -Path $leaderboardMdPath -Encoding UTF8
}

function Write-StatusFile {
    param(
        [string]$LoopRoot,
        [string]$Phase,
        [int]$Cycle,
        [int]$TopK,
        [string]$RunDir,
        [System.Collections.ArrayList]$History
    )

    $statusPath = Join-Path $LoopRoot "status.txt"
    $lines = @(
        ("timestamp={0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss")),
        ("phase={0}" -f $Phase),
        ("cycle={0}" -f $Cycle),
        ("running_topk={0}" -f $TopK),
        ("run_dir={0}" -f $RunDir),
        ("finished_runs={0}" -f $History.Count)
    )

    if ($History.Count -gt 0) {
        $bestOverall = $History | Sort-Object test_rmse, best_val_rmse | Select-Object -First 1
        $lines += ("best_topk={0}" -f $bestOverall.topk)
        $lines += ("best_test_rmse={0:F6}" -f $bestOverall.test_rmse)
        $lines += ("best_val_rmse={0:F6}" -f $bestOverall.best_val_rmse)
        $lines += ("best_epoch={0}" -f $bestOverall.best_epoch)
        $lines += ("best_15={0:F6}" -f $bestOverall.rmse15)
        $lines += ("best_30={0:F6}" -f $bestOverall.rmse30)
        $lines += ("best_60={0:F6}" -f $bestOverall.rmse60)
        $lines += ("best_run_dir={0}" -f $bestOverall.run_dir)
    }

    $lines | Set-Content -Path $statusPath -Encoding UTF8
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$loopRoot = Join-Path $RootDir ("runs\external_speed\sweeps\METR-LA_topk_16_19_20_forever_{0}" -f $timestamp)
New-Item -ItemType Directory -Path $loopRoot -Force | Out-Null

$history = New-Object System.Collections.ArrayList
$cycle = 1
$runIndex = 0

Write-Host ""
Write-Host "==> Continuous METR-LA TopK loop started" -ForegroundColor Cyan
Write-Host "    loop_root = $loopRoot"
Write-Host "    topk_order = $($TopKs -join ' -> ')"
Write-Host "    epochs = $Epochs"
Write-Host "    precision = $Precision"
Write-Host "    seed = $Seed"
Write-Host "    hidden_dim = $HiddenDim"
Write-Host "    lr = $Lr"
Write-Host "    weight_decay = $WeightDecay"
Write-Host "    mode = infinite loop"
Write-Host ""

while ($true) {
    foreach ($topk in $TopKs) {
        $runIndex += 1
        $runDir = Join-Path $loopRoot ("cycle_{0:D4}\topk_{1}" -f $cycle, $topk)
        New-Item -ItemType Directory -Path $runDir -Force | Out-Null

        Write-StatusFile -LoopRoot $loopRoot -Phase "running" -Cycle $cycle -TopK $topk -RunDir $runDir -History $history

        Write-Host ""
        Write-Host ("===== Cycle {0} | Run {1} | TopK={2} =====" -f $cycle, $runIndex, $topk) -ForegroundColor Yellow
        Write-Host "run_dir = $runDir"
        if ($history.Count -gt 0) {
            $bestBefore = $history | Sort-Object test_rmse, best_val_rmse | Select-Object -First 1
            Write-Host ("current best before launch: TopK={0} | TestRMSE={1:F4} | ValRMSE={2:F4} | 15/30/60={3:F3}/{4:F3}/{5:F3}" -f `
                $bestBefore.topk, $bestBefore.test_rmse, $bestBefore.best_val_rmse, $bestBefore.rmse15, $bestBefore.rmse30, $bestBefore.rmse60) -ForegroundColor Green
        }
        else {
            Write-Host "current best before launch: none yet" -ForegroundColor DarkGray
        }
        Write-Host ""

        try {
            & (Join-Path $RootDir "local_4090\run_metrla_bestseed_valrmse.ps1") `
                -PythonExe $PythonExe `
                -RootDir $RootDir `
                -Seed $Seed `
                -Epochs $Epochs `
                -EarlyStopPatience 0 `
                -MinEpochs 0 `
                -Optimizer "adam" `
                -VLoss "mse" `
                -WarmupEpochs 0 `
                -SchedulerMonitor "val_loss" `
                -Precision $Precision `
                -AdaptiveTopK $topk `
                -HiddenDim $HiddenDim `
                -Lr $Lr `
                -WeightDecay $WeightDecay `
                -RunDir $runDir

            if ($LASTEXITCODE -ne $null -and $LASTEXITCODE -ne 0) {
                Write-Host ("TopK={0} exited with code {1}" -f $topk, $LASTEXITCODE) -ForegroundColor Red
                Write-StatusFile -LoopRoot $loopRoot -Phase "failed_exit_$LASTEXITCODE" -Cycle $cycle -TopK $topk -RunDir $runDir -History $history
                Start-Sleep -Seconds $PollSecondsBetweenRuns
                continue
            }

            $row = Get-RunMetrics -RunDir $runDir -TopK $topk -Cycle $cycle -RunIndex $runIndex
            [void]$history.Add($row)
            Write-HistoryFiles -LoopRoot $loopRoot -History $history

            $bestOverall = $history | Sort-Object test_rmse, best_val_rmse | Select-Object -First 1
            $bestForTopK = $history | Where-Object { $_.topk -eq $topk } | Sort-Object test_rmse, best_val_rmse | Select-Object -First 1

            Write-Host ""
            Write-Host ("completed TopK={0} | cycle={1} | test_rmse={2:F4} | val_rmse={3:F4} | best_epoch={4} | 15/30/60={5:F3}/{6:F3}/{7:F3}" -f `
                $topk, $cycle, $row.test_rmse, $row.best_val_rmse, $row.best_epoch, $row.rmse15, $row.rmse30, $row.rmse60) -ForegroundColor Cyan
            Write-Host ("best ever for TopK={0}: TestRMSE={1:F4} | ValRMSE={2:F4} | cycle={3} | run={4}" -f `
                $topk, $bestForTopK.test_rmse, $bestForTopK.best_val_rmse, $bestForTopK.cycle, $bestForTopK.run_index) -ForegroundColor Magenta
            Write-Host ("CURRENT OVERALL BEST: TopK={0} | TestRMSE={1:F4} | ValRMSE={2:F4} | 15/30/60={3:F3}/{4:F3}/{5:F3} | cycle={6} | run={7}" -f `
                $bestOverall.topk, $bestOverall.test_rmse, $bestOverall.best_val_rmse, $bestOverall.rmse15, $bestOverall.rmse30, $bestOverall.rmse60, $bestOverall.cycle, $bestOverall.run_index) -ForegroundColor Green
            Write-Host ("leaderboard = {0}" -f (Join-Path $loopRoot "leaderboard.md"))
            Write-Host ""

            Write-StatusFile -LoopRoot $loopRoot -Phase "completed" -Cycle $cycle -TopK $topk -RunDir $runDir -History $history
        }
        catch {
            Write-Host ("TopK={0} failed: {1}" -f $topk, $_.Exception.Message) -ForegroundColor Red
            Write-StatusFile -LoopRoot $loopRoot -Phase "failed_exception" -Cycle $cycle -TopK $topk -RunDir $runDir -History $history
        }

        Start-Sleep -Seconds $PollSecondsBetweenRuns
    }

    $cycle += 1
}
