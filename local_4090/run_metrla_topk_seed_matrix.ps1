param(
    [string]$PythonExe = "C:\Users\WMNL\anaconda3\envs\STDR\python.exe",
    [string]$RootDir = "D:\Citi\STDR\STDR_STGAT",
    [ValidateSet("auto", "bf16", "fp32")]
    [string]$Precision = "bf16",
    [int[]]$TopKs = @(16, 19, 20),
    [int[]]$Seeds = @(76, 77, 78),
    [int]$Epochs = 30,
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
        [int]$Seed,
        [int]$OrderIndex
    )

    $metricsPath = Join-Path $RunDir "predictor_test_metrics.json"
    if (-not (Test-Path -LiteralPath $metricsPath)) {
        throw "predictor_test_metrics.json not found: $metricsPath"
    }

    $metrics = Get-Content -Path $metricsPath -Raw -Encoding UTF8 | ConvertFrom-Json
    [pscustomobject]@{
        order_index = $OrderIndex
        topk = $TopK
        seed = $Seed
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
        [string]$MatrixRoot,
        [System.Collections.ArrayList]$History
    )

    $historyPath = Join-Path $MatrixRoot "history.csv"
    $overallCsvPath = Join-Path $MatrixRoot "leaderboard_overall.csv"
    $topkCsvPath = Join-Path $MatrixRoot "leaderboard_by_topk.csv"
    $mdPath = Join-Path $MatrixRoot "leaderboard.md"

    $History | Export-Csv -Path $historyPath -NoTypeInformation -Encoding UTF8

    $overall = $History | Sort-Object test_rmse, best_val_rmse
    $byTopK = $History |
        Group-Object -Property topk |
        ForEach-Object {
            $_.Group | Sort-Object test_rmse, best_val_rmse | Select-Object -First 1
        } |
        Sort-Object test_rmse, best_val_rmse

    $overall | Export-Csv -Path $overallCsvPath -NoTypeInformation -Encoding UTF8
    $byTopK | Export-Csv -Path $topkCsvPath -NoTypeInformation -Encoding UTF8

    $mdLines = @(
        "# METR-LA TopK x Seed Matrix",
        "",
        ("- Matrix root: {0}" -f $MatrixRoot),
        ("- TopK candidates: {0}" -f (($TopKs | ForEach-Object { "$_" }) -join ", ")),
        ("- Seeds: {0}" -f (($Seeds | ForEach-Object { "$_" }) -join ", ")),
        ("- Finished runs: {0}/{1}" -f $History.Count, ($TopKs.Count * $Seeds.Count)),
        "",
        "## Overall ranking",
        "",
        "| Rank | TopK | Seed | Test RMSE | Best Val RMSE | Best Epoch | 15 min | 30 min | 60 min |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )

    $rank = 1
    foreach ($row in $overall) {
        $mdLines += ("| {0} | {1} | {2} | {3:F4} | {4:F4} | {5} | {6:F4} | {7:F4} | {8:F4} |" -f `
            $rank, $row.topk, $row.seed, $row.test_rmse, $row.best_val_rmse, $row.best_epoch, $row.rmse15, $row.rmse30, $row.rmse60)
        $rank += 1
    }

    $mdLines += ""
    $mdLines += "## Best seed per TopK"
    $mdLines += ""
    $mdLines += "| Rank | TopK | Seed | Test RMSE | Best Val RMSE | Best Epoch | 15 min | 30 min | 60 min |"
    $mdLines += "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"

    $rank = 1
    foreach ($row in $byTopK) {
        $mdLines += ("| {0} | {1} | {2} | {3:F4} | {4:F4} | {5} | {6:F4} | {7:F4} | {8:F4} |" -f `
            $rank, $row.topk, $row.seed, $row.test_rmse, $row.best_val_rmse, $row.best_epoch, $row.rmse15, $row.rmse30, $row.rmse60)
        $rank += 1
    }

    $mdLines | Set-Content -Path $mdPath -Encoding UTF8
}

function Write-StatusFile {
    param(
        [string]$MatrixRoot,
        [string]$Phase,
        [int]$TopK,
        [int]$Seed,
        [string]$RunDir,
        [System.Collections.ArrayList]$History
    )

    $statusPath = Join-Path $MatrixRoot "status.txt"
    $lines = @(
        ("timestamp={0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss")),
        ("phase={0}" -f $Phase),
        ("running_topk={0}" -f $TopK),
        ("running_seed={0}" -f $Seed),
        ("run_dir={0}" -f $RunDir),
        ("finished_runs={0}" -f $History.Count),
        ("total_runs={0}" -f ($TopKs.Count * $Seeds.Count))
    )

    if ($History.Count -gt 0) {
        $bestOverall = $History | Sort-Object test_rmse, best_val_rmse | Select-Object -First 1
        $lines += ("best_topk={0}" -f $bestOverall.topk)
        $lines += ("best_seed={0}" -f $bestOverall.seed)
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
$matrixRoot = Join-Path $RootDir ("runs\external_speed\sweeps\METR-LA_topk_seed_matrix_ep{0}_{1}" -f $Epochs, $timestamp)
New-Item -ItemType Directory -Path $matrixRoot -Force | Out-Null

$history = New-Object System.Collections.ArrayList
$orderIndex = 0
$totalRuns = $TopKs.Count * $Seeds.Count

Write-Host ""
Write-Host "==> METR-LA TopK x Seed matrix started" -ForegroundColor Cyan
Write-Host "    matrix_root = $matrixRoot"
Write-Host "    topks = $($TopKs -join ', ')"
Write-Host "    seeds = $($Seeds -join ', ')"
Write-Host "    epochs = $Epochs"
Write-Host "    precision = $Precision"
Write-Host "    hidden_dim = $HiddenDim"
Write-Host "    lr = $Lr"
Write-Host "    weight_decay = $WeightDecay"
Write-Host "    total_runs = $totalRuns"
Write-Host ""

foreach ($topk in $TopKs) {
    foreach ($seed in $Seeds) {
        $orderIndex += 1
        $runDir = Join-Path $matrixRoot ("topk_{0}\seed_{1}" -f $topk, $seed)
        New-Item -ItemType Directory -Path $runDir -Force | Out-Null

        Write-StatusFile -MatrixRoot $matrixRoot -Phase "running" -TopK $topk -Seed $seed -RunDir $runDir -History $history

        Write-Host ""
        Write-Host ("===== Run {0}/{1} | TopK={2} | Seed={3} =====" -f $orderIndex, $totalRuns, $topk, $seed) -ForegroundColor Yellow
        Write-Host "run_dir = $runDir"
        if ($history.Count -gt 0) {
            $bestBefore = $history | Sort-Object test_rmse, best_val_rmse | Select-Object -First 1
            Write-Host ("current best before launch: TopK={0} | Seed={1} | TestRMSE={2:F4} | ValRMSE={3:F4} | 15/30/60={4:F3}/{5:F3}/{6:F3}" -f `
                $bestBefore.topk, $bestBefore.seed, $bestBefore.test_rmse, $bestBefore.best_val_rmse, $bestBefore.rmse15, $bestBefore.rmse30, $bestBefore.rmse60) -ForegroundColor Green
        }
        else {
            Write-Host "current best before launch: none yet" -ForegroundColor DarkGray
        }
        Write-Host ""

        try {
            & (Join-Path $RootDir "local_4090\run_metrla_bestseed_valrmse.ps1") `
                -PythonExe $PythonExe `
                -RootDir $RootDir `
                -Seed $seed `
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
                Write-Host ("TopK={0} Seed={1} exited with code {2}" -f $topk, $seed, $LASTEXITCODE) -ForegroundColor Red
                Write-StatusFile -MatrixRoot $matrixRoot -Phase "failed_exit_$LASTEXITCODE" -TopK $topk -Seed $seed -RunDir $runDir -History $history
                Start-Sleep -Seconds $PollSecondsBetweenRuns
                continue
            }

            $row = Get-RunMetrics -RunDir $runDir -TopK $topk -Seed $seed -OrderIndex $orderIndex
            [void]$history.Add($row)
            Write-HistoryFiles -MatrixRoot $matrixRoot -History $history

            $bestOverall = $history | Sort-Object test_rmse, best_val_rmse | Select-Object -First 1
            $bestForTopK = $history | Where-Object { $_.topk -eq $topk } | Sort-Object test_rmse, best_val_rmse | Select-Object -First 1

            Write-Host ""
            Write-Host ("completed TopK={0} Seed={1} | test_rmse={2:F4} | val_rmse={3:F4} | best_epoch={4} | 15/30/60={5:F3}/{6:F3}/{7:F3}" -f `
                $topk, $seed, $row.test_rmse, $row.best_val_rmse, $row.best_epoch, $row.rmse15, $row.rmse30, $row.rmse60) -ForegroundColor Cyan
            Write-Host ("best so far for TopK={0}: Seed={1} | TestRMSE={2:F4} | ValRMSE={3:F4}" -f `
                $topk, $bestForTopK.seed, $bestForTopK.test_rmse, $bestForTopK.best_val_rmse) -ForegroundColor Magenta
            Write-Host ("CURRENT OVERALL BEST: TopK={0} | Seed={1} | TestRMSE={2:F4} | ValRMSE={3:F4} | 15/30/60={4:F3}/{5:F3}/{6:F3}" -f `
                $bestOverall.topk, $bestOverall.seed, $bestOverall.test_rmse, $bestOverall.best_val_rmse, $bestOverall.rmse15, $bestOverall.rmse30, $bestOverall.rmse60) -ForegroundColor Green
            Write-Host ("leaderboard = {0}" -f (Join-Path $matrixRoot "leaderboard.md"))
            Write-Host ""

            Write-StatusFile -MatrixRoot $matrixRoot -Phase "completed" -TopK $topk -Seed $seed -RunDir $runDir -History $history
        }
        catch {
            Write-Host ("TopK={0} Seed={1} failed: {2}" -f $topk, $seed, $_.Exception.Message) -ForegroundColor Red
            Write-StatusFile -MatrixRoot $matrixRoot -Phase "failed_exception" -TopK $topk -Seed $seed -RunDir $runDir -History $history
        }

        Start-Sleep -Seconds $PollSecondsBetweenRuns
    }
}

Write-Host ""
Write-Host "All matrix runs finished." -ForegroundColor Cyan
Write-Host ("matrix_root = {0}" -f $matrixRoot)
Write-Host ("leaderboard = {0}" -f (Join-Path $matrixRoot "leaderboard.md"))
