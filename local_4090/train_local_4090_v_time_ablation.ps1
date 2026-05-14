param(
    [string]$CondaEnvName = "STDR",
    [string]$TimeFeatureModes = "baseline,day_of_month,week_of_month,day_of_month_and_week_of_month",
    [int]$BatchSize = 32,
    [int]$Epochs = 20,
    [double]$BaseLr = 0.001,
    [int]$BaseBatchSize = 32,
    [string]$Precision = "bf16",
    [string]$Device = "cuda",
    [int]$NumWorkers = 0,
    [int]$ValInterval = 1,
    [int]$LogInterval = 1,
    [int]$NumStBlocks = 2,
    [int]$AdaptiveTopK = 16,
    [bool]$SpeedUseAdaptive = $true,
    [int]$SpeedAdaptiveTopK = 16,
    [int]$PredHorizon = 4,
    [string]$ReportHorizonsMinutes = "15,30,60",
    [int]$MaxTimeSteps = 0,
    [string]$RunPrefix = "local4090_v_time_ablation_adaptive",
    [switch]$Compile
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$lr = $BaseLr * [Math]::Sqrt($BatchSize / [double]$BaseBatchSize)
$modes = $TimeFeatureModes.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ } | Select-Object -Unique
$requestedMinutes = $ReportHorizonsMinutes.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ }

Push-Location $repoRoot
try {
    $pythonPathBackup = $env:PYTHONPATH
    $pythonUnbufferedBackup = $env:PYTHONUNBUFFERED
    $env:PYTHONPATH = $null
    $env:PYTHONUNBUFFERED = "1"

    foreach ($mode in $modes) {
        $runName = "{0}_{1}_bs{2}_ep{3}_{4}" -f $RunPrefix, $mode, $BatchSize, $Epochs, $timestamp
        $runDir = Join-Path $repoRoot (Join-Path "runs\\local_4090_v" $runName)
        $trainArgs = @(
            "train_predictor.py",
            "--data-dir", "data",
            "--log-dir", $runDir,
            "--device", $Device,
            "--precision", $Precision,
            "--batch-size", $BatchSize,
            "--epochs", $Epochs,
            "--lr", $lr,
            "--num-workers", $NumWorkers,
            "--val-interval", $ValInterval,
            "--log-interval", $LogInterval,
            "--num-st-blocks", $NumStBlocks,
            "--adaptive-topk", $AdaptiveTopK,
            "--train-task", "v",
            "--monitor-task", "v",
            "--hist-len", 12,
            "--pred-horizon", $PredHorizon,
            "--report-horizons-minutes", $ReportHorizonsMinutes,
            "--time-feature-mode", $mode
        )

        if ($MaxTimeSteps -gt 0) {
            $trainArgs += @("--max-time-steps", $MaxTimeSteps)
        }

        if ($SpeedUseAdaptive) {
            $trainArgs += "--speed-use-adaptive"
            $trainArgs += @("--speed-adaptive-topk", $SpeedAdaptiveTopK)
        }

        if ($Compile.IsPresent) {
            $trainArgs += "--compile"
            $trainArgs += @("--compile-mode", "reduce-overhead")
        }

        Write-Host "Launching local 4090 NYC V ablation"
        Write-Host "  mode        = $mode"
        Write-Host "  run_dir     = $runDir"
        Write-Host "  batch_size  = $BatchSize"
        Write-Host "  epochs      = $Epochs"
        Write-Host "  lr          = $lr"
        Write-Host "  log_every   = $LogInterval"
        Write-Host "  topology    = $(if ($SpeedUseAdaptive) { "fixed+adaptive(k=$SpeedAdaptiveTopK)" } else { "fixed_only" })"
        Write-Host "  report      = $ReportHorizonsMinutes"

        & conda run --no-capture-output -n $CondaEnvName python @trainArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Training failed for mode '$mode' with exit code $LASTEXITCODE."
        }

        $metricsPath = Join-Path $runDir "predictor_test_metrics.json"
        if (-not (Test-Path $metricsPath)) {
            throw "Expected metrics file was not produced: $metricsPath"
        }

        $metrics = Get-Content $metricsPath -Raw | ConvertFrom-Json
        $speedReport = $metrics.raw_metrics_report.speed
        $best = $metrics.selected_checkpoint_metric
        Write-Host "  best_val_rmse = $best"

        foreach ($minutes in $requestedMinutes) {
            $reportKey = "{0}min" -f $minutes
            if (-not ($speedReport.PSObject.Properties.Name -contains $reportKey)) {
                throw "Missing report horizon '$reportKey' in $metricsPath"
            }

            $rmse = $speedReport.$reportKey.rmse
            Write-Host ("  test_rmse_{0}  = {1}" -f $minutes, $rmse)
        }
    }
}
finally {
    $env:PYTHONPATH = $pythonPathBackup
    $env:PYTHONUNBUFFERED = $pythonUnbufferedBackup
    Pop-Location
}
