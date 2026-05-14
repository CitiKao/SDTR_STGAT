param(
    [string]$CondaEnvName = "STDR",
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
    [int]$SpeedAdaptiveTopK = 16,
    [int]$PredHorizon = 4,
    [string]$ReportHorizonsMinutes = "15,30,60",
    [string]$TimeFeatureMode = "baseline",
    [string]$SplitPolicy = "project_monthly",
    [string]$SplitAlignment = "none",
    [int]$MaxTimeSteps = 0,
    [string]$RunPrefix = "local4090_v_adaptive_newsplit",
    [string]$ResumeFromRun = "",
    [int]$TargetEpochs = 0,
    [string]$InitCheckpoint = "",
    [switch]$Compile
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$lr = $BaseLr * [Math]::Sqrt($BatchSize / [double]$BaseBatchSize)
$requestedMinutes = $ReportHorizonsMinutes.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ }
$runName = "{0}_{1}_bs{2}_ep{3}_{4}" -f $RunPrefix, $TimeFeatureMode, $BatchSize, $Epochs, $timestamp
$runDir = Join-Path $repoRoot (Join-Path "runs\\local_4090_v" $runName)
$epochsToRun = $Epochs
$resumeLastEpoch = 0

if ($ResumeFromRun) {
    $resumeDirResolved = Resolve-Path -LiteralPath $ResumeFromRun -ErrorAction Stop
    $runDir = $resumeDirResolved.Path
    $historyPath = Join-Path $runDir "predictor_log.json"
    if (-not (Test-Path $historyPath)) {
        throw "Resume history not found: $historyPath"
    }
    $history = Get-Content $historyPath -Raw | ConvertFrom-Json
    if (-not $history -or $history.Count -eq 0) {
        throw "Resume history is empty: $historyPath"
    }
    $resumeLastEpoch = [int]$history[-1].epoch
    if ($TargetEpochs -gt 0) {
        $epochsToRun = $TargetEpochs - $resumeLastEpoch
    }
    if ($epochsToRun -le 0) {
        throw "No additional epochs to run. resume_last_epoch=$resumeLastEpoch target_epochs=$TargetEpochs"
    }
    if (-not $InitCheckpoint) {
        $finalCheckpoint = Join-Path $runDir "stgat_final.pt"
        $bestCheckpoint = Join-Path $runDir "stgat_best.pt"
        if (Test-Path $finalCheckpoint) {
            $InitCheckpoint = $finalCheckpoint
        }
        elseif (Test-Path $bestCheckpoint) {
            $InitCheckpoint = $bestCheckpoint
        }
        else {
            throw "Could not find stgat_final.pt or stgat_best.pt in $runDir"
        }
    }
}

Push-Location $repoRoot
try {
    $pythonPathBackup = $env:PYTHONPATH
    $pythonUnbufferedBackup = $env:PYTHONUNBUFFERED
    $env:PYTHONPATH = $null
    $env:PYTHONUNBUFFERED = "1"

    $trainArgs = @(
        "train_predictor.py",
        "--data-dir", "data",
        "--log-dir", $runDir,
        "--device", $Device,
        "--precision", $Precision,
        "--batch-size", $BatchSize,
        "--epochs", $epochsToRun,
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
        "--time-feature-mode", $TimeFeatureMode,
        "--split-policy", $SplitPolicy,
        "--split-alignment", $SplitAlignment,
        "--speed-use-adaptive",
        "--speed-adaptive-topk", $SpeedAdaptiveTopK
    )

    if ($ResumeFromRun) {
        $trainArgs += @("--resume-run-dir", $runDir)
        $trainArgs += @("--init-checkpoint", $InitCheckpoint)
    }

    if ($MaxTimeSteps -gt 0) {
        $trainArgs += @("--max-time-steps", $MaxTimeSteps)
    }

    if ($Compile.IsPresent) {
        $trainArgs += "--compile"
        $trainArgs += @("--compile-mode", "reduce-overhead")
    }

    Write-Host "Launching local 4090 NYC V adaptive new-split run"
    Write-Host "  run_dir     = $runDir"
    Write-Host "  batch_size  = $BatchSize"
    if ($ResumeFromRun) {
        Write-Host "  resume      = $ResumeFromRun"
        Write-Host "  init_ckpt   = $InitCheckpoint"
        Write-Host "  epochs      = $epochsToRun (resume $resumeLastEpoch -> $($resumeLastEpoch + $epochsToRun))"
    }
    else {
        Write-Host "  epochs      = $Epochs"
    }
    Write-Host "  lr          = $lr"
    Write-Host "  log_every   = $LogInterval"
    Write-Host "  topology    = fixed+adaptive(k=$SpeedAdaptiveTopK)"
    Write-Host "  time_mode   = $TimeFeatureMode"
    Write-Host "  split       = $SplitPolicy / $SplitAlignment"
    Write-Host "  report      = $ReportHorizonsMinutes"
    Write-Host "  note        = each epoch prints train loss plus validation RMSE in real time"

    & conda run --no-capture-output -n $CondaEnvName python @trainArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Training failed with exit code $LASTEXITCODE."
    }

    $metricsPath = Join-Path $runDir "predictor_test_metrics.json"
    if (-not (Test-Path $metricsPath)) {
        throw "Expected metrics file was not produced: $metricsPath"
    }

    $metrics = Get-Content $metricsPath -Raw | ConvertFrom-Json
    $speedReport = $metrics.raw_metrics_report.speed
    Write-Host "  best_val_rmse = $($metrics.selected_checkpoint_metric)"

    foreach ($minutes in $requestedMinutes) {
        $reportKey = "{0}min" -f $minutes
        if (-not ($speedReport.PSObject.Properties.Name -contains $reportKey)) {
            throw "Missing report horizon '$reportKey' in $metricsPath"
        }

        $rmse = $speedReport.$reportKey.rmse
        Write-Host ("  test_rmse_{0}  = {1}" -f $minutes, $rmse)
    }
}
finally {
    $env:PYTHONPATH = $pythonPathBackup
    $env:PYTHONUNBUFFERED = $pythonUnbufferedBackup
    Pop-Location
}
