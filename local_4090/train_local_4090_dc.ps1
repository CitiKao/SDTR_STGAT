param(
    [string]$CondaEnvName = "STDR",
    [int]$BatchSize = 32,
    [int]$Epochs = 20,
    [double]$BaseLr = 0.001,
    [int]$BaseBatchSize = 32,
    [string]$Precision = "bf16",
    [string]$Device = "cuda",
    [int]$NumWorkers = 0,
    [int]$ValInterval = 5,
    [int]$NumStBlocks = 2,
    [int]$AdaptiveTopK = 16,
    [string]$TrainTask = "dc",
    [string]$MonitorTask = "dc",
    [int]$MaxTimeSteps = 0,
    [string]$RunPrefix = "local4090_dc",
    [switch]$Compile
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$lr = $BaseLr * [Math]::Sqrt($BatchSize / [double]$BaseBatchSize)
$runName = "{0}_bs{1}_ep{2}_{3}" -f $RunPrefix, $BatchSize, $Epochs, $timestamp
$runDir = Join-Path $repoRoot ("runs\local_4090_dc\" + $runName)

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
    "--num-st-blocks", $NumStBlocks,
    "--adaptive-topk", $AdaptiveTopK,
    "--train-task", $TrainTask,
    "--monitor-task", $MonitorTask
)

if ($MaxTimeSteps -gt 0) {
    $trainArgs += @("--max-time-steps", $MaxTimeSteps)
}

if ($Compile.IsPresent) {
    $trainArgs += "--compile"
    $trainArgs += @("--compile-mode", "reduce-overhead")
}

Push-Location $repoRoot
try {
    $pythonPathBackup = $env:PYTHONPATH
    $env:PYTHONPATH = $null
    Write-Host "Launching local 4090 training"
    Write-Host "  repo_root   = $repoRoot"
    Write-Host "  conda_env   = $CondaEnvName"
    Write-Host "  run_dir     = $runDir"
    Write-Host "  batch_size  = $BatchSize"
    Write-Host "  epochs      = $Epochs"
    Write-Host "  lr          = $lr"
    Write-Host "  device      = $Device"
    Write-Host "  precision   = $Precision"
    Write-Host "  num_workers = $NumWorkers"
    & conda run --no-capture-output -n $CondaEnvName python @trainArgs
}
finally {
    $env:PYTHONPATH = $pythonPathBackup
    Pop-Location
}
