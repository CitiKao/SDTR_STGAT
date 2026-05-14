param(
    [string]$PythonExe = "C:\Users\WMNL\anaconda3\python.exe",
    [string]$RootDir = "D:\Citi\STDR\STDR_STGAT"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $RootDir

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runDir = Join-Path $RootDir ("runs\external_speed\METR-LA_resultpush_v1_" + $timestamp)
New-Item -ItemType Directory -Path $runDir -Force | Out-Null

$args = @(
    "external_speed_benchmarks/train_sensor_speed.py",
    "--dataset-dir", "data\external_datasets\processed\METR-LA",
    "--dataset-name", "METR-LA",
    "--log-dir", $runDir,
    "--fresh",
    "--epochs", "40",
    "--batch-size", "32",
    "--optimizer", "adam",
    "--lr", "0.001",
    "--weight-decay", "1e-5",
    "--warmup-epochs", "0",
    "--grad-clip-norm", "5.0",
    "--v-loss", "mse",
    "--scheduler-monitor", "val_loss",
    "--scheduler-patience", "5",
    "--scheduler-cooldown", "1",
    "--scheduler-min-lr", "1e-5",
    "--split-policy", "benchmark_contiguous",
    "--split-alignment", "none",
    "--outlier-cleaning-mode", "train_quantile_clip",
    "--log-interval", "1",
    "--val-interval", "1",
    "--adaptive-topk", "16",
    "--hidden-dim", "32",
    "--precision", "bf16"
)

$stdoutPath = Join-Path $runDir "launch.stdout.log"
$stderrPath = Join-Path $runDir "launch.stderr.log"
$commandText = "$PythonExe $($args -join ' ')"

Write-Host ""
Write-Host "==> Launching METR-LA result-push run"
Write-Host "    run_dir = $runDir"
Write-Host "    command = $commandText"

& $PythonExe @args 2>&1 | Tee-Object -FilePath $stdoutPath
