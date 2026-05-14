param(
    [string]$PythonExe = "python",
    [string]$RootDir = "D:\Citi\STDR\STDR_STGAT"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $RootDir

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$baseRunDir = Join-Path $RootDir "runs\external_speed"
$datasets = @(
    @{
        Name = "METR-LA"
        DatasetDir = "data\external_datasets\processed\METR-LA"
        RunDir = Join-Path $baseRunDir ("METR-LA_tuned_benchmark_v1_" + $timestamp)
    },
    @{
        Name = "PEMS-BAY"
        DatasetDir = "data\external_datasets\processed\PEMS-BAY"
        RunDir = Join-Path $baseRunDir ("PEMS-BAY_tuned_benchmark_v1_" + $timestamp)
    }
)

foreach ($item in $datasets) {
    $runDir = $item.RunDir
    New-Item -ItemType Directory -Path $runDir -Force | Out-Null

    $args = @(
        "external_speed_benchmarks/train_sensor_speed.py",
        "--dataset-dir", $item.DatasetDir,
        "--dataset-name", $item.Name,
        "--log-dir", $runDir,
        "--epochs", "180",
        "--batch-size", "32",
        "--optimizer", "adamw",
        "--lr", "0.001",
        "--weight-decay", "5e-5",
        "--warmup-epochs", "5",
        "--grad-clip-norm", "5.0",
        "--v-loss", "huber",
        "--huber-delta", "1.0",
        "--scheduler-monitor", "val_rmse",
        "--scheduler-patience", "12",
        "--scheduler-cooldown", "2",
        "--scheduler-min-lr", "1e-6",
        "--split-policy", "benchmark_contiguous",
        "--split-alignment", "none",
        "--outlier-cleaning-mode", "none",
        "--log-interval", "1",
        "--val-interval", "1"
    )

    $stdoutPath = Join-Path $runDir "launch.stdout.log"
    $stderrPath = Join-Path $runDir "launch.stderr.log"
    $commandText = "$PythonExe $($args -join ' ')"

    Write-Host ""
    Write-Host "==> Launching $($item.Name)"
    Write-Host "    run_dir = $runDir"
    Write-Host "    command = $commandText"

    & $PythonExe @args 2>&1 | Tee-Object -FilePath $stdoutPath
}
