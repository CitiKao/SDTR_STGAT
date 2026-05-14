param(
    [string]$PythonExe = "C:\Users\WMNL\anaconda3\envs\STDR\python.exe",
    [string]$RepoRoot = "D:\Citi\STDR\STDR_STGAT",
    [string]$RootLogDir = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RootLogDir)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $RootLogDir = Join-Path $RepoRoot ("runs\external_speed\ablation_matrix\METR-LA_weighted_missing_ablation_ep30_" + $stamp)
}

$null = New-Item -ItemType Directory -Path $RootLogDir -Force
$summaryPath = Join-Path $RootLogDir "summary.tsv"
"variant`tlog_dir`tbest_val_rmse`tbest_epoch`ttest_rmse`trmse_15m`trmse_30m`trmse_60m" | Set-Content -Path $summaryPath -Encoding utf8

$baseArgs = @(
    "external_speed_benchmarks\train_sensor_speed.py",
    "--dataset-dir", (Join-Path $RepoRoot "data\external_datasets\processed\METR-LA"),
    "--dataset-name", "METR-LA",
    "--hist-len", "12",
    "--pred-horizon", "12",
    "--report-horizons-minutes", "15,30,60",
    "--hidden-dim", "32",
    "--num-heads", "4",
    "--num-st-blocks", "2",
    "--num-gtcn-layers", "2",
    "--kernel-size", "3",
    "--adaptive-topk", "19",
    "--epochs", "30",
    "--batch-size", "32",
    "--lr", "1e-3",
    "--optimizer", "adam",
    "--weight-decay", "1e-5",
    "--warmup-epochs", "0",
    "--grad-clip-norm", "5.0",
    "--v-loss", "mse",
    "--seed", "76",
    "--device", "auto",
    "--precision", "bf16",
    "--val-interval", "1",
    "--log-interval", "1",
    "--split-policy", "benchmark_contiguous",
    "--split-alignment", "none",
    "--scheduler-monitor", "val_loss",
    "--scheduler-factor", "0.5",
    "--scheduler-patience", "8",
    "--scheduler-cooldown", "2",
    "--scheduler-min-lr", "1e-5",
    "--early-stop-patience", "0",
    "--min-epochs", "0",
    "--outlier-cleaning-mode", "train_quantile_clip",
    "--fresh"
)

$variants = @(
    @{
        Name = "A_weighted_only"
        ExtraArgs = @("--disable-missing-aware-history")
    },
    @{
        Name = "B_missingaware_only"
        ExtraArgs = @("--disable-weighted-fixed-graph")
    },
    @{
        Name = "C_both_on"
        ExtraArgs = @()
    }
)

$env:PYTHONUNBUFFERED = "1"
Set-Location $RepoRoot

foreach ($variant in $variants) {
    $runDir = Join-Path $RootLogDir $variant.Name
    $null = New-Item -ItemType Directory -Path $runDir -Force
    $stdoutPath = Join-Path $runDir "live.stdout.log"
    $activePath = Join-Path $RootLogDir "active_run.txt"
    @(
        "variant=$($variant.Name)",
        "run_dir=$runDir",
        "started_at=$(Get-Date -Format o)"
    ) | Set-Content -Path $activePath -Encoding utf8

    $args = @($baseArgs + @("--log-dir", $runDir) + $variant.ExtraArgs)
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Starting $($variant.Name)"
    Write-Host "Run dir: $runDir"
    Write-Host "============================================================"
    & $PythonExe @args 2>&1 | Tee-Object -FilePath $stdoutPath

    $metricsPath = Join-Path $runDir "predictor_test_metrics.json"
    $metaPath = Join-Path $runDir "stgat_meta.json"
    if ((Test-Path $metricsPath) -and (Test-Path $metaPath)) {
        $metrics = Get-Content -Path $metricsPath -Raw | ConvertFrom-Json
        $meta = Get-Content -Path $metaPath -Raw | ConvertFrom-Json
        $report = $metrics.raw_metrics_report.speed
        $line = @(
            $variant.Name,
            $runDir,
            [string]$metrics.selected_checkpoint_metric,
            [string]$meta.training_control.best_epoch,
            [string]$metrics.raw_metrics.speed.rmse,
            [string]$report.'15min'.rmse,
            [string]$report.'30min'.rmse,
            [string]$report.'60min'.rmse
        ) -join "`t"
        Add-Content -Path $summaryPath -Value $line -Encoding utf8
        Write-Host ""
        Write-Host "Summary for $($variant.Name):"
        Write-Host $line
    } else {
        Write-Warning "Missing metrics for $($variant.Name): $metricsPath / $metaPath"
    }
}

Write-Host ""
Write-Host "All ablations finished."
Write-Host "Summary: $summaryPath"
