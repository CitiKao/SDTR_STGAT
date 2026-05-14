param(
    [string]$PythonExe = "C:\Users\WMNL\anaconda3\envs\STDR\python.exe",
    [string]$RootDir = "D:\Citi\STDR\STDR_STGAT",
    [int]$Seed = 77,
    [int]$Epochs = 40,
    [int]$HistLen = 12,
    [int]$EarlyStopPatience = 0,
    [int]$MinEpochs = 0,
    [ValidateSet("adam", "adamw")]
    [string]$Optimizer = "adamw",
    [ValidateSet("mse", "huber", "charbonnier")]
    [string]$VLoss = "huber",
    [int]$WarmupEpochs = 5,
    [ValidateSet("val_loss", "val_rmse")]
    [string]$SchedulerMonitor = "val_rmse",
    [ValidateSet("auto", "bf16", "fp32")]
    [string]$Precision = "bf16",
    [int]$AdaptiveTopK = 16,
    [int]$HiddenDim = 32,
    [double]$Lr = 0.001,
    [double]$WeightDecay = 1e-5,
    [string]$RunDir = ""
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $RootDir

$resolvedRunDir = $RunDir
if ([string]::IsNullOrWhiteSpace($resolvedRunDir)) {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $lrTag = "{0:g}" -f $Lr
    $wdTag = "{0:g}" -f $WeightDecay
    $resolvedRunDir = Join-Path $RootDir ("runs\external_speed\METR-LA_seed{0}_{1}_{2}_hist{3}_wu{4}_{5}_{6}_topk{7}_hd{8}_lr{9}_wd{10}_{11}" -f $Seed, $Optimizer, $VLoss, $HistLen, $WarmupEpochs, $SchedulerMonitor, $Precision, $AdaptiveTopK, $HiddenDim, $lrTag, $wdTag, $timestamp)
}
New-Item -ItemType Directory -Path $resolvedRunDir -Force | Out-Null

$combinedLogPath = Join-Path $resolvedRunDir "live.stdout.log"

$args = @(
    "-u",
    "external_speed_benchmarks/train_sensor_speed.py",
    "--dataset-dir", "data\external_datasets\processed\METR-LA",
    "--dataset-name", "METR-LA",
    "--log-dir", $resolvedRunDir,
    "--fresh",
    "--epochs", "$Epochs",
    "--hist-len", "$HistLen",
    "--batch-size", "32",
    "--optimizer", $Optimizer,
    "--lr", "$Lr",
    "--weight-decay", "$WeightDecay",
    "--warmup-epochs", "$WarmupEpochs",
    "--grad-clip-norm", "5.0",
    "--v-loss", $VLoss,
    "--scheduler-monitor", $SchedulerMonitor,
    "--scheduler-factor", "0.5",
    "--scheduler-patience", "8",
    "--scheduler-cooldown", "2",
    "--scheduler-min-lr", "1e-5",
    "--split-policy", "benchmark_contiguous",
    "--split-alignment", "none",
    "--outlier-cleaning-mode", "train_quantile_clip",
    "--log-interval", "1",
    "--val-interval", "1",
    "--adaptive-topk", "$AdaptiveTopK",
    "--hidden-dim", "$HiddenDim",
    "--precision", $Precision,
    "--num-workers", "0",
    "--seed", "$Seed",
    "--early-stop-patience", "$EarlyStopPatience",
    "--min-epochs", "$MinEpochs"
)

Write-Host ""
Write-Host "==> Launching METR-LA best-seed rerun with $($Optimizer.ToUpper()) + $($VLoss.ToUpper()) + warmup=$WarmupEpochs + $SchedulerMonitor control"
Write-Host "    seed = $Seed"
Write-Host "    epochs = $Epochs"
Write-Host "    hist_len = $HistLen"
Write-Host "    early_stop_patience = $EarlyStopPatience"
Write-Host "    min_epochs = $MinEpochs"
Write-Host "    optimizer = $Optimizer"
Write-Host "    v_loss = $VLoss"
Write-Host "    warmup_epochs = $WarmupEpochs"
Write-Host "    scheduler_monitor = $SchedulerMonitor"
Write-Host "    precision = $Precision"
Write-Host "    adaptive_topk = $AdaptiveTopK"
Write-Host "    hidden_dim = $HiddenDim"
Write-Host "    lr = $Lr"
Write-Host "    weight_decay = $WeightDecay"
Write-Host "    run_dir = $resolvedRunDir"
Write-Host "    log = $combinedLogPath"
Write-Host "    python = $PythonExe"
Write-Host ""

& $PythonExe @args 2>&1 | Tee-Object -FilePath $combinedLogPath
