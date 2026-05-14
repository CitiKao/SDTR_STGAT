param(
    [string]$CondaEnvName = "STDR",
    [string]$Datasets = "metr-la,pems-bay",
    [int]$BatchSize = 32,
    [int]$Epochs = 180,
    [int]$EarlyStopPatience = 0,
    [int]$MinEpochs = 10,
    [double]$LearningRate = 0.001,
    [double]$WeightDecay = 0.00001,
    [string]$SplitPolicy = "project_monthly",
    [int]$GraphK = 8,
    [int]$AdaptiveTopK = 16,
    [int]$ValInterval = 1,
    [int]$NumWorkers = 4,
    [string]$SplitAlignment = "none",
    [string]$SchedulerMonitor = "val_loss",
    [double]$SchedulerFactor = 0.5,
    [int]$SchedulerPatience = 8,
    [int]$SchedulerCooldown = 2,
    [double]$SchedulerMinLr = 1e-5,
    [string]$RepresentationDomain = "sensor_node",
    [string]$OutlierCleaningMode = "train_quantile_clip",
    [switch]$DisableAdaptive
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$datasetsList = $Datasets.Split(",") | ForEach-Object { $_.Trim().ToLower() } | Where-Object { $_ }
$runDirs = @()
$stopTag = if ($EarlyStopPatience -gt 0) { "es${EarlyStopPatience}min${MinEpochs}" } else { "esoff" }

Push-Location $repoRoot
try {
    $pythonPathBackup = $env:PYTHONPATH
    $env:PYTHONPATH = $null

    & conda run --no-capture-output -n $CondaEnvName python external_speed_benchmarks\prepare_dcrnn_sensor_datasets.py `
        --datasets $Datasets `
        --graph-k $GraphK `
        --representation-domain $RepresentationDomain

    foreach ($dataset in $datasetsList) {
        switch ($dataset) {
            "metr-la" { $displayName = "METR-LA" }
            "pems-bay" { $displayName = "PEMS-BAY" }
            default { throw "Unsupported dataset: $dataset" }
        }

        $datasetDirName = if ($RepresentationDomain -eq "pseudo_edge") {
            "${displayName}_pseudo_edge"
        }
        else {
            $displayName
        }

        $runDir = Join-Path $repoRoot ("runs\external_speed\" + $displayName + "_" + $RepresentationDomain + "_" + $SplitPolicy + "_bs${BatchSize}_ep${Epochs}_${stopTag}_$timestamp")
        $args = @(
            "external_speed_benchmarks\train_sensor_speed.py",
            "--dataset-dir", ("data\external_datasets\processed\" + $datasetDirName),
            "--dataset-name", $displayName,
            "--log-dir", $runDir,
            "--batch-size", $BatchSize,
            "--epochs", $Epochs,
            "--lr", $LearningRate,
            "--weight-decay", $WeightDecay,
            "--split-policy", $SplitPolicy,
            "--adaptive-topk", $AdaptiveTopK,
            "--val-interval", $ValInterval,
            "--num-workers", $NumWorkers,
            "--scheduler-monitor", $SchedulerMonitor,
            "--scheduler-factor", $SchedulerFactor,
            "--scheduler-patience", $SchedulerPatience,
            "--scheduler-cooldown", $SchedulerCooldown,
            "--scheduler-min-lr", $SchedulerMinLr,
            "--outlier-cleaning-mode", $OutlierCleaningMode
        )
        if ($SplitPolicy -eq "benchmark_contiguous") {
            $args += @("--split-alignment", $SplitAlignment)
        }
        if ($EarlyStopPatience -gt 0) {
            $args += @("--early-stop-patience", $EarlyStopPatience, "--min-epochs", $MinEpochs)
        }
        if ($DisableAdaptive.IsPresent) {
            $args += "--disable-adaptive"
        }

        $runDirs += $runDir
        & conda run --no-capture-output -n $CondaEnvName python @args
    }

    & conda run --no-capture-output -n $CondaEnvName python external_speed_benchmarks\generate_sensor_benchmark_report.py `
        --run-dirs ($runDirs -join ",")
}
finally {
    $env:PYTHONPATH = $pythonPathBackup
    Pop-Location
}
