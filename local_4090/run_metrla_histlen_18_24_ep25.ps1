param(
    [string]$PythonExe = "C:\Users\WMNL\anaconda3\envs\STDR\python.exe",
    [string]$RootDir = "D:\Citi\STDR\STDR_STGAT",
    [ValidateSet("auto", "bf16", "fp32")]
    [string]$Precision = "bf16"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $RootDir

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$sequenceRoot = Join-Path $RootDir ("runs\external_speed\sweeps\METR-LA_topk19_seed76_histlen18_24_ep25_{0}" -f $timestamp)
New-Item -ItemType Directory -Path $sequenceRoot -Force | Out-Null

$summaryPath = Join-Path $sequenceRoot "sequence_summary.csv"
$summaryLines = @("hist_len,status,run_dir")

Write-Host ""
Write-Host "==> METR-LA hist_len sweep: 18 -> 24 | TopK=19 | seed=76 | 25 epochs" -ForegroundColor Cyan
Write-Host "    sequence_root = $sequenceRoot"
Write-Host ""

foreach ($histLen in @(18, 24)) {
    $runDir = Join-Path $sequenceRoot ("hist_{0}" -f $histLen)
    Write-Host ""
    Write-Host "===== Starting hist_len=$histLen =====" -ForegroundColor Yellow
    Write-Host "run_dir = $runDir"
    Write-Host ""

    try {
        & (Join-Path $RootDir "local_4090\run_metrla_bestseed_valrmse.ps1") `
            -PythonExe $PythonExe `
            -RootDir $RootDir `
            -Seed 76 `
            -Epochs 25 `
            -HistLen $histLen `
            -EarlyStopPatience 0 `
            -MinEpochs 0 `
            -Optimizer "adam" `
            -VLoss "mse" `
            -WarmupEpochs 0 `
            -SchedulerMonitor "val_loss" `
            -Precision $Precision `
            -AdaptiveTopK 19 `
            -HiddenDim 32 `
            -Lr 0.001 `
            -WeightDecay 1e-5 `
            -RunDir $runDir

        if ($LASTEXITCODE -ne $null -and $LASTEXITCODE -ne 0) {
            Write-Host "hist_len=$histLen exited with code $LASTEXITCODE" -ForegroundColor Red
            $summaryLines += "{0},failed_exit_{1},{2}" -f $histLen, $LASTEXITCODE, $runDir
            continue
        }

        Write-Host "hist_len=$histLen completed." -ForegroundColor Green
        $summaryLines += "{0},completed,{1}" -f $histLen, $runDir
    }
    catch {
        Write-Host "hist_len=$histLen failed: $($_.Exception.Message)" -ForegroundColor Red
        $summaryLines += "{0},failed_exception,{1}" -f $histLen, $runDir
    }
}

$summaryLines | Set-Content -Path $summaryPath

Write-Host ""
Write-Host "All requested hist_len runs finished." -ForegroundColor Cyan
Write-Host "summary = $summaryPath"
