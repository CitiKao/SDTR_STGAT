param(
    [string]$PythonExe = "C:\Users\WMNL\anaconda3\envs\STDR\python.exe",
    [string]$RootDir = "D:\Citi\STDR\STDR_STGAT",
    [ValidateSet("auto", "bf16", "fp32")]
    [string]$Precision = "bf16"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $RootDir

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$sequenceRoot = Join-Path $RootDir ("runs\external_speed\sweeps\METR-LA_topk_19_20_16_ep20_{0}" -f $timestamp)
New-Item -ItemType Directory -Path $sequenceRoot -Force | Out-Null

$summaryPath = Join-Path $sequenceRoot "sequence_summary.csv"
$summaryLines = @("topk,status,run_dir")

Write-Host ""
Write-Host "==> Sequential METR-LA run: TopK 19 -> 20 -> 16 | full 20 epochs | early stopping OFF" -ForegroundColor Cyan
Write-Host "    sequence_root = $sequenceRoot"
Write-Host ""

foreach ($topk in @(19, 20, 16)) {
    $runDir = Join-Path $sequenceRoot ("topk_{0}" -f $topk)
    Write-Host ""
    Write-Host "===== Starting TopK=$topk =====" -ForegroundColor Yellow
    Write-Host "run_dir = $runDir"
    Write-Host ""

    try {
        & (Join-Path $RootDir "local_4090\run_metrla_bestseed_valrmse.ps1") `
            -PythonExe $PythonExe `
            -RootDir $RootDir `
            -Seed 77 `
            -Epochs 20 `
            -EarlyStopPatience 0 `
            -MinEpochs 0 `
            -Optimizer "adam" `
            -VLoss "mse" `
            -WarmupEpochs 0 `
            -SchedulerMonitor "val_loss" `
            -Precision $Precision `
            -AdaptiveTopK $topk `
            -HiddenDim 32 `
            -Lr 0.001 `
            -WeightDecay 1e-5 `
            -RunDir $runDir

        if ($LASTEXITCODE -ne $null -and $LASTEXITCODE -ne 0) {
            Write-Host "TopK=$topk exited with code $LASTEXITCODE" -ForegroundColor Red
            $summaryLines += "{0},failed_exit_{1},{2}" -f $topk, $LASTEXITCODE, $runDir
            continue
        }

        Write-Host "TopK=$topk completed." -ForegroundColor Green
        $summaryLines += "{0},completed,{1}" -f $topk, $runDir
    }
    catch {
        Write-Host "TopK=$topk failed: $($_.Exception.Message)" -ForegroundColor Red
        $summaryLines += "{0},failed_exception,{1}" -f $topk, $runDir
    }
}

$summaryLines | Set-Content -Path $summaryPath

Write-Host ""
Write-Host "All requested TopK runs finished." -ForegroundColor Cyan
Write-Host "summary = $summaryPath"
