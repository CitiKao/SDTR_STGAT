param(
    [string]$PythonExe = "C:\Users\WMNL\anaconda3\envs\STDR\python.exe",
    [string]$RootDir = "D:\Citi\STDR\STDR_STGAT",
    [int]$StartTopK = 10,
    [int]$EndTopK = 24,
    [int]$Seed = 77,
    [int]$Epochs = 40,
    [int]$EarlyStopPatience = 8,
    [int]$MinEpochs = 18,
    [ValidateSet("auto", "bf16", "fp32")]
    [string]$Precision = "bf16"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $RootDir

if ($EndTopK -lt $StartTopK) {
    throw "EndTopK must be greater than or equal to StartTopK."
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$sweepRoot = Join-Path $RootDir ("runs\external_speed\sweeps\METR-LA_topk_{0}_{1}_seed{2}_{3}" -f $StartTopK, $EndTopK, $Seed, $timestamp)
New-Item -ItemType Directory -Path $sweepRoot -Force | Out-Null

$sweepLogPath = Join-Path $sweepRoot "sweep.log"
$summaryCsvPath = Join-Path $sweepRoot "summary.csv"
$summaryLines = @("topk,status,run_dir")

function Write-SweepLog {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Host $line
    Add-Content -Path $sweepLogPath -Value $line
}

Write-SweepLog "Starting sequential AdaptiveTopK sweep from $StartTopK to $EndTopK."
Write-SweepLog "Sweep root: $sweepRoot"

for ($topk = $StartTopK; $topk -le $EndTopK; $topk++) {
    $runDir = Join-Path $sweepRoot ("topk_{0}" -f $topk)
    Write-SweepLog "Launching TopK=$topk -> $runDir"

    try {
        & (Join-Path $RootDir "local_4090\run_metrla_bestseed_valrmse.ps1") `
            -PythonExe $PythonExe `
            -RootDir $RootDir `
            -Seed $Seed `
            -Epochs $Epochs `
            -EarlyStopPatience $EarlyStopPatience `
            -MinEpochs $MinEpochs `
            -Optimizer "adam" `
            -VLoss "mse" `
            -WarmupEpochs 0 `
            -SchedulerMonitor "val_loss" `
            -Precision $Precision `
            -AdaptiveTopK $topk `
            -RunDir $runDir

        if ($LASTEXITCODE -ne $null -and $LASTEXITCODE -ne 0) {
            Write-SweepLog "TopK=$topk finished with native exit code $LASTEXITCODE."
            $summaryLines += "{0},failed_exit_{1},{2}" -f $topk, $LASTEXITCODE, $runDir
            continue
        }

        Write-SweepLog "TopK=$topk completed successfully."
        $summaryLines += "{0},completed,{1}" -f $topk, $runDir
    }
    catch {
        Write-SweepLog "TopK=$topk failed: $($_.Exception.Message)"
        $summaryLines += "{0},failed_exception,{1}" -f $topk, $runDir
    }
}

$summaryLines | Set-Content -Path $summaryCsvPath
Write-SweepLog "Sweep finished. Summary CSV: $summaryCsvPath"
