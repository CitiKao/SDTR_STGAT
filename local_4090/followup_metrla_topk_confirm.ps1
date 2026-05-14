param(
    [string]$PythonExe = "C:\Users\WMNL\anaconda3\envs\STDR\python.exe",
    [string]$RootDir = "D:\Citi\STDR\STDR_STGAT",
    [string]$BaseSweepRoot,
    [int]$ExtraTopK = 9,
    [int]$TopCandidateCount = 3,
    [int]$RepeatCount = 3,
    [int]$Seed = 77,
    [int]$Epochs = 40,
    [int]$EarlyStopPatience = 8,
    [int]$MinEpochs = 18,
    [ValidateSet("auto", "bf16", "fp32")]
    [string]$Precision = "bf16"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $RootDir

if ([string]::IsNullOrWhiteSpace($BaseSweepRoot)) {
    throw "BaseSweepRoot is required."
}

if (-not (Test-Path -LiteralPath $BaseSweepRoot)) {
    throw "BaseSweepRoot does not exist: $BaseSweepRoot"
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$followupRoot = Join-Path (Split-Path -Parent $BaseSweepRoot) ("METR-LA_topk_followup_{0}" -f $timestamp)
New-Item -ItemType Directory -Path $followupRoot -Force | Out-Null

$followupLogPath = Join-Path $followupRoot "followup.log"
$candidateCsvPath = Join-Path $followupRoot "candidates.csv"
$confirmationCsvPath = Join-Path $followupRoot "confirmation_summary.csv"

function Write-FollowupLog {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Host $line
    Add-Content -Path $followupLogPath -Value $line
}

function Wait-ForSweepFinish {
    param([string]$SweepRoot)

    $summaryCsv = Join-Path $SweepRoot "summary.csv"
    while (-not (Test-Path -LiteralPath $summaryCsv)) {
        Write-FollowupLog "Waiting for base sweep to finish: $SweepRoot"
        Start-Sleep -Seconds 60
    }
    Write-FollowupLog "Base sweep finished. Found summary: $summaryCsv"
}

function Get-BestEpoch {
    param([string]$RunDir)

    foreach ($path in @((Join-Path $RunDir "run_summary.json"), (Join-Path $RunDir "stgat_meta.json"))) {
        if (-not (Test-Path -LiteralPath $path)) {
            continue
        }
        try {
            $match = Select-String -Path $path -Pattern '"best_epoch"\s*:\s*(\d+)' -AllMatches
            if ($match -and $match.Matches.Count -gt 0) {
                return [int]$match.Matches[0].Groups[1].Value
            }
        }
        catch {
        }
    }
    return $null
}

function Get-RunMetrics {
    param([string]$RunDir, [int]$TopK)

    $metricsPath = Join-Path $RunDir "predictor_test_metrics.json"
    if (-not (Test-Path -LiteralPath $metricsPath)) {
        return $null
    }

    $metrics = Get-Content -Path $metricsPath -Raw -Encoding UTF8 | ConvertFrom-Json
    return [pscustomobject]@{
        TopK = $TopK
        BestValRMSE = [double]$metrics.selected_checkpoint_metric
        BestEpoch = Get-BestEpoch -RunDir $RunDir
        TestRMSE = [double]$metrics.raw_metrics.speed.rmse
        RMSE15 = [double]$metrics.raw_metrics_report.speed.'15min'.rmse
        RMSE30 = [double]$metrics.raw_metrics_report.speed.'30min'.rmse
        RMSE60 = [double]$metrics.raw_metrics_report.speed.'60min'.rmse
        CompletedEpochs = [int]$metrics.training_control.completed_epochs
        RunDir = $RunDir
    }
}

function Invoke-TopKRun {
    param(
        [int]$TopK,
        [string]$RunDir
    )

    Write-FollowupLog "Launching TopK=$TopK -> $RunDir"
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
        -AdaptiveTopK $TopK `
        -RunDir $RunDir

    if ($LASTEXITCODE -ne $null -and $LASTEXITCODE -ne 0) {
        throw "TopK=$TopK exited with code $LASTEXITCODE"
    }

    $metrics = Get-RunMetrics -RunDir $RunDir -TopK $TopK
    if ($null -eq $metrics) {
        throw "TopK=$TopK completed but metrics file was not found in $RunDir"
    }
    return $metrics
}

Wait-ForSweepFinish -SweepRoot $BaseSweepRoot

$extraRunDir = Join-Path $followupRoot ("topk_{0}" -f $ExtraTopK)
$extraMetrics = Invoke-TopKRun -TopK $ExtraTopK -RunDir $extraRunDir
Write-FollowupLog ("TopK={0} done. Test RMSE={1:F6}" -f $extraMetrics.TopK, $extraMetrics.TestRMSE)

$allRows = @()
for ($topk = 10; $topk -le 24; $topk++) {
    $runDir = Join-Path $BaseSweepRoot ("topk_{0}" -f $topk)
    $metrics = Get-RunMetrics -RunDir $runDir -TopK $topk
    if ($null -ne $metrics) {
        $allRows += $metrics
    }
}
$allRows += $extraMetrics

$candidateRows = $allRows | Sort-Object TestRMSE, BestValRMSE | Select-Object -First $TopCandidateCount
$candidateLines = @("topk,best_val_rmse,best_epoch,test_rmse,rmse15,rmse30,rmse60,completed_epochs,run_dir")
foreach ($row in $candidateRows) {
    $candidateLines += ("{0},{1:F6},{2},{3:F6},{4:F6},{5:F6},{6:F6},{7},{8}" -f `
        $row.TopK, $row.BestValRMSE, $row.BestEpoch, $row.TestRMSE, $row.RMSE15, $row.RMSE30, $row.RMSE60, $row.CompletedEpochs, $row.RunDir)
}
$candidateLines | Set-Content -Path $candidateCsvPath

Write-FollowupLog ("Selected candidates for confirmation: {0}" -f (($candidateRows | ForEach-Object { $_.TopK }) -join ", "))

$confirmationLines = @("topk,repeat,best_val_rmse,best_epoch,test_rmse,rmse15,rmse30,rmse60,completed_epochs,run_dir")
foreach ($candidate in $candidateRows) {
    for ($repeat = 1; $repeat -le $RepeatCount; $repeat++) {
        $confirmDir = Join-Path $followupRoot ("confirm_topk_{0}_r{1}" -f $candidate.TopK, $repeat)
        $metrics = Invoke-TopKRun -TopK ([int]$candidate.TopK) -RunDir $confirmDir
        Write-FollowupLog ("Confirm TopK={0} repeat={1} done. Test RMSE={2:F6}" -f $candidate.TopK, $repeat, $metrics.TestRMSE)
        $confirmationLines += ("{0},{1},{2:F6},{3},{4:F6},{5:F6},{6:F6},{7:F6},{8},{9}" -f `
            $metrics.TopK, $repeat, $metrics.BestValRMSE, $metrics.BestEpoch, $metrics.TestRMSE, $metrics.RMSE15, $metrics.RMSE30, $metrics.RMSE60, $metrics.CompletedEpochs, $metrics.RunDir)
        $confirmationLines | Set-Content -Path $confirmationCsvPath
    }
}

Write-FollowupLog "Follow-up queue finished."
