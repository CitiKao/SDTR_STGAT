param(
    [int]$Epochs = 50,
    [int]$BatchSize = 8,
    [int]$HiddenDim = 32,
    [int]$NumHeads = 4,
    [int]$NumLayers = 2,
    [int]$GraphLayers = 1,
    [int]$AdaptiveTopK = 20,
    [double]$Lr = 0.001,
    [string]$Device = "cuda",
    [string]$Precision = "bf16",
    [int]$NumWorkers = 0,
    [int]$HistLen = 14,
    [int]$PredHorizon = 4,
    [string]$ReportHorizonsMinutes = "15,30,60",
    [int]$MaxTimeSteps = 0,
    [string]$RunRoot = ""
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
if (-not $RunRoot) {
    $RunRoot = Join-Path $repoRoot ("runs\paper_speed_benchmarks\nyc_paper7_{0}" -f $timestamp)
}
$RunRoot = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($RunRoot)
New-Item -ItemType Directory -Path $RunRoot -Force | Out-Null

$models = @(
    @{ Name = "astgnn_ltd"; Label = "1/6 ASTGNN-LTD" },
    @{ Name = "icst_dnet"; Label = "2/6 ICST-DNET" },
    @{ Name = "3s_tbln"; Label = "3/6 3S-TBLN" },
    @{ Name = "mvstgcn"; Label = "4/6 MVSTGCN" },
    @{ Name = "nexusqn"; Label = "5/6 NexuSQN" },
    @{ Name = "sgsl_gat_nlstm"; Label = "6/6 SGSL-GAT-nLSTM" }
)

function Format-Duration {
    param([double]$Seconds)
    $span = [TimeSpan]::FromSeconds([Math]::Max($Seconds, 0))
    if ($span.TotalHours -ge 1) {
        return "{0:00}:{1:00}:{2:00}" -f [Math]::Floor($span.TotalHours), $span.Minutes, $span.Seconds
    }
    return "{0:00}:{1:00}" -f $span.Minutes, $span.Seconds
}

function Read-SpeedReport {
    param([string]$MetricsPath)
    $metrics = Get-Content -LiteralPath $MetricsPath -Raw | ConvertFrom-Json
    $speed = $metrics.raw_metrics_report.speed
    $rows = @()
    foreach ($label in @("15min", "30min", "60min")) {
        if ($speed.PSObject.Properties.Name -contains $label) {
            $rows += [pscustomobject]@{
                horizon = $label
                mse = [double]$speed.$label.mse
                rmse = [double]$speed.$label.rmse
                mae = [double]$speed.$label.mae
            }
        }
    }
    return $rows
}

$env:PYTHONUNBUFFERED = "1"
$summaryRows = @()
$overallStart = Get-Date
$summaryJson = Join-Path $RunRoot "paper7_summary.json"
$summaryCsv = Join-Path $RunRoot "paper7_summary.csv"
$transcript = Join-Path $RunRoot "paper7_terminal.log"

Start-Transcript -Path $transcript -Force | Out-Null
try {
    Write-Host "NYC paper benchmark run"
    Write-Host "  run_root   = $RunRoot"
    Write-Host "  epochs     = $Epochs"
    Write-Host "  batch_size = $BatchSize"
    Write-Host "  device     = $Device"
    Write-Host "  precision  = $Precision"
    Write-Host "  horizons   = $ReportHorizonsMinutes"
    Write-Host ""

    for ($i = 0; $i -lt $models.Count; $i++) {
        $model = $models[$i]
        $modelName = [string]$model.Name
        $methodIndex = $i + 1
        $methodStart = Get-Date
        $elapsedOverall = (New-TimeSpan -Start $overallStart -End $methodStart).TotalSeconds
        $etaOverall = if ($i -gt 0) { ($elapsedOverall / $i) * ($models.Count - $i) } else { 0 }
        $modelDir = Join-Path $RunRoot $modelName
        $modelLog = Join-Path $modelDir "terminal.log"
        New-Item -ItemType Directory -Path $modelDir -Force | Out-Null

        Write-Host ""
        Write-Host ("===== Method {0}/{1}: {2} =====" -f $methodIndex, $models.Count, $modelName)
        Write-Host ("overall_progress={0}/{1} ({2:N1}%) elapsed={3} eta={4}" -f `
            $methodIndex, $models.Count, (($i / [double]$models.Count) * 100.0), `
            (Format-Duration $elapsedOverall), (Format-Duration $etaOverall))

        Push-Location $repoRoot
        try {
            $argsList = @(
                "paper_speed_benchmarks\nyc_speed_prediction.py",
                "--model", $modelName,
                "--data-dir", "data",
                "--hist-len", [string]$HistLen,
                "--pred-horizon", [string]$PredHorizon,
                "--report-horizons-minutes", $ReportHorizonsMinutes,
                "--device", $Device,
                "--precision", $Precision,
                "--batch-size", [string]$BatchSize,
                "--epochs", [string]$Epochs,
                "--lr", [string]$Lr,
                "--hidden-dim", [string]$HiddenDim,
                "--num-heads", [string]$NumHeads,
                "--num-layers", [string]$NumLayers,
                "--graph-layers", [string]$GraphLayers,
                "--adaptive-topk", [string]$AdaptiveTopK,
                "--num-workers", [string]$NumWorkers,
                "--log-dir", $modelDir
            )
            if ($MaxTimeSteps -gt 0) {
                $argsList += @("--max-time-steps", [string]$MaxTimeSteps)
            }

            & python -u @argsList 2>&1 | Tee-Object -FilePath $modelLog
            if ($LASTEXITCODE -ne 0) {
                throw "Model $modelName failed with exit code $LASTEXITCODE"
            }
        }
        finally {
            Pop-Location
        }

        $metricsPath = Join-Path $modelDir "predictor_test_metrics.json"
        if (-not (Test-Path -LiteralPath $metricsPath)) {
            throw "Missing metrics file: $metricsPath"
        }

        $methodElapsed = (New-TimeSpan -Start $methodStart -End (Get-Date)).TotalSeconds
        foreach ($row in Read-SpeedReport $metricsPath) {
            $summaryRows += [pscustomobject]@{
                model = $modelName
                horizon = $row.horizon
                mse = $row.mse
                rmse = $row.rmse
                mae = $row.mae
                method_elapsed_sec = [Math]::Round($methodElapsed, 2)
                metrics_path = $metricsPath
            }
        }

        $summaryRows | Export-Csv -LiteralPath $summaryCsv -NoTypeInformation -Encoding UTF8
        $summaryRows | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $summaryJson -Encoding UTF8

        Write-Host ("Finished {0} in {1}" -f $modelName, (Format-Duration $methodElapsed))
        Write-Host "Current 15/30/60 summary:"
        $summaryRows | Format-Table model,horizon,mse,rmse,mae -AutoSize
    }

    $totalElapsed = (New-TimeSpan -Start $overallStart -End (Get-Date)).TotalSeconds
    Write-Host ""
    Write-Host ("All paper benchmarks completed in {0}" -f (Format-Duration $totalElapsed))
    Write-Host "Summary CSV : $summaryCsv"
    Write-Host "Summary JSON: $summaryJson"
    Write-Host "Transcript  : $transcript"
}
finally {
    Stop-Transcript | Out-Null
}
