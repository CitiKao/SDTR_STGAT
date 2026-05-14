param(
    [string]$PythonExe = "C:\Users\WMNL\anaconda3\envs\STDR\python.exe",
    [string]$RepoRoot = "D:\Citi\STDR\STDR_STGAT",
    [string]$RootLogDir = "",
    [string]$PlanPath = "",
    [double]$HistoricalBestRmse = 6.084874189687469,
    [double]$InitialSessionBestRmse = 6.101389884775905,
    [string]$InitialSessionBestRunDir = "D:\Citi\STDR\STDR_STGAT\runs\external_speed\ablation_matrix\METR-LA_weighted_missing_ablation_ep30_20260422_225900\C_both_on",
    [string]$ReviewNotes = ""
)

$ErrorActionPreference = "Stop"

function Write-Banner {
    param(
        [string]$Title,
        [string]$Message
    )
    Write-Host ""
    Write-Host "============================================================"
    Write-Host $Title
    Write-Host $Message
    Write-Host "============================================================"
}

function Set-QueueStatus {
    param(
        [string]$Path,
        [hashtable]$Status
    )
    $Status | ConvertTo-Json -Depth 8 | Set-Content -Path $Path -Encoding utf8
}

function Get-BestEpochSummary {
    param([string]$PredictorLogPath)
    if (-not (Test-Path $PredictorLogPath)) {
        return @{ epoch = $null; best_val_rmse = $null }
    }
    $history = Get-Content $PredictorLogPath -Raw | ConvertFrom-Json
    if ($null -eq $history -or $history.Count -eq 0) {
        return @{ epoch = $null; best_val_rmse = $null }
    }
    $best = $history | Sort-Object val_raw_speed_rmse | Select-Object -First 1
    return @{
        epoch = [int]$best.epoch
        best_val_rmse = [double]$best.val_raw_speed_rmse
    }
}

function Get-ExpValue {
    param(
        [pscustomobject]$Exp,
        [string]$Name,
        $DefaultValue
    )
    $prop = $Exp.PSObject.Properties[$Name]
    if ($null -eq $prop) {
        return $DefaultValue
    }
    return $prop.Value
}

function Get-ComparableAnchor {
    param(
        [string]$MetricsPath,
        [string]$MetaPath
    )
    if (-not ((Test-Path $MetricsPath) -and (Test-Path $MetaPath))) {
        return $null
    }
    $metrics = Get-Content -Path $MetricsPath -Raw | ConvertFrom-Json
    $meta = Get-Content -Path $MetaPath -Raw | ConvertFrom-Json
    return @{
        processed_dataset_fingerprint = [string]$metrics.processed_dataset_fingerprint
        preprocessing_variant_id = [string]$metrics.preprocessing_variant_id
        speed_metric_protocol = [string]$metrics.speed_metric_protocol
        split_policy = [string]$metrics.split_policy
        split_alignment = [string]$metrics.split_alignment
        pred_horizon = [int]$meta.pred_horizon
        requested_minutes = [string](($meta.report_horizons.requested_minutes | ForEach-Object { [string]$_ }) -join ",")
    }
}

if ([string]::IsNullOrWhiteSpace($PlanPath)) {
    throw "PlanPath is required."
}
if (-not (Test-Path $PlanPath)) {
    throw "PlanPath does not exist: $PlanPath"
}

if ([string]::IsNullOrWhiteSpace($RootLogDir)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $RootLogDir = Join-Path $RepoRoot ("runs\external_speed\tuning_sessions\METR-LA_corrected_batch_" + $stamp)
}

$null = New-Item -ItemType Directory -Path $RootLogDir -Force
$summaryPath = Join-Path $RootLogDir "summary.tsv"
$historyPath = Join-Path $RootLogDir "queue_history.jsonl"
$breakthroughPath = Join-Path $RootLogDir "queue_breakthroughs.jsonl"
$statusPath = Join-Path $RootLogDir "queue_status.json"
$planMirrorPath = Join-Path $RootLogDir "planned_experiments.tsv"
$activePath = Join-Path $RootLogDir "active_run.txt"
$reviewPath = Join-Path $RootLogDir "review_rounds.md"

"run_id`tlog_dir`tbranch`tweighted_fixed`tmissing_aware`ttopk`tseed`tlr`tweight_decay`tprecision`thidden_dim`thist_len`tepochs`tscheduler_monitor`tscheduler_patience`tscheduler_cooldown`tbest_val_rmse`tbest_epoch`ttest_rmse`trmse_15m`trmse_30m`trmse_60m`tnew_session_best`tbeat_historical" | Set-Content -Path $summaryPath -Encoding utf8
Set-Content -Path $historyPath -Value "" -Encoding utf8
Set-Content -Path $breakthroughPath -Value "" -Encoding utf8

$planRaw = Get-Content -Path $PlanPath -Raw | ConvertFrom-Json
if ($planRaw -is [System.Array]) {
    $planObjects = $planRaw
} else {
    $planObjects = @($planRaw)
}
if ($planObjects.Count -eq 0) {
    throw "PlanPath contains no experiments: $PlanPath"
}

"run_id`tbranch`tweighted_fixed`tmissing_aware`ttopk`tseed`tlr`tweight_decay`tprecision`thidden_dim`thist_len`tepochs`tscheduler_monitor`tscheduler_patience`tscheduler_cooldown" | Set-Content -Path $planMirrorPath -Encoding utf8
foreach ($planExp in $planObjects) {
    $weighted = [bool](Get-ExpValue -Exp $planExp -Name "weighted_fixed" -DefaultValue $true)
    $missingAware = [bool](Get-ExpValue -Exp $planExp -Name "missing_aware" -DefaultValue $true)
    $branch = [string](Get-ExpValue -Exp $planExp -Name "branch" -DefaultValue $(if ($weighted) { "C" } else { "B" }))
    $runId = [string](Get-ExpValue -Exp $planExp -Name "run_id" -DefaultValue "")
    if ([string]::IsNullOrWhiteSpace($runId)) {
        throw "Each experiment needs a run_id."
    }
    $line = @(
        $runId,
        $branch,
        $weighted,
        $missingAware,
        [int](Get-ExpValue -Exp $planExp -Name "topk" -DefaultValue 19),
        [int](Get-ExpValue -Exp $planExp -Name "seed" -DefaultValue 76),
        [string](Get-ExpValue -Exp $planExp -Name "lr" -DefaultValue "1e-3"),
        [string](Get-ExpValue -Exp $planExp -Name "weight_decay" -DefaultValue "1e-5"),
        [string](Get-ExpValue -Exp $planExp -Name "precision" -DefaultValue "bf16"),
        [int](Get-ExpValue -Exp $planExp -Name "hidden_dim" -DefaultValue 32),
        [int](Get-ExpValue -Exp $planExp -Name "hist_len" -DefaultValue 12),
        [int](Get-ExpValue -Exp $planExp -Name "epochs" -DefaultValue 30),
        [string](Get-ExpValue -Exp $planExp -Name "scheduler_monitor" -DefaultValue "val_loss"),
        [int](Get-ExpValue -Exp $planExp -Name "scheduler_patience" -DefaultValue 8),
        [int](Get-ExpValue -Exp $planExp -Name "scheduler_cooldown" -DefaultValue 2)
    ) -join "`t"
    Add-Content -Path $planMirrorPath -Value $line -Encoding utf8
}

$reviewText = @"
# METR-LA Corrected Batch Review

- Historical preserved baseline test RMSE: $HistoricalBestRmse
- Session-best reference test RMSE: $InitialSessionBestRmse
- Plan source: $PlanPath
- Runs in this batch: $($planObjects.Count)
"@
if (-not [string]::IsNullOrWhiteSpace($ReviewNotes)) {
    $reviewText += "`r`n`r`n## Notes`r`n$ReviewNotes"
}
Set-Content -Path $reviewPath -Value $reviewText -Encoding utf8

$expectedComparableAnchor = Get-ComparableAnchor `
    -MetricsPath (Join-Path $InitialSessionBestRunDir "predictor_test_metrics.json") `
    -MetaPath (Join-Path $InitialSessionBestRunDir "stgat_meta.json")

$sessionBestRmse = [double]$InitialSessionBestRmse
$sessionBestRunId = "seed76_corrected_ablation_C_both_on"
$sessionBestRunDir = $InitialSessionBestRunDir
$runsFinished = 0
$runsFailed = 0
$consecutiveFailures = 0

$status = @{
    session_tag = Split-Path $RootLogDir -Leaf
    active_run_id = $null
    active_run_dir = $null
    runs_finished = 0
    runs_failed = 0
    historical_best_ref = $HistoricalBestRmse
    session_best_run_id = $sessionBestRunId
    session_best_test_rmse = $sessionBestRmse
    session_best_run_dir = $sessionBestRunDir
    last_breakthrough_at = $null
    last_breakthrough_type = $null
    plan_path = $PlanPath
}
Set-QueueStatus -Path $statusPath -Status $status

$env:PYTHONUNBUFFERED = "1"
Set-Location $RepoRoot

foreach ($planExp in $planObjects) {
    if ($consecutiveFailures -ge 3) {
        Write-Banner -Title "QUEUE STOPPED" -Message "Three consecutive failures reached. Inspect logs before continuing."
        break
    }

    $weighted = [bool](Get-ExpValue -Exp $planExp -Name "weighted_fixed" -DefaultValue $true)
    $missingAware = [bool](Get-ExpValue -Exp $planExp -Name "missing_aware" -DefaultValue $true)
    $branch = [string](Get-ExpValue -Exp $planExp -Name "branch" -DefaultValue $(if ($weighted) { "C" } else { "B" }))
    $runId = [string](Get-ExpValue -Exp $planExp -Name "run_id" -DefaultValue "")
    $topk = [int](Get-ExpValue -Exp $planExp -Name "topk" -DefaultValue 19)
    $seed = [int](Get-ExpValue -Exp $planExp -Name "seed" -DefaultValue 76)
    $lr = [string](Get-ExpValue -Exp $planExp -Name "lr" -DefaultValue "1e-3")
    $weightDecay = [string](Get-ExpValue -Exp $planExp -Name "weight_decay" -DefaultValue "1e-5")
    $precision = [string](Get-ExpValue -Exp $planExp -Name "precision" -DefaultValue "bf16")
    $hiddenDim = [int](Get-ExpValue -Exp $planExp -Name "hidden_dim" -DefaultValue 32)
    $histLen = [int](Get-ExpValue -Exp $planExp -Name "hist_len" -DefaultValue 12)
    $epochs = [int](Get-ExpValue -Exp $planExp -Name "epochs" -DefaultValue 30)
    $batchSize = [int](Get-ExpValue -Exp $planExp -Name "batch_size" -DefaultValue 32)
    $schedulerMonitor = [string](Get-ExpValue -Exp $planExp -Name "scheduler_monitor" -DefaultValue "val_loss")
    $schedulerPatience = [int](Get-ExpValue -Exp $planExp -Name "scheduler_patience" -DefaultValue 8)
    $schedulerCooldown = [int](Get-ExpValue -Exp $planExp -Name "scheduler_cooldown" -DefaultValue 2)
    $schedulerMinLr = [string](Get-ExpValue -Exp $planExp -Name "scheduler_min_lr" -DefaultValue "1e-5")

    $runDir = Join-Path $RootLogDir $runId
    $stdoutPath = Join-Path $runDir "live.stdout.log"
    $null = New-Item -ItemType Directory -Path $runDir -Force

    @(
        "run_id=$runId",
        "run_dir=$runDir",
        "started_at=$(Get-Date -Format o)"
    ) | Set-Content -Path $activePath -Encoding utf8

    $status.active_run_id = $runId
    $status.active_run_dir = $runDir
    Set-QueueStatus -Path $statusPath -Status $status

    $args = @(
        "external_speed_benchmarks\train_sensor_speed.py",
        "--dataset-dir", (Join-Path $RepoRoot "data\external_datasets\processed\METR-LA"),
        "--dataset-name", "METR-LA",
        "--log-dir", $runDir,
        "--hist-len", [string]$histLen,
        "--pred-horizon", "12",
        "--report-horizons-minutes", "15,30,60",
        "--hidden-dim", [string]$hiddenDim,
        "--num-heads", "4",
        "--num-st-blocks", "2",
        "--num-gtcn-layers", "2",
        "--kernel-size", "3",
        "--adaptive-topk", [string]$topk,
        "--epochs", [string]$epochs,
        "--batch-size", [string]$batchSize,
        "--lr", [string]$lr,
        "--optimizer", "adam",
        "--weight-decay", [string]$weightDecay,
        "--warmup-epochs", "0",
        "--grad-clip-norm", "5.0",
        "--v-loss", "mse",
        "--seed", [string]$seed,
        "--device", "auto",
        "--precision", $precision,
        "--val-interval", "1",
        "--log-interval", "1",
        "--split-policy", "benchmark_contiguous",
        "--split-alignment", "none",
        "--scheduler-monitor", $schedulerMonitor,
        "--scheduler-factor", "0.5",
        "--scheduler-patience", [string]$schedulerPatience,
        "--scheduler-cooldown", [string]$schedulerCooldown,
        "--scheduler-min-lr", [string]$schedulerMinLr,
        "--early-stop-patience", "0",
        "--min-epochs", "0",
        "--outlier-cleaning-mode", "train_quantile_clip",
        "--fresh"
    )
    if (-not $weighted) {
        $args += "--disable-weighted-fixed-graph"
    }
    if (-not $missingAware) {
        $args += "--disable-missing-aware-history"
    }

    Write-Banner -Title ("Starting " + $runId) -Message ("Run dir: " + $runDir)
    & $PythonExe @args 2>&1 | Tee-Object -FilePath $stdoutPath
    $exitCode = $LASTEXITCODE

    $metricsPath = Join-Path $runDir "predictor_test_metrics.json"
    $metaPath = Join-Path $runDir "stgat_meta.json"
    $predictorLogPath = Join-Path $runDir "predictor_log.json"

    if (($exitCode -ne 0) -or -not (Test-Path $metricsPath) -or -not (Test-Path $metaPath)) {
        $runsFailed += 1
        $consecutiveFailures += 1
        $status.runs_failed = $runsFailed
        $status.active_run_id = $null
        $status.active_run_dir = $null
        Set-QueueStatus -Path $statusPath -Status $status

        $failureRecord = @{
            session_tag = $status.session_tag
            run_id = $runId
            launched_at = (Get-Date -Format o)
            finished_at = (Get-Date -Format o)
            run_dir = $runDir
            seed = $seed
            status = "failed"
            exit_code = $exitCode
            branch = $branch
        } | ConvertTo-Json -Compress
        Add-Content -Path $historyPath -Value $failureRecord -Encoding utf8
        Write-Warning ("Run failed: " + $runId + " (exit=" + $exitCode + ")")
        continue
    }

    $consecutiveFailures = 0
    $runsFinished += 1

    $metrics = Get-Content -Path $metricsPath -Raw | ConvertFrom-Json
    $meta = Get-Content -Path $metaPath -Raw | ConvertFrom-Json
    $best = Get-BestEpochSummary -PredictorLogPath $predictorLogPath

    $testRmse = [double]$metrics.raw_metrics.speed.rmse
    $rmse15 = [double]$metrics.raw_metrics_report.speed.'15min'.rmse
    $rmse30 = [double]$metrics.raw_metrics_report.speed.'30min'.rmse
    $rmse60 = [double]$metrics.raw_metrics_report.speed.'60min'.rmse
    $bestValRmse = [double]$metrics.selected_checkpoint_metric
    $bestEpoch = $best.epoch
    $sessionBestBefore = $sessionBestRmse

    if ($null -eq $expectedComparableAnchor) {
        $expectedComparableAnchor = @{
            processed_dataset_fingerprint = [string]$metrics.processed_dataset_fingerprint
            preprocessing_variant_id = [string]$metrics.preprocessing_variant_id
            speed_metric_protocol = [string]$metrics.speed_metric_protocol
            split_policy = [string]$metrics.split_policy
            split_alignment = [string]$metrics.split_alignment
            pred_horizon = [int]$meta.pred_horizon
            requested_minutes = [string](($meta.report_horizons.requested_minutes | ForEach-Object { [string]$_ }) -join ",")
        }
    }
    $comparable = (
        [string]$metrics.processed_dataset_fingerprint -eq [string]$expectedComparableAnchor.processed_dataset_fingerprint -and
        [string]$metrics.preprocessing_variant_id -eq [string]$expectedComparableAnchor.preprocessing_variant_id -and
        [string]$metrics.speed_metric_protocol -eq [string]$expectedComparableAnchor.speed_metric_protocol -and
        [string]$metrics.split_policy -eq [string]$expectedComparableAnchor.split_policy -and
        [string]$metrics.split_alignment -eq [string]$expectedComparableAnchor.split_alignment -and
        [int]$meta.pred_horizon -eq [int]$expectedComparableAnchor.pred_horizon -and
        [string](($meta.report_horizons.requested_minutes | ForEach-Object { [string]$_ }) -join ",") -eq [string]$expectedComparableAnchor.requested_minutes
    )

    $isNewSessionBest = $false
    if ($comparable -and $testRmse -lt $sessionBestRmse) {
        $isNewSessionBest = $true
        $sessionBestRmse = $testRmse
        $sessionBestRunId = $runId
        $sessionBestRunDir = $runDir
    }
    $beatHistorical = $comparable -and ($testRmse -lt $HistoricalBestRmse)

    $line = @(
        $runId,
        $runDir,
        $branch,
        $weighted,
        $missingAware,
        $topk,
        $seed,
        $lr,
        $weightDecay,
        $precision,
        $hiddenDim,
        $histLen,
        $epochs,
        $schedulerMonitor,
        $schedulerPatience,
        $schedulerCooldown,
        $bestValRmse,
        $bestEpoch,
        $testRmse,
        $rmse15,
        $rmse30,
        $rmse60,
        $isNewSessionBest,
        $beatHistorical
    ) -join "`t"
    Add-Content -Path $summaryPath -Value $line -Encoding utf8

    $historyRecord = @{
        session_tag = $status.session_tag
        run_id = $runId
        launched_at = (Get-Date -Format o)
        finished_at = (Get-Date -Format o)
        run_dir = $runDir
        branch = $branch
        seed = $seed
        status = "completed"
        exit_code = $exitCode
        configured_epochs = $epochs
        completed_epochs = [int]$meta.training_control.completed_epochs
        training_end_reason = [string]$meta.training_control.training_end_reason
        hidden_dim = $hiddenDim
        hist_len = $histLen
        pred_horizon = [int]$meta.pred_horizon
        adaptive_topk = $topk
        fixed_graph_weighted = $weighted
        use_speed_history_mask = $missingAware
        precision = $precision
        lr = [double]$lr
        weight_decay = [double]$weightDecay
        scheduler_monitor = $schedulerMonitor
        scheduler_patience = $schedulerPatience
        scheduler_cooldown = $schedulerCooldown
        scheduler_min_lr = $schedulerMinLr
        processed_dataset_fingerprint = [string]$metrics.processed_dataset_fingerprint
        preprocessing_variant_id = [string]$metrics.preprocessing_variant_id
        speed_metric_protocol = [string]$metrics.speed_metric_protocol
        split_policy = [string]$metrics.split_policy
        split_alignment = [string]$metrics.split_alignment
        best_val_rmse = $bestValRmse
        best_epoch = $bestEpoch
        test_rmse = $testRmse
        rmse15 = $rmse15
        rmse30 = $rmse30
        rmse60 = $rmse60
        selected_checkpoint = [string]$metrics.selected_checkpoint
        session_best_before = $sessionBestBefore
        session_best_after = $sessionBestRmse
        delta_vs_session_best = ($testRmse - $sessionBestBefore)
        is_new_session_best = $isNewSessionBest
        historical_best_ref = $HistoricalBestRmse
        delta_vs_historical_best = ($testRmse - $HistoricalBestRmse)
        beat_historical_best = $beatHistorical
        comparable = $comparable
    } | ConvertTo-Json -Compress
    Add-Content -Path $historyPath -Value $historyRecord -Encoding utf8

    if ($isNewSessionBest -or $beatHistorical) {
        $eventType = if ($beatHistorical) { "new_historical_best" } else { "new_session_best" }
        $breakthroughRecord = @{
            timestamp = (Get-Date -Format o)
            event_type = $eventType
            run_id = $runId
            run_dir = $runDir
            branch = $branch
            seed = $seed
            test_rmse = $testRmse
            delta_vs_session_best = ($testRmse - $sessionBestBefore)
            delta_vs_historical_best = ($testRmse - $HistoricalBestRmse)
            hidden_dim = $hiddenDim
            hist_len = $histLen
            adaptive_topk = $topk
            fixed_graph_weighted = $weighted
            use_speed_history_mask = $missingAware
            precision = $precision
            lr = [double]$lr
            weight_decay = [double]$weightDecay
            scheduler_monitor = $schedulerMonitor
            scheduler_patience = $schedulerPatience
            scheduler_cooldown = $schedulerCooldown
            scheduler_min_lr = $schedulerMinLr
        } | ConvertTo-Json -Compress
        Add-Content -Path $breakthroughPath -Value $breakthroughRecord -Encoding utf8

        if ($beatHistorical) {
            Write-Banner -Title "NEW HISTORICAL BEST" -Message ($runId + " | test_rmse=" + $testRmse.ToString("F6") + " | vs_6.084874=" + ($testRmse - $HistoricalBestRmse).ToString("F6"))
        } else {
            Write-Banner -Title "NEW SESSION BEST" -Message ($runId + " | test_rmse=" + $testRmse.ToString("F6") + " | delta=" + ($testRmse - $sessionBestBefore).ToString("F6"))
        }
        $status.last_breakthrough_at = (Get-Date -Format o)
        $status.last_breakthrough_type = $eventType
    }

    $status.runs_finished = $runsFinished
    $status.active_run_id = $null
    $status.active_run_dir = $null
    $status.session_best_run_id = $sessionBestRunId
    $status.session_best_test_rmse = $sessionBestRmse
    $status.session_best_run_dir = $sessionBestRunDir
    Set-QueueStatus -Path $statusPath -Status $status
}

Write-Banner -Title "QUEUE FINISHED" -Message ("Session best RMSE=" + $sessionBestRmse.ToString("F6") + " | run=" + $sessionBestRunId)
