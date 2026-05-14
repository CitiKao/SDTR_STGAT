param(
    [string]$PythonExe = "C:\Users\WMNL\anaconda3\envs\STDR\python.exe",
    [string]$RepoRoot = "D:\Citi\STDR\STDR_STGAT",
    [string]$RootLogDir = "",
    [string]$PlanPath = "",
    [string]$DatasetName = "PEMS-BAY",
    [string]$DatasetDir = "D:\Citi\STDR\STDR_STGAT\data\external_datasets\processed\PEMS-BAY_sensor_node_haversine",
    [string]$SplitPolicy = "project_monthly",
    [string]$SplitAlignment = "none",
    [string]$ReportHorizonsMinutes = "15,30,60",
    [double]$ReferenceRmse = 3.3369038658799606,
    [string]$ReferenceRunDir = "D:\Citi\STDR\STDR_STGAT\runs\external_speed\PEMS-BAY_project_monthly_bs32_ep100_resumeable_20260420_142901",
    [double]$InitialSessionBestRmse = 1e18,
    [string]$InitialSessionBestRunDir = "__none__",
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
    $history = Get-Content -Path $PredictorLogPath -Raw | ConvertFrom-Json
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
    $RootLogDir = Join-Path $RepoRoot ("runs\external_speed\autopilot_sessions\PEMS-BAY_batch_" + $stamp)
}

$null = New-Item -ItemType Directory -Path $RootLogDir -Force
$summaryPath = Join-Path $RootLogDir "summary.tsv"
$historyPath = Join-Path $RootLogDir "queue_history.jsonl"
$breakthroughPath = Join-Path $RootLogDir "queue_breakthroughs.jsonl"
$statusPath = Join-Path $RootLogDir "queue_status.json"
$planMirrorPath = Join-Path $RootLogDir "planned_experiments.tsv"
$activePath = Join-Path $RootLogDir "active_run.txt"
$reviewPath = Join-Path $RootLogDir "review_rounds.md"

"run_id`tlog_dir`tlineage`tdataset_dir`tweighted_fixed`tmissing_aware`ttopk`tseed`tlr`tweight_decay`tprecision`thidden_dim`thist_len`tepochs`tscheduler_monitor`tscheduler_patience`tscheduler_cooldown`tbest_val_rmse`tbest_epoch`ttest_rmse`trmse_15m`trmse_30m`trmse_60m`tnew_session_best`tbeat_reference" | Set-Content -Path $summaryPath -Encoding utf8
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

"run_id`tlineage`tdataset_dir`tweighted_fixed`tmissing_aware`ttopk`tseed`tlr`tweight_decay`tprecision`thidden_dim`thist_len`tepochs`tscheduler_monitor`tscheduler_patience`tscheduler_cooldown" | Set-Content -Path $planMirrorPath -Encoding utf8
foreach ($planExp in $planObjects) {
    $weighted = [bool](Get-ExpValue -Exp $planExp -Name "weighted_fixed" -DefaultValue $false)
    $missingAware = [bool](Get-ExpValue -Exp $planExp -Name "missing_aware" -DefaultValue $false)
    $runId = [string](Get-ExpValue -Exp $planExp -Name "run_id" -DefaultValue "")
    if ([string]::IsNullOrWhiteSpace($runId)) {
        throw "Each experiment needs a run_id."
    }
    $line = @(
        $runId,
        [string](Get-ExpValue -Exp $planExp -Name "lineage" -DefaultValue "legacy_haversine"),
        [string](Get-ExpValue -Exp $planExp -Name "dataset_dir" -DefaultValue $DatasetDir),
        $weighted,
        $missingAware,
        [int](Get-ExpValue -Exp $planExp -Name "topk" -DefaultValue 16),
        [int](Get-ExpValue -Exp $planExp -Name "seed" -DefaultValue 42),
        [string](Get-ExpValue -Exp $planExp -Name "lr" -DefaultValue "1e-3"),
        [string](Get-ExpValue -Exp $planExp -Name "weight_decay" -DefaultValue "1e-5"),
        [string](Get-ExpValue -Exp $planExp -Name "precision" -DefaultValue "bf16"),
        [int](Get-ExpValue -Exp $planExp -Name "hidden_dim" -DefaultValue 32),
        [int](Get-ExpValue -Exp $planExp -Name "hist_len" -DefaultValue 12),
        [int](Get-ExpValue -Exp $planExp -Name "epochs" -DefaultValue 40),
        [string](Get-ExpValue -Exp $planExp -Name "scheduler_monitor" -DefaultValue "val_loss"),
        [int](Get-ExpValue -Exp $planExp -Name "scheduler_patience" -DefaultValue 8),
        [int](Get-ExpValue -Exp $planExp -Name "scheduler_cooldown" -DefaultValue 2)
    ) -join "`t"
    Add-Content -Path $planMirrorPath -Value $line -Encoding utf8
}

$reviewText = @"
# PEMS-BAY Autopilot Batch Review

- Reference RMSE snapshot: $ReferenceRmse
- Reference run: $ReferenceRunDir
- Plan source: $PlanPath
- Runs in this batch: $($planObjects.Count)
"@
if (-not [string]::IsNullOrWhiteSpace($ReviewNotes)) {
    $reviewText += "`r`n`r`n## Notes`r`n$ReviewNotes"
}
Set-Content -Path $reviewPath -Value $reviewText -Encoding utf8

$expectedComparableAnchor = $null
if ($InitialSessionBestRunDir -eq "__none__") {
    $InitialSessionBestRunDir = ""
}
if (-not [string]::IsNullOrWhiteSpace($InitialSessionBestRunDir)) {
    $expectedComparableAnchor = Get-ComparableAnchor `
        -MetricsPath (Join-Path $InitialSessionBestRunDir "predictor_test_metrics.json") `
        -MetaPath (Join-Path $InitialSessionBestRunDir "stgat_meta.json")
}

$hasSessionBest = $InitialSessionBestRmse -lt 1e17
$sessionBestRmse = if ($hasSessionBest) { [double]$InitialSessionBestRmse } else { $null }
$sessionBestRunId = if ($hasSessionBest) { Split-Path $InitialSessionBestRunDir -Leaf } else { $null }
$sessionBestRunDir = if ($hasSessionBest) { $InitialSessionBestRunDir } else { $null }
$runsFinished = 0
$runsFailed = 0
$consecutiveFailures = 0

$status = @{
    session_tag = Split-Path $RootLogDir -Leaf
    dataset_name = $DatasetName
    dataset_dir = $DatasetDir
    active_run_id = $null
    active_run_dir = $null
    runs_finished = 0
    runs_failed = 0
    reference_rmse = $ReferenceRmse
    reference_run_dir = $ReferenceRunDir
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

    $lineage = [string](Get-ExpValue -Exp $planExp -Name "lineage" -DefaultValue "legacy_haversine")
    $weighted = [bool](Get-ExpValue -Exp $planExp -Name "weighted_fixed" -DefaultValue $false)
    $missingAware = [bool](Get-ExpValue -Exp $planExp -Name "missing_aware" -DefaultValue $false)
    $runId = [string](Get-ExpValue -Exp $planExp -Name "run_id" -DefaultValue "")
    $topk = [int](Get-ExpValue -Exp $planExp -Name "topk" -DefaultValue 16)
    $seed = [int](Get-ExpValue -Exp $planExp -Name "seed" -DefaultValue 42)
    $lr = [string](Get-ExpValue -Exp $planExp -Name "lr" -DefaultValue "1e-3")
    $weightDecay = [string](Get-ExpValue -Exp $planExp -Name "weight_decay" -DefaultValue "1e-5")
    $precision = [string](Get-ExpValue -Exp $planExp -Name "precision" -DefaultValue "bf16")
    $hiddenDim = [int](Get-ExpValue -Exp $planExp -Name "hidden_dim" -DefaultValue 32)
    $histLen = [int](Get-ExpValue -Exp $planExp -Name "hist_len" -DefaultValue 12)
    $epochs = [int](Get-ExpValue -Exp $planExp -Name "epochs" -DefaultValue 40)
    $batchSize = [int](Get-ExpValue -Exp $planExp -Name "batch_size" -DefaultValue 32)
    $schedulerMonitor = [string](Get-ExpValue -Exp $planExp -Name "scheduler_monitor" -DefaultValue "val_loss")
    $schedulerPatience = [int](Get-ExpValue -Exp $planExp -Name "scheduler_patience" -DefaultValue 8)
    $schedulerCooldown = [int](Get-ExpValue -Exp $planExp -Name "scheduler_cooldown" -DefaultValue 2)
    $schedulerMinLr = [string](Get-ExpValue -Exp $planExp -Name "scheduler_min_lr" -DefaultValue "1e-5")
    $datasetDir = [string](Get-ExpValue -Exp $planExp -Name "dataset_dir" -DefaultValue $DatasetDir)
    $splitPolicy = [string](Get-ExpValue -Exp $planExp -Name "split_policy" -DefaultValue $SplitPolicy)
    $splitAlignment = [string](Get-ExpValue -Exp $planExp -Name "split_alignment" -DefaultValue $SplitAlignment)
    $outlierCleaningMode = [string](Get-ExpValue -Exp $planExp -Name "outlier_cleaning_mode" -DefaultValue "train_quantile_clip")

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
        "--dataset-dir", $datasetDir,
        "--dataset-name", $DatasetName,
        "--log-dir", $runDir,
        "--hist-len", [string]$histLen,
        "--pred-horizon", "12",
        "--report-horizons-minutes", $ReportHorizonsMinutes,
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
        "--num-workers", "0",
        "--val-interval", "1",
        "--log-interval", "1",
        "--split-policy", $splitPolicy,
        "--split-alignment", $splitAlignment,
        "--scheduler-monitor", $schedulerMonitor,
        "--scheduler-factor", "0.5",
        "--scheduler-patience", [string]$schedulerPatience,
        "--scheduler-cooldown", [string]$schedulerCooldown,
        "--scheduler-min-lr", [string]$schedulerMinLr,
        "--early-stop-patience", "0",
        "--min-epochs", "0",
        "--outlier-cleaning-mode", $outlierCleaningMode,
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
            lineage = $lineage
            seed = $seed
            status = "failed"
            exit_code = $exitCode
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
    if ($comparable -and ((-not $hasSessionBest) -or ($testRmse -lt $sessionBestRmse))) {
        $isNewSessionBest = $true
        $hasSessionBest = $true
        $sessionBestRmse = $testRmse
        $sessionBestRunId = $runId
        $sessionBestRunDir = $runDir
    }
    $beatReference = $testRmse -lt $ReferenceRmse

    $line = @(
        $runId,
        $runDir,
        $lineage,
        $datasetDir,
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
        $beatReference
    ) -join "`t"
    Add-Content -Path $summaryPath -Value $line -Encoding utf8

    $historyRecord = @{
        session_tag = $status.session_tag
        run_id = $runId
        launched_at = (Get-Date -Format o)
        finished_at = (Get-Date -Format o)
        run_dir = $runDir
        lineage = $lineage
        dataset_name = $DatasetName
        dataset_dir = $datasetDir
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
        delta_vs_session_best = $(if ($null -eq $sessionBestBefore) { $null } else { $testRmse - $sessionBestBefore })
        is_new_session_best = $isNewSessionBest
        reference_rmse = $ReferenceRmse
        reference_run_dir = $ReferenceRunDir
        delta_vs_reference = ($testRmse - $ReferenceRmse)
        beat_reference = $beatReference
        comparable = $comparable
        fixed_edge_length_feature_mode = [string]$meta.fixed_edge_length_feature_mode
        history_missing_mode = [string]$meta.history_missing_mode
    } | ConvertTo-Json -Compress
    Add-Content -Path $historyPath -Value $historyRecord -Encoding utf8

    if ($isNewSessionBest -or $beatReference) {
        $eventType = if ($beatReference) { "new_reference_beat" } else { "new_session_best" }
        $breakthroughRecord = @{
            timestamp = (Get-Date -Format o)
            event_type = $eventType
            run_id = $runId
            run_dir = $runDir
            lineage = $lineage
            dataset_dir = $datasetDir
            seed = $seed
            test_rmse = $testRmse
            delta_vs_session_best = $(if ($null -eq $sessionBestBefore) { $null } else { $testRmse - $sessionBestBefore })
            delta_vs_reference = ($testRmse - $ReferenceRmse)
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

        if ($beatReference) {
            Write-Banner -Title "NEW REFERENCE BEAT" -Message ($runId + " | test_rmse=" + $testRmse.ToString("F6") + " | delta_vs_ref=" + ($testRmse - $ReferenceRmse).ToString("F6"))
        } else {
            Write-Banner -Title "NEW SESSION BEST" -Message ($runId + " | test_rmse=" + $testRmse.ToString("F6"))
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

Write-Banner -Title "QUEUE FINISHED" -Message ("Session best RMSE=" + $(if ($null -eq $sessionBestRmse) { "n/a" } else { $sessionBestRmse.ToString("F6") }) + " | run=" + [string]$sessionBestRunId)
