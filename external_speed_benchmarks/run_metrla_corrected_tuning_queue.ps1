param(
    [string]$PythonExe = "C:\Users\WMNL\anaconda3\envs\STDR\python.exe",
    [string]$RepoRoot = "D:\Citi\STDR\STDR_STGAT",
    [string]$RootLogDir = "",
    [double]$HistoricalBestRmse = 6.084874189687469,
    [double]$InitialSessionBestRmse = 6.101389884775905,
    [string]$InitialSessionBestRunDir = "D:\Citi\STDR\STDR_STGAT\runs\external_speed\ablation_matrix\METR-LA_weighted_missing_ablation_ep30_20260422_225900\C_both_on"
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

function Set-QueueStatus {
    param(
        [string]$Path,
        [hashtable]$Status
    )
    $Status | ConvertTo-Json -Depth 8 | Set-Content -Path $Path -Encoding utf8
}

if ([string]::IsNullOrWhiteSpace($RootLogDir)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $RootLogDir = Join-Path $RepoRoot ("runs\external_speed\tuning_sessions\METR-LA_corrected_tuning_queue_" + $stamp)
}

$null = New-Item -ItemType Directory -Path $RootLogDir -Force
$summaryPath = Join-Path $RootLogDir "summary.tsv"
$historyPath = Join-Path $RootLogDir "queue_history.jsonl"
$breakthroughPath = Join-Path $RootLogDir "queue_breakthroughs.jsonl"
$statusPath = Join-Path $RootLogDir "queue_status.json"
$planPath = Join-Path $RootLogDir "planned_experiments.tsv"
$activePath = Join-Path $RootLogDir "active_run.txt"
$reviewPath = Join-Path $RootLogDir "review_rounds.md"

"run_id`tlog_dir`tweighted_fixed`tmissing_aware`ttopk`tseed`tlr`tscheduler_patience`tscheduler_cooldown`tbest_val_rmse`tbest_epoch`ttest_rmse`trmse_15m`trmse_30m`trmse_60m`tnew_session_best`tbeat_historical" | Set-Content -Path $summaryPath -Encoding utf8
"run_id`tweighted_fixed`tmissing_aware`ttopk`tseed`tlr`tscheduler_patience`tscheduler_cooldown" | Set-Content -Path $planPath -Encoding utf8
Set-Content -Path $historyPath -Value "" -Encoding utf8
Set-Content -Path $breakthroughPath -Value "" -Encoding utf8

$reviewText = @"
# METR-LA Corrected Tuning Review

Round 1 notes:
- Historical old baseline test RMSE: $HistoricalBestRmse
- Current corrected session best test RMSE: $InitialSessionBestRmse
- A_weighted_only underperformed badly, so weighted-fixed-only is not a main line.
- B_missingaware_only and C_both_on are both competitive; follow-up search focuses on these two branches.
- Round-1 agent consensus:
  - weighted fixed graph is a secondary branch, not the main line
  - first attack should be small LR reductions around the current best corrected topology
  - then use seed/topk diversity to check whether the gain is robust or just seed luck

Round 2+:
- Round-2/3 update:
  - low LR at 7e-4 and 5e-4 underperformed, so this queue returns to lr=1e-3
  - next focus is TopK neighborhood and earlier ReduceLROnPlateau timing
  - after topk/scheduler tests, the next main branch is seed search on corrected B/C with historical strong seeds
"@
Set-Content -Path $reviewPath -Value $reviewText -Encoding utf8

$expectedComparableAnchor = $null
$initialMetricsPath = Join-Path $InitialSessionBestRunDir "predictor_test_metrics.json"
$initialMetaPath = Join-Path $InitialSessionBestRunDir "stgat_meta.json"
if ((Test-Path $initialMetricsPath) -and (Test-Path $initialMetaPath)) {
    $initialMetrics = Get-Content -Path $initialMetricsPath -Raw | ConvertFrom-Json
    $initialMeta = Get-Content -Path $initialMetaPath -Raw | ConvertFrom-Json
    $expectedComparableAnchor = @{
        processed_dataset_fingerprint = [string]$initialMetrics.processed_dataset_fingerprint
        preprocessing_variant_id = [string]$initialMetrics.preprocessing_variant_id
        speed_metric_protocol = [string]$initialMetrics.speed_metric_protocol
        split_policy = [string]$initialMetrics.split_policy
        split_alignment = [string]$initialMetrics.split_alignment
        hist_len = [int]$initialMeta.hist_len
        pred_horizon = [int]$initialMeta.pred_horizon
    }
}

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
    "--epochs", "30",
    "--batch-size", "32",
    "--optimizer", "adam",
    "--weight-decay", "1e-5",
    "--warmup-epochs", "0",
    "--grad-clip-norm", "5.0",
    "--v-loss", "mse",
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

$experiments = @(
    @{ RunId = "B_topk19_seed31_lr1e3_sched8c2"; Weighted = $false; MissingAware = $true; TopK = 19; Seed = 31; Lr = "1e-3"; SchedulerPatience = 8; SchedulerCooldown = 2 },
    @{ RunId = "C_topk19_seed31_lr1e3_sched8c2"; Weighted = $true; MissingAware = $true; TopK = 19; Seed = 31; Lr = "1e-3"; SchedulerPatience = 8; SchedulerCooldown = 2 },
    @{ RunId = "B_topk19_seed43_lr1e3_sched8c2"; Weighted = $false; MissingAware = $true; TopK = 19; Seed = 43; Lr = "1e-3"; SchedulerPatience = 8; SchedulerCooldown = 2 },
    @{ RunId = "C_topk19_seed43_lr1e3_sched8c2"; Weighted = $true; MissingAware = $true; TopK = 19; Seed = 43; Lr = "1e-3"; SchedulerPatience = 8; SchedulerCooldown = 2 },
    @{ RunId = "B_topk19_seed17_lr1e3_sched8c2"; Weighted = $false; MissingAware = $true; TopK = 19; Seed = 17; Lr = "1e-3"; SchedulerPatience = 8; SchedulerCooldown = 2 },
    @{ RunId = "C_topk19_seed17_lr1e3_sched8c2"; Weighted = $true; MissingAware = $true; TopK = 19; Seed = 17; Lr = "1e-3"; SchedulerPatience = 8; SchedulerCooldown = 2 },
    @{ RunId = "B_topk19_seed19_lr1e3_sched8c2"; Weighted = $false; MissingAware = $true; TopK = 19; Seed = 19; Lr = "1e-3"; SchedulerPatience = 8; SchedulerCooldown = 2 },
    @{ RunId = "C_topk19_seed19_lr1e3_sched8c2"; Weighted = $true; MissingAware = $true; TopK = 19; Seed = 19; Lr = "1e-3"; SchedulerPatience = 8; SchedulerCooldown = 2 }
)

foreach ($exp in $experiments) {
    Add-Content -Path $planPath -Value ($exp.RunId + "`t" + $exp.Weighted + "`t" + $exp.MissingAware + "`t" + $exp.TopK + "`t" + $exp.Seed + "`t" + $exp.Lr + "`t" + $exp.SchedulerPatience + "`t" + $exp.SchedulerCooldown) -Encoding utf8
}

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
}
Set-QueueStatus -Path $statusPath -Status $status

$env:PYTHONUNBUFFERED = "1"
Set-Location $RepoRoot

foreach ($exp in $experiments) {
    if ($consecutiveFailures -ge 3) {
        Write-Banner -Title "QUEUE STOPPED" -Message "Three consecutive failures reached. Inspect logs before continuing."
        break
    }

    $runDir = Join-Path $RootLogDir $exp.RunId
    $stdoutPath = Join-Path $runDir "live.stdout.log"
    $null = New-Item -ItemType Directory -Path $runDir -Force

    @(
        "run_id=$($exp.RunId)",
        "run_dir=$runDir",
        "started_at=$(Get-Date -Format o)"
    ) | Set-Content -Path $activePath -Encoding utf8

    $status.active_run_id = $exp.RunId
    $status.active_run_dir = $runDir
    Set-QueueStatus -Path $statusPath -Status $status

    $args = @(
        $baseArgs +
        @(
            "--log-dir", $runDir,
            "--adaptive-topk", [string]$exp.TopK,
            "--seed", [string]$exp.Seed,
            "--lr", [string]$exp.Lr,
            "--scheduler-patience", [string]$exp.SchedulerPatience,
            "--scheduler-cooldown", [string]$exp.SchedulerCooldown
        )
    )
    if (-not $exp.Weighted) {
        $args += "--disable-weighted-fixed-graph"
    }
    if (-not $exp.MissingAware) {
        $args += "--disable-missing-aware-history"
    }

    Write-Banner -Title ("Starting " + $exp.RunId) -Message ("Run dir: " + $runDir)
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
            run_id = $exp.RunId
            launched_at = (Get-Date -Format o)
            finished_at = (Get-Date -Format o)
            run_dir = $runDir
            seed = $exp.Seed
            status = "failed"
            exit_code = $exitCode
        } | ConvertTo-Json -Compress
        Add-Content -Path $historyPath -Value $failureRecord -Encoding utf8
        Write-Warning ("Run failed: " + $exp.RunId + " (exit=" + $exitCode + ")")
        continue
    }

    $consecutiveFailures = 0
    $runsFinished += 1

    $metrics = Get-Content -Path $metricsPath -Raw | ConvertFrom-Json
    $meta = Get-Content -Path $metaPath -Raw | ConvertFrom-Json
    $best = Get-BestEpochSummary -PredictorLogPath $predictorLogPath
    $schedulerVariantId = (
        "plateau_val_loss_factor0p5_pat" +
        [string]$exp.SchedulerPatience +
        "_cd" +
        [string]$exp.SchedulerCooldown +
        "_minlr1e-5_lr" +
        ([string]$exp.Lr).Replace(".", "p").Replace("-", "m")
    )

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
            hist_len = [int]$meta.hist_len
            pred_horizon = [int]$meta.pred_horizon
        }
    }
    $comparable = (
        [string]$metrics.processed_dataset_fingerprint -eq [string]$expectedComparableAnchor.processed_dataset_fingerprint -and
        [string]$metrics.preprocessing_variant_id -eq [string]$expectedComparableAnchor.preprocessing_variant_id -and
        [string]$metrics.speed_metric_protocol -eq [string]$expectedComparableAnchor.speed_metric_protocol -and
        [string]$metrics.split_policy -eq [string]$expectedComparableAnchor.split_policy -and
        [string]$metrics.split_alignment -eq [string]$expectedComparableAnchor.split_alignment -and
        [int]$meta.hist_len -eq [int]$expectedComparableAnchor.hist_len -and
        [int]$meta.pred_horizon -eq [int]$expectedComparableAnchor.pred_horizon
    )

    $isNewSessionBest = $false
    if ($comparable -and $testRmse -lt $sessionBestRmse) {
        $isNewSessionBest = $true
        $sessionBestRmse = $testRmse
        $sessionBestRunId = $exp.RunId
        $sessionBestRunDir = $runDir
    }
    $beatHistorical = $comparable -and ($testRmse -lt $HistoricalBestRmse)

    $line = @(
        $exp.RunId,
        $runDir,
        $exp.Weighted,
        $exp.MissingAware,
        $exp.TopK,
        $exp.Seed,
        $exp.Lr,
        $exp.SchedulerPatience,
        $exp.SchedulerCooldown,
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
        run_id = $exp.RunId
        launched_at = (Get-Date -Format o)
        finished_at = (Get-Date -Format o)
        run_dir = $runDir
        seed = $exp.Seed
        status = "completed"
        exit_code = $exitCode
        configured_epochs = 30
        completed_epochs = [int]$meta.training_control.completed_epochs
        training_end_reason = [string]$meta.training_control.training_end_reason
        hidden_dim = 32
        hist_len = 12
        pred_horizon = 12
        adaptive_topk = $exp.TopK
        fixed_graph_weighted = $exp.Weighted
        use_speed_history_mask = $exp.MissingAware
        scheduler_monitor = "val_loss"
        scheduler_patience = $exp.SchedulerPatience
        scheduler_cooldown = $exp.SchedulerCooldown
        scheduler_min_lr = "1e-5"
        scheduler_variant_id = $schedulerVariantId
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
            run_id = $exp.RunId
            run_dir = $runDir
            seed = $exp.Seed
            test_rmse = $testRmse
            delta_vs_session_best = ($testRmse - $sessionBestBefore)
            delta_vs_historical_best = ($testRmse - $HistoricalBestRmse)
            hidden_dim = 32
            hist_len = 12
            adaptive_topk = $exp.TopK
            fixed_graph_weighted = $exp.Weighted
            use_speed_history_mask = $exp.MissingAware
            scheduler_monitor = "val_loss"
            scheduler_patience = $exp.SchedulerPatience
            scheduler_cooldown = $exp.SchedulerCooldown
            scheduler_min_lr = "1e-5"
            scheduler_variant_id = $schedulerVariantId
        } | ConvertTo-Json -Compress
        Add-Content -Path $breakthroughPath -Value $breakthroughRecord -Encoding utf8

        if ($beatHistorical) {
            Write-Banner -Title "NEW HISTORICAL BEST" -Message ($exp.RunId + " | test_rmse=" + $testRmse.ToString("F6") + " | vs_6.084874=" + ($testRmse - $HistoricalBestRmse).ToString("F6"))
        } else {
            Write-Banner -Title "NEW SESSION BEST" -Message ($exp.RunId + " | test_rmse=" + $testRmse.ToString("F6") + " | delta=" + ($testRmse - $sessionBestBefore).ToString("F6"))
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
