param(
    [string]$PythonExe = "C:\Users\WMNL\anaconda3\envs\STDR\python.exe",
    [string]$RootDir = "D:\Citi\STDR\STDR_STGAT",
    [double]$TargetValRmse = 5.652028831220459,
    [int[]]$Seeds = @(7, 42, 13, 17, 19, 23, 29, 31, 37, 43, 59, 73),
    [int]$Epochs = 40,
    [double]$SchedulerFactor = 0.5,
    [int]$SchedulerPatience = 8,
    [int]$SchedulerCooldown = 2,
    [double]$SchedulerMinLr = 1e-5,
    [int]$EarlyStopPatience = 8,
    [int]$MinEpochs = 18,
    [int]$NumWorkers = 0,
    [bool]$ContinuousSearch = $true,
    [bool]$StopOnBeat = $false,
    [int]$SeedIncrement = 1,
    [string]$SessionTag = (Get-Date -Format "yyyyMMdd_HHmmss")
)

$ErrorActionPreference = "Stop"

function Write-SessionRecord {
    param(
        [string]$Path,
        [hashtable]$Record
    )
    [pscustomobject]$Record |
        ConvertTo-Json -Compress -Depth 8 |
        Add-Content -LiteralPath $Path -Encoding UTF8
}

function Write-RunSummary {
    param(
        [string]$Path,
        [hashtable]$Record
    )
    [pscustomobject]$Record |
        ConvertTo-Json -Depth 8 |
        Set-Content -LiteralPath $Path -Encoding UTF8
}

function Write-SessionStatus {
    param(
        [string]$Path,
        [hashtable]$Record
    )
    [pscustomobject]$Record |
        ConvertTo-Json -Depth 8 |
        Set-Content -LiteralPath $Path -Encoding UTF8
}

function New-BaseRecord {
    param(
        [int]$Seed,
        [string]$RunDir,
        [string]$StdoutPath,
        [string]$StderrPath,
        [string]$SummaryPath,
        [double]$TargetValRmse,
        [int]$Epochs,
        [double]$SchedulerFactor,
        [int]$SchedulerPatience,
        [int]$SchedulerCooldown,
        [double]$SchedulerMinLr,
        [int]$EarlyStopPatience,
        [int]$MinEpochs,
        [int]$NumWorkers,
        [double]$SessionBestValRmse,
        [bool]$ContinuousSearch,
        [bool]$StopOnBeat
    )
    return [ordered]@{
        timestamp = (Get-Date).ToString("s")
        seed = $Seed
        run_dir = $RunDir
        stdout = $StdoutPath
        stderr = $StderrPath
        summary_json = $SummaryPath
        target_val_rmse = $TargetValRmse
        session_best_before = $SessionBestValRmse
        epochs = $Epochs
        num_workers = $NumWorkers
        continuous_search = $ContinuousSearch
        stop_on_beat = $StopOnBeat
        early_stopping = @{
            patience = $EarlyStopPatience
            min_epochs = $MinEpochs
        }
        scheduler = @{
            factor = $SchedulerFactor
            patience = $SchedulerPatience
            cooldown = $SchedulerCooldown
            min_lr = $SchedulerMinLr
        }
    }
}

Set-Location -LiteralPath $RootDir

$entrypoint = Join-Path $RootDir "external_speed_benchmarks\train_sensor_speed.py"
$baseRunDir = Join-Path $RootDir "runs\external_speed"
$sessionTag = $SessionTag
$summaryPath = Join-Path $baseRunDir ("METR-LA_untilbeat_session_" + $sessionTag + ".jsonl")
$statusPath = Join-Path $baseRunDir ("METR-LA_untilbeat_session_" + $sessionTag + ".status.json")

if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Python executable not found: $PythonExe"
}
if (-not (Test-Path -LiteralPath $entrypoint -PathType Leaf)) {
    throw "Training entrypoint not found: $entrypoint"
}

New-Item -ItemType Directory -Path $baseRunDir -Force | Out-Null

$sessionBestValRmse = [double]$TargetValRmse
$sessionBestSeed = $null
$sessionBestRunDir = $null
$completedRuns = 0
$activeSeed = $null

$seedQueue = [System.Collections.Generic.Queue[int]]::new()
$seenSeeds = [System.Collections.Generic.HashSet[int]]::new()
foreach ($seed in $Seeds) {
    if ($seenSeeds.Add([int]$seed)) {
        $seedQueue.Enqueue([int]$seed)
    }
}
$nextSeedCandidate = if ($seenSeeds.Count -gt 0) {
    ([int](($seenSeeds | Measure-Object -Maximum).Maximum)) + [int]$SeedIncrement
} else {
    1
}

Write-Host ""
Write-Host "==> METR-LA continuous best-refresh session"
Write-Host "    historical best baseline = $TargetValRmse"
Write-Host "    initial seeds = $($Seeds -join ', ')"
Write-Host "    epochs per seed = $Epochs"
Write-Host "    num_workers = $NumWorkers"
Write-Host "    early_stop = patience:$EarlyStopPatience min_epochs:$MinEpochs"
Write-Host "    scheduler = factor:$SchedulerFactor patience:$SchedulerPatience cooldown:$SchedulerCooldown min_lr:$SchedulerMinLr"
Write-Host "    continuous_search = $ContinuousSearch"
Write-Host "    stop_on_beat = $StopOnBeat"
Write-Host "    summary = $summaryPath"
Write-Host "    status = $statusPath"

while ($true) {
    if ($seedQueue.Count -eq 0) {
        if (-not $ContinuousSearch) {
            break
        }
        while (-not $seenSeeds.Add([int]$nextSeedCandidate)) {
            $nextSeedCandidate += [int]$SeedIncrement
        }
        $seedQueue.Enqueue([int]$nextSeedCandidate)
        $nextSeedCandidate += [int]$SeedIncrement
    }

    $seed = $seedQueue.Dequeue()
    $activeSeed = $seed
    $runDir = Join-Path $baseRunDir ("METR-LA_untilbeat_seed{0}_{1}" -f $seed, $sessionTag)
    $stdoutPath = Join-Path $runDir "run.stdout.log"
    $stderrPath = Join-Path $runDir "run.stderr.log"
    $metaPath = Join-Path $runDir "stgat_meta.json"
    $metricsPath = Join-Path $runDir "predictor_test_metrics.json"
    $runSummaryPath = Join-Path $runDir "run_summary.json"
    $process = $null

    New-Item -ItemType Directory -Path $runDir -Force | Out-Null
    Write-SessionStatus -Path $statusPath -Record ([ordered]@{
        updated_at = (Get-Date).ToString("s")
        session_tag = $sessionTag
        active_seed = $activeSeed
        completed_runs = $completedRuns
        historical_best_baseline = $TargetValRmse
        session_best_val_rmse = $sessionBestValRmse
        session_best_seed = $sessionBestSeed
        session_best_run_dir = $sessionBestRunDir
        continuous_search = $ContinuousSearch
        stop_on_beat = $StopOnBeat
        summary_path = $summaryPath
    })

    $record = New-BaseRecord `
        -Seed $seed `
        -RunDir $runDir `
        -StdoutPath $stdoutPath `
        -StderrPath $stderrPath `
        -SummaryPath $runSummaryPath `
        -TargetValRmse $TargetValRmse `
        -Epochs $Epochs `
        -SchedulerFactor $SchedulerFactor `
        -SchedulerPatience $SchedulerPatience `
        -SchedulerCooldown $SchedulerCooldown `
        -SchedulerMinLr $SchedulerMinLr `
        -EarlyStopPatience $EarlyStopPatience `
        -MinEpochs $MinEpochs `
        -NumWorkers $NumWorkers `
        -SessionBestValRmse $sessionBestValRmse `
        -ContinuousSearch $ContinuousSearch `
        -StopOnBeat $StopOnBeat

    try {
        $args = @(
            "-u",
            "external_speed_benchmarks/train_sensor_speed.py",
            "--dataset-dir", "data\external_datasets\processed\METR-LA",
            "--dataset-name", "METR-LA",
            "--log-dir", $runDir,
            "--fresh",
            "--epochs", "$Epochs",
            "--batch-size", "32",
            "--optimizer", "adam",
            "--lr", "0.001",
            "--weight-decay", "1e-5",
            "--warmup-epochs", "0",
            "--grad-clip-norm", "5.0",
            "--v-loss", "mse",
            "--scheduler-monitor", "val_rmse",
            "--scheduler-factor", "$SchedulerFactor",
            "--scheduler-patience", "$SchedulerPatience",
            "--scheduler-cooldown", "$SchedulerCooldown",
            "--scheduler-min-lr", "$SchedulerMinLr",
            "--split-policy", "benchmark_contiguous",
            "--split-alignment", "none",
            "--outlier-cleaning-mode", "train_quantile_clip",
            "--log-interval", "1",
            "--val-interval", "1",
            "--adaptive-topk", "16",
            "--hidden-dim", "32",
            "--precision", "bf16",
            "--num-workers", "$NumWorkers",
            "--seed", "$seed",
            "--early-stop-patience", "$EarlyStopPatience",
            "--min-epochs", "$MinEpochs"
        )

        Write-Host ""
        Write-Host "==> Running seed $seed"
        Write-Host "    run_dir = $runDir"
        Write-Host "    current session best = $sessionBestValRmse"

        $process = Start-Process `
            -FilePath $PythonExe `
            -ArgumentList $args `
            -WorkingDirectory $RootDir `
            -RedirectStandardOutput $stdoutPath `
            -RedirectStandardError $stderrPath `
            -PassThru `
            -Wait

        $record.exit_code = $process.ExitCode

        if ($process.ExitCode -ne 0) {
            $record.status = "nonzero_exit"
            Write-SessionRecord -Path $summaryPath -Record $record
            Write-RunSummary -Path $runSummaryPath -Record $record
            Write-Warning "Seed $seed exited with code $($process.ExitCode). Continuing to next seed."
            continue
        }

        if (-not (Test-Path -LiteralPath $metaPath -PathType Leaf)) {
            $record.status = "missing_meta"
            Write-SessionRecord -Path $summaryPath -Record $record
            Write-RunSummary -Path $runSummaryPath -Record $record
            Write-Warning "Seed $seed finished without stgat_meta.json. Continuing to next seed."
            continue
        }
        if (-not (Test-Path -LiteralPath $metricsPath -PathType Leaf)) {
            $record.status = "missing_metrics"
            Write-SessionRecord -Path $summaryPath -Record $record
            Write-RunSummary -Path $runSummaryPath -Record $record
            Write-Warning "Seed $seed finished without predictor_test_metrics.json. Continuing to next seed."
            continue
        }

        $meta = Get-Content -LiteralPath $metaPath -Raw | ConvertFrom-Json
        $metrics = Get-Content -LiteralPath $metricsPath -Raw | ConvertFrom-Json
        $selected = $meta.selected_checkpoint

        if ($null -eq $selected) {
            $record.status = "invalid_meta_missing_selected_checkpoint"
            Write-SessionRecord -Path $summaryPath -Record $record
            Write-RunSummary -Path $runSummaryPath -Record $record
            Write-Warning "Seed $seed meta is missing selected_checkpoint. Continuing to next seed."
            continue
        }

        $bestVal = $selected.best_val_raw_speed_rmse -as [double]
        $bestEpoch = $selected.best_epoch -as [int]
        $selectedCheckpointPath = [string]$selected.path

        if (
            ($null -eq $bestVal) -or
            ($bestEpoch -le 0) -or
            ([string]::IsNullOrWhiteSpace($selectedCheckpointPath)) -or
            (-not (Test-Path -LiteralPath $selectedCheckpointPath -PathType Leaf))
        ) {
            $record.status = "artifact_invalid"
            $record.selected_checkpoint = $selectedCheckpointPath
            $record.best_val_rmse = $bestVal
            $record.best_epoch = $bestEpoch
            Write-SessionRecord -Path $summaryPath -Record $record
            Write-RunSummary -Path $runSummaryPath -Record $record
            Write-Warning "Seed $seed produced invalid best-checkpoint metadata. Continuing to next seed."
            continue
        }

        $report15 = $metrics.val_raw_metrics_report.speed.'15min'.rmse
        $report30 = $metrics.val_raw_metrics_report.speed.'30min'.rmse
        $report60 = $metrics.val_raw_metrics_report.speed.'60min'.rmse
        $beatHistorical = $bestVal -lt $TargetValRmse
        $isNewSessionBest = $bestVal -lt $sessionBestValRmse

        if ($isNewSessionBest) {
            $sessionBestValRmse = [double]$bestVal
            $sessionBestSeed = [int]$seed
            $sessionBestRunDir = $runDir
        }

        $record.status = $(if ($beatHistorical) { "beat_historical" } else { "completed" })
        $record.best_val_rmse = [double]$bestVal
        $record.best_epoch = [int]$bestEpoch
        $record.val_15min = $report15
        $record.val_30min = $report30
        $record.val_60min = $report60
        $record.beat_target = [bool]$beatHistorical
        $record.beat_historical = [bool]$beatHistorical
        $record.is_new_session_best = [bool]$isNewSessionBest
        $record.session_best_after = [double]$sessionBestValRmse
        $record.session_best_seed = $sessionBestSeed
        $record.selected_checkpoint = $selectedCheckpointPath
        $record.training_end_reason = [string]$meta.training_control.training_end_reason
        $record.seed_recorded_in_meta = $meta.seed

        Write-SessionRecord -Path $summaryPath -Record $record
        Write-RunSummary -Path $runSummaryPath -Record $record

        $completedRuns += 1
        $activeSeed = $null
        Write-SessionStatus -Path $statusPath -Record ([ordered]@{
            updated_at = (Get-Date).ToString("s")
            session_tag = $sessionTag
            active_seed = $activeSeed
            completed_runs = $completedRuns
            historical_best_baseline = $TargetValRmse
            session_best_val_rmse = $sessionBestValRmse
            session_best_seed = $sessionBestSeed
            session_best_run_dir = $sessionBestRunDir
            last_completed_seed = $seed
            continuous_search = $ContinuousSearch
            stop_on_beat = $StopOnBeat
            summary_path = $summaryPath
        })

        Write-Host "    best_val_rmse = $bestVal @ epoch $bestEpoch"
        Write-Host "    best-val 15/30/60 = $report15 / $report30 / $report60"
        if ($isNewSessionBest) {
            Write-Host "    NEW SESSION BEST: seed $seed => $bestVal" -ForegroundColor Green
        }
        if ($beatHistorical) {
            Write-Host "    Historical best beaten by seed $seed." -ForegroundColor Green
            if ($StopOnBeat) {
                Write-Host "    stop_on_beat=true, ending session." -ForegroundColor Yellow
                break
            }
            Write-Host "    Continuing search because stop_on_beat=false." -ForegroundColor Yellow
        }
    }
    catch {
        $record.status = "exception"
        $record.error_message = $_.Exception.Message
        if ($process -ne $null) {
            $record.exit_code = $process.ExitCode
        }
        Write-SessionRecord -Path $summaryPath -Record $record
        Write-RunSummary -Path $runSummaryPath -Record $record
        Write-Warning "Seed $seed hit an exception: $($_.Exception.Message)"
        continue
    }
}

Write-Host ""
Write-Host "Session finished. Summary: $summaryPath"
