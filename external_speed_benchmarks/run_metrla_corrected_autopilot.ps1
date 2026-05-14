param(
    [string]$PythonExe = "C:\Users\WMNL\anaconda3\envs\STDR\python.exe",
    [string]$RepoRoot = "D:\Citi\STDR\STDR_STGAT",
    [string]$CampaignRoot = "",
    [double]$HistoricalBestRmse = 6.084874189687469,
    [double]$InitialSessionBestRmse = 6.101389884775905,
    [string]$InitialSessionBestRunDir = "D:\Citi\STDR\STDR_STGAT\runs\external_speed\ablation_matrix\METR-LA_weighted_missing_ablation_ep30_20260422_225900\C_both_on",
    [string]$WatchSessionRoot = "D:\Citi\STDR\STDR_STGAT\runs\external_speed\tuning_sessions\METR-LA_corrected_tuning_queue_round3_20260423_233708",
    [int]$PollSeconds = 20,
    [int]$MaxAutoBatches = 12
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

function Write-JsonFile {
    param(
        [string]$Path,
        [object]$Value
    )
    $Value | ConvertTo-Json -Depth 10 | Set-Content -Path $Path -Encoding utf8
}

function Get-LineCountWithoutHeader {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        return 0
    }
    $lines = Get-Content $Path
    if ($lines.Count -le 1) {
        return 0
    }
    return ($lines.Count - 1)
}

function Get-ActiveRunPythonProcess {
    param([string]$RunDir)
    if ([string]::IsNullOrWhiteSpace($RunDir)) {
        return $null
    }
    $escaped = [regex]::Escape($RunDir)
    return Get-CimInstance Win32_Process |
        Where-Object {
            $_.Name -eq "python.exe" -and
            $_.CommandLine -match "train_sensor_speed.py" -and
            $_.CommandLine -match $escaped
        } |
        Select-Object -First 1
}

function Wait-ForSessionCompletion {
    param(
        [string]$SessionRoot,
        [int]$PollIntervalSeconds
    )
    if (-not (Test-Path $SessionRoot)) {
        return
    }
    $statusPath = Join-Path $SessionRoot "queue_status.json"
    $planPath = Join-Path $SessionRoot "planned_experiments.tsv"
    $summaryPath = Join-Path $SessionRoot "summary.tsv"
    if (-not (Test-Path $statusPath)) {
        return
    }

    while ($true) {
        $status = Get-Content -Path $statusPath -Raw | ConvertFrom-Json
        $plannedCount = Get-LineCountWithoutHeader -Path $planPath
        $finishedCount = Get-LineCountWithoutHeader -Path $summaryPath
        $activeRunDir = [string]$status.active_run_dir
        $activeRunId = [string]$status.active_run_id
        $pyProc = Get-ActiveRunPythonProcess -RunDir $activeRunDir

        if ($plannedCount -gt 0 -and $finishedCount -ge $plannedCount -and [string]::IsNullOrWhiteSpace($activeRunId)) {
            break
        }

        if (-not [string]::IsNullOrWhiteSpace($activeRunDir) -and $null -ne $pyProc) {
            $liveLog = Join-Path $activeRunDir "live.stdout.log"
            $tail = ""
            if (Test-Path $liveLog) {
                $tail = (Get-Content -Path $liveLog -Tail 1)
            }
            $msg = if ([string]::IsNullOrWhiteSpace($tail)) { $activeRunId } else { $activeRunId + " | " + $tail }
            Write-Banner -Title "AUTOPILOT WATCH" -Message $msg
            Start-Sleep -Seconds $PollIntervalSeconds
            continue
        }

        if ([string]::IsNullOrWhiteSpace($activeRunId) -and $finishedCount -lt $plannedCount) {
            $statusAge = ((Get-Date) - (Get-Item $statusPath).LastWriteTime).TotalMinutes
            if ($statusAge -ge 10) {
                Write-Warning ("Watch session appears stale: " + $SessionRoot)
                break
            }
        }

        Start-Sleep -Seconds $PollIntervalSeconds
    }
}

function Get-SeededBaselineRuns {
    return @(
        [pscustomobject]@{
            session_tag = "seeded_baseline"
            run_id = "baseline_B_missingaware_only"
            run_dir = "D:\Citi\STDR\STDR_STGAT\runs\external_speed\ablation_matrix\METR-LA_weighted_missing_ablation_ep30_20260422_225900\B_missingaware_only"
            branch = "B"
            seed = 76
            status = "completed"
            adaptive_topk = 19
            lr = 0.001
            weight_decay = 1e-5
            scheduler_monitor = "val_loss"
            scheduler_patience = 8
            scheduler_cooldown = 2
            precision = "bf16"
            hidden_dim = 32
            hist_len = 12
            fixed_graph_weighted = $false
            use_speed_history_mask = $true
            best_val_rmse = 5.6190
            best_epoch = 16
            test_rmse = 6.1034
            rmse15 = 5.2069
            rmse30 = 6.1556
            rmse60 = 7.1743
            comparable = $true
        }
        [pscustomobject]@{
            session_tag = "seeded_baseline"
            run_id = "baseline_C_both_on"
            run_dir = $InitialSessionBestRunDir
            branch = "C"
            seed = 76
            status = "completed"
            adaptive_topk = 19
            lr = 0.001
            weight_decay = 1e-5
            scheduler_monitor = "val_loss"
            scheduler_patience = 8
            scheduler_cooldown = 2
            precision = "bf16"
            hidden_dim = 32
            hist_len = 12
            fixed_graph_weighted = $true
            use_speed_history_mask = $true
            best_val_rmse = 5.600035660027046
            best_epoch = 16
            test_rmse = 6.101389884775905
            rmse15 = 5.217573308933963
            rmse30 = 6.16852790151769
            rmse60 = 7.143384908680595
            comparable = $true
        }
    )
}

function Get-ComparableRuns {
    param([string]$RepoRootPath)
    $records = New-Object System.Collections.Generic.List[object]
    foreach ($baseline in (Get-SeededBaselineRuns)) {
        $records.Add($baseline)
    }
    $historyFiles = Get-ChildItem -Path (Join-Path $RepoRootPath "runs\external_speed\tuning_sessions") -Filter "queue_history.jsonl" -Recurse -ErrorAction SilentlyContinue
    foreach ($historyFile in $historyFiles) {
        foreach ($line in Get-Content $historyFile.FullName) {
            if ([string]::IsNullOrWhiteSpace($line)) {
                continue
            }
            $item = $line | ConvertFrom-Json
            if ([string]$item.status -ne "completed") {
                continue
            }
            if ($null -eq $item.comparable -or -not [bool]$item.comparable) {
                continue
            }
            if ($null -eq $item.branch) {
                $item | Add-Member -NotePropertyName branch -NotePropertyValue $(if ([bool]$item.fixed_graph_weighted) { "C" } else { "B" }) -Force
            }
            if ($null -eq $item.precision) {
                $item | Add-Member -NotePropertyName precision -NotePropertyValue "bf16" -Force
            }
            if ($null -eq $item.weight_decay) {
                $item | Add-Member -NotePropertyName weight_decay -NotePropertyValue 1e-5 -Force
            }
            $records.Add($item)
        }
    }
    return $records.ToArray()
}

function Get-ConfigSignature {
    param(
        [string]$Branch,
        [int]$TopK,
        [int]$Seed,
        [double]$Lr,
        [double]$WeightDecay,
        [string]$Precision,
        [int]$HiddenDim,
        [int]$HistLen,
        [int]$SchedulerPatience,
        [int]$SchedulerCooldown
    )
    return ($Branch + "|" + $TopK + "|" + $Seed + "|" + $Lr.ToString("G17") + "|" + $WeightDecay.ToString("G17") + "|" + $Precision + "|" + $HiddenDim + "|" + $HistLen + "|" + $SchedulerPatience + "|" + $SchedulerCooldown)
}

function Get-SeenSignatureSet {
    param([object[]]$Runs)
    $set = New-Object 'System.Collections.Generic.HashSet[string]'
    foreach ($run in $Runs) {
        $sig = Get-ConfigSignature `
            -Branch ([string]$run.branch) `
            -TopK ([int]$run.adaptive_topk) `
            -Seed ([int]$run.seed) `
            -Lr ([double]$run.lr) `
            -WeightDecay ([double]$run.weight_decay) `
            -Precision ([string]$run.precision) `
            -HiddenDim ([int]$run.hidden_dim) `
            -HistLen ([int]$run.hist_len) `
            -SchedulerPatience ([int]$run.scheduler_patience) `
            -SchedulerCooldown ([int]$run.scheduler_cooldown)
        $null = $set.Add($sig)
    }
    return $set
}

function Get-BranchRanking {
    param([object[]]$Runs)
    $branchGroups = $Runs | Group-Object branch
    $ranked = foreach ($group in $branchGroups) {
        $best = $group.Group | Sort-Object test_rmse | Select-Object -First 1
        [pscustomobject]@{
            branch = [string]$group.Name
            best_test_rmse = [double]$best.test_rmse
            best_seed = [int]$best.seed
            best_topk = [int]$best.adaptive_topk
            best_patience = [int]$best.scheduler_patience
            best_cooldown = [int]$best.scheduler_cooldown
            best_precision = [string]$best.precision
            best_weight_decay = [double]$best.weight_decay
            best_hist_len = [int]$best.hist_len
            best_hidden_dim = [int]$best.hidden_dim
        }
    }
    return @($ranked | Sort-Object best_test_rmse)
}

function Get-TopSeedsForBranch {
    param(
        [object[]]$Runs,
        [string]$Branch,
        [int]$Limit = 3
    )
    $seedRanks = $Runs |
        Where-Object { $_.branch -eq $Branch } |
        Group-Object seed |
        ForEach-Object {
            $best = $_.Group | Sort-Object test_rmse | Select-Object -First 1
            [pscustomobject]@{
                seed = [int]$_.Name
                test_rmse = [double]$best.test_rmse
            }
        } |
        Sort-Object test_rmse, seed
    return @($seedRanks | Select-Object -First $Limit | ForEach-Object { [int]$_.seed })
}

function New-ExperimentObject {
    param(
        [string]$Branch,
        [int]$TopK,
        [int]$Seed,
        [double]$Lr,
        [double]$WeightDecay,
        [string]$Precision,
        [int]$HiddenDim,
        [int]$HistLen,
        [int]$SchedulerPatience,
        [int]$SchedulerCooldown,
        [string]$RunIdPrefix
    )
    $weighted = $Branch -eq "C"
    $missingAware = $true
    $lrLabel = ([string]$Lr).Replace(".", "p").Replace("-", "m")
    $wdLabel = ([string]$WeightDecay).Replace(".", "p").Replace("-", "m")
    $runId = "{0}_topk{1}_seed{2}_lr{3}_wd{4}_{5}_hist{6}_sched{7}c{8}" -f $RunIdPrefix, $TopK, $Seed, $lrLabel, $wdLabel, $Precision, $HistLen, $SchedulerPatience, $SchedulerCooldown
    return [ordered]@{
        run_id = $runId
        branch = $Branch
        weighted_fixed = $weighted
        missing_aware = $missingAware
        topk = $TopK
        seed = $Seed
        lr = [string]$Lr
        weight_decay = [string]$WeightDecay
        precision = $Precision
        hidden_dim = $HiddenDim
        hist_len = $HistLen
        epochs = 30
        batch_size = 32
        scheduler_monitor = "val_loss"
        scheduler_patience = $SchedulerPatience
        scheduler_cooldown = $SchedulerCooldown
        scheduler_min_lr = "1e-5"
    }
}

function Add-PlanIfNew {
    param(
        [System.Collections.Generic.List[object]]$Plan,
        [System.Collections.Generic.HashSet[string]]$SeenSignatures,
        [hashtable]$Exp
    )
    $sig = Get-ConfigSignature `
        -Branch ([string]$Exp.branch) `
        -TopK ([int]$Exp.topk) `
        -Seed ([int]$Exp.seed) `
        -Lr ([double]$Exp.lr) `
        -WeightDecay ([double]$Exp.weight_decay) `
        -Precision ([string]$Exp.precision) `
        -HiddenDim ([int]$Exp.hidden_dim) `
        -HistLen ([int]$Exp.hist_len) `
        -SchedulerPatience ([int]$Exp.scheduler_patience) `
        -SchedulerCooldown ([int]$Exp.scheduler_cooldown)
    if ($SeenSignatures.Contains($sig)) {
        return
    }
    $null = $SeenSignatures.Add($sig)
    $Plan.Add([pscustomobject]$Exp)
}

function Build-StagePlan {
    param(
        [string]$StageName,
        [object[]]$Runs,
        [System.Collections.Generic.HashSet[string]]$SeenSignatures
    )
    $branchRanking = Get-BranchRanking -Runs $Runs
    if ($branchRanking.Count -eq 0) {
        throw "No comparable runs available for stage planning."
    }
    $bestBranch = $branchRanking[0]
    $shadowBranch = if ($branchRanking.Count -ge 2) { $branchRanking[1] } else { $null }
    $topSeeds = Get-TopSeedsForBranch -Runs $Runs -Branch $bestBranch.branch -Limit 3
    if ($topSeeds.Count -eq 0) {
        $topSeeds = @(76)
    }

    $plan = New-Object System.Collections.Generic.List[object]

    switch ($StageName) {
        "seed_expand_primary" {
            $seedPool = @(23, 29, 37, 41, 47, 53, 59, 61, 67, 71, 73, 79)
            foreach ($seed in $seedPool | Select-Object -First 6) {
                $exp = New-ExperimentObject `
                    -Branch $bestBranch.branch `
                    -TopK ([int]$bestBranch.best_topk) `
                    -Seed $seed `
                    -Lr 0.001 `
                    -WeightDecay ([double]$bestBranch.best_weight_decay) `
                    -Precision ([string]$bestBranch.best_precision) `
                    -HiddenDim ([int]$bestBranch.best_hidden_dim) `
                    -HistLen ([int]$bestBranch.best_hist_len) `
                    -SchedulerPatience ([int]$bestBranch.best_patience) `
                    -SchedulerCooldown ([int]$bestBranch.best_cooldown) `
                    -RunIdPrefix ("auto_seed_primary_" + $bestBranch.branch)
                Add-PlanIfNew -Plan $plan -SeenSignatures $SeenSignatures -Exp $exp
            }
        }
        "seed_expand_shadow" {
            if ($null -ne $shadowBranch -and (($shadowBranch.best_test_rmse - $bestBranch.best_test_rmse) -le 0.015)) {
                $seedPool = @(23, 29, 37, 41)
                foreach ($seed in $seedPool) {
                    $exp = New-ExperimentObject `
                        -Branch $shadowBranch.branch `
                        -TopK ([int]$shadowBranch.best_topk) `
                        -Seed $seed `
                        -Lr 0.001 `
                        -WeightDecay ([double]$shadowBranch.best_weight_decay) `
                        -Precision ([string]$shadowBranch.best_precision) `
                        -HiddenDim ([int]$shadowBranch.best_hidden_dim) `
                        -HistLen ([int]$shadowBranch.best_hist_len) `
                        -SchedulerPatience ([int]$shadowBranch.best_patience) `
                        -SchedulerCooldown ([int]$shadowBranch.best_cooldown) `
                        -RunIdPrefix ("auto_seed_shadow_" + $shadowBranch.branch)
                    Add-PlanIfNew -Plan $plan -SeenSignatures $SeenSignatures -Exp $exp
                }
            }
        }
        "scheduler_local" {
            $seedCandidates = $topSeeds | Select-Object -First 2
            foreach ($seed in $seedCandidates) {
                foreach ($patience in @(4, 5, 6)) {
                    $exp = New-ExperimentObject `
                        -Branch $bestBranch.branch `
                        -TopK ([int]$bestBranch.best_topk) `
                        -Seed $seed `
                        -Lr 0.001 `
                        -WeightDecay ([double]$bestBranch.best_weight_decay) `
                        -Precision ([string]$bestBranch.best_precision) `
                        -HiddenDim ([int]$bestBranch.best_hidden_dim) `
                        -HistLen ([int]$bestBranch.best_hist_len) `
                        -SchedulerPatience $patience `
                        -SchedulerCooldown 0 `
                        -RunIdPrefix ("auto_sched_" + $bestBranch.branch)
                    Add-PlanIfNew -Plan $plan -SeenSignatures $SeenSignatures -Exp $exp
                }
            }
        }
        "weight_decay_local" {
            $bestSeed = $topSeeds[0]
            foreach ($wd in @(0.0, 3e-6, 3e-5)) {
                $exp = New-ExperimentObject `
                    -Branch $bestBranch.branch `
                    -TopK ([int]$bestBranch.best_topk) `
                    -Seed $bestSeed `
                    -Lr 0.001 `
                    -WeightDecay $wd `
                    -Precision ([string]$bestBranch.best_precision) `
                    -HiddenDim ([int]$bestBranch.best_hidden_dim) `
                    -HistLen ([int]$bestBranch.best_hist_len) `
                    -SchedulerPatience ([int]$bestBranch.best_patience) `
                    -SchedulerCooldown ([int]$bestBranch.best_cooldown) `
                    -RunIdPrefix ("auto_wd_" + $bestBranch.branch)
                Add-PlanIfNew -Plan $plan -SeenSignatures $SeenSignatures -Exp $exp
            }
        }
        "fp32_confirm" {
            foreach ($seed in ($topSeeds | Select-Object -First 2)) {
                $exp = New-ExperimentObject `
                    -Branch $bestBranch.branch `
                    -TopK ([int]$bestBranch.best_topk) `
                    -Seed $seed `
                    -Lr 0.001 `
                    -WeightDecay ([double]$bestBranch.best_weight_decay) `
                    -Precision "fp32" `
                    -HiddenDim ([int]$bestBranch.best_hidden_dim) `
                    -HistLen ([int]$bestBranch.best_hist_len) `
                    -SchedulerPatience ([int]$bestBranch.best_patience) `
                    -SchedulerCooldown ([int]$bestBranch.best_cooldown) `
                    -RunIdPrefix ("auto_fp32_" + $bestBranch.branch)
                Add-PlanIfNew -Plan $plan -SeenSignatures $SeenSignatures -Exp $exp
            }
        }
        "topk_local" {
            $bestSeed = $topSeeds[0]
            foreach ($topk in @(18, 20)) {
                $exp = New-ExperimentObject `
                    -Branch $bestBranch.branch `
                    -TopK $topk `
                    -Seed $bestSeed `
                    -Lr 0.001 `
                    -WeightDecay ([double]$bestBranch.best_weight_decay) `
                    -Precision ([string]$bestBranch.best_precision) `
                    -HiddenDim ([int]$bestBranch.best_hidden_dim) `
                    -HistLen ([int]$bestBranch.best_hist_len) `
                    -SchedulerPatience ([int]$bestBranch.best_patience) `
                    -SchedulerCooldown ([int]$bestBranch.best_cooldown) `
                    -RunIdPrefix ("auto_topk_" + $bestBranch.branch)
                Add-PlanIfNew -Plan $plan -SeenSignatures $SeenSignatures -Exp $exp
            }
        }
        "histlen_local" {
            $bestSeed = $topSeeds[0]
            foreach ($histLen in @(10, 14)) {
                $exp = New-ExperimentObject `
                    -Branch $bestBranch.branch `
                    -TopK ([int]$bestBranch.best_topk) `
                    -Seed $bestSeed `
                    -Lr 0.001 `
                    -WeightDecay ([double]$bestBranch.best_weight_decay) `
                    -Precision ([string]$bestBranch.best_precision) `
                    -HiddenDim ([int]$bestBranch.best_hidden_dim) `
                    -HistLen $histLen `
                    -SchedulerPatience ([int]$bestBranch.best_patience) `
                    -SchedulerCooldown ([int]$bestBranch.best_cooldown) `
                    -RunIdPrefix ("auto_hist_" + $bestBranch.branch)
                Add-PlanIfNew -Plan $plan -SeenSignatures $SeenSignatures -Exp $exp
            }
        }
        default {
            throw "Unknown stage: $StageName"
        }
    }

    return $plan.ToArray()
}

if ([string]::IsNullOrWhiteSpace($CampaignRoot)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $CampaignRoot = Join-Path $RepoRoot ("runs\external_speed\autopilot_sessions\METR-LA_corrected_autopilot_" + $stamp)
}

$null = New-Item -ItemType Directory -Path $CampaignRoot -Force
$planDir = Join-Path $CampaignRoot "plans"
$batchIndexPath = Join-Path $CampaignRoot "batch_index.tsv"
$campaignStatusPath = Join-Path $CampaignRoot "campaign_status.json"
$campaignBreakthroughPath = Join-Path $CampaignRoot "campaign_breakthroughs.jsonl"
$campaignReviewPath = Join-Path $CampaignRoot "review_rounds.md"
$autopilotLogPath = Join-Path $CampaignRoot "autopilot.log"
$null = New-Item -ItemType Directory -Path $planDir -Force

"batch_index`tstage_name`tsession_root`tsession_best_test_rmse`tsession_best_run_id" | Set-Content -Path $batchIndexPath -Encoding utf8
Set-Content -Path $campaignBreakthroughPath -Value "" -Encoding utf8

$reviewText = @"
# METR-LA Corrected Autopilot Review

Round 1 multi-agent notes:
- Feynman: prioritize fresh-seed expansion around the strongest corrected family, then do narrow weight-decay refinement.
- Harvey: keep B/C both alive only if they stay very close; otherwise concentrate budget on the stronger branch.
- Ohm: after seed search, test a tighter early scheduler around the strongest seed before opening broad new axes.
- Leibniz: keep local topk and small history-window probes as later-stage refinements.

Autopilot policy:
1. Watch the already-running round3 queue to completion.
2. Rank corrected comparable runs by test RMSE.
3. Automatically try the next batch from this stage order:
   - primary seed expansion
   - shadow seed expansion
   - local scheduler refinement
   - local weight-decay refinement
   - fp32 confirmation
   - local topk refinement
   - local history-window refinement
4. After that, continue extra seed-expansion batches until a better result appears or the batch budget is exhausted.
"@
Set-Content -Path $campaignReviewPath -Value $reviewText -Encoding utf8

$campaignStatus = @{
    started_at = (Get-Date -Format o)
    campaign_root = $CampaignRoot
    current_best_rmse = $InitialSessionBestRmse
    current_best_run_dir = $InitialSessionBestRunDir
    current_best_run_id = "seed76_corrected_ablation_C_both_on"
    historical_best_ref = $HistoricalBestRmse
    watch_session_root = $WatchSessionRoot
    active_stage = $null
    completed_batches = 0
}
Write-JsonFile -Path $campaignStatusPath -Value $campaignStatus

Write-Banner -Title "AUTOPILOT STARTED" -Message ("Campaign root: " + $CampaignRoot)

if (-not [string]::IsNullOrWhiteSpace($WatchSessionRoot)) {
    Write-Banner -Title "WATCHING CURRENT SESSION" -Message $WatchSessionRoot
    Wait-ForSessionCompletion -SessionRoot $WatchSessionRoot -PollIntervalSeconds $PollSeconds
}

$startupRuns = Get-ComparableRuns -RepoRootPath $RepoRoot
if ($startupRuns.Count -gt 0) {
    $startupBest = $startupRuns | Sort-Object test_rmse | Select-Object -First 1
    $campaignStatus.current_best_rmse = [double]$startupBest.test_rmse
    $campaignStatus.current_best_run_id = [string]$startupBest.run_id
    $campaignStatus.current_best_run_dir = [string]$startupBest.run_dir
    Write-JsonFile -Path $campaignStatusPath -Value $campaignStatus
}

$stageOrder = @(
    "seed_expand_primary",
    "seed_expand_shadow",
    "scheduler_local",
    "weight_decay_local",
    "fp32_confirm",
    "topk_local",
    "histlen_local"
)

$batchCounter = 0
foreach ($stageName in $stageOrder) {
    if ($batchCounter -ge $MaxAutoBatches) {
        break
    }

    $runs = Get-ComparableRuns -RepoRootPath $RepoRoot
    $seen = Get-SeenSignatureSet -Runs $runs
    $plan = Build-StagePlan -StageName $stageName -Runs $runs -SeenSignatures $seen
    if ($plan.Count -eq 0) {
        Add-Content -Path $autopilotLogPath -Value ("[" + (Get-Date -Format o) + "] skip stage " + $stageName + " because all candidate configs already exist.") -Encoding utf8
        continue
    }

    $batchCounter += 1
    $batchTag = "{0:D2}_{1}" -f $batchCounter, $stageName
    $planPath = Join-Path $planDir ($batchTag + ".json")
    $sessionRoot = Join-Path $CampaignRoot ("batch_" + $batchTag)
    Write-JsonFile -Path $planPath -Value $plan

    $campaignStatus.active_stage = $stageName
    Write-JsonFile -Path $campaignStatusPath -Value $campaignStatus

    Write-Banner -Title ("AUTOPILOT BATCH " + $batchTag) -Message ("Launching " + $plan.Count + " runs")
    & powershell -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "external_speed_benchmarks\run_metrla_corrected_batch.ps1") `
        -PythonExe $PythonExe `
        -RepoRoot $RepoRoot `
        -RootLogDir $sessionRoot `
        -PlanPath $planPath `
        -HistoricalBestRmse $HistoricalBestRmse `
        -InitialSessionBestRmse $campaignStatus.current_best_rmse `
        -InitialSessionBestRunDir $campaignStatus.current_best_run_dir `
        -ReviewNotes ("Autopilot stage: " + $stageName)

    $sessionStatusPath = Join-Path $sessionRoot "queue_status.json"
    $sessionBreakthroughPath = Join-Path $sessionRoot "queue_breakthroughs.jsonl"
    $sessionStatus = Get-Content -Path $sessionStatusPath -Raw | ConvertFrom-Json
    Add-Content -Path $batchIndexPath -Value ($batchCounter.ToString() + "`t" + $stageName + "`t" + $sessionRoot + "`t" + [string]$sessionStatus.session_best_test_rmse + "`t" + [string]$sessionStatus.session_best_run_id) -Encoding utf8

    if (Test-Path $sessionBreakthroughPath) {
        foreach ($line in Get-Content $sessionBreakthroughPath) {
            if ([string]::IsNullOrWhiteSpace($line)) {
                continue
            }
            Add-Content -Path $campaignBreakthroughPath -Value $line -Encoding utf8
        }
    }

    $campaignStatus.completed_batches = $batchCounter
    if ([double]$sessionStatus.session_best_test_rmse -lt [double]$campaignStatus.current_best_rmse) {
        $campaignStatus.current_best_rmse = [double]$sessionStatus.session_best_test_rmse
        $campaignStatus.current_best_run_id = [string]$sessionStatus.session_best_run_id
        $campaignStatus.current_best_run_dir = [string]$sessionStatus.session_best_run_dir
        Write-Banner -Title "AUTOPILOT IMPROVEMENT" -Message ($campaignStatus.current_best_run_id + " | test_rmse=" + ([double]$campaignStatus.current_best_rmse).ToString("F6"))
    }
    Write-JsonFile -Path $campaignStatusPath -Value $campaignStatus
}

if ($batchCounter -lt $MaxAutoBatches) {
    $extraSeedChunks = @(
        @(83, 89, 97, 101),
        @(103, 107, 109, 113),
        @(127, 131, 137, 139)
    )
    foreach ($chunk in $extraSeedChunks) {
        if ($batchCounter -ge $MaxAutoBatches) {
            break
        }
        $runs = Get-ComparableRuns -RepoRootPath $RepoRoot
        $branchRanking = Get-BranchRanking -Runs $runs
        if ($branchRanking.Count -eq 0) {
            break
        }
        $bestBranch = $branchRanking[0]
        $seen = Get-SeenSignatureSet -Runs $runs
        $plan = New-Object System.Collections.Generic.List[object]
        foreach ($seed in $chunk) {
            $exp = New-ExperimentObject `
                -Branch $bestBranch.branch `
                -TopK ([int]$bestBranch.best_topk) `
                -Seed $seed `
                -Lr 0.001 `
                -WeightDecay ([double]$bestBranch.best_weight_decay) `
                -Precision ([string]$bestBranch.best_precision) `
                -HiddenDim ([int]$bestBranch.best_hidden_dim) `
                -HistLen ([int]$bestBranch.best_hist_len) `
                -SchedulerPatience ([int]$bestBranch.best_patience) `
                -SchedulerCooldown ([int]$bestBranch.best_cooldown) `
                -RunIdPrefix ("auto_seed_extra_" + $bestBranch.branch)
            Add-PlanIfNew -Plan $plan -SeenSignatures $seen -Exp $exp
        }
        if ($plan.Count -eq 0) {
            continue
        }

        $batchCounter += 1
        $stageName = "seed_expand_extra"
        $batchTag = "{0:D2}_{1}" -f $batchCounter, $stageName
        $planPath = Join-Path $planDir ($batchTag + ".json")
        $sessionRoot = Join-Path $CampaignRoot ("batch_" + $batchTag)
        Write-JsonFile -Path $planPath -Value $plan

        $campaignStatus.active_stage = $stageName
        Write-JsonFile -Path $campaignStatusPath -Value $campaignStatus

        Write-Banner -Title ("AUTOPILOT BATCH " + $batchTag) -Message ("Launching " + $plan.Count + " runs")
        & powershell -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "external_speed_benchmarks\run_metrla_corrected_batch.ps1") `
            -PythonExe $PythonExe `
            -RepoRoot $RepoRoot `
            -RootLogDir $sessionRoot `
            -PlanPath $planPath `
            -HistoricalBestRmse $HistoricalBestRmse `
            -InitialSessionBestRmse $campaignStatus.current_best_rmse `
            -InitialSessionBestRunDir $campaignStatus.current_best_run_dir `
            -ReviewNotes ("Autopilot stage: " + $stageName)

        $sessionStatusPath = Join-Path $sessionRoot "queue_status.json"
        $sessionBreakthroughPath = Join-Path $sessionRoot "queue_breakthroughs.jsonl"
        $sessionStatus = Get-Content -Path $sessionStatusPath -Raw | ConvertFrom-Json
        Add-Content -Path $batchIndexPath -Value ($batchCounter.ToString() + "`t" + $stageName + "`t" + $sessionRoot + "`t" + [string]$sessionStatus.session_best_test_rmse + "`t" + [string]$sessionStatus.session_best_run_id) -Encoding utf8

        if (Test-Path $sessionBreakthroughPath) {
            foreach ($line in Get-Content $sessionBreakthroughPath) {
                if ([string]::IsNullOrWhiteSpace($line)) {
                    continue
                }
                Add-Content -Path $campaignBreakthroughPath -Value $line -Encoding utf8
            }
        }

        $campaignStatus.completed_batches = $batchCounter
        if ([double]$sessionStatus.session_best_test_rmse -lt [double]$campaignStatus.current_best_rmse) {
            $campaignStatus.current_best_rmse = [double]$sessionStatus.session_best_test_rmse
            $campaignStatus.current_best_run_id = [string]$sessionStatus.session_best_run_id
            $campaignStatus.current_best_run_dir = [string]$sessionStatus.session_best_run_dir
            Write-Banner -Title "AUTOPILOT IMPROVEMENT" -Message ($campaignStatus.current_best_run_id + " | test_rmse=" + ([double]$campaignStatus.current_best_rmse).ToString("F6"))
        }
        Write-JsonFile -Path $campaignStatusPath -Value $campaignStatus
    }
}

$campaignStatus.active_stage = $null
$campaignStatus.finished_at = (Get-Date -Format o)
Write-JsonFile -Path $campaignStatusPath -Value $campaignStatus
Write-Banner -Title "AUTOPILOT FINISHED" -Message ("Best corrected RMSE=" + ([double]$campaignStatus.current_best_rmse).ToString("F6") + " | run=" + [string]$campaignStatus.current_best_run_id)
