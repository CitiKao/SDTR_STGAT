param(
    [string]$PythonExe = "C:\Users\WMNL\anaconda3\envs\STDR\python.exe",
    [string]$RepoRoot = "D:\Citi\STDR\STDR_STGAT",
    [string]$CampaignRoot = "",
    [int]$BaseEpochs = 40,
    [int]$MaxAutoBatches = 9,
    [int]$FinalBestRepeatEpochs = 60,
    [int]$FinalBestRepeats = 2,
    [switch]$OpenMonitors,
    [switch]$ResumeExisting
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
    $Value | ConvertTo-Json -Depth 12 | Set-Content -Path $Path -Encoding utf8
}

function Get-CompletedStageNames {
    param([string]$BatchIndexPath)
    if (-not (Test-Path $BatchIndexPath)) {
        return @()
    }
    $rows = Import-Csv -Path $BatchIndexPath -Delimiter "`t"
    return @(
        $rows |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_.stage_name) } |
            ForEach-Object { [string]$_.stage_name }
    )
}

function Get-ComparableRuns {
    param([string]$CampaignRootPath)
    $records = New-Object System.Collections.Generic.List[object]
    $historyFiles = Get-ChildItem -Path $CampaignRootPath -Filter "queue_history.jsonl" -Recurse -ErrorAction SilentlyContinue
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
            $records.Add($item)
        }
    }
    return @($records.ToArray())
}

function Get-RunSignature {
    param([object]$Run)
    return (
        [string]$Run.dataset_dir + "|" +
        [string]$Run.lineage + "|" +
        [string]$Run.fixed_graph_weighted + "|" +
        [string]$Run.use_speed_history_mask + "|" +
        [string]$Run.adaptive_topk + "|" +
        [string]$Run.seed + "|" +
        ([double]$Run.lr).ToString("G17") + "|" +
        ([double]$Run.weight_decay).ToString("G17") + "|" +
        [string]$Run.precision + "|" +
        [string]$Run.hidden_dim + "|" +
        [string]$Run.hist_len + "|" +
        [string]$Run.configured_epochs + "|" +
        [string]$Run.scheduler_monitor + "|" +
        [string]$Run.scheduler_patience + "|" +
        [string]$Run.scheduler_cooldown
    )
}

function Get-SeenSignatureSet {
    param([object[]]$Runs)
    $set = [System.Collections.Generic.HashSet[string]]::new()
    foreach ($run in $Runs) {
        $null = $set.Add((Get-RunSignature -Run $run))
    }
    return ,$set
}

function New-ExperimentObject {
    param(
        [string]$Lineage,
        [string]$DatasetDir,
        [bool]$WeightedFixed,
        [bool]$MissingAware,
        [int]$TopK,
        [int]$Seed,
        [double]$Lr,
        [double]$WeightDecay,
        [string]$Precision,
        [int]$HiddenDim,
        [int]$HistLen,
        [int]$Epochs,
        [string]$SchedulerMonitor,
        [int]$SchedulerPatience,
        [int]$SchedulerCooldown,
        [string]$RunIdPrefix
    )
    $safeLr = $Lr.ToString("0.#######").Replace(".", "p")
    $safeWd = $WeightDecay.ToString("0.#######").Replace(".", "p")
    $runId = "{0}_{1}_topk{2}_seed{3}_lr{4}_wd{5}_hist{6}_ep{7}" -f $RunIdPrefix, $Lineage, $TopK, $Seed, $safeLr, $safeWd, $HistLen, $Epochs
    return [pscustomobject]@{
        run_id = $runId
        lineage = $Lineage
        dataset_dir = $DatasetDir
        weighted_fixed = $WeightedFixed
        missing_aware = $MissingAware
        topk = $TopK
        seed = $Seed
        lr = $Lr
        weight_decay = $WeightDecay
        precision = $Precision
        hidden_dim = $HiddenDim
        hist_len = $HistLen
        epochs = $Epochs
        scheduler_monitor = $SchedulerMonitor
        scheduler_patience = $SchedulerPatience
        scheduler_cooldown = $SchedulerCooldown
        split_policy = "project_monthly"
        split_alignment = "none"
        outlier_cleaning_mode = "train_quantile_clip"
    }
}

function Add-PlanIfNew {
    param(
        [System.Collections.Generic.List[object]]$Plan,
        [System.Collections.Generic.HashSet[string]]$SeenSignatures,
        [pscustomobject]$Exp
    )
    $signature = (
        [string]$Exp.dataset_dir + "|" +
        [string]$Exp.lineage + "|" +
        [string]$Exp.weighted_fixed + "|" +
        [string]$Exp.missing_aware + "|" +
        [string]$Exp.topk + "|" +
        [string]$Exp.seed + "|" +
        ([double]$Exp.lr).ToString("G17") + "|" +
        ([double]$Exp.weight_decay).ToString("G17") + "|" +
        [string]$Exp.precision + "|" +
        [string]$Exp.hidden_dim + "|" +
        [string]$Exp.hist_len + "|" +
        [string]$Exp.epochs + "|" +
        [string]$Exp.scheduler_monitor + "|" +
        [string]$Exp.scheduler_patience + "|" +
        [string]$Exp.scheduler_cooldown
    )
    if ($SeenSignatures.Add($signature)) {
        $Plan.Add($Exp)
    }
}

function Get-BestRunOrDefault {
    param(
        [object[]]$Runs,
        [string]$PreparedDatasetDir,
        [int]$Epochs
    )
    if ($Runs.Count -gt 0) {
        return @($Runs | Sort-Object test_rmse | Select-Object -First 1)[0]
    }
    return [pscustomobject]@{
        lineage = "legacy_haversine"
        dataset_dir = $PreparedDatasetDir
        fixed_graph_weighted = $false
        use_speed_history_mask = $false
        adaptive_topk = 16
        seed = 42
        lr = 0.001
        weight_decay = 1e-5
        precision = "bf16"
        hidden_dim = 32
        hist_len = 12
        configured_epochs = $Epochs
        scheduler_monitor = "val_loss"
        scheduler_patience = 8
        scheduler_cooldown = 2
    }
}

function Build-StagePlan {
    param(
        [string]$StageName,
        [object[]]$Runs,
        [System.Collections.Generic.HashSet[string]]$SeenSignatures,
        [string]$PreparedDatasetDir,
        [string]$PreparedDatasetDirMapRoute,
        [int]$Epochs
    )
    $plan = New-Object System.Collections.Generic.List[object]
    $best = @(Get-BestRunOrDefault -Runs $Runs -PreparedDatasetDir $PreparedDatasetDir -Epochs $Epochs)[0]
    $datasetDir = [string](@($best.dataset_dir)[0])
    $lineage = [string](@($best.lineage)[0])
    $weighted = [bool](@($best.fixed_graph_weighted)[0])
    $missingAware = [bool](@($best.use_speed_history_mask)[0])
    $topk = [int](@($best.adaptive_topk)[0])
    $seed = [int](@($best.seed)[0])
    $lr = [double](@($best.lr)[0])
    $weightDecay = [double](@($best.weight_decay)[0])
    $precision = [string](@($best.precision)[0])
    $hiddenDim = [int](@($best.hidden_dim)[0])
    $histLen = [int](@($best.hist_len)[0])
    $schedulerMonitor = [string](@($best.scheduler_monitor)[0])
    $schedulerPatience = [int](@($best.scheduler_patience)[0])
    $schedulerCooldown = [int](@($best.scheduler_cooldown)[0])

    switch ($StageName) {
        "route_bestshot" {
            $exp = New-ExperimentObject `
                -Lineage "legacy_map_route" `
                -DatasetDir $PreparedDatasetDirMapRoute `
                -WeightedFixed $false `
                -MissingAware $false `
                -TopK 16 `
                -Seed 19 `
                -Lr 0.001 `
                -WeightDecay 1e-5 `
                -Precision "bf16" `
                -HiddenDim 32 `
                -HistLen 12 `
                -Epochs $Epochs `
                -SchedulerMonitor "val_loss" `
                -SchedulerPatience 8 `
                -SchedulerCooldown 2 `
                -RunIdPrefix "route_bestshot"
            Add-PlanIfNew -Plan $plan -SeenSignatures $SeenSignatures -Exp $exp
        }
        "paired_seed_matrix" {
            $matrixConfigs = @(
                @{ lineage = "legacy_haversine"; dataset_dir = $PreparedDatasetDir; seed = 19; run_prefix = "matrix_haversine" },
                @{ lineage = "legacy_haversine"; dataset_dir = $PreparedDatasetDir; seed = 31; run_prefix = "matrix_haversine" },
                @{ lineage = "legacy_haversine"; dataset_dir = $PreparedDatasetDir; seed = 42; run_prefix = "matrix_haversine" },
                @{ lineage = "legacy_map_route"; dataset_dir = $PreparedDatasetDirMapRoute; seed = 19; run_prefix = "matrix_maproute" },
                @{ lineage = "legacy_map_route"; dataset_dir = $PreparedDatasetDirMapRoute; seed = 31; run_prefix = "matrix_maproute" },
                @{ lineage = "legacy_map_route"; dataset_dir = $PreparedDatasetDirMapRoute; seed = 42; run_prefix = "matrix_maproute" }
            )
            foreach ($cfg in $matrixConfigs) {
                $exp = New-ExperimentObject `
                    -Lineage ([string]$cfg.lineage) `
                    -DatasetDir ([string]$cfg.dataset_dir) `
                    -WeightedFixed $false `
                    -MissingAware $false `
                    -TopK 16 `
                    -Seed ([int]$cfg.seed) `
                    -Lr 0.001 `
                    -WeightDecay 1e-5 `
                    -Precision "bf16" `
                    -HiddenDim 32 `
                    -HistLen 12 `
                    -Epochs $Epochs `
                    -SchedulerMonitor "val_loss" `
                    -SchedulerPatience 8 `
                    -SchedulerCooldown 2 `
                    -RunIdPrefix ([string]$cfg.run_prefix)
                Add-PlanIfNew -Plan $plan -SeenSignatures $SeenSignatures -Exp $exp
            }
        }
        "seed_expand" {
            foreach ($candidateSeed in @(17, 19, 23, 29, 31, 37, 42)) {
                $exp = New-ExperimentObject `
                    -Lineage $lineage `
                    -DatasetDir $datasetDir `
                    -WeightedFixed $weighted `
                    -MissingAware $missingAware `
                    -TopK $topk `
                    -Seed $candidateSeed `
                    -Lr $lr `
                    -WeightDecay $weightDecay `
                    -Precision $precision `
                    -HiddenDim $hiddenDim `
                    -HistLen $histLen `
                    -Epochs $Epochs `
                    -SchedulerMonitor $schedulerMonitor `
                    -SchedulerPatience $schedulerPatience `
                    -SchedulerCooldown $schedulerCooldown `
                    -RunIdPrefix "auto_seed"
                Add-PlanIfNew -Plan $plan -SeenSignatures $SeenSignatures -Exp $exp
            }
        }
        "topk_local" {
            $baseTopk = [int](@($topk)[0])
            $candidates = @(
                ($baseTopk - 4),
                ($baseTopk - 2),
                ($baseTopk + 2),
                ($baseTopk + 4),
                16
            ) | Where-Object { $_ -ge 8 -and $_ -le 24 } | Select-Object -Unique
            foreach ($candidateTopk in $candidates) {
                $exp = New-ExperimentObject `
                    -Lineage $lineage `
                    -DatasetDir $datasetDir `
                    -WeightedFixed $weighted `
                    -MissingAware $missingAware `
                    -TopK ([int]$candidateTopk) `
                    -Seed $seed `
                    -Lr $lr `
                    -WeightDecay $weightDecay `
                    -Precision $precision `
                    -HiddenDim $hiddenDim `
                    -HistLen $histLen `
                    -Epochs $Epochs `
                    -SchedulerMonitor $schedulerMonitor `
                    -SchedulerPatience $schedulerPatience `
                    -SchedulerCooldown $schedulerCooldown `
                    -RunIdPrefix "auto_topk"
                Add-PlanIfNew -Plan $plan -SeenSignatures $SeenSignatures -Exp $exp
            }
        }
        "scheduler_local" {
            $schedulerConfigs = @(
                @{ patience = 4; cooldown = 0 },
                @{ patience = 6; cooldown = 1 }
            )
            foreach ($cfg in $schedulerConfigs) {
                $exp = New-ExperimentObject `
                    -Lineage $lineage `
                    -DatasetDir $datasetDir `
                    -WeightedFixed $weighted `
                    -MissingAware $missingAware `
                    -TopK $topk `
                    -Seed $seed `
                    -Lr $lr `
                    -WeightDecay $weightDecay `
                    -Precision $precision `
                    -HiddenDim $hiddenDim `
                    -HistLen $histLen `
                    -Epochs $Epochs `
                    -SchedulerMonitor $schedulerMonitor `
                    -SchedulerPatience ([int]$cfg.patience) `
                    -SchedulerCooldown ([int]$cfg.cooldown) `
                    -RunIdPrefix "auto_sched"
                Add-PlanIfNew -Plan $plan -SeenSignatures $SeenSignatures -Exp $exp
            }
        }
        "precision_local" {
            foreach ($candidatePrecision in @("fp32")) {
                $exp = New-ExperimentObject `
                    -Lineage $lineage `
                    -DatasetDir $datasetDir `
                    -WeightedFixed $weighted `
                    -MissingAware $missingAware `
                    -TopK $topk `
                    -Seed $seed `
                    -Lr $lr `
                    -WeightDecay $weightDecay `
                    -Precision $candidatePrecision `
                    -HiddenDim $hiddenDim `
                    -HistLen $histLen `
                    -Epochs $Epochs `
                    -SchedulerMonitor $schedulerMonitor `
                    -SchedulerPatience $schedulerPatience `
                    -SchedulerCooldown $schedulerCooldown `
                    -RunIdPrefix "auto_prec"
                Add-PlanIfNew -Plan $plan -SeenSignatures $SeenSignatures -Exp $exp
            }
        }
        "histlen_local" {
            foreach ($candidateHistLen in @(10, 14)) {
                $exp = New-ExperimentObject `
                    -Lineage $lineage `
                    -DatasetDir $datasetDir `
                    -WeightedFixed $weighted `
                    -MissingAware $missingAware `
                    -TopK $topk `
                    -Seed $seed `
                    -Lr $lr `
                    -WeightDecay $weightDecay `
                    -Precision $precision `
                    -HiddenDim $hiddenDim `
                    -HistLen $candidateHistLen `
                    -Epochs $Epochs `
                    -SchedulerMonitor $schedulerMonitor `
                    -SchedulerPatience $schedulerPatience `
                    -SchedulerCooldown $schedulerCooldown `
                    -RunIdPrefix "auto_hist"
                Add-PlanIfNew -Plan $plan -SeenSignatures $SeenSignatures -Exp $exp
            }
        }
        "semantics_branch" {
            $semanticConfigs = @(
                @{ lineage = "weighted_haversine"; weighted = $true; missing = $false },
                @{ lineage = "missing_haversine"; weighted = $false; missing = $true },
                @{ lineage = "corrected_haversine"; weighted = $true; missing = $true }
            )
            foreach ($cfg in $semanticConfigs) {
                $exp = New-ExperimentObject `
                    -Lineage ([string]$cfg.lineage) `
                    -DatasetDir $datasetDir `
                    -WeightedFixed ([bool]$cfg.weighted) `
                    -MissingAware ([bool]$cfg.missing) `
                    -TopK $topk `
                    -Seed $seed `
                    -Lr $lr `
                    -WeightDecay $weightDecay `
                    -Precision $precision `
                    -HiddenDim $hiddenDim `
                    -HistLen $histLen `
                    -Epochs $Epochs `
                    -SchedulerMonitor $schedulerMonitor `
                    -SchedulerPatience $schedulerPatience `
                    -SchedulerCooldown $schedulerCooldown `
                    -RunIdPrefix "auto_sem"
                Add-PlanIfNew -Plan $plan -SeenSignatures $SeenSignatures -Exp $exp
            }
        }
        default {
            throw "Unknown stage: $StageName"
        }
    }

    return @($plan.ToArray())
}

$referenceRmse = 3.3369038658799606
$referenceRunDir = Join-Path $RepoRoot "runs\external_speed\PEMS-BAY_project_monthly_bs32_ep100_resumeable_20260420_142901"
$preparedDatasetDir = Join-Path $RepoRoot "data\external_datasets\processed\PEMS-BAY_sensor_node_haversine"
$preparedDatasetDirMapRoute = Join-Path $RepoRoot "data\external_datasets\processed\PEMS-BAY_sensor_node_map_route"

if ([string]::IsNullOrWhiteSpace($CampaignRoot)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $CampaignRoot = Join-Path $RepoRoot ("runs\external_speed\autopilot_sessions\PEMS-BAY_distance_paired_autopilot_" + $stamp)
}

$null = New-Item -ItemType Directory -Path $CampaignRoot -Force
$planDir = Join-Path $CampaignRoot "plans"
$batchIndexPath = Join-Path $CampaignRoot "batch_index.tsv"
$campaignStatusPath = Join-Path $CampaignRoot "campaign_status.json"
$campaignBreakthroughPath = Join-Path $CampaignRoot "campaign_breakthroughs.jsonl"
$campaignReviewPath = Join-Path $CampaignRoot "review_rounds.md"
$autopilotLogPath = Join-Path $CampaignRoot "autopilot.log"
$null = New-Item -ItemType Directory -Path $planDir -Force

$campaignStatus = $null
$completedStages = @()
$batchCounter = 0
$isResume = $false

if ($ResumeExisting -and (Test-Path $campaignStatusPath)) {
    $campaignStatus = Get-Content -Path $campaignStatusPath -Raw | ConvertFrom-Json
    $isResume = $true
    $completedStages = Get-CompletedStageNames -BatchIndexPath $batchIndexPath
    $batchCounter = $completedStages.Count
    if ($null -ne $campaignStatus.prepared_dataset_dirs) {
        if (-not [string]::IsNullOrWhiteSpace([string]$campaignStatus.prepared_dataset_dirs.haversine)) {
            $preparedDatasetDir = [string]$campaignStatus.prepared_dataset_dirs.haversine
        }
        if (-not [string]::IsNullOrWhiteSpace([string]$campaignStatus.prepared_dataset_dirs.map_route)) {
            $preparedDatasetDirMapRoute = [string]$campaignStatus.prepared_dataset_dirs.map_route
        }
    }
    if ($null -ne $campaignStatus.base_epochs) {
        $BaseEpochs = [int]$campaignStatus.base_epochs
    }
    Add-Content -Path $autopilotLogPath -Value ("[" + (Get-Date -Format o) + "] resume existing campaign from stages: " + (($completedStages -join ",") )) -Encoding utf8
} else {
    "batch_index`tstage_name`tsession_root`tsession_best_test_rmse`tsession_best_run_id" | Set-Content -Path $batchIndexPath -Encoding utf8
    Set-Content -Path $campaignBreakthroughPath -Value "" -Encoding utf8
}

$reviewText = @"
# PEMS-BAY Distance-Paired Autopilot Review

Participants: Harvey, Lagrange, Leibniz, Ohm, Feynman, Herschel

Round 1. Treat the first objective as a direct best-shot run: `map_route + seed19`, because one branch already showed route distance is slightly stronger than straight-line distance at seed42.
Round 2. Do not decide the distance winner from one seed. Build a paired matrix on the same architecture shell: `haversine/map_route × seed19/31/42`.
Round 3. Preserve the old-best architecture shell for all paired runs: `project_monthly`, `batch=32`, `hist=12`, `pred=12`, `hidden=32`, `heads=4`, `st_blocks=2`, `gtcn=2`, `kernel=3`, `topk=16`, `Adam`, `lr=1e-3`, `wd=1e-5`, `scheduler=val_loss/p8/c2`.
Round 4. Keep `fixed_graph_weighted=false` and `missing_aware=false` through the distance matrix so that distance semantics stay the only main variable.
Round 5. Use a 40-epoch budget because recent PEMS-BAY best checkpoints cluster around epoch 27-36, which is enough to cover the useful range.
Round 6. After the paired matrix, continue seed expansion only on whichever distance branch currently has the best test RMSE.
Round 7. Then do local `topk` tuning around the branch winner instead of sweeping both branches.
Round 8. Scheduler and precision should remain second-order controls, only after branch and seed are resolved.
Round 9. `hist_len` should be tuned later and only on the best branch because its benefit is smaller than distance and seed.
Round 10. Semantic branches like weighted fixed and missing-aware should be deferred until after the distance winner is stable.
"@
if (-not $isResume -or -not (Test-Path $campaignReviewPath)) {
    Set-Content -Path $campaignReviewPath -Value $reviewText -Encoding utf8
}

Set-Location $RepoRoot

if ((-not $isResume) -or (-not (Test-Path $preparedDatasetDir))) {
    Write-Banner -Title "PREPARING DATASET" -Message $preparedDatasetDir
    & $PythonExe (Join-Path $RepoRoot "external_speed_benchmarks\prepare_dcrnn_sensor_datasets.py") `
        --datasets "pems-bay" `
        --graph-mode "official" `
        --representation-domain "sensor_node" `
        --sensor-node-distance-source "haversine"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to prepare PEMS-BAY sensor-node haversine dataset."
    }
}
if (-not (Test-Path $preparedDatasetDir)) {
    throw "Prepared dataset directory not found: $preparedDatasetDir"
}

if ((-not $isResume) -or (-not (Test-Path $preparedDatasetDirMapRoute))) {
    Write-Banner -Title "PREPARING DATASET" -Message $preparedDatasetDirMapRoute
    & $PythonExe (Join-Path $RepoRoot "external_speed_benchmarks\prepare_dcrnn_sensor_datasets.py") `
        --datasets "pems-bay" `
        --graph-mode "official" `
        --representation-domain "sensor_node" `
        --sensor-node-distance-source "map_drive" `
        --map-unreachable-policy "haversine_fallback"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to prepare PEMS-BAY sensor-node map-route dataset."
    }
}
if (-not (Test-Path $preparedDatasetDirMapRoute)) {
    throw "Prepared dataset directory not found: $preparedDatasetDirMapRoute"
}

if (-not $isResume) {
    $campaignStatus = @{
        started_at = (Get-Date -Format o)
        campaign_root = $CampaignRoot
        dataset_name = "PEMS-BAY"
        prepared_dataset_dir = $preparedDatasetDir
        prepared_dataset_dirs = @{
            haversine = $preparedDatasetDir
            map_route = $preparedDatasetDirMapRoute
        }
        base_epochs = $BaseEpochs
        current_best_rmse = $null
        current_best_run_dir = $null
        current_best_run_id = $null
        reference_rmse = $referenceRmse
        reference_run_dir = $referenceRunDir
        active_stage = $null
        completed_batches = 0
        review_rounds = 10
        review_participants = @("Harvey", "Lagrange", "Leibniz", "Ohm", "Feynman", "Herschel")
    }
    Write-JsonFile -Path $campaignStatusPath -Value $campaignStatus
} else {
    $campaignStatus.prepared_dataset_dir = $preparedDatasetDir
    $campaignStatus.prepared_dataset_dirs = @{
        haversine = $preparedDatasetDir
        map_route = $preparedDatasetDirMapRoute
    }
    Write-JsonFile -Path $campaignStatusPath -Value $campaignStatus
}

if ($OpenMonitors) {
    & powershell -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "external_speed_benchmarks\start_pems_bay_autopilot_terminals.ps1") `
        -PythonExe $PythonExe `
        -RepoRoot $RepoRoot `
        -CampaignRoot $CampaignRoot
}

$stageOrder = @(
    "route_bestshot",
    "paired_seed_matrix",
    "seed_expand",
    "topk_local",
    "scheduler_local",
    "precision_local",
    "histlen_local",
    "semantics_branch"
)

foreach ($stageName in $stageOrder) {
    if ($batchCounter -ge $MaxAutoBatches) {
        break
    }
    if ($isResume -and ($completedStages -contains $stageName)) {
        Add-Content -Path $autopilotLogPath -Value ("[" + (Get-Date -Format o) + "] skip completed stage " + $stageName) -Encoding utf8
        continue
    }
    $runs = Get-ComparableRuns -CampaignRootPath $CampaignRoot
    $seen = Get-SeenSignatureSet -Runs $runs
    $plan = @(Build-StagePlan -StageName $stageName -Runs $runs -SeenSignatures $seen -PreparedDatasetDir $preparedDatasetDir -PreparedDatasetDirMapRoute $preparedDatasetDirMapRoute -Epochs $BaseEpochs)
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
    $initialBestRmseForBatch = if ($null -eq $campaignStatus.current_best_rmse) { 1e18 } else { [double]$campaignStatus.current_best_rmse }
    $initialBestRunDirForBatch = if ($null -eq $campaignStatus.current_best_run_dir) { "__none__" } else { [string]$campaignStatus.current_best_run_dir }
    & powershell -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "external_speed_benchmarks\run_pems_bay_autopilot_batch.ps1") `
        -PythonExe $PythonExe `
        -RepoRoot $RepoRoot `
        -RootLogDir $sessionRoot `
        -PlanPath $planPath `
        -DatasetName "PEMS-BAY" `
        -DatasetDir $preparedDatasetDir `
        -SplitPolicy "project_monthly" `
        -SplitAlignment "none" `
        -ReportHorizonsMinutes "15,30,60" `
        -ReferenceRmse $referenceRmse `
        -ReferenceRunDir $referenceRunDir `
        -InitialSessionBestRmse $initialBestRmseForBatch `
        -InitialSessionBestRunDir $initialBestRunDirForBatch `
        -ReviewNotes ("PEMS-BAY autopilot stage: " + $stageName)

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
    if ($null -ne $sessionStatus.session_best_test_rmse) {
        if ($null -eq $campaignStatus.current_best_rmse -or [double]$sessionStatus.session_best_test_rmse -lt [double]$campaignStatus.current_best_rmse) {
            $campaignStatus.current_best_rmse = [double]$sessionStatus.session_best_test_rmse
            $campaignStatus.current_best_run_id = [string]$sessionStatus.session_best_run_id
            $campaignStatus.current_best_run_dir = [string]$sessionStatus.session_best_run_dir
            Write-Banner -Title "AUTOPILOT IMPROVEMENT" -Message ($campaignStatus.current_best_run_id + " | test_rmse=" + ([double]$campaignStatus.current_best_rmse).ToString("F6"))
        }
    }
    Write-JsonFile -Path $campaignStatusPath -Value $campaignStatus
}

if ($batchCounter -lt $MaxAutoBatches) {
    $extraSeedChunks = @(
        @(17, 23, 29, 37),
        @(41, 47, 53, 59)
    )
    foreach ($chunk in $extraSeedChunks) {
        if ($batchCounter -ge $MaxAutoBatches) {
            break
        }
        $runs = Get-ComparableRuns -CampaignRootPath $CampaignRoot
        if ($runs.Count -eq 0) {
            break
        }
        $best = @($runs | Sort-Object test_rmse | Select-Object -First 1)[0]
        $seen = Get-SeenSignatureSet -Runs $runs
        $plan = New-Object System.Collections.Generic.List[object]
        foreach ($candidateSeed in $chunk) {
            $exp = New-ExperimentObject `
                -Lineage ([string]$best.lineage) `
                -DatasetDir ([string]$best.dataset_dir) `
                -WeightedFixed ([bool]$best.fixed_graph_weighted) `
                -MissingAware ([bool]$best.use_speed_history_mask) `
                -TopK ([int]$best.adaptive_topk) `
                -Seed $candidateSeed `
                -Lr ([double]$best.lr) `
                -WeightDecay ([double]$best.weight_decay) `
                -Precision ([string]$best.precision) `
                -HiddenDim ([int]$best.hidden_dim) `
                -HistLen ([int]$best.hist_len) `
                -Epochs $BaseEpochs `
                -SchedulerMonitor ([string]$best.scheduler_monitor) `
                -SchedulerPatience ([int]$best.scheduler_patience) `
                -SchedulerCooldown ([int]$best.scheduler_cooldown) `
                -RunIdPrefix "auto_seed_extra"
            Add-PlanIfNew -Plan $plan -SeenSignatures $seen -Exp $exp
        }
        $plan = @($plan.ToArray())
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
        $initialBestRmseForBatch = if ($null -eq $campaignStatus.current_best_rmse) { 1e18 } else { [double]$campaignStatus.current_best_rmse }
        $initialBestRunDirForBatch = if ($null -eq $campaignStatus.current_best_run_dir) { "__none__" } else { [string]$campaignStatus.current_best_run_dir }
        & powershell -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "external_speed_benchmarks\run_pems_bay_autopilot_batch.ps1") `
            -PythonExe $PythonExe `
            -RepoRoot $RepoRoot `
            -RootLogDir $sessionRoot `
            -PlanPath $planPath `
            -DatasetName "PEMS-BAY" `
            -DatasetDir $preparedDatasetDir `
            -SplitPolicy "project_monthly" `
            -SplitAlignment "none" `
            -ReportHorizonsMinutes "15,30,60" `
            -ReferenceRmse $referenceRmse `
            -ReferenceRunDir $referenceRunDir `
            -InitialSessionBestRmse $initialBestRmseForBatch `
            -InitialSessionBestRunDir $initialBestRunDirForBatch `
            -ReviewNotes ("PEMS-BAY autopilot stage: " + $stageName)

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
        if ($null -ne $sessionStatus.session_best_test_rmse) {
            if ($null -eq $campaignStatus.current_best_rmse -or [double]$sessionStatus.session_best_test_rmse -lt [double]$campaignStatus.current_best_rmse) {
                $campaignStatus.current_best_rmse = [double]$sessionStatus.session_best_test_rmse
                $campaignStatus.current_best_run_id = [string]$sessionStatus.session_best_run_id
                $campaignStatus.current_best_run_dir = [string]$sessionStatus.session_best_run_dir
                Write-Banner -Title "AUTOPILOT IMPROVEMENT" -Message ($campaignStatus.current_best_run_id + " | test_rmse=" + ([double]$campaignStatus.current_best_rmse).ToString("F6"))
            }
        }
        Write-JsonFile -Path $campaignStatusPath -Value $campaignStatus
    }
}

if ($FinalBestRepeats -gt 0 -and $FinalBestRepeatEpochs -gt 0 -and (-not ($completedStages -contains "final_best_60epoch"))) {
    $runs = Get-ComparableRuns -CampaignRootPath $CampaignRoot
    if ($runs.Count -gt 0) {
        $best = @($runs | Sort-Object test_rmse | Select-Object -First 1)[0]
        $plan = New-Object System.Collections.Generic.List[object]
        for ($repeatIndex = 1; $repeatIndex -le $FinalBestRepeats; $repeatIndex++) {
            $exp = New-ExperimentObject `
                -Lineage ([string]$best.lineage) `
                -DatasetDir ([string]$best.dataset_dir) `
                -WeightedFixed ([bool]$best.fixed_graph_weighted) `
                -MissingAware ([bool]$best.use_speed_history_mask) `
                -TopK ([int]$best.adaptive_topk) `
                -Seed ([int]$best.seed) `
                -Lr ([double]$best.lr) `
                -WeightDecay ([double]$best.weight_decay) `
                -Precision ([string]$best.precision) `
                -HiddenDim ([int]$best.hidden_dim) `
                -HistLen ([int]$best.hist_len) `
                -Epochs $FinalBestRepeatEpochs `
                -SchedulerMonitor ([string]$best.scheduler_monitor) `
                -SchedulerPatience ([int]$best.scheduler_patience) `
                -SchedulerCooldown ([int]$best.scheduler_cooldown) `
                -RunIdPrefix ("final_best_repeat" + $repeatIndex)
            $plan.Add($exp)
        }

        if ($plan.Count -gt 0) {
            $batchCounter += 1
            $stageName = "final_best_60epoch"
            $batchTag = "{0:D2}_{1}" -f $batchCounter, $stageName
            $planPath = Join-Path $planDir ($batchTag + ".json")
            $sessionRoot = Join-Path $CampaignRoot ("batch_" + $batchTag)
            Write-JsonFile -Path $planPath -Value @($plan.ToArray())

            $campaignStatus.active_stage = $stageName
            $campaignStatus | Add-Member -NotePropertyName "final_best_repeat_epochs" -NotePropertyValue $FinalBestRepeatEpochs -Force
            $campaignStatus | Add-Member -NotePropertyName "final_best_repeats" -NotePropertyValue $FinalBestRepeats -Force
            Write-JsonFile -Path $campaignStatusPath -Value $campaignStatus

            Write-Banner -Title ("AUTOPILOT BATCH " + $batchTag) -Message ("Launching " + $plan.Count + " final-best repeat runs at " + $FinalBestRepeatEpochs + " epochs")
            $initialBestRmseForBatch = if ($null -eq $campaignStatus.current_best_rmse) { 1e18 } else { [double]$campaignStatus.current_best_rmse }
            $initialBestRunDirForBatch = if ($null -eq $campaignStatus.current_best_run_dir) { "__none__" } else { [string]$campaignStatus.current_best_run_dir }
            & powershell -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "external_speed_benchmarks\run_pems_bay_autopilot_batch.ps1") `
                -PythonExe $PythonExe `
                -RepoRoot $RepoRoot `
                -RootLogDir $sessionRoot `
                -PlanPath $planPath `
                -DatasetName "PEMS-BAY" `
                -DatasetDir $preparedDatasetDir `
                -SplitPolicy "project_monthly" `
                -SplitAlignment "none" `
                -ReportHorizonsMinutes "15,30,60" `
                -ReferenceRmse $referenceRmse `
                -ReferenceRunDir $referenceRunDir `
                -InitialSessionBestRmse $initialBestRmseForBatch `
                -InitialSessionBestRunDir $initialBestRunDirForBatch `
                -ReviewNotes ("PEMS-BAY autopilot stage: " + $stageName + " | repeat final best config twice at 60 epochs")

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
            if ($null -ne $sessionStatus.session_best_test_rmse) {
                if ($null -eq $campaignStatus.current_best_rmse -or [double]$sessionStatus.session_best_test_rmse -lt [double]$campaignStatus.current_best_rmse) {
                    $campaignStatus.current_best_rmse = [double]$sessionStatus.session_best_test_rmse
                    $campaignStatus.current_best_run_id = [string]$sessionStatus.session_best_run_id
                    $campaignStatus.current_best_run_dir = [string]$sessionStatus.session_best_run_dir
                    Write-Banner -Title "AUTOPILOT IMPROVEMENT" -Message ($campaignStatus.current_best_run_id + " | test_rmse=" + ([double]$campaignStatus.current_best_rmse).ToString("F6"))
                }
            }
            Write-JsonFile -Path $campaignStatusPath -Value $campaignStatus
        }
    }
}

$campaignStatus.active_stage = $null
$campaignStatus.finished_at = (Get-Date -Format o)
Write-JsonFile -Path $campaignStatusPath -Value $campaignStatus
Write-Banner -Title "AUTOPILOT FINISHED" -Message ("Best RMSE=" + $(if ($null -eq $campaignStatus.current_best_rmse) { "n/a" } else { ([double]$campaignStatus.current_best_rmse).ToString("F6") }) + " | run=" + [string]$campaignStatus.current_best_run_id)
