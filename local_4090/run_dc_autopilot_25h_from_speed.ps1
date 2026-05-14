param(
    [string]$CondaEnvName = "STDR",
    [string]$PythonExe = "python",
    [string]$RepoRoot = "",
    [string]$CampaignRoot = "",
    [string]$DataDir = "data",
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Device = "cuda",
    [ValidateSet("auto", "bf16", "fp32")]
    [string]$Precision = "bf16",
    [int]$NumWorkers = 0,
    [int]$BatchSize = 32,
    [int]$PredHorizon = 4,
    [string]$ReportHorizonsMinutes = "15,30,60",
    [ValidateSet("project_monthly", "benchmark_contiguous")]
    [string]$SplitPolicy = "project_monthly",
    [ValidateSet("none", "day", "week", "month")]
    [string]$SplitAlignment = "none",
    [switch]$ResumeExisting
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $RepoRoot = Split-Path -Parent $scriptDir
}
$RepoRoot = (Resolve-Path $RepoRoot).Path

if ([string]::IsNullOrWhiteSpace($CampaignRoot)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $CampaignRoot = Join-Path $RepoRoot ("runs\local_4090_dc\autopilot_sessions\DC_autopilot_25h_from_speed_" + $stamp)
}

. (Join-Path $RepoRoot "local_4090\run_dc_autopilot.ps1") `
    -CondaEnvName $CondaEnvName `
    -PythonExe $PythonExe `
    -RepoRoot $RepoRoot `
    -CampaignRoot $CampaignRoot `
    -DataDir $DataDir `
    -BatchSize $BatchSize `
    -Device $Device `
    -Precision $Precision `
    -NumWorkers $NumWorkers `
    -PredHorizon $PredHorizon `
    -ReportHorizonsMinutes $ReportHorizonsMinutes `
    -SplitPolicy $SplitPolicy `
    -SplitAlignment $SplitAlignment `
    -ResumeExisting:$($ResumeExisting.IsPresent) `
    -LoadOnly

function New-PlanList {
    return ,([System.Collections.Generic.List[object]]::new())
}

function Add-ExperimentSafe {
    param(
        [System.Collections.Generic.List[object]]$Plan,
        [System.Collections.Generic.HashSet[string]]$Seen,
        [int]$StageIndex,
        [string]$StageName,
        [int]$Ordinal,
        [int]$TopK,
        [int]$HistLen,
        [string]$Lr,
        [string]$WeightDecay,
        [int]$HiddenDim,
        [int]$Blocks,
        [string]$TimeFeatureMode,
        [int]$Seed,
        [int]$Epochs,
        [int]$ValInterval
    )
    Add-UniqueExperiment -Plan $Plan -Seen $Seen -Experiment (New-Experiment `
        -StageIndex $StageIndex -StageName $StageName -Ordinal $Ordinal `
        -TopK $TopK -HistLen $HistLen -Lr $Lr -WeightDecay $WeightDecay `
        -HiddenDim $HiddenDim -Blocks $Blocks -TimeFeatureMode $TimeFeatureMode `
        -Seed $Seed -Epochs $Epochs -ValInterval $ValInterval)
}

function Build-Smoke25hPlan {
    $plan = New-PlanList
    $seen = [System.Collections.Generic.HashSet[string]]::new()
    Add-ExperimentSafe -Plan $plan -Seen $seen -StageIndex 0 -StageName "smoke" -Ordinal 1 `
        -TopK 20 -HistLen 14 -Lr "1e-3" -WeightDecay "1e-5" `
        -HiddenDim 32 -Blocks 2 -TimeFeatureMode "baseline" -Seed 42 -Epochs 2 -ValInterval 1
    return @($plan.ToArray())
}

function Build-TopkHist25hPlan {
    $plan = New-PlanList
    $seen = [System.Collections.Generic.HashSet[string]]::new()
    $ordinal = 1
    foreach ($topk in @(16, 20, 24)) {
        foreach ($histLen in @(12, 14, 16)) {
            Add-ExperimentSafe -Plan $plan -Seen $seen -StageIndex 1 -StageName "topk_hist" -Ordinal $ordinal `
                -TopK $topk -HistLen $histLen -Lr "1e-3" -WeightDecay "1e-5" `
                -HiddenDim 32 -Blocks 2 -TimeFeatureMode "baseline" -Seed 42 -Epochs 10 -ValInterval 2
            $ordinal += 1
        }
    }
    return @($plan.ToArray())
}

function Build-Optimizer25hPlan {
    param([object[]]$Parents)
    $plan = New-PlanList
    $seen = [System.Collections.Generic.HashSet[string]]::new()
    $ordinal = 1
    foreach ($parent in $Parents) {
        foreach ($lr in @("5e-4", "7e-4", "1e-3")) {
            Add-ExperimentSafe -Plan $plan -Seen $seen -StageIndex 2 -StageName "optimizer" -Ordinal $ordinal `
                -TopK ([int]$parent.topk) -HistLen ([int]$parent.hist_len) -Lr $lr -WeightDecay "1e-5" `
                -HiddenDim 32 -Blocks 2 -TimeFeatureMode "baseline" -Seed 42 -Epochs 14 -ValInterval 2
            $ordinal += 1
        }
    }
    return @($plan.ToArray())
}

function Build-CapacityTime25hPlan {
    param([object[]]$Parents)
    $plan = New-PlanList
    $seen = [System.Collections.Generic.HashSet[string]]::new()
    $ordinal = 1
    foreach ($parent in $Parents) {
        Add-ExperimentSafe -Plan $plan -Seen $seen -StageIndex 3 -StageName "capacity_time" -Ordinal $ordinal `
            -TopK ([int]$parent.topk) -HistLen ([int]$parent.hist_len) `
            -Lr ([string]$parent.lr) -WeightDecay ([string]$parent.weight_decay) `
            -HiddenDim 48 -Blocks 2 -TimeFeatureMode "baseline" -Seed 42 -Epochs 16 -ValInterval 2
        $ordinal += 1
        Add-ExperimentSafe -Plan $plan -Seen $seen -StageIndex 3 -StageName "capacity_time" -Ordinal $ordinal `
            -TopK ([int]$parent.topk) -HistLen ([int]$parent.hist_len) `
            -Lr ([string]$parent.lr) -WeightDecay ([string]$parent.weight_decay) `
            -HiddenDim 32 -Blocks 2 -TimeFeatureMode "day_of_month_and_week_of_month" -Seed 42 -Epochs 16 -ValInterval 2
        $ordinal += 1
    }
    return @($plan.ToArray())
}

function Build-Final25hPlan {
    param([object[]]$Parents)
    $plan = New-PlanList
    $seen = [System.Collections.Generic.HashSet[string]]::new()
    $ordinal = 1
    foreach ($parent in ($Parents | Select-Object -First 3)) {
        foreach ($seed in @(19, 42, 123)) {
            Add-ExperimentSafe -Plan $plan -Seen $seen -StageIndex 4 -StageName "final" -Ordinal $ordinal `
                -TopK ([int]$parent.topk) -HistLen ([int]$parent.hist_len) `
                -Lr ([string]$parent.lr) -WeightDecay ([string]$parent.weight_decay) `
                -HiddenDim ([int]$parent.hidden_dim) -Blocks ([int]$parent.num_st_blocks) `
                -TimeFeatureMode ([string]$parent.time_feature_mode) -Seed $seed -Epochs 45 -ValInterval 1
            $ordinal += 1
        }
    }
    return @($plan.ToArray())
}

$null = New-Item -ItemType Directory -Path $CampaignRoot -Force
$planDir = Join-Path $CampaignRoot "plans"
$null = New-Item -ItemType Directory -Path $planDir -Force
$historyPath = Join-Path $CampaignRoot "queue_history.jsonl"
$statusPath = Join-Path $CampaignRoot "queue_status.json"
$summaryPath = Join-Path $CampaignRoot "summary.tsv"
$finalSummaryPath = Join-Path $CampaignRoot "final_summary.md"

if (-not $ResumeExisting.IsPresent -or -not (Test-Path $historyPath)) {
    Set-Content -Path $historyPath -Value "" -Encoding utf8
}

$campaignConfig = [pscustomobject]@{
    started_at = (Get-Date -Format o)
    repo_root = $RepoRoot
    campaign_root = $CampaignRoot
    conda_env_name = $CondaEnvName
    python_exe = $PythonExe
    data_dir = $DataDir
    split_policy = $SplitPolicy
    split_alignment = $SplitAlignment
    resume_existing = [bool]$ResumeExisting.IsPresent
    batch_size = $BatchSize
    train_task = "dc"
    monitor_task = "raw_dc"
    pred_horizon = $PredHorizon
    report_horizons_minutes = $ReportHorizonsMinutes
    plan = "25h_from_speed"
    stage_epochs = @{
        smoke = 2
        topk_hist = 10
        optimizer = 14
        capacity_time = 16
        final = 45
    }
    keep_counts = @{
        stage1 = 4
        stage2 = 4
        stage3 = 3
        final_config_count = 3
    }
    stage_plan_counts = @{
        smoke = 1
        topk_hist = 9
        optimizer = 12
        capacity_time = 8
        final = 9
        total = 39
    }
}
Write-JsonFile -Path (Join-Path $CampaignRoot "campaign_config.json") -Value $campaignConfig

Write-Banner -Title "DC AUTOPILOT 25H FROM SPEED" -Message ("Campaign root: " + $CampaignRoot)

$allRecords = New-Object System.Collections.Generic.List[object]
$env:PYTHONUNBUFFERED = "1"
$pythonPathBackup = $env:PYTHONPATH
$env:PYTHONPATH = $null

Push-Location $RepoRoot
try {
    $smokeRecords = Invoke-Stage -StageName "smoke" -Plan (Build-Smoke25hPlan) -RootLogDir $CampaignRoot -PlanDir $planDir -HistoryPath $historyPath -StatusPath $statusPath
    foreach ($record in $smokeRecords) { $allRecords.Add($record) }
    Export-SummaryTsv -Records @($allRecords.ToArray()) -Path $summaryPath
    if ((Select-TopRecords -Records $smokeRecords -Count 1).Count -eq 0) {
        throw "Smoke stage produced no successful run."
    }

    $stage1Records = Invoke-Stage -StageName "topk_hist" -Plan (Build-TopkHist25hPlan) -RootLogDir $CampaignRoot -PlanDir $planDir -HistoryPath $historyPath -StatusPath $statusPath
    foreach ($record in $stage1Records) { $allRecords.Add($record) }
    $selected = Select-TopRecords -Records $stage1Records -Count 4
    Write-JsonFile -Path (Join-Path $CampaignRoot "selected_after_topk_hist.json") -Value $selected
    Export-SummaryTsv -Records @($allRecords.ToArray()) -Path $summaryPath
    if ($selected.Count -eq 0) { throw "topk_hist stage produced no successful run." }

    $stage2Records = Invoke-Stage -StageName "optimizer" -Plan (Build-Optimizer25hPlan -Parents $selected) -RootLogDir $CampaignRoot -PlanDir $planDir -HistoryPath $historyPath -StatusPath $statusPath
    foreach ($record in $stage2Records) { $allRecords.Add($record) }
    $selected = Select-TopRecords -Records $stage2Records -Count 4
    Write-JsonFile -Path (Join-Path $CampaignRoot "selected_after_optimizer.json") -Value $selected
    Export-SummaryTsv -Records @($allRecords.ToArray()) -Path $summaryPath
    if ($selected.Count -eq 0) { throw "optimizer stage produced no successful run." }

    $stage3Records = Invoke-Stage -StageName "capacity_time" -Plan (Build-CapacityTime25hPlan -Parents $selected) -RootLogDir $CampaignRoot -PlanDir $planDir -HistoryPath $historyPath -StatusPath $statusPath
    foreach ($record in $stage3Records) { $allRecords.Add($record) }
    $selected = Select-TopRecords -Records $stage3Records -Count 3
    Write-JsonFile -Path (Join-Path $CampaignRoot "selected_after_capacity_time.json") -Value $selected
    Export-SummaryTsv -Records @($allRecords.ToArray()) -Path $summaryPath
    if ($selected.Count -eq 0) { throw "capacity_time stage produced no successful run." }

    $finalRecords = Invoke-Stage -StageName "final" -Plan (Build-Final25hPlan -Parents $selected) -RootLogDir $CampaignRoot -PlanDir $planDir -HistoryPath $historyPath -StatusPath $statusPath
    foreach ($record in $finalRecords) { $allRecords.Add($record) }
    $selectedFinal = Select-TopRecords -Records $finalRecords -Count 20
    Write-JsonFile -Path (Join-Path $CampaignRoot "selected_final.json") -Value $selectedFinal
    Export-SummaryTsv -Records @($allRecords.ToArray()) -Path $summaryPath
    Write-FinalMarkdown -Records @($allRecords.ToArray()) -Path $finalSummaryPath
    Set-CampaignComplete -StatusPath $statusPath -StageName "final" -SummaryPath $summaryPath -FinalSummaryPath $finalSummaryPath
}
finally {
    Pop-Location
    $env:PYTHONPATH = $pythonPathBackup
}
