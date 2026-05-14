param(
    [string]$PythonExe = "C:\Users\WMNL\anaconda3\envs\STDR\python.exe",
    [string]$RepoRoot = "D:\Citi\STDR\STDR_STGAT",
    [string]$CampaignRoot = "",
    [int]$RepeatEpochs = 40,
    [int]$Repeats = 3
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($CampaignRoot)) {
    throw "CampaignRoot is required."
}

function Write-JsonFile {
    param(
        [string]$Path,
        [object]$Value
    )
    $Value | ConvertTo-Json -Depth 12 | Set-Content -Path $Path -Encoding utf8
}

function Add-OrReplaceNoteProperty {
    param(
        [object]$Object,
        [string]$Name,
        [object]$Value
    )
    $Object | Add-Member -NotePropertyName $Name -NotePropertyValue $Value -Force
}

$plansDir = Join-Path $CampaignRoot "plans"
$campaignStatusPath = Join-Path $CampaignRoot "campaign_status.json"
$batchIndexPath = Join-Path $CampaignRoot "batch_index.tsv"
$templatePlanPath = Join-Path $plansDir "10_final_best_60epoch.json"

if (-not (Test-Path $campaignStatusPath)) {
    throw "Missing campaign_status.json: $campaignStatusPath"
}
if (-not (Test-Path $templatePlanPath)) {
    throw "Missing final-best template plan: $templatePlanPath"
}

$campaignStatus = Get-Content -Path $campaignStatusPath -Raw | ConvertFrom-Json
$templateRaw = Get-Content -Path $templatePlanPath -Raw | ConvertFrom-Json
$template = @($templateRaw)[0]

$completedBatches = 0
if ($null -ne $campaignStatus.completed_batches) {
    $completedBatches = [int]$campaignStatus.completed_batches
}
$batchIndex = [Math]::Max(11, $completedBatches + 1)
$stageName = "final_best_40epoch"
$batchTag = "{0:D2}_{1}" -f $batchIndex, $stageName
$planPath = Join-Path $plansDir ($batchTag + ".json")
$sessionRoot = Join-Path $CampaignRoot ("batch_" + $batchTag)

$plan = New-Object System.Collections.Generic.List[object]
for ($i = 1; $i -le $Repeats; $i++) {
    $runId = "final_best_40repeat{0}_{1}_topk{2}_seed{3}_lr0p001_wd0p00001_hist{4}_ep{5}" -f `
        $i, [string]$template.lineage, [int]$template.topk, [int]$template.seed, [int]$template.hist_len, $RepeatEpochs

    $plan.Add([pscustomobject]@{
        run_id = $runId
        lineage = [string]$template.lineage
        dataset_dir = [string]$template.dataset_dir
        weighted_fixed = [bool]$template.weighted_fixed
        missing_aware = [bool]$template.missing_aware
        topk = [int]$template.topk
        seed = [int]$template.seed
        lr = [double]$template.lr
        weight_decay = [double]$template.weight_decay
        precision = [string]$template.precision
        hidden_dim = [int]$template.hidden_dim
        hist_len = [int]$template.hist_len
        epochs = $RepeatEpochs
        scheduler_monitor = [string]$template.scheduler_monitor
        scheduler_patience = [int]$template.scheduler_patience
        scheduler_cooldown = [int]$template.scheduler_cooldown
        split_policy = [string]$template.split_policy
        split_alignment = [string]$template.split_alignment
        outlier_cleaning_mode = [string]$template.outlier_cleaning_mode
    })
}

Write-JsonFile -Path $planPath -Value @($plan.ToArray())

$campaignStatus.active_stage = $stageName
Add-OrReplaceNoteProperty -Object $campaignStatus -Name "final_best_40_repeat_epochs" -Value $RepeatEpochs
Add-OrReplaceNoteProperty -Object $campaignStatus -Name "final_best_40_repeats" -Value $Repeats
Write-JsonFile -Path $campaignStatusPath -Value $campaignStatus

$initialBestRmse = if ($null -eq $campaignStatus.current_best_rmse) { 1e18 } else { [double]$campaignStatus.current_best_rmse }
$initialBestRunDir = if ($null -eq $campaignStatus.current_best_run_dir) { "__none__" } else { [string]$campaignStatus.current_best_run_dir }
$referenceRmse = if ($null -eq $campaignStatus.reference_rmse) { 3.3369038658799606 } else { [double]$campaignStatus.reference_rmse }
$referenceRunDir = if ($null -eq $campaignStatus.reference_run_dir) { Join-Path $RepoRoot "runs\external_speed\PEMS-BAY_project_monthly_bs32_ep100_resumeable_20260420_142901" } else { [string]$campaignStatus.reference_run_dir }
$preparedDatasetDir = if ($null -eq $campaignStatus.prepared_dataset_dir) { Join-Path $RepoRoot "data\external_datasets\processed\PEMS-BAY_sensor_node_haversine" } else { [string]$campaignStatus.prepared_dataset_dir }

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
    -InitialSessionBestRmse $initialBestRmse `
    -InitialSessionBestRunDir $initialBestRunDir `
    -ReviewNotes ("PEMS-BAY manual final-best 40epoch repeats")

$statusPath = Join-Path $sessionRoot "queue_status.json"
if (-not (Test-Path $statusPath)) {
    throw "Missing queue_status.json after final-best 40epoch repeats: $statusPath"
}

$sessionStatus = Get-Content -Path $statusPath -Raw | ConvertFrom-Json
if (Test-Path $batchIndexPath) {
    $batchIndexContent = Get-Content -Path $batchIndexPath -Raw
    if ($batchIndexContent -notlike ("*" + $sessionRoot + "*")) {
        Add-Content -Path $batchIndexPath -Value ($batchIndex.ToString() + "`t" + $stageName + "`t" + $sessionRoot + "`t" + [string]$sessionStatus.session_best_test_rmse + "`t" + [string]$sessionStatus.session_best_run_id) -Encoding utf8
    }
}

$campaignStatus = Get-Content -Path $campaignStatusPath -Raw | ConvertFrom-Json
$campaignStatus.completed_batches = $batchIndex
if ($null -ne $sessionStatus.session_best_test_rmse) {
    if ($null -eq $campaignStatus.current_best_rmse -or [double]$sessionStatus.session_best_test_rmse -lt [double]$campaignStatus.current_best_rmse) {
        $campaignStatus.current_best_rmse = [double]$sessionStatus.session_best_test_rmse
        $campaignStatus.current_best_run_id = [string]$sessionStatus.session_best_run_id
        $campaignStatus.current_best_run_dir = [string]$sessionStatus.session_best_run_dir
    }
}
$campaignStatus.active_stage = $null
Add-OrReplaceNoteProperty -Object $campaignStatus -Name "finished_final_best_40epoch_at" -Value (Get-Date -Format o)
Write-JsonFile -Path $campaignStatusPath -Value $campaignStatus
