param(
    [string]$CondaEnvName = "STDR",
    [string]$PythonExe = "python",
    [string]$RepoRoot = "",
    [string]$SourceCampaignRoot = "",
    [string]$ContinuationRoot = "",
    [string]$DataDir = "data",
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Device = "cuda",
    [ValidateSet("auto", "bf16", "fp32")]
    [string]$Precision = "bf16",
    [int]$NumWorkers = 0,
    [int]$TargetEpoch = 100,
    [string]$ExcludeOrdinals = "004,005",
    [ValidateSet("project_monthly", "benchmark_contiguous")]
    [string]$SplitPolicy = "project_monthly",
    [ValidateSet("none", "day", "week", "month")]
    [string]$SplitAlignment = "none"
)

$ErrorActionPreference = "Stop"

function Initialize-Utf8Console {
    try {
        chcp.com 65001 | Out-Null
    }
    catch {
    }
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [Console]::InputEncoding = $utf8NoBom
    [Console]::OutputEncoding = $utf8NoBom
    $script:OutputEncoding = $utf8NoBom
    $env:PYTHONUTF8 = "1"
    $env:PYTHONIOENCODING = "utf-8"
    $env:PYTHONUNBUFFERED = "1"
}

Initialize-Utf8Console

function Write-OrangeHost {
    param([string]$Text)
    $esc = [char]27
    Write-Host ("{0}[38;5;208m{1}{0}[0m" -f $esc, $Text)
}

function Write-Banner {
    param([string]$Title, [string]$Message = "")
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host $Title -ForegroundColor Cyan
    if (-not [string]::IsNullOrWhiteSpace($Message)) {
        Write-Host $Message
    }
    Write-Host "============================================================" -ForegroundColor Cyan
}

function Write-JsonFile {
    param([string]$Path, [object]$Value)
    $parent = Split-Path -Parent $Path
    if (-not [string]::IsNullOrWhiteSpace($parent)) {
        $null = New-Item -ItemType Directory -Path $parent -Force
    }
    $Value | ConvertTo-Json -Depth 40 | Set-Content -Path $Path -Encoding utf8
}

function Read-JsonFile {
    param([string]$Path)
    return Get-Content -Path $Path -Raw | ConvertFrom-Json
}

function Get-ObjectProperty {
    param([object]$Object, [string]$Name)
    if ($null -eq $Object) {
        return $null
    }
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) {
        return $null
    }
    return $property.Value
}

function Get-NestedValue {
    param([object]$Object, [string[]]$Path)
    $cursor = $Object
    foreach ($part in $Path) {
        $cursor = Get-ObjectProperty -Object $cursor -Name $part
        if ($null -eq $cursor) {
            return $null
        }
    }
    return $cursor
}

function Convert-ToNullableDouble {
    param([object]$Value)
    if ($null -eq $Value -or [string]::IsNullOrWhiteSpace([string]$Value)) {
        return $null
    }
    try {
        return [double]$Value
    }
    catch {
        return $null
    }
}

function Format-Value {
    param([object]$Value, [int]$Digits = 4)
    if ($null -eq $Value -or [string]::IsNullOrWhiteSpace([string]$Value)) {
        return "-"
    }
    try {
        return ([double]$Value).ToString("F$Digits")
    }
    catch {
        return [string]$Value
    }
}

function Get-SortValue {
    param([object]$Value)
    $number = Convert-ToNullableDouble -Value $Value
    if ($null -eq $number) {
        return [double]::PositiveInfinity
    }
    return [double]$number
}

function Get-BestEpochFromHistory {
    param([object[]]$History)
    $best = $null
    foreach ($record in $History) {
        $metric = Convert-ToNullableDouble (Get-ObjectProperty -Object $record -Name "val_raw_dc")
        if ($null -eq $metric) {
            continue
        }
        if ($null -eq $best -or $metric -lt $best.metric) {
            $best = [pscustomobject]@{
                epoch = [int](Get-ObjectProperty -Object $record -Name "epoch")
                metric = $metric
            }
        }
    }
    return $best
}

function Get-History {
    param([string]$RunDir)
    $path = Join-Path $RunDir "predictor_log.json"
    if (-not (Test-Path $path)) {
        throw "History not found: $path"
    }
    $history = Read-JsonFile -Path $path
    if (-not ($history -is [array]) -or $history.Count -eq 0) {
        throw "History is empty or invalid: $path"
    }
    return @($history)
}

function Get-CheckpointEpoch {
    param([string]$RunDir)
    $statePath = Join-Path $RunDir "training_state_latest.pt"
    if (-not (Test-Path $statePath)) {
        return $null
    }
    $code = "import pathlib,sys,torch; p=pathlib.Path(sys.argv[1])/'training_state_latest.pt'; payload=torch.load(p,map_location='cpu',weights_only=False); print(payload.get('epoch') or payload.get('last_epoch') or '')"
    try {
        $output = & conda run --no-capture-output -n $CondaEnvName $PythonExe -c $code $RunDir 2>$null
        foreach ($line in @($output)) {
            $text = ([string]$line).Trim()
            if ($text -match "^\d+$") {
                return [int]$text
            }
        }
    }
    catch {
        return $null
    }
    return $null
}

function Get-ResumeEpoch {
    param([string]$RunDir)
    $history = Get-History -RunDir $RunDir
    $historyEpoch = [int]$history[-1].epoch
    $checkpointEpoch = Get-CheckpointEpoch -RunDir $RunDir
    if ($null -ne $checkpointEpoch -and [int]$checkpointEpoch -gt $historyEpoch) {
        return [int]$checkpointEpoch
    }
    return $historyEpoch
}

function Add-Nullable {
    param([Nullable[double]]$A, [Nullable[double]]$B)
    if ($null -eq $A -or $null -eq $B) {
        return $null
    }
    return [double]$A + [double]$B
}

function Get-MetricsSummary {
    param([string]$MetricsPath)
    if (-not (Test-Path $MetricsPath)) {
        return [pscustomobject]@{
            metrics_status = "missing"
            selection_score = $null
            val_raw_dc = $null
            val_gap_rmse = $null
            test_raw_dc = $null
            test_demand_rmse = $null
            test_supply_rmse = $null
            test_gap_rmse = $null
            d_rmse_15 = $null
            d_rmse_30 = $null
            d_rmse_60 = $null
            c_rmse_15 = $null
            c_rmse_30 = $null
            c_rmse_60 = $null
            gap_rmse_15 = $null
            gap_rmse_30 = $null
            gap_rmse_60 = $null
            selected_checkpoint_metric = $null
            selected_checkpoint_task = $null
        }
    }

    $payload = Read-JsonFile -Path $MetricsPath
    $testDemandRmse = Convert-ToNullableDouble (Get-NestedValue -Object $payload -Path @("raw_metrics", "demand", "rmse"))
    $testSupplyRmse = Convert-ToNullableDouble (Get-NestedValue -Object $payload -Path @("raw_metrics", "supply", "rmse"))
    $valDemandRmse = Convert-ToNullableDouble (Get-NestedValue -Object $payload -Path @("val_raw_metrics", "demand", "rmse"))
    $valSupplyRmse = Convert-ToNullableDouble (Get-NestedValue -Object $payload -Path @("val_raw_metrics", "supply", "rmse"))
    $selectedMetric = Convert-ToNullableDouble (Get-NestedValue -Object $payload -Path @("selected_checkpoint_metric"))
    $selectedTask = [string](Get-NestedValue -Object $payload -Path @("selected_checkpoint_task"))
    $valRawDc = Add-Nullable -A $valDemandRmse -B $valSupplyRmse
    if ($null -eq $valRawDc -and $selectedTask -eq "raw_dc") {
        $valRawDc = $selectedMetric
    }

    return [pscustomobject]@{
        metrics_status = "ok"
        selection_score = $valRawDc
        val_raw_dc = $valRawDc
        val_gap_rmse = Convert-ToNullableDouble (Get-NestedValue -Object $payload -Path @("val_raw_metrics", "gap", "rmse"))
        test_raw_dc = Add-Nullable -A $testDemandRmse -B $testSupplyRmse
        test_demand_rmse = $testDemandRmse
        test_supply_rmse = $testSupplyRmse
        test_gap_rmse = Convert-ToNullableDouble (Get-NestedValue -Object $payload -Path @("raw_metrics", "gap", "rmse"))
        d_rmse_15 = Convert-ToNullableDouble (Get-NestedValue -Object $payload -Path @("raw_metrics_report", "demand", "15min", "rmse"))
        d_rmse_30 = Convert-ToNullableDouble (Get-NestedValue -Object $payload -Path @("raw_metrics_report", "demand", "30min", "rmse"))
        d_rmse_60 = Convert-ToNullableDouble (Get-NestedValue -Object $payload -Path @("raw_metrics_report", "demand", "60min", "rmse"))
        c_rmse_15 = Convert-ToNullableDouble (Get-NestedValue -Object $payload -Path @("raw_metrics_report", "supply", "15min", "rmse"))
        c_rmse_30 = Convert-ToNullableDouble (Get-NestedValue -Object $payload -Path @("raw_metrics_report", "supply", "30min", "rmse"))
        c_rmse_60 = Convert-ToNullableDouble (Get-NestedValue -Object $payload -Path @("raw_metrics_report", "supply", "60min", "rmse"))
        gap_rmse_15 = Convert-ToNullableDouble (Get-NestedValue -Object $payload -Path @("raw_metrics_report", "gap", "15min", "rmse"))
        gap_rmse_30 = Convert-ToNullableDouble (Get-NestedValue -Object $payload -Path @("raw_metrics_report", "gap", "30min", "rmse"))
        gap_rmse_60 = Convert-ToNullableDouble (Get-NestedValue -Object $payload -Path @("raw_metrics_report", "gap", "60min", "rmse"))
        selected_checkpoint_metric = $selectedMetric
        selected_checkpoint_task = $selectedTask
    }
}

function Join-CommandPreview {
    param([object[]]$Args)
    $quoted = @(
        $Args | ForEach-Object {
            $item = [string]$_
            if ($item -match "\s") {
                '"' + $item.Replace('"', '\"') + '"'
            }
            else {
                $item
            }
        }
    )
    return ("conda run --no-capture-output -n {0} {1} {2}" -f $CondaEnvName, $PythonExe, ($quoted -join " "))
}

function Export-SummaryTsv {
    param([object[]]$Records, [string]$Path)
    $columns = @(
        "ordinal", "source_ordinal", "run_id", "status", "resume_from_epoch", "target_epoch", "extra_epochs",
        "selection_score", "val_raw_dc", "val_gap_rmse", "test_raw_dc", "test_demand_rmse", "test_supply_rmse", "test_gap_rmse",
        "d_rmse_15", "d_rmse_30", "d_rmse_60", "c_rmse_15", "c_rmse_30", "c_rmse_60",
        "topk", "hist_len", "lr", "weight_decay", "hidden_dim", "num_st_blocks", "time_feature_mode", "seed", "run_dir"
    )
    $lines = New-Object System.Collections.Generic.List[string]
    $lines.Add(($columns -join "`t"))
    foreach ($record in $Records) {
        $values = foreach ($column in $columns) {
            $value = Get-ObjectProperty -Object $record -Name $column
            if ($null -eq $value) { "" } else { ([string]$value).Replace("`t", " ") }
        }
        $lines.Add(($values -join "`t"))
    }
    Set-Content -Path $Path -Value $lines -Encoding utf8
}

function Write-FinalMarkdown {
    param([object[]]$Records, [string]$Path)
    $top = @(
        $Records |
            Where-Object { $_.status -in @("ok", "already_complete") } |
            Sort-Object `
                @{ Expression = { Get-SortValue $_.selection_score }; Ascending = $true }, `
                @{ Expression = { Get-SortValue $_.val_gap_rmse }; Ascending = $true }, `
                @{ Expression = { [string]$_.run_id }; Ascending = $true }
    )
    $lines = New-Object System.Collections.Generic.List[string]
    $lines.Add("# DC Final Continue To 100 Summary")
    $lines.Add("")
    $lines.Add("- Continued source final runs in-place from their latest training state.")
    $lines.Add("- Excluded source ordinals: $ExcludeOrdinals")
    $lines.Add("- Target epoch: $TargetEpoch")
    $lines.Add("")
    $lines.Add("| Rank | Source | Run | Best Val raw DC | Test raw DC | D RMSE | C RMSE | Gap RMSE | Target | TopK | Hist | LR | Seed |")
    $lines.Add("| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |")
    $rank = 1
    foreach ($record in $top) {
        $lines.Add((
            "| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} | {8} | {9} | {10} | {11} | {12} |" -f `
                $rank,
                $record.source_ordinal,
                $record.run_id,
                (Format-Value $record.selection_score),
                (Format-Value $record.test_raw_dc),
                (Format-Value $record.test_demand_rmse),
                (Format-Value $record.test_supply_rmse),
                (Format-Value $record.test_gap_rmse),
                $record.target_epoch,
                $record.topk,
                $record.hist_len,
                $record.lr,
                $record.seed
        ))
        $rank += 1
    }
    Set-Content -Path $Path -Value $lines -Encoding utf8
}

function Get-RecordFromRun {
    param(
        [object]$PlanItem,
        [string]$Status,
        [int]$ExitCode
    )
    $metrics = Get-MetricsSummary -MetricsPath (Join-Path $PlanItem.run_dir "predictor_test_metrics.json")
    $history = Get-History -RunDir $PlanItem.run_dir
    $lastEpoch = [int]$history[-1].epoch
    $bestEpoch = Get-BestEpochFromHistory -History $history
    return [pscustomobject]@{
        ordinal = $PlanItem.ordinal
        source_ordinal = $PlanItem.source_ordinal
        run_id = $PlanItem.run_id
        status = $Status
        exit_code = $ExitCode
        resume_from_epoch = $PlanItem.resume_from_epoch
        target_epoch = $PlanItem.target_epoch
        extra_epochs = $PlanItem.extra_epochs
        completed_epoch = $lastEpoch
        best_epoch = if ($null -eq $bestEpoch) { $null } else { $bestEpoch.epoch }
        selection_score = $metrics.selection_score
        val_raw_dc = $metrics.val_raw_dc
        val_gap_rmse = $metrics.val_gap_rmse
        test_raw_dc = $metrics.test_raw_dc
        test_demand_rmse = $metrics.test_demand_rmse
        test_supply_rmse = $metrics.test_supply_rmse
        test_gap_rmse = $metrics.test_gap_rmse
        d_rmse_15 = $metrics.d_rmse_15
        d_rmse_30 = $metrics.d_rmse_30
        d_rmse_60 = $metrics.d_rmse_60
        c_rmse_15 = $metrics.c_rmse_15
        c_rmse_30 = $metrics.c_rmse_30
        c_rmse_60 = $metrics.c_rmse_60
        topk = $PlanItem.topk
        hist_len = $PlanItem.hist_len
        lr = $PlanItem.lr
        weight_decay = $PlanItem.weight_decay
        hidden_dim = $PlanItem.hidden_dim
        num_st_blocks = $PlanItem.num_st_blocks
        time_feature_mode = $PlanItem.time_feature_mode
        seed = $PlanItem.seed
        run_dir = $PlanItem.run_dir
    }
}

function Write-BestSoFar {
    param([object[]]$Records)
    $best = @(
        $Records |
            Where-Object { $_.status -in @("ok", "already_complete") } |
            Sort-Object `
                @{ Expression = { Get-SortValue $_.selection_score }; Ascending = $true }, `
                @{ Expression = { Get-SortValue $_.val_gap_rmse }; Ascending = $true }, `
                @{ Expression = { [string]$_.run_id }; Ascending = $true } |
            Select-Object -First 1
    )
    if ($best.Count -eq 0) {
        Write-Host "BEST CONFIG SO FAR: none yet" -ForegroundColor Green
        return
    }
    $row = $best[0]
    Write-Host (
        "BEST CONFIG SO FAR: {0} | ValRawDC={1} | TestRawDC={2} | D={3} | C={4} | BestEpoch={5}" -f `
            $row.run_id,
            (Format-Value $row.selection_score),
            (Format-Value $row.test_raw_dc),
            (Format-Value $row.test_demand_rmse),
            (Format-Value $row.test_supply_rmse),
            $row.best_epoch
    ) -ForegroundColor Green
}

function Build-TrainArgs {
    param([object]$Item)
    return @(
        "-u",
        "train_predictor.py",
        "--data-dir", $DataDir,
        "--log-dir", $Item.run_dir,
        "--resume-run-dir", $Item.run_dir,
        "--resume-state-mode", "latest",
        "--device", $Device,
        "--precision", $Precision,
        "--batch-size", [string]$Item.batch_size,
        "--epochs", [string]$Item.extra_epochs,
        "--lr", [string]$Item.lr,
        "--optimizer", "adam",
        "--weight-decay", [string]$Item.weight_decay,
        "--num-workers", [string]$NumWorkers,
        "--val-interval", [string]$Item.val_interval,
        "--log-interval", "1",
        "--hidden-dim", [string]$Item.hidden_dim,
        "--num-st-blocks", [string]$Item.num_st_blocks,
        "--adaptive-topk", [string]$Item.topk,
        "--hist-len", [string]$Item.hist_len,
        "--pred-horizon", [string]$Item.pred_horizon,
        "--report-horizons-minutes", [string]$Item.report_horizons_minutes,
        "--train-task", "dc",
        "--monitor-task", "raw_dc",
        "--split-policy", $SplitPolicy,
        "--split-alignment", $SplitAlignment,
        "--seed", [string]$Item.seed,
        "--time-feature-mode", [string]$Item.time_feature_mode,
        "--scheduler-factor", "0.5",
        "--scheduler-patience", "10",
        "--scheduler-cooldown", "0",
        "--scheduler-min-lr", "0.0"
    )
}

if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $RepoRoot = Split-Path -Parent $scriptDir
}
$RepoRoot = (Resolve-Path $RepoRoot).Path

if ([string]::IsNullOrWhiteSpace($SourceCampaignRoot)) {
    $sessionRoot = Join-Path $RepoRoot "runs\local_4090_dc\autopilot_sessions"
    $SourceCampaignRoot = (Get-ChildItem -Path $sessionRoot -Directory -ErrorAction Stop |
        Where-Object { Test-Path (Join-Path $_.FullName "stage_s04_final_results.json") } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1).FullName
}
$SourceCampaignRoot = (Resolve-Path $SourceCampaignRoot).Path

if ([string]::IsNullOrWhiteSpace($ContinuationRoot)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $ContinuationRoot = Join-Path $RepoRoot ("runs\local_4090_dc\autopilot_sessions\DC_final_continue_to100_" + $stamp)
}

$null = New-Item -ItemType Directory -Path $ContinuationRoot -Force
$planDir = Join-Path $ContinuationRoot "plans"
$null = New-Item -ItemType Directory -Path $planDir -Force
$historyPath = Join-Path $ContinuationRoot "queue_history.jsonl"
$statusPath = Join-Path $ContinuationRoot "queue_status.json"
$summaryPath = Join-Path $ContinuationRoot "summary.tsv"
$finalSummaryPath = Join-Path $ContinuationRoot "final_summary.md"
Set-Content -Path $historyPath -Value "" -Encoding utf8

$excludedSet = [System.Collections.Generic.HashSet[string]]::new()
foreach ($item in ($ExcludeOrdinals.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ })) {
    $null = $excludedSet.Add($item.PadLeft(3, "0"))
}

$sourceRuns = @(
    Get-ChildItem -Path $SourceCampaignRoot -Directory -Filter "s04_final_*" |
        Sort-Object Name |
        Where-Object {
            $match = [regex]::Match($_.Name, "^s04_final_(\d{3})")
            $match.Success -and -not $excludedSet.Contains($match.Groups[1].Value)
        }
)

if ($sourceRuns.Count -eq 0) {
    throw "No source final runs found under $SourceCampaignRoot after excluding $ExcludeOrdinals"
}

$plan = New-Object System.Collections.Generic.List[object]
$ordinal = 1
foreach ($run in $sourceRuns) {
    $sourceOrdinal = [regex]::Match($run.Name, "^s04_final_(\d{3})").Groups[1].Value
    $configPath = Join-Path $run.FullName "experiment_config.json"
    $statePath = Join-Path $run.FullName "training_state_latest.pt"
    if (-not (Test-Path $configPath)) {
        throw "Missing experiment_config.json: $configPath"
    }
    if (-not (Test-Path $statePath)) {
        throw "Missing training_state_latest.pt: $statePath"
    }
    $cfg = Read-JsonFile -Path $configPath
    $lastEpoch = Get-ResumeEpoch -RunDir $run.FullName
    $extraEpochs = [Math]::Max(0, $TargetEpoch - $lastEpoch)
    $plan.Add([pscustomobject]@{
        stage_index = 4
        stage_name = "final"
        ordinal = $ordinal
        source_ordinal = $sourceOrdinal
        run_id = $run.Name
        train_task = "dc"
        monitor_task = "raw_dc"
        batch_size = [int]$cfg.batch_size
        topk = [int]$cfg.topk
        hist_len = [int]$cfg.hist_len
        lr = [string]$cfg.lr
        weight_decay = [string]$cfg.weight_decay
        hidden_dim = [int]$cfg.hidden_dim
        num_st_blocks = [int]$cfg.num_st_blocks
        time_feature_mode = [string]$cfg.time_feature_mode
        seed = [int]$cfg.seed
        epochs = $extraEpochs
        val_interval = [int]$cfg.val_interval
        pred_horizon = [int]$cfg.pred_horizon
        report_horizons_minutes = [string]$cfg.report_horizons_minutes
        resume_from_epoch = $lastEpoch
        target_epoch = $TargetEpoch
        extra_epochs = $extraEpochs
        run_dir = $run.FullName
    })
    $ordinal += 1
}

$campaignConfig = [pscustomobject]@{
    started_at = (Get-Date -Format o)
    repo_root = $RepoRoot
    source_campaign_root = $SourceCampaignRoot
    campaign_root = $ContinuationRoot
    conda_env_name = $CondaEnvName
    python_exe = $PythonExe
    data_dir = $DataDir
    split_policy = $SplitPolicy
    split_alignment = $SplitAlignment
    batch_size = "from_source_config"
    train_task = "dc"
    monitor_task = "raw_dc"
    plan = "final_continue_to_target"
    excluded_source_ordinals = @($ExcludeOrdinals.Split(",") | ForEach-Object { $_.Trim().PadLeft(3, "0") } | Where-Object { $_ })
    target_epoch = $TargetEpoch
    stage_plan_counts = @{
        final = $plan.Count
        total = $plan.Count
    }
}

Write-JsonFile -Path (Join-Path $ContinuationRoot "campaign_config.json") -Value $campaignConfig
Write-JsonFile -Path (Join-Path $planDir ("s04_final_continue_to{0}.json" -f $TargetEpoch)) -Value @($plan.ToArray())

Write-Banner -Title ("DC FINAL CONTINUE TO {0}" -f $TargetEpoch) -Message ("Continuation root: " + $ContinuationRoot)
Write-Host ("Source campaign: {0}" -f $SourceCampaignRoot)
Write-Host ("Excluded source ordinals: {0}" -f $ExcludeOrdinals) -ForegroundColor Red
Write-Host ("Total continuation runs: {0}" -f $plan.Count) -ForegroundColor Yellow

$records = New-Object System.Collections.Generic.List[object]
$pythonPathBackup = $env:PYTHONPATH
$env:PYTHONPATH = $null

Push-Location $RepoRoot
try {
    foreach ($item in @($plan.ToArray())) {
        Write-JsonFile -Path (Join-Path $item.run_dir "continuation_config.json") -Value $item
        Write-JsonFile -Path $statusPath -Value @{
            active_stage = "final"
            active_run_id = $item.run_id
            active_run_dir = $item.run_dir
            updated_at = (Get-Date -Format o)
            continuation_ordinal = $item.ordinal
            total = $plan.Count
            target_epoch = $TargetEpoch
        }

        Write-Host ""
        Write-Host ("Experiment={0}/{1} | source_final={2} | target_epoch={3}" -f $item.ordinal, $plan.Count, $item.source_ordinal, $TargetEpoch) -ForegroundColor Yellow
        Write-Host ("CURRENT PARAMS: run={0} | topk={1} | hist={2} | lr={3} | wd={4} | hidden={5} | blocks={6} | seed={7}" -f `
                $item.run_id, $item.topk, $item.hist_len, $item.lr, $item.weight_decay, $item.hidden_dim, $item.num_st_blocks, $item.seed) -ForegroundColor Red
        Write-Host ("RESUME: epoch {0} -> {1} ({2} extra epochs)" -f $item.resume_from_epoch, $item.target_epoch, $item.extra_epochs) -ForegroundColor Red
        Write-BestSoFar -Records @($records.ToArray())

        if ($item.extra_epochs -le 0) {
            $record = Get-RecordFromRun -PlanItem $item -Status "already_complete" -ExitCode 0
            $records.Add($record)
            Add-Content -Path $historyPath -Value ($record | ConvertTo-Json -Depth 40 -Compress) -Encoding utf8
            Export-SummaryTsv -Records @($records.ToArray()) -Path $summaryPath
            Write-FinalMarkdown -Records @($records.ToArray()) -Path $finalSummaryPath
            continue
        }

        $trainArgs = Build-TrainArgs -Item $item
        $commandPreview = Join-CommandPreview -Args $trainArgs
        Set-Content -Path (Join-Path $item.run_dir "continuation_command.txt") -Value $commandPreview -Encoding utf8
        Add-Content -Path (Join-Path $item.run_dir "live.stdout.log") -Value ("`n===== CONTINUE TO 100 START {0} =====" -f (Get-Date -Format o)) -Encoding utf8

        $historyBefore = Get-History -RunDir $item.run_dir
        $bestEpoch = Get-BestEpochFromHistory -History $historyBefore
        if ($null -ne $bestEpoch) {
            Write-OrangeHost ("CURRENT BEST EPOCH BEFORE RESUME: epoch={0} | ValRawDC={1}" -f $bestEpoch.epoch, (Format-Value $bestEpoch.metric))
        }

        & conda run --no-capture-output -n $CondaEnvName $PythonExe @trainArgs 2>&1 |
            ForEach-Object {
                $line = [string]$_
                Add-Content -Path (Join-Path $item.run_dir "live.stdout.log") -Value $line -Encoding utf8
                if ($line -match "^\[Ep\s+(\d+)\].*ValRawDC=([0-9.]+)") {
                    $epoch = [int]$Matches[1]
                    $metric = [double]$Matches[2]
                    if ($null -eq $script:StreamBestMetric -or $metric -lt $script:StreamBestMetric) {
                        $script:StreamBestMetric = $metric
                        $script:StreamBestEpoch = $epoch
                        Write-OrangeHost $line
                        Write-OrangeHost ("CURRENT BEST EPOCH: epoch={0} | ValRawDC={1}" -f $epoch, (Format-Value $metric))
                    }
                    else {
                        Write-Host $line -ForegroundColor Yellow
                    }
                }
                elseif ($line -match "^\[Ep\s+\d+\]") {
                    Write-Host $line -ForegroundColor Yellow
                }
                elseif ($line -match "Best|Test|RMSE|ValRawDC") {
                    Write-Host $line -ForegroundColor White
                }
                else {
                    Write-Host $line
                }
            }
        $exitCode = $LASTEXITCODE
        $script:StreamBestMetric = $null
        $script:StreamBestEpoch = $null

        $status = if ($exitCode -eq 0) { "ok" } else { "failed" }
        $record = Get-RecordFromRun -PlanItem $item -Status $status -ExitCode $exitCode
        $records.Add($record)
        Add-Content -Path $historyPath -Value ($record | ConvertTo-Json -Depth 40 -Compress) -Encoding utf8
        Export-SummaryTsv -Records @($records.ToArray()) -Path $summaryPath
        Write-FinalMarkdown -Records @($records.ToArray()) -Path $finalSummaryPath

        if ($status -eq "ok") {
            Write-Host ("COMPLETED: {0} | completed_epoch={1} | best_epoch={2} | ValRawDC={3} | TestRawDC={4}" -f `
                    $record.run_id, $record.completed_epoch, $record.best_epoch, (Format-Value $record.selection_score), (Format-Value $record.test_raw_dc)) -ForegroundColor Cyan
            Write-BestSoFar -Records @($records.ToArray())
        }
        else {
            Write-Host ("FAILED: {0} | exit_code={1}" -f $item.run_id, $exitCode) -ForegroundColor Red
            throw "Continuation run failed: $($item.run_id)"
        }
    }

    Write-JsonFile -Path $statusPath -Value @{
        active_stage = "final"
        active_run_id = $null
        active_run_dir = $null
        completed_at = (Get-Date -Format o)
        summary_path = $summaryPath
        final_summary_path = $finalSummaryPath
        target_epoch = $TargetEpoch
    }
}
finally {
    Pop-Location
    $env:PYTHONPATH = $pythonPathBackup
}

Write-Banner -Title "DC FINAL CONTINUE TO 100 COMPLETE" -Message ("Summary: " + $finalSummaryPath)
