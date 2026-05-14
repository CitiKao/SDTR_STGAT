param(
    [string]$CondaEnvName = "STDR",
    [string]$PythonExe = "python",
    [string]$RepoRoot = "",
    [string]$CampaignRoot = "",
    [string]$DataDir = "data",
    [int]$BatchSize = 32,
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Device = "cuda",
    [ValidateSet("auto", "bf16", "fp32")]
    [string]$Precision = "bf16",
    [int]$NumWorkers = 0,
    [int]$TuningSeed = 42,
    [string]$FinalSeeds = "19,42,123,3407,2026",
    [string]$TopKs = "8,12,16,20,24,32",
    [string]$HistLens = "8,12,16,24",
    [string]$LearningRates = "5e-4,7e-4,1e-3,1.5e-3",
    [string]$WeightDecays = "0,1e-5,1e-4",
    [string]$HiddenDims = "32,48,64",
    [string]$NumStBlocks = "2,3",
    [string]$TimeFeatureModes = "baseline,day_of_month,week_of_month,day_of_month_and_week_of_month",
    [int]$SmokeEpochs = 3,
    [int]$Stage1Epochs = 25,
    [int]$RefineEpochs = 40,
    [int]$FinalEpochs = 60,
    [int]$Stage1Keep = 4,
    [int]$Stage2Keep = 4,
    [int]$Stage3Keep = 3,
    [int]$Stage4Keep = 3,
    [int]$FinalConfigCount = 3,
    [int]$PredHorizon = 4,
    [string]$ReportHorizonsMinutes = "15,30,60",
    [ValidateSet("project_monthly", "benchmark_contiguous")]
    [string]$SplitPolicy = "project_monthly",
    [ValidateSet("none", "day", "week", "month")]
    [string]$SplitAlignment = "none",
    [int]$Stage1ValInterval = 2,
    [int]$RefineValInterval = 1,
    [int]$FinalValInterval = 1,
    [int]$MaxTimeSteps = 0,
    [ValidateSet("all", "smoke", "topk_hist", "optimizer", "structure", "time_features", "final")]
    [string]$StopAfterStage = "all",
    [switch]$DryRun,
    [switch]$ResumeExisting,
    [switch]$Compile,
    [switch]$LoadOnly
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
}

Initialize-Utf8Console

function Write-Banner {
    param(
        [string]$Title,
        [string]$Message = ""
    )
    Write-Host ""
    Write-Host "============================================================"
    Write-Host $Title
    if (-not [string]::IsNullOrWhiteSpace($Message)) {
        Write-Host $Message
    }
    Write-Host "============================================================"
}

function Write-JsonFile {
    param(
        [string]$Path,
        [object]$Value
    )
    $parent = Split-Path -Parent $Path
    if (-not [string]::IsNullOrWhiteSpace($parent)) {
        $null = New-Item -ItemType Directory -Path $parent -Force
    }
    $Value | ConvertTo-Json -Depth 30 | Set-Content -Path $Path -Encoding utf8
}

function Read-JsonFile {
    param([string]$Path)
    return Get-Content -Path $Path -Raw | ConvertFrom-Json
}

function Split-StringList {
    param([string]$Raw)
    return @(
        $Raw.Split(",") |
            ForEach-Object { $_.Trim() } |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    )
}

function Split-IntList {
    param([string]$Raw)
    return @(Split-StringList -Raw $Raw | ForEach-Object { [int]$_ })
}

function Convert-ToNullableDouble {
    param([object]$Value)
    if ($null -eq $Value) {
        return $null
    }
    $text = [string]$Value
    if ([string]::IsNullOrWhiteSpace($text)) {
        return $null
    }
    try {
        return [double]::Parse(
            $text,
            [System.Globalization.NumberStyles]::Float,
            [System.Globalization.CultureInfo]::InvariantCulture
        )
    }
    catch {
        return $null
    }
}

function Get-PropertyValue {
    param(
        [object]$Object,
        [string]$Name
    )
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
    param(
        [object]$Object,
        [string[]]$Path
    )
    $cursor = $Object
    foreach ($part in $Path) {
        $cursor = Get-PropertyValue -Object $cursor -Name $part
        if ($null -eq $cursor) {
            return $null
        }
    }
    return $cursor
}

function Add-Nullable {
    param(
        [Nullable[double]]$A,
        [Nullable[double]]$B
    )
    if ($null -eq $A -or $null -eq $B) {
        return $null
    }
    return [double]$A + [double]$B
}

function Get-SortDouble {
    param([object]$Value)
    $number = Convert-ToNullableDouble -Value $Value
    if ($null -eq $number) {
        return [double]::PositiveInfinity
    }
    return [double]$number
}

function Format-NumberToken {
    param([object]$Value)
    $text = ([string]$Value).Trim().ToLowerInvariant()
    $text = $text.Replace("+", "")
    $text = $text.Replace("-", "m")
    $text = $text.Replace(".", "p")
    return $text
}

function Format-TimeFeatureToken {
    param([string]$Value)
    switch ($Value) {
        "baseline" { return "base" }
        "day_of_month" { return "dom" }
        "week_of_month" { return "wom" }
        "day_of_month_and_week_of_month" { return "domwom" }
        default { return $Value.Replace("_", "") }
    }
}

function New-Experiment {
    param(
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

    $runId = "s{0:D2}_{1}_{2:D3}_tk{3}_h{4}_lr{5}_wd{6}_hd{7}_b{8}_tf{9}_s{10}_ep{11}" -f `
        $StageIndex,
        $StageName,
        $Ordinal,
        $TopK,
        $HistLen,
        (Format-NumberToken -Value $Lr),
        (Format-NumberToken -Value $WeightDecay),
        $HiddenDim,
        $Blocks,
        (Format-TimeFeatureToken -Value $TimeFeatureMode),
        $Seed,
        $Epochs

    return [pscustomobject]@{
        stage_index = $StageIndex
        stage_name = $StageName
        ordinal = $Ordinal
        run_id = $runId
        train_task = "dc"
        monitor_task = "raw_dc"
        batch_size = $BatchSize
        topk = $TopK
        hist_len = $HistLen
        lr = [string]$Lr
        weight_decay = [string]$WeightDecay
        hidden_dim = $HiddenDim
        num_st_blocks = $Blocks
        time_feature_mode = $TimeFeatureMode
        seed = $Seed
        epochs = $Epochs
        val_interval = $ValInterval
        pred_horizon = $PredHorizon
        report_horizons_minutes = $ReportHorizonsMinutes
    }
}

function Get-ExperimentSignature {
    param([object]$Experiment)
    return @(
        [string]$Experiment.topk,
        [string]$Experiment.hist_len,
        [string]$Experiment.lr,
        [string]$Experiment.weight_decay,
        [string]$Experiment.hidden_dim,
        [string]$Experiment.num_st_blocks,
        [string]$Experiment.time_feature_mode,
        [string]$Experiment.seed,
        [string]$Experiment.epochs
    ) -join "|"
}

function Add-UniqueExperiment {
    param(
        [System.Collections.Generic.List[object]]$Plan,
        [System.Collections.Generic.HashSet[string]]$Seen,
        [object]$Experiment
    )
    $signature = Get-ExperimentSignature -Experiment $Experiment
    if ($Seen.Add($signature)) {
        $Plan.Add($Experiment)
    }
}

function Get-MetricsSummary {
    param([string]$MetricsPath)

    if (-not (Test-Path $MetricsPath)) {
        return [pscustomobject]@{
            metrics_status = "missing"
            selection_score = $null
            selection_source = "missing"
            val_raw_dc = $null
            val_gap_rmse = $null
            test_raw_dc = $null
            test_demand_rmse = $null
            test_supply_rmse = $null
            test_gap_rmse = $null
            test_demand_mae = $null
            test_supply_mae = $null
            test_gap_mae = $null
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
    $testGapRmse = Convert-ToNullableDouble (Get-NestedValue -Object $payload -Path @("raw_metrics", "gap", "rmse"))
    $testDemandMae = Convert-ToNullableDouble (Get-NestedValue -Object $payload -Path @("raw_metrics", "demand", "mae"))
    $testSupplyMae = Convert-ToNullableDouble (Get-NestedValue -Object $payload -Path @("raw_metrics", "supply", "mae"))
    $testGapMae = Convert-ToNullableDouble (Get-NestedValue -Object $payload -Path @("raw_metrics", "gap", "mae"))
    $testRawDc = Add-Nullable -A $testDemandRmse -B $testSupplyRmse

    $valDemandRmse = Convert-ToNullableDouble (Get-NestedValue -Object $payload -Path @("val_raw_metrics", "demand", "rmse"))
    $valSupplyRmse = Convert-ToNullableDouble (Get-NestedValue -Object $payload -Path @("val_raw_metrics", "supply", "rmse"))
    $valGapRmse = Convert-ToNullableDouble (Get-NestedValue -Object $payload -Path @("val_raw_metrics", "gap", "rmse"))
    $valRawDc = Add-Nullable -A $valDemandRmse -B $valSupplyRmse

    $selectedMetric = Convert-ToNullableDouble (Get-NestedValue -Object $payload -Path @("selected_checkpoint_metric"))
    $selectedTask = [string](Get-NestedValue -Object $payload -Path @("selected_checkpoint_task"))
    if ($null -eq $valRawDc -and $selectedTask -eq "raw_dc") {
        $valRawDc = $selectedMetric
    }

    $selectionScore = $valRawDc
    $selectionSource = "val_raw_dc"
    if ($null -eq $selectionScore) {
        $selectionScore = $testRawDc
        $selectionSource = "test_raw_dc_fallback"
    }

    return [pscustomobject]@{
        metrics_status = "ok"
        selection_score = $selectionScore
        selection_source = $selectionSource
        val_raw_dc = $valRawDc
        val_gap_rmse = $valGapRmse
        test_raw_dc = $testRawDc
        test_demand_rmse = $testDemandRmse
        test_supply_rmse = $testSupplyRmse
        test_gap_rmse = $testGapRmse
        test_demand_mae = $testDemandMae
        test_supply_mae = $testSupplyMae
        test_gap_mae = $testGapMae
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

function Get-TrainArgs {
    param(
        [object]$Experiment,
        [string]$RunDir
    )
    $trainArgs = @(
        "-u",
        "train_predictor.py",
        "--data-dir", $DataDir,
        "--log-dir", $RunDir,
        "--device", $Device,
        "--precision", $Precision,
        "--batch-size", [string]$Experiment.batch_size,
        "--epochs", [string]$Experiment.epochs,
        "--lr", [string]$Experiment.lr,
        "--optimizer", "adam",
        "--weight-decay", [string]$Experiment.weight_decay,
        "--num-workers", [string]$NumWorkers,
        "--val-interval", [string]$Experiment.val_interval,
        "--log-interval", "1",
        "--hidden-dim", [string]$Experiment.hidden_dim,
        "--num-st-blocks", [string]$Experiment.num_st_blocks,
        "--adaptive-topk", [string]$Experiment.topk,
        "--hist-len", [string]$Experiment.hist_len,
        "--pred-horizon", [string]$Experiment.pred_horizon,
        "--report-horizons-minutes", [string]$Experiment.report_horizons_minutes,
        "--train-task", "dc",
        "--monitor-task", "raw_dc",
        "--split-policy", $SplitPolicy,
        "--split-alignment", $SplitAlignment,
        "--seed", [string]$Experiment.seed,
        "--time-feature-mode", [string]$Experiment.time_feature_mode,
        "--scheduler-factor", "0.5",
        "--scheduler-patience", "10",
        "--scheduler-cooldown", "0",
        "--scheduler-min-lr", "0.0"
    )

    if ($MaxTimeSteps -gt 0) {
        $trainArgs += @("--max-time-steps", [string]$MaxTimeSteps)
    }
    if ($Compile.IsPresent) {
        $trainArgs += @("--compile", "--compile-mode", "reduce-overhead")
    }
    return $trainArgs
}

function Join-CommandPreview {
    param([object[]]$TrainArgs)
    $quoted = @(
        $TrainArgs | ForEach-Object {
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

function New-RunRecord {
    param(
        [object]$Experiment,
        [string]$RunDir,
        [string]$Status,
        [int]$ExitCode,
        [object]$Metrics,
        [string]$CommandPreview
    )
    return [pscustomobject]@{
        stage_index = $Experiment.stage_index
        stage_name = $Experiment.stage_name
        ordinal = $Experiment.ordinal
        run_id = $Experiment.run_id
        status = $Status
        exit_code = $ExitCode
        selection_score = $Metrics.selection_score
        selection_source = $Metrics.selection_source
        val_raw_dc = $Metrics.val_raw_dc
        val_gap_rmse = $Metrics.val_gap_rmse
        test_raw_dc = $Metrics.test_raw_dc
        test_demand_rmse = $Metrics.test_demand_rmse
        test_supply_rmse = $Metrics.test_supply_rmse
        test_gap_rmse = $Metrics.test_gap_rmse
        test_demand_mae = $Metrics.test_demand_mae
        test_supply_mae = $Metrics.test_supply_mae
        test_gap_mae = $Metrics.test_gap_mae
        d_rmse_15 = $Metrics.d_rmse_15
        d_rmse_30 = $Metrics.d_rmse_30
        d_rmse_60 = $Metrics.d_rmse_60
        c_rmse_15 = $Metrics.c_rmse_15
        c_rmse_30 = $Metrics.c_rmse_30
        c_rmse_60 = $Metrics.c_rmse_60
        gap_rmse_15 = $Metrics.gap_rmse_15
        gap_rmse_30 = $Metrics.gap_rmse_30
        gap_rmse_60 = $Metrics.gap_rmse_60
        selected_checkpoint_metric = $Metrics.selected_checkpoint_metric
        selected_checkpoint_task = $Metrics.selected_checkpoint_task
        batch_size = $Experiment.batch_size
        topk = $Experiment.topk
        hist_len = $Experiment.hist_len
        lr = $Experiment.lr
        weight_decay = $Experiment.weight_decay
        hidden_dim = $Experiment.hidden_dim
        num_st_blocks = $Experiment.num_st_blocks
        time_feature_mode = $Experiment.time_feature_mode
        seed = $Experiment.seed
        epochs = $Experiment.epochs
        val_interval = $Experiment.val_interval
        run_dir = $RunDir
        command = $CommandPreview
    }
}

function Invoke-Experiment {
    param(
        [object]$Experiment,
        [string]$RootLogDir,
        [string]$HistoryPath,
        [string]$StatusPath
    )

    $runDir = Join-Path $RootLogDir $Experiment.run_id
    $null = New-Item -ItemType Directory -Path $runDir -Force
    $stdoutPath = Join-Path $runDir "live.stdout.log"
    $metricsPath = Join-Path $runDir "predictor_test_metrics.json"
    $trainArgs = Get-TrainArgs -Experiment $Experiment -RunDir $runDir
    $commandPreview = Join-CommandPreview -TrainArgs $trainArgs
    Set-Content -Path (Join-Path $runDir "command.txt") -Value $commandPreview -Encoding utf8
    Write-JsonFile -Path (Join-Path $runDir "experiment_config.json") -Value $Experiment

    $status = @{
        active_stage = $Experiment.stage_name
        active_run_id = $Experiment.run_id
        active_run_dir = $runDir
        updated_at = (Get-Date -Format o)
    }
    Write-JsonFile -Path $StatusPath -Value $status

    if ($DryRun.IsPresent) {
        $dryScore = 1000000.0 + ([double]$Experiment.stage_index * 10000.0) + [double]$Experiment.ordinal
        $metrics = [pscustomobject]@{
            metrics_status = "dry_run"
            selection_score = $dryScore
            selection_source = "dry_run_order"
            val_raw_dc = $dryScore
            val_gap_rmse = $dryScore
            test_raw_dc = $dryScore
            test_demand_rmse = $null
            test_supply_rmse = $null
            test_gap_rmse = $null
            test_demand_mae = $null
            test_supply_mae = $null
            test_gap_mae = $null
            d_rmse_15 = $null
            d_rmse_30 = $null
            d_rmse_60 = $null
            c_rmse_15 = $null
            c_rmse_30 = $null
            c_rmse_60 = $null
            gap_rmse_15 = $null
            gap_rmse_30 = $null
            gap_rmse_60 = $null
            selected_checkpoint_metric = $dryScore
            selected_checkpoint_task = "raw_dc"
        }
        Set-Content -Path $stdoutPath -Value ("DRY RUN: " + $commandPreview) -Encoding utf8
        $record = New-RunRecord -Experiment $Experiment -RunDir $runDir -Status "dry_run" -ExitCode 0 -Metrics $metrics -CommandPreview $commandPreview
        Add-Content -Path $HistoryPath -Value ($record | ConvertTo-Json -Depth 30 -Compress) -Encoding utf8
        return $record
    }

    if ($ResumeExisting.IsPresent -and (Test-Path $metricsPath)) {
        $metrics = Get-MetricsSummary -MetricsPath $metricsPath
        $record = New-RunRecord -Experiment $Experiment -RunDir $runDir -Status "resumed" -ExitCode 0 -Metrics $metrics -CommandPreview $commandPreview
        Add-Content -Path $HistoryPath -Value ($record | ConvertTo-Json -Depth 30 -Compress) -Encoding utf8
        return $record
    }

    Write-Banner -Title ("Starting " + $Experiment.run_id) -Message ("Run dir: " + $runDir)
    if (Test-Path $stdoutPath) {
        Remove-Item -LiteralPath $stdoutPath -Force
    }
    & conda run --no-capture-output -n $CondaEnvName $PythonExe @trainArgs 2>&1 |
        ForEach-Object {
            $line = [string]$_
            Add-Content -Path $stdoutPath -Value $line -Encoding utf8
            Write-Host $line
        }
    $exitCode = $LASTEXITCODE

    $metrics = Get-MetricsSummary -MetricsPath $metricsPath
    $runStatus = if ($exitCode -eq 0 -and $metrics.metrics_status -eq "ok") { "ok" } else { "failed" }
    $record = New-RunRecord -Experiment $Experiment -RunDir $runDir -Status $runStatus -ExitCode $exitCode -Metrics $metrics -CommandPreview $commandPreview
    Add-Content -Path $HistoryPath -Value ($record | ConvertTo-Json -Depth 30 -Compress) -Encoding utf8
    return $record
}

function Select-TopRecords {
    param(
        [object[]]$Records,
        [int]$Count
    )
    return @(
        $Records |
            Where-Object { $_.status -in @("ok", "resumed", "dry_run") } |
            Sort-Object `
                @{ Expression = { Get-SortDouble $_.selection_score }; Ascending = $true }, `
                @{ Expression = { Get-SortDouble $_.val_gap_rmse }; Ascending = $true }, `
                @{ Expression = { Get-SortDouble $_.test_raw_dc }; Ascending = $true }, `
                @{ Expression = { [string]$_.run_id }; Ascending = $true } |
            Select-Object -First $Count
    )
}

function Select-TopConfigs {
    param(
        [object[]]$Runs,
        [int]$Limit = 3
    )
    $normalized = @(
        foreach ($run in $Runs) {
            $selectionScore = Get-PropertyValue -Object $run -Name "selection_score"
            if ($null -eq $selectionScore) {
                $selectionScore = Get-PropertyValue -Object $run -Name "val_raw_dc"
            }
            if ($null -eq $selectionScore) {
                $selectionScore = Get-PropertyValue -Object $run -Name "test_raw_dc"
            }
            [pscustomobject]@{
                run_id = Get-PropertyValue -Object $run -Name "run_id"
                status = "ok"
                selection_score = $selectionScore
                val_gap_rmse = Get-PropertyValue -Object $run -Name "val_gap_rmse"
                test_raw_dc = Get-PropertyValue -Object $run -Name "test_raw_dc"
                topk = Get-PropertyValue -Object $run -Name "topk"
                hist_len = Get-PropertyValue -Object $run -Name "hist_len"
                lr = Get-PropertyValue -Object $run -Name "lr"
                weight_decay = Get-PropertyValue -Object $run -Name "weight_decay"
                hidden_dim = Get-PropertyValue -Object $run -Name "hidden_dim"
                num_st_blocks = Get-PropertyValue -Object $run -Name "num_st_blocks"
                time_feature_mode = Get-PropertyValue -Object $run -Name "time_feature_mode"
                seed = Get-PropertyValue -Object $run -Name "seed"
                original = $run
            }
        }
    )
    $selected = Select-TopRecords -Records $normalized -Count $Limit
    return @($selected | ForEach-Object { $_.original })
}

function Export-SummaryTsv {
    param(
        [object[]]$Records,
        [string]$Path
    )
    $columns = @(
        "stage_name", "run_id", "status", "selection_score", "selection_source",
        "val_raw_dc", "val_gap_rmse", "test_raw_dc",
        "test_demand_rmse", "test_supply_rmse", "test_gap_rmse",
        "test_demand_mae", "test_supply_mae", "test_gap_mae",
        "d_rmse_15", "d_rmse_30", "d_rmse_60",
        "c_rmse_15", "c_rmse_30", "c_rmse_60",
        "gap_rmse_15", "gap_rmse_30", "gap_rmse_60",
        "topk", "hist_len", "lr", "weight_decay", "hidden_dim", "num_st_blocks",
        "time_feature_mode", "seed", "epochs", "val_interval", "run_dir"
    )
    $lines = New-Object System.Collections.Generic.List[string]
    $lines.Add(($columns -join "`t"))
    foreach ($record in $Records) {
        $values = foreach ($column in $columns) {
            $value = Get-PropertyValue -Object $record -Name $column
            if ($null -eq $value) { "" } else { ([string]$value).Replace("`t", " ") }
        }
        $lines.Add(($values -join "`t"))
    }
    Set-Content -Path $Path -Value $lines -Encoding utf8
}

function Write-FinalMarkdown {
    param(
        [object[]]$Records,
        [string]$Path
    )
    $eligibleRecords = @($Records | Where-Object { $_.stage_name -eq "final" })
    if ($eligibleRecords.Count -eq 0) {
        $eligibleRecords = $Records
    }
    $top = Select-TopRecords -Records $eligibleRecords -Count 20
    $lines = New-Object System.Collections.Generic.List[string]
    $lines.Add("# DC Autopilot Summary")
    $lines.Add("")
    $lines.Add("- Selection metric: validation raw DC = demand RMSE + supply RMSE when available.")
    $lines.Add("- Tie-breaker: validation Gap RMSE, then test raw DC.")
    $bt = [char]96
    $lines.Add("- Batch size: $bt$BatchSize$bt")
    $lines.Add("- Prediction/report horizons: $bt$PredHorizon$bt steps, $bt$ReportHorizonsMinutes$bt minutes")
    $lines.Add("")
    $lines.Add("| Rank | Stage | Run | Sel. raw DC | Test raw DC | D RMSE | C RMSE | Gap RMSE | TopK | Hist | LR | WD | Hidden | Blocks | Time | Seed |")
    $lines.Add("| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | --- | ---: |")
    $rank = 1
    foreach ($record in $top) {
        $lines.Add((
            "| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} | {8} | {9} | {10} | {11} | {12} | {13} | {14} | {15} |" -f `
                $rank,
                $record.stage_name,
                $record.run_id,
                $record.selection_score,
                $record.test_raw_dc,
                $record.test_demand_rmse,
                $record.test_supply_rmse,
                $record.test_gap_rmse,
                $record.topk,
                $record.hist_len,
                $record.lr,
                $record.weight_decay,
                $record.hidden_dim,
                $record.num_st_blocks,
                $record.time_feature_mode,
                $record.seed
        ))
        $rank += 1
    }
    Set-Content -Path $Path -Value $lines -Encoding utf8
}

function Set-CampaignComplete {
    param(
        [string]$StatusPath,
        [string]$StageName,
        [string]$SummaryPath,
        [string]$FinalSummaryPath
    )
    Write-JsonFile -Path $StatusPath -Value @{
        active_stage = $StageName
        active_run_id = $null
        active_run_dir = $null
        completed_at = (Get-Date -Format o)
        dry_run = [bool]$DryRun.IsPresent
        summary_path = $SummaryPath
        final_summary_path = $FinalSummaryPath
    }
}

function Build-SmokePlan {
    $plan = [System.Collections.Generic.List[object]]::new()
    $seen = [System.Collections.Generic.HashSet[string]]::new()
    Add-UniqueExperiment -Plan $plan -Seen $seen -Experiment (New-Experiment `
        -StageIndex 0 -StageName "smoke" -Ordinal 1 `
        -TopK 16 -HistLen 12 -Lr "1e-3" -WeightDecay "1e-5" `
        -HiddenDim 32 -Blocks 2 -TimeFeatureMode "baseline" `
        -Seed $TuningSeed -Epochs $SmokeEpochs -ValInterval 1)
    return @($plan.ToArray())
}

function Build-TopkHistPlan {
    $plan = [System.Collections.Generic.List[object]]::new()
    $seen = [System.Collections.Generic.HashSet[string]]::new()
    $ordinal = 1
    foreach ($topk in (Split-IntList -Raw $TopKs)) {
        foreach ($histLen in (Split-IntList -Raw $HistLens)) {
            Add-UniqueExperiment -Plan $plan -Seen $seen -Experiment (New-Experiment `
                -StageIndex 1 -StageName "topk_hist" -Ordinal $ordinal `
                -TopK $topk -HistLen $histLen -Lr "1e-3" -WeightDecay "1e-5" `
                -HiddenDim 32 -Blocks 2 -TimeFeatureMode "baseline" `
                -Seed $TuningSeed -Epochs $Stage1Epochs -ValInterval $Stage1ValInterval)
            $ordinal += 1
        }
    }
    return @($plan.ToArray())
}

function Build-OptimizerPlan {
    param([object[]]$Parents)
    $plan = [System.Collections.Generic.List[object]]::new()
    $seen = [System.Collections.Generic.HashSet[string]]::new()
    $ordinal = 1
    foreach ($parent in $Parents) {
        foreach ($lr in (Split-StringList -Raw $LearningRates)) {
            foreach ($wd in (Split-StringList -Raw $WeightDecays)) {
                Add-UniqueExperiment -Plan $plan -Seen $seen -Experiment (New-Experiment `
                    -StageIndex 2 -StageName "optimizer" -Ordinal $ordinal `
                    -TopK ([int]$parent.topk) -HistLen ([int]$parent.hist_len) `
                    -Lr $lr -WeightDecay $wd `
                    -HiddenDim ([int]$parent.hidden_dim) -Blocks ([int]$parent.num_st_blocks) `
                    -TimeFeatureMode ([string]$parent.time_feature_mode) `
                    -Seed $TuningSeed -Epochs $RefineEpochs -ValInterval $RefineValInterval)
                $ordinal += 1
            }
        }
    }
    return @($plan.ToArray())
}

function Build-StructurePlan {
    param([object[]]$Parents)
    $plan = [System.Collections.Generic.List[object]]::new()
    $seen = [System.Collections.Generic.HashSet[string]]::new()
    $ordinal = 1
    foreach ($parent in $Parents) {
        foreach ($hidden in (Split-IntList -Raw $HiddenDims)) {
            foreach ($blocks in (Split-IntList -Raw $NumStBlocks)) {
                Add-UniqueExperiment -Plan $plan -Seen $seen -Experiment (New-Experiment `
                    -StageIndex 3 -StageName "structure" -Ordinal $ordinal `
                    -TopK ([int]$parent.topk) -HistLen ([int]$parent.hist_len) `
                    -Lr ([string]$parent.lr) -WeightDecay ([string]$parent.weight_decay) `
                    -HiddenDim $hidden -Blocks $blocks `
                    -TimeFeatureMode ([string]$parent.time_feature_mode) `
                    -Seed $TuningSeed -Epochs $RefineEpochs -ValInterval $RefineValInterval)
                $ordinal += 1
            }
        }
    }
    return @($plan.ToArray())
}

function Build-TimeFeaturePlan {
    param([object[]]$Parents)
    $plan = [System.Collections.Generic.List[object]]::new()
    $seen = [System.Collections.Generic.HashSet[string]]::new()
    $ordinal = 1
    foreach ($parent in $Parents) {
        foreach ($mode in (Split-StringList -Raw $TimeFeatureModes)) {
            Add-UniqueExperiment -Plan $plan -Seen $seen -Experiment (New-Experiment `
                -StageIndex 4 -StageName "time_features" -Ordinal $ordinal `
                -TopK ([int]$parent.topk) -HistLen ([int]$parent.hist_len) `
                -Lr ([string]$parent.lr) -WeightDecay ([string]$parent.weight_decay) `
                -HiddenDim ([int]$parent.hidden_dim) -Blocks ([int]$parent.num_st_blocks) `
                -TimeFeatureMode $mode `
                -Seed $TuningSeed -Epochs $RefineEpochs -ValInterval $RefineValInterval)
            $ordinal += 1
        }
    }
    return @($plan.ToArray())
}

function Build-FinalPlan {
    param([object[]]$Parents)
    $plan = [System.Collections.Generic.List[object]]::new()
    $seen = [System.Collections.Generic.HashSet[string]]::new()
    $ordinal = 1
    $seedValues = Split-IntList -Raw $FinalSeeds
    foreach ($parent in ($Parents | Select-Object -First $FinalConfigCount)) {
        foreach ($seed in $seedValues) {
            Add-UniqueExperiment -Plan $plan -Seen $seen -Experiment (New-Experiment `
                -StageIndex 5 -StageName "final" -Ordinal $ordinal `
                -TopK ([int]$parent.topk) -HistLen ([int]$parent.hist_len) `
                -Lr ([string]$parent.lr) -WeightDecay ([string]$parent.weight_decay) `
                -HiddenDim ([int]$parent.hidden_dim) -Blocks ([int]$parent.num_st_blocks) `
                -TimeFeatureMode ([string]$parent.time_feature_mode) `
                -Seed $seed -Epochs $FinalEpochs -ValInterval $FinalValInterval)
            $ordinal += 1
        }
    }
    return @($plan.ToArray())
}

function Invoke-Stage {
    param(
        [string]$StageName,
        [object[]]$Plan,
        [string]$RootLogDir,
        [string]$PlanDir,
        [string]$HistoryPath,
        [string]$StatusPath
    )
    if ($Plan.Count -eq 0) {
        throw "Stage $StageName has no experiments to run."
    }

    $stageIndex = [int]$Plan[0].stage_index
    $planPath = Join-Path $PlanDir ("s{0:D2}_{1}.json" -f $stageIndex, $StageName)
    $resultsPath = Join-Path $RootLogDir ("stage_s{0:D2}_{1}_results.json" -f $stageIndex, $StageName)
    Write-JsonFile -Path $planPath -Value $Plan

    Write-Banner -Title ("STAGE " + $StageName) -Message ("Runs: " + $Plan.Count + " | plan: " + $planPath)
    $records = New-Object System.Collections.Generic.List[object]
    foreach ($experiment in $Plan) {
        $record = Invoke-Experiment -Experiment $experiment -RootLogDir $RootLogDir -HistoryPath $HistoryPath -StatusPath $StatusPath
        $records.Add($record)
        Write-JsonFile -Path $resultsPath -Value @($records.ToArray())
    }
    return @($records.ToArray())
}

if ($LoadOnly.IsPresent) {
    return
}

if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $RepoRoot = Split-Path -Parent $scriptDir
}
$RepoRoot = (Resolve-Path $RepoRoot).Path

if ([string]::IsNullOrWhiteSpace($CampaignRoot)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $CampaignRoot = Join-Path $RepoRoot ("runs\local_4090_dc\autopilot_sessions\DC_autopilot_" + $stamp)
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
    dry_run = [bool]$DryRun.IsPresent
    resume_existing = [bool]$ResumeExisting.IsPresent
    batch_size = $BatchSize
    train_task = "dc"
    monitor_task = "raw_dc"
    pred_horizon = $PredHorizon
    report_horizons_minutes = $ReportHorizonsMinutes
    stage_epochs = @{
        smoke = $SmokeEpochs
        topk_hist = $Stage1Epochs
        optimizer = $RefineEpochs
        structure = $RefineEpochs
        time_features = $RefineEpochs
        final = $FinalEpochs
    }
    keep_counts = @{
        stage1 = $Stage1Keep
        stage2 = $Stage2Keep
        stage3 = $Stage3Keep
        stage4 = $Stage4Keep
        final_config_count = $FinalConfigCount
    }
}
Write-JsonFile -Path (Join-Path $CampaignRoot "campaign_config.json") -Value $campaignConfig

Write-Banner -Title "DC AUTOPILOT" -Message ("Campaign root: " + $CampaignRoot)

$allRecords = New-Object System.Collections.Generic.List[object]
$env:PYTHONUNBUFFERED = "1"
$pythonPathBackup = $env:PYTHONPATH
$env:PYTHONPATH = $null

Push-Location $RepoRoot
try {
    $smokeRecords = Invoke-Stage -StageName "smoke" -Plan (Build-SmokePlan) -RootLogDir $CampaignRoot -PlanDir $planDir -HistoryPath $historyPath -StatusPath $statusPath
    foreach ($record in $smokeRecords) { $allRecords.Add($record) }
    Export-SummaryTsv -Records @($allRecords.ToArray()) -Path $summaryPath
    if ((Select-TopRecords -Records $smokeRecords -Count 1).Count -eq 0) {
        throw "Smoke stage produced no successful run."
    }
    if ($StopAfterStage -eq "smoke") {
        Set-CampaignComplete -StatusPath $statusPath -StageName "smoke" -SummaryPath $summaryPath -FinalSummaryPath $finalSummaryPath
        return
    }

    $stage1Records = Invoke-Stage -StageName "topk_hist" -Plan (Build-TopkHistPlan) -RootLogDir $CampaignRoot -PlanDir $planDir -HistoryPath $historyPath -StatusPath $statusPath
    foreach ($record in $stage1Records) { $allRecords.Add($record) }
    $selected = Select-TopRecords -Records $stage1Records -Count $Stage1Keep
    Write-JsonFile -Path (Join-Path $CampaignRoot "selected_after_topk_hist.json") -Value $selected
    Export-SummaryTsv -Records @($allRecords.ToArray()) -Path $summaryPath
    if ($selected.Count -eq 0) { throw "topk_hist stage produced no successful run." }
    if ($StopAfterStage -eq "topk_hist") {
        Set-CampaignComplete -StatusPath $statusPath -StageName "topk_hist" -SummaryPath $summaryPath -FinalSummaryPath $finalSummaryPath
        return
    }

    $stage2Records = Invoke-Stage -StageName "optimizer" -Plan (Build-OptimizerPlan -Parents $selected) -RootLogDir $CampaignRoot -PlanDir $planDir -HistoryPath $historyPath -StatusPath $statusPath
    foreach ($record in $stage2Records) { $allRecords.Add($record) }
    $selected = Select-TopRecords -Records $stage2Records -Count $Stage2Keep
    Write-JsonFile -Path (Join-Path $CampaignRoot "selected_after_optimizer.json") -Value $selected
    Export-SummaryTsv -Records @($allRecords.ToArray()) -Path $summaryPath
    if ($selected.Count -eq 0) { throw "optimizer stage produced no successful run." }
    if ($StopAfterStage -eq "optimizer") {
        Set-CampaignComplete -StatusPath $statusPath -StageName "optimizer" -SummaryPath $summaryPath -FinalSummaryPath $finalSummaryPath
        return
    }

    $stage3Records = Invoke-Stage -StageName "structure" -Plan (Build-StructurePlan -Parents $selected) -RootLogDir $CampaignRoot -PlanDir $planDir -HistoryPath $historyPath -StatusPath $statusPath
    foreach ($record in $stage3Records) { $allRecords.Add($record) }
    $selected = Select-TopRecords -Records $stage3Records -Count $Stage3Keep
    Write-JsonFile -Path (Join-Path $CampaignRoot "selected_after_structure.json") -Value $selected
    Export-SummaryTsv -Records @($allRecords.ToArray()) -Path $summaryPath
    if ($selected.Count -eq 0) { throw "structure stage produced no successful run." }
    if ($StopAfterStage -eq "structure") {
        Set-CampaignComplete -StatusPath $statusPath -StageName "structure" -SummaryPath $summaryPath -FinalSummaryPath $finalSummaryPath
        return
    }

    $stage4Records = Invoke-Stage -StageName "time_features" -Plan (Build-TimeFeaturePlan -Parents $selected) -RootLogDir $CampaignRoot -PlanDir $planDir -HistoryPath $historyPath -StatusPath $statusPath
    foreach ($record in $stage4Records) { $allRecords.Add($record) }
    $selected = Select-TopRecords -Records $stage4Records -Count $Stage4Keep
    Write-JsonFile -Path (Join-Path $CampaignRoot "selected_after_time_features.json") -Value $selected
    Export-SummaryTsv -Records @($allRecords.ToArray()) -Path $summaryPath
    if ($selected.Count -eq 0) { throw "time_features stage produced no successful run." }
    if ($StopAfterStage -eq "time_features") {
        Set-CampaignComplete -StatusPath $statusPath -StageName "time_features" -SummaryPath $summaryPath -FinalSummaryPath $finalSummaryPath
        return
    }

    $finalRecords = Invoke-Stage -StageName "final" -Plan (Build-FinalPlan -Parents $selected) -RootLogDir $CampaignRoot -PlanDir $planDir -HistoryPath $historyPath -StatusPath $statusPath
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

Write-Banner -Title "DC AUTOPILOT COMPLETE" -Message ("Summary: " + $summaryPath)
