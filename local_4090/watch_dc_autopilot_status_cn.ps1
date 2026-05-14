param(
    [string]$CampaignRoot = "",
    [int]$RefreshSeconds = 5,
    [int]$TailLines = 18
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

function Read-JsonIfExists {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        return $null
    }
    try {
        return Get-Content -Path $Path -Raw | ConvertFrom-Json
    }
    catch {
        return $null
    }
}

function Read-HistoryRecords {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        return @()
    }
    $records = New-Object System.Collections.Generic.List[object]
    foreach ($line in Get-Content -Path $Path -ErrorAction SilentlyContinue) {
        if ([string]::IsNullOrWhiteSpace($line)) {
            continue
        }
        try {
            $records.Add(($line | ConvertFrom-Json))
        }
        catch {
        }
    }
    return @($records.ToArray())
}

function Get-SortValue {
    param([object]$Value)
    if ($null -eq $Value) {
        return [double]::PositiveInfinity
    }
    try {
        return [double]$Value
    }
    catch {
        return [double]::PositiveInfinity
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

function Write-OrangeHost {
    param([string]$Text)
    $esc = [char]27
    Write-Host ("{0}[38;5;208m{1}{0}[0m" -f $esc, $Text)
}

function Write-Title {
    param([string]$Text)
    Write-Host ""
    Write-Host $Text -ForegroundColor Cyan
    Write-Host ("-" * 72) -ForegroundColor DarkCyan
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

function Get-CampaignTotalRuns {
    param([object]$CampaignConfig)
    $total = Get-ObjectProperty -Object (Get-ObjectProperty -Object $CampaignConfig -Name "stage_plan_counts") -Name "total"
    if ($null -ne $total) {
        try {
            return [int]$total
        }
        catch {
        }
    }
    return $null
}

function Get-StageOffset {
    param([object]$StageCounts, [string]$StageName)
    if ($null -eq $StageCounts -or [string]::IsNullOrWhiteSpace($StageName)) {
        return $null
    }
    $stageOrder = @(
        "smoke",
        "topk_hist",
        "optimizer",
        "capacity_time",
        "structure",
        "time_features",
        "final"
    )
    $offset = 0
    foreach ($name in $stageOrder) {
        if ($name -eq $StageName) {
            return $offset
        }
        $count = Get-ObjectProperty -Object $StageCounts -Name $name
        if ($null -ne $count) {
            try {
                $offset += [int]$count
            }
            catch {
            }
        }
    }
    return $null
}

function Get-ExperimentProgress {
    param([object]$CampaignConfig, [object]$ActiveConfig, [object]$Status)
    $total = Get-CampaignTotalRuns -CampaignConfig $CampaignConfig
    if ($null -eq $total) {
        return $null
    }

    $stageName = [string](Get-ObjectProperty -Object $ActiveConfig -Name "stage_name")
    if ([string]::IsNullOrWhiteSpace($stageName)) {
        $stageName = [string](Get-ObjectProperty -Object $Status -Name "active_stage")
    }
    $ordinal = Get-ObjectProperty -Object $ActiveConfig -Name "ordinal"
    $offset = Get-StageOffset -StageCounts (Get-ObjectProperty -Object $CampaignConfig -Name "stage_plan_counts") -StageName $stageName
    if ($null -eq $offset -or $null -eq $ordinal) {
        return $null
    }
    try {
        return [pscustomobject]@{
            current = $offset + [int]$ordinal
            total = [int]$total
            stage = $stageName
        }
    }
    catch {
        return $null
    }
}

function Write-ExperimentProgress {
    param([object]$Progress)
    if ($null -eq $Progress) {
        Write-Host "Experiment progress: waiting for active experiment config ..." -ForegroundColor Yellow
        return
    }
    Write-Host (
        "Experiment={0}/{1} | Stage={2}" -f `
            $Progress.current,
            $Progress.total,
            $Progress.stage
    ) -ForegroundColor Yellow
}

function Write-CurrentConfig {
    param([object]$Config, [string]$RunId)
    Write-Title "CURRENT RUN PARAMS (red)"
    if ($null -eq $Config) {
        Write-Host "Waiting for active run experiment_config.json ..." -ForegroundColor Red
        return
    }
    Write-Host ("Run: {0}" -f $RunId) -ForegroundColor Red
    Write-Host (
        "topk={0} | hist={1} | lr={2} | wd={3} | hidden={4} | blocks={5} | time={6} | seed={7} | epochs={8} | val_interval={9}" -f `
            $Config.topk,
            $Config.hist_len,
            $Config.lr,
            $Config.weight_decay,
            $Config.hidden_dim,
            $Config.num_st_blocks,
            $Config.time_feature_mode,
            $Config.seed,
            $Config.epochs,
            $Config.val_interval
    ) -ForegroundColor Red
    $resumeFromEpoch = Get-ObjectProperty -Object $Config -Name "resume_from_epoch"
    $targetEpoch = Get-ObjectProperty -Object $Config -Name "target_epoch"
    $extraEpochs = Get-ObjectProperty -Object $Config -Name "extra_epochs"
    if ($null -ne $targetEpoch) {
        Write-Host (
            "resume_from_epoch={0} | target_epoch={1} | extra_epochs={2}" -f `
                $resumeFromEpoch,
                $targetEpoch,
                $extraEpochs
        ) -ForegroundColor Red
    }
}

function Write-BestRecord {
    param([object]$Best)
    Write-Title "BEST CONFIG SO FAR (green)"
    if ($null -eq $Best) {
        Write-Host "No completed run yet." -ForegroundColor Green
        return
    }
    Write-Host ("Run: {0}" -f $Best.run_id) -ForegroundColor Green
    Write-Host (
        "SelRawDC={0} | ValRawDC={1} | TestRawDC={2} | D_RMSE={3} | C_RMSE={4} | Gap_RMSE={5}" -f `
            (Format-Value $Best.selection_score),
            (Format-Value $Best.val_raw_dc),
            (Format-Value $Best.test_raw_dc),
            (Format-Value $Best.test_demand_rmse),
            (Format-Value $Best.test_supply_rmse),
            (Format-Value $Best.test_gap_rmse)
    ) -ForegroundColor Green
    Write-Host (
        "topk={0} | hist={1} | lr={2} | wd={3} | hidden={4} | blocks={5} | time={6} | seed={7} | epochs={8}" -f `
            $Best.topk,
            $Best.hist_len,
            $Best.lr,
            $Best.weight_decay,
            $Best.hidden_dim,
            $Best.num_st_blocks,
            $Best.time_feature_mode,
            $Best.seed,
            $Best.epochs
    ) -ForegroundColor Green
    Write-Host (
        "15/30/60 RMSE: D=({0}/{1}/{2}) C=({3}/{4}/{5}) Gap=({6}/{7}/{8})" -f `
            (Format-Value $Best.d_rmse_15),
            (Format-Value $Best.d_rmse_30),
            (Format-Value $Best.d_rmse_60),
            (Format-Value $Best.c_rmse_15),
            (Format-Value $Best.c_rmse_30),
            (Format-Value $Best.c_rmse_60),
            (Format-Value $Best.gap_rmse_15),
            (Format-Value $Best.gap_rmse_30),
            (Format-Value $Best.gap_rmse_60)
    ) -ForegroundColor Green
}

function Get-BestEpochFromHistoryRecords {
    param([object[]]$History)
    $best = $null
    foreach ($record in $History) {
        $metric = Convert-ToNullableDouble $record.val_raw_dc
        if ($null -eq $metric) {
            $metric = Convert-ToNullableDouble $record.val_dc
        }
        if ($null -eq $metric) {
            continue
        }
        if ($null -eq $best -or $metric -lt $best.metric) {
            $best = [pscustomobject]@{
                epoch = [int]$record.epoch
                metric = $metric
                source = "predictor_log.json"
                line = $null
            }
        }
    }
    return $best
}

function Get-BestEpochFromLiveLog {
    param([string]$StdoutPath)
    if (-not (Test-Path $StdoutPath)) {
        return $null
    }
    $best = $null
    foreach ($line in Get-Content -Path $StdoutPath -Encoding UTF8 -ErrorAction SilentlyContinue) {
        if ($line -notmatch "^\[Ep\s+(\d+)\].*ValRawDC=([0-9.]+)") {
            continue
        }
        $metric = Convert-ToNullableDouble $Matches[2]
        if ($null -eq $metric) {
            continue
        }
        if ($null -eq $best -or $metric -lt $best.metric) {
            $best = [pscustomobject]@{
                epoch = [int]$Matches[1]
                metric = $metric
                source = "live.stdout.log"
                line = $line
            }
        }
    }
    return $best
}

function Get-CurrentBestEpoch {
    param([string]$RunDir)
    if ([string]::IsNullOrWhiteSpace($RunDir)) {
        return $null
    }
    $historyPath = Join-Path $RunDir "predictor_log.json"
    $history = Read-JsonIfExists -Path $historyPath
    if ($history -is [array]) {
        $bestFromHistory = Get-BestEpochFromHistoryRecords -History $history
        if ($null -ne $bestFromHistory) {
            return $bestFromHistory
        }
    }
    return Get-BestEpochFromLiveLog -StdoutPath (Join-Path $RunDir "live.stdout.log")
}

function Write-CurrentBestEpoch {
    param([object]$BestEpoch)
    Write-Title "CURRENT BEST EPOCH (orange)"
    if ($null -eq $BestEpoch) {
        Write-OrangeHost "Waiting for the first validated epoch ..."
        return
    }
    Write-OrangeHost (
        "Best epoch so far: epoch={0} | ValRawDC/metric={1} | source={2}" -f `
            $BestEpoch.epoch,
            (Format-Value $BestEpoch.metric),
            $BestEpoch.source
    )
}

function Write-LiveLog {
    param([string]$RunDir, [object]$BestEpoch)
    Write-Title "LIVE TRAINING OUTPUT"
    if ([string]::IsNullOrWhiteSpace($RunDir)) {
        Write-Host "Waiting for next active run ..."
        return
    }
    $stdoutPath = Join-Path $RunDir "live.stdout.log"
    if (-not (Test-Path $stdoutPath)) {
        Write-Host ("Waiting for log file: {0}" -f $stdoutPath)
        return
    }
    foreach ($line in Get-Content -Path $stdoutPath -Encoding UTF8 -Tail $TailLines -ErrorAction SilentlyContinue) {
        if ($line -match "^\[Ep\s+(\d+)\]" -and $null -ne $BestEpoch -and [int]$Matches[1] -eq [int]$BestEpoch.epoch) {
            Write-OrangeHost $line
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
}

if ([string]::IsNullOrWhiteSpace($CampaignRoot)) {
    $root = Join-Path (Get-Location) "runs\local_4090_dc\autopilot_sessions"
    $latest = Get-ChildItem -Path $root -Directory -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -eq $latest) {
        throw "CampaignRoot is required because no autopilot session was found."
    }
    $CampaignRoot = $latest.FullName
}

while ($true) {
    Clear-Host
    $statusPath = Join-Path $CampaignRoot "queue_status.json"
    $historyPath = Join-Path $CampaignRoot "queue_history.jsonl"
    $campaignConfigPath = Join-Path $CampaignRoot "campaign_config.json"
    $status = Read-JsonIfExists -Path $statusPath
    $campaignConfig = Read-JsonIfExists -Path $campaignConfigPath
    $records = Read-HistoryRecords -Path $historyPath
    $finished = @($records | Where-Object { $_.status -in @("ok", "resumed") })
    $failed = @($records | Where-Object { $_.status -eq "failed" })
    $best = @(
        $finished |
            Sort-Object `
                @{ Expression = { Get-SortValue $_.selection_score }; Ascending = $true }, `
                @{ Expression = { Get-SortValue $_.val_gap_rmse }; Ascending = $true }, `
                @{ Expression = { [string]$_.run_id }; Ascending = $true } |
            Select-Object -First 1
    )

    Write-Host ("DC Autopilot Monitor | {0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss")) -ForegroundColor Cyan
    Write-Host ("Session: {0}" -f $CampaignRoot)
    if ($null -eq $status) {
        Write-Host "Status: waiting for queue_status.json ..." -ForegroundColor DarkYellow
        Start-Sleep -Seconds $RefreshSeconds
        continue
    }

    Write-Host (
        "Stage={0} | Finished={1} | Failed={2} | Active={3}" -f `
            $status.active_stage,
            $finished.Count,
            $failed.Count,
            $status.active_run_id
    )

    $activeRunDir = [string]$status.active_run_dir
    $activeConfig = $null
    if (-not [string]::IsNullOrWhiteSpace($activeRunDir)) {
        $activeConfig = Read-JsonIfExists -Path (Join-Path $activeRunDir "continuation_config.json")
        if ($null -eq $activeConfig) {
            $activeConfig = Read-JsonIfExists -Path (Join-Path $activeRunDir "experiment_config.json")
        }
    }

    Write-ExperimentProgress -Progress (Get-ExperimentProgress -CampaignConfig $campaignConfig -ActiveConfig $activeConfig -Status $status)
    Write-CurrentConfig -Config $activeConfig -RunId ([string]$status.active_run_id)
    Write-BestRecord -Best (@($best)[0])
    $activeBestEpoch = Get-CurrentBestEpoch -RunDir $activeRunDir
    Write-CurrentBestEpoch -BestEpoch $activeBestEpoch
    Write-LiveLog -RunDir $activeRunDir -BestEpoch $activeBestEpoch

    if ($null -ne $status.completed_at -and [string]::IsNullOrWhiteSpace([string]$status.active_run_id)) {
        Write-Title "STATUS"
        Write-Host ("Completed at: {0}" -f $status.completed_at) -ForegroundColor Green
        Write-Host ("Summary: {0}" -f $status.summary_path) -ForegroundColor Green
        Write-Host ("Final: {0}" -f $status.final_summary_path) -ForegroundColor Green
    }

    Start-Sleep -Seconds $RefreshSeconds
}
