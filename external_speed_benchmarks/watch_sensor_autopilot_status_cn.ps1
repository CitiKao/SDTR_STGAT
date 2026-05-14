param(
    [string]$CampaignRoot = "",
    [int]$IntervalSeconds = 15
)

$ErrorActionPreference = "Continue"

if ([string]::IsNullOrWhiteSpace($CampaignRoot)) {
    throw "CampaignRoot is required."
}

function Format-Decimal {
    param($Value)
    if ($null -eq $Value) {
        return "n/a"
    }
    try {
        return ([double]$Value).ToString("F4")
    } catch {
        return "n/a"
    }
}

function Get-LatestBatchDir {
    param([string]$Root)
    return Get-ChildItem -Path $Root -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like "batch_*" } |
        Sort-Object Name |
        Select-Object -Last 1
}

function Get-StageDescription {
    param([string]$StageName)
    switch ([string]$StageName) {
        "route_bestshot" { return "优先突破阶段：先直接开跑实际路径距离 + seed19，快速验证最有希望的单条配置。" }
        "paired_seed_matrix" { return "配对矩阵阶段：并排比较 haversine/map_route 在 seed19/31/42 上的表现，判断哪种距离语义更稳更强。" }
        "baseline" { return "基线阶段：先复现旧历史最佳风格，只改成 1-6 月 + 直线距离，建立新的可比基线。" }
        "seed_expand" { return "种子搜索阶段：不改结构，优先换 seed，确认当前结果是不是随机性造成的。" }
        "topk_local" { return "TopK 微调阶段：围绕当前最好值，小范围调整 adaptive_topk。" }
        "scheduler_local" { return "调度器微调阶段：在主结构不变时，尝试更早或更紧的学习率下降节奏。" }
        "precision_local" { return "精度确认阶段：用 fp32 对当前较优配方做确认，看数值精度是否带来提升。" }
        "histlen_local" { return "历史窗口微调阶段：小范围调整 hist_len，看长时距预测能否进一步改善。" }
        "semantics_branch" { return "语义分支阶段：在同一数据线上测试 weighted fixed / missing-aware 等语义增强。" }
        "seed_expand_extra" { return "额外种子阶段：在已有最优方向上继续扩充 seed，争取刷出更好的结果。" }
        "final_best_60epoch" { return "最终确认阶段：把全部实验结束后的最优配置再复跑两次 60 epoch，确认稳定性和是否还能刷新。" }
        "final_best_40epoch" { return "最终稳定性阶段：把当前最优配置再复跑三次 40 epoch，确认短预算稳定性。" }
        default { return "当前阶段：按 autopilot 规则继续推进下一轮实验。" }
    }
}

function Get-BoolText {
    param($Value)
    try {
        if ([bool]$Value) {
            return "开"
        }
    } catch {
    }
    return "关"
}

function Get-DistanceModeText {
    param([string]$Mode)
    switch ([string]$Mode) {
        "sensor_geodesic_haversine_km" { return "直线距离(haversine)" }
        "external_map_directed_route_distance_km_between_snapped_sensor_points" { return "真实路径距离(地图路网)" }
        "disabled_zero_no_custom_length_feature" { return "未使用额外距离特征" }
        default {
            if ([string]::IsNullOrWhiteSpace($Mode)) {
                return "未知"
            }
            return [string]$Mode
        }
    }
}

function Get-HistoryModeText {
    param([string]$Mode)
    switch ([string]$Mode) {
        "zero_fill_no_mask" { return "关(直接补 0)" }
        "causal_impute_with_mask" { return "开(causal 插补 + mask)" }
        default {
            if ([string]::IsNullOrWhiteSpace($Mode)) {
                return "未知"
            }
            return [string]$Mode
        }
    }
}

function Get-CurrentBestSummary {
    param($CampaignStatus)
    if ($null -eq $CampaignStatus) {
        return $null
    }

    $bestRunDir = [string]$CampaignStatus.current_best_run_dir
    if ([string]::IsNullOrWhiteSpace($bestRunDir) -or -not (Test-Path $bestRunDir)) {
        return $null
    }

    $metaPath = Join-Path $bestRunDir "stgat_meta.json"
    if (-not (Test-Path $metaPath)) {
        return $null
    }

    $meta = Get-Content -Path $metaPath -Raw | ConvertFrom-Json
    $metricsPath = Join-Path $bestRunDir "predictor_test_metrics.json"
    $metrics = $null
    if (Test-Path $metricsPath) {
        $metrics = Get-Content -Path $metricsPath -Raw | ConvertFrom-Json
    }

    $distanceMode = $null
    if ($null -ne $meta.graph_topology) {
        $distanceMode = [string]$meta.graph_topology.fixed_edge_length_feature_mode
    }
    if ([string]::IsNullOrWhiteSpace($distanceMode) -and $null -ne $metrics) {
        $distanceMode = [string]$metrics.fixed_edge_length_feature_mode
    }

    $historyMode = $null
    if ($null -ne $meta.graph_topology) {
        $historyMode = [string]$meta.graph_topology.history_missing_mode
    }
    if ([string]::IsNullOrWhiteSpace($historyMode) -and $null -ne $metrics) {
        $historyMode = [string]$metrics.history_missing_mode
    }

    return [pscustomobject]@{
        run_id = [string]$CampaignStatus.current_best_run_id
        run_dir = $bestRunDir
        dataset_dir = [string]$meta.dataset_dir
        split_policy = [string]$meta.split_policy
        topk = if ($null -ne $meta.graph_topology) { [string]$meta.graph_topology.adaptive_topk } else { "n/a" }
        seed = [string]$meta.seed
        lr = if ($null -ne $meta.optimizer) { [string]$meta.optimizer.lr } else { "n/a" }
        weight_decay = if ($null -ne $meta.optimizer) { [string]$meta.optimizer.weight_decay } else { "n/a" }
        precision = if ($null -ne $meta.training_runtime -and $null -ne $meta.training_runtime.precision) { [string]$meta.training_runtime.precision } elseif ($null -ne $metrics -and $null -ne $metrics.training_runtime -and $null -ne $metrics.training_runtime.precision) { [string]$metrics.training_runtime.precision } else { "n/a" }
        hidden_dim = [string]$meta.hidden_dim
        hist_len = [string]$meta.hist_len
        weighted_fixed = if ($null -ne $meta.graph_topology) { Get-BoolText $meta.graph_topology.fixed_graph_weighted } else { "未知" }
        history_mode = Get-HistoryModeText $historyMode
        distance_mode = Get-DistanceModeText $distanceMode
        best_epoch = if ($null -ne $meta.selected_checkpoint) { [string]$meta.selected_checkpoint.best_epoch } else { "n/a" }
        best_val_rmse = if ($null -ne $meta.selected_checkpoint) { Format-Decimal $meta.selected_checkpoint.best_val_raw_speed_rmse } else { "n/a" }
        test_rmse = if ($null -ne $metrics -and $null -ne $metrics.raw_metrics -and $null -ne $metrics.raw_metrics.speed) { Format-Decimal $metrics.raw_metrics.speed.rmse } else { Format-Decimal $CampaignStatus.current_best_rmse }
    }
}

function Get-ActivePlanEntry {
    param(
        [string]$PlanPath,
        [string]$RunId
    )
    if ([string]::IsNullOrWhiteSpace($PlanPath) -or -not (Test-Path $PlanPath)) {
        return $null
    }
    $planRaw = Get-Content -Path $PlanPath -Raw | ConvertFrom-Json
    $planItems = if ($planRaw -is [System.Array]) { $planRaw } else { @($planRaw) }
    foreach ($item in $planItems) {
        if ([string]$item.run_id -eq [string]$RunId) {
            return $item
        }
    }
    return $null
}

function Get-PlanDistanceText {
    param($ActivePlan)
    if ($null -eq $ActivePlan) {
        return "未知"
    }

    $lineage = [string]$ActivePlan.lineage
    $datasetDir = [string]$ActivePlan.dataset_dir
    if ($lineage -like "*map_route*" -or $datasetDir -like "*map_route*") {
        return "真实路径距离(map_route)"
    }
    if ($lineage -like "*haversine*" -or $datasetDir -like "*haversine*") {
        return "直线距离(haversine)"
    }
    return "未知"
}

function Get-ActiveTuningHighlights {
    param(
        [string]$StageName,
        $ActivePlan
    )

    if ($null -eq $ActivePlan) {
        return @("当前调整项：等待 plan 载入，暂时无法判断。")
    }

    $distanceText = Get-PlanDistanceText -ActivePlan $ActivePlan
    switch ([string]$StageName) {
        "route_bestshot" {
            return @(
                "当前调整方向：先试最有希望的路径距离分支。",
                ("当前参数：距离={0} | topk={1} | seed={2}" -f $distanceText, [string]$ActivePlan.topk, [string]$ActivePlan.seed)
            )
        }
        "paired_seed_matrix" {
            return @(
                "当前调整方向：距离方案与 seed 的配对矩阵。",
                ("当前参数：距离={0} | seed={1} | topk={2}" -f $distanceText, [string]$ActivePlan.seed, [string]$ActivePlan.topk)
            )
        }
        "baseline" {
            return @(
                "当前调整方向：建立 1-6 月标准资料范围下的新基线。",
                ("当前参数：距离={0} | topk={1} | seed={2}" -f $distanceText, [string]$ActivePlan.topk, [string]$ActivePlan.seed)
            )
        }
        "seed_expand" {
            return @(
                "当前调整方向：只换 seed，确认随机初始化/采样顺序带来的波动。",
                ("当前参数：seed={0} | 距离={1} | topk={2}" -f [string]$ActivePlan.seed, $distanceText, [string]$ActivePlan.topk)
            )
        }
        "seed_expand_extra" {
            return @(
                "当前调整方向：继续扩充 seed，尝试刷新当前最优。",
                ("当前参数：seed={0} | 距离={1} | topk={2}" -f [string]$ActivePlan.seed, $distanceText, [string]$ActivePlan.topk)
            )
        }
        "topk_local" {
            return @(
                "当前调整方向：只微调 adaptive_topk。",
                ("当前参数：adaptive_topk={0} | seed={1} | 距离={2}" -f [string]$ActivePlan.topk, [string]$ActivePlan.seed, $distanceText)
            )
        }
        "scheduler_local" {
            return @(
                "当前调整方向：只微调学习率调度器节奏。",
                ("当前参数：scheduler_monitor={0} | patience={1} | cooldown={2}" -f [string]$ActivePlan.scheduler_monitor, [string]$ActivePlan.scheduler_patience, [string]$ActivePlan.scheduler_cooldown)
            )
        }
        "precision_local" {
            return @(
                "当前调整方向：只确认数值精度。",
                ("当前参数：precision={0} | seed={1} | topk={2}" -f [string]$ActivePlan.precision, [string]$ActivePlan.seed, [string]$ActivePlan.topk)
            )
        }
        "histlen_local" {
            return @(
                "当前调整方向：只微调历史窗口长度。",
                ("当前参数：hist_len={0} | seed={1} | topk={2}" -f [string]$ActivePlan.hist_len, [string]$ActivePlan.seed, [string]$ActivePlan.topk)
            )
        }
        "semantics_branch" {
            return @(
                "当前调整方向：测试 fixed graph / missing-aware 语义增强。",
                ("当前参数：fixed_graph_weighted={0} | missing_aware_history={1} | 距离={2}" -f (Get-BoolText $ActivePlan.weighted_fixed), (Get-BoolText $ActivePlan.missing_aware), $distanceText)
            )
        }
        "final_best_60epoch" {
            return @(
                "当前调整方向：最终最优配置 60 epoch 复跑确认。",
                ("当前参数：epochs={0} | repeat run={1} | topk={2} | seed={3} | hist_len={4}" -f [string]$ActivePlan.epochs, [string]$ActivePlan.run_id, [string]$ActivePlan.topk, [string]$ActivePlan.seed, [string]$ActivePlan.hist_len)
            )
        }
        "final_best_40epoch" {
            return @(
                "当前调整方向：当前最优配置 40 epoch 三次复跑。",
                ("当前参数：epochs={0} | repeat run={1} | topk={2} | seed={3} | hist_len={4}" -f [string]$ActivePlan.epochs, [string]$ActivePlan.run_id, [string]$ActivePlan.topk, [string]$ActivePlan.seed, [string]$ActivePlan.hist_len)
            )
        }
        default {
            return @(
                "当前调整方向：按 autopilot 计划推进。",
                ("当前参数：距离={0} | topk={1} | seed={2} | lr={3} | wd={4}" -f $distanceText, [string]$ActivePlan.topk, [string]$ActivePlan.seed, [string]$ActivePlan.lr, [string]$ActivePlan.weight_decay)
            )
        }
    }
}

function Parse-EpochLine {
    param([string]$Line)
    if ([string]::IsNullOrWhiteSpace($Line)) {
        return $null
    }

    $pattern = '^\[(?<ds>[^\]]+)\]\[Ep\s+(?<epoch>\d+)\]\s+TrainV=(?<trainv>[0-9.]+)\s+ValV=(?<valv>[0-9.]+)\s+ValRMSE=(?<valrmse>[0-9.]+)\s+Best=(?<best>[0-9.]+)\s+\|\s+15min:RMSE=(?<rmse15>[0-9.]+)\s+30min:RMSE=(?<rmse30>[0-9.]+)\s+60min:RMSE=(?<rmse60>[0-9.]+)\s+\|\s+ETA=(?<eta>.+)$'
    $m = [regex]::Match($Line, $pattern)
    if (-not $m.Success) {
        return $null
    }
    return [pscustomobject]@{
        epoch = [int]$m.Groups["epoch"].Value
        trainv = [double]$m.Groups["trainv"].Value
        valv = [double]$m.Groups["valv"].Value
        valrmse = [double]$m.Groups["valrmse"].Value
        best = [double]$m.Groups["best"].Value
        rmse15 = [double]$m.Groups["rmse15"].Value
        rmse30 = [double]$m.Groups["rmse30"].Value
        rmse60 = [double]$m.Groups["rmse60"].Value
        eta = [string]$m.Groups["eta"].Value
    }
}

function Get-LatestEpochRecords {
    param([string]$LiveLogPath)
    if (-not (Test-Path $LiveLogPath)) {
        return @()
    }
    $epochLines = Get-Content -Path $LiveLogPath | Where-Object { $_ -match '^\[PEMS-BAY\]\[Ep\s+\d+\]' }
    $records = @()
    foreach ($line in $epochLines) {
        $record = Parse-EpochLine -Line $line
        if ($null -ne $record) {
            $records += $record
        }
    }
    return $records
}

function Get-JudgementText {
    param(
        $Current,
        $Previous,
        [double]$ReferenceRmse,
        $CampaignStatus,
        $QueueStatus
    )
    if ($null -eq $Current) {
        return @(
            "判断：当前还在启动或预处理阶段，模型尚未写出可读 checkpoint。",
            "建议：继续等待第一个 epoch 完成后再看趋势。"
        )
    }

    $lines = New-Object System.Collections.Generic.List[string]
    $deltaToBest = [double]$Current.valrmse - [double]$Current.best
    if ($null -ne $Previous) {
        $deltaVal = [double]$Current.valrmse - [double]$Previous.valrmse
        if ($deltaVal -lt -0.02) {
            $lines.Add("判断：当前验证 RMSE 还在明显下降，模型处于有效收敛阶段。")
        } elseif ($deltaVal -lt 0) {
            $lines.Add("判断：当前验证 RMSE 仍在缓慢变好，可以继续观察。")
        } elseif ($deltaVal -le 0.02) {
            $lines.Add("判断：当前验证 RMSE 基本进入平台区，短期提升开始变慢。")
        } else {
            $lines.Add("判断：当前这一轮验证比上一轮略差，可能出现正常波动或开始靠近平台。")
        }
    } else {
        $lines.Add("判断：目前样本还太少，只能确认训练已经顺利启动。")
    }

    if ($deltaToBest -le 0.005) {
        $lines.Add("补充：最新一轮几乎就是当前最佳附近，说明还没有明显跑偏。")
    } elseif ($deltaToBest -le 0.03) {
        $lines.Add("补充：最新一轮比最佳值稍差一点，但还在合理波动范围内。")
    } else {
        $lines.Add("补充：最新一轮已经明显离开当前最佳，若持续多轮如此，就要留意是否进入过拟合或平台期。")
    }

    if ($Current.rmse15 -le $Current.rmse30 -and $Current.rmse30 -le $Current.rmse60) {
        $lines.Add("多时距判断：15/30/60 的误差排序正常，长时距更难。")
    } else {
        $lines.Add("多时距判断：15/30/60 的排序有点异常，后面需要继续确认是不是随机波动。")
    }

    if ($null -ne $CampaignStatus -and $null -ne $CampaignStatus.current_best_rmse) {
        $diffToCampaign = [double]$CampaignStatus.current_best_rmse - [double]$ReferenceRmse
        if ($diffToCampaign -lt 0) {
            $lines.Add("全局判断：autopilot 已经超过旧历史最佳。")
        } elseif ([math]::Abs($diffToCampaign) -lt 0.02) {
            $lines.Add("全局判断：当前最优结果已经很接近旧历史最佳，后续重点是小幅微调。")
        } else {
            $lines.Add("全局判断：当前还没有接近旧历史最佳，现阶段更像是在建立新基线。")
        }
    } else {
        $lines.Add("全局判断：当前还没有产出完整 test 结果，暂时只能看验证集趋势。")
    }

    return $lines.ToArray()
}

$campaignStatusPath = Join-Path $CampaignRoot "campaign_status.json"

while ($true) {
    Clear-Host
    Write-Host "PEMS-BAY 自动监督器中文播报"
    Write-Host ("活动目录：{0}" -f $CampaignRoot)
    Write-Host ("时间：{0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
    Write-Host ""

    $campaignStatus = $null
    if (Test-Path $campaignStatusPath) {
        $campaignStatus = Get-Content -Path $campaignStatusPath -Raw | ConvertFrom-Json
        Write-Host ("当前阶段：{0}" -f [string]$campaignStatus.active_stage)
        Write-Host ("阶段说明：{0}" -f (Get-StageDescription -StageName ([string]$campaignStatus.active_stage)))
        Write-Host ("已完成批次：{0}" -f [string]$campaignStatus.completed_batches)
        Write-Host ("参考旧最佳 RMSE：{0}" -f (Format-Decimal $campaignStatus.reference_rmse))
        Write-Host ("当前 campaign 最优 RMSE：{0}" -f (Format-Decimal $campaignStatus.current_best_rmse))
        Write-Host ("当前 campaign 最优 run：{0}" -f [string]$campaignStatus.current_best_run_id)
        Write-Host ""
    } else {
        Write-Host "当前还在等待 campaign_status.json ..."
        Start-Sleep -Seconds $IntervalSeconds
        continue
    }

    $currentBest = Get-CurrentBestSummary -CampaignStatus $campaignStatus
    if ($null -ne $currentBest) {
        Write-Host "目前最佳配置：" -ForegroundColor Green
        Write-Host ("- run：{0}" -f $currentBest.run_id) -ForegroundColor Green
        Write-Host ("- 组合：topk={0} | seed={1} | lr={2} | wd={3} | hidden_dim={4} | hist_len={5} | precision={6}" -f $currentBest.topk, $currentBest.seed, $currentBest.lr, $currentBest.weight_decay, $currentBest.hidden_dim, $currentBest.hist_len, $currentBest.precision) -ForegroundColor Green
        Write-Host ("- 语义：split={0} | fixed_weighted={1} | missing_aware={2} | 距离={3}" -f $currentBest.split_policy, $currentBest.weighted_fixed, $currentBest.history_mode, $currentBest.distance_mode) -ForegroundColor Green
        Write-Host ("- 最佳 epoch：{0} | Best Val RMSE：{1} | Test RMSE：{2}" -f $currentBest.best_epoch, $currentBest.best_val_rmse, $currentBest.test_rmse) -ForegroundColor Green
        Write-Host ""
    }

    $latestBatch = Get-LatestBatchDir -Root $CampaignRoot
    if ($null -eq $latestBatch) {
        Write-Host "当前还没有 batch 目录。"
        Start-Sleep -Seconds $IntervalSeconds
        continue
    }

    $queueStatusPath = Join-Path $latestBatch.FullName "queue_status.json"
    if (-not (Test-Path $queueStatusPath)) {
        Write-Host ("最新批次 {0} 还没有 queue_status.json" -f $latestBatch.Name)
        Start-Sleep -Seconds $IntervalSeconds
        continue
    }

    $queueStatus = Get-Content -Path $queueStatusPath -Raw | ConvertFrom-Json
    Write-Host ("最新批次：{0}" -f $latestBatch.Name)
    Write-Host ("当前 run：{0}" -f [string]$queueStatus.active_run_id)
    Write-Host ("批次已完成：{0} | 失败：{1}" -f [string]$queueStatus.runs_finished, [string]$queueStatus.runs_failed)
    Write-Host ("批次当前最优 test RMSE：{0}" -f (Format-Decimal $queueStatus.session_best_test_rmse))
    Write-Host ""

    $activeRunDir = [string]$queueStatus.active_run_dir
    if ([string]::IsNullOrWhiteSpace($activeRunDir) -or -not (Test-Path $activeRunDir)) {
        Write-Host "当前没有活动中的 run，可能在收尾、切换下一轮，或本批已结束。"
        Start-Sleep -Seconds $IntervalSeconds
        continue
    }

    Write-Host ("现在在做的事情：正在训练 {0}" -f [string]$queueStatus.active_run_id)
    Write-Host ("运行目录：{0}" -f $activeRunDir)
    Write-Host ""

    $activePlan = Get-ActivePlanEntry -PlanPath ([string]$queueStatus.plan_path) -RunId ([string]$queueStatus.active_run_id)
    $tuningHighlights = Get-ActiveTuningHighlights -StageName ([string]$campaignStatus.active_stage) -ActivePlan $activePlan
    Write-Host "本轮红色调整项：" -ForegroundColor Red
    foreach ($item in $tuningHighlights) {
        Write-Host ("- {0}" -f $item) -ForegroundColor Red
    }
    Write-Host ""

    if ($null -ne $activePlan) {
        Write-Host "当前训练组合："
        Write-Host ("- 探索方向：{0}" -f [string]$activePlan.lineage)
        Write-Host ("- 数据目录：{0}" -f [string]$activePlan.dataset_dir)
        Write-Host ("- fixed_graph_weighted：{0}" -f (Get-BoolText $activePlan.weighted_fixed))
        Write-Host ("- missing_aware_history：{0}" -f (Get-BoolText $activePlan.missing_aware))
        Write-Host ("- adaptive_topk：{0}" -f [string]$activePlan.topk)
        Write-Host ("- seed：{0}" -f [string]$activePlan.seed)
        Write-Host ("- lr：{0}" -f [string]$activePlan.lr)
        Write-Host ("- weight_decay：{0}" -f [string]$activePlan.weight_decay)
        Write-Host ("- precision：{0}" -f [string]$activePlan.precision)
        Write-Host ("- hidden_dim：{0}" -f [string]$activePlan.hidden_dim)
        Write-Host ("- hist_len：{0}" -f [string]$activePlan.hist_len)
        Write-Host ("- epochs：{0}" -f [string]$activePlan.epochs)
        Write-Host ("- scheduler：{0} / patience={1} / cooldown={2}" -f [string]$activePlan.scheduler_monitor, [string]$activePlan.scheduler_patience, [string]$activePlan.scheduler_cooldown)
        Write-Host ""
    }

    $liveLog = Join-Path $activeRunDir "live.stdout.log"
    $epochRecords = Get-LatestEpochRecords -LiveLogPath $liveLog
    $current = $null
    $previous = $null
    if ($epochRecords.Count -ge 1) {
        $current = $epochRecords[-1]
    }
    if ($epochRecords.Count -ge 2) {
        $previous = $epochRecords[-2]
    }

    if ($null -eq $current) {
        Write-Host "当前进度：还在准备数据或刚进入训练，checkpoint 尚未准备好。"
    } else {
        Write-Host ("当前进度：第 {0} 轮" -f $current.epoch)
        Write-Host ("训练损失 TrainV：{0}" -f (Format-Decimal $current.trainv))
        Write-Host ("验证损失 ValV：{0}" -f (Format-Decimal $current.valv))
        Write-Host ("当前验证 RMSE：{0}" -f (Format-Decimal $current.valrmse))
        Write-Host ("当前最好验证 RMSE：{0}" -f (Format-Decimal $current.best))
        Write-Host ("15/30/60 分钟 RMSE：{0} / {1} / {2}" -f (Format-Decimal $current.rmse15), (Format-Decimal $current.rmse30), (Format-Decimal $current.rmse60))
        Write-Host ("预计完成时间：{0}" -f $current.eta)
    }

    Write-Host ""
    Write-Host "结果判断："
    $judgements = Get-JudgementText -Current $current -Previous $previous -ReferenceRmse ([double]$campaignStatus.reference_rmse) -CampaignStatus $campaignStatus -QueueStatus $queueStatus
    foreach ($item in $judgements) {
        Write-Host ("- {0}" -f $item)
    }

    $metricsPath = Join-Path $activeRunDir "predictor_test_metrics.json"
    if (Test-Path $metricsPath) {
        $metrics = Get-Content -Path $metricsPath -Raw | ConvertFrom-Json
        Write-Host ""
        Write-Host "最终结果："
        Write-Host ("- Test RMSE：{0}" -f (Format-Decimal $metrics.raw_metrics.speed.rmse))
        Write-Host ("- 15/30/60：{0} / {1} / {2}" -f (Format-Decimal $metrics.raw_metrics_report.speed.'15min'.rmse), (Format-Decimal $metrics.raw_metrics_report.speed.'30min'.rmse), (Format-Decimal $metrics.raw_metrics_report.speed.'60min'.rmse))
    }

    Write-Host ""
    Write-Host "原始训练日志尾部："
    if (Test-Path $liveLog) {
        Get-Content -Path $liveLog -Tail 5
    } else {
        Write-Host "live.stdout.log 尚未生成。"
    }

    Start-Sleep -Seconds $IntervalSeconds
}
