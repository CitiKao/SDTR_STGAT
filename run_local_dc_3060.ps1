param(
    [int]$Epochs = 100,
    [int]$BatchSize = 16,
    [int]$NumStBlocks = 2,
    [int]$AdaptiveTopK = 16,
    [string]$MonitorTask = "dc",
    [string]$Precision = "fp32"
)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$runId = "local_dc_${NumStBlocks}b_3060_" + (Get-Date -Format "yyyyMMdd_HHmmss")
$logDir = Join-Path $root ("runs\" + $runId)

python (Join-Path $root "train_predictor.py") `
  --data-dir (Join-Path $root "data") `
  --log-dir $logDir `
  --device cuda `
  --precision $Precision `
  --batch-size $BatchSize `
  --epochs $Epochs `
  --val-interval 5 `
  --num-workers 0 `
  --train-task dc `
  --monitor-task $MonitorTask `
  --num-st-blocks $NumStBlocks `
  --adaptive-topk $AdaptiveTopK
