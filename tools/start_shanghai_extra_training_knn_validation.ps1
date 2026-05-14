$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'D:\Citi\STDR\STDR_STGAT'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$BenchmarkOut = "runs\shanghai_benchmarks_selected_ep150_$Stamp"
$Knn4Out = "runs\shanghai_stgat_knn4_ep180_$Stamp"
$Knn8Out = "runs\shanghai_stgat_knn8_ep180_$Stamp"

Write-Host "Shanghai extra suite started at $Stamp" -ForegroundColor Cyan
Write-Host "1-6/8: selected neural benchmarks, fresh 150 epochs because old benchmark checkpoints were not saved." -ForegroundColor Yellow
Write-Host "7/8: STGAT KNN=4, 180 epochs." -ForegroundColor Yellow
Write-Host "8/8: STGAT KNN=8, 180 epochs." -ForegroundColor Yellow
Write-Host "Outputs:" -ForegroundColor Cyan
Write-Host "  $BenchmarkOut"
Write-Host "  $Knn4Out"
Write-Host "  $Knn8Out"

python -u dc_benchmark\run_all_benchmarks_colored.py `
  --methods deep_multiconvlstm,graph_wavenet,mt_mf_gcn,stgcn,mlrnn_taxi_demand,st_resnet `
  --source-data-dir data\shanghai_dc `
  --dataset-dir data\shanghai_dc_benchmark `
  --output-dir $BenchmarkOut `
  --hist-len 14 `
  --pred-horizon 4 `
  --report-horizons-minutes 15,30,60 `
  --split-policy project_monthly `
  --time-feature-mode baseline `
  --epochs 150 `
  --early-stop-patience 0 `
  --batch-size 32 `
  --max-train-rows 200000 `
  --hidden-dim 32 `
  --lr 1e-3 `
  --device auto `
  --max-train-samples 0 `
  --max-eval-samples 0 `
  --checkpoint-dir "$BenchmarkOut\checkpoints" `
  --ordinal-offset 0 `
  --total-override 8

python -u tools\run_chengdu_dc_full_colored.py `
  --city-label Shanghai-KNN4 `
  --output-dir $Knn4Out `
  --source-data-dir data\shanghai_dc_knn4 `
  --benchmark-dir data\shanghai_dc_knn4_benchmark `
  --hist-len 14 `
  --pred-horizon 4 `
  --report-horizons-minutes 15,30,60 `
  --epochs 180 `
  --stgat-batch-size 8 `
  --hidden-dim 32 `
  --lr 1e-3 `
  --device auto `
  --precision auto `
  --skip-benchmarks `
  --ordinal-offset 6 `
  --total-override 8

python -u tools\run_chengdu_dc_full_colored.py `
  --city-label Shanghai-KNN8 `
  --output-dir $Knn8Out `
  --source-data-dir data\shanghai_dc_knn8 `
  --benchmark-dir data\shanghai_dc_knn8_benchmark `
  --hist-len 14 `
  --pred-horizon 4 `
  --report-horizons-minutes 15,30,60 `
  --epochs 180 `
  --stgat-batch-size 8 `
  --hidden-dim 32 `
  --lr 1e-3 `
  --device auto `
  --precision auto `
  --skip-benchmarks `
  --ordinal-offset 7 `
  --total-override 8

Write-Host "Shanghai extra suite complete." -ForegroundColor Green
