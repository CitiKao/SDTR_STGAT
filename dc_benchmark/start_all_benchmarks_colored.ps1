$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$Host.UI.RawUI.WindowTitle = "STDR DC Benchmark formal h16 100ep es15"

Set-Location -LiteralPath "D:\Citi\STDR\STDR_STGAT"

python -u dc_benchmark\run_all_benchmarks_colored.py `
  --methods all `
  --source-data-dir data `
  --dataset-dir data\dc_benchmark_h16 `
  --force-dataset `
  --hist-len 16 `
  --pred-horizon 4 `
  --report-horizons-minutes 15,30,60 `
  --split-policy project_monthly `
  --time-feature-mode baseline `
  --epochs 100 `
  --early-stop-patience 15 `
  --batch-size 32 `
  --hidden-dim 32 `
  --lr 1e-3 `
  --device auto
