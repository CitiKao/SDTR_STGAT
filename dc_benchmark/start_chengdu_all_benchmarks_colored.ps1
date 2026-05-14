$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$Host.UI.RawUI.WindowTitle = "Chengdu DC 12 benchmark methods"

Set-Location -LiteralPath "D:\Citi\STDR\STDR_STGAT"

python -u dc_benchmark\run_all_benchmarks_colored.py `
  --methods all `
  --source-data-dir data\chengdu_dc `
  --dataset-dir data\chengdu_dc_benchmark `
  --hist-len 14 `
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
