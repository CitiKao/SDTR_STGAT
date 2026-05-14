$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$Host.UI.RawUI.WindowTitle = "深圳 DC 自动实验 STGAT + 12 benchmarks"

Set-Location -LiteralPath "D:\Citi\STDR\STDR_STGAT"

python -u tools\run_shenzhen_dc_full_colored.py `
  --source-data-dir data\shenzhen_dc `
  --benchmark-dir data\shenzhen_dc_benchmark `
  --hist-len 14 `
  --pred-horizon 4 `
  --report-horizons-minutes 15,30,60 `
  --epochs 100 `
  --early-stop-patience 15 `
  --stgat-batch-size 8 `
  --benchmark-batch-size 32 `
  --hidden-dim 32 `
  --lr 1e-3 `
  --device auto `
  --precision auto
