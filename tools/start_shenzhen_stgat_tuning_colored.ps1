$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$Host.UI.RawUI.WindowTitle = "深圳 STGAT 调参 h14 p4"

Set-Location -LiteralPath "D:\Citi\STDR\STDR_STGAT"

python -u tools\run_shenzhen_stgat_tuning_colored.py `
  --data-dir data\shenzhen_dc `
  --baseline-dir runs\shenzhen_dc_full_20260508_002254 `
  --hist-len 14 `
  --pred-horizon 4 `
  --report-horizons-minutes 15,30,60 `
  --epochs 100 `
  --device auto `
  --precision auto
