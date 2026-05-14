$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$Host.UI.RawUI.WindowTitle = "Shanghai DC preprocessing 20x20"

Set-Location -LiteralPath "D:\Citi\STDR\STDR_STGAT"

python -u tools\build_shanghai_dc_dataset.py `
  --input-dir data\external_datasets\raw\dc_candidate_downloads\sh_taxi_data `
  --output-dir data\shanghai_dc `
  --benchmark-output-dir data\shanghai_dc_benchmark `
  --start-date 2015-04-01 `
  --end-date 2015-04-30 `
  --slot-minutes 15 `
  --grid-rows 20 `
  --grid-cols 20 `
  --bbox 121.16,121.82,30.91,31.46 `
  --knn 5 `
  --bucket-count 1024 `
  --max-open-buckets 1024 `
  --progress-rows 1000000 `
  --hist-len 14 `
  --pred-horizon 4
