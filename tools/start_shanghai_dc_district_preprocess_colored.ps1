$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$Host.UI.RawUI.WindowTitle = "Shanghai DC preprocessing - real district polygons"

Set-Location -LiteralPath "D:\Citi\STDR\STDR_STGAT"

python -u tools\build_shanghai_dc_dataset.py `
  --input-dir data\external_datasets\raw\dc_candidate_downloads\sh_taxi_data `
  --output-dir data\shanghai_dc_district `
  --benchmark-output-dir data\shanghai_dc_district_benchmark `
  --start-date 2015-04-01 `
  --end-date 2015-04-30 `
  --slot-minutes 15 `
  --partition-mode district_polygon `
  --boundary-geojson data\external_datasets\raw\dc_candidate_downloads\shanghai_boundaries\shanghai_datav_310000_districts.geojson `
  --polygon-lookup-resolution 1600 `
  --polygon-touch-tolerance-deg 1e-7 `
  --bucket-count 1024 `
  --max-open-buckets 1024 `
  --progress-rows 1000000 `
  --hist-len 14 `
  --pred-horizon 4
