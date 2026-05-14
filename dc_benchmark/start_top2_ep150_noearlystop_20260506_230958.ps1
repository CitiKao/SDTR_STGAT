Set-Location -LiteralPath 'D:\Citi\STDR\STDR_STGAT'
$env:PYTHONUNBUFFERED='1'
[Console]::OutputEncoding=[System.Text.Encoding]::UTF8
$Host.UI.RawUI.WindowTitle = 'STDR benchmark top2 ep150 no early stop'
python -u dc_benchmark\run_all_benchmarks_colored.py --methods deep_multiconvlstm,mlrnn_taxi_demand --source-data-dir data --dataset-dir data\dc_benchmark_h16 --hist-len 16 --pred-horizon 4 --report-horizons-minutes 15,30,60 --split-policy project_monthly --time-feature-mode baseline --epochs 150 --early-stop-patience 0 --batch-size 32 --hidden-dim 32 --lr 1e-3 --device auto --output-dir runs\dc_benchmark_top2_h16_ep150_noearlystop_20260506_230958
