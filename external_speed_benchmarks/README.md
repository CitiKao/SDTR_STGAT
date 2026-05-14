# External Speed Benchmarks

This folder adapts the current STGAT speed path to node-speed sensor datasets such as `METR-LA` and `PEMS-BAY`.

Files:

- `prepare_dcrnn_sensor_datasets.py`: downloads the official DCRNN traffic files and sensor coordinates, prefers official benchmark graph assets when available, writes maps, and stores processed arrays plus a `speed_valid_mask.npy`.
- `train_sensor_speed.py`: trains the node-speed adaptation with masked speed loss/metrics, always-on train-only quantile clipping, `batch_size=32`, `pred_horizon=12`, and default `15/30/60` reporting. By default `--epochs` is the exact training budget; early stopping is opt-in via `--early-stop-patience > 0`.
- `generate_sensor_benchmark_report.py`: merges the trained run outputs into a single markdown report.
- `run_local_4090_sensor_benchmarks.ps1`: a convenience wrapper for local 4090 runs.

Default assumptions:

- `METR-LA` uses the official DCRNN `metr-la.h5`, which currently ends on `2012-06-27 23:55:00`.
- `PEMS-BAY` uses `2017-01-01 00:00:00` through `2017-05-31 23:55:00`.
- The default external sensor split now follows the same project monthly rule as the main predictor: days `1-20` train, days `21-24` val, days `25+` test, with full `history+target` window containment.
- If you need the older benchmark-style contiguous split for side experiments, `train_sensor_speed.py` still supports `--split-policy benchmark_contiguous` plus optional `--split-alignment day|week|month`.
- Speed benchmarks should be reported with masked metrics that exclude missing zero readings, especially for `METR-LA`.
- Outlier cleaning is now always enabled in the sensor benchmark training path and is recorded in reports as part of the preprocessing variant.
- Early stopping is disabled by default so a run configured with `--epochs 180` will complete all `180` epochs unless you explicitly enable early stopping.
- The default external speed scheduler is intentionally more conservative now: it monitors `val_loss`, uses `patience=8`, `cooldown=2`, `min_lr=1e-5`, keeps `weight_decay=1e-5`, and records the full optimizer/scheduler config in each run directory.

## Current Best Local METR-LA Record

Current reference run:

- `runs/external_speed/METR-LA_untilbeat_seed31_20260421_004136`

Verified setup:

- dataset: `METR-LA`
- representation: `sensor_node`
- split policy: `benchmark_contiguous`
- fixed graph source: `official_distance_csv`
- adaptive graph: enabled with `topk=16`
- cleaning: `train_quantile_clip`
- metric protocol: `masked_zero_excluded`
- optimizer LR: `0.001`
- `pred_horizon=12`
- report horizons: `15 / 30 / 60`

Best validation checkpoint:

- epoch `19`
- best val raw speed `RMSE`: `5.61047`
- val `15 / 30 / 60 RMSE`: `4.90045 / 5.68330 / 6.44777`

Recorded test metrics:

- test raw speed `RMSE / MAE / MAPE`: `6.15990 / 3.43850 / 9.15800%`
- test `15 / 30 / 60 RMSE`: `5.31431 / 6.23545 / 7.12792`

Benchmark note:

- this run is not fully official-like because `train_quantile_clip` is enabled before normalization
- use it as the current local best reference for the cleaned `METR-LA` line unless a later run improves it
