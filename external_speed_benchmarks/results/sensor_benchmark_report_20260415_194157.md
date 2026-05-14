# Sensor Benchmark Report

Generated at `2026-04-15T19:41:57`.

## METR-LA [cleaned:train_quantile_clip]

- Run dir: `runs\external_speed\smoke_METR-LA_epoch_budget_ep3`
- Completed epochs: `3/3`
- Stop reason: `epoch_budget_exhausted`
- Early stopping: `disabled`
- Best epoch: `3`
- Best val RMSE: `5.9203`
- Preprocessing variant: `train_quantile_clip_q0p01_0p99_masked_zero_excluded`
- Dataset fingerprint: `3834c00239b50702b9985b3cd59ca3cc58de2191015ca583467450c29d6a68b4`
- Sensors: `207`
- Graph edges: `1515`
- Split: `contiguous_time_70_10_20_full_window_containment` with train/val/test = `23967/3405/6831`
- Adaptive graph: `on`
- Fixed graph source: `official_distance_csv`
- Metric protocol: `masked_zero_excluded`
- Outlier cleaning: `enabled | method=train_quantile_clip | fit=train_history_only | apply=all_valid_points | replace=clip`
- Cleaned points: `124070 (1.90% of valid points)`
- Benchmark comparability: `custom / not directly official-like`
- Benchmark deviations: `train-only quantile clipping applied before normalization; custom STGAT adaptation rather than the original benchmark model`
- Actual date range: `2012-03-01T00:00:00` to `2012-06-27T23:55:00`
- Dataset note: `METR-LA official DCRNN h5 currently ends on 2012-06-27 23:55, so requested dates beyond that are clipped automatically. Speed zeros are stored with an explicit validity mask so masked benchmark metrics can exclude missing readings.`

| Horizon | RMSE | MAE | MAPE | MSE |
| --- | ---: | ---: | ---: | ---: |
| 15min | 5.5619 | 3.3014 | 8.48% | 30.9348 |
| 30min | 6.4648 | 3.7372 | 10.09% | 41.7931 |
| 60min | 7.3843 | 4.2412 | 11.90% | 54.5272 |
