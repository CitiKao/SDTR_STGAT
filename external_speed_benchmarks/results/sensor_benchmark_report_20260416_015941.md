# Sensor Benchmark Report

Generated at `2026-04-16T01:59:41`.

## METR-LA [cleaned:train_quantile_clip]

- Run dir: `runs\external_speed\METR-LA_bs32_ep180_cleaned_full180_20260415_195651`
- Completed epochs: `180/180`
- Stop reason: `epoch_budget_exhausted`
- Early stopping: `disabled`
- Best epoch: `14`
- Best val RMSE: `5.6757`
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
| 15min | 5.3557 | 3.0568 | 7.96% | 28.6839 |
| 30min | 6.2815 | 3.5197 | 9.51% | 39.4568 |
| 60min | 7.2517 | 4.0674 | 11.35% | 52.5873 |

## PEMS-BAY [cleaned:train_quantile_clip]

- Run dir: `runs\external_speed\PEMS-BAY_bs32_ep180_cleaned_full180_20260415_221528`
- Completed epochs: `180/180`
- Stop reason: `epoch_budget_exhausted`
- Early stopping: `disabled`
- Best epoch: `26`
- Best val RMSE: `3.3897`
- Preprocessing variant: `train_quantile_clip_q0p01_0p99_masked_zero_excluded`
- Dataset fingerprint: `e746e92634e8dec54253af1e2291758f3584d34b7cd7d598ead248d2a72ec48b`
- Sensors: `325`
- Graph edges: `2369`
- Split: `contiguous_time_70_10_20_full_window_containment` with train/val/test = `30410/4325/8672`
- Adaptive graph: `on`
- Fixed graph source: `official_adj_pkl`
- Metric protocol: `masked_zero_excluded`
- Outlier cleaning: `enabled | method=train_quantile_clip | fit=train_history_only | apply=all_valid_points | replace=clip`
- Cleaned points: `301195 (2.13% of valid points)`
- Benchmark comparability: `custom / not directly official-like`
- Benchmark deviations: `train-only quantile clipping applied before normalization; custom STGAT adaptation rather than the original benchmark model`
- Actual date range: `2017-01-01T00:00:00` to `2017-05-31T23:55:00`
- Dataset note: `PEMS-BAY was clipped to the requested 2017-05-31 cutoff. Official adjacency is loaded from the public adj_mx_bay.pkl release when graph_mode=official.`

| Horizon | RMSE | MAE | MAPE | MSE |
| --- | ---: | ---: | ---: | ---: |
| 15min | 2.8000 | 1.4694 | 3.02% | 7.8398 |
| 30min | 3.6996 | 1.8409 | 4.00% | 13.6871 |
| 60min | 4.4849 | 2.2034 | 5.08% | 20.1140 |
