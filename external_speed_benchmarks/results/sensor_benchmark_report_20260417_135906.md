# Sensor Benchmark Report

Generated at `2026-04-17T13:59:06`.

## METR-LA [cleaned:train_quantile_clip]

- Run dir: `runs\external_speed\METR-LA_project_monthly_bs32_ep20_20260417`
- Completed epochs: `20/20`
- Stop reason: `epoch_budget_exhausted`
- Early stopping: `disabled`
- Best epoch: `20`
- Best val RMSE: `5.9406`
- Preprocessing variant: `train_quantile_clip_q0p01_0p99_masked_zero_excluded`
- Dataset fingerprint: `3834c00239b50702b9985b3cd59ca3cc58de2191015ca583467450c29d6a68b4`
- Sensors: `207`
- Graph edges: `1515`
- Split: `per_month_day_1_20_train_21_24_val_25_plus_test_full_window_containment` with train/val/test = `22948/4516/6532`
- Split rule: `train=days 1-20 | val=days 21-24 | test=days 25+`
- Train first/last target: `2012-03-01T01:00:00` to `2012-06-20T23:55:00`
- Val first/last target: `2012-03-21T01:00:00` to `2012-06-24T23:55:00`
- Test first/last target: `2012-03-25T01:00:00` to `2012-06-27T23:55:00`
- Split description: `每月 1-20 train / 21-24 val / 25+ test，且整個 history+target window 必須完整落在同一 split`
- Adaptive graph: `on`
- Fixed graph source: `official_distance_csv`
- Metric protocol: `masked_zero_excluded`
- Optimizer: `Adam | lr=0.001 | weight_decay=1e-05`
- Scheduler: `ReduceLROnPlateau | monitor=val_loss | patience=8 | cooldown=2 | min_lr=1e-05`
- Outlier cleaning: `enabled | method=train_quantile_clip | fit=train_history_only | apply=all_valid_points | replace=clip`
- Cleaned points: `120706 (1.85% of valid points)`
- Benchmark comparability: `custom / not directly official-like`
- Benchmark deviations: `train-only quantile clipping applied before normalization; project monthly split used instead of external benchmark contiguous split; custom STGAT adaptation rather than the original benchmark model`
- Actual date range: `2012-03-01T00:00:00` to `2012-06-27T23:55:00`
- Dataset note: `METR-LA official DCRNN h5 currently ends on 2012-06-27 23:55, so requested dates beyond that are clipped automatically. Speed zeros are stored with an explicit validity mask so masked benchmark metrics can exclude missing readings.`

| Horizon | RMSE | MAE | MAPE | MSE |
| --- | ---: | ---: | ---: | ---: |
| 15min | 5.0790 | 2.9716 | 7.17% | 25.7959 |
| 30min | 5.9680 | 3.3950 | 8.49% | 35.6167 |
| 60min | 6.8781 | 3.9390 | 10.10% | 47.3088 |

## PEMS-BAY [cleaned:train_quantile_clip]

- Run dir: `runs\external_speed\PEMS-BAY_project_monthly_bs32_ep20_20260417`
- Completed epochs: `20/20`
- Stop reason: `epoch_budget_exhausted`
- Early stopping: `disabled`
- Best epoch: `19`
- Best val RMSE: `3.5301`
- Preprocessing variant: `train_quantile_clip_q0p01_0p99_masked_zero_excluded`
- Dataset fingerprint: `e746e92634e8dec54253af1e2291758f3584d34b7cd7d598ead248d2a72ec48b`
- Sensors: `325`
- Graph edges: `2369`
- Split: `per_month_day_1_20_train_21_24_val_25_plus_test_full_window_containment` with train/val/test = `28673/5645/8813`
- Split rule: `train=days 1-20 | val=days 21-24 | test=days 25+`
- Train first/last target: `2017-01-01T01:00:00` to `2017-05-20T23:55:00`
- Val first/last target: `2017-01-21T01:00:00` to `2017-05-24T23:55:00`
- Test first/last target: `2017-01-25T01:00:00` to `2017-05-31T23:55:00`
- Split description: `每月 1-20 train / 21-24 val / 25+ test，且整個 history+target window 必須完整落在同一 split`
- Adaptive graph: `on`
- Fixed graph source: `official_adj_pkl`
- Metric protocol: `masked_zero_excluded`
- Optimizer: `Adam | lr=0.001 | weight_decay=1e-05`
- Scheduler: `ReduceLROnPlateau | monitor=val_loss | patience=8 | cooldown=2 | min_lr=1e-05`
- Outlier cleaning: `enabled | method=train_quantile_clip | fit=train_history_only | apply=all_valid_points | replace=clip`
- Cleaned points: `266804 (1.89% of valid points)`
- Benchmark comparability: `custom / not directly official-like`
- Benchmark deviations: `train-only quantile clipping applied before normalization; project monthly split used instead of external benchmark contiguous split; custom STGAT adaptation rather than the original benchmark model`
- Actual date range: `2017-01-01T00:00:00` to `2017-05-31T23:55:00`
- Dataset note: `PEMS-BAY was clipped to the requested 2017-05-31 cutoff. Official adjacency is loaded from the public adj_mx_bay.pkl release when graph_mode=official.`

| Horizon | RMSE | MAE | MAPE | MSE |
| --- | ---: | ---: | ---: | ---: |
| 15min | 2.6494 | 1.3625 | 2.78% | 7.0191 |
| 30min | 3.4505 | 1.7110 | 3.63% | 11.9060 |
| 60min | 4.0579 | 2.0444 | 4.43% | 16.4668 |
