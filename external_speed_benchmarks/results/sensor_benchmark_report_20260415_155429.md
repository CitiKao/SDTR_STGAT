# Sensor Benchmark Report

Generated at `2026-04-15T15:54:29`.

## METR-LA [raw]

- Run dir: `runs\external_speed\smoke_METR-LA_raw_switch_ep1`
- Best epoch: `1`
- Best val RMSE: `6.5322`
- Preprocessing variant: `raw_masked_zero_excluded`
- Dataset fingerprint: `0efbdabbe46a4897408236d8c7af3aee68b19ee890b0f24c3641f7f3314d243d`
- Sensors: `207`
- Graph edges: `1515`
- Split: `contiguous_time_70_10_20_full_window_containment` with train/val/test = `23967/3405/6831`
- Adaptive graph: `on`
- Fixed graph source: `official_distance_csv`
- Metric protocol: `masked_zero_excluded`
- Outlier cleaning: `disabled`
- Cleaned points: `0 (0.00% of valid points)`
- Benchmark comparability: `official-like preprocessing`
- Benchmark deviations: `custom STGAT adaptation rather than the original benchmark model`
- Actual date range: `2012-03-01T00:00:00` to `2012-06-27T23:55:00`
- Dataset note: `METR-LA official DCRNN h5 currently ends on 2012-06-27 23:55, so requested dates beyond that are clipped automatically. Speed zeros are stored with an explicit validity mask so masked benchmark metrics can exclude missing readings.`

| Horizon | RMSE | MAE | MAPE | MSE |
| --- | ---: | ---: | ---: | ---: |
| 15min | 6.3474 | 3.7506 | 9.89% | 40.2901 |
| 30min | 6.9482 | 4.0505 | 11.02% | 48.2769 |
| 60min | 7.8137 | 4.6177 | 13.59% | 61.0538 |

## METR-LA [cleaned:train_quantile_clip]

- Run dir: `runs\external_speed\smoke_METR-LA_clean_switch_ep1`
- Best epoch: `1`
- Best val RMSE: `6.2805`
- Preprocessing variant: `train_quantile_clip_q0p01_0p99_masked_zero_excluded`
- Dataset fingerprint: `0efbdabbe46a4897408236d8c7af3aee68b19ee890b0f24c3641f7f3314d243d`
- Sensors: `207`
- Graph edges: `1515`
- Split: `contiguous_time_70_10_20_full_window_containment` with train/val/test = `23967/3405/6831`
- Adaptive graph: `on`
- Fixed graph source: `official_distance_csv`
- Metric protocol: `masked_zero_excluded`
- Outlier cleaning: `enabled | method=train_quantile_clip | fit=train_history_only | apply=all_valid_points | replace=clip`
- Cleaned points: `124070 (1.90% of valid points)`
- Benchmark comparability: `custom / not directly official-like`
- Benchmark deviations: `optional train-only quantile clipping applied before normalization; custom STGAT adaptation rather than the original benchmark model`
- Actual date range: `2012-03-01T00:00:00` to `2012-06-27T23:55:00`
- Dataset note: `METR-LA official DCRNN h5 currently ends on 2012-06-27 23:55, so requested dates beyond that are clipped automatically. Speed zeros are stored with an explicit validity mask so masked benchmark metrics can exclude missing readings.`

| Horizon | RMSE | MAE | MAPE | MSE |
| --- | ---: | ---: | ---: | ---: |
| 15min | 6.1332 | 3.6525 | 9.26% | 37.6157 |
| 30min | 6.7716 | 3.9844 | 10.27% | 45.8542 |
| 60min | 7.6065 | 4.5782 | 12.56% | 57.8585 |
