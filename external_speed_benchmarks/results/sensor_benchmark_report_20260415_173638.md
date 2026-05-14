# Sensor Benchmark Report

Generated at `2026-04-15T17:36:38`.

## METR-LA [raw]

- Run dir: `runs\external_speed\smoke20_METR-LA_raw`
- Best epoch: `16`
- Best val RMSE: `5.9148`
- Preprocessing variant: `raw_masked_zero_excluded`
- Dataset fingerprint: `3834c00239b50702b9985b3cd59ca3cc58de2191015ca583467450c29d6a68b4`
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
| 15min | 5.5321 | 3.2532 | 8.70% | 30.6040 |
| 30min | 6.4892 | 3.7146 | 10.44% | 42.1101 |
| 60min | 7.4629 | 4.2710 | 12.45% | 55.6948 |

## METR-LA [cleaned:train_quantile_clip]

- Run dir: `runs\external_speed\smoke20_METR-LA_clean`
- Best epoch: `17`
- Best val RMSE: `5.6785`
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
- Benchmark deviations: `optional train-only quantile clipping applied before normalization; custom STGAT adaptation rather than the original benchmark model`
- Actual date range: `2012-03-01T00:00:00` to `2012-06-27T23:55:00`
- Dataset note: `METR-LA official DCRNN h5 currently ends on 2012-06-27 23:55, so requested dates beyond that are clipped automatically. Speed zeros are stored with an explicit validity mask so masked benchmark metrics can exclude missing readings.`

| Horizon | RMSE | MAE | MAPE | MSE |
| --- | ---: | ---: | ---: | ---: |
| 15min | 5.3627 | 3.0940 | 7.89% | 28.7590 |
| 30min | 6.3503 | 3.5905 | 9.58% | 40.3260 |
| 60min | 7.3909 | 4.1640 | 11.52% | 54.6258 |

## PEMS-BAY [raw]

- Run dir: `runs\external_speed\smoke20_PEMS-BAY_raw`
- Best epoch: `13`
- Best val RMSE: `3.9469`
- Preprocessing variant: `raw_masked_zero_excluded`
- Dataset fingerprint: `e746e92634e8dec54253af1e2291758f3584d34b7cd7d598ead248d2a72ec48b`
- Sensors: `325`
- Graph edges: `2369`
- Split: `contiguous_time_70_10_20_full_window_containment` with train/val/test = `30410/4325/8672`
- Adaptive graph: `on`
- Fixed graph source: `official_adj_pkl`
- Metric protocol: `masked_zero_excluded`
- Outlier cleaning: `disabled`
- Cleaned points: `0 (0.00% of valid points)`
- Benchmark comparability: `official-like preprocessing`
- Benchmark deviations: `custom STGAT adaptation rather than the original benchmark model`
- Actual date range: `2017-01-01T00:00:00` to `2017-05-31T23:55:00`
- Dataset note: `PEMS-BAY was clipped to the requested 2017-05-31 cutoff. Official adjacency is loaded from the public adj_mx_bay.pkl release when graph_mode=official.`

| Horizon | RMSE | MAE | MAPE | MSE |
| --- | ---: | ---: | ---: | ---: |
| 15min | 3.3265 | 1.6342 | 3.56% | 11.0658 |
| 30min | 4.1249 | 2.0199 | 4.69% | 17.0148 |
| 60min | 4.8397 | 2.3657 | 5.82% | 23.4227 |

## PEMS-BAY [cleaned:train_quantile_clip]

- Run dir: `runs\external_speed\smoke20_PEMS-BAY_clean`
- Best epoch: `19`
- Best val RMSE: `3.3917`
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
- Benchmark deviations: `optional train-only quantile clipping applied before normalization; custom STGAT adaptation rather than the original benchmark model`
- Actual date range: `2017-01-01T00:00:00` to `2017-05-31T23:55:00`
- Dataset note: `PEMS-BAY was clipped to the requested 2017-05-31 cutoff. Official adjacency is loaded from the public adj_mx_bay.pkl release when graph_mode=official.`

| Horizon | RMSE | MAE | MAPE | MSE |
| --- | ---: | ---: | ---: | ---: |
| 15min | 2.8028 | 1.4538 | 3.00% | 7.8555 |
| 30min | 3.7372 | 1.8417 | 4.03% | 13.9669 |
| 60min | 4.6178 | 2.2805 | 5.33% | 21.3244 |
