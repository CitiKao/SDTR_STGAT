# Sensor Benchmark Report

Generated at `2026-04-15T15:01:01`.

## METR-LA

- Run dir: `D:\Citi\STDR\STDR_STGAT\runs\external_speed\smoke_METR-LA_official_masked_ep2`
- Best epoch: `2`
- Best val RMSE: `6.2639`
- Sensors: `207`
- Graph edges: `1515`
- Split: `contiguous_time_70_10_20_full_window_containment` with train/val/test = `23967/3405/6831`
- Adaptive graph: `on`
- Fixed graph source: `official_distance_csv`
- Metric protocol: `masked_zero_excluded`

| Horizon | RMSE | MAE | MAPE | MSE |
| --- | ---: | ---: | ---: | ---: |
| 15min | 5.8372 | 3.4045 | 0.00% | 34.0733 |
| 30min | 6.7585 | 3.8866 | 0.00% | 45.6776 |
| 60min | 7.6922 | 4.4864 | 0.00% | 59.1705 |

## PEMS-BAY

- Run dir: `D:\Citi\STDR\STDR_STGAT\runs\external_speed\smoke_PEMS-BAY_official_masked_ep2`
- Best epoch: `1`
- Best val RMSE: `5.1720`
- Sensors: `325`
- Graph edges: `2369`
- Split: `contiguous_time_70_10_20_full_window_containment` with train/val/test = `30410/4325/8672`
- Adaptive graph: `on`
- Fixed graph source: `official_adj_pkl`
- Metric protocol: `masked_zero_excluded`

| Horizon | RMSE | MAE | MAPE | MSE |
| --- | ---: | ---: | ---: | ---: |
| 15min | 4.0029 | 2.0296 | 0.00% | 16.0232 |
| 30min | 4.9352 | 2.5171 | 0.00% | 24.3561 |
| 60min | 5.7249 | 2.9185 | 0.00% | 32.7741 |
