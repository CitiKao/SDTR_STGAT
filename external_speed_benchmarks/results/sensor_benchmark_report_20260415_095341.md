# Sensor Benchmark Report

Generated at `2026-04-15T09:53:41`.

## METR-LA

- Run dir: `D:\Citi\STDR\STDR_STGAT\runs\external_speed\METR-LA_bs32_ep180_final_20260415_025640`
- Best epoch: `5`
- Best val RMSE: `11.1528`
- Sensors: `207`
- Graph edges: `2024`
- Split: `contiguous_time_70_10_20_full_window_containment` with train/val/test = `23967/3405/6831`
- Adaptive graph: `on`

| Horizon | RMSE | MAE | MSE |
| --- | ---: | ---: | ---: |
| 15min | 9.9152 | 5.1513 | 98.3114 |
| 20min | 10.7311 | 5.5487 | 115.1562 |
| 60min | 14.8135 | 7.7994 | 219.4400 |

## PEMS-BAY

- Run dir: `D:\Citi\STDR\STDR_STGAT\runs\external_speed\PEMS-BAY_bs32_ep180_final_20260415_025929`
- Best epoch: `10`
- Best val RMSE: `4.1847`
- Sensors: `325`
- Graph edges: `3182`
- Split: `contiguous_time_70_10_20_full_window_containment` with train/val/test = `30410/4325/8672`
- Adaptive graph: `on`

| Horizon | RMSE | MAE | MSE |
| --- | ---: | ---: | ---: |
| 15min | 3.4487 | 1.6647 | 11.8936 |
| 20min | 3.7581 | 1.7977 | 14.1235 |
| 60min | 5.1182 | 2.4705 | 26.1960 |
