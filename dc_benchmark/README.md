# DC Benchmark

这个目录把项目里的 D/C 任务整理成一个可复用 benchmark 协议，目标是让 HA、ARIMA、XGBoost、LSTM、ConvLSTM、ST-ResNet、STGCN、DCRNN、Graph WaveNet、MLRNN、Deep MultiConvLSTM、MT-MF-GCN 都在同一份数据、同一组 split、同一套 raw-scale metric 下比较。

这里的 `DC` 沿用当前项目定义：

```text
raw_dc = D_RMSE + C_RMSE
```

也就是说本协议默认预测 `D=demand` 和 `C=supply/capacity` 两个目标，再统一计算 `D_RMSE`、`C_RMSE`、`gap`、`raw_dc`。这不是把目标改成单独的 `D/C ratio`。

## 导出资料集

```powershell
cd D:\Citi\STDR\STDR_STGAT
python -m dc_benchmark.export_dc_benchmark --source-data-dir data --output-dir data/dc_benchmark --hist-len 14 --pred-horizon 4 --force
```

导出后会得到：

```text
data/dc_benchmark/
  manifest.json
  splits.json
  targets_dc.npy        # (T, N, 2), channels = demand/supply
  node_demand.npy
  node_supply.npy
  time_features.npy
  adjacency_matrix.npy
  edge_index.npy
  time_meta.csv
```

`splits.json` 物化 train/val/test window 起点，所有论文入口都只读这个 split，不允许自己重新切资料。

## 运行论文名入口

每个入口脚本都以论文/方法名命名：

```powershell
python dc_benchmark\paper_runs\HA.py --dataset-dir data/dc_benchmark
python dc_benchmark\paper_runs\ARIMA.py --dataset-dir data/dc_benchmark
python dc_benchmark\paper_runs\XGBoost.py --dataset-dir data/dc_benchmark
python dc_benchmark\paper_runs\LSTM.py --dataset-dir data/dc_benchmark --epochs 100 --early-stop-patience 15
python dc_benchmark\paper_runs\ConvLSTM.py --dataset-dir data/dc_benchmark --epochs 100 --early-stop-patience 15
python dc_benchmark\paper_runs\ST_ResNet.py --dataset-dir data/dc_benchmark --epochs 100 --early-stop-patience 15
python dc_benchmark\paper_runs\STGCN.py --dataset-dir data/dc_benchmark --epochs 100 --early-stop-patience 15
python dc_benchmark\paper_runs\DCRNN.py --dataset-dir data/dc_benchmark --epochs 100 --early-stop-patience 15
python dc_benchmark\paper_runs\Graph_WaveNet.py --dataset-dir data/dc_benchmark --epochs 100 --early-stop-patience 15
python dc_benchmark\paper_runs\MLRNN.py --dataset-dir data/dc_benchmark --epochs 100 --early-stop-patience 15
python dc_benchmark\paper_runs\Deep_MultiConvLSTM.py --dataset-dir data/dc_benchmark --epochs 100 --early-stop-patience 15
python dc_benchmark\paper_runs\MT_MF_GCN.py --dataset-dir data/dc_benchmark --epochs 100 --early-stop-patience 15
```

Neural paper baselines default to `--epochs 100 --early-stop-patience 15` and select the checkpoint with the lowest validation `raw_dc` before reporting test metrics.

结果默认写到 `runs/dc_benchmark/*_dc_metrics.json`。

## 结果边界

当前入口是 `paper_inspired_reimplementation`，不是论文官方代码复现。论文名只表示架构来源和对照类别，结果 JSON 里会写：

```json
"not_official_paper_result": true
```

论文中如果要写得严谨，建议表头用：

```text
Paper-inspired same-data DC benchmark
```

不要写成：

```text
Official reproduction of PaperName
```

除非之后真的接入论文官方代码、官方模块、同等调参预算，并保留完整 run manifest。

## 15 点正确性验证

1. 数据源只来自 `manifest.json` 指向的 demand/supply/time/graph 文件。
2. train/val/test split 必须使用 `splits.json` 里的物化 window indices。
3. 每个 window 的 history 和 target 必须完整落在同一 split。
4. normalization 统计只允许来自 train time mask。
5. raw metric 必须在原始计数尺度计算。
6. `raw_dc` 唯一等于 `D_RMSE + C_RMSE`。
7. D、C、gap 都输出 overall、per-step、report horizon。
8. report horizon 的分钟数必须能映射到 prediction step。
9. 预测张量协议固定为 `(batch, nodes, horizon, 2)`。
10. graph 方法只能使用 benchmark manifest 里的 graph。
11. time feature 是否使用必须记录在 manifest/run config。
12. HA/ARIMA/XGBoost 等非神经方法也必须走同一 evaluator。
13. test split 只用于最终报告，不能调参。
14. 结果 JSON 必须记录 split hash、hist_len、pred_horizon。
15. 不同 feature contract 或不同 split 的结果不能放进同一主榜硬排名。
