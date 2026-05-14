# Sensor Benchmark Report

Generated at `2026-04-18T06:21:01`.

## METR-LA [pseudo_edge | contiguous_ratio_70_10_20 | cleaned:train_quantile_clip]

- Run dir: `runs\external_speed\smoke_METR-LA_pseudo_edge_guardrails_ep1`
- Completed epochs: `1/1`
- Stop reason: `epoch_budget_exhausted`
- Early stopping: `disabled`
- Best epoch: `1`
- Best val RMSE: `6.9081`
- Preprocessing variant: `train_quantile_clip_q0p01_0p99_masked_zero_excluded`
- Dataset fingerprint: `548f8e88a141309221d8e15db55cb752330c9ef558fe6ac669718541663d4cf4`
- Representation: `pseudo_edge`
- Representation variant: `pseudo_edge__base-official_distance_csv__radius-0.250km__halfscale-0.450__clusterscale-1.250__fallbackk-3`
- V domain: `edge`
- Split alignment: `none`
- Train span: `2012-03-01T00:00:00` to `2012-05-23T07:05:00`
- Val span: `2012-05-23T07:10:00` to `2012-06-04T04:45:00`
- Test span: `2012-06-04T04:50:00` to `2012-06-27T23:55:00`
- Sensors: `207`
- Graph nodes: `213`
- Graph edges: `207`
- Speed items: `207`
- Dataset summary source: `run_snapshot:prepared_dataset_summary.json`
- Split: `contiguous_time_70_10_20_full_window_containment` with train/val/test = `23967/3405/6831`
- Split description: `Contiguous 70/10/20 time split with full history+target window containment.`
- Adaptive graph: `on`
- Fixed graph source: `pseudo_edge_from_sensor_graph`
- Metric protocol: `masked_zero_excluded`
- Optimizer: `Adam | lr=0.001 | weight_decay=1e-05`
- Scheduler: `ReduceLROnPlateau | monitor=val_loss | patience=2 | cooldown=0 | min_lr=1e-05`
- Outlier cleaning: `enabled | method=train_quantile_clip | fit=train_window_only | apply=all_valid_points | replace=clip`
- Cleaned points: `124047 (1.90% of valid points)`
- Benchmark comparability: `custom / not directly official-like`
- Benchmark deviations: `train-only quantile clipping applied before normalization; experimental pseudo-edge representation used instead of the official sensor-node benchmark target; custom STGAT adaptation rather than the original benchmark model`
- Actual date range: `2012-03-01T00:00:00` to `2012-06-27T23:55:00`
- Pseudo-edge topology: `self_loop_fixes=44 (21.26%) | isolated_edges=52 (25.12%) | unique_structural_edges=204 | weak_components=87 | largest_component=12.08%`
- Pseudo-edge construction: `base=official_distance_csv | cluster_radius_km=0.25 | cluster_radius_scale=1.25 | half_length_scale=0.45 | fallback_neighbor_k=3`
- Pseudo-edge warnings: `self_loop_fix_ratio=0.213 exceeds 0.100; isolated_edge_ratio=0.251 exceeds 0.150; line_graph_largest_component_ratio=0.121 is below 0.250`
- Dataset note: `METR-LA official DCRNN h5 currently ends on 2012-06-27 23:55, so requested dates beyond that are clipped automatically. Speed zeros are stored with an explicit validity mask so masked benchmark metrics can exclude missing readings. Experimental pseudo-edge representation: each sensor is reinterpreted as one directed pseudo-edge built from the sensor graph; results are not directly comparable to official sensor-node benchmarks.`

| Horizon | RMSE | MAE | MAPE | MSE |
| --- | ---: | ---: | ---: | ---: |
| 15min | 6.5698 | 4.0939 | 11.12% | 43.1625 |
| 30min | 7.2389 | 4.3561 | 12.00% | 52.4015 |
| 60min | 8.7233 | 5.2297 | 15.67% | 76.0965 |
