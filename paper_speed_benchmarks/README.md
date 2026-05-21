# Paper Speed Benchmarks

This folder is the isolated benchmark area for paper-style traffic speed
prediction experiments. It intentionally stays separate from the main
`train_predictor.py` workflow so paper baselines can be added and compared under
one NYC speed dataset protocol.

## Current Program

`nyc_speed_prediction.py` trains or evaluates NYC edge-speed predictors with:

- the canonical `data/` files in this repo
- monthly split: days 1-20 train, 21-24 validation, 25+ test
- full-window containment to avoid leakage
- train-only per-edge speed normalization
- raw speed metrics in km/h
- report horizons such as 15, 30, and 60 minutes

Available runnable models:

- `persistence`: repeat the last observed edge speed
- `temporal_mlp`: shared per-edge temporal MLP baseline
- `line_graph_gru`: GRU plus directed line-graph message passing
- `stgat_edge`: the current repo STGAT edge-speed path
- `astgnn_ltd`: paper-inspired long-term attention STGNN adapter
- `icst_dnet`: paper-inspired causal spatio-temporal diffusion adapter
- `3s_tbln`: paper-inspired self-supervised bilateral learning adapter
- `mvstgcn`: paper-inspired multi-view spatial-temporal GCN adapter
- `nexusqn`: paper-inspired spatiotemporal MLP-Mixer adapter
- `dagcan`: paper-inspired decoupled adaptive graph convolution attention adapter
- `sgsl_gat_nlstm`: paper-inspired task-oriented graph learning + GAT-nLSTM adapter

The listed 2024-2026 paper adapters are currently paper-inspired
implementations, not official reproductions. Promote an adapter to faithful
reproduction only after checking the paper details and any official code.

## Example

```powershell
cd D:\STDR
python paper_speed_benchmarks\nyc_speed_prediction.py `
  --model temporal_mlp `
  --data-dir data `
  --hist-len 14 `
  --pred-horizon 4 `
  --report-horizons-minutes 15,30,60 `
  --device cuda `
  --precision bf16 `
  --batch-size 32 `
  --epochs 50 `
  --log-dir runs\paper_speed_benchmarks\nyc_temporal_mlp
```

Smoke test on CPU:

```powershell
python paper_speed_benchmarks\nyc_speed_prediction.py --model persistence --max-time-steps 3000 --device cpu --log-dir runs\paper_speed_benchmarks\smoke_persistence
```
