# Local 4090 Training

This folder contains the local RTX 4090 workflow for this repo.

Files:

- `train_local_4090_dc.ps1`
  Local Windows launcher for DC training on the current dataset.
- `benchmark_batch_sizes.py`
  Runs a short 2-epoch batch-size sweep and writes a machine-readable summary.
- `results/`
  Committed benchmark summaries for the local machine.

Notes:

- Real checkpoints and raw training logs are written under `runs/`.
- The benchmark script scales learning rate with `sqrt(batch_size / 32)` to stay close to the existing H200 sweep heuristic.
- The local launcher defaults to `bf16`, `cuda`, and `num_workers=0`, which is a safer default for a single Windows workstation.
