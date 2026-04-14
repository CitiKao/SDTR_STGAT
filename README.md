# STDR-STGAT

`STDR-STGAT` is the standalone predictor training component inside the broader `STDR` project. It is the focused training bundle for the modified STGAT model, with separate workflows for:

- `DC` training
- `V` training
- H200 Slurm execution

## Core Files

- `train_predictor.py`
  Main training entrypoint. Supports `joint`, `dc`, and `v` training modes.
- `stgat_model.py`
  Modified STGAT predictor definition.
- `predictor_normalization.py`
  Normalization and inverse-normalization utilities for `D/C/V`.
- `data_loader.py`
  NYC graph and feature loading utilities.
- `recompute_predictor_metrics.py`
  Rebuilds metrics JSON from checkpoints and metadata.
- `submit_nano5_stgat_dc.slurm`
  H200 Slurm entrypoint for `DC` training.
- `submit_nano5_stgat_v.slurm`
  H200 Slurm entrypoint for `V` training.
- `H200_V_SWEEP_COMMANDS.md`
  H200 command handbook for the `V` training branch.

## Tasks

### DC

- predicts `demand`
- predicts `supply`

### V

- predicts `speed`

## Model Structure

The predictor follows the STGAT design idea with stacked `ST-block`s:

- temporal convolution first
- graph attention second

Adjustable structure parameters include:

- `--num-st-blocks`
- `--num-gtcn-layers`
- `--hidden-dim`
- `--num-heads`
- `--kernel-size`

## Graph Paths

### DC path

- fixed node adjacency path
- adaptive node adjacency path

The adaptive adjacency is now a real learned graph:

- learned from node embeddings
- sparsified with `top-k`
- default `--adaptive-topk 16`

Set:

```bash
--adaptive-topk 0
```

to fall back to a dense learned graph.

### V path

- fixed line-graph adjacency path
- no adaptive edge graph in the mainline setup

## Time Features

The predictor uses six calendar features:

- `month_sin`
- `month_cos`
- `weekday_sin`
- `weekday_cos`
- `slot_sin`
- `slot_cos`

This supports learning:

- intra-day adjacent time-window patterns
- weekday differences
- recurring month-level differences

## Split Policy

The monthly split rule is:

- days `1-20`: `train`
- days `21-24`: `val`
- days `25+`: `test`

To avoid leakage, a sample is kept only when the full `history + target` window stays inside a single split.

This prevents:

- train samples from reading `test` history
- val/test windows from crossing split boundaries

## Normalization

- `demand`: `log1p + z-score`
- `supply`: `log1p + z-score`
- `speed`: per-edge `z-score`

All statistics are computed from train windows only.

## Training Modes

### DC-only

```bash
python train_predictor.py --train-task dc --monitor-task dc
```

### V-only

```bash
python train_predictor.py --train-task v --monitor-task v
```

### Joint

```bash
python train_predictor.py --train-task joint --monitor-task raw_dc
```

`raw_dc` means:

- `demand_rmse + supply_rmse`

## H200 Usage

### Train DC

```bash
NUM_ST_BLOCKS=2 sbatch submit_nano5_stgat_dc.slurm
```

Common overridable variables:

- `BATCH_SIZE`
- `EPOCHS`
- `PRECISION`
- `NUM_ST_BLOCKS`
- `ADAPTIVE_TOPK`
- `TRAIN_TASK`
- `MONITOR_TASK`
- `RUN_PREFIX`

### Train V

```bash
NUM_ST_BLOCKS=2 sbatch submit_nano5_stgat_v.slurm
```

Common overridable variables:

- `BATCH_SIZE`
- `EPOCHS`
- `LR`
- `MAX_TIME_STEPS`
- `HIST_LEN`
- `PRED_HORIZON`
- `REPORT_HORIZONS_MINUTES`
- `PRECISION`
- `NUM_ST_BLOCKS`
- `ADAPTIVE_TOPK`
- `RUN_PREFIX`

### V H200 Sweep Workflow

For paper-style V reporting, the V runner now defaults to:

- `HIST_LEN=12`
- `PRED_HORIZON=4`
- `REPORT_HORIZONS_MINUTES=15,30,60`

With 15-minute slots, this yields:

- `15 min = step 1`
- `30 min = step 2`
- `60 min = step 4`

### V H200 Sweep Workflow

This branch is the canonical `V` training surface.

Use:

- `submit_nano5_stgat_v.slurm`
- `H200_V_SWEEP_COMMANDS.md`

The command handbook is organized into:

- `smoke`
- `pilot`
- `extended`

and every command explicitly shows:

- `TRAIN_TASK=v`
- `MONITOR_TASK=v`
- `BATCH_SIZE`
- `EPOCHS`
- `LR`

Both Slurm scripts are configured for:

- H200
- `#SBATCH --time=07:00:00`

## Outputs

Each training run writes:

- `stgat_best.pt`
- `stgat_final.pt`
- `stgat_meta.json`
- `predictor_log.json`
- `predictor_test_metrics.json`

For V runs with report horizons enabled, `predictor_test_metrics.json` now contains:

- legacy aggregate `raw_metrics`
- additive `raw_metrics_per_step`
- additive `raw_metrics_report`
- `report_horizons` metadata with slot/minute-to-step mapping

## Recompute

`recompute_predictor_metrics.py` is used to:

- rebuild missing metrics JSON
- regenerate metadata and test metrics from checkpoints

It respects saved run metadata whenever available.

## Source Of Truth

For predictor work, treat these files as canonical:

- `train_predictor.py`
- `stgat_model.py`
- `predictor_normalization.py`

If any older notes or temporary scripts disagree with the code, follow the code in `STDR-STGAT`.
