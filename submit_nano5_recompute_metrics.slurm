#!/bin/bash
#SBATCH --job-name=sdtr_stgat_recompute
#SBATCH --partition=normal2
#SBATCH --account=ACD114200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=slurm-recompute-%j.out
#SBATCH --error=slurm-recompute-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gaodeefreetaipei@gmail.com

set -euo pipefail

# Under Slurm, the batch script is executed from a spool copy in /var/spool/slurmd.
# Use the original submission directory instead of BASH_SOURCE[0].
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
ENV_NAME="${CONDA_ENV_NAME:-sdtr-stgat-h200}"
RUN_ID="${RUN_ID:-nano5_159619}"
CHECKPOINT="${CHECKPOINT:-}"
if [[ -n "$CHECKPOINT" ]]; then
  RUN_DIR="${RUN_DIR:-$(dirname "$CHECKPOINT")}"
else
  RUN_DIR="${RUN_DIR:-$SCRIPT_DIR/runs/$RUN_ID}"
  CHECKPOINT="$RUN_DIR/stgat_best.pt"
fi
BATCH_SIZE="${BATCH_SIZE:-32}"
PRECISION="${PRECISION:-bf16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EDGE_LENGTH_SOURCE="${EDGE_LENGTH_SOURCE:-osrm}"
HIST_LEN="${HIST_LEN:-12}"
REPORT_HORIZONS_MINUTES="${REPORT_HORIZONS_MINUTES:-}"

if [[ ! -f "$SCRIPT_DIR/recompute_predictor_metrics.py" ]]; then
  echo "Expected recompute_predictor_metrics.py under $SCRIPT_DIR" >&2
  echo "Submit this job from the project directory, for example:" >&2
  echo "  cd /work/<your_account>/STDR-STGAT" >&2
  echo "  RUN_ID=nano5_159619 sbatch submit_nano5_recompute_metrics.slurm" >&2
  exit 1
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  echo "Set RUN_ID, RUN_DIR, or CHECKPOINT before submitting." >&2
  echo "Example:" >&2
  echo "  RUN_ID=nano5_159619 sbatch submit_nano5_recompute_metrics.slurm" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"

# Avoid `ml purge` here for the same reason as create_nano5_env.sh:
# some Nano5 shells call `conda deactivate` during purge before the conda hook
# is initialized.
# Disable `nounset` around `ml load` for compatibility with the site Lmod init
# script, which may reference SLURM_JOBID directly.
set +u
ml load miniconda3/24.11.1
set -u
eval "$(conda shell.bash hook)"

echo "job_id=${SLURM_JOB_ID}"
echo "job_node_list=${SLURM_JOB_NODELIST}"
echo "project_dir=${SCRIPT_DIR}"
echo "run_id=${RUN_ID}"
echo "run_dir=${RUN_DIR}"
echo "checkpoint=${CHECKPOINT}"
echo "env_name=${ENV_NAME}"
echo "batch_size=${BATCH_SIZE}"
echo "precision=${PRECISION}"
echo "num_workers=${NUM_WORKERS}"
echo "edge_length_source=${EDGE_LENGTH_SOURCE}"
echo "hist_len=${HIST_LEN}"
echo "report_horizons_minutes=${REPORT_HORIZONS_MINUTES}"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Conda environment '$ENV_NAME' was not found." >&2
  echo "On the Nano5 login node, run:" >&2
  echo "  cd $SCRIPT_DIR && bash create_nano5_env.sh" >&2
  exit 1
fi

RECOMPUTE_ARGS=(
  --checkpoint "$CHECKPOINT"
  --log-dir "$RUN_DIR"
  --data-dir "$SCRIPT_DIR/data"
  --device auto
  --precision "$PRECISION"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --edge-length-source "$EDGE_LENGTH_SOURCE"
  --hist-len "$HIST_LEN"
)

if [[ -n "$REPORT_HORIZONS_MINUTES" ]]; then
  RECOMPUTE_ARGS+=(--report-horizons-minutes "$REPORT_HORIZONS_MINUTES")
fi

conda run --no-capture-output -n "$ENV_NAME" \
  python "$SCRIPT_DIR/recompute_predictor_metrics.py" "${RECOMPUTE_ARGS[@]}"

echo "Recompute finished. Outputs are in: $RUN_DIR"
echo "Generated:"
echo "  $RUN_DIR/stgat_meta.json"
echo "  $RUN_DIR/predictor_test_metrics.json"
echo "Note: predictor_log.json cannot be reconstructed from a checkpoint alone."
