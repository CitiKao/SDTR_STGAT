#!/bin/bash
#SBATCH --job-name=stdr_stgat_v
#SBATCH --partition=normal2
#SBATCH --account=ACD114200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=07:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gaodeefreetaipei@gmail.com

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
ENV_NAME="${CONDA_ENV_NAME:-sdtr-stgat-h200}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-1e-3}"
MAX_TIME_STEPS="${MAX_TIME_STEPS:-0}"
PRECISION="${PRECISION:-bf16}"
VAL_INTERVAL="${VAL_INTERVAL:-5}"
HIST_LEN="${HIST_LEN:-12}"
PRED_HORIZON="${PRED_HORIZON:-4}"
REPORT_HORIZONS_MINUTES="${REPORT_HORIZONS_MINUTES:-15,30,60}"
COMPILE="${COMPILE:-0}"
COMPILE_MODE="${COMPILE_MODE:-reduce-overhead}"
NUM_ST_BLOCKS="${NUM_ST_BLOCKS:-2}"
ADAPTIVE_TOPK="${ADAPTIVE_TOPK:-16}"
SPEED_USE_ADAPTIVE="${SPEED_USE_ADAPTIVE:-0}"
SPEED_ADAPTIVE_TOPK="${SPEED_ADAPTIVE_TOPK:-$ADAPTIVE_TOPK}"
RUN_PREFIX="${RUN_PREFIX:-nano5_v_${NUM_ST_BLOCKS}b}"
RUN_DIR="${SCRIPT_DIR}/runs/${RUN_PREFIX}_${SLURM_JOB_ID}"

if [[ ! -f "$SCRIPT_DIR/train_predictor.py" ]]; then
  echo "Expected train_predictor.py under $SCRIPT_DIR" >&2
  echo "Submit this job from the project directory, for example:" >&2
  echo "  cd /work/<your_account>/STDR-STGAT && sbatch submit_nano5_stgat_v.slurm" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"

set +u
ml load miniconda3/24.11.1
set -u
eval "$(conda shell.bash hook)"

echo "job_id=${SLURM_JOB_ID}"
echo "job_node_list=${SLURM_JOB_NODELIST}"
echo "project_dir=${SCRIPT_DIR}"
echo "run_dir=${RUN_DIR}"
echo "env_name=${ENV_NAME}"
echo "batch_size=${BATCH_SIZE}"
echo "epochs=${EPOCHS}"
echo "lr=${LR}"
echo "max_time_steps=${MAX_TIME_STEPS}"
echo "precision=${PRECISION}"
echo "val_interval=${VAL_INTERVAL}"
echo "hist_len=${HIST_LEN}"
echo "pred_horizon=${PRED_HORIZON}"
echo "report_horizons_minutes=${REPORT_HORIZONS_MINUTES}"
echo "compile=${COMPILE}"
echo "compile_mode=${COMPILE_MODE}"
echo "num_st_blocks=${NUM_ST_BLOCKS}"
echo "adaptive_topk=${ADAPTIVE_TOPK}"
echo "speed_use_adaptive=${SPEED_USE_ADAPTIVE}"
echo "speed_adaptive_topk=${SPEED_ADAPTIVE_TOPK}"
echo "train_task=v"
echo "monitor_task=v"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Conda environment '$ENV_NAME' was not found." >&2
  echo "On the Nano5 login node, run:" >&2
  echo "  cd $SCRIPT_DIR && bash create_nano5_env.sh" >&2
  exit 1
fi

TRAIN_ARGS=(
  --data-dir "$SCRIPT_DIR/data"
  --log-dir "$RUN_DIR"
  --device auto
  --precision "$PRECISION"
  --batch-size "$BATCH_SIZE"
  --epochs "$EPOCHS"
  --lr "$LR"
  --val-interval "$VAL_INTERVAL"
  --hist-len "$HIST_LEN"
  --pred-horizon "$PRED_HORIZON"
  --report-horizons-minutes "$REPORT_HORIZONS_MINUTES"
  --num-st-blocks "$NUM_ST_BLOCKS"
  --adaptive-topk "$ADAPTIVE_TOPK"
  --speed-adaptive-topk "$SPEED_ADAPTIVE_TOPK"
  --train-task v
  --monitor-task v
)

if [[ "$MAX_TIME_STEPS" != "0" ]]; then
  TRAIN_ARGS+=(--max-time-steps "$MAX_TIME_STEPS")
fi

if [[ "$COMPILE" == "1" ]]; then
  TRAIN_ARGS+=(--compile --compile-mode "$COMPILE_MODE")
fi

if [[ "$SPEED_USE_ADAPTIVE" == "1" ]]; then
  TRAIN_ARGS+=(--speed-use-adaptive)
fi

conda run -n "$ENV_NAME" python "$SCRIPT_DIR/train_predictor.py" "${TRAIN_ARGS[@]}"

echo "Training finished. Outputs are in: $RUN_DIR"
