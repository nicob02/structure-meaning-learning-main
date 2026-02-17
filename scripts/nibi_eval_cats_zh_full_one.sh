#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --account=def-eporte2

set -euo pipefail

module purge
module load python/3.10

source ~/venvs/graminduct/bin/activate

cd "$SLURM_SUBMIT_DIR/vc-pcfg"

MODEL="${MODEL:-joint}"
SEED="${SEED:-1018}"
DATA_PATH="../preprocessed-data/abstractscenes_zh"
EPOCH="${EPOCH:-}"

BASE_DIR="$SLURM_SUBMIT_DIR/results/parses/zh_full_one"
OUT_DIR="$BASE_DIR/${MODEL}_${SEED}"
mkdir -p "$OUT_DIR"

RUN_DIR="$SLURM_SUBMIT_DIR/runs/zh_full_${MODEL}_s${SEED}"
if [ -z "$EPOCH" ]; then
  CKPT="$(ls -1 "$RUN_DIR"/checkpoints/*.pth.tar 2>/dev/null | sort -V | tail -n 1)"
else
  CKPT="$RUN_DIR/checkpoints/${EPOCH}.pth.tar"
fi
OUT_BASENAME="${MODEL}_${SEED}_parses.json"

if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
  echo "Missing checkpoint in: $RUN_DIR/checkpoints"
  exit 1
fi

python "$SLURM_SUBMIT_DIR/vc-pcfg/as_extra_evals.py" \
  --mode trees \
  --model_init "$CKPT" \
  --data_path "$DATA_PATH" \
  --logger_name "$OUT_DIR" \
  --out_file "$OUT_BASENAME" \
  --skip_syntactic_bootstrapping

python "$SLURM_SUBMIT_DIR/vc-pcfg/as_extra_evals.py" \
  --mode cats \
  --data_path "$DATA_PATH" \
  --logger_name "$OUT_DIR" \
  --out_file "$OUT_BASENAME" \
  --skip_syntactic_bootstrapping
