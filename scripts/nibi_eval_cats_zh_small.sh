#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --account=def-eporte2

set -euo pipefail

module purge
module load python/3.10

source ~/venvs/graminduct/bin/activate

cd "$SLURM_SUBMIT_DIR/vc-pcfg"

BASE_DIR="$SLURM_SUBMIT_DIR/results/parses/zh_small"
PRED_DIR="$BASE_DIR/pred"
DF_DIR="$BASE_DIR/df_cats"
CT_DIR="$BASE_DIR/ct_cats"

mkdir -p "$PRED_DIR" "$DF_DIR" "$CT_DIR"

SEEDS=(91 214 527 627 1018)
MODELS=("joint" "sem-first" "syn-first" "visual-labels")
DATA_PATH="../preprocessed-data/abstractscenes_zh_small"
EPOCH=4

for MODEL in "${MODELS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    RUN_DIR="$SLURM_SUBMIT_DIR/runs/zh_small_${MODEL}_s${SEED}"
    CKPT="$RUN_DIR/checkpoints/${EPOCH}.pth.tar"
    OUT_BASENAME="${MODEL}_${SEED}_parses.json"
    OUT_DIR="$PRED_DIR/${MODEL}_${SEED}"

    if [ ! -f "$CKPT" ]; then
      echo "Missing checkpoint: $CKPT"
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

    cp "$OUT_DIR/${OUT_BASENAME}_df.csv" "$DF_DIR/"
    cp "$OUT_DIR/${OUT_BASENAME}_ct.csv" "$CT_DIR/"
  done
done
