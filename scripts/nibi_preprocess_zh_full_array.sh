#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=def-eporte2
#SBATCH --array=0-30

set -euo pipefail

module purge
module load python/3.10

source ~/venvs/graminduct/bin/activate

cd "$SLURM_SUBMIT_DIR/vc-pcfg"

CHUNK_SIZE="${CHUNK_SIZE:-2000}"
LOG_EVERY="${LOG_EVERY:-200}"
INPUT_CAPS="../preprocessed-data/abstractscenes/all_caps_zh.jsonl"
INPUT_IDS="../preprocessed-data/abstractscenes/all.id_zh"
SHARDS_DIR="../preprocessed-data/abstractscenes_zh_shards"

TOTAL=$(wc -l < "$INPUT_IDS")
START=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE))
END=$((START + CHUNK_SIZE))

if [ "$START" -ge "$TOTAL" ]; then
  echo "Shard $SLURM_ARRAY_TASK_ID start $START >= total $TOTAL; exiting."
  exit 0
fi

OUT_DIR="$SHARDS_DIR/shard_$(printf "%02d" "$SLURM_ARRAY_TASK_ID")"
mkdir -p "$OUT_DIR"

COPY_FROM=""
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
  COPY_FROM="../preprocessed-data/abstractscenes"
fi

PYTHONUNBUFFERED=1 python -u "data preprocessing/as_prepare_zh.py" \
  --input_caps "$INPUT_CAPS" \
  --input_ids "$INPUT_IDS" \
  --output_dir "$OUT_DIR" \
  --copy_features_from "$COPY_FROM" \
  --start "$START" \
  --end "$END" \
  --log_every "$LOG_EVERY"
