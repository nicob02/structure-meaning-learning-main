#!/bin/bash
#SBATCH --time=23:59:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=def-eporte2

set -euo pipefail

module purge
module load python/3.10

if [ ! -d "$HOME/venvs/graminduct" ]; then
  python -m venv ~/venvs/graminduct
  source ~/venvs/graminduct/bin/activate
  pip install --upgrade pip
  # Pin transformers <5 to keep encode_plus API for hanlp tokenizers
  pip install "transformers<5" "tokenizers<0.20"
  pip install hanlp opencc-python-reimplemented nltk torch torchvision torchaudio matplotlib
  if [ ! -d "$HOME/pytorch-struct" ]; then
    git clone --branch infer_pos_tag https://github.com/zhaoyanpeng/pytorch-struct.git ~/pytorch-struct
    cd ~/pytorch-struct
    pip install -e .
  fi
else
  source ~/venvs/graminduct/bin/activate
fi

cd "$SLURM_SUBMIT_DIR/vc-pcfg"

LOG_EVERY="${LOG_EVERY:-500}"
VOCAB_SIZE="${VOCAB_SIZE:-2000}"
KEEP_SIMPLIFIED_TOKENS="${KEEP_SIMPLIFIED_TOKENS:-0}"
INPUT_CAPS="${INPUT_CAPS:-../preprocessed-data/abstractscenes/all_caps_zh.jsonl}"
INPUT_IDS="${INPUT_IDS:-../preprocessed-data/abstractscenes/all.id_zh}"
OUTPUT_DIR="${OUTPUT_DIR:-../preprocessed-data/abstractscenes_zh}"
COPY_FEATURES_FROM="${COPY_FEATURES_FROM:-../preprocessed-data/abstractscenes}"
EXTRA_PREP_ARGS=()
if [ "$KEEP_SIMPLIFIED_TOKENS" = "1" ]; then
  EXTRA_PREP_ARGS+=(--keep_simplified_tokens)
fi

PYTHONUNBUFFERED=1 python -u "data preprocessing/as_prepare_zh.py" \
  --input_caps "$INPUT_CAPS" \
  --input_ids "$INPUT_IDS" \
  --output_dir "$OUTPUT_DIR" \
  --copy_features_from "$COPY_FEATURES_FROM" \
  --vocab_size "$VOCAB_SIZE" \
  --log_every "$LOG_EVERY" \
  "${EXTRA_PREP_ARGS[@]}"
