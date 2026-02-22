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

PYTHONUNBUFFERED=1 python -u "data preprocessing/as_prepare_zh.py" \
  --input_caps "../preprocessed-data/abstractscenes/all_caps_zh.jsonl" \
  --input_ids "../preprocessed-data/abstractscenes/all.id_zh" \
  --output_dir "../preprocessed-data/abstractscenes_zh" \
  --copy_features_from "../preprocessed-data/abstractscenes" \
  --log_every "$LOG_EVERY"
