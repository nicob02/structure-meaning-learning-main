#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --array=0-19
#SBATCH --account=def-eporte2

set -euo pipefail

module purge
module load python/3.10 cuda/12.2

source ~/venvs/graminduct/bin/activate

if [ ! -d "$HOME/pytorch-struct" ]; then
  git clone --branch infer_pos_tag https://github.com/zhaoyanpeng/pytorch-struct.git ~/pytorch-struct
  cd ~/pytorch-struct
  pip install -e .
fi

cd "$SLURM_SUBMIT_DIR/vc-pcfg"
mkdir -p "$SLURM_SUBMIT_DIR/runs"

SEEDS=(91 214 527 627 1018)
MODELS=("joint" "sem-first" "syn-first" "visual-labels")

MODEL_IDX=$((SLURM_ARRAY_TASK_ID / 5))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 5))

MODEL=${MODELS[$MODEL_IDX]}
SEED=${SEEDS[$SEED_IDX]}

ENCODER_FILE="all_as-resn-50.npy"
IMG_DIM=2048
EXTRA_ARGS=""

if [ "$MODEL" = "sem-first" ]; then
  EXTRA_ARGS="--sem_first"
elif [ "$MODEL" = "syn-first" ]; then
  EXTRA_ARGS="--syn_first"
elif [ "$MODEL" = "visual-labels" ]; then
  ENCODER_FILE="all_flat_features_gold.npy"
  IMG_DIM=756
fi

python ./as_train.py \
  --num_epochs 30 \
  --encoder_file "$ENCODER_FILE" \
  --img_dim "$IMG_DIM" \
  --visual_mode \
  --logger_name "$SLURM_SUBMIT_DIR/runs/zh_full_${MODEL}_s${SEED}" \
  --seed "$SEED" \
  --data_path "../preprocessed-data/abstractscenes_zh" \
  --skip_syntactic_bootstrapping \
  --resume \
  $EXTRA_ARGS
