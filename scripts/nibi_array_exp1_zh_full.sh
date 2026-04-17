#!/bin/bash
#SBATCH --time=09:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
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
DATA_PATH="${DATA_PATH:-../preprocessed-data/abstractscenes_zh}"
RUN_PREFIX="${RUN_PREFIX:-zh_full}"
USE_RESUME="${USE_RESUME:-1}"
VSE_MT_ALPHA="${VSE_MT_ALPHA:-1.0}"
VSE_LM_ALPHA="${VSE_LM_ALPHA:-1.0}"
SWITCH_EPOCH="${SWITCH_EPOCH:-}"
PHASE1_VSE_MT_ALPHA="${PHASE1_VSE_MT_ALPHA:-}"
PHASE1_VSE_LM_ALPHA="${PHASE1_VSE_LM_ALPHA:-}"
PHASE2_VSE_MT_ALPHA="${PHASE2_VSE_MT_ALPHA:-}"
PHASE2_VSE_LM_ALPHA="${PHASE2_VSE_LM_ALPHA:-}"
RESUME_ARG=()
if [ "$USE_RESUME" = "1" ]; then
  RESUME_ARG+=(--resume)
fi
LOSS_ARGS=(--vse_mt_alpha "$VSE_MT_ALPHA" --vse_lm_alpha "$VSE_LM_ALPHA")
if [ -n "$SWITCH_EPOCH" ]; then
  LOSS_ARGS+=(--switch_epoch "$SWITCH_EPOCH")
fi
if [ -n "$PHASE1_VSE_MT_ALPHA" ]; then
  LOSS_ARGS+=(--phase1_vse_mt_alpha "$PHASE1_VSE_MT_ALPHA")
fi
if [ -n "$PHASE1_VSE_LM_ALPHA" ]; then
  LOSS_ARGS+=(--phase1_vse_lm_alpha "$PHASE1_VSE_LM_ALPHA")
fi
if [ -n "$PHASE2_VSE_MT_ALPHA" ]; then
  LOSS_ARGS+=(--phase2_vse_mt_alpha "$PHASE2_VSE_MT_ALPHA")
fi
if [ -n "$PHASE2_VSE_LM_ALPHA" ]; then
  LOSS_ARGS+=(--phase2_vse_lm_alpha "$PHASE2_VSE_LM_ALPHA")
fi

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
  --logger_name "$SLURM_SUBMIT_DIR/runs/${RUN_PREFIX}_${MODEL}_s${SEED}" \
  --seed "$SEED" \
  --data_path "$DATA_PATH" \
  "${LOSS_ARGS[@]}" \
  --skip_syntactic_bootstrapping \
  "${RESUME_ARG[@]}" \
  $EXTRA_ARGS
