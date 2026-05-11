#!/bin/bash
#SBATCH --time=09:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
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
NUM_EPOCHS="${NUM_EPOCHS:-30}"
USE_STRUCT_NEG="${USE_STRUCT_NEG:-0}"
STRUCT_NEG_MARGIN="${STRUCT_NEG_MARGIN:-}"
STRUCT_NEG_WEIGHT="${STRUCT_NEG_WEIGHT:-}"
STRUCT_NEG_STYLE="${STRUCT_NEG_STYLE:-}"
USE_MI_REG="${USE_MI_REG:-0}"
MI_MARGIN="${MI_MARGIN:-}"
MI_WEIGHT="${MI_WEIGHT:-}"
MI_STYLE="${MI_STYLE:-}"
USE_TEMP_ANNEAL="${USE_TEMP_ANNEAL:-0}"
TEMP_START="${TEMP_START:-}"
TEMP_END="${TEMP_END:-}"
TEMP_ANNEAL_FRAC="${TEMP_ANNEAL_FRAC:-}"
TEMP_MODE="${TEMP_MODE:-}"
USE_ENTROPY_BONUS="${USE_ENTROPY_BONUS:-0}"
ENTROPY_WEIGHT="${ENTROPY_WEIGHT:-}"
ENTROPY_ANNEAL_FRAC="${ENTROPY_ANNEAL_FRAC:-}"
ENTROPY_MODE="${ENTROPY_MODE:-}"
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
EXTRA_OBJ_ARGS=()
if [ "$USE_STRUCT_NEG" = "1" ]; then
  EXTRA_OBJ_ARGS+=(--use_structural_negatives)
  [ -n "$STRUCT_NEG_MARGIN" ] && EXTRA_OBJ_ARGS+=(--struct_neg_margin "$STRUCT_NEG_MARGIN")
  [ -n "$STRUCT_NEG_WEIGHT" ] && EXTRA_OBJ_ARGS+=(--struct_neg_weight "$STRUCT_NEG_WEIGHT")
  [ -n "$STRUCT_NEG_STYLE" ] && EXTRA_OBJ_ARGS+=(--struct_neg_style "$STRUCT_NEG_STYLE")
fi
if [ "$USE_MI_REG" = "1" ]; then
  EXTRA_OBJ_ARGS+=(--use_mi_regularizer)
  [ -n "$MI_MARGIN" ] && EXTRA_OBJ_ARGS+=(--mi_margin "$MI_MARGIN")
  [ -n "$MI_WEIGHT" ] && EXTRA_OBJ_ARGS+=(--mi_weight "$MI_WEIGHT")
  [ -n "$MI_STYLE" ] && EXTRA_OBJ_ARGS+=(--mi_style "$MI_STYLE")
fi
if [ "$USE_TEMP_ANNEAL" = "1" ]; then
  EXTRA_OBJ_ARGS+=(--use_temperature_annealing)
  [ -n "$TEMP_START" ] && EXTRA_OBJ_ARGS+=(--temp_start "$TEMP_START")
  [ -n "$TEMP_END" ] && EXTRA_OBJ_ARGS+=(--temp_end "$TEMP_END")
  [ -n "$TEMP_ANNEAL_FRAC" ] && EXTRA_OBJ_ARGS+=(--temp_anneal_frac "$TEMP_ANNEAL_FRAC")
  [ -n "$TEMP_MODE" ] && EXTRA_OBJ_ARGS+=(--temp_mode "$TEMP_MODE")
fi
if [ "$USE_ENTROPY_BONUS" = "1" ]; then
  EXTRA_OBJ_ARGS+=(--use_entropy_bonus)
  [ -n "$ENTROPY_WEIGHT" ] && EXTRA_OBJ_ARGS+=(--entropy_weight "$ENTROPY_WEIGHT")
  [ -n "$ENTROPY_ANNEAL_FRAC" ] && EXTRA_OBJ_ARGS+=(--entropy_anneal_frac "$ENTROPY_ANNEAL_FRAC")
  [ -n "$ENTROPY_MODE" ] && EXTRA_OBJ_ARGS+=(--entropy_mode "$ENTROPY_MODE")
fi

LR_PARSER="${LR_PARSER:-}"
LR_TXT_ENC="${LR_TXT_ENC:-}"
LR_IMG_ENC="${LR_IMG_ENC:-}"
PARSER_GRAD_NOISE="${PARSER_GRAD_NOISE:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
[ -n "$LR_PARSER" ] && EXTRA_OBJ_ARGS+=(--lr_parser "$LR_PARSER")
[ -n "$LR_TXT_ENC" ] && EXTRA_OBJ_ARGS+=(--lr_txt_enc "$LR_TXT_ENC")
[ -n "$LR_IMG_ENC" ] && EXTRA_OBJ_ARGS+=(--lr_img_enc "$LR_IMG_ENC")
[ -n "$PARSER_GRAD_NOISE" ] && EXTRA_OBJ_ARGS+=(--parser_grad_noise "$PARSER_GRAD_NOISE")
[ -n "$BATCH_SIZE" ] && EXTRA_OBJ_ARGS+=(--batch_size "$BATCH_SIZE")

# Architectural priors against the left-branching attractor.
BRANCHING_WEIGHT="${BRANCHING_WEIGHT:-}"
RIGHT_PRIOR_WEIGHT="${RIGHT_PRIOR_WEIGHT:-}"
BRANCHING_INIT="${BRANCHING_INIT:-}"
BRANCHING_INIT_MODE="${BRANCHING_INIT_MODE:-}"
REVERSE_TEXT="${REVERSE_TEXT:-0}"
[ -n "$BRANCHING_WEIGHT" ] && EXTRA_OBJ_ARGS+=(--branching_weight "$BRANCHING_WEIGHT")
[ -n "$RIGHT_PRIOR_WEIGHT" ] && EXTRA_OBJ_ARGS+=(--right_prior_weight "$RIGHT_PRIOR_WEIGHT")
[ -n "$BRANCHING_INIT" ] && EXTRA_OBJ_ARGS+=(--branching_init "$BRANCHING_INIT")
[ -n "$BRANCHING_INIT_MODE" ] && EXTRA_OBJ_ARGS+=(--branching_init_mode "$BRANCHING_INIT_MODE")
[ "$REVERSE_TEXT" = "1" ] && EXTRA_OBJ_ARGS+=(--reverse_text)

# H2 test: SGD on parser instead of Adam.
PARSER_OPTIM="${PARSER_OPTIM:-}"
PARSER_SGD_MOMENTUM="${PARSER_SGD_MOMENTUM:-}"
[ -n "$PARSER_OPTIM" ] && EXTRA_OBJ_ARGS+=(--parser_optim "$PARSER_OPTIM")
[ -n "$PARSER_SGD_MOMENTUM" ] && EXTRA_OBJ_ARGS+=(--parser_sgd_momentum "$PARSER_SGD_MOMENTUM")

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
  --num_epochs "$NUM_EPOCHS" \
  --encoder_file "$ENCODER_FILE" \
  --img_dim "$IMG_DIM" \
  --visual_mode \
  --logger_name "$SLURM_SUBMIT_DIR/runs/${RUN_PREFIX}_${MODEL}_s${SEED}" \
  --seed "$SEED" \
  --data_path "$DATA_PATH" \
  "${LOSS_ARGS[@]}" \
  "${EXTRA_OBJ_ARGS[@]}" \
  --skip_syntactic_bootstrapping \
  "${RESUME_ARG[@]}" \
  $EXTRA_ARGS
