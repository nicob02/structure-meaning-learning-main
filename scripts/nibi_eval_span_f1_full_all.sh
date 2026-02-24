#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --account=def-eporte2

set -euo pipefail

module purge
module load python/3.10
source ~/venvs/graminduct/bin/activate

cd "$SLURM_SUBMIT_DIR"

OUT_DIR="${OUT_DIR:-results/span_f1/zh_full_all}"
MAX_EPOCH="${MAX_EPOCH:-30}"
mkdir -p "$OUT_DIR"

RUN_ARGS=()
for model in joint sem-first syn-first visual-labels; do
  for seed in 91 214 527 627 1018; do
    run_dir="runs/zh_full_${model}_s${seed}"
    if [ ! -d "$run_dir/semantic_bootstrapping_results" ]; then
      echo "Missing run outputs: $run_dir/semantic_bootstrapping_results"
      exit 1
    fi
    RUN_ARGS+=(--run "${model}=${run_dir}")
  done
done

python analyses/analysis_span_f1.py \
  "${RUN_ARGS[@]}" \
  --max_epoch "$MAX_EPOCH" \
  --switch_epoch 15 \
  --out_table "$OUT_DIR/table1_zh_full.csv" \
  --out_plot "$OUT_DIR/figure4_zh_full.png"

echo "Done. Wrote:"
echo "  $OUT_DIR/table1_zh_full.csv"
echo "  $OUT_DIR/figure4_zh_full.png"
