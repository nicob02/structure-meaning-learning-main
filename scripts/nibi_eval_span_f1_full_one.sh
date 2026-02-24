#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --account=def-eporte2

set -euo pipefail

module purge
module load python/3.10
source ~/venvs/graminduct/bin/activate

cd "$SLURM_SUBMIT_DIR"

RUN_DIR="${RUN_DIR:-runs/zh_full_joint_s1018}"
OUT_DIR="${OUT_DIR:-results/span_f1/zh_full_joint_s1018}"
MAX_EPOCH="${MAX_EPOCH:-30}"

mkdir -p "$OUT_DIR"

python - <<'PY'
import csv
from pathlib import Path
from statistics import mean
import os

run_dir = Path(os.environ["RUN_DIR"])
out_dir = Path(os.environ["OUT_DIR"])
out_csv = out_dir / "epoch_mean_f1.csv"

res_dir = run_dir / "semantic_bootstrapping_results"
if not res_dir.exists():
    raise FileNotFoundError(f"Missing directory: {res_dir}")

rows = []
for p in sorted(res_dir.glob("*.csv"), key=lambda x: int(x.stem)):
    epoch = int(p.stem)
    vals = []
    with p.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for r in reader:
            if not r:
                continue
            vals.append(float(r[3]))
    if vals:
        rows.append((epoch, mean(vals), len(vals)))

with out_csv.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["epoch", "mean_sentence_f1", "num_sentences"])
    for r in rows:
        w.writerow(r)

print(f"Wrote {out_csv}")
PY

python analyses/analysis_span_f1.py \
  --run "joint=$RUN_DIR" \
  --max_epoch "$MAX_EPOCH" \
  --out_table "$OUT_DIR/table1_single_run.csv" \
  --out_plot "$OUT_DIR/figure4_single_run.png"

echo "Done. Outputs in: $OUT_DIR"
