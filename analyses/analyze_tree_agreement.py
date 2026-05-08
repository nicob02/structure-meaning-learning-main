"""Compare predicted parses across runs that reach similar F1.

For each pair of run directories, this script:
  1. Loads the per-sentence prediction CSV from each run (same epoch).
  2. Aligns sentences by id.
  3. For each pair (run_i, run_j), computes:
       - Constituent agreement: average per-sentence F1 of pred_i vs pred_j
         (treats run_j's predictions as if they were gold).
       - Span-level Jaccard: |pred_i ∩ pred_j| / |pred_i ∪ pred_j|.
       - Per-length precision/recall against gold and against each other.
  4. Identifies systematic errors:
       - Spans predicted by ALL runs but absent from gold ("shared FP").
       - Spans in gold but missed by ALL runs ("shared FN").
       - Frequency of specific (start, end) span types in shared errors.

Outputs:
  - <out_dir>/agreement_matrix.csv  (mean F1 of run_i vs run_j over all sentences)
  - <out_dir>/per_length_metrics.csv (precision/recall by span width)
  - <out_dir>/shared_errors.csv     (top recurring error spans)
  - Plot: <out_dir>/agreement_heatmap.png

Usage:
  python analyses/analyze_tree_agreement.py \
      --run s91=runs/zh_fix1_phase_noLR_30ep_joint_s91 \
      --run s214=runs/zh_fix1_phase_noLR_30ep_joint_s214 \
      --run s527=runs/zh_fix1_phase_noLR_30ep_joint_s527 \
      --run s627=runs/zh_fix1_phase_noLR_30ep_joint_s627 \
      --run s1018=runs/zh_fix1_phase_noLR_30ep_joint_s1018 \
      --epoch 26 \
      --out_dir analyses/results/phase_noLR_agreement
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


def parse_list(cell):
    try:
        return json.loads(cell)
    except Exception:
        return eval(cell)


def f1_score(pred, gold):
    pred_set = set(tuple(s) for s in pred)
    gold_set = set(tuple(s) for s in gold)
    overlap = pred_set & gold_set
    p = len(overlap) / (len(pred_set) + 1e-8)
    r = len(overlap) / (len(gold_set) + 1e-8)
    if not gold_set:
        r = 1.0
        if not pred_set:
            p = 1.0
    return 2 * p * r / (p + r + 1e-8)


def jaccard(a, b):
    sa = set(tuple(s) for s in a)
    sb = set(tuple(s) for s in b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


def load_run(run_dir, epoch):
    csv_path = Path(run_dir) / "semantic_bootstrapping_results" / f"{epoch}.csv"
    if not csv_path.exists():
        # fall back to last available
        cands = sorted((Path(run_dir) / "semantic_bootstrapping_results").glob("*.csv"))
        if not cands:
            raise FileNotFoundError(f"No CSVs in {run_dir}")
        csv_path = cands[-1]
        print(f"[warn] epoch {epoch} not found in {run_dir}, using {csv_path.stem}")
    rows = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            sid = row[0]
            gold = parse_list(row[1])
            pred = parse_list(row[2])
            rows[sid] = (gold, pred)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    help="label=path/to/run_dir")
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    runs = {}
    for spec in args.run:
        label, path = spec.split("=", 1)
        runs[label] = load_run(path, args.epoch)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = list(runs.keys())

    # 1. Pairwise agreement matrix (F1 of run_i preds vs run_j preds, ignoring gold)
    agreement = {}
    for i in labels:
        for j in labels:
            common_ids = set(runs[i]) & set(runs[j])
            scores = []
            for sid in common_ids:
                _, pi = runs[i][sid]
                _, pj = runs[j][sid]
                scores.append(f1_score(pi, pj))
            agreement[(i, j)] = sum(scores) / max(1, len(scores))
    with open(out_dir / "agreement_matrix.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + labels)
        for i in labels:
            w.writerow([i] + [f"{agreement[(i, j)]:.4f}" for j in labels])

    if HAS_MPL:
        n = len(labels)
        mat = [[agreement[(labels[i], labels[j])] for j in range(n)] for i in range(n)]
        plt.figure(figsize=(6, 5))
        plt.imshow(mat, cmap="viridis", vmin=0.5, vmax=1.0)
        plt.colorbar(label="Pred-pred F1")
        plt.xticks(range(n), labels, rotation=45)
        plt.yticks(range(n), labels)
        for i in range(n):
            for j in range(n):
                plt.text(j, i, f"{mat[i][j]:.2f}", ha="center", va="center",
                         color="white" if mat[i][j] < 0.75 else "black")
        plt.title(f"Per-sentence F1 between predictions (epoch {args.epoch})")
        plt.tight_layout()
        plt.savefig(out_dir / "agreement_heatmap.png", dpi=200)
        plt.close()

    # 2. F1 vs gold by span length
    per_length = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for label, rows in runs.items():
        for sid, (gold, pred) in rows.items():
            gold_set = set(tuple(s) for s in gold)
            pred_set = set(tuple(s) for s in pred)
            for s in pred_set:
                w = s[1] - s[0]
                if s in gold_set:
                    per_length[(label, w)]["tp"] += 1
                else:
                    per_length[(label, w)]["fp"] += 1
            for s in gold_set:
                w = s[1] - s[0]
                if s not in pred_set:
                    per_length[(label, w)]["fn"] += 1

    with open(out_dir / "per_length_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "span_width", "tp", "fp", "fn", "precision", "recall", "f1"])
        for (label, width), d in sorted(per_length.items()):
            tp, fp, fn = d["tp"], d["fp"], d["fn"]
            p = tp / max(1, tp + fp)
            r = tp / max(1, tp + fn)
            f1 = 2 * p * r / max(1e-8, p + r)
            w.writerow([label, width, tp, fp, fn, f"{p:.4f}", f"{r:.4f}", f"{f1:.4f}"])

    # 3. Shared errors (across all runs)
    common_ids = set.intersection(*[set(r) for r in runs.values()])
    shared_fp = Counter()
    shared_fn = Counter()
    for sid in common_ids:
        gold_set = set(tuple(s) for s in runs[labels[0]][sid][0])
        pred_sets = [set(tuple(s) for s in runs[lab][sid][1]) for lab in labels]
        # FP shared: in every pred set but not in gold
        common_pred = set.intersection(*pred_sets)
        for s in common_pred - gold_set:
            shared_fp[(s[1] - s[0], s[0], s[1])] += 1
        # FN shared: in gold but in NO pred set
        union_pred = set.union(*pred_sets)
        for s in gold_set - union_pred:
            shared_fn[(s[1] - s[0], s[0], s[1])] += 1

    with open(out_dir / "shared_errors.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["error_type", "span_width", "start", "end", "count"])
        for (width, start, end), c in shared_fp.most_common(50):
            w.writerow(["shared_FP", width, start, end, c])
        for (width, start, end), c in shared_fn.most_common(50):
            w.writerow(["shared_FN", width, start, end, c])

    # 4. Console summary
    print(f"\n=== Pairwise prediction-prediction F1 (epoch {args.epoch}) ===")
    print(f"{'':12s}" + "".join(f"{l:>10s}" for l in labels))
    for i in labels:
        print(f"{i:12s}" + "".join(f"{agreement[(i, j)]:>10.4f}" for j in labels))

    print(f"\n=== Top 10 shared false positives (predicted by ALL but not in gold) ===")
    for (width, start, end), c in shared_fp.most_common(10):
        print(f"  span ({start},{end}) width={width}  count={c}")

    print(f"\n=== Top 10 shared false negatives (in gold but missed by ALL) ===")
    for (width, start, end), c in shared_fn.most_common(10):
        print(f"  span ({start},{end}) width={width}  count={c}")

    print(f"\nResults written to {out_dir}/")


if __name__ == "__main__":
    main()
