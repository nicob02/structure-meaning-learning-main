"""Per-width F1 aggregator across all seeds of a single configuration.

Usage:
  python analyses/per_width_f1.py \
      --runs runs/zh_fix1_phase_noLR_30ep_joint_s* \
      --epoch 26
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def parse_list_cell(cell):
    """CSV cells may be JSON-ish or Python-repr-ish. Try JSON first, fall back."""
    try:
        return json.loads(cell)
    except Exception:
        try:
            return eval(cell)  # safe: cells are list-of-tuples produced by us
        except Exception:
            return []


def find_csv(run_dir, epoch):
    p = Path(run_dir) / "semantic_bootstrapping_results" / f"{epoch}.csv"
    if p.exists():
        return p
    cands = sorted((p.parent).glob("*.csv"),
                   key=lambda x: int(x.stem) if x.stem.lstrip('-').isdigit() else -1)
    if not cands:
        return None
    print(f"[warn] epoch {epoch} not found in {run_dir}, using {cands[-1].stem}",
          file=sys.stderr)
    return cands[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="One or more run directories. Globs OK.")
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--out_csv", default=None,
                    help="If set, write per-width metrics to this CSV.")
    args = ap.parse_args()

    per_w = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    n_runs_seen = 0
    n_rows = 0
    for run in args.runs:
        csvf = find_csv(run, args.epoch)
        if csvf is None:
            print(f"[warn] no CSV in {run}", file=sys.stderr)
            continue
        n_runs_seen += 1
        with open(csvf, "r") as f:
            for row in csv.reader(f):
                if not row:
                    continue
                gold = set(tuple(s) for s in parse_list_cell(row[1]))
                pred = set(tuple(s) for s in parse_list_cell(row[2]))
                for s in pred:
                    w = s[1] - s[0]
                    if s in gold:
                        per_w[w]["tp"] += 1
                    else:
                        per_w[w]["fp"] += 1
                for s in gold:
                    w = s[1] - s[0]
                    if s not in pred:
                        per_w[w]["fn"] += 1
                n_rows += 1

    print(f"\nAggregated over {n_runs_seen} run(s), {n_rows} sentences total\n")
    print(f"{'width':>6} {'tp':>8} {'fp':>8} {'fn':>8} "
          f"{'prec':>6} {'recall':>6} {'f1':>6}")
    print("-" * 60)
    rows_out = []
    for w in sorted(per_w):
        d = per_w[w]
        p = d["tp"] / max(1, d["tp"] + d["fp"])
        r = d["tp"] / max(1, d["tp"] + d["fn"])
        f1 = 2 * p * r / max(1e-8, p + r)
        print(f"{w:>6} {d['tp']:>8} {d['fp']:>8} {d['fn']:>8} "
              f"{p:>6.3f} {r:>6.3f} {f1:>6.3f}")
        rows_out.append([w, d["tp"], d["fp"], d["fn"], p, r, f1])

    if args.out_csv:
        with open(args.out_csv, "w", newline="") as f:
            w_csv = csv.writer(f)
            w_csv.writerow(["width", "tp", "fp", "fn", "precision", "recall", "f1"])
            w_csv.writerows(rows_out)
        print(f"\nWritten to {args.out_csv}")


if __name__ == "__main__":
    main()
