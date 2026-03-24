import argparse
import ast
import csv
import itertools
import json
from pathlib import Path


def parse_cell(cell):
    try:
        return json.loads(cell)
    except Exception:
        return ast.literal_eval(cell)


def get_epoch_csv(run_dir, epoch):
    sem_dir = run_dir / "semantic_bootstrapping_results"
    files = sorted(sem_dir.glob("*.csv"), key=lambda p: int(p.stem))
    if not files:
        raise FileNotFoundError(f"No eval CSVs in {sem_dir}")
    if epoch is None:
        return files[-1]
    target = sem_dir / f"{epoch}.csv"
    if not target.exists():
        raise FileNotFoundError(target)
    return target


def load_f1_by_id(csv_path):
    out = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            idx = int(row[0])
            f1 = float(row[3])
            out.append((idx, f1))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="label=run_dir")
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=1000)
    parser.add_argument("--show_common", type=int, default=30)
    args = parser.parse_args()

    runs = {}
    for item in args.run:
        if "=" not in item:
            raise ValueError(f"Expected label=run_dir, got: {item}")
        label, path = item.split("=", 1)
        runs[label] = Path(path)

    worst = {}
    for label, run_dir in runs.items():
        csv_path = get_epoch_csv(run_dir, args.epoch)
        vals = load_f1_by_id(csv_path)
        vals.sort(key=lambda x: (x[1], x[0]))  # lowest f1 first
        worst[label] = vals[: args.top_k]
        print(
            f"{label}: epoch={csv_path.stem}, top_k={args.top_k}, "
            f"worst_f1_range=[{worst[label][0][1]:.4f}, {worst[label][-1][1]:.4f}]"
        )

    worst_sets = {k: {idx for idx, _ in v} for k, v in worst.items()}

    print("\nPairwise overlap:")
    for a, b in itertools.combinations(worst_sets.keys(), 2):
        ia = worst_sets[a]
        ib = worst_sets[b]
        inter = ia.intersection(ib)
        union = ia.union(ib)
        jacc = len(inter) / max(1, len(union))
        print(
            f"{a} vs {b}: overlap={len(inter)}/{args.top_k} "
            f"({len(inter)/max(1,args.top_k):.2%}), jaccard={jacc:.3f}"
        )

    common_all = set.intersection(*worst_sets.values()) if worst_sets else set()
    print(
        f"\nCommon worst IDs across ALL runs: {len(common_all)}/{args.top_k} "
        f"({len(common_all)/max(1,args.top_k):.2%})"
    )
    if common_all:
        print("Sample common IDs:", sorted(list(common_all))[: args.show_common])


if __name__ == "__main__":
    main()
