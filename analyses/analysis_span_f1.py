import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt


def parse_list_cell(cell):
    try:
        return json.loads(cell)
    except Exception:
        # fallback for python list repr
        return eval(cell)


def f1(span1, span2):
    overlap = span1.intersection(span2)
    prec = float(len(overlap)) / (len(span1) + 1e-8)
    reca = float(len(overlap)) / (len(span2) + 1e-8)
    if len(span2) == 0:
        reca = 1.0
        if len(span1) == 0:
            prec = 1.0
    return 2 * prec * reca / (prec + reca + 1e-8)


def right_left_baselines(gold_spans):
    # gold_spans length is L-1 for L tokens
    n = len(gold_spans)
    right_spans = [(c, n) for c in range(0, n)]
    left_spans = [(0, c + 1) for c in range(0, n)]
    gold_set = set(tuple(s) for s in gold_spans[:-1])
    right_set = set(right_spans[1:])
    left_set = set(left_spans[:-1])
    return f1(gold_set, right_set), f1(gold_set, left_set)


def read_epoch_csv(csv_path):
    sent_f1 = []
    right_f1 = []
    left_f1 = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            gold_spans = parse_list_cell(row[1])
            sent_f1.append(float(row[3]))
            r_f1, l_f1 = right_left_baselines(gold_spans)
            right_f1.append(r_f1)
            left_f1.append(l_f1)
    return sent_f1, right_f1, left_f1


def collect_runs(run_specs, last_epoch):
    runs = {}
    for label, run_dir in run_specs.items():
        run_dir = Path(run_dir)
        epoch_files = sorted(run_dir.glob("semantic_bootstrapping_results/*.csv"))
        if not epoch_files:
            raise FileNotFoundError(f"No CSVs found in {run_dir}")
        by_epoch = {int(p.stem): p for p in epoch_files}
        epochs = sorted(by_epoch.keys())
        target_epoch = last_epoch if last_epoch is not None else epochs[-1]
        if target_epoch not in by_epoch:
            raise ValueError(f"Epoch {target_epoch} not found in {run_dir}")
        runs[label] = by_epoch
    return runs


def build_table(runs_by_label, last_epoch):
    rows = []
    right_all = []
    left_all = []
    for label, runs in runs_by_label.items():
        sent_means = []
        for run_dir, epoch_map in runs.items():
            sent_f1, right_f1, left_f1 = read_epoch_csv(epoch_map[last_epoch])
            sent_means.append(mean(sent_f1))
            right_all.extend(right_f1)
            left_all.extend(left_f1)
        rows.append((label, mean(sent_means), pstdev(sent_means)))
    right_mean, right_sd = mean(right_all), pstdev(right_all)
    left_mean, left_sd = mean(left_all), pstdev(left_all)
    return rows, (right_mean, right_sd), (left_mean, left_sd)


def build_curve(runs_by_label, max_epoch):
    curves = {}
    for label, runs in runs_by_label.items():
        epochs = sorted(next(iter(runs.values())).keys())
        epochs = [e for e in epochs if e <= max_epoch]
        means = []
        ses = []
        for e in epochs:
            vals = []
            for _, epoch_map in runs.items():
                sent_f1, _, _ = read_epoch_csv(epoch_map[e])
                vals.append(mean(sent_f1))
            mu = mean(vals)
            se = (pstdev(vals) / (len(vals) ** 0.5)) if len(vals) > 1 else 0.0
            means.append(mu)
            ses.append(se)
        curves[label] = (epochs, means, ses)
    return curves


def plot_curves(curves, out_png, switch_epoch=None):
    plt.figure(figsize=(8, 5))
    for label, (epochs, means, ses) in curves.items():
        plt.plot(epochs, means, label=label)
        lower = [m - s for m, s in zip(means, ses)]
        upper = [m + s for m, s in zip(means, ses)]
        plt.fill_between(epochs, lower, upper, alpha=0.2)
    if switch_epoch is not None:
        plt.axvline(switch_epoch, linestyle="--", color="black", linewidth=1)
    plt.xlabel("Epoch")
    plt.ylabel("Mean F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)


def parse_run_args(run_args):
    runs = {}
    for item in run_args:
        if "=" not in item:
            raise ValueError(f"Run spec must be label=path, got {item}")
        label, path = item.split("=", 1)
        runs[label] = Path(path).resolve()
    return runs


def group_runs(run_specs):
    grouped = {}
    for label, run_dir in run_specs.items():
        if not run_dir.exists():
            raise FileNotFoundError(run_dir)
        if run_dir.is_dir() and (run_dir / "semantic_bootstrapping_results").exists():
            grouped.setdefault(label, {})[str(run_dir)] = {
                int(p.stem): p for p in (run_dir / "semantic_bootstrapping_results").glob("*.csv")
            }
        else:
            raise FileNotFoundError(f"Missing semantic_bootstrapping_results in {run_dir}")
    return grouped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True,
                        help="label=path to a run directory")
    parser.add_argument("--last_epoch", type=int, default=None,
                        help="Epoch to use for Table 1 (default: last)")
    parser.add_argument("--max_epoch", type=int, default=30)
    parser.add_argument("--switch_epoch", type=int, default=None)
    parser.add_argument("--out_table", default="table1.csv")
    parser.add_argument("--out_plot", default="figure4.png")
    args = parser.parse_args()

    run_specs = parse_run_args(args.run)
    runs_by_label = group_runs(run_specs)
    last_epoch = args.last_epoch
    if last_epoch is None:
        # use last epoch of first run
        first_run = next(iter(next(iter(runs_by_label.values())).values()))
        last_epoch = max(first_run.keys())

    rows, right_stats, left_stats = build_table(runs_by_label, last_epoch)

    with open(args.out_table, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Mean", "SD"])
        writer.writerow(["Right-branching", right_stats[0], right_stats[1]])
        writer.writerow(["Left-branching", left_stats[0], left_stats[1]])
        for label, mu, sd in rows:
            writer.writerow([label, mu, sd])

    curves = build_curve(runs_by_label, args.max_epoch)
    plot_curves(curves, args.out_plot, switch_epoch=args.switch_epoch)


if __name__ == "__main__":
    main()
