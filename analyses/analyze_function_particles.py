import argparse
import ast
import csv
import json
from pathlib import Path
from statistics import mean


DEFAULT_PARTICLES = ["了", "的", "得", "把", "被", "是"]


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


def load_caps_tokens(data_path):
    caps = {}
    with (data_path / "all_caps.json").open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            cap, _ = json.loads(line)
            caps[i] = cap.split(" ") if cap else []
    return caps


def length_bin(n):
    if n <= 8:
        return "<=8"
    if n <= 12:
        return "9-12"
    return ">=13"


def safe_mean(vals):
    return mean(vals) if vals else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument(
        "--particles",
        default=",".join(DEFAULT_PARTICLES),
        help="Comma-separated token list to stratify on.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    data_path = Path(args.data_path)
    particles = [p for p in args.particles.split(",") if p]
    csv_path = get_epoch_csv(run_dir, args.epoch)

    caps = load_caps_tokens(data_path)
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            idx = int(row[0])
            f1 = float(row[3])
            rows.append((idx, f1))

    by_any = {"has_any": [], "no_any": []}
    by_len_any = {"<=8": {"has_any": [], "no_any": []}, "9-12": {"has_any": [], "no_any": []}, ">=13": {"has_any": [], "no_any": []}}
    by_particle = {p: [] for p in particles}
    by_particle_not = {p: [] for p in particles}

    for idx, f1 in rows:
        toks = caps.get(idx, [])
        n = len(toks)
        b = length_bin(n)
        has_any = any(p in toks for p in particles)
        key = "has_any" if has_any else "no_any"
        by_any[key].append(f1)
        by_len_any[b][key].append(f1)

        for p in particles:
            if p in toks:
                by_particle[p].append(f1)
            else:
                by_particle_not[p].append(f1)

    print(f"Run: {run_dir}")
    print(f"Epoch: {csv_path.stem}")
    print(f"Items: {len(rows)}")
    print(f"Particles: {particles}")
    print(
        f"Any-particle split: has_any={safe_mean(by_any['has_any'])*100:.2f} "
        f"(n={len(by_any['has_any'])}), "
        f"no_any={safe_mean(by_any['no_any'])*100:.2f} (n={len(by_any['no_any'])})"
    )

    print("\nBy length x any-particle:")
    for b in ("<=8", "9-12", ">=13"):
        ha = by_len_any[b]["has_any"]
        na = by_len_any[b]["no_any"]
        print(
            f"{b}: has_any={safe_mean(ha)*100:.2f} (n={len(ha)}), "
            f"no_any={safe_mean(na)*100:.2f} (n={len(na)})"
        )

    print("\nPer-particle split:")
    for p in particles:
        yes = by_particle[p]
        no = by_particle_not[p]
        print(
            f"{p}: has={safe_mean(yes)*100:.2f} (n={len(yes)}), "
            f"not={safe_mean(no)*100:.2f} (n={len(no)}), "
            f"delta={(safe_mean(yes)-safe_mean(no))*100:+.2f}"
        )


if __name__ == "__main__":
    main()
