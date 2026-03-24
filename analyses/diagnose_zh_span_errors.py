import argparse
import ast
import csv
import json
from pathlib import Path
from statistics import mean


TERMINAL_PUNCT = {"。", "！", "？", "!", "?", ".", "｡", "．", "…"}


def parse_cell(cell):
    try:
        return json.loads(cell)
    except Exception:
        return ast.literal_eval(cell)


def read_eval_rows(csv_path):
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            idx = int(row[0])
            gold = [tuple(x) for x in parse_cell(row[1])]
            pred = [tuple(x) for x in parse_cell(row[2])]
            sent_f1 = float(row[3])
            rows.append((idx, gold, pred, sent_f1))
    return rows


def load_caption_maps(data_path):
    caps = {}
    gold_meta = {}

    caps_path = data_path / "all_caps.json"
    with caps_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            cap, _ = json.loads(line)
            tokens = cap.split(" ") if cap else []
            caps[i] = tokens

    gold_path = data_path / "all_gold_caps.json"
    with gold_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            cap, tree_spans, tree_labels, pos_tags = json.loads(line)
            gold_meta[i] = {
                "caption": cap,
                "tree_spans": tree_spans,
                "tree_labels": tree_labels,
                "pos_tags": pos_tags,
            }
    return caps, gold_meta


def bin_name(length):
    if length <= 8:
        return "<=8"
    if length <= 12:
        return "9-12"
    return ">=13"


def safe_mean(vals):
    return mean(vals) if vals else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, help="Path to a run dir under runs/")
    parser.add_argument("--data_path", required=True, help="Path to dataset (e.g. preprocessed-data/abstractscenes_zh_tree_simp_v2000)")
    parser.add_argument("--epoch", type=int, default=None, help="Epoch to inspect. Default: latest.")
    parser.add_argument("--top_k", type=int, default=15, help="Number of worst examples to print.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    data_path = Path(args.data_path)
    sem_dir = run_dir / "semantic_bootstrapping_results"
    csvs = sorted(sem_dir.glob("*.csv"), key=lambda p: int(p.stem))
    if not csvs:
        raise FileNotFoundError(f"No semantic eval CSVs in {sem_dir}")

    if args.epoch is None:
        csv_path = csvs[-1]
    else:
        csv_path = sem_dir / f"{args.epoch}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)

    rows = read_eval_rows(csv_path)
    caps, gold_meta = load_caption_maps(data_path)

    all_f1 = []
    punct_f1 = []
    no_punct_f1 = []
    by_len = {"<=8": [], "9-12": [], ">=13": []}
    invalid_gold = 0
    invalid_pred = 0
    diagnostics = []

    for idx, gold, pred, sent_f1 in rows:
        toks = caps.get(idx, [])
        n = len(toks)
        if n >= 2 and len(gold) != n - 1:
            invalid_gold += 1
        if n >= 2 and len(pred) != n - 1:
            invalid_pred += 1
        final_tok = toks[-1] if toks else ""
        ends_punct = final_tok in TERMINAL_PUNCT
        all_f1.append(sent_f1)
        by_len[bin_name(n)].append(sent_f1)
        if ends_punct:
            punct_f1.append(sent_f1)
        else:
            no_punct_f1.append(sent_f1)

        gold_set = set(gold[:-1]) if gold else set()
        pred_set = set(pred[:-1]) if pred else set()
        missed = sorted(gold_set - pred_set)
        extra = sorted(pred_set - gold_set)
        diagnostics.append(
            {
                "idx": idx,
                "f1": sent_f1,
                "len": n,
                "final_tok": final_tok,
                "ends_punct": ends_punct,
                "caption": " ".join(toks),
                "missed": missed,
                "extra": extra,
                "gold_n": len(gold),
                "pred_n": len(pred),
                "pos_tail": gold_meta.get(idx, {}).get("pos_tags", [])[-3:],
            }
        )

    print(f"Run: {run_dir}")
    print(f"Dataset: {data_path}")
    print(f"Epoch file: {csv_path.name}")
    print(f"Items: {len(rows)}")
    print(f"Mean sentence F1: {safe_mean(all_f1) * 100:.2f}")
    print(
        "By final punctuation: "
        f"punct={safe_mean(punct_f1) * 100:.2f} (n={len(punct_f1)}), "
        f"no_punct={safe_mean(no_punct_f1) * 100:.2f} (n={len(no_punct_f1)})"
    )
    print(
        "By length: "
        + ", ".join(
            f"{k}={safe_mean(v) * 100:.2f} (n={len(v)})"
            for k, v in by_len.items()
        )
    )
    print(
        f"Span cardinality mismatches: gold={invalid_gold}, pred={invalid_pred}"
    )

    diagnostics.sort(key=lambda x: x["f1"])
    print("\nWorst examples:")
    for item in diagnostics[: args.top_k]:
        print(
            f"- idx={item['idx']} f1={item['f1'] * 100:.2f} len={item['len']} "
            f"final={item['final_tok']} punct={item['ends_punct']} "
            f"gold_n={item['gold_n']} pred_n={item['pred_n']}"
        )
        print(f"  caption: {item['caption']}")
        print(f"  pos_tail: {item['pos_tail']}")
        print(f"  missed[:8]={item['missed'][:8]}")
        print(f"  extra[:8]={item['extra'][:8]}")


if __name__ == "__main__":
    main()
