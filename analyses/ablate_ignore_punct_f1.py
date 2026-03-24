import argparse
import ast
import csv
import json
from pathlib import Path
from statistics import mean


PUNCT_TOKENS = {"。", "！", "？", "!", "?", ".", "｡", "．", "…", "，", ",", "、", "；", ";", "：", ":"}


def parse_cell(cell):
    try:
        return json.loads(cell)
    except Exception:
        return ast.literal_eval(cell)


def sent_f1(pred_set, gold_set):
    overlap = pred_set.intersection(gold_set)
    prec = float(len(overlap)) / (len(pred_set) + 1e-8)
    reca = float(len(overlap)) / (len(gold_set) + 1e-8)
    if len(gold_set) == 0:
        reca = 1.0
        if len(pred_set) == 0:
            prec = 1.0
    return 2 * prec * reca / (prec + reca + 1e-8)


def load_caps_tokens(data_path):
    caps = {}
    with (data_path / "all_caps.json").open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            cap, _ = json.loads(line)
            caps[i] = cap.split(" ") if cap else []
    return caps


def load_rows(csv_path):
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            idx = int(row[0])
            gold = [tuple(x) for x in parse_cell(row[1])]
            pred = [tuple(x) for x in parse_cell(row[2])]
            rows.append((idx, gold, pred))
    return rows


def filter_spans(spans, punct_idx):
    out = []
    for l, r in spans:
        # Drop any span that touches punctuation anywhere in coverage [l, r].
        if any(l <= pi <= r for pi in punct_idx):
            continue
        out.append((l, r))
    return out


def corpus_f1(tp, fp, fn):
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    return 2 * prec * rec / (prec + rec + 1e-8)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--epoch", type=int, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    data_path = Path(args.data_path)
    csv_path = get_epoch_csv(run_dir, args.epoch)

    caps = load_caps_tokens(data_path)
    rows = load_rows(csv_path)

    orig_sent = []
    filt_sent = []
    removed_gold_spans = 0
    removed_pred_spans = 0
    total_gold_spans = 0
    total_pred_spans = 0

    o_tp = o_fp = o_fn = 0
    f_tp = f_fp = f_fn = 0

    for idx, gold, pred in rows:
        toks = caps.get(idx, [])
        punct_idx = {i for i, t in enumerate(toks) if t in PUNCT_TOKENS}

        gold_set = set(gold[:-1]) if gold else set()
        pred_set = set(pred[:-1]) if pred else set()
        orig_sent.append(sent_f1(pred_set, gold_set))

        o_overlap = pred_set.intersection(gold_set)
        o_tp += len(o_overlap)
        o_fp += len(pred_set - gold_set)
        o_fn += len(gold_set - pred_set)

        fg = set(filter_spans(list(gold_set), punct_idx))
        fp = set(filter_spans(list(pred_set), punct_idx))
        filt_sent.append(sent_f1(fp, fg))

        f_overlap = fp.intersection(fg)
        f_tp += len(f_overlap)
        f_fp += len(fp - fg)
        f_fn += len(fg - fp)

        total_gold_spans += len(gold_set)
        total_pred_spans += len(pred_set)
        removed_gold_spans += len(gold_set) - len(fg)
        removed_pred_spans += len(pred_set) - len(fp)

    print(f"Run: {run_dir}")
    print(f"Epoch: {csv_path.stem}")
    print(f"Items: {len(rows)}")
    print(f"Original sentence-F1 mean: {mean(orig_sent)*100:.2f}")
    print(f"Punct-ignored sentence-F1 mean: {mean(filt_sent)*100:.2f}")
    print(f"Delta (ignored - original): {(mean(filt_sent)-mean(orig_sent))*100:+.2f}")
    print(f"Original corpus-F1: {corpus_f1(o_tp, o_fp, o_fn)*100:.2f}")
    print(f"Punct-ignored corpus-F1: {corpus_f1(f_tp, f_fp, f_fn)*100:.2f}")
    print(
        "Removed spans "
        f"(gold={removed_gold_spans}/{max(1,total_gold_spans)}="
        f"{removed_gold_spans/max(1,total_gold_spans):.2%}, "
        f"pred={removed_pred_spans}/{max(1,total_pred_spans)}="
        f"{removed_pred_spans/max(1,total_pred_spans):.2%})"
    )


if __name__ == "__main__":
    main()
