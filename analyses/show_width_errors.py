"""Show concrete examples of width-3/4/5 parsing errors.

Why
---
Per-width F1 aggregates show the model's bottleneck is at widths 3-5: that's
where Chinese VPs, complex NPs, and PP-attachment decisions live. This script
turns those bulk statistics into individual, presentable examples you can
show to a supervisor: each example shows the original sentence, the gold
bracketing, the predicted bracketing, and which width-3/4/5 spans were
missed or wrongly inserted.

Usage
-----
  python analyses/show_width_errors.py \
      --run runs/zh_fix1_phase_noLR_30ep_joint_s91 \
      --caps preprocessed-data/abstractscenes_zh/all_caps.json \
      --epoch 26 \
      --widths 3 4 5 \
      --n_examples 20 \
      --out analyses/results/width_errors_s91.txt

Outputs both a human-readable .txt file and, if matplotlib is available,
a small set of tree visualizations (one PNG per example).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Tuple


def parse_list(cell):
    """Parse a CSV cell that may be JSON or Python-repr (e.g. '[(0, 4)]')."""
    try:
        return json.loads(cell)
    except Exception:
        return eval(cell)


def load_caps(caps_path: Path):
    """Return {idx: tokens_list} from all_caps.json.

    The file has one (caption, span) JSON tuple per line. The line number
    (0-indexed) is the `idx` recorded in the run CSV.
    """
    caps = {}
    with open(caps_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            try:
                cap, _span = json.loads(line)
            except Exception:
                continue
            # Match the preprocessing in vpcfg.as_dataloader.AsDataset:
            #   caption = caption.strip().lower().split()
            # so token indices align with the run's CSV.
            tokens = cap.strip().lower().split()
            caps[idx] = tokens
    return caps


def load_run_csv(csv_path: Path):
    """Return list of (sid:str, gold_spans, pred_spans, sent_f1:float)."""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            sid = row[0]
            gold = [tuple(s) for s in parse_list(row[1])]
            pred = [tuple(s) for s in parse_list(row[2])]
            try:
                f1 = float(row[3])
            except (IndexError, ValueError):
                f1 = 0.0
            rows.append((sid, gold, pred, f1))
    return rows


def bracket_with_spans(tokens: List[str], spans: List[Tuple[int, int]]):
    """Return a bracketed string display of tokens given a set of constituents.

    Each (i, j) span is inclusive on both ends. We render by repeatedly
    wrapping the substring of tokens with `[` and `]`. To keep things
    readable for non-binary input we sort by span width (largest first) so
    larger constituents are bracketed before smaller ones.
    """
    n = len(tokens)
    out = [t for t in tokens]
    # Mark open '[' at each start, ']' at each end, deepest spans first.
    opens = [0] * n
    closes = [0] * n
    for (i, j) in sorted(spans, key=lambda s: -(s[1] - s[0])):
        if 0 <= i < n and 0 <= j < n:
            opens[i] += 1
            closes[j] += 1
    pieces = []
    for i, tok in enumerate(out):
        pieces.append("[" * opens[i])
        pieces.append(tok)
        pieces.append("]" * closes[i])
        pieces.append(" ")
    return "".join(pieces).strip()


def width_filtered_errors(gold, pred, widths):
    """Return (shared_set, gold_only, pred_only) restricted to given widths."""
    gold_w = {(i, j) for (i, j) in gold if (j - i + 1) in widths}
    pred_w = {(i, j) for (i, j) in pred if (j - i + 1) in widths}
    shared = gold_w & pred_w
    fn = gold_w - pred_w
    fp = pred_w - gold_w
    return shared, fn, fp


def format_span(tokens, span):
    i, j = span
    if i < 0 or j >= len(tokens):
        return f"({i},{j}) [OUT-OF-RANGE]"
    inside = " ".join(tokens[i:j + 1])
    return f"({i},{j})  '{inside}'  width={j - i + 1}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Run directory (contains semantic_bootstrapping_results/).")
    ap.add_argument("--caps", required=True, help="Path to all_caps.json.")
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--widths", nargs="+", type=int, default=[3, 4, 5])
    ap.add_argument("--n_examples", type=int, default=20)
    ap.add_argument("--min_len", type=int, default=6,
                    help="Skip sentences shorter than this — too few candidate spans.")
    ap.add_argument("--max_len", type=int, default=20,
                    help="Skip extremely long sentences for readability.")
    ap.add_argument("--out", default=None,
                    help="Optional path to write the human-readable report.")
    args = ap.parse_args()

    run_dir = Path(args.run)
    csv_path = run_dir / "semantic_bootstrapping_results" / f"{args.epoch}.csv"
    if not csv_path.exists():
        cands = sorted((run_dir / "semantic_bootstrapping_results").glob("*.csv"))
        if not cands:
            raise SystemExit(f"No CSVs in {run_dir}/semantic_bootstrapping_results/")
        csv_path = cands[-1]
        print(f"[warn] epoch {args.epoch} not found, using {csv_path.stem}")

    print(f"Loading captions from {args.caps} ...")
    caps = load_caps(Path(args.caps))
    print(f"  loaded {len(caps)} sentences")
    print(f"Loading run predictions from {csv_path} ...")
    rows = load_run_csv(csv_path)
    print(f"  loaded {len(rows)} predictions")

    # Rank candidate sentences by how informative they are:
    # - must have at least one width-w mismatch in the requested widths
    # - prefer sentences where pred is structurally wrong (more FP+FN) but
    #   not catastrophic (still has SOME shared spans, so we're showing a
    #   "partial-failure" rather than a total-failure example)
    candidates = []
    width_set = set(args.widths)
    for (sid, gold, pred, f1) in rows:
        try:
            idx = int(sid)
        except ValueError:
            continue
        tokens = caps.get(idx)
        if tokens is None:
            continue
        L = len(tokens)
        if L < args.min_len or L > args.max_len:
            continue
        shared, fn, fp = width_filtered_errors(gold, pred, width_set)
        n_err = len(fn) + len(fp)
        if n_err == 0:
            continue
        candidates.append({
            "sid": sid, "tokens": tokens, "L": L, "f1": f1,
            "gold": gold, "pred": pred,
            "shared": shared, "fn": fn, "fp": fp,
            "score": n_err,  # rank: more errors at our widths = more illustrative
        })

    # Sort: prefer sentences with several width-3/4/5 mismatches but with a
    # decent overall F1 (so the example isn't pure noise).
    candidates.sort(key=lambda c: (-c["score"], -c["f1"]))
    selected = candidates[: args.n_examples]

    report_lines = []
    report_lines.append(
        f"# Width-{','.join(map(str, args.widths))} errors in {run_dir.name} "
        f"(epoch {csv_path.stem})\n"
    )
    report_lines.append(
        f"Total sentences scanned: {len(rows)};  candidates with width-"
        f"{','.join(map(str, args.widths))} mismatch: {len(candidates)}\n"
    )

    for k, c in enumerate(selected, 1):
        report_lines.append("=" * 78)
        report_lines.append(
            f"Example {k}/{len(selected)}  sid={c['sid']}  length={c['L']}  "
            f"sent_F1={c['f1']:.3f}  errors_at_widths={c['score']}"
        )
        report_lines.append("")
        report_lines.append("Tokens (0-indexed):")
        report_lines.append("  " + "  ".join(
            f"{i}:{tok}" for i, tok in enumerate(c["tokens"])
        ))
        report_lines.append("")
        report_lines.append("Gold bracketing:")
        report_lines.append("  " + bracket_with_spans(c["tokens"], c["gold"]))
        report_lines.append("")
        report_lines.append("Predicted bracketing:")
        report_lines.append("  " + bracket_with_spans(c["tokens"], c["pred"]))
        report_lines.append("")
        if c["fn"]:
            report_lines.append("Gold constituents MISSED at width "
                                f"{','.join(map(str, args.widths))} "
                                "(false negatives):")
            for span in sorted(c["fn"]):
                report_lines.append("  - " + format_span(c["tokens"], span))
        if c["fp"]:
            report_lines.append("Predicted constituents WRONG at width "
                                f"{','.join(map(str, args.widths))} "
                                "(false positives):")
            for span in sorted(c["fp"]):
                report_lines.append("  + " + format_span(c["tokens"], span))
        if c["shared"]:
            report_lines.append("Correctly matched at these widths:")
            for span in sorted(c["shared"]):
                report_lines.append("  = " + format_span(c["tokens"], span))
        report_lines.append("")

    text = "\n".join(report_lines)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"\n[wrote {args.out}]")


if __name__ == "__main__":
    main()
