import argparse
import json
import os
import pickle
import shutil
from collections import Counter
from pathlib import Path

from opencc import OpenCC
import hanlp
from nltk import Tree

import sys

# Allow importing vpcfg utils for vocab creation
CURRENT_DIR = Path(__file__).resolve().parent
VC_PCFG_DIR = CURRENT_DIR.parent
sys.path.append(str(VC_PCFG_DIR))
from vpcfg.utils import Vocabulary  # noqa: E402


TOK_MODEL = hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH
CON_MODEL = hanlp.pretrained.constituency.CTB9_CON_FULL_TAG_ERNIE_GRAM


def load_hanlp_models():
    tok = hanlp.load(TOK_MODEL)
    parser = hanlp.load(CON_MODEL)
    return tok, parser


def hanlp_to_tree(parser, tokens):
    """Return an NLTK Tree from HanLP constituency outputs."""
    out = parser([tokens])
    res = out[0] if isinstance(out, list) else out
    if isinstance(res, Tree):
        return res
    if isinstance(res, str):
        return Tree.fromstring(res)
    if isinstance(res, dict):
        for key in ("con", "tree", "brackets"):
            if key in res:
                val = res[key]
                if isinstance(val, list):
                    val = val[0]
                if isinstance(val, str):
                    return Tree.fromstring(val)
    text = str(res)
    if "(" in text and ")" in text:
        return Tree.fromstring(text)
    return None


def binarize_tree(tree):
    """Force a binary tree so we get exactly L-1 constituent spans."""
    tree = tree.copy(deep=True)
    tree.collapse_unary(collapsePOS=True, collapseRoot=True)
    tree.chomsky_normal_form(factor="right", horzMarkov=2)
    return tree


def tree_to_spans_and_labels(tree):
    """Return all multi-token spans as (start, end) with their labels."""
    leaf_positions = list(tree.treepositions("leaves"))
    pos2idx = {pos: i for i, pos in enumerate(leaf_positions)}
    spans = []
    for node_pos in tree.treepositions():
        node = tree[node_pos]
        if isinstance(node, str):
            continue
        rel_leaves = node.treepositions("leaves")
        if len(rel_leaves) < 2:
            continue
        idxs = [pos2idx[node_pos + rel] for rel in rel_leaves]
        span = (min(idxs), max(idxs))
        spans.append((span, node.label()))
    spans = sorted(spans, key=lambda x: (x[0][0], x[0][1]))
    span_list = [list(span) for span, _ in spans]
    label_list = [label for _, label in spans]
    return span_list, label_list


def align_tokens_to_text(tokens, text):
    """Greedy alignment of tokens to character offsets in text."""
    offsets = []
    i = 0
    n = len(text)
    for tok in tokens:
        while i < n and text[i].isspace():
            i += 1
        L = len(tok)
        j = i
        if text[i:i + L] != tok:
            found = text.find(tok, i)
            if found != -1:
                j = found
        start = max(0, min(j, n))
        end = max(start, min(j + L, n))
        offsets.append((start, end))
        i = end
    return offsets


def char_spans_to_token_spans(char_spans, token_offsets):
    """Map inclusive char spans to inclusive token spans."""
    token_spans = []
    for c_start, c_end in char_spans:
        matched = []
        for ti, (t_start, t_end) in enumerate(token_offsets):
            if t_end == t_start:
                continue
            # token offset is [t_start, t_end) and char span is inclusive
            if t_start <= c_end and (t_end - 1) >= c_start:
                matched.append(ti)
        if not matched:
            continue
        token_spans.append([min(matched), max(matched)])
    return token_spans


def build_vocab(word_counts, vocab_size):
    vocab = Vocabulary()
    for word, _ in sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:vocab_size]:
        vocab.add_word(word)
    return vocab


def copy_features(copy_from, output_dir):
    if not copy_from:
        return
    copy_from = Path(copy_from)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("all_as-resn-50.npy", "all_flat_features_gold.npy"):
        src = copy_from / fname
        dst = output_dir / fname
        if src.exists() and not dst.exists():
            shutil.copyfile(src, dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_caps", required=True, help="Path to all_caps_zh.jsonl")
    parser.add_argument("--input_ids", required=True, help="Path to all.id_zh")
    parser.add_argument("--output_dir", required=True, help="Output dir for preprocessed-data/abstractscenes_zh")
    parser.add_argument("--vocab_size", default=2000, type=int)
    parser.add_argument("--copy_features_from", default="", help="Optional dir to copy image features from")
    parser.add_argument("--use_existing_char_spans", action="store_true",
                        help="Use char spans from input jsonl instead of re-parsing")
    parser.add_argument("--limit", type=int, default=0,
                        help="If > 0, limit to first N sentences for a quick smoke test")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tok, parser_model = load_hanlp_models()
    t2s = OpenCC("t2s")
    s2t = OpenCC("s2t")

    ids = []
    with open(args.input_ids, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.limit and i >= args.limit:
                break
            ids.append(line.strip())

    caps_out_path = output_dir / "all_caps.json"
    caps_text_path = output_dir / "all_caps.text"
    ids_out_path = output_dir / "all.id"
    vocab_counts_path = output_dir / "complete_word_list_counts.json"
    vocab_pickle_path = output_dir / "vocab_dict.pkl"
    gold_caps_path = output_dir / "all_gold_caps.json"

    word_counts = Counter()
    error_count = 0

    with open(args.input_caps, "r", encoding="utf-8") as fin, \
        open(caps_out_path, "w", encoding="utf-8") as fout, \
        open(caps_text_path, "w", encoding="utf-8") as ftext, \
        open(gold_caps_path, "w", encoding="utf-8") as fgold:
        for i, line in enumerate(fin):
            if not line.strip():
                continue
            if args.limit and i >= args.limit:
                break
            sent_trad, char_spans = json.loads(line)
            sent_simp = t2s.convert(sent_trad)
            tokens_simp = tok(sent_simp)
            tokens_trad = [s2t.convert(tok_) for tok_ in tokens_simp]

            tree = hanlp_to_tree(parser_model, tokens_simp)
            if tree is None:
                error_count += 1
                continue
            tree = binarize_tree(tree)
            tree_spans, tree_labels = tree_to_spans_and_labels(tree)

            spans = []
            if args.use_existing_char_spans:
                token_offsets = align_tokens_to_text(tokens_trad, sent_trad)
                spans = char_spans_to_token_spans(char_spans, token_offsets)
                spans = sorted({tuple(span) for span in spans}, key=lambda x: (x[0], x[1]))
                spans = [list(span) for span in spans]

            if not spans or len(spans) != max(0, len(tokens_trad) - 1):
                spans = tree_spans
                if len(spans) != max(0, len(tokens_trad) - 1):
                    error_count += 1
                    continue

            caption = " ".join(tokens_trad)
            json.dump([caption, spans], fout, ensure_ascii=False)
            fout.write("\n")
            ftext.write(caption + "\n")
            word_counts.update(tokens_trad)

            pos_tags = [tag for _, tag in tree.pos()]
            json.dump([caption, tree_spans, tree_labels, pos_tags], fgold, ensure_ascii=False)
            fgold.write("\n")

    with open(ids_out_path, "w", encoding="utf-8") as f:
        for id_ in ids:
            f.write(f"{id_}\n")

    with open(vocab_counts_path, "w", encoding="utf-8") as f:
        json.dump(word_counts, f, ensure_ascii=False)

    vocab = build_vocab(word_counts, args.vocab_size)
    with open(vocab_pickle_path, "wb") as f:
        pickle.dump(vocab, f)

    copy_features(args.copy_features_from, output_dir)

    if error_count:
        print(f"Finished with {error_count} parsing/tokenization mismatches.")


if __name__ == "__main__":
    main()
