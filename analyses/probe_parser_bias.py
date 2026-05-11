"""Probe a (possibly untrained) parser for left/right-branching bias.

Why this exists
---------------
We see ALL trained seeds converge to nearly the same left-branching parse
(95-98% pairwise F1). That has two possible explanations:

  1. The bias is *born at init* — the CKY argmax decoder breaks ties toward
     one direction, and Adam dynamics quickly amplify whatever shape comes
     out first. If true: parsing the corpus with an UNTRAINED parser
     should already show strong leftness.

  2. The bias is *born during training* — at init the parser produces
     diverse / balanced shapes, but the loss landscape has a left-branching
     attractor everyone falls into. If true: untrained parsers should
     produce balanced shapes.

This script tells you which.

Outputs
-------
- Per-seed: leftness ratio (fraction of non-root spans starting at 0),
  rightness ratio (ending at L-1), branching index in [-1, +1] where -1 is
  fully left, +1 is fully right.
- A histogram of (start, end) span counts.
- If you run multiple seeds, an across-seed agreement score on argmax trees.

Usage
-----
  # Probe untrained parsers across 5 seeds:
  python analyses/probe_parser_bias.py \
      --data_path preprocessed-data/abstractscenes_zh \
      --prefix all \
      --seeds 91 214 527 627 1018 \
      --max_sents 1000 \
      --out_dir analyses/results/parser_init_probe

  # Probe an existing checkpoint (single seed):
  python analyses/probe_parser_bias.py \
      --data_path preprocessed-data/abstractscenes_zh \
      --prefix all \
      --checkpoint runs/zh_full_joint_s91/checkpoints/checkpoint.pth.tar \
      --max_sents 1000 \
      --out_dir analyses/results/parser_trained_probe_s91
"""
from __future__ import annotations
import argparse
import os
import sys
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

# So we can import vpcfg from the repo root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vc-pcfg"))

import vpcfg.as_dataloader as data  # noqa: E402
from vpcfg.model_vis import VGCPCFGs  # noqa: E402
from vpcfg import utils  # noqa: E402


def _build_default_opt(data_path, vocab_size, prefix, batch_size, encoder_file,
                       img_dim, max_length):
    """Build a minimal `opt` namespace matching as_train.py defaults."""
    opt = argparse.Namespace()
    # Loss / model knobs (defaults match as_train.py).
    opt.log_step = 100
    opt.grad_clip = 5.0
    opt.vse_mt_alpha = 1.0
    opt.vse_lm_alpha = 1.0
    opt.sem_first = False
    opt.use_structural_negatives = False
    opt.struct_neg_margin = 0.1
    opt.struct_neg_weight = 1.0
    opt.struct_neg_style = 'hinge'
    opt.use_mi_regularizer = False
    opt.mi_margin = 0.1
    opt.mi_weight = 1.0
    opt.mi_style = 'ratio'
    opt.use_entropy_bonus = False
    opt.entropy_weight = 0.1
    opt.entropy_mode = 'category'
    opt.parser_grad_noise = 0.0
    opt.lr_parser = None
    opt.lr_txt_enc = None
    opt.lr_img_enc = None
    opt.branching_weight = 0.0
    opt.right_prior_weight = 0.0
    opt.branching_init = 0.0
    opt.branching_init_mode = 'right'
    opt.lr = 5e-4
    opt.beta1 = 0.75
    opt.beta2 = 0.999
    opt.margin = 0.1
    # Architecture sizes (defaults from as_module.py / as_train.py).
    opt.vocab_size = vocab_size
    opt.nt_states = 30
    opt.t_states = 60
    opt.h_dim = 512
    opt.w_dim = 512
    opt.z_dim = 64
    opt.state_dim = 256
    opt.word_dim = 512
    opt.lstm_dim = 512
    opt.sem_dim = 512
    opt.syn_dim = 512
    opt.img_dim = img_dim
    opt.no_imgnorm = False
    return opt


def branching_summary(spans, length):
    """Return (n_left_edge, n_right_edge, n_total_nontrivial, branching_index).

    `spans` is a list of (i, j) inclusive span endpoints. We exclude width-1
    spans (preterminal singletons) and the root span (covering the whole
    sentence) before computing leftness / rightness.
    """
    n_left = 0
    n_right = 0
    n_total = 0
    for (i, j) in spans:
        w = j - i + 1
        if w < 2 or w >= length:  # skip preterminals and the root
            continue
        n_total += 1
        if i == 0:
            n_left += 1
        if j == length - 1:
            n_right += 1
    if n_total == 0:
        return n_left, n_right, n_total, 0.0
    # branching_index in [-1, +1]: -1 means every non-trivial span is at the
    # left edge (fully left-branching), +1 means every non-trivial span is at
    # the right edge (fully right-branching), 0 is balanced.
    bi = (n_right - n_left) / n_total
    return n_left, n_right, n_total, bi


def parse_one_loader(model, loader, max_sents):
    """Run model.forward_parser over a loader; return list of (length, spans)."""
    out = []
    seen = 0
    with torch.no_grad():
        for batch in loader:
            # collate_fun returns (images, captions, lengths, ids, gold_spans).
            images, captions, lengths, ids, gold_spans = batch
            if not isinstance(lengths, torch.Tensor):
                lengths = torch.tensor(lengths).long()
            if torch.cuda.is_available():
                captions = captions.cuda()
                lengths = lengths.cuda()
            parser_outs = model.forward_parser(captions, lengths)
            argmax_spans = parser_outs[-3]
            # argmax_spans[b] is a list of (i, j, A) tuples
            lengths_list = lengths.tolist() if isinstance(lengths, torch.Tensor) else list(lengths)
            for b, L in enumerate(lengths_list):
                pred = [(int(s[0]), int(s[1])) for s in argmax_spans[b]
                        if int(s[0]) != int(s[1])]
                out.append((int(L), pred))
                seen += 1
                if seen >= max_sents:
                    return out
    return out


def aggregate(parses, name=""):
    n_left = n_right = n_total = 0
    bi_per_sent = []
    span_counter = Counter()
    width_left_counter = Counter()  # width -> count of left-edge spans
    width_right_counter = Counter()  # width -> count of right-edge spans
    for (L, spans) in parses:
        nl, nr, nt, bi = branching_summary(spans, L)
        n_left += nl
        n_right += nr
        n_total += nt
        if nt > 0:
            bi_per_sent.append(bi)
        for (i, j) in spans:
            w = j - i + 1
            if w < 2 or w >= L:
                continue
            span_counter[(i, j)] += 1
            if i == 0:
                width_left_counter[w] += 1
            if j == L - 1:
                width_right_counter[w] += 1
    summary = {
        'name': name,
        'n_sents': len(parses),
        'n_total_spans': n_total,
        'leftness': n_left / max(1, n_total),
        'rightness': n_right / max(1, n_total),
        'branching_index_global': (n_right - n_left) / max(1, n_total),
        'branching_index_mean_per_sent': float(np.mean(bi_per_sent))
            if bi_per_sent else 0.0,
        'top_spans': span_counter.most_common(10),
        'left_by_width': dict(sorted(width_left_counter.items())),
        'right_by_width': dict(sorted(width_right_counter.items())),
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_path', required=True)
    ap.add_argument('--prefix', default='all')
    ap.add_argument('--encoder_file', default='all_as-resn-50.npy')
    ap.add_argument('--img_dim', type=int, default=2048)
    ap.add_argument('--max_length', type=int, default=1000)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--max_sents', type=int, default=1000)
    ap.add_argument('--seeds', nargs='+', type=int, default=[91])
    ap.add_argument('--checkpoint', default=None,
                    help='Optional path to a .pth.tar checkpoint to load.')
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load vocab from data_path. The pickle was created when `utils`
    # was importable at top-level with a `Vocabulary` class. The local
    # version lives in vpcfg.utils — register it under both names so
    # pickle.load can resolve it regardless of how it was saved.
    import pickle
    from vpcfg.utils import Vocabulary as _Vocabulary
    import types
    if 'utils' not in sys.modules or not hasattr(sys.modules['utils'], 'Vocabulary'):
        _u = sys.modules.get('utils')
        if _u is None or not isinstance(_u, types.ModuleType):
            _u = types.ModuleType('utils')
            sys.modules['utils'] = _u
        _u.Vocabulary = _Vocabulary
    import __main__
    if not hasattr(__main__, 'Vocabulary'):
        __main__.Vocabulary = _Vocabulary

    vocab_path = os.path.join(args.data_path, 'vocab_dict.pkl')
    if not os.path.exists(vocab_path):
        # Fallback: try parent (preprocessed) location
        vocab_path = os.path.join(args.data_path, '..', 'vocab_dict.pkl')
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    data.set_constant(visual_mode=True, max_length=args.max_length)

    all_summaries = []
    for seed in args.seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        opt = _build_default_opt(
            args.data_path, vocab_size=len(vocab),
            prefix=args.prefix, batch_size=args.batch_size,
            encoder_file=args.encoder_file, img_dim=args.img_dim,
            max_length=args.max_length,
        )

        train_loader, _, _ = data.get_data_iters(
            args.data_path, args.prefix, vocab, args.batch_size, 0,
            shuffle=False, sampler=None, tiny=False, one_shot=False,
            encoder_file=args.encoder_file, img_dim=args.img_dim,
            use_syntactic_bootstrapping=False, reverse_text=False,
        )

        # Build model. The logger here is a stub — we don't need to write logs.
        import logging
        log = logging.getLogger(f'probe_s{seed}')
        log.setLevel(logging.WARNING)
        model = VGCPCFGs(opt, vocab, log)
        if args.checkpoint:
            ckpt = torch.load(args.checkpoint, map_location='cpu')
            sd = ckpt.get('model', ckpt)
            try:
                model.set_state_dict(sd)
            except Exception as e:
                print(f"  [warn] full set_state_dict failed ({e}); "
                      f"loading parser only.")
                model.parser.load_state_dict(sd.get('parser', sd), strict=False)
        model.parser.eval()
        if hasattr(model, 'txt_enc'):
            model.txt_enc.eval()
        if torch.cuda.is_available():
            model.parser.cuda()
            if hasattr(model, 'txt_enc'):
                model.txt_enc.cuda()
            if hasattr(model, 'img_enc'):
                model.img_enc.cuda()

        parses = parse_one_loader(model, train_loader, args.max_sents)
        summary = aggregate(parses, name=f'seed{seed}')
        all_summaries.append((seed, summary, parses))

        print(f"\n=== seed {seed} ({'TRAINED' if args.checkpoint else 'UNTRAINED'}) ===")
        print(f"  n_sents={summary['n_sents']}  n_spans={summary['n_total_spans']}")
        print(f"  leftness  (start==0):           {summary['leftness']:.4f}")
        print(f"  rightness (end==L-1):           {summary['rightness']:.4f}")
        print(f"  branching index (global):       {summary['branching_index_global']:+.4f}   "
              f"(-1=fully left, +1=fully right, 0=balanced)")
        print(f"  branching index (per-sent mean): {summary['branching_index_mean_per_sent']:+.4f}")
        print(f"  top 10 (start, end) spans:")
        for span, c in summary['top_spans']:
            print(f"     {span}  count={c}")

    # Cross-seed agreement of argmax trees.
    if len(all_summaries) >= 2:
        print(f"\n=== Cross-seed pairwise span overlap (untrained or trained) ===")
        sets_per_seed = []
        for (seed, _, parses) in all_summaries:
            sets_per_sent = [set(spans) for (_, spans) in parses]
            sets_per_seed.append(sets_per_sent)
        n_sents = min(len(s) for s in sets_per_seed)
        n_seeds = len(sets_per_seed)
        for i in range(n_seeds):
            row = []
            for j in range(n_seeds):
                if i == j:
                    row.append("1.000")
                    continue
                num = den = 0
                for s in range(n_sents):
                    pi = sets_per_seed[i][s]
                    pj = sets_per_seed[j][s]
                    if not pi and not pj:
                        continue
                    inter = len(pi & pj)
                    if inter == 0:
                        continue
                    num += 2 * inter
                    den += len(pi) + len(pj)
                f1 = num / max(1, den)
                row.append(f"{f1:.3f}")
            print(f"  seed{all_summaries[i][0]:>5}: " + " ".join(row))

    # Dump JSON for downstream plotting.
    out_path = os.path.join(args.out_dir, 'summary.json')
    dump = []
    for (seed, summary, _) in all_summaries:
        d = dict(summary)
        d['top_spans'] = [[list(s), int(c)] for s, c in d['top_spans']]
        d['seed'] = seed
        dump.append(d)
    with open(out_path, 'w') as f:
        json.dump(dump, f, indent=2)
    print(f"\nWritten to {out_path}")


if __name__ == '__main__':
    main()
