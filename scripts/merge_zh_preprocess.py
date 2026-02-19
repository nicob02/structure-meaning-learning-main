import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import sys


def load_vocab_class():
    repo_root = Path(__file__).resolve().parent.parent
    vpcfg_dir = repo_root / "vc-pcfg" / "vpcfg"
    sys.path.insert(0, str(vpcfg_dir))
    from utils import Vocabulary  # type: ignore
    return Vocabulary


def build_vocab(word_counts, vocab_size):
    Vocabulary = load_vocab_class()
    vocab = Vocabulary()
    for word, _ in sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:vocab_size]:
        vocab.add_word(word)
    return vocab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--vocab_size", type=int, default=2000)
    parser.add_argument("--copy_features_from", default="")
    args = parser.parse_args()

    shards_dir = Path(args.shards_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    caps_out = out_dir / "all_caps.json"
    caps_text_out = out_dir / "all_caps.text"
    gold_out = out_dir / "all_gold_caps.json"
    ids_out = out_dir / "all.id"
    counts_out = out_dir / "complete_word_list_counts.json"
    vocab_out = out_dir / "vocab_dict.pkl"

    shard_dirs = sorted(shards_dir.glob("shard_*"))
    if not shard_dirs:
        raise FileNotFoundError(f"No shard_* directories found in {shards_dir}")

    word_counts = Counter()

    with caps_out.open("w", encoding="utf-8") as f_caps, \
        caps_text_out.open("w", encoding="utf-8") as f_text, \
        gold_out.open("w", encoding="utf-8") as f_gold, \
        ids_out.open("w", encoding="utf-8") as f_ids:
        for shard in shard_dirs:
            shard_caps = shard / "all_caps.json"
            shard_text = shard / "all_caps.text"
            shard_gold = shard / "all_gold_caps.json"
            shard_ids = shard / "all.id"
            shard_counts = shard / "complete_word_list_counts.json"

            if not shard_caps.exists() or not shard_text.exists() or not shard_gold.exists() or not shard_ids.exists():
                raise FileNotFoundError(f"Missing expected files in {shard}")

            with shard_caps.open("r", encoding="utf-8") as f:
                for line in f:
                    f_caps.write(line)
            with shard_text.open("r", encoding="utf-8") as f:
                for line in f:
                    f_text.write(line)
            with shard_gold.open("r", encoding="utf-8") as f:
                for line in f:
                    f_gold.write(line)
            with shard_ids.open("r", encoding="utf-8") as f:
                for line in f:
                    f_ids.write(line)

            with shard_counts.open("r", encoding="utf-8") as f:
                counts = json.load(f)
            word_counts.update(counts)

    with counts_out.open("w", encoding="utf-8") as f:
        json.dump(word_counts, f, ensure_ascii=False)

    vocab = build_vocab(word_counts, args.vocab_size)
    with vocab_out.open("wb") as f:
        import pickle
        pickle.dump(vocab, f)

    if args.copy_features_from:
        src_dir = Path(args.copy_features_from)
        for fname in ("all_as-resn-50.npy", "all_flat_features_gold.npy"):
            src = src_dir / fname
            dst = out_dir / fname
            if src.exists() and not dst.exists():
                shutil.copyfile(src, dst)


if __name__ == "__main__":
    main()
