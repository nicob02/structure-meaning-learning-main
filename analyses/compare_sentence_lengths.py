import argparse
import json
import math
from pathlib import Path
from statistics import mean, median


PUNCT_TOKENS = {"。", "！", "？", "!", "?", ".", "｡", "．", "…", "，", ",", "、", "；", ";", "：", ":"}


def percentile(sorted_vals, pct):
    if not sorted_vals:
        return float("nan")
    if pct <= 0:
        return sorted_vals[0]
    if pct >= 100:
        return sorted_vals[-1]
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return sorted_vals[lo]
    w = k - lo
    return sorted_vals[lo] * (1 - w) + sorted_vals[hi] * w


def load_lengths(data_path):
    caps_path = Path(data_path) / "all_caps.json"
    if not caps_path.exists():
        raise FileNotFoundError(caps_path)
    lengths = []
    end_punct = 0
    punct_tokens = 0
    total_tokens = 0
    with caps_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            cap, _ = json.loads(line)
            toks = cap.split(" ") if cap else []
            n = len(toks)
            lengths.append(n)
            if toks and toks[-1] in PUNCT_TOKENS:
                end_punct += 1
            punct_tokens += sum(1 for t in toks if t in PUNCT_TOKENS)
            total_tokens += n
    return lengths, end_punct, punct_tokens, total_tokens


def summarize(name, lengths, end_punct, punct_tokens, total_tokens):
    s = sorted(lengths)
    mu = mean(s)
    med = median(s)
    std = (sum((x - mu) ** 2 for x in s) / len(s)) ** 0.5
    print(f"=== {name} ===")
    print(f"sentences={len(s)}")
    print(f"mean_len={mu:.3f}  median_len={med:.3f}  std_len={std:.3f}")
    print(
        f"p10={percentile(s,10):.3f}  p25={percentile(s,25):.3f}  "
        f"p75={percentile(s,75):.3f}  p90={percentile(s,90):.3f}  p95={percentile(s,95):.3f}"
    )
    print(
        f"end_punct_rate={end_punct/max(1,len(s)):.2%}  "
        f"token_punct_rate={punct_tokens/max(1,total_tokens):.2%}"
    )
    print(
        f"len<=8={sum(1 for x in s if x<=8)/len(s):.2%}  "
        f"len9-12={sum(1 for x in s if 9<=x<=12)/len(s):.2%}  "
        f"len>=13={sum(1 for x in s if x>=13)/len(s):.2%}"
    )
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--en_path", required=True, help="Path to English dataset dir with all_caps.json")
    parser.add_argument("--zh_path", required=True, help="Path to Chinese dataset dir with all_caps.json")
    parser.add_argument("--zh_label", default="zh", help="Display label for Chinese dataset")
    args = parser.parse_args()

    en_lengths, en_end_punct, en_punct_tokens, en_total_tokens = load_lengths(args.en_path)
    zh_lengths, zh_end_punct, zh_punct_tokens, zh_total_tokens = load_lengths(args.zh_path)

    summarize("English", en_lengths, en_end_punct, en_punct_tokens, en_total_tokens)
    summarize(args.zh_label, zh_lengths, zh_end_punct, zh_punct_tokens, zh_total_tokens)

    print("=== Difference (zh - en) ===")
    print(f"mean_len_delta={mean(zh_lengths)-mean(en_lengths):+.3f}")
    print(f"median_len_delta={median(zh_lengths)-median(en_lengths):+.3f}")
    print(
        f"end_punct_rate_delta="
        f"{(zh_end_punct/max(1,len(zh_lengths)))-(en_end_punct/max(1,len(en_lengths))):+.2%}"
    )


if __name__ == "__main__":
    main()
