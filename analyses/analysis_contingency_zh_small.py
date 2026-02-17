import argparse
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CTB9_MAP = {
    "verb": {"VV", "VA", "VC", "VE"},
    "adjective": {"JJ"},
    "noun": {"NN"},
    "proper noun": {"NR"},
    "function word": {"P", "LC", "AS", "DEC", "DEG", "DER", "DEV"},
    "determiner": {"DT"},
    "conjunction": {"CC", "CS"},
    "modal": {"MSP"},
    "pronoun": {"PN"},
    "adverb": {"AD"},
}

MODEL_LABELS = {
    "joint": "Joint-learning",
    "sem-first": "Semantics-first",
    "syn-first": "Syntax-first",
    "visual-labels": "Visual-labels",
}


def map_syn_cat(tag):
    for syn_cat, tags in CTB9_MAP.items():
        if tag in tags:
            return syn_cat
    return None


def entropy(counts):
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs)).sum())


def v_measure(df, beta=0.3):
    # df has columns: syn_cat, pred_cat
    syn = df["syn_cat"]
    pred = df["pred_cat"]
    contingency = pd.crosstab(syn, pred)
    h_syn = entropy(contingency.sum(axis=1).values)
    h_pred = entropy(contingency.sum(axis=0).values)

    # H(C|K)
    h_syn_given_pred = 0.0
    for pred_cat in contingency.columns:
        col = contingency[pred_cat].values
        h_syn_given_pred += entropy(col) * (col.sum() / contingency.values.sum())

    # H(K|C)
    h_pred_given_syn = 0.0
    for syn_cat in contingency.index:
        row = contingency.loc[syn_cat].values
        h_pred_given_syn += entropy(row) * (row.sum() / contingency.values.sum())

    homogeneity = 1.0 if h_syn == 0 else 1.0 - (h_syn_given_pred / h_syn)
    completeness = 1.0 if h_pred == 0 else 1.0 - (h_pred_given_syn / h_pred)

    if homogeneity + completeness == 0:
        v = 0.0
    else:
        v = (1 + beta) * homogeneity * completeness / (beta * homogeneity + completeness)
    return v, homogeneity, completeness


def load_df(path):
    df = pd.read_csv(path)
    df["syn_cat"] = df["gold_cat"].map(map_syn_cat)
    mapped = df[df["syn_cat"].notna()].copy()
    if mapped.empty:
        tag_counts = df["gold_cat"].value_counts().head(20).to_dict()
        raise ValueError(
            f"No gold tags matched CTB9_MAP in {path}. "
            f"Top gold tags: {tag_counts}"
        )
    return mapped


def parse_model_seed(filename):
    name = Path(filename).name
    parts = name.split("_")
    model = "_".join(parts[:-3])
    seed = parts[-3]
    return model, int(seed)


def build_figure(df_dir, seed, out_fig):
    df_files = sorted(Path(df_dir).glob("*_df.csv"))
    rows = []
    for f in df_files:
        model, f_seed = parse_model_seed(f)
        if f_seed != seed:
            continue
        df = load_df(f)
        df["model"] = MODEL_LABELS.get(model, model)
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No df_cats found for seed {seed} in {df_dir}")
    df_all = pd.concat(rows, ignore_index=True)
    if df_all.empty:
        raise ValueError(f"No mapped rows for seed {seed} in {df_dir}")

    syn_cats = list(CTB9_MAP.keys())
    pred_cats = [f"C{i}" for i in range(60)]

    models = [MODEL_LABELS[m] for m in ["joint", "sem-first", "syn-first", "visual-labels"]]
    fig, axes = plt.subplots(1, 4, figsize=(14, 6), sharey=True)
    for ax, model in zip(axes, models):
        sub = df_all[df_all["model"] == model]
        ct = pd.crosstab(sub["syn_cat"], sub["pred_cat"])
        ct = ct.reindex(index=syn_cats, columns=pred_cats, fill_value=0)
        proportions = ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

        im = ax.imshow(
            proportions.T.values,
            aspect="auto",
            cmap="Purples",
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_title(model)
        ax.set_xticks(range(len(syn_cats)))
        ax.set_xticklabels(syn_cats, rotation=90)
        ax.set_yticks([0, 9, 19, 29, 39, 49, 59])
        ax.set_yticklabels([f"C{i}" for i in [0, 9, 19, 29, 39, 49, 59]])
        if ax is axes[0]:
            ax.set_ylabel("Predicted category")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="Proportion")
    fig.supxlabel("Syntactic category")
    fig.tight_layout()
    fig.savefig(out_fig, dpi=300)


def build_table(df_dir, out_table, beta):
    df_files = sorted(Path(df_dir).glob("*_df.csv"))
    rows = []
    for f in df_files:
        model, seed = parse_model_seed(f)
        df = load_df(f)
        v, h, c = v_measure(df, beta=beta)
        rows.append((MODEL_LABELS.get(model, model), seed, v, h, c))

    df_res = pd.DataFrame(rows, columns=["Model", "Seed", "V", "H", "C"])
    summary = (
        df_res.groupby("Model")[["V", "H", "C"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["Model", "V_mean", "V_sd", "H_mean", "H_sd", "C_mean", "C_sd"]
    summary.to_csv(out_table, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--df_dir", required=True)
    parser.add_argument("--out_fig", default="figure6_zh_small.png")
    parser.add_argument("--out_table", default="table2_zh_small.csv")
    parser.add_argument("--seed", type=int, default=1018)
    parser.add_argument("--beta", type=float, default=0.3)
    args = parser.parse_args()

    build_figure(args.df_dir, args.seed, args.out_fig)
    build_table(args.df_dir, args.out_table, args.beta)


if __name__ == "__main__":
    main()
