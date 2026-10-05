#!/usr/bin/env python3
import os
"""Compare model-ranked CRISPR target recovery with random target selection."""

SOURCE_GENE = "Icam1"
SOURCE_TO_DGE_SHAPE = {"Icam1": "Icam_shapes", "Cxcr4": "Cxcr4_shapes"}
TOP_K_VALUES = (10, 20, 50)
RANDOM_REPEATS = 10_000
RANDOM_SEED = 20260812
OUTPUT_FORMAT = os.environ.get("CELLATTENTION_PLOT_FORMAT", "png")
PNG_DPI = 600

from pathlib import Path
from font_helper import font_family
import sys
import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(os.environ["CELLATTENTION_CRISPR_WORKDIR"]) / "visualization"
PROJECT = HERE.parent.parent
MODEL_RESULTS = HERE.parent / "analysis/source_focused_immune/source_target_receiver_results.csv"
DGE_ALL = HERE.parent / "data/preprocessed/immune_cells_spatial_DGE_all_shapes.csv"
DGE_SIGNIFICANT = HERE.parent / "data/preprocessed/immune_cells_spatial_DGE_significant.csv"
FONT = HERE.parent / "font"


def configure_style():
    family = font_family(FONT)
    mpl.rcParams.update({
        "font.family": family, "font.sans-serif": [family], "font.size": 7,
        "axes.titlesize": 8, "axes.labelsize": 7, "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5, "legend.fontsize": 6.2,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def average_precision(labels):
    labels = np.asarray(labels, dtype=bool)
    positives = int(labels.sum())
    if positives == 0:
        return np.nan
    ranks = np.arange(1, len(labels) + 1)
    precision = np.cumsum(labels) / ranks
    return float(precision[labels].sum() / positives)


def prepare_ranked_targets():
    if SOURCE_GENE not in SOURCE_TO_DGE_SHAPE:
        raise ValueError(f"No DGE shape mapping is defined for {SOURCE_GENE!r}")
    shape = SOURCE_TO_DGE_SHAPE[SOURCE_GENE]
    model = pd.read_csv(MODEL_RESULTS)
    model = model[(model.source_gene.eq(SOURCE_GENE)) & model.receiver_cell_type.eq("Immune_all")].copy()
    if model.empty:
        raise ValueError(f"No Immune_all model results found for {SOURCE_GENE}")
    all_dge = pd.read_csv(DGE_ALL, usecols=["names", "target_shape"])
    significant = pd.read_csv(DGE_SIGNIFICANT, usecols=["names", "target_shape"])
    tested = set(all_dge.loc[all_dge.target_shape.eq(shape), "names"].astype(str))
    positive = set(significant.loc[significant.target_shape.eq(shape), "names"].astype(str))
    ranked = (model[model.target_gene.isin(tested)]
              .sort_values(["physical_rank_within_source_receiver", "target_gene"])
              .drop_duplicates("target_gene").reset_index(drop=True))
    if ranked.empty:
        raise ValueError(f"No model targets overlap the complete DGE results for {shape}")
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked["experimental_positive"] = ranked.target_gene.isin(positive)
    if not ranked.experimental_positive.any():
        raise ValueError(f"No significant experimental targets overlap the model universe for {shape}")
    return ranked, shape


def random_recovery(labels):
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(labels)
    curves = np.empty((RANDOM_REPEATS, n), dtype=np.int16)
    random_ap = np.empty(RANDOM_REPEATS, dtype=float)
    for repeat in range(RANDOM_REPEATS):
        shuffled = rng.permutation(labels)
        curves[repeat] = np.cumsum(shuffled)
        random_ap[repeat] = average_precision(shuffled)
    return curves, random_ap


def main():
    fmt = OUTPUT_FORMAT.lower()
    if fmt not in {"png", "pdf"}:
        raise ValueError('OUTPUT_FORMAT must be "png" or "pdf"')
    configure_style()
    ranked, shape = prepare_ranked_targets()
    labels = ranked.experimental_positive.to_numpy(dtype=bool)
    n, positives = len(labels), int(labels.sum())
    observed = np.cumsum(labels)
    random_curves, random_ap = random_recovery(labels)
    random_mean = random_curves.mean(axis=0)
    random_low, random_high = np.quantile(random_curves, [.025, .975], axis=0)
    model_ap = average_precision(labels)
    ap_p = (1 + np.count_nonzero(random_ap >= model_ap)) / (RANDOM_REPEATS + 1)

    ranked["cumulative_experimental_targets"] = observed
    ranked["cumulative_recall"] = observed / positives
    ranked["random_mean_hits"] = random_mean
    ranked["random_lower_95"] = random_low
    ranked["random_upper_95"] = random_high
    ranked["random_expected_hits_exact"] = ranked["rank"] * positives / n

    summary_rows = []
    for k in TOP_K_VALUES:
        if k > n:
            continue
        hits = int(observed[k - 1])
        expected = k * positives / n
        empirical_p = (1 + np.count_nonzero(random_curves[:, k - 1] >= hits)) / (RANDOM_REPEATS + 1)
        summary_rows.append({
            "source_gene": SOURCE_GENE, "dge_shape": shape,
            "candidate_targets": n, "experimental_positive_targets": positives,
            "top_k": k, "model_hits": hits, "model_recall": hits / positives,
            "model_precision": hits / k, "random_expected_hits": expected,
            "enrichment_over_random": hits / expected if expected else np.nan,
            "random_lower_95": random_low[k - 1], "random_upper_95": random_high[k - 1],
            "empirical_p": empirical_p, "model_average_precision": model_ap,
            "random_average_precision_mean": float(random_ap.mean()),
            "average_precision_empirical_p": ap_p,
            "random_repeats": RANDOM_REPEATS, "random_seed": RANDOM_SEED,
        })
    summary = pd.DataFrame(summary_rows)

    data_dir = HERE / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    curve_out = data_dir / f"{SOURCE_GENE}_target_recovery_curve.csv"
    summary_out = data_dir / f"{SOURCE_GENE}_target_recovery_summary.csv"
    ranked.to_csv(curve_out, index=False)
    summary.to_csv(summary_out, index=False)

    fig, ax = plt.subplots(figsize=(100 / 25.4, 86 / 25.4))
    fig.subplots_adjust(left=.16, right=.95, bottom=.17, top=.86)
    x = ranked["rank"].to_numpy()
    ax.fill_between(x, random_low, random_high, color="#D9D9D9", alpha=.72,
                    linewidth=0, label="Random 95% interval", zorder=1)
    ax.plot(x, random_mean, color="#858585", linewidth=1.05, linestyle="--",
            label="Random mean", zorder=2)
    ax.plot(x, observed, color="#2878B5", linewidth=1.65,
            label="Physical-model ranking", zorder=3)
    colors = ["#56B4E9", "#2878B5", "#164A73"]
    for color, row in zip(colors, summary.itertuples(index=False)):
        ax.scatter(row.top_k, row.model_hits, s=25, color=color, edgecolor="white",
                   linewidth=.65, zorder=4)
        ax.annotate(f"Top {row.top_k}: {row.model_hits}", (row.top_k, row.model_hits),
                    xytext=(5, 5), textcoords="offset points", fontsize=6.1,
                    color="#202020", ha="left", va="bottom")
    ax.set_xlabel("Top K targets ranked by the physical model")
    ax.set_ylabel("Cumulative CRISPR-supported targets")
    ax.set_xlim(1, n)
    ax.set_ylim(0, positives + max(1.5, positives * .04))
    ax.set_title(f"{SOURCE_GENE} target recovery", loc="left", fontweight="bold", pad=5)
    ax.text(.98, .06,
            f"{positives} experimental positives among {n} tested targets\n"
            f"Average precision = {model_ap:.2f}; permutation P = {ap_p:.2g}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.2)
    ax.grid(axis="y", color="#E8E8E8", linewidth=.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.5, width=.6, pad=2)
    ax.legend(loc="upper left", frameon=False, handlelength=2.2)

    figure_out = HERE / "figures" / f"{SOURCE_GENE}_target_recovery_random_baseline.{fmt}"
    figure_out.parent.mkdir(parents=True, exist_ok=True)
    options = {"facecolor": "white"}
    if fmt == "png":
        options.update({"dpi": PNG_DPI, "pil_kwargs": {"compress_level": 6}})
    fig.savefig(figure_out, **options)
    plt.close(fig)
    print(f"source_gene={SOURCE_GENE}")
    print(f"candidate_targets={n}")
    print(f"experimental_positive_targets={positives}")
    for row in summary.itertuples(index=False):
        print(f"top{row.top_k}_hits={row.model_hits}; expected={row.random_expected_hits:.3f}; enrichment={row.enrichment_over_random:.3f}; empirical_p={row.empirical_p:.6g}")
    print(f"average_precision={model_ap:.6f}; empirical_p={ap_p:.6g}")
    print(f"curve_data={curve_out}")
    print(f"summary_data={summary_out}")
    print(f"figure={figure_out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise
