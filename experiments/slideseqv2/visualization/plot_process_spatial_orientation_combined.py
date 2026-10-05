#!/usr/bin/env python3
"""Compare model and database spatial gene patterns as source-to-sink streamlines."""

OUTPUT_FORMAT = "png"
PNG_DPI = 600
FIGURE_WIDTH_CM = 36.0
FIGURE_HEIGHT_CM = 25.2
GRID_SIZE = 58
MAX_CHARGES_PER_ROLE = 350
MIN_SUPPORTED_EDGES = 10
MAXIMUM_INTERACTION_DISTANCE = 0.1
STREAMLINE_DENSITY = 0.46
ARROW_GRID_STEP = 5
ARROW_LENGTH_FRACTION = 0.056
MINIMUM_CROP_HALF_WIDTH = 0.085

from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import mmread

from slideseq_common import DATASET, HERE, configure_style


def lines(path):
    return [x.strip() for x in path.read_text().splitlines() if x.strip()]


def crop_from_edges(points, minimum_half_width=MINIMUM_CROP_HALF_WIDTH):
    distances = ((points[:, None] - points[None, :]) ** 2).sum(axis=2)
    k = min(14, len(points))
    seed = np.argmin(np.partition(distances, k - 1, axis=1)[:, k - 1])
    focus = points[np.argsort(distances[seed])[:k]]
    center = np.median(focus, axis=0)
    required_half = np.maximum(np.ptp(focus, axis=0) * 1.05, minimum_half_width)
    half = np.repeat(required_half.max(), 2)
    return center - half, center + half


def normalize_weights(values):
    positive = values[values > 0]
    if not len(positive): return values
    scale = max(float(np.quantile(positive, .98)), 1e-8)
    return np.clip(values / scale, 0, 1)


def charge_subset(xy, values, visible):
    idx = np.flatnonzero(visible & (values > 0))
    if len(idx) > MAX_CHARGES_PER_ROLE:
        idx = idx[np.argpartition(values[idx], -MAX_CHARGES_PER_ROLE)[-MAX_CHARGES_PER_ROLE:]]
    return xy[idx], normalize_weights(values[idx])


def vector_field(grid_x, grid_y, positive, negative, softening):
    u = np.zeros_like(grid_x); v = np.zeros_like(grid_y); density = np.zeros_like(grid_x)
    for sign, (positions, weights) in ((1.0, positive), (-1.0, negative)):
        if not len(positions): continue
        for start in range(0, len(positions), 64):
            pos = positions[start:start + 64]; weight = weights[start:start + 64]
            dx = grid_x[..., None] - pos[:, 0]
            dy = grid_y[..., None] - pos[:, 1]
            distance2 = dx * dx + dy * dy
            r2 = distance2 + softening * softening
            within_range = distance2 <= MAXIMUM_INTERACTION_DISTANCE ** 2
            coefficient = sign * weight * within_range / np.power(r2, 1.5)
            u += (coefficient * dx).sum(axis=2)
            v += (coefficient * dy).sum(axis=2)
            density += sign * (weight * within_range * np.exp(-distance2 / (2 * (2.2 * softening) ** 2))).sum(axis=2)
    magnitude = np.hypot(u, v)
    scale = np.quantile(magnitude[magnitude > 0], .98) if np.any(magnitude > 0) else 1
    u /= scale; v /= scale
    return u, v, density


def directional_similarity(a, b):
    ua, va = a; ub, vb = b
    ma = np.hypot(ua, va); mb = np.hypot(ub, vb)
    keep = (ma > np.quantile(ma, .35)) & (mb > np.quantile(mb, .35))
    cosine = (ua * ub + va * vb) / np.maximum(ma * mb, 1e-12)
    weight = np.sqrt(ma * mb)
    denominator = np.sum(weight[keep])
    return float(np.sum(cosine[keep] * weight[keep]) / denominator) if denominator > 0 else np.nan


def main():
    configure_style()
    ids = lines(DATASET / "data/raw/full_cell_ids.txt")
    xy = np.loadtxt(DATASET / "data/preprocessed/spatial_coordinates.csv", delimiter=",")
    cell_index = {x: i for i, x in enumerate(ids)}
    genes = lines(DATASET / "data/raw/full_gene_ids.txt")
    gene_index = {x: i for i, x in enumerate(genes)}
    expression = mmread(DATASET / "data/raw/full_expression.mtx").tocsr()
    if expression.shape[0] != len(ids): expression = expression.T.tocsr()

    @lru_cache(maxsize=None)
    def gene_values(specification):
        parts = tuple(x.strip() for x in specification.split("+") if x.strip())
        values = np.column_stack([expression[:, gene_index[g]].toarray().ravel() for g in parts])
        if values.shape[1] == 1: return values[:, 0]
        result = np.exp(np.log1p(values).mean(axis=1)) - 1
        result[(values <= 0).any(axis=1)] = 0
        return result

    database_specs = {
        "MetaChat": DATASET / "analysis/metachat_support_expanded/results",
        "CellChat": DATASET / "analysis/cellchat_support_expanded/results",
        "NeuronChat": DATASET / "analysis/neuronchat_support_expanded/results",
    }
    candidates = []
    field_cache = {}
    for database, result_dir in database_specs.items():
        stats = pd.read_csv(result_dir / "lr_distance_matched_enrichment_long.csv")
        stats = stats[(stats.model_supported_edges >= MIN_SUPPORTED_EDGES) & (stats.log2_odds_ratio > 0)]
        significant_positive = stats[stats.fdr_q_within_triplet < .05]
        stats = pd.concat([
            stats.sort_values(["model_supported_edges", "log2_odds_ratio"], ascending=False).head(120),
            significant_positive,
        ]).drop_duplicates(["cluster", "source_gene", "target_gene", "lr_pair"])
        support = pd.read_csv(result_dir / "model_edge_lr_support.csv")
        for row in stats.itertuples(index=False):
            key = (row.cluster, row.source_gene, row.target_gene, row.lr_pair)
            edges = support[(support.cluster == row.cluster) &
                            (support.source_gene == row.source_gene) &
                            (support.target_gene == row.target_gene) &
                            (support.lr_pair == row.lr_pair)].drop_duplicates(["sender_cell_id", "receiver_cell_id"])
            edges = edges[edges.sender_cell_id.isin(cell_index) & edges.receiver_cell_id.isin(cell_index)]
            if len(edges) < MIN_SUPPORTED_EDGES: continue
            si = edges.sender_cell_id.map(cell_index).to_numpy(int)
            ri = edges.receiver_cell_id.map(cell_index).to_numpy(int)
            lo, hi = crop_from_edges(np.vstack([xy[si], xy[ri]]))
            visible = ((xy[:, 0] >= lo[0]) & (xy[:, 0] <= hi[0]) &
                       (xy[:, 1] >= lo[1]) & (xy[:, 1] <= hi[1]))
            gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], GRID_SIZE), np.linspace(lo[1], hi[1], GRID_SIZE))
            softening = max(hi[0] - lo[0], hi[1] - lo[1]) / 42
            ligand, receptor = [x.strip() for x in row.lr_pair.split("->")]
            model_field = vector_field(gx, gy,
                charge_subset(xy, gene_values(row.source_gene), visible),
                charge_subset(xy, gene_values(row.target_gene), visible), softening)
            database_field = vector_field(gx, gy,
                charge_subset(xy, gene_values(ligand), visible),
                charge_subset(xy, gene_values(receptor), visible), softening)
            similarity = directional_similarity(model_field[:2], database_field[:2])
            candidates.append({"database": database, "cluster": row.cluster,
                "source_gene": row.source_gene, "target_gene": row.target_gene,
                "database_relation": row.lr_pair, "supported_edges": len(edges),
                "log2_odds_ratio": row.log2_odds_ratio, "field_direction_similarity": similarity})
            field_cache[(database,) + key] = (lo, hi, gx, gy, model_field, database_field)

    candidate_table = pd.DataFrame(candidates).sort_values(
        ["database", "field_direction_similarity", "supported_edges"], ascending=[True, False, False])
    fixed_cases = [
        ("CellChat", 1, "Mylk", "Hs3st4", "Mdk -> Ncl"),
        ("CellChat", 4, "Itga11", "Cplx1", "Lama2 -> Sv2a"),
        ("NeuronChat", 4, "Snhg11", "Plp1", "Nrxn2 -> Nlgn1"),
        ("NeuronChat", 1, "Pvalb", "Tshz2", "Gad1+Slc32a1 -> Gabra2"),
    ]
    selected = []
    for database, cluster, source, target, relation in fixed_cases:
        match = candidate_table[(candidate_table.database == database) &
            (candidate_table.cluster == cluster) & (candidate_table.source_gene == source) &
            (candidate_table.target_gene == target) & (candidate_table.database_relation == relation)]
        if match.empty: raise RuntimeError(f"Missing fixed streamline case: {database} C{cluster} {source}->{target} {relation}")
        selected.append(next(match.head(1).itertuples(index=False)))
    metachat_fixed = [
        (1, "Drc7", "Igfbp4", "Nadsyn1 -> Gria2"),
        (1, "Thbs4", "Igfbp4", "Aldh7a1 -> Vdac1"),
    ]
    for cluster, source, target, relation in metachat_fixed:
        match = candidate_table[(candidate_table.database == "MetaChat") &
            (candidate_table.cluster == cluster) & (candidate_table.source_gene == source) &
            (candidate_table.target_gene == target) & (candidate_table.database_relation == relation)]
        if match.empty: raise RuntimeError(f"Missing MetaChat case: C{cluster} {source}->{target} {relation}")
        selected.append(next(match.head(1).itertuples(index=False)))

    out_data = HERE / "data/database_streamline_concordance/combined"
    out_fig = HERE / "figures/database_streamline_concordance"
    out_data.mkdir(parents=True, exist_ok=True); out_fig.mkdir(parents=True, exist_ok=True)
    candidate_table.to_csv(out_data / "candidate_field_similarity.csv", index=False)
    pd.DataFrame(selected).to_csv(out_data / "selected_streamline_cases.csv", index=False)

    process_labels = {
        "CellChat": ("Protein-mediated signaling", "protein_signaling"),
        "NeuronChat": ("Neural signaling", "neural_signaling"),
        "MetaChat": ("Metabolite-associated signaling", "metabolite_signaling"),
    }
    metabolite_labels = {
        "Nadsyn1 -> Gria2": "L-glutamic acid (Nadsyn1 → Gria2)",
        "Aldh7a1 -> Vdac1": "NADH (Aldh7a1 → Vdac1)",
    }

    def render_database(database, cases):
        process_name, output_stem = process_labels[database]
        fig, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH_CM / 2.54, FIGURE_HEIGHT_CM / 2.54))
        fig.subplots_adjust(left=.018, right=.995, bottom=.072, top=.988, wspace=.008, hspace=.035)
        for row_number, row in enumerate(cases):
            ax_model, ax_db = axes[row_number]
            key = (row.database, row.cluster, row.source_gene, row.target_gene, row.database_relation)
            lo, hi, gx, gy, model_field, database_field = field_cache[key]
            process_relation = metabolite_labels.get(
                row.database_relation, row.database_relation.replace(" -> ", " → ")
            )
            panels = [
                (ax_model, model_field, f"Model relation: {row.source_gene} → {row.target_gene}"),
                (ax_db, database_field, f"{process_name}\n{process_relation}"),
            ]
            ax_model.set_anchor("E")
            ax_db.set_anchor("W")
            for ax, field, title in panels:
                u, v, density = field
                limit = max(float(np.quantile(np.abs(density), .98)), 1e-8)
                ax.imshow(density, extent=[lo[0], hi[0], lo[1], hi[1]], origin="lower",
                          cmap="RdBu_r", vmin=-limit, vmax=limit, alpha=.50,
                          interpolation="bilinear")
                visible_cells = ((xy[:, 0] >= lo[0]) & (xy[:, 0] <= hi[0]) &
                                 (xy[:, 1] >= lo[1]) & (xy[:, 1] <= hi[1]))
                ax.scatter(xy[visible_cells, 0], xy[visible_cells, 1], s=5.0,
                           color="#858585", alpha=.42, linewidths=0,
                           rasterized=True, zorder=2)
                magnitude = np.hypot(u, v)
                spatial_span = max(hi[0] - lo[0], hi[1] - lo[1])
                sampled = (slice(None, None, ARROW_GRID_STEP), slice(None, None, ARROW_GRID_STEP))
                smag = magnitude[sampled]
                threshold = np.quantile(magnitude[magnitude > 0], .52) if np.any(magnitude > 0) else np.inf
                direction_u = np.divide(u[sampled], smag, out=np.zeros_like(smag), where=smag > 0)
                direction_v = np.divide(v[sampled], smag, out=np.zeros_like(smag), where=smag > 0)
                strength = np.clip(smag / max(np.quantile(magnitude, .92), 1e-8), .45, 1.0)
                arrow_length = spatial_span * ARROW_LENGTH_FRACTION * strength
                direction_u = np.where(smag >= threshold, direction_u * arrow_length, np.nan)
                direction_v = np.where(smag >= threshold, direction_v * arrow_length, np.nan)
                ax.quiver(gx[sampled], gy[sampled], direction_u, direction_v, angles="xy",
                          scale_units="xy", scale=1, color="#202020", width=.0052,
                          headwidth=4.5, headlength=5.7, headaxislength=5.0,
                          pivot="middle", alpha=.86, zorder=4)
                ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
                ax.set_xticks([]); ax.set_yticks([])
                ax.text(.018, .982, title, transform=ax.transAxes, ha="left", va="top",
                        fontsize=14.5, fontweight="bold", linespacing=1.12, zorder=8,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=.84, pad=2.2))
                for spine in ax.spines.values():
                    spine.set_linewidth(.7); spine.set_color("#555555")
            ax_db.text(.985, .025,
                       f"Directional cosine similarity = {row.field_direction_similarity:.2f}",
                       transform=ax_db.transAxes, ha="right", va="bottom", fontsize=12.0,
                       bbox=dict(facecolor="white", edgecolor="none", alpha=.82, pad=1.7))
        fig.text(.5, .015,
                 "Red: source-side expression density   ·   Blue: target-side expression density   ·   Arrows: local source-to-target orientation",
                 ha="center", va="bottom", fontsize=12.0)
        output = out_fig / f"{output_stem}_model_short_arrow_concordance.png"
        fig.savefig(output, dpi=PNG_DPI, facecolor="white", pil_kwargs={"compress_level": 6})
        fig.savefig(output.with_suffix(".pdf"), dpi=PNG_DPI, facecolor="white")
        plt.close(fig); print(output)

    for database in ("CellChat", "NeuronChat", "MetaChat"):
        render_database(database, [row for row in selected if row.database == database])

if __name__ == "__main__": main()
