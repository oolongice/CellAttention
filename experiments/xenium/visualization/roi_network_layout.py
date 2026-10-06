"""Deterministic, label-aware stress layout for compact ROI networks."""
import hashlib
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse.csgraph import shortest_path


def _label(node, cell_labels, display_names):
    if node[0] == "cell":
        return cell_labels.get(node[1], display_names.get(node[1], node[1]))
    return node[1]


def stress_layout(
    nodes, edges, seed_text, *, figure_width_in, figure_height_in,
    axes_width_fraction, axes_height_fraction, node_font_size,
    gene_node_size, cell_node_size, cell_labels, display_names,
    restarts=8, edge_length_in=.36, collision_padding_in=.025,
    collision_weight=7.0, boundary_weight=8.0,
):
    """Optimize graph-distance stress while preventing label/node collisions."""
    n = len(nodes)
    index = {node: i for i, node in enumerate(nodes)}
    adjacency = np.full((n, n), np.inf)
    np.fill_diagonal(adjacency, 0.0)
    for left, right, _ in edges:
        i, j = index[left], index[right]
        adjacency[i, j] = adjacency[j, i] = 1.0
    graph_distance = shortest_path(adjacency, directed=False, unweighted=True)
    finite = graph_distance[np.isfinite(graph_distance)]
    fallback = float(finite.max() + 1) if finite.size else 1.0
    graph_distance[~np.isfinite(graph_distance)] = fallback

    axes_width = figure_width_in * axes_width_fraction
    axes_height = figure_height_in * axes_height_fraction
    labels = [_label(node, cell_labels, display_names) for node in nodes]
    sizes = np.asarray([cell_node_size if node[0] == "cell" else gene_node_size for node in nodes])
    circle_diameter = 2.0 * np.sqrt(sizes / np.pi) / 72.0
    text_width = np.asarray([max(len(label), 1) * node_font_size * .54 / 72.0 for label in labels])
    text_height = np.full(n, node_font_size * 1.15 / 72.0)
    half_width = np.maximum(circle_diameter, text_width) / 2 + collision_padding_in
    half_height = np.maximum(circle_diameter, text_height) / 2 + collision_padding_in
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def residual(flat):
        normalized = flat.reshape(n, 2)
        physical = normalized * np.asarray([axes_width, axes_height])
        values = []
        for i, j in pairs:
            delta = physical[i] - physical[j]
            distance = max(float(np.linalg.norm(delta)), 1e-7)
            desired = min(edge_length_in * graph_distance[i, j], .72 * max(axes_width, axes_height))
            weight = 1.0 / max(graph_distance[i, j], 1.0)
            values.append(weight * (distance - desired) / max(desired, .08))
            elliptical = np.sqrt(
                (delta[0] / (half_width[i] + half_width[j])) ** 2
                + (delta[1] / (half_height[i] + half_height[j])) ** 2
            )
            values.append(collision_weight * max(0.0, 1.0 - elliptical))
        for i in range(n):
            x, y = physical[i]
            values.extend([
                boundary_weight * max(0.0, half_width[i] - x),
                boundary_weight * max(0.0, x + half_width[i] - axes_width),
                boundary_weight * max(0.0, half_height[i] - y),
                boundary_weight * max(0.0, y + half_height[i] - axes_height),
            ])
        return np.asarray(values)

    base_seed = int(hashlib.sha256(seed_text.encode()).hexdigest()[:8], 16)
    best = None
    for restart in range(restarts):
        rng = np.random.default_rng(base_seed + restart)
        initial = rng.uniform(.12, .88, size=(n, 2)).ravel()
        fit = least_squares(residual, initial, bounds=(.001, .999),
                            max_nfev=1800, ftol=1e-9, xtol=1e-9, gtol=1e-9)
        cost = float(np.dot(residual(fit.x), residual(fit.x)))
        if best is None or cost < best[0]:
            best = (cost, fit.x.reshape(n, 2))
    return {node: tuple(best[1][i]) for i, node in enumerate(nodes)}
