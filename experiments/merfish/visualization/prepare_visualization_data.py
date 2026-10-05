#!/usr/bin/env python3
"""Prepare canonical visualization inputs from formal 3D analysis outputs."""
from pathlib import Path
import hashlib
import json
import shutil
import pandas as pd
from merfish_common import HERE, RESULTS

stats_source = RESULTS / "directional_edge_statistics.csv"
edges_source = RESULTS / "directional_edges.csv"
out = HERE / "data/directional_arrow_heatmaps"
out.mkdir(parents=True, exist_ok=True)
shutil.copy2(stats_source, out / "directional_arrow_statistics.csv")
shutil.copy2(edges_source, HERE / "data/directional_edges.csv")
stats = pd.read_csv(stats_source)
selected = (stats[stats.modeled & stats.directional_enrichment.gt(0)]
            .sort_values(["cluster", "directional_enrichment"], ascending=[True, False])
            .groupby("cluster").head(3))
selected.to_csv(HERE / "data/selected_pairs.csv", index=False)
if len(selected) != 24 or not selected.groupby("cluster").size().eq(3).all():
    raise RuntimeError("Expected three selected directional pairs for each of eight receiver groups")
manifest = {
    "edge_definition": "abs(p*beta*K3D*x/field_sd)*min(abs(residual_z),3)/3 with source-positive receiver exclusion and signed direction support",
    "spatial_dimensions": 3,
    "top_fraction": 0.5,
    "maximum_edges_per_triplet": 1000,
    "selected_pairs_per_cluster": 3,
    "candidate_grid_entries": len(stats),
    "selected_pairs": len(selected),
    "input_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in (stats_source, edges_source)},
}
(HERE / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(f"Prepared {len(stats)} standard 3D directions and {len(selected)} selected pairs")
