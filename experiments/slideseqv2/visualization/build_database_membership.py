"""Build model gene-pair support membership for the SlideSeqV2 UpSet plot."""
from __future__ import annotations

import pandas as pd
from slideseq_common import DATASET, HERE

DATABASES = {
    "protein_signaling": "cellchat_support_expanded",
    "neural_signaling": "neuronchat_support_expanded",
    "metabolite_signaling": "metachat_support_expanded",
}
MIN_SUPPORTED_EDGES = 10


def main():
    universes, supported_sets = [], {}
    for name, directory in DATABASES.items():
        path = DATASET / "analysis" / directory / "results/lr_distance_matched_enrichment_long.csv"
        table = pd.read_csv(path, usecols=["source_gene", "target_gene", "model_supported_edges", "log2_odds_ratio"])
        universes.append(set(map(tuple, table[["source_gene", "target_gene"]].drop_duplicates().to_numpy())))
        supported = table[(table.model_supported_edges >= MIN_SUPPORTED_EDGES) & (table.log2_odds_ratio > 0)]
        supported_sets[name] = set(map(tuple, supported[["source_gene", "target_gene"]].drop_duplicates().to_numpy()))
    membership = pd.DataFrame(sorted(set.union(*universes)), columns=["source_gene", "target_gene"])
    pairs = list(map(tuple, membership[["source_gene", "target_gene"]].to_numpy()))
    for name in DATABASES:
        membership[name] = [pair in supported_sets[name] for pair in pairs]
    output = HERE / "data/database_support_venn_membership.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    membership.to_csv(output, index=False)
    print(f"gene_pairs={len(membership)} output={output}")


if __name__ == "__main__":
    main()
