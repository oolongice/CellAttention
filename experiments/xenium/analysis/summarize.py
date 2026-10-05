#!/usr/bin/env python3
"""Derive compact Xenium tables from source-target analysis results."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True, type=Path)
    args = parser.parse_args()
    root = args.workdir.expanduser().resolve()
    result = root / "analysis/results"
    data = root / "data/preprocessed"
    targets = pd.read_csv(result / "target_scores.csv")
    pairs = pd.read_csv(result / "source_target_scores.csv")
    composition = pd.read_csv(result / "receiver_group_annotation_composition.csv")
    assignments = pd.read_csv(result / "receiver_group_assignments.csv")
    samples = [x.strip() for x in (data / "sample_ids.txt").read_text().splitlines() if x.strip()]
    annotations = [x.strip() for x in (data / "evaluation_cell_groups.txt").read_text().splitlines() if x.strip()]
    if not (len(assignments) == len(samples) == len(annotations)):
        raise ValueError("assignment, sample, and annotation row counts differ")
    assignments["sample_id"] = samples
    assignments["annotation"] = annotations
    sample = assignments.groupby(["receiver_group", "sample_id"], as_index=False).size().rename(columns={"size": "count"})
    sample["fraction_within_context"] = sample["count"] / sample.groupby("receiver_group")["count"].transform("sum")
    sample.to_csv(result / "receiver_group_sample_composition.csv", index=False)
    (composition.sort_values(["receiver_group", "fraction_within_context"], ascending=[True, False])
     .groupby("receiver_group").head(3).to_csv(result / "receiver_group_summary.csv", index=False))
    (targets.sort_values(["receiver_group", "target_rank"]).groupby("receiver_group").head(20)
     .to_csv(result / "top_targets_by_receiver_group.csv", index=False))
    selected = pairs[pairs["selected"].astype(str).str.lower().eq("true")]
    (selected.sort_values(["receiver_group", "target_rank", "source_rank"]).groupby("receiver_group").head(100)
     .to_csv(result / "top_selected_pairs.csv", index=False))
    print(f"receiver_groups={assignments.receiver_group.nunique()}")
    print(f"assignments={len(assignments)}")
    print(f"active_effects={len(selected)}")


if __name__ == "__main__":
    main()
