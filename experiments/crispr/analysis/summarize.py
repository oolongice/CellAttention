#!/usr/bin/env python3
from pathlib import Path
import os
import pandas as pd

root=Path(os.environ["CELLATTENTION_CRISPR_WORKDIR"])/"analysis"; result=root/"results"
targets=pd.read_csv(result/"target_scores.csv")
pairs=pd.read_csv(result/"source_target_scores.csv")
composition=pd.read_csv(result/"receiver_group_annotation_composition.csv")
top_annotations=(composition.sort_values(["receiver_group","fraction_within_context"],ascending=[True,False]).groupby("receiver_group").head(3))
top_annotations.to_csv(result/"receiver_group_summary.csv",index=False)
targets.sort_values(["receiver_group","target_rank"]).groupby("receiver_group").head(20).to_csv(result/"top_targets_by_receiver_group.csv",index=False)
selected=pairs[pairs["selected"].astype(str).str.lower().eq("true")]
selected.sort_values(["receiver_group","target_rank","source_rank"]).groupby("receiver_group").head(100).to_csv(result/"top_selected_pairs.csv",index=False)
