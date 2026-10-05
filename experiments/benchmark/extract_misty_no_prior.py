#!/usr/bin/env python3
from pathlib import Path
import os,pandas as pd
R=Path(os.environ['CELLATTENTION_BENCHMARK_WORKDIR']); TAG=os.environ['HIER_VARIANT']; O=R/'results'/TAG/'misty_no_gene_role_prior'; D=R/'data'/TAG
genes=(D/'gene_ids.txt').read_text().splitlines(); sources={g for g in genes if g.startswith('source_ligand_')}; targets={g for g in genes if g.startswith('target_')}
d=pd.read_csv(O/'raw_scores.csv').rename(columns={'target_group':'receiver_group'}); d=d[d.source_gene.isin(sources)&d.target_gene.isin(targets)].groupby(['receiver_group','source_gene','target_gene'],as_index=False).score.max()
d.to_csv(O/'triplet_scores.csv',index=False)
d.groupby(['receiver_group','target_gene'],as_index=False).score.max().to_csv(O/'scores.csv',index=False)
