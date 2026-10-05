#!/usr/bin/env python3
from pathlib import Path
import os
import numpy as np,pandas as pd
DATASET=Path(os.environ.get('CELLATTENTION_SLIDESEQ_WORKDIR',Path(__file__).resolve().parents[3]/'.local/slideseqv2')).expanduser().resolve();HERE=DATASET/'analysis';R=HERE/'results';PRE=DATASET/'data/preprocessed';V=DATASET/'visualization/data';V.mkdir(parents=True,exist_ok=True)
p=pd.read_csv(R/'source_target_scores.csv');p=p[p.selected.astype(str).str.lower().eq('true')].copy();p['cluster']=p.receiver_group+1;p['effect_direction']=np.where(p.signed_beta>=0,'positive','negative')
a=pd.read_csv(R/'receiver_group_assignments.csv');ann=np.array((PRE/'evaluation_cell_groups.txt').read_text().splitlines(),object);a['annotation']=ann;a['cluster']=a.receiver_group+1
meta=[]
for c,g in a.groupby('cluster'):
 comp=g.annotation.value_counts();meta.append({'cluster':c,'cell_count':len(g),'dominant_cell_type':comp.index[0],'dominant_cell_type_fraction':comp.iloc[0]/len(g)})
meta=pd.DataFrame(meta);p=p.merge(meta,on='cluster',how='left');p=p.sort_values(['cluster','target_rank','derived_attention'],ascending=[True,True,False]);p['display_order_within_cluster']=p.groupby('cluster').cumcount()+1
p.to_csv(HERE/'all_selected_cluster_source_target_relations.csv',index=False);p[p.display_order_within_cluster<=8].assign(relation_label=lambda x:x.source_gene+' → '+x.target_gene).to_csv(V/'top_cluster_source_target_relations.csv',index=False);meta.to_csv(HERE/'cluster_metadata.csv',index=False)
print(f'receiver_groups={p.cluster.nunique()} selected_relations={len(p)}')
