#!/usr/bin/env python3
from pathlib import Path
import os
import numpy as np,pandas as pd
HERE=Path(os.environ["CELLATTENTION_MOSTA_WORKDIR"])/"analysis";R=HERE/'results';PRE=HERE.parent/'data/preprocessed';V=HERE.parent/'visualization/data';V.mkdir(parents=True,exist_ok=True)
p=pd.read_csv(R/'source_target_scores.csv');p=p[p.selected.astype(str).str.lower().eq('true')].copy();p['cluster']=p.receiver_group+1;p['effect_direction']=np.where(p.signed_beta>=0,'positive','negative')
a=pd.read_csv(R/'receiver_group_assignments.csv');samples=np.array((PRE/'sample_ids.txt').read_text().splitlines(),object);ann=np.array((PRE/'evaluation_cell_groups.txt').read_text().splitlines(),object);a['sample_id']=samples;a['annotation']=ann;a['cluster']=a.receiver_group+1
meta=[]
for c,g in a.groupby('cluster'):
 comp=g.annotation.value_counts();sample=g.sample_id.value_counts();meta.append({'cluster':c,'cell_count':len(g),'dominant_cell_type':comp.index[0],'dominant_cell_type_fraction':comp.iloc[0]/len(g),'dominant_sample':sample.index[0],'dominant_sample_fraction':sample.iloc[0]/len(g)})
meta=pd.DataFrame(meta);p=p.merge(meta,on='cluster',how='left');p=p.sort_values(['cluster','target_rank','derived_attention'],ascending=[True,True,False]);p['display_order_within_cluster']=p.groupby('cluster').cumcount()+1
p.to_csv(HERE/'all_selected_cluster_source_target_relations.csv',index=False);p[p.display_order_within_cluster<=8].assign(relation_label=lambda x:x.source_gene+' → '+x.target_gene).to_csv(V/'top_cluster_source_target_relations.csv',index=False);meta.to_csv(HERE/'cluster_metadata.csv',index=False)
print(f'clusters={p.cluster.nunique()} selected_relations={len(p)}')
