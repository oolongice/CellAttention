#!/usr/bin/env python3
from pathlib import Path
import os
import numpy as np,pandas as pd
HERE=Path(os.environ["CELLATTENTION_MERFISH_WORKDIR"])/"analysis";R=HERE/'results';PRE=HERE.parent/'data/preprocessed';V=HERE.parent/'visualization/data';V.mkdir(parents=True,exist_ok=True)
p=pd.read_csv(R/'source_target_scores.csv');p=p[p.selected.astype(str).str.lower().eq('true')].copy();p['cluster']=p.receiver_group+1;p['effect_direction']=np.where(p.signed_beta>=0,'positive','negative')
a=pd.read_csv(R/'receiver_group_assignments.csv');meta=pd.read_parquet(PRE/'cell_metadata.parquet');a['cluster']=a.receiver_group+1
for c in ['clusters','germlayer','type','tissue']: a[c]=meta[c].astype(str).to_numpy()
rows=[]
for c,g in a.groupby('cluster'):
 row={'cluster':c,'cell_count':len(g)}
 for field in ['clusters','germlayer','type','tissue']:
  vc=g[field].value_counts();row[f'dominant_{field}']=vc.index[0];row[f'dominant_{field}_fraction']=vc.iloc[0]/len(g)
 rows.append(row)
cm=pd.DataFrame(rows);p=p.merge(cm,on='cluster',how='left').sort_values(['cluster','target_rank','derived_attention'],ascending=[True,True,False]);p['display_order_within_cluster']=p.groupby('cluster').cumcount()+1
p.to_csv(HERE/'all_selected_cluster_source_target_relations.csv',index=False);p[p.display_order_within_cluster<=10].assign(relation_label=lambda x:x.source_gene+' → '+x.target_gene).to_csv(V/'top_cluster_source_target_relations.csv',index=False);cm.to_csv(HERE/'cluster_metadata.csv',index=False)
a.to_parquet(V/'cell_receiver_groups.parquet',index=False)
print(f'clusters={len(cm)} selected_relations={len(p)}')
