#!/usr/bin/env python3
"""Run information-matched adapters and emit both benchmark output granularities."""
from pathlib import Path
import argparse,json,subprocess,os
import numpy as np
import pandas as pd
ROOT=Path(os.environ["CELLATTENTION_BENCHMARK_WORKDIR"]); CODE=Path(__file__).resolve().parent; TAG=os.environ.get('HIER_VARIANT','shared_prior_seed_168'); DATA=ROOT/'data'/TAG; OUT=ROOT/'results'/TAG
INFERRED=os.environ.get('HIER_GROUPS','inferred')=='inferred'; GROUP_PATH=DATA/('inferred_groups.txt' if INFERRED else 'cell_groups.txt')
def load():
 genes=(DATA/'gene_ids.txt').read_text().splitlines()
 return genes,{x:i for i,x in enumerate(genes)},np.loadtxt(DATA/'expression.csv',delimiter=','),np.loadtxt(DATA/'spatial_coordinates.csv',delimiter=','),np.loadtxt(GROUP_PATH,dtype=int)
def write_receiver(method,rows,meta):
 d=OUT/method; d.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows,columns=['receiver_group','target_gene','score']).to_csv(d/'scores.csv',index=False)
 meta.update(group_source='shared unsupervised KMeans labels' if INFERRED else 'oracle cell groups'); (d/'method_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
def cellattention():
 import sys; sys.path.insert(0,str(CODE.parent/'synthetic/analysis')); import benchmark_core as core
 genes,gi,x,xy,groups=load(); ligands=[z for z in genes if z.startswith('source_ligand_')]; targets=[z for z in genes if z.startswith('target_')]; receivers=sorted(set(groups)) if INFERRED else [0,1,2]
 fields=core.standardize(core.source_fields(xy,x)[0]); acc={(rg,l,t):[] for rg in receivers for l in ligands for t in targets}; models=sorted((ROOT/'models'/TAG).glob('seed_*'))
 if not models: raise FileNotFoundError(f'no CellAttention models for {TAG}')
 for model in models:
  observed=np.loadtxt(model/'standardized_expression.csv',delimiter=','); reconstruction=np.loadtxt(model/'reconstruction.csv',delimiter=','); embedding=np.loadtxt(model/'cell_embeddings.csv',delimiter=',')
  receiver_groups=core.cluster(core.whiten(embedding),int(model.name.split('_')[-1])); _,scores=core.scores(receiver_groups,observed-reconstruction,fields,'transformer')
  for rg in receivers:
   selected=receiver_groups[groups==rg]
   for l in ligands:
    for t in targets: acc[(rg,l,t)].append(float(np.mean(scores[selected,gi[l],gi[t]])))
 triplet=pd.DataFrame([(g,l,t,float(np.mean(v))) for (g,l,t),v in acc.items()],columns=['receiver_group','source_gene','target_gene','score']); d=OUT/'cellattention'; d.mkdir(parents=True,exist_ok=True); triplet.to_csv(d/'triplet_scores.csv',index=False)
 receiver=triplet.groupby(['receiver_group','target_gene'],as_index=False).score.max(); receiver.to_csv(d/'scores.csv',index=False)
 (d/'method_metadata.json').write_text(json.dumps({'method':'CellAttention','training_seeds':[int(x.name.split('_')[-1]) for x in models],'groups_used_for_fit':False,'receiver_aggregation':'shared inferred clusters used only when exporting scores','source_aggregation_for_receiver_target':'maximum after averaging training seeds','truth_used':False},indent=2)+'\n')
def external(method):
 env=Path(os.environ['CELLATTENTION_BENCHMARK_ENVS'])/method; subprocess.run([str(env/'bin/python'),str(CODE/f'run_{method}.py')],check=True,env=os.environ.copy())
def main():
 p=argparse.ArgumentParser(); p.add_argument('methods',nargs='*',default=['cellattention','holonet','commot']); args=p.parse_args()
 for method in args.methods: print('running',method,flush=True); {'cellattention':cellattention,'holonet':lambda:external('holonet'),'commot':lambda:external('commot')}[method]()
if __name__=='__main__': main()
