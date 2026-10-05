#!/usr/bin/env python3
"""Information-matched hierarchical generator with deliberately overlapping cell groups."""
from pathlib import Path
import argparse, json
import numpy as np
from scipy.special import k0
import pandas as pd

def generate(out,seed,n,marker_strength,type_preference,spatial_separation,effect_scale,dispersion,dropout_rate,target_background_scale,decoy_scale,ligand_spatial_scale,aligned_kernel,direct_field_signal,ligand_hotspot_scale,binary_receptor_gate,receptor_gate_quantile,bimodal_receptor,receptor_gate_threshold,communication_genes,simple_one_to_one,separated_ligand_domains,ligand_domain_scale,ligand_domain_radius,ligand_domain_sigma):
 rng=np.random.default_rng(seed); out.mkdir(parents=True,exist_ok=True)
 groups=np.arange(n)%6; rng.shuffle(groups)
 # Overlapping Gaussian clouds: group identity is only weakly spatially informative.
 theta=groups*2*np.pi/6; centers=np.c_[250+spatial_separation*np.cos(theta),250+spatial_separation*np.sin(theta)]
 xy=centers+rng.normal(0,105,(n,2)); xy=np.clip(xy,0,500)
 L=[f'source_ligand_{c}' for c in list('HAFDGBCE')[:communication_genes]]; R=[f'receptor_{c}' for c in list('CEHABGDF')[:communication_genes]]; T=[f'target_{c}' for c in list('FBHDECAG')[:communication_genes]]
 markers=[f'marker_group_{g}_{j}' for g in range(6) for j in range(3)]; noise=[f'noise_{j}' for j in range(36)]
 genes=markers+noise+L+R+T; gi={g:i for i,g in enumerate(genes)}
 state=rng.normal(size=(n,4)); mu=np.full((n,len(genes)),.35)
 receptor_positive=np.zeros((n,len(R)),dtype=bool)
 # Each marker is shared by three groups; own-group enrichment is moderate, not diagnostic.
 for g in range(6):
  for j in range(3):
   v=.45+.35*np.maximum(state[:,j],-1)
   v+=marker_strength*(groups==g)
   v+=.58*marker_strength*np.isin(groups,[(g-1)%6,(g+1)%6])
   mu[:,gi[f'marker_group_{g}_{j}']]+=v
 # Communication genes are broadly expressed; preferences overlap across sender/receiver populations.
 for k in range(len(L)):
  if separated_ligand_domains:
   angle=2*np.pi*k/len(L); center=np.array([250+ligand_domain_radius*np.cos(angle),250+ligand_domain_radius*np.sin(angle)]); domain=np.exp(-np.sum((xy-center)**2,axis=1)/(2*ligand_domain_sigma**2))
   mu[:,gi[L[k]]]=.08+ligand_domain_scale*domain*(groups==(3+k%3))
  else:
   mu[:,gi[L[k]]]+=.65+.35*np.maximum(state[:,k%4],-1)
  if bimodal_receptor:
   compatible=np.isin(groups,[k%3,(k+1)%3]); receptor_positive[:,k]=rng.random(n)<np.where(compatible,.78,.12)
   low=np.minimum(rng.lognormal(np.log(.28),.22,n),receptor_gate_threshold*.8); high=np.maximum(rng.lognormal(np.log(4.0),.18,n),receptor_gate_threshold*1.2)
   mu[:,gi[R[k]]]=np.where(receptor_positive[:,k],high,low)
  else:
   mu[:,gi[R[k]]]+=.65+.35*np.maximum(state[:,(k+1)%4],-1)
   mu[:,gi[R[k]]]+=type_preference*np.isin(groups,[k%3,(k+1)%3])
  pattern=.25*(1+np.sin(xy[:,0]/37+1.7*k))*(1+np.cos(xy[:,1]/43-1.3*k))
  mu[:,gi[L[k]]]+=ligand_spatial_scale*pattern*np.isin(groups,[3,4,5])
  hotspots=(rng.random(n)<.08)*rng.uniform(.6,1.4,n)*np.isin(groups,[3,4,5]) if ligand_hotspot_scale>0 else np.zeros(n)
  mu[:,gi[L[k]]]+=ligand_hotspot_scale*hotspots
 # Random many-to-many network by default; the easy pilot uses identifiable one-to-one chains.
 if simple_one_to_one:
  chains=[(3+k%3,L[k],R[k],k%3,T[k]) for k in range(len(L))]
 else:
  chains=[]; used=set()
  while len(chains)<9:
   li=int(rng.integers(0,min(6,len(L)))); ri=int(rng.integers(0,min(6,len(R)))); rg=int(rng.choice([ri%3,(ri+1)%3])) if separated_ligand_domains else int(rng.integers(0,3)); ti=int(rng.integers(0,min(6,len(T)))); sg=3+li%3 if separated_ligand_domains else 3+int(rng.integers(0,3))
   key=(sg,L[li],R[ri],rg,T[ti])
   if key not in used: used.add(key); chains.append(key)
 for k,t in enumerate(T):
  mu[:,gi[t]]+=target_background_scale*(.9+.55*(groups==k%3)+.42*state[:,k%4]+.55*np.sin(xy[:,0]/58+k)+.42*np.cos(xy[:,1]/73-k))
 d=np.sqrt(((xy[:,None,:]-xy[None,:,:])**2).sum(2)); K=np.where((d<=320)&(d>0),k0(np.maximum(d,5)/130),0) if aligned_kernel else np.exp(-d/105); np.fill_diagonal(K,0)
 lig=np.maximum(mu[:,[gi[z] for z in L]],0); rec=np.maximum(mu[:,[gi[z] for z in R]],0); records=[]
 for sg,l,r,rg,t in chains:
  source_mask=groups==sg; field=K[:,source_mask]@lig[source_mask,L.index(l)]; field/=np.quantile(field,.9)+1e-8
  receptor_values=rec[:,R.index(r)]; gate=(receptor_positive[:,R.index(r)] if bimodal_receptor else receptor_values>=np.quantile(receptor_values,receptor_gate_quantile)) if binary_receptor_gate else np.ones(n,dtype=bool); signal=field if (direct_field_signal or binary_receptor_gate) else field*receptor_values; mask=(groups==rg)&gate; beta=float(rng.uniform(.85,1.25))*effect_scale; mu[mask,gi[t]]+=beta*signal[mask]; records.append((sg,l,r,rg,t,beta))
 decoys=[]
 for q in range(6,min(8,len(L))):
  pair=(L[q],R[(q+1)%8]); decoys.append(pair); mu[:,gi[L[q]]]+=.5*decoy_scale*np.exp(-((xy[:,0]-(120+q*35))/90)**2); mu[:,gi[R[(q+1)%8]]]+=.5*decoy_scale*np.exp(-((xy[:,1]-(390-q*30))/90)**2)
 mu=np.clip(mu,.03,None)*rng.lognormal(0,.25,n)[:,None]; disp=dispersion
 x=rng.negative_binomial(disp,disp/(disp+mu)).astype(float); x[rng.random(x.shape)<np.exp(-mu/1.8)*dropout_rate]=0
 np.savetxt(out/'expression.csv',x,delimiter=',',fmt='%.0f'); np.savetxt(out/'spatial_coordinates.csv',xy,delimiter=',',fmt='%.7g')
 (out/'gene_ids.txt').write_text('\n'.join(genes)+'\n'); (out/'cell_ids.txt').write_text('\n'.join(f'cell_{i}' for i in range(n))+'\n'); (out/'cell_groups.txt').write_text('\n'.join(map(str,groups))+'\n')
 pd.DataFrame(records,columns=['sender_group','ligand','receptor','receiver_group','target','effect']).to_csv(out/'truth_hierarchical.csv',index=False)
 pd.DataFrame([(L[k],R[k]) for k in range(len(L))] if simple_one_to_one else [(l,r) for l in L for r in R],columns=['ligand','receptor']).to_csv(out/'candidate_lr.csv',index=False)
 pd.DataFrame([(l,t) for l in L for t in T],columns=['source_gene','target_gene']).to_csv(out/'candidate_source_target.csv',index=False)
 (out/'metadata.json').write_text(json.dumps({'scenario':'shared_inferred_prior','generation_seed':seed,'cells':n,'genes':len(genes),'marker_strength':marker_strength,'type_preference':type_preference,'spatial_separation':spatial_separation,'communication_effect_scale':effect_scale,'dispersion':dispersion,'dropout_rate':dropout_rate,'target_background_scale':target_background_scale,'decoy_scale':decoy_scale,'ligand_spatial_scale':ligand_spatial_scale,'aligned_kernel':aligned_kernel,'direct_field_signal':direct_field_signal,'ligand_hotspot_scale':ligand_hotspot_scale,'binary_receptor_gate':binary_receptor_gate,'receptor_gate_quantile':receptor_gate_quantile,'bimodal_receptor':bimodal_receptor,'receptor_gate_threshold':receptor_gate_threshold,'true_chains':len(records),'unique_receiver_targets':len({(x[3],x[4]) for x in records}),'candidate_lr_pairs':len(L) if simple_one_to_one else len(L)*len(R),'candidate_source_target_pairs':len(L)*len(T),'communication_genes':communication_genes,'simple_one_to_one':simple_one_to_one,'separated_ligand_domains':separated_ligand_domains,'ligand_domain_scale':ligand_domain_scale,'ligand_domain_radius':ligand_domain_radius,'ligand_domain_sigma':ligand_domain_sigma,'decoy_lr_pairs':decoys},indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True); p.add_argument('--seed',type=int,default=167); p.add_argument('--cells',type=int,default=600); p.add_argument('--marker-strength',type=float,default=3.5); p.add_argument('--type-preference',type=float,default=1.8); p.add_argument('--spatial-separation',type=float,default=80); p.add_argument('--effect-scale',type=float,default=1.0); p.add_argument('--dispersion',type=float,default=3.5); p.add_argument('--dropout-rate',type=float,default=.18); p.add_argument('--target-background-scale',type=float,default=1.0); p.add_argument('--decoy-scale',type=float,default=1.0); p.add_argument('--ligand-spatial-scale',type=float,default=0.0); p.add_argument('--aligned-kernel',action='store_true'); p.add_argument('--direct-field-signal',action='store_true'); p.add_argument('--ligand-hotspot-scale',type=float,default=0.0); p.add_argument('--binary-receptor-gate',action='store_true'); p.add_argument('--receptor-gate-quantile',type=float,default=.6); p.add_argument('--bimodal-receptor',action='store_true'); p.add_argument('--receptor-gate-threshold',type=float,default=1.0); p.add_argument('--communication-genes',type=int,default=8); p.add_argument('--simple-one-to-one',action='store_true'); p.add_argument('--separated-ligand-domains',action='store_true'); p.add_argument('--ligand-domain-scale',type=float,default=20.0); p.add_argument('--ligand-domain-radius',type=float,default=175.0); p.add_argument('--ligand-domain-sigma',type=float,default=42.0)
 a=p.parse_args(); generate(a.out,a.seed,a.cells,a.marker_strength,a.type_preference,a.spatial_separation,a.effect_scale,a.dispersion,a.dropout_rate,a.target_background_scale,a.decoy_scale,a.ligand_spatial_scale,a.aligned_kernel,a.direct_field_signal,a.ligand_hotspot_scale,a.binary_receptor_gate,a.receptor_gate_quantile,a.bimodal_receptor,a.receptor_gate_threshold,a.communication_genes,a.simple_one_to_one,a.separated_ligand_domains,a.ligand_domain_scale,a.ligand_domain_radius,a.ligand_domain_sigma)
