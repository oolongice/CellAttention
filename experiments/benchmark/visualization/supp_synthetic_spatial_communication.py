#!/usr/bin/env python3
"""Supplementary spatial plots for all planted chains in three benchmark scenarios.

Run with the benchmark report environment. SEED selects the representative replicate.
Expression colors show raw counts, clipped at each gene's 99th percentile, as in
figures/data/representative_chain_expression.png. No candidate/negative chains
are included: all truth rows are assembled into one compact montage per scenario.
"""
from pathlib import Path
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd

# User settings
SEED = int(os.environ.get('CELLATTENTION_VISUALIZATION_SEED', '168'))
DPI = 600

ROOT=Path(os.environ['CELLATTENTION_BENCHMARK_WORKDIR']); FONT=ROOT/'font'; OUT=ROOT/'visualization'/'supp_figures'
GROUP_COLORS=['#0072B2','#E69F00','#009E73','#CC79A7','#56B4E9','#D55E00']
SCENARIOS = [('simple', 'Simple'), ('complex_lr', 'Complex LR'), ('spatial_overlap', 'Spatial overlap')]
SOURCE_CMAP=mpl.colors.LinearSegmentedColormap.from_list('source',['#F2F2F2','#56B4E9','#002B5B'])
RECEPTOR_CMAP=mpl.colors.LinearSegmentedColormap.from_list('receptor',['#F2F2F2','#009E73','#004D38'])
TARGET_CMAP=mpl.colors.LinearSegmentedColormap.from_list('target',['#F2F2F2','#E69F00','#7A3E00'])

def style():
 files=[FONT/'arial.ttf',FONT/'arialbd.ttf',FONT/'ariali.ttf',FONT/'arialbi.ttf']
 if all(p.is_file() for p in files):
  for p in files: fm.fontManager.addfont(p)
  family=fm.FontProperties(fname=str(files[0])).get_name()
 else:
  family='DejaVu Sans'
 mpl.rcParams.update({'font.family':family,'font.sans-serif':[family],'font.size':7,'axes.titlesize':8,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,'legend.fontsize':5.5,'axes.linewidth':.6,'xtick.major.width':.6,'ytick.major.width':.6,'pdf.fonttype':42,'ps.fonttype':42})

def load(scenario):
 d=ROOT/'data'/f'scenario_{scenario}_seed_{SEED}'
 genes=(d/'gene_ids.txt').read_text().splitlines()
 x=np.loadtxt(d/'expression.csv',delimiter=',')
 xy=np.loadtxt(d/'spatial_coordinates.csv',delimiter=',')
 groups=np.loadtxt(d/'cell_groups.txt',dtype=int)
 truth=pd.read_csv(d/'truth_hierarchical.csv')
 if x.shape != (len(groups),len(genes)) or xy.shape != (len(groups),2):
  raise ValueError(f'Inconsistent data dimensions: {d}')
 if not np.isfinite(x).all() or not np.isfinite(xy).all():
  raise ValueError(f'Nonfinite expression or coordinates: {d}')
 required=set(truth.ligand)|set(truth.receptor)|set(truth.target)
 if not required.issubset(genes):
  raise ValueError(f'Missing chain genes: {required-set(genes)}')
 return x,xy,groups,genes,truth

def clean(ax):
 ax.set_aspect('equal'); ax.set_xlabel('Spatial x (µm)'); ax.set_ylabel('Spatial y (µm)'); ax.spines[['top','right']].set_visible(False); ax.tick_params(length=2.5,pad=1.5)

def save(fig,name):
 OUT.mkdir(parents=True,exist_ok=True); path=OUT/name; fig.savefig(path,dpi=DPI,facecolor='white',pil_kwargs={'compress_level':6}); fig.savefig(path.with_suffix('.pdf'),facecolor='white'); plt.close(fig); print(path)

def group_figure(xy,labels,title,prefix,name):
 fig,ax=plt.subplots(figsize=(89/25.4,82/25.4),constrained_layout=True)
 for g in sorted(np.unique(labels)):
  m=labels==g; ax.scatter(xy[m,0],xy[m,1],s=5,color=GROUP_COLORS[int(g)],edgecolors='none',alpha=.88,label=f'{prefix} {g}')
 ax.set_title(title,loc='left',fontweight='bold',pad=4); clean(ax); ax.legend(loc='upper right',ncol=2,frameon=True,framealpha=.92,borderpad=.35,handletextpad=.25,columnspacing=.55,markerscale=1.25,labelspacing=.25)
 save(fig,name)

def expression_panel(ax,xy,values,title,gene,cmap,vmax=None):
 upper=max(float(np.quantile(values,.99)),1) if vmax is None else vmax
 ax.scatter(xy[:,0],xy[:,1],s=4.2,color='#D9D9D9',edgecolors='none',alpha=.5)
 m=values>0; pts=ax.scatter(xy[m,0],xy[m,1],c=values[m],s=6,cmap=cmap,vmin=0,vmax=upper,edgecolors='none',alpha=.95)
 ax.set_title(title,loc='left',fontweight='bold',pad=4); clean(ax)
 ax.text(.02,.02,gene,transform=ax.transAxes,ha='left',va='bottom',fontsize=6.5,bbox={'facecolor':'white','edgecolor':'none','alpha':.82,'pad':1.5})
 cb=ax.figure.colorbar(pts,ax=ax,fraction=.045,pad=.025); cb.set_label('Expression',labelpad=2); cb.outline.set_linewidth(.5); cb.ax.tick_params(length=2,width=.5,pad=1)

def chain_expression(x,xy,genes,chain,name):
 gi={g:i for i,g in enumerate(genes)}; names=[chain.ligand,chain.receptor,chain.target]; cmaps=[SOURCE_CMAP,RECEPTOR_CMAP,TARGET_CMAP]; titles=['Ligand expression','Receptor expression','Target expression']
 fig,axes=plt.subplots(1,3,figsize=(183/25.4,62/25.4),constrained_layout=True); fig.set_constrained_layout_pads(w_pad=.025,h_pad=.025,wspace=.08,hspace=.05)
 for ax,gene,cmap,title in zip(axes,names,cmaps,titles): expression_panel(ax,xy,x[:,gi[gene]],title,gene,cmap)
 fig.suptitle(f'Group {chain.sender_group} → Group {chain.receiver_group}',fontsize=8,fontweight='bold')
 save(fig,name)

def network_figure(truth):
 fig,ax=plt.subplots(figsize=(60/25.4,62/25.4)); fig.subplots_adjust(left=.04,right=.96,bottom=.13,top=.88)
 ligands=sorted(truth.ligand.unique()); receptors=sorted(truth.receptor.unique()); targets=sorted(truth.target.unique())
 layers=[ligands,receptors,targets]; xs=[.16,.50,.84]; ys=[dict(zip(v,np.linspace(.82,.18,len(v)))) for v in layers]
 for row in truth.itertuples():
  color=GROUP_COLORS[int(row.receiver_group)]
  ax.add_patch(FancyArrowPatch((xs[0]+.035,ys[0][row.ligand]),(xs[1]-.035,ys[1][row.receptor]),arrowstyle='-|>',mutation_scale=4.5,linewidth=.65,color=color,alpha=.72,connectionstyle='arc3,rad=.05'))
  ax.add_patch(FancyArrowPatch((xs[1]+.035,ys[1][row.receptor]),(xs[2]-.035,ys[2][row.target]),arrowstyle='-|>',mutation_scale=4.5,linewidth=.65,color=color,alpha=.72,connectionstyle='arc3,rad=-.05'))
 for x,nodes,pos,color,marker in zip(xs,layers,ys,['#56B4E9','#009E73','#E69F00'],['o','D','s']):
  ax.scatter([x]*len(nodes),list(pos.values()),s=30,color=color,marker=marker,edgecolor='#222',linewidth=.35,zorder=3)
  for gene,y in pos.items():
   label=gene.replace('source_ligand_','L').replace('receptor_','R').replace('target_','T')
   ax.text(x,y-.032,label,ha='center',va='top',fontsize=5)
 for g in sorted(truth.receiver_group.unique()):
  ax.plot([],[],color=GROUP_COLORS[int(g)],linewidth=1.1,label=f'Receiver {g}')
 ax.legend(loc='lower center',bbox_to_anchor=(.5,.005),ncol=3,frameon=False,handlelength=.9,columnspacing=.5,handletextpad=.25,fontsize=5)
 for x,title in zip(xs,['Ligands','Receptors','Targets']):
  ax.text(x,.97,title,ha='center',va='top',fontweight='bold',fontsize=6)
 ax.set_title('Planted communication network',loc='left',fontweight='bold',pad=3,fontsize=7)
 ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off'); save(fig,'planted_communication_network.png')


def chain_montage(x,xy,genes,truth,label):
 gi={gene:index for index,gene in enumerate(genes)}
 chains=list(truth.itertuples(index=False)); nrows=int(np.ceil(len(chains)/3)); ncols=9
 height=22+28*nrows
 fig,axes=plt.subplots(nrows,ncols,figsize=(180/25.4,height/25.4),squeeze=False)
 fig.subplots_adjust(left=.025,right=.975,bottom=.04,top=.84,wspace=.08,hspace=.34)
 roles=[('ligand',SOURCE_CMAP,'Ligand'),('receptor',RECEPTOR_CMAP,'Receptor'),('target',TARGET_CMAP,'Target')]
 vmax=[]
 for field,_,_ in roles:
  values=[x[:,gi[getattr(chain,field)]] for chain in chains]
  vmax.append(max(1.0,max(float(np.quantile(value,.99)) for value in values)))
 for index,chain in enumerate(chains):
  row=index//3; block=index%3; block_axes=axes[row,block*3:block*3+3]
  for role_index,(ax,(field,cmap,_)) in enumerate(zip(block_axes,roles)):
   gene=getattr(chain,field); values=x[:,gi[gene]]
   ax.scatter(xy[:,0],xy[:,1],s=1.5,color='#D9D9D9',edgecolors='none',alpha=.45)
   positive=values>0
   ax.scatter(xy[positive,0],xy[positive,1],c=values[positive],s=2.2,cmap=cmap,vmin=0,vmax=vmax[role_index],edgecolors='none',alpha=.95)
   ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([]); ax.spines[:].set_visible(False)
   ax.set_title(gene,fontsize=5.5,fontweight='bold',pad=1.5)
  block_axes[1].text(.5,1.23,f'Group {chain.sender_group} → Group {chain.receiver_group}',transform=block_axes[1].transAxes,ha='center',va='bottom',fontsize=6.5,fontweight='bold')
 for index in range(len(chains),nrows*3):
  row=index//3; block=index%3
  for ax in axes[row,block*3:block*3+3]:ax.axis('off')
 fig.suptitle(f'{label}: planted L–R–T expression',x=.025,y=.975,ha='left',fontsize=8,fontweight='bold')
 fig.text(.665,.953,'Expression',ha='right',va='center',fontsize=5.5)
 for role_index,(_,cmap,role) in enumerate(roles):
  cax=fig.add_axes([.68+.10*role_index,.945,.075,.012])
  sm=mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0,vmax[role_index]),cmap=cmap)
  cb=fig.colorbar(sm,cax=cax,orientation='horizontal'); cb.set_ticks([0,vmax[role_index]])
  cb.ax.tick_params(labelsize=5,length=1.5,width=.4,pad=1); cb.outline.set_linewidth(.4)
  cb.ax.set_title(role,fontsize=5,pad=1)
 save(fig,'all_chain_expression.png')


def main():
 global OUT
 style()
 for scenario,label in SCENARIOS:
  if scenario not in os.environ.get('CELLATTENTION_SCENARIOS', 'simple,complex_lr,spatial_overlap').split(','):
   continue
  OUT=ROOT/'visualization'/'supp_figures'/scenario
  x,xy,groups,genes,truth=load(scenario)
  group_figure(xy,groups,f'{label}: cell-group organization','Group','true_cell_groups.png')
  network_figure(truth)
  chain_montage(x,xy,genes,truth,label)
  records=[]
  for index,chain in enumerate(truth.itertuples(index=False),1):
   records.append(dict(chain._asdict(),expression_figure='all_chain_expression.png',
                       panel_row=(index-1)//3+1,panel_group=(index-1)%3+1,
                       seed=SEED,scenario=scenario))
  pd.DataFrame(records).to_csv(OUT/'chain_manifest.csv',index=False)
if __name__=='__main__': main()
