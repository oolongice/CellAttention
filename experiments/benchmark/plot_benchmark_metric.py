#!/usr/bin/env python3
"""Shared fixed-size plotting helper for benchmark metric panels."""
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

FIGSIZE=(11.2,6.4)
DPI=240
POSITIONS=[0,2,3,4,5,7,8,9]
XTICK_LABELS=[
 'CellAttention',
 'MISTy\nall genes',
 'MISTy\nsource/target\nspecified',
 'HoloNet\nno database',
 'COMMOT\nno database',
 'HoloNet\nfull database',
 'COMMOT\npartial database',
 'COMMOT\nfull database',
]
COLORS=['#7b3294','#4c78a8','#72b7b2','#bdbdbd','#bdbdbd','#e07b39','#f6c85f','#f2b134']
DIFFICULTIES=['hard','moderate','easy']
DIFFICULTY_COLORS=['#355f8a','#2a9d8f','#e9c46a']
PARAMETER_LABELS=['Parameter 1','Parameter 2','Parameter 3']

def plot_metric(detail,items,metric,metric_title,task_title,output,random_line=None):
 fig,ax=plt.subplots(figsize=FIGSIZE)
 ax.axvspan(-.55,.55,color='#7b3294',alpha=.055,zorder=0)
 ax.axvspan(1.45,6.55,color='#4c78a8',alpha=.045,zorder=0)
 ax.axvspan(7.45,9.55,color='#f2b134',alpha=.09,zorder=0)
 stats=[]
 for method in items:
  z=detail[detail.method==method][metric] if method is not None else []
  stats.append((float(z.mean()),float(z.std(ddof=1) if len(z)>1 else 0.0)) if len(z) else (np.nan,np.nan))
 tops=[m+s for m,s in stats if np.isfinite(m)]
 ymax=max(max(tops)*1.58,.075)
 for pos,color,(mean,sd) in zip(POSITIONS,COLORS,stats):
  if np.isfinite(mean):
   ax.bar(pos,mean,yerr=sd,capsize=5,width=.72,color=color,zorder=3)
   ax.text(pos,mean+sd+ymax*.026,f'{mean:.3f}',ha='center',va='bottom',fontsize=15,fontweight='semibold')
  else:
   ax.text(pos,ymax*.06,'N/A',ha='center',va='center',fontsize=15,fontweight='semibold',color='#555')
 if random_line is not None:
  ax.axhline(random_line,color='#d62728',linestyle='--',linewidth=1.8,zorder=2)
  ax.text(9.5,random_line+ymax*.018,f'Random AP = {random_line:.3f}',ha='right',fontsize=13,color='#d62728',fontweight='semibold')
 ax.set_xlim(-.75,9.75); ax.set_ylim(0,ymax)
 ax.set_xticks(POSITIONS,XTICK_LABELS)
 for label in ax.get_xticklabels():
  label.set_fontsize(15); label.set_rotation(30); label.set_ha('right'); label.set_rotation_mode('anchor')
 ax.tick_params(axis='x',pad=10,length=0); ax.tick_params(axis='y',labelsize=14)
 ax.set_ylabel(metric_title,fontsize=17,labelpad=9)
 ax.grid(axis='y',alpha=.22,zorder=0)
 ax.text(0,1.055,'No predefined clusters\nor database',transform=ax.get_xaxis_transform(),ha='center',fontsize=14,fontweight='semibold',clip_on=False)
 ax.text(3.5,1.055,'Predefined clusters;\nno database',transform=ax.get_xaxis_transform(),ha='center',fontsize=14,fontweight='semibold',clip_on=False)
 ax.text(8,1.055,'Predefined clusters\nand database',transform=ax.get_xaxis_transform(),ha='center',fontsize=14,fontweight='semibold',clip_on=False)
 fig.suptitle(f'{task_title} — {metric_title}',fontsize=21,fontweight='semibold',y=.97)
 fig.subplots_adjust(left=.105,right=.985,bottom=.29,top=.70)
 fig.savefig(Path(output),dpi=DPI,facecolor='white')
 fig.savefig(Path(output).with_suffix('.pdf'),facecolor='white')
 plt.close(fig)

def plot_metric_by_difficulty(detail,items,metric,metric_title,task_title,output,random_lines=None):
 fig,ax=plt.subplots(figsize=(12.4,6.8)); width=.22; offsets=(-.25,0,.25)
 ax.axvspan(-.65,.65,color='#7b3294',alpha=.055,zorder=0); ax.axvspan(1.35,5.65,color='#4c78a8',alpha=.045,zorder=0); ax.axvspan(6.35,9.65,color='#f2b134',alpha=.09,zorder=0)
 stats={}
 for method in items:
  for difficulty in DIFFICULTIES:
   z=detail[(detail.method==method)&(detail.difficulty==difficulty)][metric] if method is not None else []
   stats[(method,difficulty)]=(float(z.mean()),float(z.std(ddof=1) if len(z)>1 else 0.0)) if len(z) else (np.nan,np.nan)
 tops=[m+s for m,s in stats.values() if np.isfinite(m)]; ymax=max(max(tops)*1.65,.075)
 for pos,method,method_color in zip(POSITIONS,items,COLORS):
  available=False
  for offset,difficulty,hatch in zip(offsets,DIFFICULTIES,('///','...','')):
   mean,sd=stats[(method,difficulty)]
   if np.isfinite(mean):
    available=True; ax.bar(pos+offset,mean,yerr=sd,capsize=3,width=width,color=method_color,edgecolor='#333333',linewidth=.45,hatch=hatch,zorder=3)
    ax.text(pos+offset,mean+sd+ymax*.018,f'{mean:.2f}',ha='center',va='bottom',fontsize=12,fontweight='semibold',rotation=90)
  if not available: ax.text(pos,ymax*.055,'N/A',ha='center',va='center',fontsize=14,fontweight='semibold',color='#555')
 if random_lines:
  values=sorted({round(float(v),10) for v in random_lines.values()}); value=values[0]
  ax.axhline(value,color='#d62728',linestyle='--',linewidth=1.6,alpha=.9,zorder=2)
  for extra in values[1:]: ax.axhline(extra,color='#d62728',linestyle=':',linewidth=1.4,alpha=.9,zorder=2)
 ax.set_xlim(-.85,9.85); ax.set_ylim(0,ymax); ax.set_xticks(POSITIONS,XTICK_LABELS)
 for label in ax.get_xticklabels(): label.set_fontsize(15); label.set_rotation(30); label.set_ha('right'); label.set_rotation_mode('anchor')
 ax.tick_params(axis='x',pad=10,length=0); ax.tick_params(axis='y',labelsize=14); ax.set_ylabel(metric_title,fontsize=17,labelpad=9); ax.grid(axis='y',alpha=.22,zorder=0)
 ax.text(0,1.055,'No predefined clusters\nor database',transform=ax.get_xaxis_transform(),ha='center',fontsize=14,fontweight='semibold',clip_on=False)
 ax.text(3.5,1.055,'Predefined clusters;\nno database',transform=ax.get_xaxis_transform(),ha='center',fontsize=14,fontweight='semibold',clip_on=False)
 ax.text(8,1.055,'Predefined clusters\nand database',transform=ax.get_xaxis_transform(),ha='center',fontsize=14,fontweight='semibold',clip_on=False)
 fig.legend(handles=[Patch(facecolor='white',edgecolor='#333',hatch=h,label=label) for label,h in zip(PARAMETER_LABELS,('///','...',''))],frameon=False,ncol=3,loc='lower right',bbox_to_anchor=(.985,.018),fontsize=13)
 fig.suptitle(f'{task_title} — {metric_title}',fontsize=21,fontweight='semibold',y=.93); fig.subplots_adjust(left=.10,right=.985,bottom=.34,top=.76)
 fig.savefig(Path(output),dpi=DPI,facecolor='white')
 fig.savefig(Path(output).with_suffix('.pdf'),facecolor='white')
 plt.close(fig)
