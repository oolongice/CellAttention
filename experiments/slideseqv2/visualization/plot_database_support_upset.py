#!/usr/bin/env python3
"""Compact UpSet plot of process support for model source-target pairs."""
OUTPUT_FORMAT="png";PNG_DPI=600;FIGURE_WIDTH_IN=2.2;FIGURE_HEIGHT_IN=1.3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from slideseq_common import HERE,configure_style
MEMBERSHIP=HERE/"data/database_support_venn_membership.csv"
ROWS=["CellChatDB","NeuronChatDB","MetaChatDB"]
COLORS={(True,False,True):"#6F9FB9",(False,False,True):"#DEA16C",(True,True,True):"#716D99",(False,False,False):"#AFAFAF",(False,True,True):"#9A86BC"}
def main():
 configure_style();data=pd.read_csv(MEMBERSHIP);fields=["protein_signaling","neural_signaling","metabolite_signaling"]
 patterns=data.groupby(fields).size().rename("pair_count").reset_index();patterns["pattern"]=list(map(tuple,patterns[fields].to_numpy(dtype=bool)));patterns=patterns[patterns.pair_count>0];patterns["is_others"]=[not any(p) for p in patterns.pattern];patterns=patterns.sort_values(["is_others","pair_count"],ascending=[True,False]).reset_index(drop=True);patterns["display_label"]=["Others" if is_others else "" for is_others in patterns.is_others]
 patterns.to_csv(HERE/"data/database_support_upset_counts.csv",index=False)
 fig=plt.figure(figsize=(FIGURE_WIDTH_IN,FIGURE_HEIGHT_IN));grid=fig.add_gridspec(2,1,height_ratios=[1.45,1],hspace=.08,left=.30,right=.98,bottom=.10,top=.82);bar=fig.add_subplot(grid[0]);mat=fig.add_subplot(grid[1],sharex=bar)
 x=np.arange(len(patterns));colors=[COLORS.get(p,"#7E98A8") for p in patterns.pattern];bar.bar(x,patterns.pair_count,width=.68,color=colors,linewidth=0)
 for xpos,count in zip(x,patterns.pair_count):bar.text(xpos,count+.7,str(int(count)),ha="center",va="bottom",fontsize=4.5,fontweight="bold")
 bar.set_ylim(0,patterns.pair_count.max()*1.25);bar.set_ylabel("Gene pairs",fontsize=4.2,labelpad=2);bar.set_xticks([]);bar.tick_params(axis="y",labelsize=3.3,length=1.8,pad=1.5);bar.spines[["top","right"]].set_visible(False);bar.spines[["left","bottom"]].set_linewidth(.45)
 for xpos,pattern in zip(x,patterns.pattern):
  selected=[i for i,present in enumerate(pattern) if present]
  if len(selected)>1:mat.plot([xpos,xpos],[min(selected),max(selected)],color="#454545",linewidth=.8,zorder=1)
  for row in range(3):mat.scatter(xpos,row,s=14 if pattern[row] else 8,color=colors[xpos] if pattern[row] else "#D8D8D8",edgecolor="none",zorder=2)
  if not any(pattern):mat.text(xpos,1,"×",ha="center",va="center",fontsize=5.2,fontweight="bold",color="#666",zorder=3)
 mat.set_yticks(range(3),ROWS,fontsize=3.45);mat.set_ylim(2.55,-.55);mat.set_xticks(x,patterns.display_label,fontsize=3.8);mat.tick_params(axis="both",length=0,pad=1.5);mat.spines[:].set_visible(False)
 for row in range(3):mat.axhline(row,color="#EFEFEF",linewidth=.4,zorder=0)
 fig.suptitle("Database support for model gene pairs",x=.02,y=.96,ha="left",fontsize=6.3,fontweight="bold")
 out=HERE/"figures"/f"database_support_gene_pair_upset.{OUTPUT_FORMAT}";kw={"facecolor":"white"}
 if OUTPUT_FORMAT=="png":kw.update({"dpi":PNG_DPI,"pil_kwargs":{"compress_level":6}})
 fig.savefig(out,**kw);fig.savefig(out.with_suffix(".pdf"),dpi=PNG_DPI,facecolor="white");plt.close(fig);print(patterns[["pattern","pair_count","display_label"]].to_string(index=False));print(f"output={out}")
if __name__=="__main__":main()
