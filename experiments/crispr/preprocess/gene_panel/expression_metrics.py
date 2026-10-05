"""Sparse-safe non-target expression and spatial metrics."""
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
def asum(x,a): return np.asarray(x.sum(axis=a)).ravel()
def sqsum(x,a): return np.asarray(x.multiply(x).sum(axis=a)).ravel() if sparse.issparse(x) else np.sum(x*x,axis=a)
def technical_flag(g):
 s=str(g); l=s.lower(); return l.startswith("mt-") or s.startswith(("Rpl","Rps","Gm")) or "Rik" in s or l in {"malat1","gfp","cas9","sgrna"} or "vector" in l
def morans_i_matrix(x,coords,k=6):
 """Compute Moran's I for every feature on a directed spatial kNN graph."""
 n=x.shape[0]
 if n<4: return np.full(x.shape[1],np.nan)
 nei=cKDTree(np.asarray(coords)).query(np.asarray(coords),k=min(k+1,n))[1][:,1:]; rows=np.repeat(np.arange(n),nei.shape[1]); cols=nei.ravel(); w=sparse.csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(n,n)); s0=float(w.sum()); means=asum(x,0)/n; wx=w@x
 term=asum(x.multiply(wx),0) if sparse.issparse(x) else np.sum(x*wx,axis=0)
 row=asum(w,1); col=asum(w,0); left=asum(x.multiply(row[:,None]),0) if sparse.issparse(x) else np.sum(x*row[:,None],axis=0); right=asum(x.multiply(col[:,None]),0) if sparse.issparse(x) else np.sum(x*col[:,None],axis=0)
 numerator=term-means*(left+right)+means*means*s0; denominator=sqsum(x,0)-n*means*means; return (n/s0)*numerator/np.where(denominator>0,denominator,np.nan)
def compute_metrics(adata,celltype_column,raw_layer=None,normalized_layer=None,sample_column=None,min_cells=30,spatial_key="spatial"):
 raw=adata.layers[raw_layer] if raw_layer else adata.X; norm=adata.layers[normalized_layer] if normalized_layer else adata.X; n=adata.n_obs; det=asum(raw>0,0); mean=asum(norm,0)/n; var=np.maximum(sqsum(norm,0)/n-mean**2,0); labels=adata.obs[celltype_column].astype(str); names=sorted(labels.unique()); ds=[]; ms=[]; vs=[]
 for ct in names:
  idx=np.flatnonzero((labels==ct).to_numpy()); x=norm[idx]; r=raw[idx]; m=asum(x,0)/len(idx); v=np.maximum(sqsum(x,0)/len(idx)-m*m,0); ds.append(asum(r>0,0)/len(idx)); ms.append(m); vs.append(v/(m+1e-6) if len(idx)>=min_cells else np.full(adata.n_vars,np.nan))
 D=np.vstack(ds); M=np.vstack(ms); P=M/(M.sum(0,keepdims=True)+1e-9); tau=np.sum(1-P,axis=0)/max(len(names)-1,1); within=np.nan_to_num(np.nanmedian(np.vstack(vs),axis=0)); spatial_i=np.full(adata.n_vars,np.nan)
 if spatial_key in adata.obsm:
  if sample_column and sample_column in adata.obs and adata.obs[sample_column].nunique()>1:
   vals=[]
   for _,idx in adata.obs.groupby(sample_column,observed=True).indices.items():
    if len(idx)>=4: vals.append(morans_i_matrix(norm[idx],adata.obsm[spatial_key][idx]))
   if vals: spatial_i=np.nanmedian(np.vstack(vals),axis=0)
  else: spatial_i=morans_i_matrix(norm,adata.obsm[spatial_key])
 frame=pd.DataFrame({"gene":adata.var_names.astype(str),"n_cells_detected":det.astype(int),"detection_rate":det/n,"mean_expression":mean,"variance":var,"nonzero_mean":asum(norm,0)/np.maximum(det,1),"max_celltype_detection":D.max(0),"celltype_specificity":tau,"within_type_variability":within,"batch_consistency":1.0,"spatial_variability":spatial_i,"technical_flag":[technical_flag(g) for g in adata.var_names]}).set_index("gene")
 return frame,pd.DataFrame(D,index=names,columns=adata.var_names),pd.DataFrame(M,index=names,columns=adata.var_names)
