#!/usr/bin/env python3
"""Prepare the 3D 6-somite zebrafish weMERFISH dataset for CellAttention."""
from pathlib import Path
import json, os, tempfile, zipfile
import anndata as ad
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

HERE=Path(__file__).resolve().parent
ROOT=Path(os.environ["CELLATTENTION_MERFISH_WORKDIR"])
ARCHIVE=ROOT/'data/raw/doi_10_5061_dryad_j0zpc86v9__v20251211.zip'
OUT=ROOT/'data/preprocessed'
MEMBER='weMERFISH_measured_C_6s_E1_rescaled_z.h5ad'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='wemfish_prepare_') as tmp:
        with zipfile.ZipFile(ARCHIVE) as z:
            if z.testzip() is not None: raise RuntimeError('archive CRC validation failed')
            z.extract(MEMBER,tmp)
        a=ad.read_h5ad(Path(tmp)/MEMBER)
    x=np.asarray(a.X,dtype=np.float32)
    coords=np.asarray(a.obsm['spatial'],dtype=np.float64)
    if x.shape!=(a.n_obs,a.n_vars) or coords.shape!=(a.n_obs,3): raise ValueError('unexpected matrix or coordinate shape')
    if not np.isfinite(x).all() or not np.isfinite(coords).all() or (x<0).any(): raise ValueError('invalid numeric values')
    np.savetxt(OUT/'expression.csv',x,delimiter=',',fmt='%.7g')
    np.savetxt(OUT/'spatial_coordinates.csv',coords,delimiter=',',fmt='%.9g')
    def lines(name,values): (OUT/name).write_text('\n'.join(map(str,values))+'\n')
    lines('cell_ids.txt',a.obs_names)
    lines('gene_ids.txt',a.var_names)
    lines('sample_ids.txt',a.obs['batch'].astype(str))
    lines('evaluation_cell_groups.txt',a.obs['clusters'].astype(str))
    obs=a.obs.copy();obs.rename(columns={'cell_id':'cell_id_original'},inplace=True);obs.insert(0,'cell_id',a.obs_names);obs.to_parquet(OUT/'cell_metadata.parquet',index=False)
    nn=cKDTree(coords).query(coords,k=2,workers=-1)[0][:,1]
    summary={'dataset':'weMERFISH_6s_E1_3D','source_archive':str(ARCHIVE),'archive_member':MEMBER,
      'cells':a.n_obs,'genes':a.n_vars,'expression_nonzero':int(np.count_nonzero(x)),
      'coordinate_source':'obsm[spatial]','coordinate_dimensions':3,
      'coordinate_ranges':{'x':[float(coords[:,0].min()),float(coords[:,0].max())],
      'y':[float(coords[:,1].min()),float(coords[:,1].max())],'z':[float(coords[:,2].min()),float(coords[:,2].max())]},
      'median_nearest_neighbor_distance':float(np.median(nn)),'samples':a.obs['batch'].value_counts().to_dict(),
      'annotation_levels':{'clusters':int(a.obs.clusters.nunique()),'germlayer':int(a.obs.germlayer.nunique()),'type':int(a.obs.type.nunique()),'tissue':int(a.obs.tissue.nunique())}}
    (OUT/'preprocess_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
