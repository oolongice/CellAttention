"""Aligned readers for common Paper_results visualization inputs."""
from pathlib import Path
import numpy as np
import pandas as pd
def read_lines(path,dtype=object):return np.asarray([x.strip() for x in Path(path).read_text().splitlines() if x.strip()],dtype=dtype)
def read_receiver_group_assignments(path,cell_ids=None,one_based=True):
    frame=pd.read_csv(path);values=frame["receiver_group"].to_numpy(dtype=int)+(1 if one_based else 0)
    if cell_ids is not None:
        if len(frame)!=len(cell_ids):raise ValueError("receiver_group assignment and cell ID row counts differ")
        if "cell_id" in frame and not np.array_equal(frame.cell_id.astype(str).to_numpy(),np.asarray(cell_ids,dtype=str)):raise ValueError("receiver_group assignments are not aligned with cell IDs")
    return values
def load_spatial_receiver_group_data(preprocessed,results,*,with_samples=False,coordinate_file="spatial_coordinates.csv",annotation_file="evaluation_cell_groups.txt"):
    preprocessed=Path(preprocessed);results=Path(results);xy=np.loadtxt(preprocessed/coordinate_file,delimiter=",");annotations=read_lines(preprocessed/annotation_file);cells=read_lines(preprocessed/"cell_ids.txt")
    receiver_groups=read_receiver_group_assignments(results/"receiver_group_assignments.csv",cells);arrays=[xy,annotations,receiver_groups]
    if with_samples:arrays.insert(1,read_lines(preprocessed/"sample_ids.txt"))
    if len({len(x) for x in arrays})!=1:raise ValueError("spatial visualization inputs have unequal row counts")
    return tuple(arrays)
