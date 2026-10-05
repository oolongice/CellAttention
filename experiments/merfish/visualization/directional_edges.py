from functools import lru_cache
import pandas as pd
from merfish_common import DATASET, HERE

@lru_cache(maxsize=1)
def load_edges():
    table = pd.read_csv(HERE/'data/directional_edges.csv')
    ids = (DATASET/'model/transformer/cell_ids.txt').read_text().splitlines()
    index = {v:i for i,v in enumerate(ids)}
    table['sender_index'] = table.sender_cell_id.map(index)
    table['receiver_index'] = table.receiver_cell_id.map(index)
    assert not table[['sender_index','receiver_index']].isna().any().any()
    return table

def pair_edges(table, cluster, source, target):
    edges = table[(table.cluster==cluster)&(table.source_gene==source)&(table.target_gene==target)]
    if edges.empty:
        raise ValueError(f'No retained edges: {cluster} {source} {target}')
    return edges
