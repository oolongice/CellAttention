#!/usr/bin/env python3
from pathlib import Path
import os,json,itertools,csv
R=Path(os.environ['CELLATTENTION_BENCHMARK_WORKDIR']); TAG=os.environ['HIER_VARIANT']; D=R/'data'/TAG; O=R/'results'/TAG/'misty_no_gene_role_prior'; O.mkdir(parents=True,exist_ok=True)
genes=(D/'gene_ids.txt').read_text().splitlines(); pairs=O/'all_gene_pairs.csv'
with pairs.open('w',newline='') as h:
 w=csv.writer(h); w.writerow(['source_gene','target_gene']); w.writerows(itertools.product(genes,genes))
m={'dataset':f'hierarchical_{TAG}_all_genes','expression_csv':str(D/'expression.csv'),'coordinates_csv':str(D/'spatial_coordinates.csv'),'cell_ids_txt':str(D/'cell_ids.txt'),'gene_ids_txt':str(D/'gene_ids.txt'),'cell_groups_txt':str(D/'inferred_groups.txt'),'candidate_pairs_csv':str(pairs)}
(O/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
