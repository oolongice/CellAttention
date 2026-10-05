#!/usr/bin/env python3
from pathlib import Path
import json,os
R=Path(os.environ['CELLATTENTION_BENCHMARK_WORKDIR']); TAG=os.environ.get('HIER_VARIANT','seed_168'); D=R/'data'/TAG; O=R/'results'/TAG/'misty_receiver'; O.mkdir(parents=True,exist_ok=True)
groups='inferred_groups.txt' if os.environ.get('HIER_GROUPS','oracle')=='inferred' else 'cell_groups.txt'
m={'dataset':f'hierarchical_{TAG}','expression_csv':str(D/'expression.csv'),'coordinates_csv':str(D/'spatial_coordinates.csv'),'cell_ids_txt':str(D/'cell_ids.txt'),'gene_ids_txt':str(D/'gene_ids.txt'),'cell_groups_txt':str(D/groups),'candidate_pairs_csv':str(D/'candidate_source_target.csv')}
(O/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
