#!/usr/bin/env python3
from pathlib import Path
import json, os, subprocess
ROOT=Path(os.environ["CELLATTENTION_BENCHMARK_WORKDIR"]); TAG=os.environ.get('HIER_VARIANT','seed_168'); DATA=ROOT/'data'/TAG; TRAINER=Path(os.environ['CELLATTENTION_BINARY'])
for seed in range(int(os.environ.get('CELLATTENTION_MODEL_SEED_START', '3001')), int(os.environ.get('CELLATTENTION_MODEL_SEED_END', '3005')) + 1):
    out=ROOT/'models'/TAG/f'seed_{seed}'; out.mkdir(parents=True,exist_ok=True)
    cfg=ROOT/'configs'/TAG/f'seed_{seed}.json'; cfg.parent.mkdir(parents=True,exist_ok=True)
    spec={'expression_matrix':str(DATA/'expression.csv'),'cell_ids':str(DATA/'cell_ids.txt'),'gene_ids':str(DATA/'gene_ids.txt'),'sample_ids':None,'output_dir':str(out),'device_index':0,
          'model':{'d_model':48,'d_ff':96,'n_heads':4,'n_layers':2,'dropout':.05},
          'training':{'epochs':int(os.environ.get('CELLATTENTION_EPOCHS', '110')),'batch_size':64,'learning_rate':.001,'mask_probability':.25,'seed':seed,'shuffle':True,'max_cells_per_epoch':int(os.environ.get('CELLATTENTION_MAX_CELLS_PER_EPOCH', str(min(600,len((DATA/'cell_ids.txt').read_text().splitlines()))))),'epoch_sampling':'stratified','minimum_cells_per_sample':0}}
    if 'CELLATTENTION_INFERENCE_CHUNK_SIZE' in os.environ:
        spec['training']['inference_mask_chunk_size'] = int(os.environ['CELLATTENTION_INFERENCE_CHUNK_SIZE'])
    if (out/'model.mpk').exists():
        if not cfg.is_file() or json.loads(cfg.read_text()) != spec:
            raise ValueError(f'existing checkpoint has a different or unknown config: {out}')
        continue
    cfg.write_text(json.dumps(spec,indent=2)+'\n')
    env=os.environ.copy(); env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',RAYON_NUM_THREADS='8')
    with (out/'training.log').open('w') as h: subprocess.run([str(TRAINER),str(cfg)],env=env,stdout=h,stderr=subprocess.STDOUT,check=True)
