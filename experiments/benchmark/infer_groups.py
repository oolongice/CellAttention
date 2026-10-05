#!/usr/bin/env python3
from pathlib import Path
import argparse,json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
p=argparse.ArgumentParser(); p.add_argument('data',type=Path); a=p.parse_args()
x=np.loadtxt(a.data/'expression.csv',delimiter=',')
z=StandardScaler().fit_transform(np.log1p(x))
z=PCA(20,random_state=9001).fit_transform(z)
labels=KMeans(6,n_init=30,random_state=9001).fit_predict(z)
(a.data/'inferred_groups.txt').write_text('\n'.join(map(str,labels))+'\n')
(a.data/'inferred_groups_metadata.json').write_text(json.dumps({'method':'KMeans','input':'log1p expression only','standardization':'per gene z-score','pca_components':20,'n_clusters':6,'random_state':9001,'truth_labels_used':False},indent=2)+'\n')
