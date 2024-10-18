import os, sys, pickle, os.path as osp
import persist_to_disk as ptd
ptd.config.set_project_path(osp.abspath("."))

sys.path.append(os.getcwd())
from _settings import GEN_PATHS
import dataeval


for dataset in list(GEN_PATHS):
    print(dataset)
    rg = dataeval.load.read_cleaned_outputs_new(GEN_PATHS[dataset]['gpt-3.5-turbo'])
    for d in rg:
        del d['generations']
    
    if dataset == 'coqa':
        raw = datasets.load_from_disk(dataeval.coqa._save_dataset())
        for d, d1 in zip(rg, raw):
            assert d['id'] == d1['id'] and 'story' not in d1
            d['story'] = d1['story']

    with open(f'./data/{dataset}.pkl', 'wb') as fout:
        pickle.dump(rg, fout)