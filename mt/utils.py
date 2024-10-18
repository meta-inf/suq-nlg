import os, os.path as osp
import numpy as onp
from typing import List, Tuple


def load_data(base_path, split, lang):
    with open(os.path.join(base_path, split, f'{split}.{lang}')) as fin:
        return [l.rstrip() for l in fin.readlines()]


class ICLkNNRetriever:

    def __init__(self, Dsrc, Dsrc_retr, Ddest_retr, emb_dict, rng: onp.random.Generator, 
                 shuffle=False):
        self.Dsrc_retr, self.Ddest_retr, self.emb_dict = Dsrc_retr, Ddest_retr, emb_dict
        self.eval_emb_dict = {src: emb for src, emb in zip(Dsrc, self.emb_dict['eval'])}
        self.len_src_retr = onp.asarray([s.count(' ') for s in Dsrc_retr])
        self.rng, self.shuffle = rng, shuffle

    def retrieve(self, src, n_samples) -> List[Tuple[str, str]]:
        src_emb = self.eval_emb_dict[src]
        sqdist = ((self.emb_dict['retr'] - src_emb[None])**2).sum(axis=-1)
        idcs = onp.argsort(sqdist)
        len_src = src.count(' ')
        length_mask = onp.logical_and(
            self.len_src_retr >= len_src//2, self.len_src_retr <= len_src*2)
        ret_idcs = ([i for i in idcs if length_mask[i]] 
                    + [i for i in idcs if not length_mask[i]])[:n_samples]
        ret = [(self.Dsrc_retr[i], self.Ddest_retr[i]) for i in ret_idcs]
        if self.shuffle:
            self.rng.shuffle(ret)
        return list(ret)
    

class ICLRandRetriever:

    def __init__(self, Dsrc, Dsrc_retr, Ddest_retr, emb_dict, rng: onp.random.Generator):
        self.Dsrc_retr, self.Ddest_retr = Dsrc_retr, Ddest_retr
        self.rng = rng

    def retrieve(self, src, n_samples) -> List[Tuple[str, str]]:
        idcs = self.rng.choice(len(self.Dsrc_retr), n_samples, replace=False)
        return [(self.Dsrc_retr[i], self.Ddest_retr[i]) for i in idcs]


class MixedRetriever:

    def __init__(self, Dsrc, Dsrc_retr, Ddest_retr, emb_dict, rng: onp.random.Generator,
                 n_rand: int):
        self.rand = ICLRandRetriever(Dsrc, Dsrc_retr, Ddest_retr, emb_dict, rng)
        self.knn = ICLkNNRetriever(Dsrc, Dsrc_retr, Ddest_retr, emb_dict, rng, shuffle=True)
        self.n_rand = n_rand

    def retrieve(self, src, n_samples):
        rand_samples = self.rand.retrieve(src, self.n_rand)
        knn_samples = self.knn.retrieve(src, n_samples)
        knn_samples = [stup for stup in knn_samples if stup not in rand_samples]
        return rand_samples + knn_samples[:n_samples-len(rand_samples)]


if __name__ == '__main__':
    Dsrc = load_data(osp.expanduser('~/data/flores/floresp-v2.0-rc.3'), 'devtest', 'eng_Latn')
    Dsrc_retr = load_data(osp.expanduser('~/data/flores/floresp-v2.0-rc.3'), 'dev', 'eng_Latn')
    Ddest_retr = load_data(osp.expanduser('~/data/flores/floresp-v2.0-rc.3'), 'dev', 'yue_Hant')
    emb_dict = onp.load(osp.expanduser('~/run/manual/guq/emb-retrieval.npz'))
    emb_dict = {
        'eval': emb_dict['emb_devtest'],
        'retr': emb_dict['emb_dev']
    }
    
    # retr = ICLkNNRetriever(Dsrc, Dsrc_retr, Ddest_retr, emb_dict, onp.random.default_rng())
    # src, retr = Dsrc[10], retr.retrieve(Dsrc[10], 10)
    # print(src)
    # for s, d in retr:
    #     print('\t', s, '\n\t->', d)
        
    retr = MixedRetriever(Dsrc, Dsrc_retr, Ddest_retr, emb_dict, onp.random.default_rng(), 2)
    src, retr = Dsrc[10], retr.retrieve(Dsrc[10], 6)
    print(src)
    for s, d in retr:
        print('\t', s, '\n\t->', d)