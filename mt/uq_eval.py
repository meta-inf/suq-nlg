import pickle, os, sys
from functools import cache
from typing import List, Dict, Tuple, Any

import tqdm
from jaxtyping import Float, Array
from matplotlib import pyplot as plt
from sacrebleu.metrics import CHRF

from mt.utils import *


@cache
def sentence_chrf(ref, hypothesis):
    return CHRF(word_order=2).sentence_score(hypothesis, [ref]).score / 100


def get_max_utility_and_action(
        posterior_samples: List[str],
        ref_set: List[str]) -> Tuple[float, str]:
    ret = (-1e10, None)
    for r in ref_set:  # action
        utility_r = []
        for pr in posterior_samples:
            utility_r.append(sentence_chrf(pr, r))
        ret = max(ret, (onp.mean(utility_r), r))
    return ret


def gibbs_utility(posterior_samples):
    ret = []
    for i, a in enumerate(posterior_samples):
        for j, b in enumerate(posterior_samples):
            if i!=j:
                ret.append(sentence_chrf(a, b))
    return onp.mean(ret)


def _fix_sId(sample_dicts):
    if len(sample_dicts) == 0:
        return []
    cId = sample_dicts[0]['id'][0]
    assert all(
        s['actual'] == sample_dicts[0]['actual'] and s['id'][0] == cId
        for s in sample_dicts[1:])
    sd_sorted = list(sorted([
        (s['id'][1], hash(s['pred']), s) for s in sample_dicts]))
    ret = []
    for j, stup in enumerate(sd_sorted):
        ret.append(stup[-1] | {'id': (cId, j)})
    return ret


def _group_and_fix_sample_dicts(sample_dicts):
    # group samples by completion chains
    sd_by_cId = {}
    for sd in sample_dicts:
        cid = sd['id'][0]
        if cid not in sd_by_cId:
            sd_by_cId[cid] = []
        sd_by_cId[cid].append(sd)
        
    # fix duplicate or nonconsecutive sample_id
    lst = []
    for cId, sd_cur in sd_by_cId.items():
        sd_by_cId[cId] = _fix_sId(sd_cur)
        lst.extend(sd_by_cId[cId])
    
    return sd_by_cId, lst


def uncertainty_stats(sample_dicts: List[Dict[str, Any]], mb_n_actions=1):
    """
    Compute Bayesian uncertainty stats for MBR and Gibbs prediction.
    `sample_dicts` consists of ICL prediction samples for a single test query.  Samples are
    identified with `(chain_id, sample_id)` where `chain_id` indicates the ICL demonstration
    chain it belongs to.  We use the samples with `sample_id<mb_n_actions` as the action set in MBR. 
    """
    sd_by_cId, sample_dicts = _group_and_fix_sample_dicts(sample_dicts)
    
    # "reference samples" define our action space
    ref = [sd['pred'] for sd in sample_dicts if sd['id'][1] < mb_n_actions]
    # the rest will be used to estimate the average utility
    psamples = [sd['pred'] for sd in sample_dicts if sd['id'][1] >= mb_n_actions]
    pred_u_mbr, mbr_action = get_max_utility_and_action(psamples, ref)
    pred_u_gibbs = gibbs_utility(psamples+ref)

    ic_mbr, ic_gibbs = [], []
    for _, intra_chain_dicts in sd_by_cId.items():  # insertion ordered
        cur_psamples = [sd['pred'] for sd in intra_chain_dicts if sd['id'][1] >= mb_n_actions]
        ic_mbr.append(get_max_utility_and_action(cur_psamples, ref)[0])  # MBR in intra-chain samples; ~ "irreducible" uncertainty
        ic_gibbs.append(
            gibbs_utility([sd['pred'] for sd in intra_chain_dicts]))
        
    pred_improved_mbr = onp.mean(ic_mbr)
    pred_improved_gibbs = onp.mean(ic_gibbs)
    
    # reality check
    actual_out = sample_dicts[0]['actual']
    act_u_gibbs = onp.mean([sentence_chrf(actual_out, ps) for ps in psamples+ref])
    act_u_mbr = sentence_chrf(actual_out, mbr_action)
    
    return {
        'eu_mbr': pred_improved_mbr - pred_u_mbr,
        'eu_gibbs': pred_improved_gibbs - pred_u_gibbs,
        'pred_u_mbr': pred_u_mbr,
        'pred_u_gibbs': pred_u_gibbs,
        'act_u_mbr': act_u_mbr,
        'act_u_gibbs': act_u_gibbs,
    } 


def process_trace(trace, mb_n_actions=1):
    trace_by_data = {}
    for t in trace:
        if t['src'] not in trace_by_data:
            trace_by_data[t['src']] = []
        trace_by_data[t['src']].append(t)

    eu = {}
    for test_query, sample_dicts in tqdm.tqdm(trace_by_data.items()):
        if len(sample_dicts) <= mb_n_actions+1:
            print(f"Skipping {test_query}: insufficient samples", file=sys.stderr)
            continue
        eu[test_query] = uncertainty_stats(sample_dicts, mb_n_actions=mb_n_actions)

    return eu, list(trace_by_data.items())


def load_pkl(path):
    with open(path, 'rb') as fin:
        return pickle.load(fin)


def load_trace(dir_or_dirs, type='eu', base_mb_n_actions=4):
    file_name, proc_kw = {
        'eu': ('trace.pkl', {}),
        'base': ('trace-nocomp.pkl', {'mb_n_actions': base_mb_n_actions}),
    }[type]
    if isinstance(dir_or_dirs, str):
        dir_or_dirs = [dir_or_dirs]

    trace = []
    for d in dir_or_dirs:
        trace.extend(load_pkl(os.path.join(d, file_name)))
    
    return process_trace(trace, **proc_kw)


def process_dicts(d_base, d_base_longer, d_eu):
    idcs = [k for k in d_base if k in d_base_longer and k in d_eu]
    if len(idcs) < len(d_base):
        print("dropping", len(d_base) - len(idcs), "samples", file=sys.stderr)
    def proc_eud(db, d):
        ret = db.copy()
        for suff in ['mbr', 'gibbs']:
            ret[f'eu_{suff}'] = d[f'eu_{suff}'] + d[f'pred_u_{suff}'] - db[f'pred_u_{suff}']
        return ret
    d_bn = [proc_eud(d_base[k], d_eu[k]) for k in idcs]
    d_base_longer = [d_base_longer[k] for k in idcs]
    return d_bn, d_base_longer
    

def EK(ret, k): return onp.asarray([s[k] for s in ret])


def cmpplot(a, b, newfig=True, createfig=True, **kwargs):
    print(a.mean(), b.mean())
    if newfig and createfig:
        plt.figure(figsize=(4, 4))
    plt.scatter(a, b, marker='+', **kwargs)
    if newfig:
        vmax = max(1, onp.max(a), onp.max(b))
        vmin = min(0, onp.min(a), onp.min(b))
        plt.xlim(vmin, vmax)
        plt.ylim(vmin, vmax)
        plt.plot([vmin, vmax], [vmin, vmax], linestyle=':', color='gray')


def improvement_curve(base_stats, improved_stats, order_fn, tgt='act_u_mbr'):
    order = order_fn(base_stats)  # [n] permutation
    base_utility = [base_stats[o][tgt] for o in order]
    improved_utility = [improved_stats[o][tgt] for o in order]
    avg_utility = []
    total_utility, n = sum(base_utility), len(base_utility)
    for base_u, improved_u in zip(base_utility, improved_utility):
        avg_utility.append(total_utility/n)
        total_utility += improved_u - base_u
    return onp.asarray(avg_utility)


def gen_random_order(stats, rng):
    return rng.permutation(len(stats))


def order_by_tu(stats):
    return onp.argsort([s['pred_u_mbr'] for s in stats])  # lowest expected utility first


def order_by_eu(stats):
    return onp.argsort([-s['eu_mbr'] for s in stats])  # highest EU first


def plot_arr(a: Float[Array, "batch_size length"],
             mode='ms', qlo=0.25, sd_scale=1., ax=None, shade_kw={}, **kw):
    shade_kw = {'alpha': 0.1} | shade_kw
    x = onp.arange(a.shape[1])
    if ax is None:
        ax = plt.gca()
    if mode == 'q':
        ax.plot(x, onp.median(a, axis=0), **kw)
        ax.fill_between(
            x, 
            onp.quantile(a, qlo, axis=0), 
            onp.quantile(a, 1-qlo, axis=0), 
            **shade_kw)
    else:
        ax.plot(x, onp.mean(a, axis=0), **kw)
        ax.fill_between(
            x,
            onp.mean(a, axis=0)-onp.std(a, axis=0)*sd_scale, 
            onp.mean(a, axis=0)+onp.std(a, axis=0)*sd_scale,
            **shade_kw)


def plot_improvement(
        base_stats, improved_stats, target_key='act_u_mbr', sig=0.05, n_bs=10000, bs_seed=23,
        ax=None, legend=True, short=False):
    """
    :return: P_{rand seq}(AUC_{EU} > AUC_{with rand seq})
    """
    if ax is None:
        ax = plt.gca()

    rng = onp.random.default_rng(bs_seed)
    rand_curves = onp.asarray([
        improvement_curve(
            base_stats, improved_stats, lambda s: gen_random_order(s, rng), tgt=target_key)
        for _ in range(n_bs)])
    x = onp.arange(len(base_stats))
    tu_curve = improvement_curve(
        base_stats, improved_stats, order_by_tu, tgt=target_key)
    eu_curve = improvement_curve(
        base_stats, improved_stats, order_by_eu, tgt=target_key)

    rand_label = 'random (exp. utl.)' if short else 'random (expected utility)'

    plot_arr(100*rand_curves, mode='q', qlo=sig/2, ax=ax, 
             shade_kw={'alpha': 0}, linestyle=':', color='gray', label=rand_label)
    ax.plot(x, 100*eu_curve, label='EU')
    ax.plot(x, 100*tu_curve, label='TU', linestyle=(0, (3, 3)))

    sAvg = onp.mean(rand_curves, axis=1)
    sObs = onp.mean(eu_curve)
    p_avg = (sObs < sAvg).mean()
    ax.set_title(f'p={p_avg:.3f}')
    if legend:
        ax.legend()
    return p_avg


def compute_improvement(
        base_stats, improved_stats, rng, target_key='act_u_mbr', n_bs=192):

    rand_curves = onp.asarray([
        improvement_curve(
            base_stats, improved_stats, lambda s: gen_random_order(s, rng), tgt=target_key)
        for _ in range(n_bs)])
    eu_curve = improvement_curve(base_stats, improved_stats, order_by_eu, tgt=target_key)
    tu_curve = improvement_curve(base_stats, improved_stats, order_by_tu, tgt=target_key)

    sAvg = onp.mean(rand_curves, axis=1)
    sObs = onp.mean(eu_curve)
    return (
        (sObs <= sAvg).mean(),
        rand_curves[0],
        eu_curve-rand_curves[0][0],
        tu_curve-rand_curves[0][0])


def bootstrap(dNew, dMore, target_key='act_u_mbr', n_bs=500):
    rng = onp.random.default_rng(1)

    def take(arr, idcs):
        return [arr[i] for i in idcs]

    pvals = []
    sA, sO, sOT = [], [], []
    for _ in range(n_bs):
        idcs = rng.choice(len(dNew), size=len(dNew))
        pval, sAvg, sObs, sObsT = compute_improvement(
            take(dNew, idcs), take(dMore, idcs), rng, target_key=target_key,
            n_bs=3)
        pvals.append(pval)
        sA.append(sAvg)
        sO.append(sObs)
        sOT.append(sObsT)
    sA, sO, sOT = map(onp.asarray, (sA, sO, sOT))
    print(onp.mean(pvals), onp.mean(sO), onp.mean(sOT))
    return sO, sOT
