#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, sys, pickle
import numpy as np

osp = os.path
from glob import glob
import functools
from typing import List, Tuple

import matplotlib as mpl, pandas as pd, seaborn as sns
from matplotlib import pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

mpl.rcParams['font.size'] = 12


# In[2]:


sys.path.append(os.getcwd())


# In[3]:


from utils.eval_tu import *


# In[4]:


# In some experiments we generated more samples. For consistency we truncate them.
# Note that the first `TRUNC_N` queries are comparable across experiments.
TRUNC_N = 1000
TRUNC_K = 10

ROOT_PATH = './run/qa/'

MODELS = [
    "llama3.1-8b", "llama3.1-70b",
    "gemini-1.5-flash", "gemini-1.5-pro", 
    "gpt-4o-mini", "gpt-4o", 
    "claude-3.5", 
]

ALL_DSETS = ['nq_open', 'trivia', 'coqa', 'sciq', 'truthful_qa']
TRIVIA_DSETS = ['nq_open', 'trivia']

data_name_map = {
    # filtered trivia datasets
    'trivia*': 'TriviaQA*',
    'nq_open*': 'NQOpen*',
    # unfiltered
    'trivia': 'TriviaQA',
    'nq_open': 'NQOpen',
    # other datasets
    'sciq': 'SciQ',
    'truthful_qa': 'TruthfulQA',
    'coqa': 'CoQA',
}

os.makedirs('/tmp/uq-figs', exist_ok=True)
def savefig(name):
    plt.savefig(f'/tmp/uq-figs/{name}')


# In[5]:


@functools.lru_cache(maxsize=200)
def load(path):
    with open(path, 'rb') as fin:
        recs = pickle.load(fin)[:TRUNC_N]        
    
    pred, actual = [], []
    for _, gen, srmat, true_rmat in recs:
        idcs = np.asarray([i for i in range(len(gen)) if gen[i].strip()])[:TRUNC_K]
        if len(idcs) == 0:
            pred.append(np.nan); actual.append(np.nan)
            continue
        pred.append(off_diagonal_mean(srmat[idcs][:, idcs]))
        actual.append(safe_mean(true_rmat[idcs, 0]))
        
    pred, actual = map(np.asarray, (pred, actual))
    vcs = []
    vts = []
    
    for rtup in recs:
        row = rtup[0]
        if 'verbalized_confidence' in row:
            gen_cleaned = map(str.lstrip, row['generations']['text_cleaned'])
            vc = [row['verbalized_confidence'][gen] for gen in gen_cleaned if gen.strip()]
            vcs.append(safe_mean(vc))
        if 'vc_tian' in row:
            gen_cleaned = map(str.strip, row['generations']['text_cleaned'])
            vc = [row['vc_tian'][gen] for gen in gen_cleaned if gen.strip()]
            vts.append(safe_mean(vc))
            
    vcs, vts = map(np.asarray, (vcs, vts))
        
    return (pred, actual, vcs, vts), recs

def align_arrays(
    mandatory: Tuple[np.ndarray], optional: Tuple[np.ndarray]
) -> Tuple[Tuple[np.ndarray], Tuple[np.ndarray], List[int]]:
    len_orig = len(mandatory[0])
    assert all(len(a)==len_orig for a in mandatory[1:])
    
    len_trunc = min(
        [len_orig] + [len(oa) for oa in optional if len(oa)>0]
    )
    
    nan_masks = [
        np.isnan(a[:len_trunc]) for a in mandatory+optional if len(a)>0
    ]
    valid_mask = ~ functools.reduce(np.logical_or, nan_masks)
    if valid_mask.sum() < len_orig:
        sys.stderr.write(f"Removing {len_orig-valid_mask.sum()}/{len_orig} samples\n")
    
    new_mand = tuple(a[:len_trunc][valid_mask] for a in mandatory)
    new_opti = tuple(
        a[:len_trunc][valid_mask] if len(a)>0 else a
        for a in optional)
    
    return new_mand, new_opti, list(np.where(valid_mask)[0])
    

def mask_arrays(arrs, mask):
    return tuple(
        a[mask] if len(a)>0 else a for a in arrs
    )

def eval_bs(pred, actual, n_bs=1000, seed=42, **kwargs):
    rng = np.random.default_rng(seed)
    eces, arcs = [], []
    n = pred.shape[0]
    for _ in range(n_bs):
        idcs = rng.choice(n, size=n)
        eces.append(eval_ece(pred[idcs], actual[idcs], **kwargs))
        # For ARC the first argument is uncertainty = -expected utility
        arcs.append(area_under_accuracy_coverage_curve(-pred[idcs], actual[idcs]))
    return eval_ece(pred, actual), eces, arcs

def bs_mean_est(arr, n_bs=1000, seed=42, return_all=False):
    n = arr.shape[0]
    rng = np.random.default_rng(seed)
    ret = []
    for _ in range(n_bs):
        idcs = rng.choice(n, size=n)
        ret.append(arr[idcs].mean())
    if return_all:
        return ret
    return arr.mean(), np.quantile(ret, [.025, .975])

def majority_vote(lst):
    return sum(lst) / len(lst) > 0.5


# In[6]:


# mask for outdated queries
trivia_masks = {}
for dset in TRIVIA_DSETS:
    with open(f"./data/mask-{dset}.pkl", 'rb') as fin:
        dct = pickle.load(fin)
        recs = dct['recs']
        trivia_masks[dset] = {r['id']: majority_vote(r['ans']) for r in recs}


# In[7]:


edf_full = []
eces = {}
for dset in ALL_DSETS:
    
    for model in MODELS:
        print(dset, model)
        paths = glob(osp.join(ROOT_PATH, f"{dset}*{model}/eval*gpt-4o-mini*/map__parsed.pkl"))
        if len(paths) != 1:
            print("skipping", (dset, model, paths), file=sys.stderr)
            continue
        
        path = paths[0]
        (pred, actual, _, _), recs = load(path)
        
        (pred, actual), _, idcs = align_arrays((pred, actual), tuple())
        recs = [recs[i] for i in idcs]
        
        if dset in TRIVIA_DSETS:
            mask_outdated = [False, True]
        else:
            mask_outdated = [False]

        for do_mask in mask_outdated:
            if do_mask:
                masks = trivia_masks[dset]
                cur_mask = np.logical_not(np.asarray([masks[tup[0]['id']] for tup in recs]))
                pred, actual = pred[cur_mask], actual[cur_mask]
                dset_viz = dset + '*'
            else:
                dset_viz = dset

            ece_est, ece_tr, arc_tr = eval_bs(pred, actual, plot_title=None)
            cur_accs = bs_mean_est(actual, return_all=True)

            eces[(dset_viz, model)] = (
                (ece_est, np.quantile(ece_tr, [.025, .975])), 
                (actual.mean(), np.quantile(cur_accs, [.025, .975]))
            )
            for e, a, ac in zip(ece_tr, arc_tr, cur_accs):
                edf_full.append({
                    "dataset": dset_viz, "model": model, "ECE": e,
                    "AUARC": a, "Utility": ac
                })


# In[8]:


LIN_DSETS = ['nq_open*', 'trivia*', 'coqa']
mfams = ['gpt', 'gemini', 'llama']


# In[9]:


df = pd.DataFrame(edf_full)

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 3.25), facecolor='w')

kw = {'order': LIN_DSETS, 'hue_order': MODELS, 'palette': 'colorblind', "errorbar": ("pi", 95)}

for i, metric in enumerate(['ECE', 'Utility', 'AUARC']):
    ax = axes[i]
    g = sns.barplot(df, x='dataset', y=metric, hue='model', ax=ax, legend=True, **kw)
    ax.get_legend().set_visible(False)
    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.set_title(metric + ' ' + '↑↓'[int(i==0)])
    if i>0:
        ax.set_ylim(0.4, None)
        
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=7, fontsize=11)

# Adjust the layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.9])
savefig('lin-all.pdf')


# In[10]:


edf = []
for dset in ['nq_open', 'trivia', 'coqa', 'truthful_qa', 'sciq']:
    
    for model in ['llama3.1-70b', 'llama3.1-70b-pret']:
        print(dset, model)
        paths = glob(osp.join(ROOT_PATH, f"{dset}*{model}/eval*gpt-4o-mini*/map__parsed.pkl"))
        if len(paths) != 1:
            print("skipping", (dset, model, paths), file=sys.stderr)
            continue
        
        path = paths[0]
        (pred, actual, _, _), recs = load(path)
        
        (pred, actual), _, idcs = align_arrays((pred, actual), tuple())
        recs = [recs[i] for i in idcs]
        
        if dset in TRIVIA_DSETS:
            mask_outdated = [False, True]
        else:
            mask_outdated = [False]
        
        for do_mask in mask_outdated:
            if do_mask:
                masks = trivia_masks[dset]
                cur_mask = np.logical_not(np.asarray([masks[tup[0]['id']] for tup in recs]))
                pred, actual = pred[cur_mask], actual[cur_mask]
                dset_viz = dset + "*"
            else:
                dset_viz = dset

            _, cur_eces, cur_aucs = eval_bs(pred, actual)
            cur_accs = bs_mean_est(actual, return_all=True)

            for e, a, ac in zip(cur_eces, cur_aucs, cur_accs):
                edf.append({
                    "model": model, "dataset": dset_viz, "ECE": e, "AUC": a, 'Utility': ac
                })

edf = pd.DataFrame(edf)


# In[11]:


def select(df, model):
    df1 = df[(df.model == model) | (df.model == model+'-pret')].copy()
    df1['model'] = df1.model.map(lambda s: ('pre-RLHF' if s.endswith('pret') else 'post-RLHF'))
    df1['1-AUC'] = 1-df1.AUC
    df1['1-u'] = 1-df1.Utility
    df1['dataset'] = df1.dataset.map(data_name_map.__getitem__)
    return df1

for model in ['llama3.1-70b']:
    for filtering in [True, False]:
        dsets = {
            True: ['trivia*', 'sciq', 'truthful_qa', 'nq_open*', 'coqa'],
            False: ['trivia', 'sciq', 'truthful_qa', 'nq_open', 'coqa']
        }[filtering]
        kw = {
            'hue_order': ['pre-RLHF', 'post-RLHF'], 'palette': 'colorblind', "errorbar": ("pi", 95),
            'order': [data_name_map[k] for k in dsets]
        }
        df = select(edf, model)
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(7.5, 2.35), sharey=False)
        sns.barplot(df, x='dataset', y='ECE', hue='model', ax=axes[0], **kw)
        handles, labels = axes[0].get_legend_handles_labels()
        axes[0].legend(handles=handles, labels=labels, fontsize=10)
        axes[0].set_ylabel(None)
        axes[0].tick_params(axis='x', labelsize=8.5, rotation=18)
        axes[0].tick_params(axis='y', labelsize=9)
        axes[0].set_xlabel(None)
        axes[0].set_title('ECE (↓)', size=11)
        axes[0].set_ylim(0, 0.65)
        ax1 = axes[1]

        for i, (y, name) in enumerate([
            ('1-u', '1 - Utility (↓)'),
            ('1-AUC', '1 - AUARC (↓)')
        ]):
            ax = axes[i+1]
            sns.barplot(df, x='dataset', y=y, hue='model', legend=False, ax=ax, **kw)
            ax.set_title(name, size=11)
            ax.tick_params(axis='x', labelsize=8.5, rotation=18)
            ax.set_xlabel(None)
            ax.tick_params(axis='y', labelsize=9)
            ax.set_ylabel(None)


        plt.tight_layout()

        savefig(f'revisit-tian-full-{model}-{filtering}.pdf')


# In[12]:


CMP_MODELS = ["llama3.1-70b", "gpt-4o-mini", "gpt-4o"]

edf, adf = [], []

for dset in ALL_DSETS:
    
    for model in CMP_MODELS:
        paths = glob(osp.join(ROOT_PATH, f"{dset}*{model}/eval*gpt-4o-mini*/map__parsed.pkl"))
        if len(paths) != 1:
            print("skipping", (dset, model, paths), file=sys.stderr)
            continue
        
        path = paths[0]
        (pred, actual, vu, vt), recs = load(path)
        len_vu, len_vt = len(vu), len(vt)
        
        if len_vu == 0 and len_vt == 0:
            print("skipping", (dset, model, len(vu), len(vt)), file=sys.stderr)
            continue
        
        (pred, actual), (vu, vt), rec_idcs = align_arrays((pred, actual), (vu, vt))
        recs = [recs[i] for i in rec_idcs]
            
        if dset in TRIVIA_DSETS:
            outdated_mask = trivia_masks[dset]
            cur_mask = np.logical_not(np.asarray([outdated_mask[tup[0]['id']] for tup in recs]))
            pred, actual, vu, vt = mask_arrays((pred, actual, vu, vt), cur_mask)
                    
        plt.figure(figsize=(7.5, 2.5), facecolor='w')
        
        dset_viz = dset if dset not in TRIVIA_DSETS else dset+"*"
        
        for i, (title, arr) in enumerate({
            'P(True)': vu, 
            'Verb.': vt,
            'Prob.': pred, 
        }.items()):
            if len(arr) == 0:
                continue
            plt.subplot(1, 3, 1+i)
            eval_ece(arr, actual, plot_title=title)
            if i>0:
                plt.yticks([]); plt.ylabel(None)
            if i<2:
                plt.xlabel('avg. confidence')
                
            _, cur_eces, cur_aucs = eval_bs(arr, actual)
            for e, a in zip(cur_eces, cur_aucs):
                edf.append({"model": model, "dataset": dset_viz, "method": title, "ECE": e})
                adf.append({"model": model, "dataset": dset_viz, "method": title, "AUC": a})
                
        plt.tight_layout()
        savefig(f'rd-{dset}_{model}.pdf')


# In[13]:


for is_main, df, name in [(True, edf, 'ECE'), (False, adf, 'AUC')]:

    df = pd.DataFrame(df)

    datasets = ['trivia*', 'nq_open*', "coqa", 'sciq', 'truthful_qa']
    methods = ["P(True)", "Verb.", "Prob."]

    # Create a wide figure with 5 subplots (1 row x 5 columns)
    if is_main:
        fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(15, 3), sharey=True)
    else:
        fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(15, 4), sharey=True)

    # Iterate over each dataset and corresponding subplot axis
    for i, (ax, dataset) in enumerate(zip(axes, datasets)):
        # Subset the DataFrame for the current dataset
        df_subset = df[df['dataset'] == dataset].copy()

        # Convert 'model' and 'method' to categorical types with specified order
        df_subset['model'] = pd.Categorical(df_subset['model'], categories=CMP_MODELS, ordered=True)
        df_subset['method'] = pd.Categorical(df_subset['method'], categories=methods, ordered=True)

        # Sort the DataFrame to match the plotting order
        df_subset_sorted = df_subset.sort_values(['model', 'method'])

        # Create the barplot without confidence intervals
        g = sns.barplot(
            data=df_subset_sorted,
            x='model',
            y=name,
            hue='method',
            hue_order=methods,
            order=CMP_MODELS,
            ax=ax,
            errorbar=("pi", 95),
            palette='colorblind',
            legend=(i==0),
        )
        ax.set_title(data_name_map[dataset])
        ax.tick_params(axis='x', labelsize=12, rotation=15)
        g.set(xlabel=None)
    
    if name == 'AUC':
        plt.ylim(0.3, 0.99)

    plt.tight_layout()
    savefig(f"method-comparison-{name}.pdf")
