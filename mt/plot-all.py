#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pprint, pandas as pd, seaborn as sns
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

sys.path.append(os.getcwd())

from mt.uq_eval import *
from utils.eval_tu import eval_ece


# In[2]:


MODELS = ['gemini-1.5-pro', 'gpt-4o-mini', 'gpt-4o']
LANG_NAMES = {
    'tam': 'Tamil', 
    'kmr': 'Kurdish' 
}

base_path_t = './run/mt/BASE-{model}-{lang}-gt4'
longer_path_t = './run/mt/BASE-{model}-{lang}-gt128'
eu_path_t = './run/mt/{model}-{lang}-N500-comp4-gt4'

os.makedirs('/tmp/uq-figs/mt/', exist_ok=True)

def savefig(file):
    plt.savefig(f'/tmp/uq-figs/mt/{file}')


# In[3]:


dNew = dMore = None

main_tbl_df = []
main_results: Dict[Tuple[str, str], Tuple[list[dict], list[dict]]] = {}

for model in MODELS:
    for lang in LANG_NAMES:
        print(model, lang)
        
        pBase, pLonger = base_path_t.format(model=model, lang=lang), longer_path_t.format(model=model, lang=lang)
        dBase, _ = load_trace(pBase, 'base', base_mb_n_actions=8)
        dLonger, _ = load_trace(pLonger, 'base', base_mb_n_actions=8)
        dEu, _ = load_trace(eu_path_t.format(model=model, lang=lang), 'eu')
        lBase, lLonger = process_dicts(dBase, dLonger, dEu)
                
        for target_suffix in ['mbr', 'gibbs']:
            plt.figure(figsize=(2.8, 2.4))
            plot_improvement(lBase, lLonger, f'act_u_{target_suffix}', short=True)
            savefig(f'full-{model}_{lang}_mbr_{target_suffix}.pdf')
            
            if target_suffix == 'mbr':
                main_results[(model, lang)] = (lBase, lLonger)
                sO, sOT = bootstrap(lBase, lLonger, n_bs=1000, target_key=f'act_u_{target_suffix}')
                for i_bs in range(sO.shape[0]):
                    main_tbl_df.append({
                        "model": model,
                        "language": lang,
                        "diff": (sO[i_bs]-sOT[i_bs]).mean()
                    })

main_tbl_df = pd.DataFrame(main_tbl_df)


# In[4]:


fig = plt.figure(figsize=(11, 2.3), facecolor='w')
for i, ((model, lang), (lBase, lLonger)) in enumerate(
    list(main_results.items())[2:]):
    plt.subplot(1, 4, i+1)
    pval = plot_improvement(
        lBase, lLonger, f'act_u_mbr', legend=False, n_bs=10000
    )
    if i==0:
        ax = plt.gca()
        plt.ylabel('chrF / %')
    plt.title(f'{model}, {lang}, p={pval:.3f}')
    
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=10)

# Adjust the layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.9])
savefig('main-combined.pdf')


# In[5]:


print(r'\begin{tabular}{cccc} \toprule')
print(' & '.join(MODELS), r'\\ \midrule')
for lang in LANG_NAMES:
    print(lang, end=' ')
    for model in MODELS:
        d = main_tbl_df[(main_tbl_df.model == model) & (main_tbl_df.language == lang)]['diff']
        d = d.to_numpy()
        lo, mid, hi = onp.quantile(d*100, [.25, .5, .75])
        print(f'\n\t & $ {mid:.2f} $', r'{\small ' + f'$[{lo:.2f}, {hi:.2f}]$' + r'}', end='')
    print(r' \\')
print(r'\bottomrule \end{tabular}')


# In[6]:


eces = {}
ustats = {}

for model in MODELS:
    for lang in LANG_NAMES:
        print(model, lang)
        pBase, pLonger = base_path_t.format(model=model, lang=lang), longer_path_t.format(model=model, lang=lang)
        dBase, _ = load_trace(pBase, 'base', base_mb_n_actions=8)
        dLonger, _ = load_trace(pLonger, 'base', base_mb_n_actions=8)
        dEu, _ = load_trace(eu_path_t.format(model=model, lang=lang), 'eu')
        lBase, lLonger = process_dicts(dBase, dLonger, dEu)        
        eces[(model, lang)] = eval_ece(
            EK(lBase, 'pred_u_mbr'), EK(lBase, 'act_u_mbr'))

pprint.pp(eces)
