import os, sys, tqdm, pickle, json
import numpy as onp
from typing import List, Tuple

from generation import apis
import exputils
from mt.utils import *
from mt.uq_eval import sentence_chrf


def get_parser():
    parser = exputils.parser("MT")
    parser.add_argument('--mode', type=str, default='preprocess', 
                        choices=['preprocess', 'mt', 'preprocess-replace'])
    parser.add_argument('--gcp_region', type=str, default='us-east1')
    # for preprocessing only
    parser.add_argument('--dest_lang', type=str, default='kmr_Latn')
    parser.add_argument('--data_path', type=str, 
                        default=os.path.expanduser('~/data/flores/floresp-v2.0-rc.3'))
    parser.add_argument('--retrieval_split', type=str, default='dev',
                        help="split to retrieve ICL demonstrations from")
    parser.add_argument('--retrieval_method', type=str, default='emb')
    parser.add_argument('--test_split', type=str, default='devtest',
                        help="split for evaluation")
    parser.add_argument('--n_data_subsample', '-N', type=int, default=500, 
                        help='subsample size for evaluation. -1: use all')
    parser.add_argument('--n_completions', '-comp', type=int, default=0, 
                        help="number of ICL completions to define the EU bound")
    parser.add_argument('--emb_path', type=str, default='./run/mt/emb-retrieval.npz')
    # for MT and preprocessing
    parser.add_argument('--load_preproc_from', '-lp', type=str, default='')
    # 
    parser.add_argument('--n_gt_demonstrations', '-gt', type=int, default=16,
                        help="number of ground truth ICL shots")
    parser.add_argument('--n_epis_chains', '-nec', type=int, default=1)
    parser.add_argument('--n_intra_chain_samples', '-nic', type=int, default=1)
    parser.add_argument('--lm', type=str, default='gemini-1.5-flash')
    parser.add_argument('--log_every', type=int, default=1)
    parser.add_argument('--debug', action='store_true', default=False)
    return parser


def rewrite(src: str, lm: apis.LMWrapper, K: int, n_retry: int = 4) -> List[str]:
    if K == 0:
        return []

    prompt = f"""
    Please generate {K} sentences that convey a broadly similar meaning as the one provided, ensuring they are of similar length and use varied wording. Place a blank line after each sentence. Avoid adding any extra explanations.
    """.strip()
    prompt = prompt + "\n\n" + src + "\n"

    ret = []
    for _ in range((n_retry+1)//2):
        if len(ret) >= K:
            break
        resp = '\n'.join(lm.complete_multi(prompt, 2))
        for ln in resp.split('\n'):
            if ln.strip() == '':
                continue
            ret.append(ln)

    if len(ret) < K:
        sys.stderr.write(f"Failed to generate {K} completions for the prompt:\n{prompt}\n")
    return ret[:K]


def build_mt_prompt(query: str, demonstrations: List[Tuple[str, str]], dest_lang_name: str):
    # base prompt taken from Agarwal et al (2024, Fig A.18)
    BASE_PROMPT = f"""
You are an expert translator. I am going to give you one or more example pairs of text snippets where the first is in English and the second is a translation of the first snippet into {dest_lang_name}. The sentences will be written
English: <first sentence>
{dest_lang_name}: <translated first sentence>
After the example pairs, I am going to provide another sentence in English and I want you to translate it into {dest_lang_name}. Give only the translation, and no extra commentary, formatting, or chattiness. Translate the text from English to {dest_lang_name}.
    """.strip()

    prompt = BASE_PROMPT
    for src, dst in demonstrations:
        prompt = prompt + f"\n\nEnglish: {src}\n{dest_lang_name}: {dst}"

    prompt = prompt + "\n\nEnglish: " + query + f"\n{dest_lang_name}: "
    return prompt


def get_lang_name(lId):
    """ get full language names to use in prompts """
    lang_map = {}
    with open('./static/mt-lang-name.txt') as fin:
        for ln in fin.readlines():
            if ln.strip() == '': continue
            cId, cName = ln.split(': ')
            lang_map[cId] = cName.strip()
    return lang_map[lId]


def preprocess_replace_destlang(args):
    """ 
    replace target language related keys (actual, gt_demo[...][1]) in preprocessed data;
    retain `extra_inp_demos` etc. so that results are comparable across languages
    """
    Dsrc  = load_data(args.data_path, args.test_split, 'eng_Latn')
    Ddest = load_data(args.data_path, args.test_split, args.dest_lang)
    Dsrc_retr  = load_data(args.data_path, args.retrieval_split, 'eng_Latn')
    Ddest_retr = load_data(args.data_path, args.retrieval_split, args.dest_lang)

    assert len(Dsrc) == len(Ddest)
    with open(os.path.join(args.load_preproc_from, "hps.txt"), 'r') as fin:
        hps_preproc = json.load(fin)
        skipped = ['dir', 'dest_lang', 'load_preproc_from', 'mode']
        for k, v in vars(args).items():
            if k not in skipped:
                assert hps_preproc[k] == v, (k, hps_preproc[k], v)

    with open(os.path.join(args.load_preproc_from, "preproc.pkl"), 'rb') as fin:
        preprocessed = pickle.load(fin)

    src_dest_map = {k: v for k, v in zip(Dsrc, Ddest)} | {k: v for k, v in zip(Dsrc_retr, Ddest_retr)}

    for item in preprocessed:
        item['actual'] = src_dest_map[item['inp']]
        for i, demo in enumerate(item['gt_demo']):
            item['gt_demo'][i] = (demo[0], src_dest_map[demo[0]])
    
    dp = os.path.join(args.dir, "preproc.pkl")
    with open(dp, 'wb') as fout:
        pickle.dump(preprocessed, fout)
    print(f"Preprocessed data saved to {dp}")


def preprocess(args):
    # for evaluation
    Dsrc  = load_data(args.data_path, args.test_split, 'eng_Latn')
    Ddest = load_data(args.data_path, args.test_split, args.dest_lang)
    # for ICL demonstration retrieval
    Dsrc_retr  = load_data(args.data_path, args.retrieval_split, 'eng_Latn')
    Ddest_retr = load_data(args.data_path, args.retrieval_split, args.dest_lang)
    assert len(Dsrc) == len(Ddest) and len(Dsrc_retr) == len(Ddest_retr)
    if args.n_epis_chains > 1:
        assert args.n_intra_chain_samples > 2
    # we will retrieve some random samples for an initial prompt, and then kNN
    retr_rng = onp.random.default_rng(23)
    emb_dict = onp.load(args.emb_path)
    emb_dict = {
        'eval': emb_dict['emb_'+args.test_split],
        'retr': emb_dict['emb_'+args.retrieval_split]
    }
    if args.retrieval_method == 'emb':
        retriever = ICLkNNRetriever(Dsrc, Dsrc_retr, Ddest_retr, emb_dict, retr_rng, shuffle=True)
    elif args.retrieval_method.startswith('mixed'):
        k_rand = int(args.retrieval_method.split('-')[1])
        retriever = MixedRetriever(Dsrc, Dsrc_retr, Ddest_retr, emb_dict, retr_rng, k_rand)
    else:
        assert args.retrieval_method == 'random', ValueError(args.retrieval_method)
        retriever = ICLRandRetriever(Dsrc, Dsrc_retr, Ddest_retr, None, retr_rng)

    # shuffle & subsample evaluation data
    ds = None if args.n_data_subsample < 0 else args.n_data_subsample
    idcs = onp.random.default_rng(42).permutation(len(Dsrc))[:ds]
    Dsrc, Ddest = [Dsrc[i] for i in idcs], [Ddest[i] for i in idcs]
    print(len(Dsrc), len(Dsrc_retr), args.n_epis_chains, args.n_intra_chain_samples)

    lm = apis.get(args.lm, args.debug, gcp_region=args.gcp_region, stop_seqs=[])
    preprocessed = []
    for d_id in tqdm.trange(len(Dsrc)):
        query_src = Dsrc[d_id]
        # get ICL demonstrations: - ground truth 
        gt_demo = retriever.retrieve(query_src, args.n_gt_demonstrations)
        # icl_idcs = rng.permutation(len(Dsrc_retr))[:args.n_gt_demonstrations]
        # gt_demo = [(Dsrc_retr[j], Ddest_retr[j]) for j in icl_idcs]
        extra_demo_inps = rewrite(query_src, lm, args.n_completions)
        preprocessed.append({
            "inp": query_src,
            "actual": Ddest[d_id],
            "gt_demo": gt_demo,
            "extra_demo_inps": extra_demo_inps
        })

    dp = os.path.join(args.dir, "preproc.pkl")
    with open(dp, 'wb') as fout:
        pickle.dump(preprocessed, fout)
    print(f"Preprocessed data saved to {dp}")


def proc_output(s: str) -> str:
    return s.split('\n')[0]


def mt_main(args):
    if args.n_epis_chains > 1:
        assert args.n_intra_chain_samples > 2

    with open(os.path.join(args.load_preproc_from, "preproc.pkl"), 'rb') as fin:
        preprocessed = pickle.load(fin)
        assert len(preprocessed) >= args.n_data_subsample
        if args.n_data_subsample > 0:
            preprocessed = preprocessed[:args.n_data_subsample]

    print('loaded preprocessed data', len(preprocessed), file=sys.stderr)

    dest_lang_name = get_lang_name(args.dest_lang)
    lm = apis.get(args.lm, args.debug, gcp_region=args.gcp_region, stop_seqs=[])
    rng = onp.random.default_rng(23)

    trace = []
    trace_nocompletion = []

    for d_id, item in enumerate(tqdm.tqdm(preprocessed)):
        query_src, gt_demo, extra_demo_inps = item['inp'], item['gt_demo'], item['extra_demo_inps']
        gt_demo = [tup for tup in gt_demo if tup[0] != query_src][:args.n_gt_demonstrations]
        # extra_demo_inps = extra_demo_inps[:args.n_completions]
        ncomp_cur = min(len(extra_demo_inps), args.n_completions)
        extra_demo_inp_mat = [
            rng.choice(extra_demo_inps, size=ncomp_cur, replace=False)
            for _ in range(args.n_epis_chains)]
        
        # add synthesized demos
        icl_demo_chains = [gt_demo.copy() for _ in range(args.n_epis_chains)]
        for j in range(ncomp_cur):
            cur_dtup = []
            # complete each chain with model generation
            for i, hist in enumerate(icl_demo_chains):
                cur_inp = extra_demo_inp_mat[i][j] 
                pred_out = lm.complete_single(build_mt_prompt(cur_inp, hist, dest_lang_name))
                if pred_out is None:
                    sys.stderr.write('invalid completion: ' + cur_inp + '\n')
                    break
                cur_dtup.append((cur_inp, proc_output(pred_out)))
            # neglecting this input if any chain failed to complete 
            if len(cur_dtup) < args.n_epis_chains:
                continue
            # update chains
            for hist, cur in zip(icl_demo_chains, cur_dtup):
                hist.append(cur)

        # complete the query
        for chain_id in range(args.n_epis_chains):
            prompt = build_mt_prompt(query_src, icl_demo_chains[chain_id], dest_lang_name)
            query_outs = lm.complete_multi(prompt, args.n_intra_chain_samples)
            if query_outs is None:
                sys.stderr.write('Failed to complete: ' + prompt + '\n')
                continue
            for completion_id, query_out in enumerate(query_outs):
                query_out = proc_output(query_out)
                trace.append({
                    "src": query_src, "pred": query_out, "actual": item['actual'],
                    "demo": icl_demo_chains[chain_id],
                    "id": (chain_id, completion_id)
                })

        # sample for no completion baseline
        query_outs = lm.complete_multi(
            build_mt_prompt(query_src, gt_demo, dest_lang_name), args.n_intra_chain_samples)
        if query_outs is not None:
            for completion_id, query_out in enumerate(query_outs):
                query_out = proc_output(query_out)
                trace_nocompletion.append({
                    "src": query_src, "pred": query_out, "actual": item['actual'],
                    "demo": gt_demo, "id": (-1, completion_id)
                })

        # misc
        if (args.log_every > 0 and d_id%args.log_every == 0) or d_id+1 == len(preprocessed):
            print(d_id, lm.gather_usage(), file=sys.stderr)
            lm.save_history(f"{args.dir}/history.pkl")
            with open(f"{args.dir}/trace.pkl", 'wb') as fout:
                pickle.dump(trace, fout)
            with open(f"{args.dir}/trace-nocomp.pkl", 'wb') as fout:
                pickle.dump(trace_nocompletion, fout)

    for cur_trace in [trace, trace_nocompletion]:
        chrf_scores = []
        for t in cur_trace:
            if t['pred'] is None:
                continue
            chrf_scores.append(sentence_chrf(t['actual'], t['pred'])) 
        chrf_scores = onp.asarray(chrf_scores)
        print(chrf_scores.mean(), chrf_scores.std(), chrf_scores.std() / chrf_scores.shape[0]**0.5 * 1.96)

    print(lm.gather_usage())


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    assert not os.path.exists(os.path.join(args.dir, 'preproc.pkl'))  # ...
    exputils.preflight(args)

    if args.mode == 'preprocess':
        preprocess(args)
    elif args.mode == 'preprocess-replace':
        preprocess_replace_destlang(args)
    else:
        mt_main(args)
