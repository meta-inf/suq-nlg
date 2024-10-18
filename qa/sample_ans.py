import os, sys, pickle, json, os.path as osp

import tqdm
import numpy as np
import argparse
import tiktoken

from generation import apis
from qa import vu


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-d', type=str, default='coqa')
parser.add_argument('--dataset_path', '-dp', type=str, default="./data/")
parser.add_argument('--n_samples_per_dp', '-nspd', type=int, default=None)
parser.add_argument('--seed', '-s', type=int, default=1)
parser.add_argument('--n_subsample', '-n', type=int, default=1000)

parser.add_argument('--gen_model', '-gm', type=str, default='gpt-4o-mini')
parser.add_argument('--gcp_region', type=str, default='us-central1')

parser.add_argument('--dump_dir', '-o', type=str)
parser.add_argument('--resume', action='store_true', default=False)
parser.add_argument('--save_every', type=int, default=30)
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--run', action='store_true', default=False)


QA_SYSTEM_PROMPT = """You are a helpful AI assistant designed for question answering tasks. Provide complete answers in a single line. If given specific instructions or demonstrations in the prompt, follow them carefully."""


def postprocess(dataset: str, model: apis.LMWrapper, dump_dir: str, save_every=30):
    with open(os.path.join(dump_dir, 'records.pkl'), 'rb') as fin:
        recs = pickle.load(fin)

    if osp.exists(osp.join(dump_dir, 'records_pp.pkl')):
        with open(os.path.join(dump_dir, 'records_pp.pkl'), 'rb') as fin:
            old_postprocessed = pickle.load(fin)
    else:
        old_postprocessed = []

    recs_pp = []
    
    for i in tqdm.trange(len(recs)):
        row = recs[i]
        add_verbalized = isinstance(model, (apis.GPT, apis.GCPOpenAIWrapper))
        if i < len(old_postprocessed):
            old_row = old_postprocessed[i]
            assert old_row['id'] == row['id'], f"ID mismatch at {i}"
            if not add_verbalized or ('vc_tian' in old_row):
                recs_pp.append(old_row)
                continue
        else:
            old_row = None

        row['generations']['text_cleaned'] = [vu.clean_text(_) for _ in row['generations']['text']]
        if add_verbalized:
            vu.add_verbalized_confidence(row, dataset, model, old_row=old_row)
            vu.add_vu_tian_etal(row, dataset, model, old_row=old_row)
        recs_pp.append(row)

        if i % save_every == 0 or i+1 == len(recs):
            with open(os.path.join(dump_dir, 'records_pp.pkl'), 'wb') as fout:
                pickle.dump(recs_pp, fout)
            print(f"Saved {i+1} postprocessed records to {dump_dir}", model.gather_usage(), file=sys.stderr)

    with open(os.path.join(args.dump_dir, 'records_pp.pkl'), 'wb') as fout:
        pickle.dump(recs, fout)
        

def main(args):
    if not args.resume:
        if args.run:
            os.makedirs(args.dump_dir, exist_ok=False)
            with open(os.path.join(args.dump_dir, 'args.json'), 'w') as fout:
                json.dump(vars(args), fout)
        # no existing record
        old_ret = []
    else:
        # check if old args are consistent
        KEYS = ['gen_model', 'seed', 'n_samples_per_dp', 'dataset']
        with open(os.path.join(args.dump_dir, 'args.json'), 'r') as fin:
            old_args = json.load(fin)
            assert all(old_args[k] == vars(args)[k] for k in KEYS)
            assert old_args['n_subsample'] <= args.n_subsample
        # load the old records
        with open(os.path.join(args.dump_dir, 'records.pkl'), 'rb') as fin:
            old_ret = pickle.load(fin)

    # load dataset and shuffle
    with open(os.path.join(args.dataset_path, f"{args.dataset}.pkl"), 'rb') as fin:
        data_recs = pickle.load(fin)

    rng = np.random.default_rng(args.seed)
    subsample_idcs = rng.permutation(len(data_recs))[:args.n_subsample]
    data_recs = [data_recs[i] for i in subsample_idcs]

    nspd = args.n_samples_per_dp or len(data_recs[0]['generations']['text'])
    # approximate token cost using the GPT tokenizer
    enc = tiktoken.encoding_for_model('gpt-4o')
    inp_tokens = sum(len(enc.encode(g['prompt'])) for g in data_recs)
    out_tokens = sum(len(enc.encode(g['answer'])) for g in data_recs)
    sys.stderr.write(f"Token cost: {inp_tokens/1e6} inp Mtokens, {out_tokens*nspd/1e6} out Mtokens\n")

    if not args.run:
        return

    model = apis.get(
        lm_name=args.gen_model, 
        debug=args.debug, 
        stop_seqs=['\n'], # this is consistent with Lin et al, see their dataeval.load_worker._clean_output
        system_prompt=QA_SYSTEM_PROMPT,
        gcp_region=args.gcp_region) 
    
    # GENERATE ANSWERS
    ret = []
    for i in tqdm.trange(len(subsample_idcs)):
        rec_i = data_recs[i]
        if i < len(old_ret):  # skipping completed ones
            if old_ret[i]['id'] != rec_i['id']:
                raise ValueError(f"ID mismatch: {old_ret[i]['id']} != {rec_i['id']}")
            ret.append(old_ret[i])
            continue

        generations = model.complete_multi(rec_i['prompt'], n_samples=nspd)
        new_rec = rec_i | {'generations': {'text': generations}}
        ret.append(new_rec)
        
        if i % args.save_every == 0 or i+1 == len(subsample_idcs):
            with open(os.path.join(args.dump_dir, 'records.pkl'), 'wb') as fout:
                pickle.dump(ret, fout)
            print(f"Saved {i+1} records to {args.dump_dir}", model.gather_usage(), file=sys.stderr)

    postprocess(args.dataset, model, args.dump_dir)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)