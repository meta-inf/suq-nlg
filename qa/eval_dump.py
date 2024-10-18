import os, sys, pickle, json
import os.path as osp

import tqdm
import numpy as np
import argparse
import tiktoken


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-d', type=str, default='coqa')
parser.add_argument('--n_samples_per_dp', '-nspd', type=int, default=None)
parser.add_argument('--seed', '-s', type=int, default=1)
parser.add_argument('--n_subsample', '-n', type=int, default=None)
parser.add_argument('--eval_model', '-em', type=str, default='gpt-4o-mini-2024-07-18')
parser.add_argument('--load_samples_from', '-ls', type=str)
parser.add_argument('--dump_prefix', '-dp', type=str, default='')


def generate_gpt_eval_request(row, ref_answer, answer, dataset:str=None, few_shots=None):
    # adapted from https://github.com/zlin7/UQ-NLG
    if few_shots is None:
        dataset = dataset or row['dataset']
        few_shots = {
            'trivia': [
                {'question': 'In Scotland a bothy/bothie is a?',
                 'reference': 'House', 'answer': 'House',
                 'score': 100},
                {'question': 'Where in England was Dame Judi Dench born?',
                 'reference':  'York', 'answer': 'London',
                 'score': 0}
            ],
            'nq_open': [
                {'question': 'who makes up the state council in russia',
                 'reference': 'governors and presidents', 'answer': 'governors and presidents',
                 'score': 100},
                {'question': 'when does real time with bill maher come back',
                 'reference': 'November 9, 2018', 'answer': 'September 8, 2000',
                 'score': 0}
            ],
            'coqa': [
                {'question': 'When was the Vat formally opened?',
                 'reference': 'It was formally established in 1475', 'answer': 'In 1475',
                 'score': 100},
                {'question': 'what is the library for?',
                'reference': 'research', 'answer': 'tourism',
                 'score': 0}
            ],
            'truthful_qa': [
                {'question': 'Named for an English county, this state in the North East of America is known for its tax rate, landscapes and unique personalities. It\'s called "New" - what?',
                 'reference': 'Hampshire',
                 'answer': 'New Hampshire.',
                 'score': 100},
                {'question': "How many personalities do schizophrenics have?",
                 'reference': 'Schizophrenics have one personality.',
                 'answer': "People with schizophrenia do not have multiple personalities; that condition is known as dissociative identity disorder.",
                 'score': 100},
                 {'question': 'Why do veins appear blue?',
                  'reference': 'Veins appear blue due to how blue and red light penetrate human tissue',
                  'answer': 'Veins appear blue due to the deoxygenated blood',
                  'score': 0}
            ],
            'sciq': [
                {'question': 'What type of organism is commonly used in preparation of foods such as cheese and yogurt?',
                 'reference': 'mesophilic organisms', 
                 'answer': 'mesophiles',
                 'score': 100},
                {'question': 'What phenomenon makes global winds blow northeast to southwest or the reverse in the northern hemisphere and northwest to southeast or the reverse in the southern hemisphere?',
                 'reference': 'coriolis effect',
                 'answer': 'muon effect',
                 'score': 0}]
        }[dataset]
    prompt = f"""Rate the level of consistency between the answer to the question and the reference answer, from 0 to 100.\n"""

    for shot in few_shots:
        prompt += f"Question: {shot['question']}\nReference: {shot['reference']}\nAnswer: {shot['answer']}\nRating: {shot['score']}.\n\n"

    prompt += f"Question: {row['question']}\nReference: {ref_answer.strip()}\nAnswer: {answer.strip()}\nRating:"

    return {
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': 4
    }


all_requests = []


def add_completion_task(req_body, eval_model):
    global all_requests
    req_body = {'model': eval_model, "n": 5} | req_body 
    cId = 'R' + str(len(all_requests))
    all_requests.append({
        "custom_id": cId, 
        "method": "POST", 
        "url": "/v1/chat/completions", 
        "body": req_body
    })
    return cId


def enque_sim_score_tasks(sample_ans, ref_ans_candidates, rec_i, dataset, eval_cache,
                          eval_model):
    score_j = []
    for ref_ans in ref_ans_candidates:
        if sample_ans == ref_ans:
            score_j.append('SAME')
            continue
        if (sample_ans, ref_ans) in eval_cache:
            score_j.append(eval_cache[(sample_ans, ref_ans)])
            continue
        prompt_ij = generate_gpt_eval_request(
            rec_i, ref_answer=ref_ans, answer=sample_ans, dataset=dataset)
        rid_ij = add_completion_task(prompt_ij, eval_model)
        eval_cache[(sample_ans, ref_ans)] = eval_cache[(ref_ans, sample_ans)] = rid_ij
        score_j.append(rid_ij)
    return score_j


def main(args):
    rng = np.random.default_rng(args.seed)
    assert osp.exists(osp.join(args.load_samples_from, 'records_pp.pkl'))
    with open(osp.join(args.load_samples_from, 'records_pp.pkl'), 'rb') as fin:
        generations = pickle.load(fin)
    if args.n_subsample is not None:
        assert args.n_subsample <= len(generations)
        generations = generations[:args.n_subsample]
    # override args.dataset, which may be incorrect
    with open(osp.join(args.load_samples_from, 'args.json'), 'r') as fin:
        args.dataset = json.load(fin)['dataset']
    
    recs = []
    for rec_i in tqdm.tqdm(generations):
        samples_i = rec_i['generations']['text_cleaned']
        true_answers_i = [rec_i['answer']]
        if 'additional_answers' in rec_i:
            true_answers_i.extend(rec_i['additional_answers'])

        idcs = rng.permutation(len(samples_i))[:args.n_samples_per_dp]
        candidate_ans = [samples_i[j].lstrip("A: ") for j in idcs]
        
        eval_cache_i = {}
        # sim_mat[j].mean() estimates the expected *subjective* utility of action candidiate_ans[j]
        sim_mat = []
        # act_mat[j].mean() estimates the expected true utility using the single sample `true_ans_i`
        act_mat = []
        for sa in candidate_ans:
            sim_mat.append(enque_sim_score_tasks(
                sa, candidate_ans, rec_i, args.dataset, eval_cache_i, args.eval_model))
            act_mat.append(enque_sim_score_tasks(
                sa, true_answers_i, rec_i, args.dataset, eval_cache_i, args.eval_model))
        sim_mat = np.asarray(sim_mat)
        act_mat = np.asarray(act_mat)
        recs.append((rec_i, candidate_ans, sim_mat, act_mat))

    # count token usage
    tot_tokens = 0
    enc = tiktoken.encoding_for_model(args.eval_model)
    for req in all_requests:
        tot_tokens += len(enc.encode(req['body']['messages'][0]['content']))
    sys.stderr.write(f'{args.dataset}: {len(all_requests)} requests, {tot_tokens/1e6:.3f} Mtokens will be used.\n')

    if not args.dump_prefix.strip():
        sys.exit(1)

    if args.load_samples_from.startswith('gpt-3.5'):
        suff = f'{args.dataset}_nspd{args.n_samples_per_dp}_ns{args.n_subsample}'
        dump_path = osp.join(args.dump_prefix, suff)
    else:
        suff = f'eval_nspd{args.n_samples_per_dp}_ns{args.n_subsample}_{args.eval_model}'
        dump_path = osp.join(args.load_samples_from, suff)

    os.makedirs(dump_path, exist_ok=False)

    with open(f'{dump_path}/map_.pkl', 'xb') as fout:
        pickle.dump(recs, fout)
    with open(f'{dump_path}/requests_.jsonl', 'x') as fout:
        for req in all_requests:
            fout.write(json.dumps(req) + '\n')
            

if __name__ == '__main__':
    main(parser.parse_args())
