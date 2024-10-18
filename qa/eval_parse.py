import argparse, json, os, sys, pickle, re, glob
import os.path as osp
import numpy as np
import openai, tqdm

from qa.eval_submit import online_outfile_path
from utils.eval_tu import *


parser = argparse.ArgumentParser()
parser.add_argument('--rec_file', '-r', type=str)
parser.add_argument('--submission_rec_file', '-sr', type=str, default=None)


def download_batch_outputs(submission_rec_file, openai_client):

    out_file_ids = []
    with open(submission_rec_file, 'r') as fin:
        lines = fin.readlines()
        for ln in tqdm.tqdm(lines, "retrieving responses"):
            batch_id = json.loads(ln)["id"]
            b = openai_client.batches.retrieve(batch_id)
            if b.status != 'completed':
                print('batch not completed:', b, file=sys.stderr)
                sys.exit(1)
            out_file_ids.append(b.output_file_id)

    out_file_paths = []
    for out_file_id in tqdm.tqdm(out_file_ids, "downloading output"):
        path = os.path.join(os.path.dirname(submission_rec_file), f'out-{out_file_id}.jsonl')
        out_file_paths.append(path)
        if os.path.exists(path):
            print(f"file {path} already exists, skipping", file=sys.stderr)
        else:
            out_file = openai_client.files.content(out_file_id)
            out_file.write_to_file(path)

    return out_file_paths


def main(args):
    openai_client = openai.Client()

    if args.submission_rec_file is None:
        args.submission_rec_file = args.rec_file.replace('.pkl', '_submission.jsonl')

    if os.path.exists(args.submission_rec_file):
        sys.stderr.write('retrieving batch outputs\n')
        out_file_paths = download_batch_outputs(args.submission_rec_file, openai_client)
    else:
        # online
        req_files = glob.glob(osp.join(osp.dirname(args.rec_file), 'requests_*.jsonl'))
        out_file_paths = [
            opath for req_file in req_files if osp.exists(opath := online_outfile_path(req_file))]
   
    out_by_id = {}
    for path in tqdm.tqdm(out_file_paths, "loading responses"):
        with open(path, 'r') as fin:
            for line in fin:
                rec = json.loads(line)
                try:
                    # resp = rec['response']['body']['choices'][0]['message']['content']
                    resps = [
                        choice['message']['content'] for choice in rec['response']['body']['choices']]
                except Exception as e:
                    print("error loading response:", rec, e, file=sys.stderr)
                    continue
                resp_ints = []
                for resp in resps:
                    if (resp_int := re.search(r'\d+', resp)) is not None:
                        resp_ints.append(int(resp_int.group()))
                if len(resp_ints) == 0:
                    print("error parsing response:", resps, file=sys.stderr)
                    continue
                out_by_id[rec['custom_id']] = np.mean(resp_ints)
    
    def parse_r_entry(e):
        if e == 'SAME': return 1.
        if e not in out_by_id:
            print(f"entry {e} not found in out_by_id", file=sys.stderr)
            return np.nan
        return out_by_id[e] / 100.
    
    with open(args.rec_file, 'rb') as fin:
        recs = pickle.load(fin)
    
    new_recs = []
    for (row, ans_samples, sim_rmat, true_rmat) in recs:
        sim_rmat  = np.asarray([[parse_r_entry(e) for e in row] for row in sim_rmat])
        true_rmat = np.asarray([[parse_r_entry(e) for e in row] for row in true_rmat])
        new_recs.append((row, ans_samples, sim_rmat, true_rmat))

    dump_path = args.rec_file.replace('.pkl', '_parsed.pkl')
    with open(dump_path, 'wb') as fout:
        pickle.dump(new_recs, fout)


if __name__ == '__main__':
    main(parser.parse_args())