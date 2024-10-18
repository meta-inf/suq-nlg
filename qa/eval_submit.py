import argparse, json, os, sys, pickle
import openai, tiktoken, tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--inp_file', '-i', type=str)
parser.add_argument('--rec_file', '-r', type=str, default=None)
parser.add_argument('--dry', action='store_true', default=True)
parser.add_argument('--run', action='store_false', dest='dry')
parser.add_argument('--online', action='store_true', default=False, help='use online APIs')


def online_outfile_path(inp_path):
    return inp_path.replace('.jsonl', '_out.jsonl')


def main(args):
    openai_client = openai.Client()

    if args.rec_file is None:
        rec_file = args.inp_file.replace('requests_', 'map_').replace('.jsonl', '.pkl')
    else:
        rec_file = args.rec_file
    assert os.path.exists(rec_file)
    
    # load input
    with open(args.inp_file) as fin:
        lines = fin.readlines()

    # get model name and encoder
    mname = json.loads(lines[0])['body']['model']
    sys.stderr.write(f'using model {mname}\n')
    encoder = tiktoken.encoding_for_model(mname)
        
    # break into multiple files if necessary
    paths = []
    n_tokens = [len(encoder.encode(json.loads(line)['body']['messages'][0]['content']))
                 for line in lines]
    if sum(n_tokens) <= 2e6 and len(lines) <= 10000:
        paths = [args.inp_file]
    else:
        i_s, n_lines = 0, len(lines)
        while i_s < n_lines:
            i_e, n_tokens_i = i_s, 0
            while i_e < min(n_lines, i_s+10000) and n_tokens_i+n_tokens[i_e] < 2e6:
                n_tokens_i += n_tokens[i_e]
                i_e += 1
            path_i = args.inp_file.replace('.jsonl', f'_{i_s}.jsonl')
            sys.stderr.write(f'writing {path_i} with {i_s}:{i_e} ({n_tokens_i} tokens)\n')
            with open(path_i, 'w') as fout:
                fout.writelines(lines[i_s:i_e])
            paths.append(path_i)
            i_s = i_e
    
    # validate rec_file: check if all `custom_id` referenced there presents
    req_ids = set()
    for req in lines:
        req_ids.add(json.loads(req)['custom_id'])
    
    with open(rec_file, 'rb') as fin:
        recs = pickle.load(fin)
        for (_, _, sim_rmat, true_rmat) in recs:
            for item in sim_rmat.reshape((-1,)):
                assert item == 'SAME' or item in req_ids
            for item in true_rmat.reshape((-1,)):
                assert item == 'SAME' or item in req_ids
    
    if args.dry:
        sys.exit(0)
    
    if args.online:
        # query the ChatCompletion API online and save results to multiple files
        for inp_path in paths:
            with open(inp_path) as fin:
                lines = fin.readlines()
            out_path = online_outfile_path(inp_path)
            with open(out_path, 'w') as fout:
                for line in tqdm.tqdm(lines, "querying completion"):
                    req = json.loads(line)
                    ret = openai_client.chat.completions.create(**req['body'])
                    ret = {"custom_id": req['custom_id'], "response": {"body": ret.to_dict()}}
                    fout.write(json.dumps(ret) + '\n')
    else:
        # submit offline jobs
        srec_path = rec_file.replace('.pkl', '_submission.jsonl')
        if os.path.exists(srec_path):
            raise FileExistsError(srec_path)
        
        s_recs = []
        for path in tqdm.tqdm(paths):
            with open(path, 'rb') as fin:
                ret = openai_client.files.create(file=fin, purpose='batch')
                print('file created:', ret)
            ret = openai_client.batches.create(
                input_file_id=ret.id, endpoint='/v1/chat/completions', completion_window='24h')
            print('job submitted:', ret)
            s_recs.append(ret.to_dict())
        
        with open(srec_path, 'w') as fout:
            for s in s_recs:
                fout.write(json.dumps(s) + '\n')


if __name__ == '__main__':
    main(parser.parse_args())