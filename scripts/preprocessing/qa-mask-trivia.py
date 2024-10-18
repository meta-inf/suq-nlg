""" filter possibly outdated questions from trivia datasets """
import argparse, os, sys, pickle, tqdm

sys.path.append(os.getcwd())

from utils.eval_tu import *
from generation import apis


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-d', type=str, default='nq_open')
parser.add_argument('--dataset_path', '-dp', type=str, default="./data/")
parser.add_argument('--output_path', '-o', type=str, default='/tmp/last.pkl')


def generate_check_prompt(row, dataset):
    year = {
        'trivia': 2017,
        'nq_open': 2019
    }[dataset]
    return f"""
I will provide you with a trivia question and its reference answer as of {year}. Please evaluate if the answer to this question is likely to have changed between {year} - 2023. 

Examples of questions that might have changed answers include:
* Recent events or records
* The latest season of an ongoing TV series
* Current holders of public offices or titles

Use your best judgment in the evaluation. In scenarios where your internal knowledge could be inaccurate, rely on common sense and the provided reference answer. Please format your response in two lines, as follows:
1. A brief explanation of your reasoning (one sentence)
2. A "Yes" or "No" answer indicating whether the answer likely changed

Please adhere to the format of the example below:

Question: when did the botswana currency first come into circulation 
Reference Answer: 1976
Reasoning: The date a currency first entered circulation is a historical fact that doesn't change.
Answer: No

Now please answer the following:

Question: {row['question']}
Reference Answer: {row['answer']}
    """.strip() + '\n'


def parse_completion(text):
    lines = text.strip().split('\n')
    return lines[-1].find('Yes') != -1


def main(args):
    with open(os.path.join(args.dataset_path, f"{args.dataset}.pkl"), 'rb') as fin:
        data_recs = pickle.load(fin)

    model = apis.get("gpt-4o", debug=False, stop_seqs=[])

    lst = []
    mask = []
    for i, row in enumerate(tqdm.tqdm(data_recs)):
        prompt = generate_check_prompt(row, 'nq_open')
        responses = model.complete_multi(prompt, 5)
        ret = [parse_completion(resp) for resp in responses]
        lst.append({'id': row['id'], 'raw_response': responses, 'ans': ret})
        mask.append(sum(ret) > 2)

        if i % 100 == 0 or i+1 == len(data_recs):
            with open(args.output_path, 'wb') as fout:
                to_dump = {'recs': lst, 'mask': mask}
                pickle.dump(to_dump, fout)


if __name__ == '__main__':
    main(parser.parse_args())
