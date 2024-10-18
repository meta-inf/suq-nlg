import sys
import numpy as np
from typing import List, Tuple
import tiktoken

from generation.apis import GPT, GCPOpenAIWrapper
from generation.utils import parse_float_between_0_and_1


def clean_text(old_text: str):
    cleaned_text = old_text
    strings_to_filter_on = [
                '\n', 'Q:', 'question:', 'Question:', 'Questions:', 'questions:', 'QUESTION:',
            ]
    for string in strings_to_filter_on:
        if string in cleaned_text:
            cleaned_text = cleaned_text.split(string)[0]
    return cleaned_text


def add_vu_tian_etal(row, dataset, model, text_key='text_cleaned', old_row=None):
    conf_prompt = """
Provide the probability that your last answer is correct. Give ONLY the probability, no other words or explanation.

For example:

Probability: <the probability between 0.0 and 1.0, without any extra commentary whatsoever; just the probability!>
""".strip()
    anss = [_.strip() for _ in row['generations'][text_key]]
    
    conf_of_text = {}
    if old_row is not None and 'vc_tian' in old_row:
        conf_of_text = {
            k: old_row['vc_tian'][k] for k in anss if k in old_row['vc_tian']
        }

    for generation in anss:
        if generation in conf_of_text:
            continue
        if generation == '':
            conf_of_text[generation] = np.nan
            continue
        prompt = [{"role": "user", "content": row["prompt"]}, 
                  {"role": "assistant", "content": generation},
                  {"role": "user", "content": conf_prompt}]
        ret = model.complete_multi(prompt, n_samples=8)
        confs = [parse_float_between_0_and_1(_) for _ in ret]
        conf = np.mean([c for c in confs if c is not None])  # nan if no valid confidences
        conf_of_text[generation] = conf
        # print(conf, file=sys.stderr)

    row['vc_tian'] = conf_of_text


def add_verbalized_confidence(row, dataset, model, text_key='text_cleaned', old_row=None):
    """ adapted from https://github.com/zlin7/UQ-NLG """

    anss = [_.lstrip() for _ in row['generations'][text_key]]
    unique_answers = set(anss)
    few_shots = '\n* '.join(list(unique_answers)[:10])
    story = (row['story'] + '\n') if dataset == 'coqa' else ''

    if isinstance(model, GPT):
        tokenizer = tiktoken.encoding_for_model(model._model_name)
        gpt_kw = {
            "max_tokens": 1,
            "logprobs": True,
            "logit_bias": {tokenizer.encode_single_token(k): +100 for k in ["(A", "(B"]},
            "top_logprobs": 10
        }
        def query_and_parse(prompt):
            resps: List[Tuple[str, float]] = model.complete_custom_kwargs(prompt, gpt_kw)
            p_true = -1
            for text, logprob in resps:
                if 'A' in text:
                    p_true = np.exp(logprob)
                elif 'B' in text:
                    p_true = 1 - np.exp(logprob)
            if p_true < 0:
                sys.stderr.write(f"Invalid response {resps} for row {row['id']}\n")
                return np.nan
            return p_true
    else:
        assert isinstance(model, GCPOpenAIWrapper)
        def query_and_parse(prompt):
            resps = model.complete_multi(prompt, n_samples=5, extra_kw={'max_tokens': 5})
            ret = []
            for text in resps:
                if 'A' in text and text.find('A: ') == -1:
                    ret.append(1)
                elif 'B' in text:
                    ret.append(0)
            if len(ret) == 0:
                sys.stderr.write(f"Invalid response {resps} for row {row['id']}\n")
                return np.nan
            return np.mean(ret)

    conf_of_text = {}
    if old_row is not None and 'verbalized_confidence' in old_row:
        conf_of_text = {
            k: old_row['verbalized_confidence'][k] 
            for k in anss if k in old_row['verbalized_confidence']
        }

    for _ans in anss:
        if _ans in conf_of_text:
            continue
        prompt = f"""{story}Question: {row['question']}
Here are some brainstormed ideas: 
* {few_shots}
Possible Answer: {_ans}
Is the possible answer:
(A) True
(B) False
The possible answer is: """
        conf_of_text[_ans] = query_and_parse(prompt)
    
    row['verbalized_confidence'] = conf_of_text
