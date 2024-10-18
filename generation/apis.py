# Wrapper around LM APIs
import gzip
import os
import pickle
import subprocess
import sys
import time
from typing import List, Union, Tuple, Dict, Any
from pprint import pp

import backoff
import numpy as onp
import anthropic
import openai
import proto
import vertexai
import vertexai.generative_models as vg
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
import google.api_core.exceptions

from generation import utils


def log_backoff(details):
    print(details)
    sys.stderr.write("Backing off {target} {exception} {wait:.1f}s\n".format(**details))


class LMWrapper:

    """ base class for all LM APIs """

    def __init__(self, n_max_retry: int=2, stop_seqs: List[str]=[], max_tokens=2000, debug=False):
        self._n_max_retry, self._debug = n_max_retry, debug
        self._stop_seqs, self._max_tokens = stop_seqs, max_tokens
        self._usage_stats: List[dict] = []
        self._history: List[dict] = []

    def complete_single(self, prompt: str) -> Union[str, Tuple[str, float]]:
        raise NotImplementedError()
    
    def complete_multi(
        self, prompt: str, n_samples: int) -> Union[List[str], List[Tuple[str, float]]]:
        raise NotImplementedError()
    
    def _sleep_before_calling(self, fn):
        if not hasattr(self, '_rpm'):
            return fn()
        last_call = getattr(self, '_last_call', 0)
        wait_secs = 1.2 * 60 / self._rpm - (time.time() - last_call)
        if wait_secs > 0:
            time.sleep(wait_secs)
        ret = fn()
        self._last_call = time.time()
        return ret
    
    def wrap_api_call(func):
        def wrapper(self: LMWrapper, *args, **kwargs):
            for _ in range(self._n_max_retry):
                try:
                    ret, (usage, history) = self._sleep_before_calling(
                        lambda: func(self, *args, **kwargs))
                    self.log_usage(usage)
                    self.log_history(history)
                    return ret
                except Exception as e:
                    print(f'Error: {str(e)}', file=sys.stderr)
                    time.sleep(10)
                    continue
            return None
        return wrapper

    def log_usage(self, usage: Dict[str, Any]):
        self._usage_stats.append(utils.flatten_dict(usage))

    def log_history(self, st: Dict):
        if self._debug:
            pp(st, stream=sys.stderr)
        self._history.append(st)

    def gather_usage(self) -> Dict[str, int]:
        return {k: sum([_[k] for _ in self._usage_stats if k in _]) 
                for k in self._usage_stats[0].keys()}

    def save_history(self, path: str):
        with gzip.open(path, 'wb') as fout:
            pickle.dump(self._history, fout)


class GeminiText(LMWrapper):

    # Uses the default (ADC) project id

    def __init__(self, model_name, stop_seqs, system_prompt, debug=False, max_tokens=2000, 
                 location='us-central1'):
        super().__init__(stop_seqs=stop_seqs, debug=debug, max_tokens=max_tokens)
        vertexai.init(location=location)
        self._model_name, self._rpm = {
            'gemini-1.5-flash': ('gemini-1.5-flash-001', 200),
            'gemini-1.5-pro': ('gemini-1.5-pro-001', 60),
        }[model_name]
        self._model = vg.GenerativeModel(
            model_name=self._model_name, system_instruction=system_prompt)

    def _safety_settings(self):
        return {h: vg.SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH for h in vg.HarmCategory}
    
    @LMWrapper.wrap_api_call
    def _complete_multi(self, prompt, n_samples):
        try:
            resp = self._model.generate_content(
                prompt, 
                safety_settings=self._safety_settings(),
                generation_config=vg.GenerationConfig(
                    stop_sequences=self._stop_seqs,
                    candidate_count=n_samples,
                    max_output_tokens=self._max_tokens,
                ))
            history = {"prompt": prompt, "response": resp.to_dict()}
            usage = proto.Message.to_dict(resp.usage_metadata)
            generations = [cd.content.parts[0].text for cd in resp.candidates if cd.content.parts]
        except Exception as e:
            print(e, file=sys.stderr)
            print(resp, file=sys.stderr)
            raise e
        return generations, (usage, history)
    
    def complete_multi(self, prompt, n_samples):
        # NOTE due to the safety setting, we may fail to get any sample for some queries
        ret = []
        for _ in range(0, 10):
            cur = self._complete_multi(prompt, min(8, n_samples))
            if cur is not None:
                n_samples -= len(cur)  # we don't always get the number of samples we ask for
                ret.extend(cur)
            if n_samples <= 0:
                break
        return ret
    
    def complete_single(self, prompt):
        ret = self.complete_multi(prompt, n_samples=1)
        return ret[0] if len(ret) > 0 else None
   

class GPT(LMWrapper):

    def __init__(self, model_name, stop_tokens, system_prompt=None, debug=False, max_tokens=2000):
        super().__init__(debug=debug, stop_seqs=stop_tokens, max_tokens=max_tokens)
        self._client = openai.OpenAI()
        self._system_prompt = system_prompt
        self._model_name, self._rpm = {
            'gpt-4o': ('gpt-4o-2024-08-06', 500),
            'gpt-4o-mini': ('gpt-4o-mini-2024-07-18', 1000),
            'gpt-3.5': ('gpt-3.5-turbo-0125', 500),
        }[model_name]

    @LMWrapper.wrap_api_call
    @backoff.on_exception(backoff.expo, openai.RateLimitError, max_time=240)
    def complete_custom_kwargs(self, prompt, kwargs):
        kwargs = {"max_tokens": self._max_tokens, "stop": self._stop_seqs} | kwargs

        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            assert all(isinstance(_, dict) for _ in prompt)
            messages = prompt
        if self._system_prompt is not None:
            messages = [{"role": "system", "content": self._system_prompt}] + messages

        resp = self._client.chat.completions.create(
            model=self._model_name, 
            messages=messages, 
            **kwargs)

        ret = []
        for ch in resp.choices:
            text_i = ch.message.content
            if "logprobs" in kwargs and kwargs["logprobs"]:
                logprobs = [t.logprob for t in ch.logprobs.content]
                ret.append((text_i, onp.sum(logprobs)))
            else:
                ret.append(text_i)

        history = {"prompt": prompt, "response": resp.to_dict()}
        usage = {'tokens': resp.usage.to_dict()}
        return ret, (usage, history)
    
    def complete_multi(self, prompt: Union[str, dict], n_samples: int):
        return self.complete_custom_kwargs(prompt, {"n": n_samples})
    
    def complete_single(self, prompt):
        return self.complete_multi(prompt, n_samples=1)[0]


def get_gcp_project_id():
    result = subprocess.run('gcloud config get-value project'.split(' '), stdout=subprocess.PIPE)
    return result.stdout.decode('ascii').strip()


class GCPOnlinePrediction(LMWrapper):

    """
    LMs accessed through the online prediction API.
    Generally pretrained models that don't support any chat prompt format.
    """

    def __init__(self, endpoint_id, stop_tokens, is_hex_llm=False, system_prompt=None, 
                 debug=False, max_tokens=2000, location='us-central1'):
        vertexai.init(location=location)
        super().__init__(debug=debug, stop_seqs=stop_tokens, max_tokens=max_tokens)
        self._location, self._project_id = location, get_gcp_project_id()
        self._endpoint_id, self._is_hex_llm = endpoint_id, is_hex_llm
        api_endpoint = f"{location}-aiplatform.googleapis.com"
        client_options = {"api_endpoint": api_endpoint}
        self._client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)

    @LMWrapper.wrap_api_call
    @backoff.on_exception(backoff.expo, google.api_core.exceptions.GoogleAPICallError, max_time=240)
    def _predict(
            self,
            instances: Union[Dict, List[Dict]],
    ):
        instances = instances if isinstance(instances, list) else [instances]
        instances = [
            json_format.ParseDict(instance_dict, Value()) for instance_dict in instances
        ]
        parameters_dict = {}
        parameters = json_format.ParseDict(parameters_dict, Value())
        endpoint = self._client.endpoint_path(
            project=self._project_id, location=self._location, endpoint=self._endpoint_id)
        response = self._client.predict(
            endpoint=endpoint, instances=instances, parameters=parameters)
        predictions = response.predictions
        usage, history = {}, json_format.MessageToDict(response._pb)
        return predictions, (usage, history)

    def complete_multi(self, prompt: str, n_samples: int):
        ret = []
        batch_size = 10 if not self._is_hex_llm else 5
        for _ in range(0, n_samples, batch_size):
            if not self._is_hex_llm:
                instance = {
                    "n": min(10, n_samples - _),
                    "max_tokens": self._max_tokens,
                    "stop": self._stop_seqs,
                    "raw_response": True,
                    "prompt": prompt
                }
                ret.extend([s.lstrip() for s in self._predict([instance])])
            else:
                # HexLLM doesn't have sane defaults
                instance = {
                    "max_tokens": min(self._max_tokens, 100),
                    "temperature": 1.,
                    "top_k": -1,
                    "stop": self._stop_seqs,
                    "raw_response": True,
                    "prompt": prompt
                }
                ret.extend([s.lstrip() for s in self._predict([instance]*5)])
        return ret
    
    def complete_single(self, prompt):
        return self.complete_multi(prompt, n_samples=1)[0]


class GCPOpenAIWrapper(LMWrapper):

    """
    GCP models that provide an OpenAI-like API, including MaaS and custom API endpoints.
    """

    def _endpoint_url(self):
        BASE = f"https://{self._location}-aiplatform.googleapis.com/v1beta1/"
        if hasattr(self, "_endpoint_id"):
            return BASE + "projects/{}/locations/{}/endpoints/{}".format(self._project_id, self._location, self._endpoint_id)
        return BASE + "projects/{self._project_id}/locations/{self._location}/endpoints/openapi/chat/completions?"
    
    def get_client(self):
        if hasattr(self, "_token_creation_time") and time.time() - self._token_creation_time < 3000:
            # token expires in 1 hour
            return self._temp_client
        
        # refresh token & create new client object
        from google.auth import default, transport            
        self._token_creation_time = time.time()
        credentials, _ = default()
        auth_request = transport.requests.Request()
        credentials.refresh(auth_request)

        self._temp_client = openai.OpenAI(base_url=self._endpoint_url(), api_key=credentials.token)
        return self._temp_client

    def __init__(self, model_name, stop_tokens, system_prompt=None, 
                 debug=False, max_tokens=2000, location='us-central1'):
        vertexai.init(location=location)
        super().__init__(debug=debug, stop_seqs=stop_tokens, max_tokens=max_tokens)
        self._location, self._project_id = location, get_gcp_project_id()
        self._system_prompt = system_prompt
        maas_models = {  # NOTE these points to Llama 3.1 models
            'llama3.1-70b': ('meta/llama3-70b-instruct-maas', 60),
            'llama3.1-8b': ('meta/llama3-8b-instruct-maas', 60),
        }
        if model_name in maas_models:
            self._model_name, self._rpm = maas_models[model_name]
        elif model_name.startswith('gcpv'):
            self._endpoint_id = model_name.split('gcpv-')[1]
            self._rpm = 1000  # don't restrict
        else:  # Gemma is restrictive
            self._max_tokens = 300
            self._endpoint_id = model_name.split('gcpg-')[1]
            self._rpm = 1000  # don't restrict
        self._is_gemma = model_name.startswith('gcpg-')

    def get_default_kw(self):
        return {
            "max_tokens": self._max_tokens,
            "extra_body": {
                "extra_body": {
                    "google": {
                        "model_safety_settings": {
                            "enabled": False,
                            "llama_guard_settings": {},
                        }
                    }
                }
            },
            "model": self._model_name,
        }
    
    def get_custom_kw(self):
        return {
            "max_tokens": self._max_tokens,
            "model": "",
            "stop": self._stop_seqs,
        }

    @LMWrapper.wrap_api_call
    @backoff.on_exception(backoff.expo, openai.RateLimitError, max_time=240)
    def _call_completion(self, prompt, kwargs):
        client = self.get_client()
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            assert all(isinstance(_, dict) for _ in prompt)
            messages = prompt
        if self._system_prompt is not None:
            if not self._is_gemma:
                messages = [{"role": "system", "content": self._system_prompt}] + messages
            else:
                assert messages[0]["role"] == "user"
                messages = [{
                    "role": "user",
                    "content": self._system_prompt + "\n" + messages[0]["content"]
                }] + messages[1:]
        resp = client.chat.completions.create(messages=messages, **kwargs)
        history = {"prompt": prompt, "response": resp.to_dict()}
        usage = {'tokens': resp.usage.to_dict()}
        _ = [ch.message.content for ch in resp.choices][0] # make sure these exist, retry if not
        return resp, (usage, history)

    def complete_single(self, prompt):
        if hasattr(self, "_endpoint_id"):
            kwargs = self.get_custom_kw()
            resp = self._call_completion(prompt, kwargs)
        else:
            kwargs = self.get_default_kw()
            resp = self._call_completion(prompt, kwargs)
        return resp.choices[0].message.content
    
    def complete_multi(self, prompt: str, n_samples: int, extra_kw={}):
        if hasattr(self, "_endpoint_id") and not self._is_gemma:
            batch_size = 10
        else:
            batch_size = 1

        ret = []
        for _ in range(0, 2 * (n_samples//batch_size + 1)):

            if batch_size == 1:
                cur = self.complete_single(prompt)
                if cur is not None:
                    n_samples -= batch_size
                    ret.append(cur)
            else:
                delt_n = min(n_samples, batch_size)
                kwargs = self.get_custom_kw() | extra_kw | {"n": delt_n}
                resp = self._call_completion(prompt, kwargs)
                if resp is not None:
                    cur = [ch.message.content for ch in resp.choices]
                    ret.extend(cur)
                    n_samples -= len(cur)

            if n_samples <= 0:
                break

        return ret


class AnthropicWrapper(LMWrapper):
    
    def __init__(self, model_name, stop_tokens, system_prompt=None, 
                 debug=False, max_tokens=2000, location='us-east5'):
        assert location in ['us-east5', 'europe-west1']
        vertexai.init(location=location)
        super().__init__(debug=debug, stop_seqs=stop_tokens, max_tokens=max_tokens)
        self._client = anthropic.AnthropicVertex(
            region=location, project_id=get_gcp_project_id())
        self._system_prompt = system_prompt
        self._model_name, self._rpm = {
            'claude-3.5': ('claude-3-5-sonnet@20240620', 90),
        }[model_name]

    @LMWrapper.wrap_api_call
    @backoff.on_exception(backoff.expo, anthropic.APIError, max_time=240, on_backoff=log_backoff)
    def complete_single(self, prompt):
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            assert all(isinstance(_, dict) for _ in prompt)
            messages = prompt
        kwargs = {
            "max_tokens": self._max_tokens, 
            "model": self._model_name,
            "stop_sequences": [s for s in self._stop_seqs if s.strip()]
        }
        if self._system_prompt is not None:
            kwargs["system"] = self._system_prompt

        resp = self._client.messages.create(messages=messages, **kwargs)
        ret = resp.content[0].text
        history = {"prompt": prompt, "response": resp.to_dict()}
        usage = {'tokens': resp.usage.to_dict()}
        return ret, (usage, history)
    
    def complete_multi(self, prompt: str, n_samples: int):
        return GCPOpenAIWrapper.complete_multi(self, prompt, n_samples)


def get(lm_name: str, debug: bool, stop_seqs=['\n'], system_prompt=None,
        gcp_region='us-central1') -> LMWrapper:
    if lm_name.startswith('gemini'):
        return GeminiText(lm_name, stop_seqs, system_prompt=system_prompt, debug=debug, location=gcp_region)
    elif lm_name.startswith('gpt'):
        return GPT(lm_name, stop_seqs, system_prompt=system_prompt, debug=debug)
    elif lm_name.startswith('llama') or lm_name.startswith('gcpv') or lm_name.startswith('gcpg'):
        return GCPOpenAIWrapper(lm_name, stop_seqs, system_prompt=system_prompt, debug=debug, location=gcp_region)
    elif lm_name.startswith('claude'):
        return AnthropicWrapper(lm_name, stop_seqs, system_prompt=system_prompt, debug=debug, location=gcp_region)
    elif lm_name.startswith('gcp'):
        is_hex = lm_name.startswith('gcp_hex')
        endpoint_id = lm_name.split('-')[1]
        return GCPOnlinePrediction(
            endpoint_id, stop_seqs, system_prompt=system_prompt, is_hex_llm=is_hex,
            debug=debug, location=gcp_region)
    raise ValueError(f'Unknown LM: {lm_name}')
