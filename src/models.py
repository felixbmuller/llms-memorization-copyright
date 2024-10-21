"""Provide different LLMs with a unified interface."""

import json
from pprint import pprint
import time
import pickle
from base64 import b64encode
from typing import Any

import openai
import aleph_alpha_client
import requests
from fire import Fire

from src.utils import timestamp
from src.api_keys import api_keys


def get_model(provider_name, model_name, temp=None, max_tokens=None, sleep_interval=1):
    """Get a `LLM` instance for the given model name.

    Parameters
    ----------
    model_name : str
        model identifier
    temp : float, optional
        softmax temperature, by default None (use model-specific default)

    Returns
    -------
    LLM
        model instance
    """

    provider = {
        "openai": OpenAI,
        "together": TogetherOpenAI,
        "alephalpha": AlephAlpha,
    }[provider_name]

    params = {
        "model": model_name,
        "temperature": temp,
        "sleep_interval": sleep_interval,
        "max_tokens": max_tokens,
        **api_keys,
    }

    return provider(**params)


class LLM:
    """Base class for different providers of LLMS. This handles the prompt generation, storage of model outputs and - if needed - sleep intervals to reduce API load. Subclasses need to overwrite `request`"""

    sleep_interval: int

    def __init__(self, sleep_interval=0):
        self.sleep_interval = sleep_interval
        pass

    def request(self, system_msg: str, prompt: str) -> tuple[str, Any, dict]:
        """
        Implemented by subclasses to actually query the model

        Returns
        -------
        str: Response text
        Any: raw response object. Will be pickled and persisted
        dict: any additional information that should be persisted in the jsonl line
        """

        raise NotImplementedError()

    def __call__(self, prompt, prompt_id, values, fp, system_msg=""):
        """Prompt a model and store the output.

        Parameters
        ----------
        prompt : str
            prompt template
        prompt_id : str
            prompt id
        values : dict[str, str]
            book metadata
        fp : filepointer
            open filepointer for storing model outputs as jsonl
        system_msg : str, optional
            system message to be used in addition to prompt, by default ""
        """
        try:
            final_prompt = prompt.format(**values)
            print(final_prompt)
        except KeyError:
            # print("SKIPPED ", prompt)
            return "skip"

        try:
            out_text, out_obj, additional_info = self.request(system_msg, final_prompt)
        except Exception as e:
            print(f"ERROR: {e.__class__}: {e}")
            return "error"

        s = json.dumps(
            {
                "timestamp": timestamp(),
                "prompt_id": prompt_id,
                "prompt": prompt,
                "output": out_text,
                "final_prompt": final_prompt,
                "values": values,
                "raw_output": b64encode(pickle.dumps(out_obj)).decode("ascii"),
                **additional_info,
            }
        )
        fp.write(s)
        fp.write("\n")
        fp.flush()

        time.sleep(self.sleep_interval)

        return "performed"


class AlephAlpha(LLM):
    def __init__(
        self,
        aleph_alpha_api_key,
        model,
        temperature=None,
        sleep_interval=1,
        max_tokens=None,
        **kwargs,
    ):
        super().__init__(sleep_interval=sleep_interval)

        if temperature is None:
            temperature = 0.8

        self.llm = aleph_alpha_client.Client(token=aleph_alpha_api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def request(self, system_msg: str, prompt: str) -> tuple[str, Any, dict]:
        if system_msg:
            prompt = system_msg + "\n\n" + prompt

        request = aleph_alpha_client.CompletionRequest(
            prompt=aleph_alpha_client.Prompt.from_text(prompt),
            maximum_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        response = self.llm.complete(
            request,
            model=self.model,
        )

        out = response.completions[0].completion

        return out, response, {}


class TogetherOpenAI(LLM):
    """
    We use this version of the Together API for Alpaca, because the default one had issues with the instruction wrapping of Alpaca.
    """

    def __init__(
        self,
        together_api_key,
        model,
        temperature=None,
        sleep_interval=1,
        max_tokens=None,
        **kwargs,
    ):
        super().__init__(sleep_interval=sleep_interval)

        if temperature is None:
            temperature = 0.7

        self.client = openai.OpenAI(
            api_key=together_api_key,
            base_url="https://api.together.xyz/v1",
        )

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def request(self, system_msg: str, prompt: str) -> tuple[str, Any, dict]:

        response = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        out = response.choices[0].message.content

        return out, response, {}


class OpenAI(LLM):
    def __init__(
        self,
        openai_api_key,
        model,
        temperature=None,
        sleep_interval=1,
        max_tokens=None,
        **kwargs,
    ):
        super().__init__(sleep_interval=sleep_interval)

        if temperature is None:
            temperature = 0.7

        self.client = openai.OpenAI(
            api_key=openai_api_key,
        )

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def request(self, system_msg: str, prompt: str) -> tuple[str, Any, dict]:

        response = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        out = response.choices[0].message.content

        return out, response, {}


def main(model_name, prompt, system_msg="", temp=None):

    model = get_model(model_name, temp=temp)

    ret = model.request(system_msg, prompt)

    pprint(ret)


if __name__ == "__main__":
    Fire(main)
