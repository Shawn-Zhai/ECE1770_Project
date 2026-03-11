# llm_client.py
from openai import OpenAI
from typing import Union

def create_llm_client(backend, base_url) -> OpenAI:
    """
    Create an OpenAI client that talks either to:
      - the real OpenAI API (BACKEND = 'openai'), or
      - a local LLM OpenAI-compatible server (BACKEND = 'gpt-nano').

    Usage is identical in the rest of the code:
      client.chat.completions.create(...)
    """
    if backend == "gpt-nano": #try different LLM
        # other API option
        return OpenAI(
            base_url=base_url,
            api_key="EMPTY",
        )
    else:
        # OpenAI cloud – base_url default, key from env OPENAI_API_KEY
        return OpenAI()
