"""LLM access layer for the Mohafiz agent.

Cheap-first cascade (roadmap quota design): the 8B model does routing, grading,
and groundedness; the 70B only writes final answers. Keeping the cheap steps off
the 70B protects its scarce daily token budget (the W1 lesson).
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

load_dotenv(Path(__file__).parent.parent / ".env")

CHEAP_MODEL = "llama-3.1-8b-instant"      # routing / grading / groundedness
BIG_MODEL = "llama-3.3-70b-versatile"     # final answer generation

_client = None


def _groq() -> Groq:
    global _client
    if _client is None:
        _client = Groq()
    return _client


def chat(messages, model=CHEAP_MODEL, temperature=0.0, json_mode=False, max_tokens=1024) -> str:
    kwargs = {"model": model, "messages": messages, "temperature": temperature,
              "max_tokens": max_tokens}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = _groq().chat.completions.create(**kwargs)
    return resp.choices[0].message.content


def chat_json(messages, model=CHEAP_MODEL, max_tokens=1024) -> dict:
    """Chat with JSON output, tolerant of stray prose around the object."""
    raw = chat(messages, model=model, json_mode=True, max_tokens=max_tokens)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1:
            return json.loads(raw[start:end + 1])
        raise
