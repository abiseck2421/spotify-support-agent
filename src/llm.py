"""Tiny shared Gemini helper (google.genai SDK): client setup + one call
with retry/backoff. Deliberately thin — batching/caching/JSON handling
live in the scripts that use it (label_train, build_golden, later days).
Loads GEMINI_API_KEY from .env (gitignored).
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

# The SDK logs a WARNING it doesn't recommend Models.generate_content for
# auto function calling on every plain call. We don't use AFC; silence it.
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

MODELS = ("gemini-3.6-flash", "gemini-3.5-flash", "gemini-2.5-flash")
CACHE: dict[tuple[str, str], str] = {}


class NoApiKeyError(RuntimeError):
    pass


def _client():
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise NoApiKeyError(
            "GEMINI_API_KEY is empty/missing. Put your key in .env, e.g. "
            "GEMINI_API_KEY=your-key-here (the file is gitignored)."
        )
    return genai.Client(api_key=key)


def generate(
    prompt: str,
    *,
    model: str | None = None,
    max_tokens: int = 4096,
    retries: int = 4,
) -> str:
    """One Gemini call with exponential backoff on quota/transient errors."""
    if model is None:
        model = pick_model()
    cache_key = (model, prompt)
    if cache_key in CACHE:
        return CACHE[cache_key]

    client = _client()
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2, max_output_tokens=max_tokens
                ),
            )
            out = (resp.text or "").strip()
            CACHE[cache_key] = out
            return out
        except NoApiKeyError:
            raise
        except Exception as e:  # noqa: BLE001 - surface after retries
            last_err = e
            time.sleep(backoff_sleep(e, attempt))
    raise RuntimeError(f"Gemini failed after {retries} retries: {last_err}") from last_err


def batch_generate(
    prompts: list[str],
    *,
    model: str | None = None,
    max_tokens: int = 8192,
    retries: int = 4,
    sleep_between: float = 0.0,
) -> list[str]:
    """Sequential generate() calls, one free-tier quota breath between batches."""
    outs: list[str] = []
    for i, p in enumerate(prompts):
        outs.append(generate(p, model=model, max_tokens=max_tokens, retries=retries))
        if sleep_between:
            time.sleep(sleep_between)
        if i % 5 == 4:
            time.sleep(1.0)
    return outs


_model: str | None = None


def pick_model() -> str:
    global _model
    if _model is not None:
        return _model
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise NoApiKeyError(
            "GEMINI_API_KEY is empty/missing. Put your key in .env, e.g. "
            "GEMINI_API_KEY=your-key-here (the file is gitignored)."
        )
    client = genai.Client(api_key=key)
    for name in MODELS:
        try:
            client.models.generate_content(model=name, contents="ping")
            _model = name
            print(f"[llm] using model: {name}")
            return name
        except Exception:  # noqa: BLE001 - model may not exist or be blocked
            continue
    raise RuntimeError("No usable Gemini model found (checked " + ", ".join(MODELS) + ")")


def jsonify(block: str) -> str:
    """Strip ```json fences if Gemini wrapped the answer in them."""
    s = block.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
    return s.strip()


def backoff_sleep(e: Exception, attempt: int) -> float:
    """Wait that respects the API's own rate-limit RTT estimate when given."""
    s = str(e)
    m = re.search(r"Please retry in (\d+(?:\.\d+)?)s?", s)
    if m:
        return float(m.group(1)) + 2.0   # API's retry delay + buffer
    if "429" in s or "RESOURCE_EXHAUSTED" in s or "QUOTA" in s.upper():
        return float(2 ** attempt) + 2.0
    return float(2 ** attempt)


__all__ = ["NoApiKeyError", "generate", "batch_generate", "jsonify", "pick_model"]