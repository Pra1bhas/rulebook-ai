"""ONE place for every LLM call.

Why a wrapper? The rest of the code just calls `generate(...)`. To switch from
Gemini to Groq (or a paid model later) we change config, not 10 files.
It also handles free-tier rate limits (429) and "server busy" (503) by waiting and
retrying, then falls back from Gemini to Groq if a Groq key is set.
"""
import json
import logging
import time
from typing import TypeVar

import httpx
from pydantic import BaseModel

from app.config import settings

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


def generate(prompt: str, schema: type[T] | None = None) -> str | T:
    """Send a prompt. If `schema` is given, return a validated Pydantic object."""
    provider = settings.llm_provider
    for attempt in range(settings.llm_max_retries + 1):
        try:
            if provider == "gemini":
                return _gemini(prompt, schema)
            if provider == "groq":
                return _groq(prompt, schema)
            if provider == "fake":
                return _fake(prompt, schema)
            raise LLMError(f"Unknown llm_provider: {provider}")
        except RateLimited:
            if attempt == settings.llm_max_retries:
                break
            wait = 2**attempt * 2  # 2s, 4s, 8s, 16s ... (exponential backoff)
            log.warning("%s busy or rate limited, retrying in %ss", provider, wait)
            time.sleep(wait)
    # Gemini still busy after all retries: try the backup provider once.
    if provider == "gemini" and settings.groq_api_key:
        log.warning("Gemini unavailable, falling back to Groq")
        try:
            return _groq(prompt, schema)
        except RateLimited:
            pass
    raise LLMError("LLM is busy or rate limited. Wait a minute and retry.")


class RateLimited(Exception):
    """Temporary problem (429 rate limit, 503 overloaded): worth waiting and retrying."""


RETRYABLE_CODES = {429, 500, 503}


# ---------------- Gemini (Google AI Studio free tier) ----------------
_gemini_client = None


def _gemini(prompt: str, schema: type[T] | None) -> str | T:
    global _gemini_client
    from google import genai
    from google.genai import errors, types

    if not settings.gemini_api_key:
        raise LLMError("GEMINI_API_KEY is missing. Put it in your .env file.")
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=settings.gemini_api_key)

    config = types.GenerateContentConfig(temperature=0)
    if schema:
        # Structured output: Gemini must reply with JSON matching our schema.
        config = types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=schema,
        )
    try:
        resp = _gemini_client.models.generate_content(
            model=settings.gemini_model, contents=prompt, config=config
        )
    except errors.APIError as e:
        if e.code in RETRYABLE_CODES:
            raise RateLimited() from e
        raise LLMError(f"Gemini error {e.code}: {e.message}") from e

    if schema:
        return schema.model_validate_json(resp.text)
    return resp.text


# ---------------- Groq (free tier, OpenAI-compatible API) ----------------
def _groq(prompt: str, schema: type[T] | None) -> str | T:
    if not settings.groq_api_key:
        raise LLMError("GROQ_API_KEY is missing. Put it in your .env file.")
    messages = [{"role": "user", "content": prompt}]
    body: dict = {"model": settings.groq_model, "messages": messages, "temperature": 0}
    if schema:
        schema_text = json.dumps(schema.model_json_schema())
        messages.insert(
            0,
            {"role": "system", "content": f"Reply ONLY with JSON matching this schema: {schema_text}"},
        )
        body["response_format"] = {"type": "json_object"}

    r = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {settings.groq_api_key}"},
        json=body,
        timeout=60,
    )
    if r.status_code in RETRYABLE_CODES:
        raise RateLimited()
    if r.status_code != 200:
        raise LLMError(f"Groq error {r.status_code}: {r.text[:300]}")
    text = r.json()["choices"][0]["message"]["content"]
    return schema.model_validate_json(text) if schema else text


# ---------------- Fake (only for automated tests, no internet) ----------------
def _fake(prompt: str, schema: type[T] | None) -> str | T:
    if schema is None:
        # used by query rewriting: just echo the question back
        return prompt.rsplit("Question:", 1)[-1].strip()
    name = schema.__name__
    if name == "LLMQueries":
        return schema.model_validate({"queries": [prompt.rsplit("Problem:", 1)[-1].strip()]})
    if name == "LLMPlan":
        found = "[S1]" in prompt
        point = {"text": "Fake step based on S1.", "source_ids": ["S1"]}
        invented = {"text": "Fake point with an invented source.", "source_ids": ["S99"]}
        return schema.model_validate(
            {
                "found": found,
                "summary": "Fake summary." if found else "",
                "rights": [point, invented] if found else [],
                "steps": [point] if found else [],
                "escalation": [],
                "draft_letter": "Dear Sir/Madam, ..." if found else "",
                "missing_info": ["When did it happen?"] if found else [],
            }
        )
    # LLMAnswer: pretend the first source answers the question.
    if "[S1]" in prompt:
        return schema.model_validate(
            {"found": True, "answer": "Fake answer based on S1.", "source_ids": ["S1"]}
        )
    return schema.model_validate({"found": False, "answer": "", "source_ids": []})
