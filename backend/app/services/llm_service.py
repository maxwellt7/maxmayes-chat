"""Thin shared adapters for OpenAI and Anthropic.

Centralizes API-call patterns and key checks so orchestrator nodes and agent
helpers don't duplicate boilerplate. Streaming Anthropic responses go through
``get_anthropic_client()``; the caller manages the stream context.
"""
from __future__ import annotations

import json
import logging

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from app.config import settings

_log = logging.getLogger(__name__)


def _require_openai_key() -> str:
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is not configured")
    return settings.openai_api_key


def _require_anthropic_key() -> str:
    if not settings.anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY is not configured")
    return settings.anthropic_api_key


async def call_openai_text(
    model: str, system: str, user: str, temperature: float = 0.0
) -> str:
    """Plain text completion via OpenAI chat completions API."""
    client = AsyncOpenAI(api_key=_require_openai_key())
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


async def call_openai_json(
    model: str, system: str, user: str, temperature: float = 0.0
) -> dict:
    """JSON-mode completion via OpenAI chat completions API.

    Returns the parsed dict. Raises ``json.JSONDecodeError`` if the model
    returns malformed JSON despite the response_format hint.
    """
    client = AsyncOpenAI(api_key=_require_openai_key())
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def get_anthropic_client() -> AsyncAnthropic:
    """Return an AsyncAnthropic client. Caller manages streaming context."""
    return AsyncAnthropic(api_key=_require_anthropic_key())
