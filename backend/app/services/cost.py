"""Pre-flight cost estimation for a chat request.

The spend ceiling has to decide *before* the expensive call whether to allow it,
which means the estimate must be made without knowing the real token counts. So
this module deliberately over-estimates: it assumes the synthesizer emits its
full `max_tokens` budget and that retrieval returns full-size chunks. An
over-estimate makes the ceiling trip early, which is the safe direction to be
wrong in.

Prices are USD per million tokens, as published at the time of writing. They are
constants here rather than configuration because a stale price makes the ceiling
inaccurate rather than inoperative, and a wrong value in an env var would be
invisible.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

# Rough characters-per-token for English prose. Real tokenizers land near 4;
# using a smaller divisor would under-count and under-count is the unsafe
# direction, so 4 is used with the deliberate over-estimates below.
_CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class ModelPrice:
    input_per_mtok: Decimal
    output_per_mtok: Decimal


_PRICES: dict[str, ModelPrice] = {
    "gpt-4.1-mini": ModelPrice(Decimal("0.40"), Decimal("1.60")),
    "claude-sonnet-4-6": ModelPrice(Decimal("3.00"), Decimal("15.00")),
    # Fallback for anything unlisted: priced at the most expensive model we use,
    # so an unrecognised model name cannot slip under the ceiling.
    "__default__": ModelPrice(Decimal("3.00"), Decimal("15.00")),
}

# Fixed-size parts of the request that do not scale with the user's message:
# system prompts, the index catalog, the retrieved context window, and the voice
# profile. Measured generously from the current prompt templates.
_OPTIMIZER_OVERHEAD_TOKENS = 400
_ROUTER_OVERHEAD_TOKENS = 2_000
_VERIFIER_OVERHEAD_TOKENS = 6_000
_SYNTH_OVERHEAD_TOKENS = 8_000

# Output budgets. The first three are structured and short; the synthesizer is
# capped by `max_tokens` in the orchestrator and assumed to use all of it.
_OPTIMIZER_OUTPUT_TOKENS = 200
_ROUTER_OUTPUT_TOKENS = 400
_VERIFIER_OUTPUT_TOKENS = 400
_SYNTH_OUTPUT_TOKENS = 2_048

# Embedding and rerank calls are sub-cent at these volumes but not zero.
_RETRIEVAL_FIXED_USD = Decimal("0.002")


def _price(model: str) -> ModelPrice:
    return _PRICES.get(model, _PRICES["__default__"])


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _call_cost(model: str, input_tokens: int, output_tokens: int) -> Decimal:
    price = _price(model)
    million = Decimal(1_000_000)
    return (
        Decimal(input_tokens) * price.input_per_mtok / million
        + Decimal(output_tokens) * price.output_per_mtok / million
    )


def estimate_chat_request_usd(
    message: str,
    *,
    optimizer_model: str = "gpt-4.1-mini",
    router_model: str = "gpt-4.1-mini",
    verifier_model: str = "gpt-4.1-mini",
    synthesizer_model: str = "claude-sonnet-4-6",
) -> Decimal:
    """Upper-bound cost of running the full pipeline for `message`.

    The user's text is counted four times because it is passed to every stage,
    which is what makes a long message disproportionately expensive and is the
    behaviour the ceiling exists to bound.
    """
    message_tokens = estimate_tokens(message)

    total = _RETRIEVAL_FIXED_USD
    total += _call_cost(
        optimizer_model,
        _OPTIMIZER_OVERHEAD_TOKENS + message_tokens,
        _OPTIMIZER_OUTPUT_TOKENS,
    )
    total += _call_cost(
        router_model,
        _ROUTER_OVERHEAD_TOKENS + message_tokens,
        _ROUTER_OUTPUT_TOKENS,
    )
    total += _call_cost(
        verifier_model,
        _VERIFIER_OVERHEAD_TOKENS + message_tokens,
        _VERIFIER_OUTPUT_TOKENS,
    )
    total += _call_cost(
        synthesizer_model,
        _SYNTH_OVERHEAD_TOKENS + message_tokens,
        _SYNTH_OUTPUT_TOKENS,
    )
    return total.quantize(Decimal("0.000001"))
