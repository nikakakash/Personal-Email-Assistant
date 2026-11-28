"""Model prices (USD per million tokens) and the cost of a call."""

import logging
from dataclasses import dataclass
from decimal import Decimal

from app.llm.client import Usage

logger = logging.getLogger(__name__)
_MILLION = Decimal(1_000_000)
_QUANTUM = Decimal("0.000001")


@dataclass(frozen=True)
class ModelPrice:
    input: Decimal
    output: Decimal
    cache_read: Decimal
    cache_write: Decimal


def _p(i: str, o: str, cr: str, cw: str) -> ModelPrice:
    return ModelPrice(Decimal(i), Decimal(o), Decimal(cr), Decimal(cw))


# TODO(before release): re-verify the cache rates against the live pricing page.
PRICES: dict[str, ModelPrice] = {
    "claude-opus-5-5": _p("4", "20", "0.20", "5.00"),
    "claude-sonnet-5-5": _p("2", "10", "0.10", "2.50"),
    "claude-haiku-5-5": _p("0.10", "0.50", "0.01", "0.125"),
    # Served by server-side fallback ("default") when the requested model declines.
    "claude-opus-4-8": _p("5", "25", "0.50", "6.25"),
    # Embeddings: input tokens only. TODO(before release): re-verify against Voyage's pricing.
    "voyage-3.5-lite": _p("0.02", "0", "0", "0"),
}

# Fail closed: a model we have no price for is charged at the highest known rate of each kind.
_MAX_PRICE = ModelPrice(
    input=max(p.input for p in PRICES.values()),
    output=max(p.output for p in PRICES.values()),
    cache_read=max(p.cache_read for p in PRICES.values()),
    cache_write=max(p.cache_write for p in PRICES.values()),
)


def price_for(model: str) -> ModelPrice:
    price = PRICES.get(model)
    if price is None:
        logger.warning("no price for model %r; charging the highest known rates", model)
        return _MAX_PRICE
    return price


def cost_of(model: str, usage: Usage) -> Decimal:
    """Cost in USD, quantized to 6 places. Unknown models are charged at the highest rates."""
    price = price_for(model)
    total = (
        usage.input_tokens * price.input
        + usage.output_tokens * price.output
        + usage.cache_read_tokens * price.cache_read
        + usage.cache_write_tokens * price.cache_write
    ) / _MILLION
    return total.quantize(_QUANTUM)
