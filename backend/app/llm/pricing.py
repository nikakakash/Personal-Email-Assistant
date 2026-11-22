"""Model prices (USD per million tokens) and the cost of a call."""

from dataclasses import dataclass
from decimal import Decimal

from app.llm.client import Usage

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
}


def cost_of(model: str, usage: Usage) -> Decimal:
    """Cost in USD, quantized to 6 places. Unknown models raise KeyError (never guess a price)."""
    price = PRICES[model]
    total = (
        usage.input_tokens * price.input
        + usage.output_tokens * price.output
        + usage.cache_read_tokens * price.cache_read
        + usage.cache_write_tokens * price.cache_write
    ) / _MILLION
    return total.quantize(_QUANTUM)
