"""Model price registry and cost arithmetic."""
from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from decimal import Decimal
from importlib import resources
from typing import Any, Mapping

from .errors import UnknownModelError
from .money import ceil_micros


@dataclass(frozen=True)
class Usage:
    """Provider-neutral token usage for one call.

    ``input_tokens`` is the uncached remainder only; the full prompt size is
    the sum of the four input-side fields. Reasoning/thinking tokens are part
    of ``output_tokens``, as both providers bill them.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_5m_tokens: int = 0
    cache_write_1h_tokens: int = 0

    @property
    def prompt_tokens(self) -> int:
        return (self.input_tokens + self.cache_read_tokens
                + self.cache_write_5m_tokens + self.cache_write_1h_tokens)


@dataclass(frozen=True)
class Rates:
    """USD per million tokens, which is also micro-dollars per token."""

    input: Decimal
    output: Decimal
    cache_read: Decimal
    cache_write_5m: Decimal
    cache_write_1h: Decimal

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "Rates":
        inp = Decimal(str(raw["input"]))
        return cls(
            input=inp,
            output=Decimal(str(raw["output"])),
            # A provider that lists no cache price bills those tokens as input.
            cache_read=Decimal(str(raw.get("cache_read", inp))),
            cache_write_5m=Decimal(str(raw.get("cache_write_5m", inp))),
            cache_write_1h=Decimal(str(raw.get("cache_write_1h", inp))),
        )


@dataclass(frozen=True)
class ModelPrice:
    base: Rates
    # Some models bill the whole request at a higher rate once the prompt
    # passes a size threshold.
    long_context_threshold: int | None = None
    long_context: Rates | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ModelPrice":
        long = raw.get("long_context")
        if long is None:
            return cls(base=Rates.from_mapping(raw))
        return cls(
            base=Rates.from_mapping(raw),
            long_context_threshold=int(long["threshold"]),
            long_context=Rates.from_mapping(long),
        )

    def rates_for(self, prompt_tokens: int) -> Rates:
        if (self.long_context is not None
                and self.long_context_threshold is not None
                and prompt_tokens > self.long_context_threshold):
            return self.long_context
        return self.base


# "gpt-5.4-2026-03-05", "claude-opus-4-5-20251101", "claude-opus-4-5@20251101"
_DATE_SUFFIX = re.compile(r"[-@]\d{4}-?\d{2}-?\d{2}$")
# Bedrock ("anthropic.claude-...") and router-style ("openai/gpt-...") prefixes.
_VENDOR_PREFIX = re.compile(r"^(?:[a-z0-9-]+\.)?(?:anthropic|openai)[./]")


class PriceRegistry:
    """Maps model IDs to prices. Unknown models raise instead of costing $0."""

    def __init__(self, prices: Mapping[str, ModelPrice] | None = None):
        self._prices: dict[str, ModelPrice] = dict(prices or {})
        self._lock = threading.Lock()

    @classmethod
    def bundled(cls) -> "PriceRegistry":
        text = resources.files("tenantbudget").joinpath("prices.json").read_text("utf-8")
        return cls.from_dict(json.loads(text)["models"])

    @classmethod
    def from_dict(cls, models: Mapping[str, Mapping[str, Any]]) -> "PriceRegistry":
        return cls({name: ModelPrice.from_mapping(raw) for name, raw in models.items()})

    def register(self, model: str, *, input: Any, output: Any, **extra: Any) -> None:
        price = ModelPrice.from_mapping({"input": input, "output": output, **extra})
        with self._lock:
            self._prices[model] = price

    def get(self, model: str) -> ModelPrice:
        price = self._prices.get(model)
        if price is not None:
            return price
        # Fall back only for spellings that are certainly the same model:
        # a vendor prefix and/or a dated snapshot suffix.
        bare = _VENDOR_PREFIX.sub("", model)
        for candidate in (bare, _DATE_SUFFIX.sub("", bare)):
            price = self._prices.get(candidate)
            if price is not None:
                return price
        raise UnknownModelError(model)

    def cost_micros(self, model: str, usage: Usage) -> int:
        """Actual cost of a finished call, rounded up to a whole micro-dollar."""
        rates = self.get(model).rates_for(usage.prompt_tokens)
        return ceil_micros(
            usage.input_tokens * rates.input
            + usage.cache_read_tokens * rates.cache_read
            + usage.cache_write_5m_tokens * rates.cache_write_5m
            + usage.cache_write_1h_tokens * rates.cache_write_1h
            + usage.output_tokens * rates.output
        )

    def worst_case_micros(self, model: str, input_tokens: int, max_output_tokens: int,
                          *, cache_write: bool = False) -> int:
        """Upper bound on the cost of a call that has not been made yet.

        Assumes no cache hits and a full ``max_output_tokens`` response. With
        ``cache_write`` set, the prompt is priced at the dearest cache-write
        rate, since writes cost more than plain input.
        """
        rates = self.get(model).rates_for(input_tokens)
        input_rate = rates.input
        if cache_write:
            input_rate = max(rates.input, rates.cache_write_5m, rates.cache_write_1h)
        return ceil_micros(input_tokens * input_rate + max_output_tokens * rates.output)
