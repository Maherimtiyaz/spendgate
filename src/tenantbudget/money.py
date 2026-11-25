"""Money is integer micro-dollars everywhere (1 USD = 1_000_000 micros).

A price of N USD per million tokens is exactly N micros per token, so token
costs need no unit conversion, only a final round-up to a whole micro.
"""
from __future__ import annotations

from decimal import ROUND_CEILING, ROUND_HALF_EVEN, Decimal

MICROS_PER_USD = 1_000_000


def usd_to_micros(usd: int | float | str | Decimal) -> int:
    value = Decimal(str(usd)) * MICROS_PER_USD
    return int(value.quantize(Decimal(1), rounding=ROUND_HALF_EVEN))


def micros_to_usd(micros: int) -> Decimal:
    return Decimal(micros) / MICROS_PER_USD


def ceil_micros(value: Decimal) -> int:
    return int(value.quantize(Decimal(1), rounding=ROUND_CEILING))
