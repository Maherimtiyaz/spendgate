"""Per-tenant LLM spend metering and hard budgets, as a library."""
from .errors import (BudgetExceeded, DeliveryError, LoopDetected, PermanentDeliveryError,
                     RunLimitExceeded, SpendgateError, UnknownModelError)
from .guard import AsyncMeteredStream, Guard, GuardReservation, MeteredStream, UsageEvent
from .ledger import BudgetKey, BudgetStatus, InMemoryStore, LedgerStore, Sweeper
from .outbox import Dispatcher
from .money import micros_to_usd, usd_to_micros
from .pricing import ModelPrice, PriceRegistry, Usage
from .run import Run
from .sinks import CloudEventsSink, StripeMeterSink

__version__ = "0.1.0"

__all__ = [
    "Guard", "GuardReservation", "UsageEvent", "MeteredStream", "AsyncMeteredStream",
    "Run", "RunLimitExceeded", "LoopDetected",
    "Dispatcher", "CloudEventsSink", "StripeMeterSink", "DeliveryError", "PermanentDeliveryError",
    "BudgetExceeded", "SpendgateError", "UnknownModelError",
    "BudgetKey", "BudgetStatus", "InMemoryStore", "LedgerStore", "Sweeper",
    "ModelPrice", "PriceRegistry", "Usage",
    "micros_to_usd", "usd_to_micros",
]
