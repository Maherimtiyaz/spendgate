from __future__ import annotations


class SpendgateError(Exception):
    """Base class for every error this library raises."""


class UnknownModelError(SpendgateError):
    """No price is registered for the model, so its cost cannot be bounded."""

    def __init__(self, model: str):
        super().__init__(
            f"no price registered for model {model!r}; add one with PriceRegistry.register()"
        )
        self.model = model


class BudgetExceeded(SpendgateError):
    """The call was refused: its worst-case cost does not fit in a budget."""

    def __init__(self, *, tenant, feature, window, period, limit_micros,
                 spent_micros, reserved_micros, requested_micros):
        self.tenant = tenant
        self.feature = feature
        self.window = window
        self.period = period
        self.limit_micros = limit_micros
        self.spent_micros = spent_micros
        self.reserved_micros = reserved_micros
        self.requested_micros = requested_micros
        scope = tenant if feature is None else f"{tenant}/{feature}"
        left = limit_micros - spent_micros - reserved_micros
        super().__init__(
            f"{window} budget for {scope} exceeded: call needs up to "
            f"${requested_micros / 1e6:.6f}, ${max(left, 0) / 1e6:.6f} left of "
            f"${limit_micros / 1e6:.2f} (period {period})"
        )

    @property
    def remaining_micros(self) -> int:
        return max(self.limit_micros - self.spent_micros - self.reserved_micros, 0)


class RunLimitExceeded(SpendgateError):
    """An agent run hit its own ceiling. ``kind`` is ``"cost"`` or ``"steps"``."""

    def __init__(self, *, run_id: str, kind: str, limit, used, requested=None):
        self.run_id = run_id
        self.kind = kind
        self.limit = limit
        self.used = used
        self.requested = requested
        if kind == "steps":
            message = f"run {run_id} reached its limit of {limit} model calls"
        else:
            message = (f"run {run_id} would exceed its ${limit / 1e6:.2f} ceiling: "
                       f"${used / 1e6:.6f} used or held, next call needs up to "
                       f"${requested / 1e6:.6f}")
        super().__init__(message)


class LoopDetected(SpendgateError):
    """The model asked for the same tool call, with the same arguments, too
    many times in a row of recent calls. ``response`` is the response that
    tripped the breaker, when the call was not streamed."""

    def __init__(self, *, run_id: str, tool: str, arguments: str, count: int, response=None):
        self.run_id = run_id
        self.tool = tool
        self.arguments = arguments
        self.count = count
        self.response = response
        super().__init__(
            f"run {run_id} is looping: tool {tool!r} was requested {count} times "
            f"with identical arguments")


class DeliveryError(SpendgateError):
    """A usage event could not be delivered to a sink; it will be retried."""


class PermanentDeliveryError(DeliveryError):
    """The sink rejected the event in a way retrying cannot fix."""
