"""Run: ceilings for one agent run, so a runaway loop stops itself."""
from __future__ import annotations

import threading
import uuid
from collections import deque
from typing import TYPE_CHECKING, Any

from .adapters import adapter_for
from .errors import LoopDetected, RunLimitExceeded
from .money import usd_to_micros

if TYPE_CHECKING:
    from .guard import Guard, GuardReservation


class Run:
    """One agent run: a series of model calls that should not go on for ever.

    Create it with ``guard.run(...)`` and make the run's calls through it.
    Every call still has to fit the tenant's budgets; the run adds ceilings
    of its own and trips on repeated identical tool calls. Once a ceiling is
    hit or a loop is detected, every further call through the run raises.

    The ceilings live in this process's memory: a run is expected to be
    driven by one process. It is safe to use from several threads or tasks.
    """

    def __init__(self, guard: "Guard", tenant: str, *, feature: str | None = None,
                 max_usd: float | str | None = None, max_steps: int | None = None,
                 loop_repeats: int | None = 3, loop_window: int = 10,
                 run_id: str | None = None):
        if loop_repeats is not None and loop_repeats < 2:
            raise ValueError("loop_repeats must be at least 2, or None to disable")
        self.guard = guard
        self.tenant = tenant
        self.feature = feature
        self.id = run_id or uuid.uuid4().hex
        self.max_micros = None if max_usd is None else usd_to_micros(max_usd)
        self.max_steps = max_steps
        self.loop_repeats = loop_repeats
        #: Model calls admitted so far.
        self.steps = 0
        #: Cost booked so far, in micro-dollars.
        self.spent_micros = 0
        self._held: dict[str, int] = {}
        self._pending = 0  # admitted, hold not yet confirmed by the store
        self._recent: deque[tuple[str, str]] = deque(maxlen=max(loop_window, loop_repeats or 0))
        self._loop: dict[str, Any] | None = None
        self._lock = threading.Lock()

    @property
    def cost_usd(self) -> float:
        return self.spent_micros / 1e6

    # -- making calls -----------------------------------------------------

    def chat(self, client: Any, *, api: str | None = None, adapter: Any = None,
             **request: Any) -> Any:
        """``Guard.chat`` within this run's ceilings."""
        adapter = adapter or adapter_for(client, api)
        response = self.guard.chat(client, tenant=self.tenant, feature=self.feature,
                                   adapter=adapter, run=self, **request)
        if not request.get("stream"):
            self._raise_if_looping(response)
        return response

    async def achat(self, client: Any, *, api: str | None = None, adapter: Any = None,
                    **request: Any) -> Any:
        """``Guard.achat`` within this run's ceilings."""
        adapter = adapter or adapter_for(client, api)
        response = await self.guard.achat(client, tenant=self.tenant, feature=self.feature,
                                          adapter=adapter, run=self, **request)
        if not request.get("stream"):
            self._raise_if_looping(response)
        return response

    def reserve(self, model: str, *, input_tokens: int, max_output_tokens: int,
                cache_write: bool = False, provider: str | None = None) -> "GuardReservation":
        """``Guard.reserve`` within this run's ceilings, for calls you wrap yourself."""
        return self.guard.reserve(self.tenant, model, input_tokens=input_tokens,
                                  max_output_tokens=max_output_tokens, feature=self.feature,
                                  cache_write=cache_write, provider=provider, run=self)

    def record_tool_call(self, name: str, arguments: Any) -> None:
        """Feed loop detection yourself, for calls made outside ``chat``.
        Raises ``LoopDetected`` if this call completes a loop."""
        from .adapters.base import canonical_arguments
        self._record_tools([(name, canonical_arguments(arguments))])
        self._raise_if_looping()

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    async def __aenter__(self) -> "Run":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    # -- called by the guard ----------------------------------------------

    def _admit(self, amount: int) -> None:
        """Check the ceilings for a call about to be reserved, and count it."""
        with self._lock:
            self._raise_if_looping()
            if self.max_steps is not None and self.steps >= self.max_steps:
                raise RunLimitExceeded(run_id=self.id, kind="steps", limit=self.max_steps,
                                       used=self.steps)
            used = self.spent_micros + self._pending + sum(self._held.values())
            if self.max_micros is not None and used + amount > self.max_micros:
                raise RunLimitExceeded(run_id=self.id, kind="cost", limit=self.max_micros,
                                       used=used, requested=amount)
            self.steps += 1
            self._pending += amount

    def _refused(self, amount: int) -> None:
        """The tenant budget refused the call after the run admitted it."""
        with self._lock:
            self.steps -= 1
            self._pending -= amount

    def _holding(self, reservation_id: str, amount: int) -> None:
        with self._lock:
            self._pending -= amount
            self._held[reservation_id] = amount

    def _settled(self, reservation_id: str, cost: int) -> None:
        with self._lock:
            self._held.pop(reservation_id, None)
            self.spent_micros += cost

    def _released(self, reservation_id: str) -> None:
        """The call never reached the provider, so it is not a step."""
        with self._lock:
            if self._held.pop(reservation_id, None) is not None:
                self.steps -= 1

    def _record_tools(self, calls: list[tuple[str, str]]) -> None:
        if self.loop_repeats is None:
            return
        with self._lock:
            for call in calls:
                self._recent.append(call)
                count = self._recent.count(call)
                if count >= self.loop_repeats and self._loop is None:
                    self._loop = {"tool": call[0], "arguments": call[1], "count": count}

    def _raise_if_looping(self, response: Any = None) -> None:
        if self._loop is not None:
            raise LoopDetected(run_id=self.id, response=response, **self._loop)
