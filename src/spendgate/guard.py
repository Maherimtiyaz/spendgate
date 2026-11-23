"""Guard: the public entry point that wraps provider calls in reserve -> settle."""
from __future__ import annotations

import asyncio
import contextvars
import logging
import time
from dataclasses import asdict, dataclass, replace
from functools import partial
from typing import Any, Callable

from .adapters import Adapter, StreamTracker, adapter_for, estimate_input_tokens
from .errors import UnknownModelError
from .run import Run
from .ledger import (BudgetKey, BudgetStatus, InMemoryStore, LedgerStore,
                     Reservation)
from .money import usd_to_micros
from .pricing import PriceRegistry, Usage

log = logging.getLogger("spendgate")

_last_event: contextvars.ContextVar["UsageEvent | None"] = contextvars.ContextVar(
    "spendgate_last_event", default=None)


@dataclass(frozen=True)
class GuardReservation:
    """A granted hold on budget for one call that is about to be made."""

    id: str
    tenant: str
    feature: str | None
    model: str
    provider: str | None
    amount_micros: int
    # Prompt size the hold was sized for; reused if usage never arrives.
    input_tokens: int = 0
    # The agent run this call belongs to, if any.
    run: Run | None = None


@dataclass(frozen=True)
class UsageEvent:
    """One metered call. ``id`` is unique per call: use it as the idempotency
    key when forwarding the event to a billing system."""

    id: str
    tenant: str
    feature: str | None
    provider: str | None
    model: str
    usage: Usage
    cost_micros: int
    reserved_micros: int
    # True when the provider's final usage never arrived (no usage in the
    # response, or a stream cut short) and the cost was estimated instead.
    estimated: bool
    timestamp: float
    run_id: str | None = None

    @property
    def cost_usd(self) -> float:
        return self.cost_micros / 1e6

    def to_dict(self) -> dict[str, Any]:
        """A JSON-safe form, as stored in the outbox."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UsageEvent":
        return cls(**{**data, "usage": Usage(**data["usage"])})


class Guard:
    """Meters LLM calls per tenant and refuses calls that would break a budget.

    ``count_tokens`` chooses how prompt size is measured before a call:
    ``"estimate"`` (offline, leans high) or ``"api"`` (the provider's token
    counter where one exists; exact, but one extra request per call).
    """

    def __init__(
        self,
        store: LedgerStore | None = None,
        prices: PriceRegistry | None = None,
        *,
        count_tokens: str = "estimate",
        default_max_output_tokens: int = 4096,
        reservation_ttl: float = 600.0,
        outbox: bool = False,
        on_usage: Callable[[UsageEvent], None] | None = None,
        on_soft_limit: Callable[[BudgetStatus], None] | None = None,
        clock: Callable[[], float] = time.time,
    ):
        if count_tokens not in ("estimate", "api"):
            raise ValueError("count_tokens must be 'estimate' or 'api'")
        self.store: LedgerStore = store if store is not None else InMemoryStore(clock=clock)
        self.prices = prices if prices is not None else PriceRegistry.bundled()
        self.count_tokens = count_tokens
        self.default_max_output_tokens = default_max_output_tokens
        self.reservation_ttl = reservation_ttl
        # Store every usage event durably, in the same transaction that books
        # its cost, for a Dispatcher to deliver to billing sinks.
        self.outbox = outbox
        self.on_usage = on_usage
        self.on_soft_limit = on_soft_limit
        self._clock = clock

    # -- budgets ----------------------------------------------------------

    def set_budget(
        self,
        tenant: str,
        *,
        feature: str | None = None,
        daily_usd: float | str | None = None,
        monthly_usd: float | str | None = None,
        total_usd: float | str | None = None,
        soft_pct: float | None = None,
    ) -> None:
        """Set hard limits for a tenant, or for one feature of a tenant.

        A call must fit in every budget that applies to it: the tenant-wide
        ones and, if it names a feature, that feature's. Passing ``None`` for
        a window removes that window's limit. ``soft_pct`` (0-100) fires
        ``on_soft_limit`` once per period when spend crosses that share.
        """
        if soft_pct is not None and not 0 < soft_pct <= 100:
            raise ValueError("soft_pct must be in (0, 100]")
        for window, usd in (("day", daily_usd), ("month", monthly_usd), ("total", total_usd)):
            key = BudgetKey(tenant, feature, window)
            if usd is None:
                self.store.remove_budget(key)
                continue
            limit = usd_to_micros(usd)
            soft = None if soft_pct is None else int(limit * soft_pct / 100)
            self.store.set_budget(key, limit, soft)

    def status(self, tenant: str, feature: str | None = None) -> list[BudgetStatus]:
        return self.store.status(tenant, feature)

    @property
    def last_event(self) -> UsageEvent | None:
        """The usage event of the most recent ``chat``/``achat`` call made in
        the current thread or asyncio task: what the call you just made cost."""
        return _last_event.get()

    # -- low-level API: wrap any call yourself ----------------------------

    def reserve(
        self,
        tenant: str,
        model: str,
        *,
        input_tokens: int,
        max_output_tokens: int,
        feature: str | None = None,
        cache_write: bool = False,
        provider: str | None = None,
        run: Run | None = None,
    ) -> GuardReservation:
        """Hold the worst-case cost of a call. Raises ``BudgetExceeded`` if it
        does not fit. Follow with ``settle`` (call finished) or ``release``
        (call never reached the provider)."""
        amount = self.prices.worst_case_micros(
            model, input_tokens, max_output_tokens, cache_write=cache_write)
        if run is not None:
            run._admit(amount)  # the run's own ceilings come first
        try:
            held: Reservation = self.store.reserve(tenant, feature, amount,
                                                   self.reservation_ttl)
        except BaseException:
            if run is not None:
                run._refused(amount)
            raise
        if run is not None:
            run._holding(held.id, amount)
        return GuardReservation(held.id, tenant, feature, model, provider, amount,
                                input_tokens, run)

    def settle(self, reservation: GuardReservation, usage: Usage | None,
               *, model: str | None = None, estimated: bool = False) -> UsageEvent:
        """Book the real cost of a finished call and free the rest of the hold.

        ``model`` is the model that actually served the call, when it can
        differ from the one requested (provider-side fallbacks). With
        ``usage=None`` the full reservation is booked. Pass ``estimated=True``
        when ``usage`` is a guess; the cost is then capped at the reservation.
        """
        billed_model = reservation.model
        if model and model != reservation.model:
            try:
                self.prices.get(model)
                billed_model = model
            except UnknownModelError:
                log.warning("served model %r has no price; billing as %r",
                            model, reservation.model)
        if usage is None:
            usage, cost, estimated = Usage(), reservation.amount_micros, True
        else:
            cost = self.prices.cost_micros(billed_model, usage)
            if estimated:
                cost = min(cost, reservation.amount_micros)

        run = reservation.run
        event = UsageEvent(
            id=reservation.id, tenant=reservation.tenant, feature=reservation.feature,
            provider=reservation.provider, model=billed_model, usage=usage,
            cost_micros=cost, reserved_micros=reservation.amount_micros,
            estimated=estimated, timestamp=self._clock(),
            run_id=run.id if run is not None else None,
        )
        result = self.store.settle(reservation.id, cost,
                                   event.to_dict() if self.outbox else None)
        if not result.applied:  # repeated settle: already booked and reported
            return replace(event, cost_micros=result.actual_micros)
        if run is not None:
            run._settled(reservation.id, cost)
        if result.overage_micros:
            log.warning("call %s cost %d micros, %d over its reservation",
                        reservation.id, result.actual_micros, result.overage_micros)
        if self.on_usage is not None:
            self._notify(self.on_usage, event)
        if self.on_soft_limit is not None:
            for status in result.soft_crossed:
                self._notify(self.on_soft_limit, status)
        return event

    def release(self, reservation: GuardReservation) -> None:
        self.store.release(reservation.id)
        if reservation.run is not None:
            reservation.run._released(reservation.id)

    def run(self, tenant: str, *, feature: str | None = None, max_usd: float | str | None = None,
            max_steps: int | None = None, loop_repeats: int | None = 3,
            loop_window: int = 10, run_id: str | None = None) -> Run:
        """Start an agent run with its own ceilings, on top of the tenant's budgets.

        ``max_usd`` caps what the whole run may cost and ``max_steps`` how many
        model calls it may make. The run also stops with ``LoopDetected`` once
        the model has requested the same tool call with the same arguments
        ``loop_repeats`` times within its last ``loop_window`` tool calls;
        pass ``loop_repeats=None`` to turn that off.
        """
        return Run(self, tenant, feature=feature, max_usd=max_usd, max_steps=max_steps,
                   loop_repeats=loop_repeats, loop_window=loop_window, run_id=run_id)

    # -- high-level API: wrap an SDK client call --------------------------

    def chat(self, client: Any, *, tenant: str, feature: str | None = None,
             api: str | None = None, adapter: Adapter | None = None,
             run: Run | None = None, **request: Any) -> Any:
        """Make one metered call with a synchronous OpenAI or Anthropic client.

        ``request`` is passed to the SDK unchanged, except that an output cap
        is added when the request has none. Returns the SDK's response, or
        with ``stream=True`` a ``MeteredStream`` that yields the SDK's events
        and settles the call when the stream ends or is closed.
        """
        adapter = adapter or adapter_for(client, api)
        prepared, tracker = self._prepare(adapter, request)
        input_tokens = None
        if self.count_tokens == "api":
            counted = adapter.count_input_tokens(client, prepared.request)
            if counted is not None:
                input_tokens = adapter.counted(counted)
        reservation = self._reserve_prepared(adapter, prepared, tenant, feature,
                                             input_tokens, run)
        try:
            response = adapter.invoke(client, prepared.request)
        except BaseException:
            self.release(reservation)
            raise
        if tracker is not None:
            return MeteredStream(self, reservation, response, tracker)
        _last_event.set(self._settle_response(adapter, reservation, response))
        return response

    async def achat(self, client: Any, *, tenant: str, feature: str | None = None,
                    api: str | None = None, adapter: Adapter | None = None,
                    run: Run | None = None, **request: Any) -> Any:
        """``chat`` for asynchronous SDK clients. With ``stream=True`` it
        returns an ``AsyncMeteredStream``."""
        adapter = adapter or adapter_for(client, api)
        prepared, tracker = self._prepare(adapter, request)
        input_tokens = None
        if self.count_tokens == "api":
            pending = adapter.count_input_tokens(client, prepared.request)
            if pending is not None:
                input_tokens = adapter.counted(await pending)
        reservation = await self._off_loop(
            self._reserve_prepared, adapter, prepared, tenant, feature, input_tokens, run)
        try:
            response = await adapter.invoke(client, prepared.request)
        except BaseException:
            await self._off_loop(self.release, reservation)
            raise
        if tracker is not None:
            return AsyncMeteredStream(self, reservation, response, tracker)
        _last_event.set(
            await self._off_loop(self._settle_response, adapter, reservation, response))
        return response

    # -- internals --------------------------------------------------------

    def _prepare(self, adapter: Adapter, request: dict[str, Any]):
        if "model" not in request:
            raise TypeError("request needs a 'model'")
        prepared = adapter.prepare(request, self.default_max_output_tokens)
        if not request.get("stream"):
            return prepared, None
        to_send, tracker = adapter.start_stream(prepared.request)
        return replace(prepared, request=to_send), tracker

    async def _off_loop(self, fn: Callable[..., Any], *args: Any) -> Any:
        """Run a ledger operation without blocking the event loop."""
        if getattr(self.store, "blocking", False):
            return await asyncio.to_thread(partial(fn, *args))
        return fn(*args)

    def _settle_stream(self, reservation: GuardReservation,
                       tracker: StreamTracker) -> UsageEvent:
        self._note_tools(reservation, tracker.tool_calls)
        usage = tracker.usage
        if tracker.complete and usage is not None:
            return self.settle(reservation, usage, model=tracker.model)
        # The stream stopped before the provider's final usage arrived. Bill
        # what is known and estimate the rest, leaning high: the prompt at the
        # size it was reserved for, the output from the text received so far.
        base = usage if usage is not None else Usage(input_tokens=reservation.input_tokens)
        guessed = tracker.output_bytes // 3 + 1 if tracker.output_bytes else 0
        partial_usage = replace(base, output_tokens=max(base.output_tokens, guessed))
        return self.settle(reservation, partial_usage, model=tracker.model, estimated=True)

    def _reserve_prepared(self, adapter: Adapter, prepared, tenant: str,
                          feature: str | None, input_tokens: int | None,
                          run: Run | None = None) -> GuardReservation:
        if input_tokens is None:
            input_tokens = estimate_input_tokens(prepared.request)
        return self.reserve(
            tenant, prepared.request["model"], input_tokens=input_tokens,
            max_output_tokens=prepared.max_output_tokens, feature=feature,
            cache_write=prepared.cache_write, provider=adapter.provider, run=run,
        )

    def _settle_response(self, adapter: Adapter, reservation: GuardReservation,
                         response: Any) -> UsageEvent:
        try:
            usage = adapter.usage(response)
        except Exception:
            log.exception("could not read usage from the response; booking the reservation")
            usage = None
        event = self.settle(reservation, usage, model=getattr(response, "model", None))
        self._note_tools(reservation, lambda: adapter.tool_calls(response))
        return event

    @staticmethod
    def _note_tools(reservation: GuardReservation, read: Callable[[], Any]) -> None:
        if reservation.run is None:
            return
        try:
            calls = read()
        except Exception:
            log.exception("could not read tool calls for loop detection")
            return
        reservation.run._record_tools(calls)

    @staticmethod
    def _notify(callback: Callable[[Any], None], payload: Any) -> None:
        # The call already happened and is booked; a failing listener must
        # not turn a successful response into an error.
        try:
            callback(payload)
        except Exception:
            log.exception("spendgate callback %r failed", callback)


class _MeteredBase:
    def __init__(self, guard: Guard, reservation: GuardReservation, stream: Any,
                 tracker: StreamTracker):
        self._guard = guard
        self._reservation = reservation
        self._stream = stream
        self._tracker = tracker
        self._settled = False
        #: The ``UsageEvent`` for this call, once the stream has been settled.
        self.usage_event: UsageEvent | None = None

    def _observe(self, item: Any) -> bool:
        try:
            return self._tracker.observe(item)
        except Exception:  # never let metering break the caller's stream
            log.exception("could not read a stream event for metering")
            return True

    def _settle(self) -> None:
        if self._settled:
            return
        self._settled = True
        self.usage_event = self._guard._settle_stream(self._reservation, self._tracker)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class MeteredStream(_MeteredBase):
    """A streamed response that settles its cost when it ends.

    Iterate it exactly like the SDK stream it wraps. The call is settled when
    the stream is exhausted, raises, or is closed; use it as a context manager
    or call ``close()`` if you stop reading early.
    """

    def __iter__(self) -> "MeteredStream":
        return self

    def __next__(self) -> Any:
        if not hasattr(self, "_iterator"):
            self._iterator = iter(self._stream)
        while True:
            try:
                item = next(self._iterator)
            except BaseException:  # includes StopIteration: the stream is over
                self._settle()
                raise
            if self._observe(item):
                return item

    def close(self) -> None:
        try:
            close = getattr(self._stream, "close", None)
            if close is not None:
                close()
        finally:
            self._settle()

    def __enter__(self) -> "MeteredStream":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __del__(self) -> None:
        # Last resort for a stream that was dropped without being closed.
        try:
            self._settle()
        except Exception:
            pass


class AsyncMeteredStream(_MeteredBase):
    """``MeteredStream`` for asynchronous clients: ``async for`` / ``async with``."""

    def __aiter__(self) -> "AsyncMeteredStream":
        return self

    async def __anext__(self) -> Any:
        if not hasattr(self, "_iterator"):
            self._iterator = self._stream.__aiter__()
        while True:
            try:
                item = await self._iterator.__anext__()
            except BaseException:  # includes StopAsyncIteration and cancellation
                await self._asettle()
                raise
            if self._observe(item):
                return item

    async def _asettle(self) -> None:
        if not self._settled:
            await self._guard._off_loop(self._settle)

    async def close(self) -> None:
        try:
            close = getattr(self._stream, "close", None)
            if close is not None:
                await close()
        finally:
            await self._asettle()

    aclose = close

    async def __aenter__(self) -> "AsyncMeteredStream":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()
