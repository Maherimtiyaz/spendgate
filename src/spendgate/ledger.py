"""Budget ledger: the reserve -> settle state machine.

A call first *reserves* its worst-case cost against every budget that applies
to it. If any budget cannot fit the reservation, nothing is reserved and the
call is refused. After the call, *settle* books the real cost and frees the
rest. A reservation that is never settled (the process died) expires after a
TTL and stops holding budget.

Invariant: for every budget, ``spent + reserved <= limit`` whenever a
reservation is granted, so concurrent callers cannot jointly overspend.
"""
from __future__ import annotations

import heapq
import logging
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Callable, Protocol

from .errors import BudgetExceeded

log = logging.getLogger("spendgate")

WINDOWS = ("day", "month", "total")


def period_for(window: str, now: float) -> str:
    """Identifier of the budget period containing ``now`` (UTC)."""
    if window == "total":
        return "all"
    moment = datetime.fromtimestamp(now, tz=timezone.utc)
    if window == "day":
        return moment.strftime("%Y-%m-%d")
    if window == "month":
        return moment.strftime("%Y-%m")
    raise ValueError(f"unknown budget window {window!r}; expected one of {WINDOWS}")


@dataclass(frozen=True)
class BudgetKey:
    tenant: str
    feature: str | None
    window: str


@dataclass(frozen=True)
class BudgetStatus:
    key: BudgetKey
    period: str
    limit_micros: int
    spent_micros: int
    reserved_micros: int
    soft_limit_micros: int | None = None

    @property
    def remaining_micros(self) -> int:
        return max(self.limit_micros - self.spent_micros - self.reserved_micros, 0)


@dataclass(frozen=True)
class Reservation:
    id: str
    tenant: str
    feature: str | None
    amount_micros: int
    expires_at: float


@dataclass(frozen=True)
class SettleResult:
    actual_micros: int
    reserved_micros: int
    # True when this call booked the cost; False for a repeated settle.
    applied: bool
    # Budgets whose soft limit this settle crossed.
    soft_crossed: tuple[BudgetStatus, ...] = ()

    @property
    def overage_micros(self) -> int:
        """How far the real cost ran past the reservation (0 in the normal case)."""
        return max(self.actual_micros - self.reserved_micros, 0)


@dataclass(frozen=True)
class OutboxItem:
    """A usage event waiting to be delivered to billing sinks."""

    id: str
    payload: dict
    # Names of the sinks that already have it.
    delivered: tuple[str, ...]
    # Delivery attempts so far, this one included.
    attempts: int


class LedgerStore(Protocol):
    def set_budget(self, key: BudgetKey, limit_micros: int,
                   soft_limit_micros: int | None = None) -> None: ...

    def remove_budget(self, key: BudgetKey) -> None: ...

    def reserve(self, tenant: str, feature: str | None, amount_micros: int,
                ttl_seconds: float) -> Reservation: ...

    def settle(self, reservation_id: str, actual_micros: int,
               event: dict | None = None) -> SettleResult:
        """Book the cost. ``event``, if given, is queued in the outbox as part
        of the same atomic step, so a booked cost always has its event."""

    def claim_events(self, limit: int, lease_seconds: float) -> list[OutboxItem]:
        """Take up to ``limit`` due events for delivery. A claimed event is
        hidden from other claimers until ``finish_event`` or the lease runs out."""

    def finish_event(self, event_id: str, *, delivered: tuple[str, ...], done: bool,
                     dead: bool = False, error: str | None = None,
                     retry_at: float | None = None) -> None: ...

    def outbox_stats(self) -> dict[str, int]:
        """Counts of ``pending`` and ``dead`` events."""

    def dead_events(self, limit: int = 100) -> list[tuple[OutboxItem, str | None]]:
        """Events that will not be retried, oldest first, each with its last error."""

    def retry_dead(self, event_ids: list[str] | None = None) -> int:
        """Put dead events (all of them, or the given IDs) back in the queue
        with a fresh set of attempts. Returns how many were revived."""

    def release(self, reservation_id: str) -> None: ...

    def extend(self, reservation_id: str, ttl_seconds: float) -> bool:
        """Push a live hold's expiry to ``ttl_seconds`` from now. Returns
        False if the hold is gone (settled, released or already expired)."""

    def sweep(self) -> int: ...

    def prune(self, older_than_seconds: float) -> int:
        """Delete records of reservations settled or expired longer ago than
        this. Returns how many were deleted. Budgets and spend are unaffected."""

    def status(self, tenant: str, feature: str | None = None) -> list[BudgetStatus]: ...


@dataclass
class _Config:
    limit: int
    soft: int | None


@dataclass
class _State:
    spent: int = 0
    reserved: int = 0
    soft_fired: bool = False


@dataclass
class _Held:
    reservation: Reservation
    # (budget key, period) pairs this reservation was booked against.
    entries: tuple[tuple[BudgetKey, str], ...]
    holding: bool = True  # False once expired: no longer counted in `reserved`


@dataclass
class _Queued:
    payload: dict
    seq: int
    delivered: tuple[str, ...] = ()
    attempts: int = 0
    next_attempt_at: float = 0.0
    locked_until: float = 0.0
    dead: bool = False
    error: str | None = None


@dataclass(order=True)
class _Expiry:
    at: float
    reservation_id: str = field(compare=False)


class Sweeper:
    """Background thread that expires overdue reservations on a schedule.

    Stores already expire overdue holds when a tenant's budget is checked, so
    budgets stay correct without this. Run a sweeper when you also want holds
    left by crashed workers cleared promptly for tenants that have gone quiet.
    """

    def __init__(self, store: LedgerStore, interval: float = 30.0):
        self._store = store
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> "Sweeper":
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="spendgate-sweeper",
                                            daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self._store.sweep()
            except Exception:
                log.exception("sweep failed; will retry")

    def __enter__(self) -> "Sweeper":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


class InMemoryStore:
    """Thread-safe, single-process ledger. For tests, dev and single workers.

    Budgets live in one process's memory, so several workers each get their
    own copy of the budget. Use a shared store for multi-process deployments.
    """

    # Settled and expired reservations are remembered so that a repeated or
    # late settle is handled correctly; the memory is bounded.
    _HISTORY = 50_000

    def __init__(self, clock: Callable[[], float] = time.time):
        self._clock = clock
        self._lock = threading.Lock()
        self._budgets: dict[BudgetKey, _Config] = {}
        self._state: dict[tuple[BudgetKey, str], _State] = {}
        self._held: dict[str, _Held] = {}
        self._expiries: list[_Expiry] = []
        self._expired: OrderedDict[str, _Held] = OrderedDict()
        self._settled: OrderedDict[str, SettleResult] = OrderedDict()
        self._outbox: dict[str, _Queued] = {}
        self._outbox_seq = 0

    # -- configuration ----------------------------------------------------

    def set_budget(self, key: BudgetKey, limit_micros: int,
                   soft_limit_micros: int | None = None) -> None:
        if key.window not in WINDOWS:
            raise ValueError(f"unknown budget window {key.window!r}; expected one of {WINDOWS}")
        if limit_micros < 0:
            raise ValueError("budget limit must not be negative")
        with self._lock:
            self._budgets[key] = _Config(limit_micros, soft_limit_micros)

    def remove_budget(self, key: BudgetKey) -> None:
        with self._lock:
            self._budgets.pop(key, None)

    # -- reserve / settle / release ---------------------------------------

    def reserve(self, tenant: str, feature: str | None, amount_micros: int,
                ttl_seconds: float) -> Reservation:
        if amount_micros < 0:
            raise ValueError("reservation amount must not be negative")
        with self._lock:
            now = self._clock()
            self._expire_due(now)
            entries = self._applicable(tenant, feature, now)
            # Check every budget before touching any, so a refusal leaves no trace.
            for key, period in entries:
                config = self._budgets[key]
                state = self._state.get((key, period)) or _State()
                if state.spent + state.reserved + amount_micros > config.limit:
                    raise BudgetExceeded(
                        tenant=key.tenant, feature=key.feature, window=key.window,
                        period=period, limit_micros=config.limit,
                        spent_micros=state.spent, reserved_micros=state.reserved,
                        requested_micros=amount_micros,
                    )
            for entry in entries:
                self._state.setdefault(entry, _State()).reserved += amount_micros
            reservation = Reservation(
                id=uuid.uuid4().hex, tenant=tenant, feature=feature,
                amount_micros=amount_micros, expires_at=now + ttl_seconds,
            )
            self._held[reservation.id] = _Held(reservation, tuple(entries))
            heapq.heappush(self._expiries, _Expiry(reservation.expires_at, reservation.id))
            return reservation

    def settle(self, reservation_id: str, actual_micros: int,
               event: dict | None = None) -> SettleResult:
        if actual_micros < 0:
            raise ValueError("actual cost must not be negative")
        with self._lock:
            done = self._settled.get(reservation_id)
            if done is not None:
                return SettleResult(done.actual_micros, done.reserved_micros, applied=False)
            # A reservation that already expired is still settled: the money
            # was spent, so it must be booked even though the hold is gone.
            held = self._held.pop(reservation_id, None) or self._expired.pop(reservation_id, None)
            if held is None:
                raise KeyError(f"unknown reservation {reservation_id!r}")
            crossed: list[BudgetStatus] = []
            for entry in held.entries:
                key, period = entry
                state = self._state.setdefault(entry, _State())
                if held.holding:
                    state.reserved -= held.reservation.amount_micros
                state.spent += actual_micros
                config = self._budgets.get(key)
                if (config is not None and config.soft is not None
                        and not state.soft_fired and state.spent >= config.soft):
                    state.soft_fired = True
                    crossed.append(self._status(key, period, config, state))
            result = SettleResult(actual_micros, held.reservation.amount_micros,
                                  applied=True, soft_crossed=tuple(crossed))
            self._remember(self._settled, reservation_id, result)
            if event is not None:
                self._outbox_seq += 1
                self._outbox[reservation_id] = _Queued(event, self._outbox_seq)
            return result

    def release(self, reservation_id: str) -> None:
        """Drop a reservation without booking any cost (the call never ran)."""
        with self._lock:
            held = self._held.pop(reservation_id, None)
            if held is None:
                self._expired.pop(reservation_id, None)
                return
            self._unhold(held)

    def extend(self, reservation_id: str, ttl_seconds: float) -> bool:
        with self._lock:
            now = self._clock()
            self._expire_due(now)
            held = self._held.get(reservation_id)
            if held is None:
                return False
            held.reservation = replace(held.reservation, expires_at=now + ttl_seconds)
            heapq.heappush(self._expiries, _Expiry(now + ttl_seconds, reservation_id))
            return True

    def sweep(self) -> int:
        """Expire overdue reservations now. Returns how many were expired."""
        with self._lock:
            return self._expire_due(self._clock())

    def prune(self, older_than_seconds: float) -> int:
        return 0  # this store's history is already bounded

    # -- outbox (in memory: lost on restart, like the budgets) -------------

    def claim_events(self, limit: int, lease_seconds: float) -> list[OutboxItem]:
        with self._lock:
            now = self._clock()
            due = sorted((q.seq, key) for key, q in self._outbox.items()
                         if not q.dead and q.next_attempt_at <= now and q.locked_until <= now)
            items = []
            for _, key in due[:limit]:
                queued = self._outbox[key]
                queued.attempts += 1
                queued.locked_until = now + lease_seconds
                items.append(OutboxItem(key, queued.payload, queued.delivered, queued.attempts))
            return items

    def finish_event(self, event_id: str, *, delivered: tuple[str, ...], done: bool,
                     dead: bool = False, error: str | None = None,
                     retry_at: float | None = None) -> None:
        with self._lock:
            queued = self._outbox.get(event_id)
            if queued is None:
                return
            if done:
                del self._outbox[event_id]
                return
            queued.delivered, queued.dead, queued.error = tuple(delivered), dead, error
            queued.locked_until = 0.0
            if retry_at is not None:
                queued.next_attempt_at = retry_at

    def outbox_stats(self) -> dict[str, int]:
        with self._lock:
            dead = sum(q.dead for q in self._outbox.values())
            return {"pending": len(self._outbox) - dead, "dead": dead}

    def dead_events(self, limit: int = 100) -> list[tuple[OutboxItem, str | None]]:
        with self._lock:
            dead = sorted((q.seq, key) for key, q in self._outbox.items() if q.dead)
            return [(OutboxItem(key, self._outbox[key].payload, self._outbox[key].delivered,
                                self._outbox[key].attempts), self._outbox[key].error)
                    for _, key in dead[:limit]]

    def retry_dead(self, event_ids: list[str] | None = None) -> int:
        with self._lock:
            revived = 0
            for key, queued in self._outbox.items():
                if queued.dead and (event_ids is None or key in event_ids):
                    queued.dead, queued.attempts = False, 0
                    queued.next_attempt_at = queued.locked_until = 0.0
                    revived += 1
            return revived

    # -- inspection -------------------------------------------------------

    def status(self, tenant: str, feature: str | None = None) -> list[BudgetStatus]:
        """Current-period status of the budgets set for exactly this scope."""
        with self._lock:
            now = self._clock()
            self._expire_due(now)
            out = []
            for window in WINDOWS:
                key = BudgetKey(tenant, feature, window)
                config = self._budgets.get(key)
                if config is None:
                    continue
                period = period_for(window, now)
                state = self._state.get((key, period)) or _State()
                out.append(self._status(key, period, config, state))
            return out

    # -- internals (call with the lock held) ------------------------------

    def _applicable(self, tenant: str, feature: str | None,
                    now: float) -> list[tuple[BudgetKey, str]]:
        scopes = [None] if feature is None else [None, feature]
        return [
            (key, period_for(window, now))
            for scope in scopes
            for window in WINDOWS
            if (key := BudgetKey(tenant, scope, window)) in self._budgets
        ]

    def _unhold(self, held: _Held) -> None:
        if not held.holding:
            return
        held.holding = False
        for entry in held.entries:
            state = self._state.get(entry)
            if state is not None:
                state.reserved -= held.reservation.amount_micros

    def _expire_due(self, now: float) -> int:
        count = 0
        while self._expiries and self._expiries[0].at <= now:
            due = heapq.heappop(self._expiries)
            held = self._held.get(due.reservation_id)
            if held is None:  # already settled or released
                continue
            if held.reservation.expires_at > now:  # extended since this entry
                continue
            del self._held[due.reservation_id]
            self._unhold(held)
            self._remember(self._expired, due.reservation_id, held)
            count += 1
        return count

    def _remember(self, history: OrderedDict, key: str, value) -> None:
        history[key] = value
        while len(history) > self._HISTORY:
            history.popitem(last=False)

    @staticmethod
    def _status(key: BudgetKey, period: str, config: _Config, state: _State) -> BudgetStatus:
        return BudgetStatus(key=key, period=period, limit_micros=config.limit,
                            spent_micros=state.spent, reserved_micros=state.reserved,
                            soft_limit_micros=config.soft)
