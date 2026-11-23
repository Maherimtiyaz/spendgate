"""Dispatcher: delivers queued usage events to billing sinks, at least once.

With ``Guard(store, outbox=True)`` every settled call leaves a usage event in
the store's outbox, written together with the booking. A dispatcher takes
events from there and hands each one to every sink. An event stays queued
until all sinks have accepted it, so a crash or an outage delays billing data
but does not lose it. Delivery can repeat; sinks deduplicate on the event ID.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Protocol, Sequence

from .errors import PermanentDeliveryError
from .guard import UsageEvent
from .ledger import LedgerStore

log = logging.getLogger("spendgate")


class Sink(Protocol):
    #: Unique among a dispatcher's sinks; recorded against delivered events.
    name: str

    def send(self, event: UsageEvent) -> None:
        """Deliver one event. Raise ``PermanentDeliveryError`` if it can never
        be accepted, any other exception if it is worth retrying."""


class Dispatcher:
    def __init__(self, store: LedgerStore, sinks: Sequence[Sink], *, batch_size: int = 100,
                 interval: float = 1.0, lease_seconds: float = 60.0, max_attempts: int = 12,
                 backoff_base: float = 5.0, backoff_max: float = 3600.0,
                 clock: Callable[[], float] = time.time):
        names = [sink.name for sink in sinks]
        if not sinks or len(set(names)) != len(names):
            raise ValueError("a dispatcher needs at least one sink, each with a unique name")
        self._store = store
        self._sinks = list(sinks)
        self._batch_size = batch_size
        self._interval = interval
        self._lease = lease_seconds
        self._max_attempts = max_attempts
        self._backoff_base = backoff_base
        self._backoff_max = backoff_max
        self._clock = clock
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def run_once(self) -> int:
        """Attempt one batch. Returns how many events were fully delivered."""
        finished = 0
        for item in self._store.claim_events(self._batch_size, self._lease):
            delivered = list(item.delivered)
            error, permanent = None, False
            try:
                event = UsageEvent.from_dict(item.payload)
            except Exception as exc:
                error, permanent = f"unreadable event: {exc!r}", True
            else:
                for sink in self._sinks:
                    if sink.name in delivered:
                        continue  # this sink has it from an earlier attempt
                    try:
                        sink.send(event)
                    except PermanentDeliveryError as exc:
                        error, permanent = f"{sink.name}: {exc}", True
                    except Exception as exc:
                        error = f"{sink.name}: {exc!r}"
                    else:
                        delivered.append(sink.name)
            if error is None:
                self._store.finish_event(item.id, delivered=tuple(delivered), done=True)
                finished += 1
            elif permanent or item.attempts >= self._max_attempts:
                log.error("usage event %s will not be retried: %s", item.id, error)
                self._store.finish_event(item.id, delivered=tuple(delivered), done=False,
                                         dead=True, error=error)
            else:
                delay = min(self._backoff_base * 2 ** (item.attempts - 1), self._backoff_max)
                self._store.finish_event(item.id, delivered=tuple(delivered), done=False,
                                         error=error, retry_at=self._clock() + delay)
        return finished

    def start(self) -> "Dispatcher":
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="spendgate-dispatcher",
                                            daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                busy = self.run_once() >= self._batch_size
            except Exception:
                log.exception("dispatch failed; will retry")
                busy = False
            if not busy:
                self._stop.wait(self._interval)

    def __enter__(self) -> "Dispatcher":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
