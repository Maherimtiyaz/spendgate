"""PostgreSQL ledger: budgets shared by every worker, surviving restarts.

Requires ``psycopg`` 3 with its connection pool: ``pip install "spendgate[postgres]"``.

Atomicity comes from the database. Reserving against a budget is one
conditional upsert that only succeeds while ``spent + reserved + amount``
stays within the limit, so two workers racing for the last dollar cannot both
win. A reservation spanning several budgets runs in one transaction and takes
its row locks in a fixed order, so it is all-or-nothing and cannot deadlock
against another reservation.
"""
from __future__ import annotations

import json
import time
import uuid
from collections import defaultdict
from typing import Any, Callable

import psycopg
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from .errors import BudgetExceeded
from .ledger import (WINDOWS, BudgetKey, BudgetStatus, OutboxItem, Reservation,
                     SettleResult, period_for)

SCHEMA = """
CREATE TABLE IF NOT EXISTS spendgate_budgets (
    tenant            text   NOT NULL,
    feature           text   NOT NULL DEFAULT '',
    win               text   NOT NULL,
    limit_micros      bigint NOT NULL CHECK (limit_micros >= 0),
    soft_limit_micros bigint,
    PRIMARY KEY (tenant, feature, win)
);
CREATE TABLE IF NOT EXISTS spendgate_periods (
    tenant          text    NOT NULL,
    feature         text    NOT NULL,
    win             text    NOT NULL,
    period          text    NOT NULL,
    spent_micros    bigint  NOT NULL DEFAULT 0,
    reserved_micros bigint  NOT NULL DEFAULT 0 CHECK (reserved_micros >= 0),
    soft_fired      boolean NOT NULL DEFAULT false,
    PRIMARY KEY (tenant, feature, win, period)
);
CREATE TABLE IF NOT EXISTS spendgate_reservations (
    id            text             PRIMARY KEY,
    tenant        text             NOT NULL,
    feature       text             NOT NULL,
    amount_micros bigint           NOT NULL,
    status        text             NOT NULL,  -- held | expired | settled
    actual_micros bigint,
    entries       jsonb            NOT NULL,  -- [[feature, win, period], ...]
    created_at    double precision NOT NULL,
    expires_at    double precision NOT NULL,
    settled_at    double precision
);
CREATE INDEX IF NOT EXISTS spendgate_reservations_due
    ON spendgate_reservations (expires_at) WHERE status = 'held';
CREATE TABLE IF NOT EXISTS spendgate_outbox (
    id              text             PRIMARY KEY,  -- the usage event id
    seq             bigserial        NOT NULL,
    payload         jsonb            NOT NULL,
    delivered       text[]           NOT NULL DEFAULT '{}',
    attempts        integer          NOT NULL DEFAULT 0,
    next_attempt_at double precision NOT NULL DEFAULT 0,
    locked_until    double precision NOT NULL DEFAULT 0,
    dead            boolean          NOT NULL DEFAULT false,
    last_error      text
);
CREATE INDEX IF NOT EXISTS spendgate_outbox_due
    ON spendgate_outbox (next_attempt_at) WHERE NOT dead;
"""

# Insert the period row holding `amount`, or add `amount` to the existing row,
# but only while the budget still has room. No returned row means refused.
_HOLD = """
INSERT INTO spendgate_periods AS p (tenant, feature, win, period, reserved_micros)
SELECT %(tenant)s, %(feature)s, %(win)s, %(period)s, %(amount)s
 WHERE %(amount)s <= %(limit)s
ON CONFLICT (tenant, feature, win, period) DO UPDATE
   SET reserved_micros = p.reserved_micros + EXCLUDED.reserved_micros
 WHERE p.spent_micros + p.reserved_micros + EXCLUDED.reserved_micros <= %(limit)s
RETURNING 1
"""

_PERIOD = ("SELECT spent_micros, reserved_micros FROM spendgate_periods "
           "WHERE tenant = %s AND feature = %s AND win = %s AND period = %s")


class _Refused(Exception):
    def __init__(self, **details: Any):
        self.details = details


class PostgresStore:
    """Ledger store on PostgreSQL. Pass a connection string or your own pool."""

    # Calls block on the network; the guard runs them off the event loop.
    blocking = True

    def __init__(self, conninfo: str | None = None, *, pool: ConnectionPool | None = None,
                 max_connections: int = 10, clock: Callable[[], float] = time.time):
        if (conninfo is None) == (pool is None):
            raise ValueError("pass exactly one of conninfo or pool")
        self._owns_pool = pool is None
        self._pool = pool or ConnectionPool(conninfo, min_size=1, max_size=max_connections,
                                            open=True)
        self._clock = clock

    def setup(self) -> "PostgresStore":
        """Create the tables if they do not exist. Safe to call on every start."""
        with self._pool.connection() as conn:
            conn.execute(SCHEMA)
        return self

    def close(self) -> None:
        if self._owns_pool:
            self._pool.close()

    # -- configuration ----------------------------------------------------

    def set_budget(self, key: BudgetKey, limit_micros: int,
                   soft_limit_micros: int | None = None) -> None:
        if key.window not in WINDOWS:
            raise ValueError(f"unknown budget window {key.window!r}; expected one of {WINDOWS}")
        if limit_micros < 0:
            raise ValueError("budget limit must not be negative")
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO spendgate_budgets (tenant, feature, win, limit_micros, soft_limit_micros) "
                "VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT (tenant, feature, win) DO UPDATE SET "
                "limit_micros = EXCLUDED.limit_micros, soft_limit_micros = EXCLUDED.soft_limit_micros",
                (key.tenant, key.feature or "", key.window, limit_micros, soft_limit_micros))

    def remove_budget(self, key: BudgetKey) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "DELETE FROM spendgate_budgets WHERE tenant = %s AND feature = %s AND win = %s",
                (key.tenant, key.feature or "", key.window))

    # -- reserve / settle / release ---------------------------------------

    def reserve(self, tenant: str, feature: str | None, amount_micros: int,
                ttl_seconds: float) -> Reservation:
        if amount_micros < 0:
            raise ValueError("reservation amount must not be negative")
        now = self._clock()
        reservation = Reservation(id=uuid.uuid4().hex, tenant=tenant, feature=feature,
                                  amount_micros=amount_micros, expires_at=now + ttl_seconds)
        with self._pool.connection() as conn:
            try:
                self._hold(conn, reservation, now)
            except _Refused as first:
                # Overdue holds may be what is filling the budget. Expiring
                # them costs a query, so it is only done when a call would
                # otherwise be refused.
                if not self._expire_due(conn, now, tenant):
                    raise BudgetExceeded(**first.details) from None
                try:
                    self._hold(conn, reservation, now)
                except _Refused as second:
                    raise BudgetExceeded(**second.details) from None
        return reservation

    def _hold(self, conn: psycopg.Connection, reservation: Reservation, now: float) -> None:
        tenant, amount = reservation.tenant, reservation.amount_micros
        features = [""] if reservation.feature is None else ["", reservation.feature]
        with conn.transaction():
            budgets = conn.execute(
                "SELECT feature, win, limit_micros FROM spendgate_budgets "
                "WHERE tenant = %s AND feature = ANY(%s) ORDER BY feature, win",
                (tenant, features)).fetchall()
            entries = []
            for feature, win, limit in budgets:  # fixed order: no deadlocks
                period = period_for(win, now)
                held = conn.execute(_HOLD, {"tenant": tenant, "feature": feature, "win": win,
                                            "period": period, "amount": amount,
                                            "limit": limit}).fetchone()
                if held is None:
                    row = conn.execute(_PERIOD, (tenant, feature, win, period)).fetchone()
                    spent, reserved = row or (0, 0)
                    # Raising rolls back the holds already taken on other budgets.
                    raise _Refused(tenant=tenant, feature=feature or None, window=win,
                                   period=period, limit_micros=limit, spent_micros=spent,
                                   reserved_micros=reserved, requested_micros=amount)
                entries.append([feature, win, period])
            conn.execute(
                "INSERT INTO spendgate_reservations "
                "(id, tenant, feature, amount_micros, status, entries, created_at, expires_at) "
                "VALUES (%s, %s, %s, %s, 'held', %s, %s, %s)",
                (reservation.id, tenant, reservation.feature or "", amount,
                 json.dumps(entries), now, reservation.expires_at))

    def settle(self, reservation_id: str, actual_micros: int,
               event: dict | None = None) -> SettleResult:
        if actual_micros < 0:
            raise ValueError("actual cost must not be negative")
        with self._pool.connection() as conn, conn.transaction():
            row = conn.execute(
                "SELECT tenant, amount_micros, status, actual_micros, entries "
                "FROM spendgate_reservations WHERE id = %s FOR UPDATE",
                (reservation_id,)).fetchone()
            if row is None:
                raise KeyError(f"unknown reservation {reservation_id!r}")
            tenant, amount, status, booked, entries = row
            if status == "settled":
                return SettleResult(booked, amount, applied=False)
            # An expired hold was already given back; the spend is still booked.
            unhold = amount if status == "held" else 0
            crossed = []
            for feature, win, period in sorted(entries):
                conn.execute(
                    "UPDATE spendgate_periods SET spent_micros = spent_micros + %s, "
                    "reserved_micros = reserved_micros - %s "
                    "WHERE tenant = %s AND feature = %s AND win = %s AND period = %s",
                    (actual_micros, unhold, tenant, feature, win, period))
                soft = conn.execute(
                    "UPDATE spendgate_periods p SET soft_fired = true FROM spendgate_budgets b "
                    "WHERE b.tenant = p.tenant AND b.feature = p.feature AND b.win = p.win "
                    "AND p.tenant = %s AND p.feature = %s AND p.win = %s AND p.period = %s "
                    "AND NOT p.soft_fired AND p.spent_micros >= b.soft_limit_micros "
                    "RETURNING b.limit_micros, b.soft_limit_micros, p.spent_micros, p.reserved_micros",
                    (tenant, feature, win, period)).fetchone()
                if soft is not None:
                    crossed.append(BudgetStatus(
                        key=BudgetKey(tenant, feature or None, win), period=period,
                        limit_micros=soft[0], soft_limit_micros=soft[1],
                        spent_micros=soft[2], reserved_micros=soft[3]))
            conn.execute(
                "UPDATE spendgate_reservations SET status = 'settled', actual_micros = %s, "
                "settled_at = %s WHERE id = %s",
                (actual_micros, self._clock(), reservation_id))
            if event is not None:
                # Same transaction as the booking: a cost is never booked
                # without its event, and never the other way round.
                conn.execute("INSERT INTO spendgate_outbox (id, payload) VALUES (%s, %s)",
                             (reservation_id, Jsonb(event)))
            return SettleResult(actual_micros, amount, applied=True, soft_crossed=tuple(crossed))

    def release(self, reservation_id: str) -> None:
        with self._pool.connection() as conn, conn.transaction():
            row = conn.execute(
                "SELECT tenant, amount_micros, status, entries FROM spendgate_reservations "
                "WHERE id = %s FOR UPDATE", (reservation_id,)).fetchone()
            if row is None or row[2] == "settled":
                return
            tenant, amount, status, entries = row
            if status == "held":
                for feature, win, period in sorted(entries):
                    conn.execute(
                        "UPDATE spendgate_periods SET reserved_micros = reserved_micros - %s "
                        "WHERE tenant = %s AND feature = %s AND win = %s AND period = %s",
                        (amount, tenant, feature, win, period))
            conn.execute("DELETE FROM spendgate_reservations WHERE id = %s", (reservation_id,))

    def extend(self, reservation_id: str, ttl_seconds: float) -> bool:
        with self._pool.connection() as conn:
            row = conn.execute(
                "UPDATE spendgate_reservations SET expires_at = %s "
                "WHERE id = %s AND status = 'held' AND expires_at > %s RETURNING 1",
                (self._clock() + ttl_seconds, reservation_id, self._clock())).fetchone()
        return row is not None

    def prune(self, older_than_seconds: float) -> int:
        cutoff = self._clock() - older_than_seconds
        with self._pool.connection() as conn:
            return conn.execute(
                "DELETE FROM spendgate_reservations WHERE "
                "(status = 'settled' AND settled_at < %s) OR "
                "(status = 'expired' AND expires_at < %s)", (cutoff, cutoff)).rowcount

    def sweep(self) -> int:
        """Expire overdue reservations for every tenant. Returns how many."""
        with self._pool.connection() as conn:
            return self._expire_due(conn, self._clock(), None)

    def _expire_due(self, conn: psycopg.Connection, now: float, tenant: str | None) -> int:
        with conn.transaction():
            # SKIP LOCKED lets several sweepers run without waiting on, or
            # double-expiring, each other's rows.
            due = conn.execute(
                "SELECT id, tenant, amount_micros, entries FROM spendgate_reservations "
                "WHERE status = 'held' AND expires_at <= %s "
                "AND (%s::text IS NULL OR tenant = %s) ORDER BY id FOR UPDATE SKIP LOCKED",
                (now, tenant, tenant)).fetchall()
            if not due:
                return 0
            give_back: dict[tuple, int] = defaultdict(int)
            for _id, owner, amount, entries in due:
                for feature, win, period in entries:
                    give_back[(owner, feature, win, period)] += amount
            for key in sorted(give_back):
                conn.execute(
                    "UPDATE spendgate_periods SET reserved_micros = reserved_micros - %s "
                    "WHERE tenant = %s AND feature = %s AND win = %s AND period = %s",
                    (give_back[key], *key))
            conn.execute(
                "UPDATE spendgate_reservations SET status = 'expired' WHERE id = ANY(%s)",
                ([row[0] for row in due],))
            return len(due)

    # -- outbox -----------------------------------------------------------

    def claim_events(self, limit: int, lease_seconds: float) -> list[OutboxItem]:
        now = self._clock()
        with self._pool.connection() as conn:
            # SKIP LOCKED: several dispatchers can run without taking the
            # same events. The lease covers a dispatcher dying mid-delivery.
            rows = conn.execute(
                "UPDATE spendgate_outbox SET attempts = attempts + 1, locked_until = %(lease)s "
                "WHERE id IN (SELECT id FROM spendgate_outbox WHERE NOT dead "
                "AND next_attempt_at <= %(now)s AND locked_until <= %(now)s "
                "ORDER BY seq LIMIT %(limit)s FOR UPDATE SKIP LOCKED) "
                "RETURNING seq, id, payload, delivered, attempts",
                {"now": now, "lease": now + lease_seconds, "limit": limit}).fetchall()
        return [OutboxItem(id, payload, tuple(delivered), attempts)
                for _seq, id, payload, delivered, attempts in sorted(rows)]

    def finish_event(self, event_id: str, *, delivered: tuple[str, ...], done: bool,
                     dead: bool = False, error: str | None = None,
                     retry_at: float | None = None) -> None:
        with self._pool.connection() as conn:
            if done:
                conn.execute("DELETE FROM spendgate_outbox WHERE id = %s", (event_id,))
                return
            conn.execute(
                "UPDATE spendgate_outbox SET delivered = %s, dead = %s, last_error = %s, "
                "locked_until = 0, next_attempt_at = COALESCE(%s, next_attempt_at) WHERE id = %s",
                (list(delivered), dead, error, retry_at, event_id))

    def dead_events(self, limit: int = 100) -> list[tuple[OutboxItem, str | None]]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT id, payload, delivered, attempts, last_error FROM spendgate_outbox "
                "WHERE dead ORDER BY seq LIMIT %s", (limit,)).fetchall()
        return [(OutboxItem(id, payload, tuple(delivered), attempts), error)
                for id, payload, delivered, attempts, error in rows]

    def retry_dead(self, event_ids: list[str] | None = None) -> int:
        with self._pool.connection() as conn:
            return conn.execute(
                "UPDATE spendgate_outbox SET dead = false, attempts = 0, next_attempt_at = 0, "
                "locked_until = 0 WHERE dead AND (%s::text[] IS NULL OR id = ANY(%s))",
                (event_ids, event_ids)).rowcount

    def outbox_stats(self) -> dict[str, int]:
        with self._pool.connection() as conn:
            dead, total = conn.execute(
                "SELECT count(*) FILTER (WHERE dead), count(*) FROM spendgate_outbox").fetchone()
        return {"pending": total - dead, "dead": dead}

    # -- inspection -------------------------------------------------------

    def status(self, tenant: str, feature: str | None = None) -> list[BudgetStatus]:
        now = self._clock()
        with self._pool.connection() as conn:
            self._expire_due(conn, now, tenant)
            budgets = {
                win: (limit, soft) for win, limit, soft in conn.execute(
                    "SELECT win, limit_micros, soft_limit_micros FROM spendgate_budgets "
                    "WHERE tenant = %s AND feature = %s", (tenant, feature or "")).fetchall()}
            out = []
            for window in WINDOWS:
                if window not in budgets:
                    continue
                limit, soft = budgets[window]
                period = period_for(window, now)
                row = conn.execute(_PERIOD, (tenant, feature or "", window, period)).fetchone()
                spent, reserved = row or (0, 0)
                out.append(BudgetStatus(
                    key=BudgetKey(tenant, feature, window), period=period, limit_micros=limit,
                    spent_micros=spent, reserved_micros=reserved, soft_limit_micros=soft))
            return out
