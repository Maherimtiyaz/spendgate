import threading

import pytest

from spendgate import BudgetExceeded, BudgetKey, Sweeper

DAY = 86_400


@pytest.fixture
def store(make_store, clock):
    s = make_store(clock)
    s.set_budget(BudgetKey("acme", None, "month"), 1000)
    return s


def month(store, tenant="acme", feature=None):
    return next(s for s in store.status(tenant, feature) if s.key.window == "month")


def test_reserve_then_settle_books_actual_and_frees_the_rest(store):
    r = store.reserve("acme", None, 400, ttl_seconds=60)
    assert (month(store).reserved_micros, month(store).spent_micros) == (400, 0)
    result = store.settle(r.id, 150)
    assert result.applied and result.overage_micros == 0
    assert (month(store).reserved_micros, month(store).spent_micros) == (0, 150)
    assert month(store).remaining_micros == 850


def test_reservations_count_against_the_limit(store):
    store.reserve("acme", None, 600, ttl_seconds=60)
    with pytest.raises(BudgetExceeded) as exc:
        store.reserve("acme", None, 401, ttl_seconds=60)
    assert exc.value.remaining_micros == 400
    store.reserve("acme", None, 400, ttl_seconds=60)  # exactly fits


def test_refusal_leaves_no_partial_hold_on_other_budgets(store):
    store.set_budget(BudgetKey("acme", "chat", "month"), 100)
    with pytest.raises(BudgetExceeded) as exc:
        store.reserve("acme", "chat", 500, ttl_seconds=60)
    assert exc.value.feature == "chat"
    assert month(store).reserved_micros == 0
    assert month(store, feature="chat").reserved_micros == 0


def test_feature_calls_draw_on_tenant_and_feature_budgets(store):
    store.set_budget(BudgetKey("acme", "chat", "month"), 300)
    r = store.reserve("acme", "chat", 200, ttl_seconds=60)
    store.settle(r.id, 200)
    assert month(store).spent_micros == 200
    assert month(store, feature="chat").spent_micros == 200
    # A different feature is limited only by the tenant budget.
    store.reserve("acme", "search", 800, ttl_seconds=60)


def test_tenants_are_isolated(store):
    store.set_budget(BudgetKey("globex", None, "month"), 50)
    store.reserve("acme", None, 1000, ttl_seconds=60)
    store.reserve("globex", None, 50, ttl_seconds=60)


def test_tenant_without_budget_is_metered_but_not_limited(store):
    r = store.reserve("nobody", None, 10**12, ttl_seconds=60)
    assert store.settle(r.id, 5).applied


def test_release_frees_the_hold_without_spend(store):
    r = store.reserve("acme", None, 1000, ttl_seconds=60)
    store.release(r.id)
    store.release(r.id)  # harmless twice
    assert (month(store).reserved_micros, month(store).spent_micros) == (0, 0)


def test_settle_is_idempotent(store):
    r = store.reserve("acme", None, 400, ttl_seconds=60)
    assert store.settle(r.id, 100).applied
    again = store.settle(r.id, 100)
    assert not again.applied
    assert month(store).spent_micros == 100


def test_actual_above_reservation_is_booked_in_full(store):
    r = store.reserve("acme", None, 100, ttl_seconds=60)
    result = store.settle(r.id, 130)
    assert result.overage_micros == 30
    assert (month(store).reserved_micros, month(store).spent_micros) == (0, 130)


def test_expired_reservation_stops_holding_budget(store, clock):
    store.reserve("acme", None, 1000, ttl_seconds=60)
    clock.now += 61
    assert month(store).reserved_micros == 0
    store.reserve("acme", None, 1000, ttl_seconds=60)  # budget is usable again


def test_late_settle_after_expiry_still_books_the_spend(store, clock):
    r = store.reserve("acme", None, 400, ttl_seconds=60)
    clock.now += 61
    assert store.sweep() == 1
    assert store.settle(r.id, 250).applied
    assert (month(store).reserved_micros, month(store).spent_micros) == (0, 250)


def test_settle_unknown_reservation_raises(store):
    with pytest.raises(KeyError):
        store.settle("nope", 1)


def test_month_window_resets_and_total_window_does_not(store, clock):
    store.set_budget(BudgetKey("acme", None, "total"), 1500)
    r = store.reserve("acme", None, 1000, ttl_seconds=60)
    store.settle(r.id, 1000)
    with pytest.raises(BudgetExceeded):
        store.reserve("acme", None, 1, ttl_seconds=60)
    clock.now += 31 * DAY  # next calendar month
    assert month(store).spent_micros == 0
    store.reserve("acme", None, 500, ttl_seconds=60)
    with pytest.raises(BudgetExceeded) as exc:
        store.reserve("acme", None, 1, ttl_seconds=60)
    assert exc.value.window == "total"


def test_spend_settled_after_rollover_lands_in_the_period_it_was_reserved_in(store, clock):
    r = store.reserve("acme", None, 400, ttl_seconds=40 * DAY)
    clock.now += 31 * DAY
    store.settle(r.id, 400)
    assert month(store).spent_micros == 0


def test_soft_limit_is_reported_once_per_period(store):
    store.set_budget(BudgetKey("acme", None, "month"), 1000, soft_limit_micros=800)
    crossed = []
    for cost in (500, 300, 100):
        r = store.reserve("acme", None, cost, ttl_seconds=60)
        crossed.append(len(store.settle(r.id, cost).soft_crossed))
    assert crossed == [0, 1, 0]


def test_concurrent_reservations_never_exceed_the_limit(make_store, clock):
    store = make_store(clock)
    store.set_budget(BudgetKey("acme", None, "month"), 1_000_000)
    granted, refused = [], []
    start = threading.Barrier(16)

    def worker():
        start.wait()
        for _ in range(400):
            try:
                r = store.reserve("acme", None, 700, ttl_seconds=60)
            except BudgetExceeded:
                refused.append(1)
            else:
                granted.append(r)
                store.settle(r.id, 700)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    status = month(store)
    assert len(granted) == 1_000_000 // 700  # every reservation that fits, no more
    assert len(granted) + len(refused) == 6400
    assert status.spent_micros == len(granted) * 700 <= status.limit_micros
    assert status.reserved_micros == 0


def test_refused_tenant_gets_budget_back_from_another_workers_dead_hold(store, clock):
    store.reserve("acme", None, 1000, ttl_seconds=60)  # its worker then died
    clock.now += 61
    store.reserve("acme", None, 1000, ttl_seconds=60)  # no sweep was needed first


def test_sweep_only_expires_overdue_holds(store, clock):
    store.reserve("acme", None, 100, ttl_seconds=30)
    store.reserve("acme", None, 200, ttl_seconds=90)
    clock.now += 31
    assert store.sweep() == 1
    assert store.sweep() == 0
    assert month(store).reserved_micros == 200


def test_sweeper_thread_expires_holds_in_the_background(store, clock):
    store.reserve("acme", None, 100, ttl_seconds=30)
    clock.now += 31
    swept = threading.Event()
    real_sweep = store.sweep

    def sweep():
        count = real_sweep()
        if count:
            swept.set()
        return count

    store.sweep = sweep
    with Sweeper(store, interval=0.01):
        assert swept.wait(5)


def test_extended_hold_outlives_its_original_ttl(store, clock):
    r = store.reserve("acme", None, 1000, ttl_seconds=60)
    clock.now += 50
    assert store.extend(r.id, 60)
    clock.now += 50  # 100s after reserving: past the original TTL, inside the new one
    assert store.sweep() == 0
    assert month(store).reserved_micros == 1000
    clock.now += 11
    assert store.sweep() == 1
    assert month(store).reserved_micros == 0


def test_extend_reports_a_hold_that_is_already_gone(store, clock):
    expired = store.reserve("acme", None, 100, ttl_seconds=60)
    settled = store.reserve("acme", None, 100, ttl_seconds=600)
    store.settle(settled.id, 50)
    clock.now += 61
    assert not store.extend(expired.id, 60)
    assert not store.extend(settled.id, 60)
    assert not store.extend("nope", 60)
    assert month(store).reserved_micros == 0


def test_prune_removes_old_records_without_touching_spend(store, clock):
    old = store.reserve("acme", None, 100, ttl_seconds=60)
    store.settle(old.id, 80)
    clock.now += 3600
    recent = store.reserve("acme", None, 100, ttl_seconds=60)
    store.settle(recent.id, 70)
    live = store.reserve("acme", None, 100, ttl_seconds=60)

    removed = store.prune(older_than_seconds=600)
    assert removed in (0, 1)  # the in-memory store keeps no unbounded history
    assert month(store).spent_micros == 150 and month(store).reserved_micros == 100
    assert not store.settle(recent.id, 70).applied  # still known: no double booking
    assert store.settle(live.id, 10).applied
