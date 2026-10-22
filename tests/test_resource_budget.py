"""Unit tests for shared worker budgets and phase admission."""

from __future__ import annotations

import threading
import time

import pytest

from scripts.resource_budget import (
    PhaseAdmissionCancelled,
    WeightedAdmissionGate,
    resolve_worker_count,
)


@pytest.mark.parametrize(
    ("requested", "budget", "items", "cap", "automatic", "expected"),
    [
        (32, 8, 100, 16, 4, 8),
        (32, 64, 3, 16, 4, 3),
        (32, 64, 100, 5, 4, 5),
        (0, 8, 100, 16, 4, 4),
        (None, 8, 100, 16, 0, 8),
        (None, None, 3, 16, 32, 3),
        (None, None, 100, 5, 32, 5),
    ],
)
def test_resolver_clamps_explicit_and_automatic_values(
        requested, budget, items, cap, automatic, expected):
    assert resolve_worker_count(
        requested,
        cpu_budget=budget,
        item_count=items,
        stage_cap=cap,
        automatic=automatic,
    ) == expected


def test_resolver_zero_items_avoids_an_empty_pool():
    assert resolve_worker_count(
        8, cpu_budget=1, item_count=0, stage_cap=1) == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"cpu_budget": 0}, "cpu_budget must be positive"),
        ({"cpu_budget": -1}, "cpu_budget must be non-negative"),
        ({"item_count": -1}, "item_count must be non-negative"),
        ({"stage_cap": -1}, "stage_cap must be non-negative"),
        ({"requested": -1}, "requested must be non-negative"),
    ],
)
def test_resolver_rejects_invalid_bounds(kwargs, message):
    with pytest.raises(ValueError, match=message):
        resolve_worker_count(**kwargs)


def test_missing_budget_preserves_stage_automatic_default():
    assert resolve_worker_count(
        None, item_count=100, stage_cap=32, automatic=24) == 24
    assert resolve_worker_count(
        48, item_count=100, stage_cap=32, automatic=24) == 32


def test_gate_weight_sum_never_exceeds_capacity():
    gate = WeightedAdmissionGate(3)
    active = 0
    peak = 0
    lock = threading.Lock()
    entered = threading.Barrier(4)

    def worker(weight):
        nonlocal active, peak
        entered.wait()
        with gate.lease(weight):
            with lock:
                active += weight
                peak = max(peak, active)
                assert active == gate.active_weight
            time.sleep(0.02)
            with lock:
                active -= weight

    threads = [threading.Thread(target=worker, args=(w,)) for w in (2, 1, 1)]
    for thread in threads:
        thread.start()
    entered.wait()
    for thread in threads:
        thread.join(timeout=2)
        assert not thread.is_alive()
    assert peak <= 3
    assert gate.active_weight == 0


def test_gate_is_fifo_for_weighted_requests():
    gate = WeightedAdmissionGate(2)
    holder_ready = threading.Event()
    release_holder = threading.Event()
    order = []

    def holder():
        with gate.lease(2):
            holder_ready.set()
            release_holder.wait(timeout=2)

    def waiter(name, weight):
        with gate.lease(weight):
            order.append(name)

    owner = threading.Thread(target=holder)
    owner.start()
    assert holder_ready.wait(timeout=1)
    first = threading.Thread(target=waiter, args=("large", 2))
    second = threading.Thread(target=waiter, args=("small", 1))
    first.start()
    time.sleep(0.03)
    second.start()
    time.sleep(0.03)
    assert order == []
    release_holder.set()
    owner.join(timeout=1)
    first.join(timeout=1)
    second.join(timeout=1)
    assert order == ["large", "small"]


def test_gate_cancellation_removes_ticket_and_wakes_following_waiter():
    gate = WeightedAdmissionGate(1)
    holder_ready = threading.Event()
    release_holder = threading.Event()
    cancel = threading.Event()
    cancelled = []
    acquired = threading.Event()

    def holder():
        with gate.lease(1):
            holder_ready.set()
            release_holder.wait(timeout=2)

    def cancelled_waiter():
        try:
            with gate.lease(1, cancel):
                raise AssertionError("cancelled waiter was admitted")
        except PhaseAdmissionCancelled:
            cancelled.append(True)

    def following_waiter():
        with gate.lease(1):
            acquired.set()

    owner = threading.Thread(target=holder)
    owner.start()
    assert holder_ready.wait(timeout=1)
    first = threading.Thread(target=cancelled_waiter)
    second = threading.Thread(target=following_waiter)
    first.start()
    time.sleep(0.03)
    second.start()
    cancel.set()
    first.join(timeout=1)
    assert cancelled == [True]
    release_holder.set()
    owner.join(timeout=1)
    second.join(timeout=1)
    assert acquired.is_set()
    assert gate.queued_count == 0


def test_gate_cancel_removes_the_requested_ticket_when_weights_match():
    gate = WeightedAdmissionGate(1)
    holder_ready = threading.Event()
    release_holder = threading.Event()
    first_acquired = threading.Event()
    second_cancel = threading.Event()
    second_cancelled = threading.Event()

    def holder():
        with gate.lease(1):
            holder_ready.set()
            release_holder.wait(timeout=2)

    def first_waiter():
        with gate.lease(1):
            first_acquired.set()

    def second_waiter():
        try:
            with gate.lease(1, second_cancel):
                raise AssertionError("second waiter was unexpectedly admitted")
        except PhaseAdmissionCancelled:
            second_cancelled.set()

    owner = threading.Thread(target=holder)
    owner.start()
    assert holder_ready.wait(timeout=1)
    first = threading.Thread(target=first_waiter)
    second = threading.Thread(target=second_waiter)
    first.start()
    time.sleep(0.03)
    second.start()
    time.sleep(0.03)
    second_cancel.set()
    gate.wake_all()
    assert second_cancelled.wait(timeout=1)
    release_holder.set()
    owner.join(timeout=1)
    first.join(timeout=1)
    second.join(timeout=1)
    assert first_acquired.is_set()
    assert gate.queued_count == 0


def test_gate_releases_after_exception_and_capacity_one_serializes():
    gate = WeightedAdmissionGate(1)
    with pytest.raises(RuntimeError):
        with gate.lease(1):
            assert gate.active_weight == 1
            raise RuntimeError("phase failed")
    assert gate.active_weight == 0
    with gate.lease(1):
        assert gate.active_weight == 1
    assert gate.active_weight == 0


@pytest.mark.parametrize("capacity", [0, -1])
def test_gate_requires_positive_capacity(capacity):
    with pytest.raises(ValueError):
        WeightedAdmissionGate(capacity)


def test_gate_rejects_phase_larger_than_capacity():
    gate = WeightedAdmissionGate(1)
    with pytest.raises(ValueError, match="exceeds gate capacity"):
        with gate.lease(2):
            pass
