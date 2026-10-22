"""Small, shared helpers for bounded pipeline worker pools.

The pipeline has two distinct resource controls:

* :func:`resolve_worker_count` turns a requested (or automatic) pool size
  into a size bounded by the caller's CPU budget, the amount of work, and the
  cap already owned by a stage.
* :class:`WeightedAdmissionGate` limits the sum of CPU weights of phases that
  are active at the same time.  Its queue is FIFO so a large phase cannot be
  starved by later small phases.

This module deliberately has no knowledge of pipeline configuration or of a
particular stage.  A missing ``cpu_budget`` means standalone execution and
therefore leaves the caller's automatic default intact, subject only to the
ordinary item-count and stage-cap bounds.
"""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
import threading
from typing import Iterator


def _positive_or_none(value: int | None, name: str) -> int | None:
    """Validate an optional positive bound.

    ``None`` is the only spelling for an absent bound.  Worker requests use
    zero for automatic selection, so zero is handled by the public resolver
    rather than by this validator.
    """

    if value is None:
        return None
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def resolve_worker_count(
    requested: int | None = None,
    *,
    cpu_budget: int | None = None,
    item_count: int | None = None,
    stage_cap: int | None = None,
    automatic: int | None = None,
) -> int:
    """Resolve one stage pool size without exceeding its available bounds.

    ``requested`` follows the convention used by the existing command line
    interfaces: a positive value is explicit and ``None`` or ``0`` asks for
    the stage's automatic value.  ``automatic`` is the already-computed
    standalone default.  If it is omitted, the CPU budget is used when one
    was supplied, otherwise the stage cap, item count, or one worker is used
    in that order.

    A positive ``cpu_budget`` is an invocation-level bound.  In the absence
    of a budget, the helper does not replace an automatic default with a
    machine-wide value; this is what preserves standalone stage behavior.
    ``item_count=0`` resolves to zero, which lets callers avoid constructing
    an empty process pool.
    """

    requested = _positive_or_none(requested, "requested")
    item_count = _positive_or_none(item_count, "item_count")
    stage_cap = _positive_or_none(stage_cap, "stage_cap")
    automatic = _positive_or_none(automatic, "automatic")
    cpu_budget = _positive_or_none(cpu_budget, "cpu_budget")
    if cpu_budget == 0:
        raise ValueError("cpu_budget must be positive")
    # Existing stage APIs use zero to mean "automatic/no explicit cap" for
    # both their worker default and optional cap.
    if automatic == 0:
        automatic = None
    if stage_cap == 0:
        stage_cap = None

    if item_count == 0:
        return 0

    if requested:
        value = requested
    elif automatic is not None:
        value = automatic
    elif cpu_budget is not None:
        value = cpu_budget
    elif stage_cap is not None:
        value = stage_cap
    elif item_count is not None:
        value = item_count
    else:
        value = 1

    bounds = [value]
    if cpu_budget is not None:
        bounds.append(cpu_budget)
    if item_count is not None:
        bounds.append(item_count)
    if stage_cap is not None:
        bounds.append(stage_cap)
    return max(1, min(bounds))


# A descriptive alias is useful at call sites that refer to the resolved
# value as a budget rather than a worker count.
resolve_worker_budget = resolve_worker_count


class PhaseAdmissionCancelled(RuntimeError):
    """Raised when a queued phase is cancelled before it acquires capacity."""


@dataclass(eq=False)
class _AdmissionTicket:
    weight: int


class WeightedAdmissionGate:
    """FIFO weighted admission for concurrently running pipeline phases.

    The gate accounts only for admitted phase work.  Queue operations remain
    outside the lease in callers, so waiting on this condition never holds a
    queue or pipeline lock.  A lease is released exactly once even when its
    body raises an exception.
    """

    def __init__(self, capacity: int):
        capacity = _positive_or_none(capacity, "capacity")
        if not capacity:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._condition = threading.Condition()
        self._waiting: deque[_AdmissionTicket] = deque()
        self._used = 0

    @property
    def active_weight(self) -> int:
        """Current sum of admitted lease weights."""

        with self._condition:
            return self._used

    @property
    def current_weight(self) -> int:
        """Compatibility spelling for callers inspecting active capacity."""

        return self.active_weight

    @property
    def queued_count(self) -> int:
        with self._condition:
            return len(self._waiting)

    def wake_all(self) -> None:
        """Wake all current waiters after an external stop/failure signal."""

        with self._condition:
            self._condition.notify_all()

    @contextmanager
    def lease(
        self,
        weight: int,
        cancel_event: threading.Event | None = None,
    ) -> Iterator[None]:
        """Wait FIFO for ``weight`` capacity and release it on exit.

        Condition waits are deliberately short so a worker failure can set a
        shared cancellation event without needing to reach into the gate.
        Cancellation removes the ticket and wakes following waiters before
        raising :class:`PhaseAdmissionCancelled`.
        """

        weight = _positive_or_none(weight, "weight")
        if not weight:
            raise ValueError("weight must be positive")
        if weight > self.capacity:
            raise ValueError(
                f"weight {weight} exceeds gate capacity {self.capacity}")
        if cancel_event is not None and not hasattr(cancel_event, "is_set"):
            raise TypeError("cancel_event must provide is_set()")

        ticket = _AdmissionTicket(weight)
        admitted = False
        with self._condition:
            self._waiting.append(ticket)
            while True:
                cancelled = (
                    cancel_event is not None and cancel_event.is_set())
                if cancelled:
                    self._waiting.remove(ticket)
                    self._condition.notify_all()
                    raise PhaseAdmissionCancelled(
                        "phase admission cancelled while waiting")
                if (self._waiting[0] is ticket
                        and self._used + weight <= self.capacity):
                    self._waiting.popleft()
                    self._used += weight
                    admitted = True
                    break
                # A bounded wait makes cancellation responsive even if no
                # currently active lease is released for a while.
                self._condition.wait(timeout=0.05)

        try:
            yield
        finally:
            if admitted:
                with self._condition:
                    self._used -= weight
                    self._condition.notify_all()


__all__ = [
    "PhaseAdmissionCancelled",
    "WeightedAdmissionGate",
    "resolve_worker_budget",
    "resolve_worker_count",
]
