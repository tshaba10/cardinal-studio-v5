"""
Calendars and shared series.

Calendar ``monthly_cycle_day``
------------------------------
One step per calendar month. The absolute step index ``g`` is "months since
1970-01", so every entity, the shared series and the random-number keys share
one clock. An entity with ``cycle_day = d`` has its statement on day ``d`` of
month ``g``; the cycle runs from the day after the previous statement (or the
open date for the first cycle) up to and including the statement date.

* first cycle: the first statement date strictly after the open date
* ``cycle_number`` is 1 for the statement in the month of ``time.output_start``
  (it is <= 0 during warm-up, which is simulated but not written)

Calendar ``monthly`` is the same with every statement on the last day of the
month (no per-entity cycle day).

Shared series - guide Step 2
----------------------------
A shared factor (unemployment, seasonality) is generated once per absolute
step and read by every entity, giving cross-entity dependence without
coupling entities to each other - every entity stays independently
simulable, which is what makes chunking and parallelism safe.
"""
from __future__ import annotations

import numpy as np

from ..rng import KeyedRNG
from ..types import month_index, month_of_year, month_start_days, parse_date


class MonthlyCalendar:
    def __init__(self, time_spec: dict, start_days: np.ndarray, cycle_day: np.ndarray | None):
        self.start = np.asarray(start_days, dtype=np.int64)          # per root ordinal
        self.cycle_day = None if cycle_day is None else np.asarray(cycle_day, dtype=np.int64)
        self.window_start = int(month_index(np.array([parse_date(time_spec["output_start"])]))[0])
        self.window_end = self.window_start + int(time_spec["output_steps"]) - 1
        open_month = month_index(self.start)
        if self.cycle_day is None:
            # statement on the last day of the month -> first statement is the open month's
            self.first_step = open_month
        else:
            open_dom = self.start - month_start_days(open_month) + 1
            self.first_step = np.where(open_dom < self.cycle_day, open_month, open_month + 1)
        wu = time_spec.get("warmup_max_steps")
        earliest = int(self.first_step.min()) if self.first_step.size else self.window_start
        self.sim_start = earliest if wu is None else max(earliest, self.window_start - int(wu))
        self.sim_start = min(self.sim_start, self.window_start)

    def statement_date(self, g: int, roots: np.ndarray) -> np.ndarray:
        if self.cycle_day is None:
            return month_start_days(np.array([g + 1]))[0] - 1 + np.zeros(roots.size, np.int64)
        return month_start_days(np.array([g]))[0] + self.cycle_day[roots] - 1

    def context(self, g: int, roots: np.ndarray, first_active: np.ndarray) -> dict:
        """Step context arrays for the given active roots at absolute step g."""
        stmt = self.statement_date(g, roots)
        prev = self.statement_date(g - 1, roots)
        start = np.maximum(self.start[roots], prev + 1)
        return {
            "statement_date": stmt,
            "cycle_start_date": start,
            "days_in_cycle": stmt - start + 1,
            "cycle_number": np.full(roots.size, g - self.window_start + 1, np.int64),
            "months_on_book": g - self.first_step[roots] + 1,
            "step_index": np.full(roots.size, g, np.int64),
            "month_of_year": np.full(roots.size, int(month_of_year(np.array([g]))[0]), np.int64),
            "_first_active": first_active,
        }


def build_shared(shared_spec: dict, g_from: int, g_to: int, seed: int) -> dict[str, np.ndarray]:
    """Generate every shared series for absolute steps g_from..g_to (inclusive).

    Returned arrays are indexed by ``g - g_from``. AR(1) innovations are keyed
    by the absolute step, so a series value does not depend on how far back
    the simulation starts other than through the explicit recursion origin.
    """
    rng = KeyedRNG(seed)
    steps = np.arange(g_from, g_to + 1, dtype=np.int64)
    out: dict[str, np.ndarray] = {}
    for name, spec in shared_spec.items():
        p = spec.get("params", {})
        proc = spec["process"]
        if proc == "constant":
            x = np.full(steps.size, float(p.get("value", 0.0)))
        elif proc == "seasonal":
            factors = np.asarray(p["by_month"], dtype=float)
            x = factors[month_of_year(steps) - 1]
        elif proc == "explicit":
            vals = {int(month_index(np.array([parse_date(k + "-01" if len(k) == 7 else k)]))[0]): float(v)
                    for k, v in p["values"].items()}
            x = np.empty(steps.size)
            last = float(p.get("default", next(iter(vals.values()))))
            for i, g in enumerate(steps):           # forward fill
                last = vals.get(int(g), last)
                x[i] = last
        elif proc == "ar1":
            mean, phi, sigma = float(p["mean"]), float(p["phi"]), float(p["sigma"])
            eps = rng.normal(f"shared.{name}", np.zeros(steps.size, np.int64), steps)
            x = np.empty(steps.size)
            prev = float(p.get("initial", mean))
            for i in range(steps.size):
                prev = mean + phi * (prev - mean) + sigma * eps[i]
                x[i] = prev
        else:  # pragma: no cover - rejected by the compiler
            raise ValueError(proc)
        for shock in spec.get("shocks", []):
            g0 = int(month_index(np.array([parse_date(shock["from"])]))[0])
            g1 = int(month_index(np.array([parse_date(shock["to"])]))[0])
            sel = (steps >= g0) & (steps <= g1)
            if "add" in shock:
                x[sel] += float(shock["add"])
            if "multiply" in shock:
                x[sel] *= float(shock["multiply"])
        lo, hi = p.get("min"), p.get("max")
        if lo is not None or hi is not None:
            x = np.clip(x, -np.inf if lo is None else lo, np.inf if hi is None else hi)
        out[name] = x
    return out
