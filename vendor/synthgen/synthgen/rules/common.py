"""
Domain-neutral rules usable by any contract (package ``common``).

Conventions for every rule in the library:
* inputs arrive already converted to the declared rule types
  (see ``synthgen.rules`` for the type vocabulary);
* a rule returns either an array, or ``(array, null_mask)``;
* rules never draw random numbers - samplers receive ``u(i)``, a function
  returning the i-th keyed uniform array for the current rows.
"""
from __future__ import annotations

import numpy as np

from ..distributions import categorical
from . import registrar

rule, sampler, group = registrar("common")


@rule("common.copy", "1.0", inputs={"x": "any"}, output="any")
def copy(x):
    """Return the input unchanged (e.g. expose a lagged state as a column)."""
    return x


@rule("common.param_value", "1.0", inputs={}, output="any", params={"value": "any"})
def param_value(value):
    """Return a (possibly per-row, category-dependent) parameter as a column."""
    return value


@rule("common.add_days", "1.0", inputs={"date": "date", "days": "int"}, output="date",
      params={"offset": ("int", 0)})
def add_days(date, days, offset):
    """date + days + offset."""
    return date + days + offset


@rule("common.date_offset", "1.0", inputs={"date": "date"}, output="date",
      params={"days": "int"})
def date_offset(date, days):
    """date + a fixed (or per-row) number of days."""
    return date + days


@rule("common.days_between", "1.0", inputs={"start": "date", "end": "date"}, output="int")
def days_between(start, end):
    """end - start in days."""
    return end - start


@rule("common.date_plus_seconds", "1.0", inputs={"date": "date", "seconds": "int"},
      output="timestamp")
def date_plus_seconds(date, seconds):
    """Combine a date and seconds-after-midnight into a timestamp."""
    return date * 86400 + seconds


@rule("common.ratio", "1.0", inputs={"numerator": "float", "denominator": "float"},
      output="float", params={"floor_numerator_at_zero": ("bool", False)})
def ratio(numerator, denominator, floor_numerator_at_zero):
    """numerator / denominator; NULL where the denominator is 0."""
    num = np.maximum(numerator, 0.0) if floor_numerator_at_zero else numerator
    zero = denominator == 0
    out = np.where(zero, 0.0, num / np.where(zero, 1.0, denominator))
    return out, zero


@rule("common.add", "1.0", inputs={"a": "fixed6", "b": "fixed6"}, output="fixed6")
def add(a, b):
    """a + b, exact to 6 decimal places."""
    return a + b


@rule("common.subtract", "1.0", inputs={"a": "fixed6", "b": "fixed6"}, output="fixed6")
def subtract(a, b):
    """a - b, exact to 6 decimal places."""
    return a - b


@rule("common.running_total", "1.0", inputs={"previous": "fixed6", "value": "fixed6"},
      output="fixed6")
def running_total(previous, value):
    """Cumulative sum carried as state: previous total + this step's value."""
    return previous + value


@rule("common.at_least", "1.0", inputs={"x": "float"}, output="bool",
      params={"threshold": "float"})
def at_least(x, threshold):
    """x >= threshold."""
    return x >= threshold


@rule("common.any_true", "1.0", inputs={"a": "bool", "b": "bool"}, output="bool")
def any_true(a, b):
    """a OR b."""
    return a | b


# --------------------------------------------------------------------------- #
# Samplers (decorative / generic)
# --------------------------------------------------------------------------- #

@sampler("text.compose", "1.0", inputs={}, output="any",
         params={"parts": "list", "separator": ("any", " ")}, uniforms=-1)
def text_compose(u, n, parts, separator):
    """Build a string by picking one word from each list in ``parts``.

    Example params: {"parts": [["Blue","Golden"], ["Oak","River"], ["Market","Cafe"]]}
    """
    out = None
    sep = np.asarray(separator, dtype=object)
    for i, words in enumerate(parts):
        words_arr = np.asarray(words, dtype=object)
        idx = np.minimum((u(i) * len(words_arr)).astype(np.int64), len(words_arr) - 1)
        piece = words_arr[idx]
        out = piece if out is None else out + sep + piece
    return out


@sampler("time.seconds_of_day", "1.0", inputs={}, output="int",
         params={"hour_weights": ("list", None)}, uniforms=2)
def seconds_of_day(u, n, hour_weights):
    """Seconds after midnight; hour drawn from ``hour_weights`` (24 values)."""
    w = np.ones(24) if hour_weights is None else np.asarray(hour_weights, dtype=float)
    if w.size != 24:
        raise ValueError("hour_weights must have 24 entries")
    hour = categorical(u(0), w)
    return hour * 3600 + np.minimum((u(1) * 3600).astype(np.int64), 3599)
