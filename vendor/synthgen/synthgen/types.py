"""
Column data types, exact fixed-point arithmetic and quantization.

Design notes
------------
* Money and every other ``decimal(p,s)`` column is stored internally as an
  ``int64`` array holding the value multiplied by ``10**s`` ("minor units").
  Integer arithmetic on minor units is exact, so balances carried across many
  time steps never drift by fractions of a cent (guide, Step 5). It is also
  fully vectorisable, which Python's ``decimal.Decimal`` is not.
* Rounding happens in exactly one place per value: when a float is converted
  into a fixed-point column (``float_to_fixed``) or when an integer quotient
  is formed (``div_round``). The rounding mode is a documented convention
  (``conventions.rounding`` in the contract), never an accident.
* Dates are stored as ``int64`` days since 1970-01-01 and timestamps as
  ``int64`` seconds since 1970-01-01T00:00:00 (UTC).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

# --------------------------------------------------------------------------- #
# Column type
# --------------------------------------------------------------------------- #

_INT_BITS = {"int8": 8, "int16": 16, "int32": 32, "int64": 64, "integer": 64, "int": 64}
_DECIMAL_RE = re.compile(r"^decimal\(\s*(\d+)\s*,\s*(\d+)\s*\)$")


@dataclass(frozen=True)
class ColumnType:
    """Logical column type as declared in the contract (``"type": ...``)."""

    kind: str                      # string | int | decimal | float | bool | date | timestamp
    bits: int = 64                 # for ints: declared width (range checked on output)
    precision: Optional[int] = None  # for decimals: total significant digits
    scale: int = 0                 # for decimals: digits after the point

    # ----- helpers --------------------------------------------------------- #
    @property
    def is_numeric(self) -> bool:
        return self.kind in ("int", "decimal", "float")

    @property
    def int_range(self) -> tuple[int, int]:
        """Representable range of a declared integer width."""
        half = 2 ** (self.bits - 1)
        return -half, half - 1

    @property
    def decimal_limit(self) -> Optional[int]:
        """Exclusive upper bound on |minor units| implied by ``decimal(p,s)``."""
        if self.kind != "decimal" or self.precision is None:
            return None
        return 10 ** self.precision

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        if self.kind == "decimal":
            return f"decimal({self.precision},{self.scale})"
        if self.kind == "int":
            return f"int{self.bits}"
        return self.kind


def parse_type(spec: str) -> ColumnType:
    """Parse a contract type string such as ``"decimal(10,2)"`` or ``"int16"``."""
    if not isinstance(spec, str):
        raise ValueError(f"column type must be a string, got {spec!r}")
    s = spec.strip().lower()
    if s in _INT_BITS:
        return ColumnType("int", bits=_INT_BITS[s])
    m = _DECIMAL_RE.match(s)
    if m:
        p, sc = int(m.group(1)), int(m.group(2))
        if sc > p:
            raise ValueError(f"{spec}: scale cannot exceed precision")
        if p > 18:
            raise ValueError(f"{spec}: precision > 18 does not fit in int64 minor units")
        return ColumnType("decimal", precision=p, scale=sc)
    if s in ("float", "double", "float64"):
        return ColumnType("float")
    if s in ("bool", "boolean"):
        return ColumnType("bool")
    if s in ("string", "str", "text", "varchar"):
        return ColumnType("string")
    if s == "date":
        return ColumnType("date")
    if s in ("timestamp", "datetime"):
        return ColumnType("timestamp")
    raise ValueError(f"unsupported column type {spec!r}")


# --------------------------------------------------------------------------- #
# Exact integer rounding
# --------------------------------------------------------------------------- #

ROUNDING_MODES = ("half_up", "half_even", "floor", "ceil", "down")


def div_round(n, d, mode: str = "half_up") -> np.ndarray:
    """Exactly divide integer array ``n`` by positive integer ``d`` and round.

    ``half_up`` means ties go away from zero (the usual financial rounding,
    equivalent to ``decimal.ROUND_HALF_UP``).
    """
    n = np.asarray(n, dtype=np.int64)
    d = np.asarray(d, dtype=np.int64)
    if np.any(d <= 0):
        raise ZeroDivisionError("div_round requires a positive divisor")
    if mode == "floor":
        return np.floor_divide(n, d)
    if mode == "ceil":
        return -np.floor_divide(-n, d)
    sign = np.where(n < 0, -1, 1).astype(np.int64)
    q, r = np.divmod(np.abs(n), d)
    if mode == "down":                       # truncate toward zero
        return sign * q
    twice = 2 * r
    if mode == "half_up":
        q = q + (twice >= d)
    elif mode == "half_even":
        q = q + ((twice > d) | ((twice == d) & (q % 2 == 1)))
    else:
        raise ValueError(f"unknown rounding mode {mode!r}")
    return sign * q


def rescale(v, from_scale: int, to_scale: int, mode: str = "half_up") -> np.ndarray:
    """Change the scale of a fixed-point array (exact when increasing scale)."""
    v = np.asarray(v, dtype=np.int64)
    if to_scale == from_scale:
        return v
    if to_scale > from_scale:
        return v * np.int64(10 ** (to_scale - from_scale))
    return div_round(v, 10 ** (from_scale - to_scale), mode)


# Floats that are within this many units of the last place of a tie are
# treated as ties. It compensates for binary representation error such as
# 2.675 being stored as 2.67499999999999982236431605997495353221893310546875.
_TIE_EPS = 1e-7


def float_to_fixed(x, scale: int, mode: str = "half_up") -> np.ndarray:
    """Convert float values (value units) to fixed-point minor units at ``scale``.

    This is the single documented rounding point for any value produced by
    floating point math (distribution draws, ratios, custom expressions).
    """
    x = np.asarray(x, dtype=np.float64) * (10.0 ** scale)
    if mode == "floor":
        r = np.floor(x + _TIE_EPS)
    elif mode == "ceil":
        r = np.ceil(x - _TIE_EPS)
    elif mode == "down":
        r = np.trunc(x + np.sign(x) * _TIE_EPS)
    elif mode == "half_up":
        r = np.sign(x) * np.floor(np.abs(x) + 0.5 + _TIE_EPS)
    elif mode == "half_even":
        a = np.abs(x)
        fl = np.floor(a)
        frac = a - fl
        is_tie = np.abs(frac - 0.5) <= _TIE_EPS
        up = (frac > 0.5 + _TIE_EPS) | (is_tie & (fl % 2 == 1))
        r = np.sign(x) * (fl + up)
    else:
        raise ValueError(f"unknown rounding mode {mode!r}")
    r = np.where(np.isfinite(r), r, 0.0)
    return r.astype(np.int64)


def fixed_to_float(v, scale: int) -> np.ndarray:
    """Minor units -> float value units (for statistics / float math only)."""
    return np.asarray(v, dtype=np.float64) / (10.0 ** scale)


def quantize_fixed(v, step_units: int, mode: str = "half_up") -> np.ndarray:
    """Round fixed-point values to a multiple of ``step_units`` (exact)."""
    if step_units <= 1:
        return np.asarray(v, dtype=np.int64)
    return div_round(v, step_units, mode) * np.int64(step_units)


def step_to_units(step: float, scale: int) -> int:
    """Convert a quantization step in value units (e.g. 0.0025) to minor units.

    Raises if the step is not representable at the column's scale, which is a
    contract error (e.g. ``round_to: 0.001`` on a ``decimal(10,2)`` column).
    """
    units = step * (10 ** scale)
    rounded = int(round(units))
    if rounded <= 0 or abs(units - rounded) > 1e-9 * max(1.0, abs(units)):
        raise ValueError(f"quantization step {step} is not a multiple of 10^-{scale}")
    return rounded


# --------------------------------------------------------------------------- #
# Dates
# --------------------------------------------------------------------------- #

_EPOCH = np.datetime64("1970-01-01", "D")


def parse_date(s: str) -> int:
    """'YYYY-MM-DD' -> days since epoch."""
    return int((np.datetime64(s, "D") - _EPOCH).astype(np.int64))


def parse_timestamp(s: str) -> int:
    """ISO timestamp -> seconds since epoch (UTC)."""
    return int((np.datetime64(s, "s") - np.datetime64("1970-01-01T00:00:00", "s")).astype(np.int64))


def days_to_datetime64(d) -> np.ndarray:
    return np.asarray(d, dtype=np.int64).astype("datetime64[D]")


def month_index(days) -> np.ndarray:
    """Days since epoch -> months since 1970-01 (0-based)."""
    return days_to_datetime64(days).astype("datetime64[M]").astype(np.int64)


def month_start_days(month_idx) -> np.ndarray:
    """Months since 1970-01 -> day number of the 1st of that month."""
    m = np.asarray(month_idx, dtype=np.int64).astype("datetime64[M]")
    return m.astype("datetime64[D]").astype(np.int64)


def day_of_month(days) -> np.ndarray:
    d = np.asarray(days, dtype=np.int64)
    return d - month_start_days(month_index(d)) + 1


def month_of_year(month_idx) -> np.ndarray:
    """1..12 for a months-since-epoch index."""
    return np.asarray(month_idx, dtype=np.int64) % 12 + 1


# --------------------------------------------------------------------------- #
# Value coercion helpers (value units <-> storage)
# --------------------------------------------------------------------------- #

def coerce_scalar_to_storage(value, ctype: ColumnType, mode: str = "half_up"):
    """Convert a contract literal (bounds, init values, ...) to storage units."""
    if value is None:
        return None
    if ctype.kind == "decimal":
        return int(float_to_fixed(np.array([float(value)]), ctype.scale, mode)[0])
    if ctype.kind == "int":
        return int(round(float(value)))
    if ctype.kind == "float":
        return float(value)
    if ctype.kind == "bool":
        return bool(value)
    if ctype.kind == "date":
        return parse_date(value) if isinstance(value, str) else int(value)
    if ctype.kind == "timestamp":
        return parse_timestamp(value) if isinstance(value, str) else int(value)
    return value


def storage_to_value_units(values: np.ndarray, ctype: ColumnType) -> np.ndarray:
    """Storage array -> natural numeric value units (decimal -> float)."""
    if ctype.kind == "decimal":
        return fixed_to_float(values, ctype.scale)
    return values
