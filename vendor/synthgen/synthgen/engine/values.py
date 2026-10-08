"""
Runtime column container and type conversions between columns and rules.

A ``Col`` is one column for the rows currently being processed:

* ``values``  numpy array in *storage* units (see ``synthgen.types``):
              decimals as int64 minor units, dates as int64 days, categoricals
              as int codes into ``labels``, identifiers/foreign keys as int64
              ordinals of the referenced entity (formatted only at write time).
* ``mask``    optional boolean NULL mask.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..expressions import Val
from ..rules import type_scale
from ..types import (ColumnType, fixed_to_float, float_to_fixed, quantize_fixed, rescale)

INT = ColumnType("int")
FLOAT = ColumnType("float")
DATE = ColumnType("date")
BOOL = ColumnType("bool")


@dataclass
class Col:
    values: np.ndarray
    ctype: ColumnType
    mask: Optional[np.ndarray] = None
    labels: Optional[tuple] = None
    ref: Optional[str] = None            # "entity.column" identifier that formats ordinals

    def take(self, idx: np.ndarray) -> "Col":
        return Col(self.values[idx], self.ctype,
                   None if self.mask is None else self.mask[idx], self.labels, self.ref)

    def __len__(self) -> int:
        return len(self.values)

    def to_val(self) -> Val:
        """For the expression language: numeric values in natural units."""
        if self.ctype.kind == "decimal":
            return Val(fixed_to_float(self.values, self.ctype.scale), self.mask)
        if self.labels is not None:
            return Val(self.values, self.mask, self.labels)
        return Val(self.values, self.mask)


def concat_cols(cols: list[Col]) -> Col:
    first = cols[0]
    values = np.concatenate([c.values for c in cols])
    if any(c.mask is not None for c in cols):
        mask = np.concatenate([c.mask if c.mask is not None else np.zeros(len(c), bool)
                               for c in cols])
    else:
        mask = None
    return Col(values, first.ctype, mask, first.labels, first.ref)


# --------------------------------------------------------------------------- #
# column -> rule input
# --------------------------------------------------------------------------- #

def to_rule_type(col: Col, rtype: str, rounding: str) -> np.ndarray:
    """Convert a column to the representation a rule declares for an input."""
    v, ct = col.values, col.ctype
    scale = type_scale(rtype)
    if scale is not None:
        if ct.kind == "decimal":
            return rescale(v, ct.scale, scale, rounding)
        if ct.kind in ("int", "bool"):
            return rescale(v.astype(np.int64), 0, scale, rounding)
        if ct.kind == "float":
            return float_to_fixed(v, scale, rounding)
        raise TypeError(f"cannot pass {ct} as {rtype}")
    if rtype == "float":
        if ct.kind == "decimal":
            return fixed_to_float(v, ct.scale)
        return v.astype(np.float64)
    if rtype == "int":
        if ct.kind == "decimal":
            return rescale(v, ct.scale, 0, rounding)
        if ct.kind == "float":
            return float_to_fixed(v, 0, rounding)
        return v.astype(np.int64)
    if rtype == "bool":
        return v.astype(bool)
    if rtype in ("date", "timestamp"):
        return v.astype(np.int64)
    if rtype in ("category", "any"):
        if col.labels is not None:
            lab = np.asarray(list(col.labels) + [None], dtype=object)
            return lab[np.where(v < 0, len(col.labels), v)]
        if ct.kind == "decimal":
            return fixed_to_float(v, ct.scale)
        return v
    raise TypeError(f"unknown rule type {rtype!r}")


def null_fill(values: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """Replace NULL slots by a neutral value (0 / False / '') for NULL-unaware rules."""
    if mask is None or not mask.any():
        return values
    if values.dtype == object:
        out = values.copy()
        out[mask] = ""
        return out
    return np.where(mask, np.zeros(1, dtype=values.dtype), values)


# --------------------------------------------------------------------------- #
# rule output / draw -> column storage
# --------------------------------------------------------------------------- #

def to_storage(values, rtype: str, node, rounding: str) -> np.ndarray:
    """Convert a rule/draw output (in ``rtype`` representation) to node storage.

    This is where decimals produced by float math are rounded - the single
    documented rounding point - followed by the node's quantization.
    Values are *not* clipped here: derived values must never be edited after
    the fact; their bounds are verified, not enforced.
    """
    ct = node.ctype
    values = np.asarray(values)
    src_scale = type_scale(rtype)
    if ct.kind == "string":
        if node.labels is not None:
            if values.dtype.kind in "iu" and rtype not in ("category", "any"):
                return values.astype(np.int64)
            # hash-factorise (C speed), then map the few distinct labels to codes
            codes_u, uniques = pd.factorize(values.astype(object), use_na_sentinel=True)
            lut = {lab: i for i, lab in enumerate(node.labels)}
            mapped = np.array([lut.get(u.decode() if isinstance(u, bytes) else u, -1)
                               for u in uniques] + [-1], dtype=np.int64)
            if (mapped[:-1] < 0).any():
                bad = [str(u) for u, m in zip(uniques, mapped) if m < 0][:5]
                raise ValueError(f"{node.full}: values {bad} are not in allowed_values")
            return mapped[codes_u]
        if values.dtype.kind == "S":
            return values
        return values.astype(object).astype("S")
    if ct.kind == "decimal":
        if src_scale is not None:
            out = rescale(values, src_scale, ct.scale, rounding)
        elif values.dtype.kind in "iub" and rtype in ("int", "bool"):
            out = rescale(values.astype(np.int64), 0, ct.scale, rounding)
        else:
            out = float_to_fixed(values.astype(np.float64), ct.scale, rounding)
    elif ct.kind == "int":
        if src_scale is not None:
            out = rescale(values, src_scale, 0, rounding)
        elif values.dtype.kind == "f":
            out = float_to_fixed(values, 0, rounding)
        else:
            out = values.astype(np.int64)
    elif ct.kind == "float":
        out = fixed_to_float(values, src_scale) if src_scale is not None else values.astype(np.float64)
    elif ct.kind == "bool":
        out = values.astype(bool)
    else:                                   # date / timestamp
        out = values.astype(np.int64)
    if node.quant is not None and ct.kind in ("decimal", "int"):
        out = quantize_fixed(out, node.quant[0], node.quant[1])
    return out


def bound_at_creation(values: np.ndarray, node) -> np.ndarray:
    """Apply a drawn node's bounds at the moment of creation (allowed by the guide).

    After quantization a truncated draw can land one step outside a bound
    that is not itself a multiple of the step; such values move to the
    nearest in-bound multiple.
    """
    lo, hi, mode = node.bounds
    if mode not in ("truncate", "clip") or (lo is None and hi is None):
        return values
    if node.ctype.kind not in ("decimal", "int", "float", "date", "timestamp"):
        return values
    if node.quant is not None and node.ctype.kind in ("decimal", "int"):
        q = node.quant[0]
        lo = None if lo is None else int(quantize_fixed(np.array([lo]), q, "ceil")[0])
        hi = None if hi is None else int(quantize_fixed(np.array([hi]), q, "floor")[0])
    if lo is not None:
        values = np.maximum(values, lo)
    if hi is not None:
        values = np.minimum(values, hi)
    return values
