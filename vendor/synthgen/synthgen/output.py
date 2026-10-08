"""
CSV output - guide Step 10.

Each grain is written as one canonical long table. Formatting is fully
vectorised: every column is rendered to a fixed-width numpy byte-string array,
the arrays are laid side by side in one ``uint8`` matrix with separators, and
the NUL padding is squeezed out in a single pass. This writes roughly a
million rows per second per core with no third-party CSV library.

Formatting conventions
----------------------
* decimals: exactly ``scale`` digits after the point, from exact minor units
* dates ``YYYY-MM-DD``; timestamps ``YYYY-MM-DDTHH:MM:SS`` (UTC)
* booleans ``true`` / ``false``
* NULL is the empty field
* strings containing ``,`` ``"`` or newlines are quoted RFC-4180 style
"""
from __future__ import annotations

import numpy as np

from .contract.compiler import Plan
from .engine.values import Col
from .identifiers import PatternCodec


class CodecCache:
    """One PatternCodec per identifier column, seeded with the contract seed."""

    def __init__(self, plan: Plan):
        self.plan = plan
        self._codecs: dict[str, PatternCodec] = {}

    def get(self, full: str) -> PatternCodec:
        c = self._codecs.get(full)
        if c is None:
            node = self.plan.node(full)
            c = PatternCodec(node.spec["format"]["pattern"], self.plan.seed, full)
            self._codecs[full] = c
        return c


def _fmt_fixed(v: np.ndarray, scale: int) -> np.ndarray:
    """Exact fixed-point -> text, written digit by digit into a byte matrix.

    Digits are right-aligned; unused leading cells stay NUL and are squeezed
    out by ``render_rows`` - no per-value Python work and no string ops.
    """
    v = np.asarray(v, dtype=np.int64)
    n = v.size
    a = np.abs(v)
    ip, fp = np.divmod(a, 10 ** scale)
    # number of integer digits per value (at least one: "0.50")
    nd = np.ones(n, dtype=np.int64)
    t = ip // 10
    while t.any():
        nd += t > 0
        t //= 10
    max_nd = int(nd.max()) if n else 1
    width = 1 + max_nd + (1 + scale if scale else 0)
    mat = np.zeros((n, width), dtype=np.uint8)
    pos = width - 1
    for _ in range(scale):                      # fractional digits, always written
        fp, d = np.divmod(fp, 10)
        mat[:, pos] = d + 48
        pos -= 1
    if scale:
        mat[:, pos] = ord(".")
        pos -= 1
    int_end = pos                               # rightmost integer digit column
    for k in range(max_nd):                     # integer digits, only where needed
        ip, d = np.divmod(ip, 10)
        rows = nd > k
        mat[rows, int_end - k] = (d + 48)[rows]
    neg = np.nonzero(v < 0)[0]
    if neg.size:
        mat[neg, int_end - nd[neg]] = ord("-")
    return mat.view(f"S{width}").reshape(-1)


def _lut_format(v: np.ndarray, fmt) -> np.ndarray:
    """Format each distinct value once, then gather (for low-cardinality columns)."""
    uniq, inv = np.unique(v, return_inverse=True)
    lut = fmt(uniq)
    width = max(int(np.strings.str_len(lut).max()), 1) if lut.size else 1
    return lut.astype(f"S{width}")[inv]                # tight width keeps rows narrow


def _repeats(v: np.ndarray) -> bool:
    """True when a sample suggests many repeated values (e.g. a foreign key)."""
    sample = v[: min(v.size, 20000)]
    return v.size > 1000 and np.unique(sample).size < sample.size // 2


def _build_time_lut() -> np.ndarray:
    s = np.arange(86400)
    h, rem = np.divmod(s, 3600)
    m, sec = np.divmod(rem, 60)
    mat = np.zeros((86400, 9), dtype=np.uint8)
    mat[:, 0] = ord("T")
    for col, x in ((1, h), (4, m), (7, sec)):
        mat[:, col] = x // 10 + 48
        mat[:, col + 1] = x % 10 + 48
    mat[:, 3] = mat[:, 6] = ord(":")
    return mat.view("S9").reshape(-1)


_TIME_LUT = _build_time_lut()


def _small_range(v: np.ndarray) -> bool:
    return v.size > 1000 and v.dtype.kind in "iu" and int(v.max()) - int(v.min()) < v.size // 4


def _quote_if_needed(s: np.ndarray) -> np.ndarray:
    needs = ((np.strings.find(s, b",") >= 0) | (np.strings.find(s, b'"') >= 0)
             | (np.strings.find(s, b"\n") >= 0))
    if not needs.any():
        return s
    q = np.strings.add(np.strings.add(b'"', np.strings.replace(s, b'"', b'""')), b'"')
    return np.where(needs, q, s)


def format_col(col: Col, codecs: CodecCache) -> np.ndarray:
    """Render a column to an 'S' array (NULL -> empty)."""
    v, ct = col.values, col.ctype
    if v.size == 0:
        return np.zeros(0, dtype="S1")
    if col.ref is not None:                             # identifiers / foreign keys
        codec = codecs.get(col.ref)
        out = _lut_format(v, codec.unique) if _repeats(v) else codec.unique(v)
    elif col.labels is not None:
        lab = np.array([s.encode("utf-8") for s in col.labels] + [b""])
        out = _quote_if_needed(lab)[np.where(v < 0, len(col.labels), v)]
    elif ct.kind == "decimal":
        out = _fmt_fixed(v, ct.scale)
    elif ct.kind == "int":
        out = _lut_format(v, lambda u: u.astype("S")) if _small_range(v) else _fmt_fixed(v, 0)
    elif ct.kind == "float":
        out = np.char.mod(b"%.10g", v.astype(float)).astype("S")
    elif ct.kind == "bool":
        out = np.where(v.astype(bool), b"true", b"false")
    elif ct.kind == "date":
        out = _lut_format(v, lambda u: u.astype("datetime64[D]").astype("S10"))
    elif ct.kind == "timestamp":
        days, secs = np.divmod(v.astype(np.int64), 86400)
        d = _lut_format(days, lambda u: u.astype("datetime64[D]").astype("S10"))
        t = _TIME_LUT[secs]
        out = np.ascontiguousarray(np.concatenate(
            [d.view(np.uint8).reshape(-1, 10), t.view(np.uint8).reshape(-1, 9)], axis=1)
        ).view("S19").reshape(-1)
    else:                                               # free text
        out = v if v.dtype.kind == "S" else v.astype(object).astype("S")
        out = _quote_if_needed(out)
    if col.mask is not None and col.mask.any():
        out = np.where(col.mask, b"", out)
    return out


def render_rows(columns: list[np.ndarray]) -> bytes:
    """Join byte-string columns into CSV rows (comma-separated, '\\n' terminated)."""
    if not columns:
        return b""
    n = columns[0].size
    if n == 0:
        return b""
    widths = [max(c.dtype.itemsize, 1) for c in columns]
    total = sum(widths) + len(columns)
    mat = np.zeros((n, total), dtype=np.uint8)
    off = 0
    for i, (c, w) in enumerate(zip(columns, widths)):
        c = np.ascontiguousarray(c.astype(f"S{w}"))
        mat[:, off:off + w] = c.view(np.uint8).reshape(n, w)
        off += w
        mat[:, off] = ord(",") if i < len(columns) - 1 else ord("\n")
        off += 1
    flat = mat.reshape(-1)
    return flat[flat != 0].tobytes()


def emitted_columns(plan: Plan, entity: str) -> list[str]:
    return [c for c, n in plan.entities[entity].nodes.items() if n.emit]


def header(plan: Plan, entity: str) -> bytes:
    return (",".join(emitted_columns(plan, entity)) + "\n").encode("utf-8")


def format_table(plan: Plan, entity: str, cols: dict[str, Col], codecs: CodecCache,
                 batch: int = 1_000_000):
    """Yield CSV bytes for a table (without header) in batches to bound memory."""
    names = emitted_columns(plan, entity)
    if not names:
        return
    n = len(cols[names[0]])
    for lo in range(0, n, batch):
        hi = min(n, lo + batch)
        part = [format_col(cols[c].take(np.arange(lo, hi)), codecs) for c in names]
        yield render_rows(part)
